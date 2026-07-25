#!/usr/bin/env python
"""Offline-verifiable primitives for the Paper C v2 matched DPO study.

This module deliberately has no torch/transformers/TRL dependency.  It defines
the normative one-token losses, deterministic Stage-2 family split, uncertainty
selection, condition grid, and canonical hashes used by lock/train/eval code.

The primary comparison is not generic sequence DPO.  With exactly two verdict
tokens and chosen=correct/rejected=wrong, DPO reduces to a frozen-reference
margin loss:

    softplus(-beta * (m_policy - m_reference))

where m is the gold-signed unsafe-minus-safe logit margin.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 2
LOSS_FORMULA_VERSION = "paper_c_binary_margin_v2"
PARTITION_ALGORITHM_VERSION = "paper_c_global_family_balanced_v2"
SELECTION_ALGORITHM_VERSION = "paper_c_stratified_entropy_v2"
OBJECTIVES = ("verdict_ce", "pair_ce", "dpo")
SAMPLERS = ("uncertain", "matched_random")
PARTITIONS = ("stage2_update", "stage2_dev")


class PaperCContractError(ValueError):
    """Raised when an input would make two Paper C cells incomparable."""


def canonical_json_bytes(value: object) -> bytes:
    """Stable JSON encoding used for all Paper C object fingerprints."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ordered(values: Iterable[object]) -> str:
    """Fingerprint values with explicit record boundaries and ordering."""
    digest = hashlib.sha256()
    for value in values:
        payload = canonical_json_bytes(value)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PaperCContractError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise PaperCContractError(
                    f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[Mapping]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False,
                                    allow_nan=False) + "\n")


def validate_execution_sources(lock: Mapping, repo_root: str | Path) -> None:
    """Rehash every behavior-changing source bound by a final Paper C lock."""
    source_record = lock.get("execution_sources")
    if not isinstance(source_record, Mapping):
        raise PaperCContractError("Paper C lock has no execution-source inventory")
    files = source_record.get("files")
    if not isinstance(files, Mapping) or not files:
        raise PaperCContractError("Paper C execution-source inventory is empty")
    rebound = []
    root = Path(repo_root)
    for relative, expected in sorted(files.items()):
        path = root / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise PaperCContractError(f"Paper C execution source drifted: {relative}")
        rebound.append({"path": str(relative), "sha256": str(expected)})
    if sha256_ordered(rebound) != source_record.get("aggregate_sha256"):
        raise PaperCContractError("Paper C execution-source aggregate hash mismatch")


def normalize_gold(value: object) -> int:
    """Return 0=safe or 1=unsafe; refuse ambiguous truthy coercions."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"safe", "0"}:
            return 0
        if normalized in {"unsafe", "harmful", "1"}:
            return 1
    raise PaperCContractError(f"invalid binary guard label: {value!r}")


def label_sign(gold: object) -> int:
    return 1 if normalize_gold(gold) == 1 else -1


def unsafe_minus_safe(safe_logit: float, unsafe_logit: float) -> float:
    return float(unsafe_logit) - float(safe_logit)


def signed_margin(safe_logit: float, unsafe_logit: float, gold: object) -> float:
    return float(label_sign(gold)) * unsafe_minus_safe(safe_logit, unsafe_logit)


def _softplus(value: float) -> float:
    """Numerically stable scalar log(1+exp(value))."""
    x = float(value)
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def _sigmoid(value: float) -> float:
    x = float(value)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def pair_ce_loss(policy_signed_margin: float, beta: float) -> float:
    beta_value = float(beta)
    if not math.isfinite(beta_value) or beta_value <= 0:
        raise PaperCContractError(f"beta must be finite and positive, got {beta!r}")
    return _softplus(-beta_value * float(policy_signed_margin))


def dpo_loss(policy_signed_margin: float, reference_signed_margin: float,
             beta: float) -> float:
    beta_value = float(beta)
    if not math.isfinite(beta_value) or beta_value <= 0:
        raise PaperCContractError(f"beta must be finite and positive, got {beta!r}")
    delta = float(policy_signed_margin) - float(reference_signed_margin)
    return _softplus(-beta_value * delta)


def dpo_logratio_loss(
    chosen_policy_logp: float,
    rejected_policy_logp: float,
    chosen_reference_logp: float,
    rejected_reference_logp: float,
    beta: float,
) -> float:
    """Canonical DPO scalar form, used to test the margin reduction."""
    delta = (
        float(chosen_policy_logp) - float(rejected_policy_logp)
        - float(chosen_reference_logp) + float(rejected_reference_logp)
    )
    return _softplus(-float(beta) * delta)


def two_verdict_probability_unsafe(safe_logit: float, unsafe_logit: float) -> float:
    return _sigmoid(unsafe_minus_safe(safe_logit, unsafe_logit))


def probability_correct(safe_logit: float, unsafe_logit: float, gold: object) -> float:
    p_unsafe = two_verdict_probability_unsafe(safe_logit, unsafe_logit)
    return p_unsafe if normalize_gold(gold) == 1 else 1.0 - p_unsafe


def binary_entropy(probability: float) -> float:
    p = float(probability)
    if not math.isfinite(p) or p < 0 or p > 1:
        raise PaperCContractError(f"probability must lie in [0,1], got {probability!r}")
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def two_verdict_entropy(safe_logit: float, unsafe_logit: float) -> float:
    return binary_entropy(two_verdict_probability_unsafe(safe_logit, unsafe_logit))


def condition_grid() -> list[dict]:
    return [
        {"objective": objective, "sampler": sampler,
         "condition": f"{objective}__{sampler}"}
        for sampler in SAMPLERS
        for objective in OBJECTIVES
    ]


def validate_config(config: Mapping) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise PaperCContractError("Paper C config schema_version must be 2")
    if not str(config.get("study_id", "")).strip():
        raise PaperCContractError("Paper C config has no study_id")
    artifact_root_value = str(config.get("artifact_root", "")).strip()
    artifact_root = Path(artifact_root_value)
    if not artifact_root_value or artifact_root.is_absolute() or ".." in artifact_root.parts:
        raise PaperCContractError("artifact_root must be a nonempty repository-relative path")
    models = list(config.get("models") or [])
    seeds = list(config.get("seeds") or [])
    if not models or len(models) != len(set(models)):
        raise PaperCContractError("models must be a nonempty unique list")
    if (not seeds or len(seeds) != len(set(seeds))
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)):
        raise PaperCContractError("seeds must be a nonempty unique integer list")
    stage2 = config.get("stage2")
    if not isinstance(stage2, Mapping):
        raise PaperCContractError("Paper C config has no stage2 object")
    if tuple(stage2.get("objectives") or ()) != OBJECTIVES:
        raise PaperCContractError(
            f"objectives must be exactly {OBJECTIVES}, in that order")
    if tuple(stage2.get("samplers") or ()) != SAMPLERS:
        raise PaperCContractError(
            f"samplers must be exactly {SAMPLERS}, in that order")
    for field in ("development_fraction", "uncertain_fraction"):
        value = float(stage2.get(field, -1))
        if not (0 < value < 0.5):
            raise PaperCContractError(f"{field} must lie in (0, 0.5)")
    if float(stage2.get("pairwise_beta", -1)) <= 0:
        raise PaperCContractError("pairwise_beta must be positive")
    if float(stage2.get("stage2_dropout", -1)) != 0.0:
        raise PaperCContractError("stage2_dropout must be exactly zero in the v2 isolation design")
    if float(stage2.get("reference_margin_atol", -1)) <= 0:
        raise PaperCContractError("reference_margin_atol must be positive")
    for field in ("learning_rate", "warmup_ratio"):
        if not math.isfinite(float(stage2.get(field, float("nan")))):
            raise PaperCContractError(f"{field} must be finite")
    if float(stage2.get("learning_rate", -1)) <= 0:
        raise PaperCContractError("learning_rate must be positive")
    if not (0 <= float(stage2.get("warmup_ratio", -1)) < 1):
        raise PaperCContractError("warmup_ratio must lie in [0,1)")
    if not str(stage2.get("scheduler", "")).strip():
        raise PaperCContractError("scheduler must be nonempty")
    for field in ("max_steps", "per_device_batch", "gradient_accumulation"):
        value = int(stage2.get(field, 0))
        if value <= 0:
            raise PaperCContractError(f"{field} must be positive")
    checkpoints = [int(v) for v in stage2.get("checkpoint_steps") or []]
    if not checkpoints or checkpoints != sorted(set(checkpoints)):
        raise PaperCContractError("checkpoint_steps must be sorted and unique")
    if checkpoints[-1] != int(stage2.get("max_steps", -1)):
        raise PaperCContractError("last checkpoint must equal max_steps")
    analysis = config.get("analysis")
    if not isinstance(analysis, Mapping):
        raise PaperCContractError("Paper C config has no analysis object")
    if int(analysis.get("bootstrap_replicates", 0)) <= 0:
        raise PaperCContractError("bootstrap_replicates must be positive")
    target_fpr = float(analysis.get("target_fpr", -1))
    if not (0 < target_fpr < 1):
        raise PaperCContractError("target_fpr must lie in (0,1)")


def _row_identity(row: Mapping) -> tuple[str, str, int, str, str]:
    try:
        sample_id = str(row["sample_id"])
        source = str(row["source"])
        gold = normalize_gold(row.get("gold", row.get("label")))
        family_id = str(row["family_id"])
        content_sha = str(row["content_sha256"])
    except KeyError as exc:
        raise PaperCContractError(f"row missing identity field: {exc.args[0]}") from exc
    if not all((sample_id, source, family_id, content_sha)):
        raise PaperCContractError("row identity fields may not be empty")
    return sample_id, source, gold, family_id, content_sha


def family_partition(
    rows: Sequence[Mapping],
    *,
    development_fraction: float,
    seed: int,
) -> list[dict]:
    """Deterministically assign whole families to Stage-2 update or dev.

    A global family may span sources.  Families are assigned once while a
    deterministic greedy objective approximates the requested row fraction in
    every source/label stratum.  No family is ever split.
    """
    fraction = float(development_fraction)
    if not (0 < fraction < 0.5):
        raise PaperCContractError("development_fraction must lie in (0, 0.5)")
    if not rows:
        raise PaperCContractError("cannot partition an empty row sequence")

    # family_id is the global leakage unit inherited from Paper A.  A family
    # crossing source/label strata is a contract error, not two local families.
    families: dict[str, list[Mapping]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        sample_id, _, _, family_id, _ = _row_identity(row)
        if sample_id in seen_ids:
            raise PaperCContractError(f"duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        families[family_id].append(row)

    family_counts: dict[str, dict[tuple[str, int], int]] = {}
    totals: dict[tuple[str, int], int] = defaultdict(int)
    for family_id, members in families.items():
        counts: dict[tuple[str, int], int] = defaultdict(int)
        for member in members:
            _, source, gold, _, _ = _row_identity(member)
            counts[(source, gold)] += 1
            totals[(source, gold)] += 1
        family_counts[family_id] = dict(counts)
    for stratum, total in totals.items():
        if total < 2:
            raise PaperCContractError(f"stratum {stratum} is too small to split")

    targets = {
        stratum: min(total - 1, max(1, int(round(total * fraction))))
        for stratum, total in totals.items()
    }
    dev_counts: dict[tuple[str, int], int] = defaultdict(int)
    dev_families: set[str] = set()

    def objective(counts: Mapping[tuple[str, int], int]) -> float:
        score = 0.0
        for stratum, total in totals.items():
            observed = int(counts.get(stratum, 0))
            if observed == 0:
                score += 1000.0
            score += ((observed - targets[stratum]) / float(total)) ** 2
        return score

    while True:
        current_score = objective(dev_counts)
        candidates = []
        for family_id, counts in family_counts.items():
            if family_id in dev_families:
                continue
            # Every represented stratum must retain at least one update row.
            if any(dev_counts[stratum] + count >= totals[stratum]
                   for stratum, count in counts.items()):
                continue
            trial = defaultdict(int, dev_counts)
            for stratum, count in counts.items():
                trial[stratum] += count
            trial_score = objective(trial)
            tie_key = hashlib.sha256(
                f"{int(seed)}|{family_id}".encode("utf-8")).hexdigest()
            candidates.append((trial_score, tie_key, family_id, trial))
        if not candidates:
            break
        best_score, _, best_family, best_counts = min(candidates)
        missing_stratum = any(dev_counts[stratum] == 0 for stratum in totals)
        if not missing_stratum and best_score >= current_score - 1e-15:
            break
        dev_families.add(best_family)
        dev_counts = best_counts

    for stratum, total in totals.items():
        if dev_counts[stratum] <= 0 or dev_counts[stratum] >= total:
            raise PaperCContractError(
                f"global family assignment cannot split stratum {stratum} safely")
    assignments = {
        family_id: ("stage2_dev" if family_id in dev_families else "stage2_update")
        for family_id in families
    }

    out: list[dict] = []
    for row in rows:
        sample_id, source, gold, family_id, content_sha = _row_identity(row)
        out.append({
            "sample_id": sample_id,
            "content_sha256": content_sha,
            "family_id": family_id,
            "source": source,
            "gold": gold,
            "stage2_partition": assignments[family_id],
        })
    return out


def build_selections(
    partition_rows: Sequence[Mapping],
    reference_rows: Sequence[Mapping],
    *,
    uncertain_fraction: float,
    seed: int,
) -> list[dict]:
    """Build entropy-high and source/label-matched random selections.

    `reference_rows` must contain one safe/unsafe logit pair for every partition
    row.  Only `stage2_update` rows are eligible.  The random control is sampled
    without overlap from the lower-entropy remainder.
    """
    fraction = float(uncertain_fraction)
    if not (0 < fraction < 0.5):
        raise PaperCContractError("uncertain_fraction must lie in (0, 0.5)")

    by_sample: dict[str, Mapping] = {}
    for record in reference_rows:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in by_sample:
            raise PaperCContractError(
                f"missing or duplicate reference sample_id: {sample_id!r}")
        try:
            float(record["safe_logit"])
            float(record["unsafe_logit"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PaperCContractError(
                f"reference row {sample_id} lacks finite logits") from exc
        by_sample[sample_id] = record

    partition_ids = {str(row.get("sample_id", "")) for row in partition_rows}
    if set(by_sample) != partition_ids:
        missing = sorted(partition_ids - set(by_sample))[:5]
        extra = sorted(set(by_sample) - partition_ids)[:5]
        raise PaperCContractError(
            f"reference/partition identity mismatch: missing={missing}, extra={extra}")

    strata: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in partition_rows:
        if row.get("stage2_partition") not in PARTITIONS:
            raise PaperCContractError("partition row has invalid stage2_partition")
        if row["stage2_partition"] != "stage2_update":
            continue
        sample_id, source, gold, family_id, content_sha = _row_identity(row)
        reference = by_sample[sample_id]
        safe_logit = float(reference["safe_logit"])
        unsafe_logit = float(reference["unsafe_logit"])
        if not math.isfinite(safe_logit) or not math.isfinite(unsafe_logit):
            raise PaperCContractError(f"non-finite reference logits for {sample_id}")
        strata[(source, gold)].append({
            "sample_id": sample_id,
            "content_sha256": content_sha,
            "family_id": family_id,
            "source": source,
            "gold": gold,
            "stage2_partition": "stage2_update",
            "safe_logit": safe_logit,
            "unsafe_logit": unsafe_logit,
            "reference_signed_margin": signed_margin(safe_logit, unsafe_logit, gold),
            "two_verdict_probability_correct": probability_correct(
                safe_logit, unsafe_logit, gold),
            "two_verdict_entropy": two_verdict_entropy(safe_logit, unsafe_logit),
        })

    selected: list[dict] = []
    for stratum, candidates in sorted(strata.items()):
        n_select = int(math.floor(len(candidates) * fraction))
        if n_select < 1:
            raise PaperCContractError(
                f"stratum {stratum} has {len(candidates)} rows; too few for fraction {fraction}")
        uncertain = sorted(
            candidates,
            key=lambda row: (-row["two_verdict_entropy"], row["sample_id"]),
        )[:n_select]
        uncertain_ids = {row["sample_id"] for row in uncertain}
        remainder = [row for row in candidates if row["sample_id"] not in uncertain_ids]
        random_rows = sorted(
            remainder,
            key=lambda row: hashlib.sha256(
                f"{int(seed)}|{row['sample_id']}".encode("utf-8")).hexdigest(),
        )[:n_select]
        if len(random_rows) != n_select:
            raise PaperCContractError(
                f"stratum {stratum} cannot supply a disjoint matched-random sample")

        for role, rows_for_role in (("uncertain", uncertain),
                                    ("matched_random", random_rows)):
            for rank, row in enumerate(rows_for_role, 1):
                record = dict(row)
                record.pop("safe_logit")
                record.pop("unsafe_logit")
                record["selection_role"] = role
                record["selection_rank"] = rank
                selected.append(record)

    validate_selections(selected)
    return sorted(
        selected,
        key=lambda row: (
            SAMPLERS.index(row["selection_role"]), row["source"], row["gold"],
            row["selection_rank"], row["sample_id"],
        ),
    )


def validate_selections(records: Sequence[Mapping]) -> None:
    if not records:
        raise PaperCContractError("selection artifact is empty")
    ids_by_role: dict[str, set[str]] = {role: set() for role in SAMPLERS}
    counts: dict[str, dict[tuple[str, int], int]] = {
        role: defaultdict(int) for role in SAMPLERS
    }
    for row in records:
        role = str(row.get("selection_role", ""))
        if role not in SAMPLERS:
            raise PaperCContractError(f"invalid selection role: {role!r}")
        sample_id, source, gold, _, _ = _row_identity(row)
        if sample_id in ids_by_role[role]:
            raise PaperCContractError(f"duplicate {role} sample_id: {sample_id}")
        ids_by_role[role].add(sample_id)
        counts[role][(source, gold)] += 1
    overlap = ids_by_role["uncertain"] & ids_by_role["matched_random"]
    if overlap:
        raise PaperCContractError(
            f"uncertain and matched-random selections overlap: {sorted(overlap)[:5]}")
    if dict(counts["uncertain"]) != dict(counts["matched_random"]):
        raise PaperCContractError("selection roles are not source/label matched")


def selection_ids(records: Sequence[Mapping], role: str) -> list[str]:
    if role not in SAMPLERS:
        raise PaperCContractError(f"unknown sampler: {role}")
    chosen = [row for row in records if row.get("selection_role") == role]
    chosen.sort(key=lambda row: (
        str(row["source"]), normalize_gold(row.get("gold", row.get("label"))),
        int(row.get("selection_rank", 0)), str(row["sample_id"]),
    ))
    return [str(row["sample_id"]) for row in chosen]


def selection_metadata(
    *,
    config: Mapping,
    train_manifest_sha256: str,
    stage1_adapter_sha256: str,
    reference_sha256: str,
    partition_rows: Sequence[Mapping],
    selection_rows: Sequence[Mapping],
) -> dict:
    validate_config(config)
    validate_selections(selection_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": SELECTION_ALGORITHM_VERSION,
        "config_sha256": canonical_sha256(config),
        "train_manifest_sha256": train_manifest_sha256,
        "stage1_adapter_sha256": stage1_adapter_sha256,
        "reference_sha256": reference_sha256,
        "partition_sha256": sha256_ordered(partition_rows),
        "selection_sha256": sha256_ordered(selection_rows),
        "ordered_ids": {
            role: sha256_ordered(selection_ids(selection_rows, role))
            for role in SAMPLERS
        },
        "counts": {
            role: len(selection_ids(selection_rows, role)) for role in SAMPLERS
        },
    }


def run_condition(objective: str, sampler: str) -> str:
    if objective not in OBJECTIVES:
        raise PaperCContractError(f"unknown objective: {objective}")
    if sampler not in SAMPLERS:
        raise PaperCContractError(f"unknown sampler: {sampler}")
    return f"{objective}__{sampler}"


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / "configs" / "paper_c_dpo_v2.json"
    config = read_json(config_path)
    validate_config(config)
    assert math.isclose(dpo_loss(0.5, 0.5, 0.1), math.log(2.0), abs_tol=1e-12)
    print("paper_c_dpo_common self-check OK", canonical_sha256(config)[:16])
