"""Family-safe Stage-2 partitioning and matched example selection."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from collections.abc import Mapping, Sequence

from .contracts import ContractError, PARTITIONS, SAMPLERS, normalize_gold, row_identity
from .objectives import signed_margin, two_verdict_entropy, two_verdict_probability_unsafe


PARTITION_VERSION = "paper_c_global_family_balanced_v1"
SELECTION_VERSION = "paper_c_stratified_entropy_v1"


def family_partition(
    rows: Sequence[Mapping], *, development_fraction: float, seed: int
) -> list[dict]:
    """Assign each global family once while balancing source/label strata."""
    fraction = float(development_fraction)
    if not 0 < fraction < 0.5:
        raise ContractError("development_fraction must lie in (0,0.5)")
    if not rows:
        raise ContractError("cannot partition an empty manifest")

    families: dict[str, list[Mapping]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        sample_id, _, _, family_id, _ = row_identity(row)
        if sample_id in seen:
            raise ContractError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        families[family_id].append(row)

    family_counts: dict[str, dict[tuple[str, int], int]] = {}
    totals: dict[tuple[str, int], int] = defaultdict(int)
    for family_id, members in families.items():
        counts: dict[tuple[str, int], int] = defaultdict(int)
        for member in members:
            _, source, gold, _, _ = row_identity(member)
            counts[(source, gold)] += 1
            totals[(source, gold)] += 1
        family_counts[family_id] = dict(counts)
    if any(total < 2 for total in totals.values()):
        raise ContractError("every source/label stratum needs at least two rows")

    targets = {
        stratum: min(total - 1, max(1, int(round(total * fraction))))
        for stratum, total in totals.items()
    }
    development_counts: dict[tuple[str, int], int] = defaultdict(int)
    development_families: set[str] = set()

    def imbalance(counts: Mapping[tuple[str, int], int]) -> float:
        value = 0.0
        for stratum, total in totals.items():
            observed = int(counts.get(stratum, 0))
            if observed == 0:
                value += 1000.0
            value += ((observed - targets[stratum]) / float(total)) ** 2
        return value

    while True:
        current = imbalance(development_counts)
        candidates = []
        for family_id, counts in family_counts.items():
            if family_id in development_families:
                continue
            if any(
                development_counts[stratum] + count >= totals[stratum]
                for stratum, count in counts.items()
            ):
                continue
            trial = defaultdict(int, development_counts)
            for stratum, count in counts.items():
                trial[stratum] += count
            tie_key = hashlib.sha256(f"{int(seed)}|{family_id}".encode()).hexdigest()
            candidates.append((imbalance(trial), tie_key, family_id, trial))
        if not candidates:
            break
        best_score, _, best_family, best_counts = min(candidates)
        missing = any(development_counts[stratum] == 0 for stratum in totals)
        if not missing and best_score >= current - 1e-15:
            break
        development_families.add(best_family)
        development_counts = best_counts

    for stratum, total in totals.items():
        observed = development_counts[stratum]
        if observed <= 0 or observed >= total:
            raise ContractError(f"cannot safely split stratum {stratum}")

    assignments = {
        family_id: (
            "stage2_dev" if family_id in development_families else "stage2_update"
        )
        for family_id in families
    }
    output = []
    for row in rows:
        sample_id, source, gold, family_id, content_hash = row_identity(row)
        output.append({
            "sample_id": sample_id,
            "source": source,
            "gold": gold,
            "family_id": family_id,
            "content_sha256": content_hash,
            "stage2_partition": assignments[family_id],
        })
    return output


def _validate_reference(reference_rows: Sequence[Mapping]) -> dict[str, Mapping]:
    by_id: dict[str, Mapping] = {}
    for row in reference_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in by_id:
            raise ContractError(f"missing or duplicate reference ID: {sample_id!r}")
        try:
            safe_logit = float(row["safe_logit"])
            unsafe_logit = float(row["unsafe_logit"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"reference row {sample_id} lacks logits") from exc
        if not math.isfinite(safe_logit) or not math.isfinite(unsafe_logit):
            raise ContractError(f"reference row {sample_id} has non-finite logits")
        by_id[sample_id] = row
    return by_id


def build_selections(
    partition_rows: Sequence[Mapping],
    reference_rows: Sequence[Mapping],
    *,
    uncertain_fraction: float,
    seed: int,
) -> list[dict]:
    fraction = float(uncertain_fraction)
    if not 0 < fraction < 0.5:
        raise ContractError("uncertain_fraction must lie in (0,0.5)")
    reference_by_id = _validate_reference(reference_rows)
    partition_ids = {str(row.get("sample_id", "")) for row in partition_rows}
    if set(reference_by_id) != partition_ids:
        raise ContractError("reference and partition sample IDs differ")

    strata: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in partition_rows:
        if row.get("stage2_partition") not in PARTITIONS:
            raise ContractError("invalid Stage-2 partition")
        if row["stage2_partition"] != "stage2_update":
            continue
        sample_id, source, gold, family_id, content_hash = row_identity(row)
        reference = reference_by_id[sample_id]
        safe_logit = float(reference["safe_logit"])
        unsafe_logit = float(reference["unsafe_logit"])
        prompt_hash = str(reference.get("prompt_sha256", ""))
        if not prompt_hash:
            raise ContractError(f"reference row {sample_id} has no prompt fingerprint")
        p_unsafe = two_verdict_probability_unsafe(safe_logit, unsafe_logit)
        strata[(source, gold)].append({
            "sample_id": sample_id,
            "source": source,
            "gold": gold,
            "family_id": family_id,
            "content_sha256": content_hash,
            "stage2_partition": "stage2_update",
            "prompt_sha256": prompt_hash,
            "reference_signed_margin": signed_margin(safe_logit, unsafe_logit, gold),
            "two_verdict_probability_correct": p_unsafe if gold == 1 else 1.0 - p_unsafe,
            "two_verdict_entropy": two_verdict_entropy(safe_logit, unsafe_logit),
        })

    selected = []
    for stratum, candidates in sorted(strata.items()):
        count = int(math.floor(len(candidates) * fraction))
        if count < 1:
            raise ContractError(f"stratum {stratum} is too small for selection")
        uncertain = sorted(
            candidates, key=lambda row: (-row["two_verdict_entropy"], row["sample_id"])
        )[:count]
        uncertain_ids = {row["sample_id"] for row in uncertain}
        remainder = [row for row in candidates if row["sample_id"] not in uncertain_ids]
        random_rows = sorted(
            remainder,
            key=lambda row: hashlib.sha256(
                f"{int(seed)}|{row['sample_id']}".encode()
            ).hexdigest(),
        )[:count]
        if len(random_rows) != count:
            raise ContractError(f"stratum {stratum} cannot supply a disjoint control")
        for role, role_rows in (("uncertain", uncertain), ("matched_random", random_rows)):
            for rank, row in enumerate(role_rows, 1):
                record = dict(row)
                record["selection_role"] = role
                record["selection_rank"] = rank
                selected.append(record)
    validate_selections(selected)
    return sorted(selected, key=lambda row: (
        SAMPLERS.index(row["selection_role"]),
        row["source"],
        row["gold"],
        row["selection_rank"],
        row["sample_id"],
    ))


def validate_selections(rows: Sequence[Mapping]) -> None:
    if not rows:
        raise ContractError("selection artifact is empty")
    identifiers = {role: set() for role in SAMPLERS}
    counts = {role: defaultdict(int) for role in SAMPLERS}
    for row in rows:
        role = str(row.get("selection_role", ""))
        if role not in SAMPLERS:
            raise ContractError(f"invalid selection role: {role}")
        sample_id, source, gold, _, _ = row_identity(row)
        if sample_id in identifiers[role]:
            raise ContractError(f"duplicate {role} sample: {sample_id}")
        identifiers[role].add(sample_id)
        counts[role][(source, gold)] += 1
    overlap = identifiers["uncertain"] & identifiers["matched_random"]
    if overlap:
        raise ContractError("uncertain and matched-random selections overlap")
    if dict(counts["uncertain"]) != dict(counts["matched_random"]):
        raise ContractError("the two selections are not source/label matched")


def selection_ids(rows: Sequence[Mapping], role: str) -> list[str]:
    if role not in SAMPLERS:
        raise ContractError(f"unknown sampler: {role}")
    selected = [row for row in rows if row.get("selection_role") == role]
    selected.sort(key=lambda row: (
        str(row["source"]),
        normalize_gold(row.get("gold", row.get("label"))),
        int(row.get("selection_rank", 0)),
        str(row["sample_id"]),
    ))
    return [str(row["sample_id"]) for row in selected]


def condition_grid() -> list[dict]:
    return [
        {"objective": objective, "sampler": sampler,
         "condition": f"{objective}__{sampler}"}
        for sampler in SAMPLERS
        for objective in ("verdict_ce", "pair_ce", "dpo")
    ]
