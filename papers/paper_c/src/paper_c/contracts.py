"""Fail-closed configuration, path, identity, and artifact contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


OBJECTIVES = ("verdict_ce", "pair_ce", "dpo")
SAMPLERS = ("uncertain", "matched_random")
PARTITIONS = ("stage2_update", "stage2_dev")
REQUIRED_MANIFESTS = (
    "train.jsonl",
    "calibration.jsonl",
    "id_test.jsonl",
    "transfer_test.jsonl",
    "orbench_safe_stress.jsonl",
    "harmbench_positive_stress.jsonl",
)


class ContractError(ValueError):
    """An input or output violates the study's comparability contract."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ContractError(f"not a directory: {root}")
    records = []
    for child in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append({"path": child.relative_to(root).as_posix(), "sha256": sha256_file(child)})
    if not records:
        raise ContractError(f"directory has no files: {root}")
    return canonical_sha256(records)


def sha256_ordered(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = canonical_json_bytes(value)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContractError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def output_path(value: str | Path, *, root: Path | None = None) -> Path:
    """Resolve an output and reject any location outside this Paper C folder."""
    base = (root or project_root()).resolve()
    raw = Path(value)
    candidate = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
    if not _inside(base, candidate):
        raise ContractError(f"Paper C output escapes isolated workspace: {value}")
    return candidate


def read_path(value: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a read-only input; parent repository inputs may live outside."""
    base = (root or project_root()).resolve()
    raw = Path(value)
    return raw.resolve() if raw.is_absolute() else (base / raw).resolve()


def write_json(path: str | Path, value: object) -> None:
    target = output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def write_jsonl(path: str | Path, rows: Iterable[Mapping]) -> None:
    target = output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False, allow_nan=False))
            handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def normalize_gold(value: object) -> int:
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
    raise ContractError(f"invalid binary guard label: {value!r}")


def row_text(row: Mapping) -> str:
    for key in ("text", "prompt", "user_input", "text_or_download_reference"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ContractError(f"row {row.get('sample_id')!r} has no scoreable text")


def row_identity(row: Mapping) -> tuple[str, str, int, str, str]:
    try:
        sample_id = str(row["sample_id"])
        source = str(row["source"])
        family_id = str(row["family_id"])
        content_hash = str(row["content_sha256"])
        gold = normalize_gold(row.get("gold", row.get("label")))
    except KeyError as exc:
        raise ContractError(f"row missing identity field: {exc.args[0]}") from exc
    if not all((sample_id, source, family_id, content_hash)):
        raise ContractError("row identity fields may not be empty")
    return sample_id, source, gold, family_id, content_hash


def load_config(path: str | Path) -> dict:
    config = read_json(output_path(path))
    validate_config(config)
    return config


def _positive_int(mapping: Mapping, field: str) -> int:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field} must be a positive integer")
    return value


def _reject_unknown(mapping: Mapping, allowed: set[str], context: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ContractError(f"unknown {context} fields: {sorted(unknown)}")


def validate_config(config: Mapping) -> None:
    _reject_unknown(config, {
        "schema_version", "study_id", "project_root_marker", "artifact_root",
        "input_root", "parent_sources", "models", "seeds", "prompt", "stage1",
        "stage2", "analysis", "confirmatory", "cloud",
    }, "top-level config")
    if int(config.get("schema_version", -1)) != 1:
        raise ContractError("schema_version must be 1")
    if not str(config.get("study_id", "")).strip():
        raise ContractError("study_id is required")
    marker = str(config.get("project_root_marker", ""))
    if not marker or Path(marker).is_absolute() or ".." in Path(marker).parts:
        raise ContractError("project_root_marker must be a safe workspace-relative path")
    if not output_path(marker).is_file():
        raise ContractError("project_root_marker does not exist")
    for field in ("artifact_root", "input_root"):
        value = str(config.get(field, ""))
        if not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ContractError(f"{field} must remain inside the Paper C workspace")
        output_path(value)
    parent_sources = config.get("parent_sources")
    if not isinstance(parent_sources, Mapping):
        raise ContractError("parent_sources is required")
    _reject_unknown(parent_sources, {"lock", "manifests"}, "parent_sources")
    if any(not str(parent_sources.get(field, "")).strip() for field in ("lock", "manifests")):
        raise ContractError("parent lock and manifest source paths are required")
    models = config.get("models")
    if not isinstance(models, Mapping) or len(models) != 4:
        raise ContractError("the primary panel must contain exactly four models")
    for model_key, model in models.items():
        if not isinstance(model, Mapping):
            raise ContractError(f"invalid model record: {model_key}")
        _reject_unknown(model, {
            "model_id", "model_revision", "tokenizer_revision", "trust_remote_code",
            "dtype", "safe_token_id", "unsafe_token_id",
        }, f"model {model_key}")
        for field in ("model_id", "model_revision", "tokenizer_revision", "dtype"):
            if not str(model.get(field, "")).strip():
                raise ContractError(f"{model_key}.{field} is required")
        safe_id = model.get("safe_token_id")
        unsafe_id = model.get("unsafe_token_id")
        if not isinstance(safe_id, int) or not isinstance(unsafe_id, int) or safe_id == unsafe_id:
            raise ContractError(f"{model_key} has invalid verdict token IDs")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 5 or len(set(seeds)) != 5:
        raise ContractError("the primary design requires five unique seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ContractError("all seeds must be integers")
    stage1 = config.get("stage1")
    stage2 = config.get("stage2")
    if not isinstance(stage1, Mapping) or not isinstance(stage2, Mapping):
        raise ContractError("stage1 and stage2 configuration objects are required")
    _reject_unknown(stage1, {
        "learning_rate", "max_steps", "per_device_batch", "gradient_accumulation",
        "scheduler", "warmup_ratio", "data_order_seed", "lora",
    }, "stage1")
    _reject_unknown(stage2, {
        "objectives", "samplers", "development_fraction", "development_split_seed",
        "uncertain_fraction", "selection_seed", "learning_rate", "max_steps",
        "checkpoint_steps", "per_device_batch", "gradient_accumulation", "scheduler",
        "warmup_ratio", "beta", "dropout", "reference_margin_atol",
        "represented_noninferiority_margin", "checkpoint_rule", "pilot",
    }, "stage2")
    pilot = stage2.get("pilot")
    if not isinstance(pilot, Mapping):
        raise ContractError(
            "stage2.pilot is required: the full panel must be gated on a measured effect size and "
            "a measured pairing variance reduction, not an assumed one"
        )
    _reject_unknown(pilot, {"models", "seeds", "gate", "purpose"}, "stage2.pilot")
    pilot_models = pilot.get("models")
    if not isinstance(pilot_models, list) or not 1 <= len(pilot_models) < len(models):
        raise ContractError("stage2.pilot.models must be a strict, nonempty subset of the panel")
    if any(key not in models for key in pilot_models):
        raise ContractError("every stage2.pilot model must exist in the panel")
    if pilot.get("gate") != "measure_effect_and_pairing_variance_before_full_panel":
        raise ContractError("stage2.pilot.gate must state the measurement it authorises")
    lora = stage1.get("lora")
    if not isinstance(lora, Mapping):
        raise ContractError("Stage-1 LoRA configuration is required")
    _reject_unknown(lora, {"r", "alpha", "dropout", "target_modules"}, "stage1.lora")
    _positive_int(lora, "r")
    _positive_int(lora, "alpha")
    if not isinstance(lora.get("target_modules"), list) or not lora["target_modules"]:
        raise ContractError("Stage-1 LoRA target_modules must be nonempty")
    if tuple(stage2.get("objectives", ())) != OBJECTIVES:
        raise ContractError(f"stage2 objectives must be exactly {OBJECTIVES}")
    if tuple(stage2.get("samplers", ())) != SAMPLERS:
        raise ContractError(f"stage2 samplers must be exactly {SAMPLERS}")
    for field in ("development_fraction", "uncertain_fraction"):
        value = float(stage2.get(field, -1))
        if not 0 < value < 0.5:
            raise ContractError(f"{field} must lie in (0, 0.5)")
    beta = float(stage2.get("beta", -1))
    if not math.isfinite(beta) or beta <= 0:
        raise ContractError("stage2 beta must be finite and positive")
    if float(stage2.get("dropout", -1)) != 0.0:
        raise ContractError("Stage-2 dropout must be exactly zero")
    for section in (stage1, stage2):
        _positive_int(section, "max_steps")
        _positive_int(section, "per_device_batch")
        _positive_int(section, "gradient_accumulation")
        learning_rate = float(section.get("learning_rate", -1))
        warmup = float(section.get("warmup_ratio", -1))
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ContractError("learning_rate must be finite and positive")
        if not math.isfinite(warmup) or not 0 <= warmup < 1:
            raise ContractError("warmup_ratio must lie in [0,1)")
    checkpoints = list(stage2.get("checkpoint_steps", ()))
    if checkpoints != sorted(set(checkpoints)) or not checkpoints:
        raise ContractError("checkpoint_steps must be a sorted unique list")
    if checkpoints[-1] != stage2["max_steps"]:
        raise ContractError("last checkpoint must equal Stage-2 max_steps")
    expected_runs = len(models) * len(seeds) * len(OBJECTIVES) * len(SAMPLERS)
    if expected_runs != 120:
        raise ContractError("the primary Stage-2 panel must contain 120 cells")
    prompt = config.get("prompt")
    if not isinstance(prompt, Mapping):
        raise ContractError("prompt configuration is required")
    _reject_unknown(prompt, {
        "version", "system", "max_length", "frozen_token_cache_required",
        "rerender_after_design_lock",
    }, "prompt")
    if prompt.get("frozen_token_cache_required") is not True:
        raise ContractError("frozen prompt token caches are required")
    if prompt.get("rerender_after_design_lock") is not False:
        raise ContractError("prompt rerendering after the design lock must be disabled")
    _positive_int(prompt, "max_length")
    analysis = config.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ContractError("analysis configuration is required")
    _reject_unknown(analysis, {
        "primary_metric", "primary_reference_contrast", "bootstrap_replicates",
        "bootstrap_seed", "target_fpr", "infeasible_cell_policy",
        "candidate_test_scoring", "primary_estimand", "secondary_estimands",
        "bootstrap_cluster_unit", "power",
    }, "analysis")
    _positive_int(analysis, "bootstrap_replicates")
    if analysis.get("primary_reference_contrast") != (
        "equal_weight_factorial_marginal_over_uncertain_and_matched_random"
    ):
        raise ContractError("the primary reference-centering estimand must be factorial-marginal")
    # The primary estimand must be the frontier form. A step-matched or single-checkpoint AP
    # difference is confounded with the effective-learning-rate change that reference centering
    # induces at initialisation (see power.effective_learning_rate_ratio), so it may only ever be
    # secondary.
    if analysis.get("primary_estimand") != "frontier_transfer_gap_at_matched_represented":
        raise ContractError(
            "analysis.primary_estimand must be "
            "'frontier_transfer_gap_at_matched_represented'; point-in-step AP differences are "
            "confounded with effective learning rate and are secondary only"
        )
    # Seeds inside a checkpoint share model, manifest, recipe and data order, so the bootstrap must
    # resample checkpoints. Resampling cells would understate the interval by roughly sqrt(seeds).
    if analysis.get("bootstrap_cluster_unit") != "model_key":
        raise ContractError("analysis.bootstrap_cluster_unit must be 'model_key'")
    power = analysis.get("power")
    if not isinstance(power, Mapping):
        raise ContractError("analysis.power is required: the design must state what it can detect")
    _reject_unknown(power, {
        "target_effect_transfer", "assumed_pairing_variance_reduction",
        "seed_sd_source", "gate_blocks_stage2",
    }, "analysis.power")
    target = float(power.get("target_effect_transfer", -1))
    if not math.isfinite(target) or not 0 < target < 0.5:
        raise ContractError("analysis.power.target_effect_transfer must lie in (0, 0.5)")
    reduction = float(power.get("assumed_pairing_variance_reduction", -1))
    if not math.isfinite(reduction) or not 0 < reduction <= 1:
        raise ContractError("analysis.power.assumed_pairing_variance_reduction must lie in (0, 1]")
    if power.get("gate_blocks_stage2") is not True:
        raise ContractError(
            "analysis.power.gate_blocks_stage2 must be true: an underpowered design has to block "
            "the full panel rather than be discovered in the discussion"
        )
    confirmatory = config.get("confirmatory")
    if not isinstance(confirmatory, Mapping):
        raise ContractError("confirmatory gate configuration is required")
    _reject_unknown(confirmatory, {
        "enabled", "requires_prospective_child_lock", "reference_transfer_superiority_margin",
        "represented_noninferiority_margin", "orbench_false_positive_absolute_harm_margin",
        "harmbench_recall_absolute_harm_margin", "simultaneous_one_sided_confidence",
        "incomplete_primary_panel_can_pass",
    }, "confirmatory")
    if confirmatory.get("requires_prospective_child_lock") is not True:
        raise ContractError("confirmatory analysis requires a prospective child lock")
    if confirmatory.get("incomplete_primary_panel_can_pass") is not False:
        raise ContractError("an incomplete primary panel may not pass")
    cloud = config.get("cloud")
    if not isinstance(cloud, Mapping):
        raise ContractError("cloud configuration is required")
    _reject_unknown(cloud, {
        "default_zone", "default_machine_type", "default_accelerator",
        "boot_disk_gb", "provisioning_model",
    }, "cloud")


def self_hashed(value: Mapping, hash_field: str = "lock_sha256") -> dict:
    out = dict(value)
    out.pop(hash_field, None)
    out[hash_field] = canonical_sha256(out)
    return out


def validate_self_hash(value: Mapping, hash_field: str = "lock_sha256") -> None:
    expected = value.get(hash_field)
    payload = {key: item for key, item in value.items() if key != hash_field}
    if not isinstance(expected, str) or canonical_sha256(payload) != expected:
        raise ContractError(f"invalid {hash_field}")
