"""Layered design, selection, and prospective lock contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping

from .contracts import (
    ContractError,
    OBJECTIVES,
    SAMPLERS,
    canonical_sha256,
    load_config,
    output_path,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_directory,
    sha256_ordered,
    self_hashed,
    validate_self_hash,
    write_json,
)
from .inputs import validate_input_manifest


PROTOCOL_LOCK_PATH = Path("artifacts/locks/PROTOCOL_LOCK.json")
DESIGN_LOCK_PATH = Path("artifacts/locks/STAGE2_DESIGN_LOCK.json")
SELECTION_LOCK_PATH = Path("artifacts/locks/SELECTION_LOCK.json")


def _write_new_lock(path: Path, lock: dict) -> None:
    target = output_path(path)
    if target.exists():
        raise ContractError(f"refusing to overwrite immutable lock: {target}")
    write_json(path, lock)


def _source_inventory() -> dict:
    root = output_path(".")
    include_roots = (
        root / "src",
        root / "config",
        root / "cloud",
        root / "tests",
        root / "manuscript",
    )
    records = []
    for name in ("PROTOCOL.md", "DEVELOPMENT_PLAN.md", "README.md", "pyproject.toml", "Makefile"):
        path = root / name
        if path.is_file():
            records.append({"path": name, "sha256": sha256_file(path)})
    for include_root in include_roots:
        if not include_root.exists():
            continue
        for path in sorted(item for item in include_root.rglob("*") if item.is_file()):
            if any(
                part in {"__pycache__", "build", "bundle", "generated"}
                for part in path.parts
            ):
                continue
            records.append({
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            })
    if not records:
        raise ContractError("execution-source inventory is empty")
    return {"files": records, "aggregate_sha256": sha256_ordered(records)}


def validate_source_inventory(inventory: Mapping) -> None:
    root = output_path(".")
    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        raise ContractError("lock execution-source inventory is empty")
    rebound = []
    for record in files:
        path = output_path(str(record.get("path", "")))
        observed = sha256_file(path)
        if observed != record.get("sha256"):
            raise ContractError(f"execution source drifted: {record.get('path')}")
        rebound.append({"path": record["path"], "sha256": observed})
    if sha256_ordered(rebound) != inventory.get("aggregate_sha256"):
        raise ContractError("execution-source aggregate hash mismatch")
    if _source_inventory() != dict(inventory):
        raise ContractError("execution-source inventory gained, lost, or changed a file")


def validate_recorded_source_inventory(inventory: Mapping) -> None:
    """Validate a historical inventory internally without rebinding it to the live tree."""
    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        raise ContractError("historical execution-source inventory is empty")
    if sha256_ordered(files) != inventory.get("aggregate_sha256"):
        raise ContractError("historical execution-source aggregate hash mismatch")


def create_protocol_lock(
    config_path: str | Path,
    *,
    out_path: str | Path = PROTOCOL_LOCK_PATH,
    supersedes_path: str | Path | None = None,
) -> dict:
    config = load_config(config_path)
    inputs = validate_input_manifest(config)
    payload = {
        "lock_schema_version": 2 if supersedes_path is not None else 1,
        "lock_kind": "protocol_lock",
        "study_id": config["study_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "stage1_candidate_generation_authorized",
        "claim_scope": "development_and_stage1_candidate_generation_only",
        "config": {
            "path": output_path(config_path).relative_to(output_path(".")).as_posix(),
            "sha256": sha256_file(output_path(config_path)),
            "object_sha256": canonical_sha256(config),
            "value": config,
        },
        "input_manifest": {
            "path": str(Path(config["input_root"]) / "INPUT_MANIFEST.json"),
            "sha256": sha256_file(output_path(Path(config["input_root"]) / "INPUT_MANIFEST.json")),
            "self_hash": inputs["input_manifest_sha256"],
        },
        "execution_sources": _source_inventory(),
        "required_children": ["stage2_design_lock", "selection_lock", "prospective_lock"],
        "stage2_candidate_training_authorized": False,
        "retrospective_scoring_authorized": False,
        "confirmatory_scoring_authorized": False,
    }
    if supersedes_path is not None:
        previous_path = output_path(supersedes_path)
        previous = read_json(previous_path)
        validate_self_hash(previous)
        if previous.get("lock_kind") != "protocol_lock":
            raise ContractError("a protocol lock can supersede only another protocol lock")
        if previous.get("study_id") != config["study_id"]:
            raise ContractError("superseded protocol lock belongs to a different study")
        if (previous.get("config") or {}).get("object_sha256") != canonical_sha256(config):
            raise ContractError("superseded protocol lock used a different configuration")
        if canonical_sha256((previous.get("config") or {}).get("value")) != (
            previous.get("config") or {}
        ).get("object_sha256"):
            raise ContractError("superseded protocol-lock configuration record is corrupt")
        validate_recorded_source_inventory(previous.get("execution_sources") or {})
        payload["supersedes"] = {
            "path": previous_path.relative_to(output_path(".")).as_posix(),
            "sha256": sha256_file(previous_path),
            "self_hash": previous["lock_sha256"],
            "reason": "execution_source_safety_amendment_before_any_gpu_run",
        }
        payload["amendment"] = {
            "scope": "cloud_safety_and_lock_completeness_only",
            "scientific_config_changed": False,
            "gpu_run_occurred_before_amendment": False,
        }
    lock = self_hashed(payload)
    _write_new_lock(out_path, lock)
    return lock


def validate_protocol_lock(path: str | Path = PROTOCOL_LOCK_PATH) -> dict:
    lock = read_json(output_path(path))
    validate_self_hash(lock)
    if lock.get("lock_kind") != "protocol_lock":
        raise ContractError("not a Paper C protocol lock")
    config = lock.get("config") or {}
    config_path = output_path(str(config.get("path", "")))
    if sha256_file(config_path) != config.get("sha256"):
        raise ContractError("protocol-lock configuration bytes drifted")
    if canonical_sha256(load_config(config_path)) != config.get("object_sha256"):
        raise ContractError("protocol-lock configuration object drifted")
    input_record = lock.get("input_manifest") or {}
    input_path = output_path(str(input_record.get("path", "")))
    if sha256_file(input_path) != input_record.get("sha256"):
        raise ContractError("protocol-lock input manifest drifted")
    validate_input_manifest(config["value"])
    validate_source_inventory(lock.get("execution_sources") or {})
    supersedes = lock.get("supersedes")
    if supersedes is not None:
        previous_path = output_path(str(supersedes.get("path", "")))
        if sha256_file(previous_path) != supersedes.get("sha256"):
            raise ContractError("superseded protocol-lock bytes drifted")
        previous = read_json(previous_path)
        validate_self_hash(previous)
        validate_recorded_source_inventory(previous.get("execution_sources") or {})
        if previous.get("lock_sha256") != supersedes.get("self_hash"):
            raise ContractError("superseded protocol-lock self-hash mismatch")
        if previous.get("study_id") != lock.get("study_id"):
            raise ContractError("superseded protocol lock belongs to a different study")
    return lock


def _cell_key(row: Mapping) -> tuple[str, int]:
    try:
        return str(row["model_key"]), int(row["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("invalid model/seed inventory row") from exc


def _expected_stage1_cells(config: Mapping) -> set[tuple[str, int]]:
    return {(model, int(seed)) for model in config["models"] for seed in config["seeds"]}


def _validate_stage1_inventory(rows: list[dict], config: Mapping) -> None:
    expected = _expected_stage1_cells(config)
    observed = {_cell_key(row) for row in rows}
    if len(observed) != len(rows) or observed != expected or len(observed) != 20:
        raise ContractError("Stage-1 inventory must contain the exact 20-cell grid")
    for row in rows:
        adapter = output_path(str(row.get("adapter_path", "")))
        run_metadata = output_path(str(row.get("run_metadata_path", "")))
        prompt_cache = output_path(str(row.get("prompt_cache_path", "")))
        if not adapter.is_dir() or sha256_directory(adapter) != row.get("adapter_sha256"):
            raise ContractError(f"Stage-1 adapter drifted: {adapter}")
        if not run_metadata.is_file() or sha256_file(run_metadata) != row.get("run_metadata_sha256"):
            raise ContractError(f"Stage-1 run metadata drifted: {run_metadata}")
        if not prompt_cache.is_file() or sha256_file(prompt_cache) != row.get("prompt_cache_sha256"):
            raise ContractError(f"Stage-1 prompt cache drifted: {prompt_cache}")


def _validate_stage2_input_inventory(rows: list[dict], config: Mapping) -> None:
    expected = _expected_stage1_cells(config)
    observed = {_cell_key(row) for row in rows}
    if len(observed) != len(rows) or observed != expected or len(observed) != 20:
        raise ContractError("Stage-2 input inventory must contain the exact 20-cell grid")
    for row in rows:
        for prefix in ("reference", "selection"):
            artifact = output_path(str(row.get(f"{prefix}_path", "")))
            if not artifact.is_file() or sha256_file(artifact) != row.get(f"{prefix}_sha256"):
                raise ContractError(f"Stage-2 {prefix} artifact drifted: {artifact}")


def create_design_lock(
    *,
    protocol_lock_path: str | Path,
    stage1_inventory_path: str | Path,
    stage2_input_inventory_path: str | Path,
    partition_path: str | Path,
) -> dict:
    protocol = validate_protocol_lock(protocol_lock_path)
    config = protocol["config"]["value"]
    stage1_path = output_path(stage1_inventory_path)
    stage2_path = output_path(stage2_input_inventory_path)
    partition = output_path(partition_path)
    stage1_rows = read_jsonl(stage1_path)
    stage2_rows = read_jsonl(stage2_path)
    _validate_stage1_inventory(stage1_rows, config)
    _validate_stage2_input_inventory(stage2_rows, config)
    if not partition.is_file():
        raise ContractError("Stage-2 family partition is missing")
    lock = self_hashed({
        "lock_schema_version": 1,
        "lock_kind": "stage2_design_lock",
        "study_id": protocol["study_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "stage2_candidate_training_authorized",
        "claim_scope": "candidate_training_and_development_scoring_only",
        "parent_protocol_lock": {
            "path": output_path(protocol_lock_path).relative_to(output_path(".")).as_posix(),
            "sha256": sha256_file(output_path(protocol_lock_path)),
            "self_hash": protocol["lock_sha256"],
        },
        "config": protocol["config"],
        "stage1_inventory": {
            "path": stage1_path.relative_to(output_path(".")).as_posix(),
            "sha256": sha256_file(stage1_path),
            "rows": len(stage1_rows),
        },
        "stage2_input_inventory": {
            "path": stage2_path.relative_to(output_path(".")).as_posix(),
            "sha256": sha256_file(stage2_path),
            "rows": len(stage2_rows),
        },
        "stage2_partition": {
            "path": partition.relative_to(output_path(".")).as_posix(),
            "sha256": sha256_file(partition),
        },
        "stage2_candidate_training_authorized": True,
        "retrospective_scoring_authorized": False,
        "confirmatory_scoring_authorized": False,
    })
    _write_new_lock(DESIGN_LOCK_PATH, lock)
    return lock


def validate_design_lock(path: str | Path = DESIGN_LOCK_PATH) -> dict:
    lock = read_json(output_path(path))
    validate_self_hash(lock)
    if lock.get("lock_kind") != "stage2_design_lock":
        raise ContractError("not a Paper C Stage-2 design lock")
    parent = lock.get("parent_protocol_lock") or {}
    parent_path = output_path(str(parent.get("path", "")))
    if sha256_file(parent_path) != parent.get("sha256"):
        raise ContractError("Stage-2 design-lock parent bytes drifted")
    protocol = validate_protocol_lock(parent_path)
    if protocol.get("lock_sha256") != parent.get("self_hash"):
        raise ContractError("Stage-2 design-lock parent self-hash mismatch")
    config = protocol["config"]["value"]
    stage1_record = lock.get("stage1_inventory") or {}
    stage2_record = lock.get("stage2_input_inventory") or {}
    partition_record = lock.get("stage2_partition") or {}
    stage1_path = output_path(str(stage1_record.get("path", "")))
    stage2_path = output_path(str(stage2_record.get("path", "")))
    partition_path = output_path(str(partition_record.get("path", "")))
    if sha256_file(stage1_path) != stage1_record.get("sha256"):
        raise ContractError("Stage-1 inventory drifted after the Stage-2 design lock")
    if sha256_file(stage2_path) != stage2_record.get("sha256"):
        raise ContractError("Stage-2 input inventory drifted after the design lock")
    if sha256_file(partition_path) != partition_record.get("sha256"):
        raise ContractError("Stage-2 partition drifted after the design lock")
    _validate_stage1_inventory(read_jsonl(stage1_path), config)
    _validate_stage2_input_inventory(read_jsonl(stage2_path), config)
    return lock


def _candidate_key(row: Mapping) -> tuple[str, int, str, str, int]:
    try:
        return (
            str(row["model_key"]),
            int(row["seed"]),
            str(row["sampler"]),
            str(row["objective"]),
            int(row["step"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("invalid candidate-inventory row") from exc


def _validate_candidate_inventory(rows: list[dict], design_lock: dict) -> None:
    config = design_lock["config"]["value"]
    expected = {
        (model, seed, sampler, objective, step)
        for model in config["models"]
        for seed in config["seeds"]
        for sampler in SAMPLERS
        for objective in OBJECTIVES
        for step in config["stage2"]["checkpoint_steps"]
    }
    observed = {_candidate_key(row) for row in rows}
    if len(observed) != len(rows):
        raise ContractError("candidate inventory contains duplicate cells")
    if observed != expected or len(observed) != 480:
        raise ContractError("candidate inventory must contain the exact 480-checkpoint grid")
    for row in rows:
        adapter_path = output_path(str(row.get("adapter_path", "")))
        if not adapter_path.is_dir():
            raise ContractError(f"candidate adapter is missing: {adapter_path}")
        expected_hash = str(row.get("adapter_sha256", ""))
        if not expected_hash or sha256_directory(adapter_path) != expected_hash:
            raise ContractError(f"candidate adapter hash mismatch: {adapter_path}")


def _validate_selection_table(rows: list[dict], design_lock: dict) -> None:
    config = design_lock["config"]["value"]
    expected = {
        (model, seed, sampler, objective)
        for model in config["models"]
        for seed in config["seeds"]
        for sampler in SAMPLERS
        for objective in OBJECTIVES
    }
    observed = set()
    allowed_steps = set(config["stage2"]["checkpoint_steps"])
    for row in rows:
        try:
            key = (
                str(row["model_key"]), int(row["seed"]),
                str(row["sampler"]), str(row["objective"]),
            )
            step = int(row["selected_step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("invalid checkpoint-selection row") from exc
        status = str(row.get("selection_status", ""))
        if step not in allowed_steps:
            raise ContractError("selected checkpoint step is outside the locked candidates")
        if status not in {"target_reached", "target_infeasible"}:
            raise ContractError("checkpoint selection has an invalid status")
        if status == "target_infeasible" and step != max(allowed_steps):
            raise ContractError("an infeasible cell must select only the max-step descriptive fallback")
        if bool(row.get("eligible_for_primary_target_matched_contrast")) != (status == "target_reached"):
            raise ContractError("checkpoint-selection eligibility disagrees with its status")
        if key in observed:
            raise ContractError("checkpoint-selection table has duplicate cells")
        observed.add(key)
    if observed != expected or len(observed) != 120:
        raise ContractError("checkpoint-selection table must contain the exact 120-cell grid")


def _validate_development_score_inventory(rows: list[dict], design_lock: dict) -> None:
    config = design_lock["config"]["value"]
    expected = {
        ("stage1", model, int(seed), None, None, None)
        for model in config["models"] for seed in config["seeds"]
    }
    expected.update({
        ("candidate", model, int(seed), sampler, objective, int(step))
        for model in config["models"]
        for seed in config["seeds"]
        for sampler in SAMPLERS
        for objective in OBJECTIVES
        for step in config["stage2"]["checkpoint_steps"]
    })
    observed = set()
    for row in rows:
        kind = str(row.get("score_kind", ""))
        try:
            key = (
                kind,
                str(row["model_key"]),
                int(row["seed"]),
                str(row["sampler"]) if kind == "candidate" else None,
                str(row["objective"]) if kind == "candidate" else None,
                int(row["step"]) if kind == "candidate" else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("invalid development-score inventory row") from exc
        if key in observed:
            raise ContractError("development-score inventory contains duplicate bundles")
        observed.add(key)
        score_path = output_path(str(row.get("score_path", "")))
        if not score_path.is_file() or sha256_file(score_path) != row.get("score_sha256"):
            raise ContractError(f"development score bundle drifted: {score_path}")
    if observed != expected or len(observed) != 500:
        raise ContractError("development-score inventory must contain exactly 20 Stage-1 and 480 candidate bundles")


def _selected_adapter_inventory(candidates: list[dict], selections: list[dict]) -> list[dict]:
    by_key = {_candidate_key(row): row for row in candidates}
    output = []
    for row in selections:
        key = (
            str(row["model_key"]), int(row["seed"]), str(row["sampler"]),
            str(row["objective"]), int(row["selected_step"]),
        )
        candidate = by_key.get(key)
        if candidate is None:
            raise ContractError(f"selected checkpoint is absent from candidate inventory: {key}")
        output.append({
            "model_key": key[0],
            "seed": key[1],
            "sampler": key[2],
            "objective": key[3],
            "step": key[4],
            "selection_status": row["selection_status"],
            "adapter_path": candidate["adapter_path"],
            "adapter_sha256": candidate["adapter_sha256"],
        })
    output.sort(key=lambda row: (
        row["model_key"], row["seed"], row["sampler"], row["objective"]
    ))
    if len(output) != 120:
        raise ContractError("selected adapter inventory must contain exactly 120 cells")
    return output


def create_selection_lock(
    *,
    design_lock_path: str | Path,
    candidate_inventory_path: str | Path,
    development_scores_path: str | Path,
    selection_table_path: str | Path,
) -> dict:
    design_lock = validate_design_lock(design_lock_path)
    candidates_path = output_path(candidate_inventory_path)
    scores_path = output_path(development_scores_path)
    selections_path = output_path(selection_table_path)
    candidates = read_jsonl(candidates_path)
    selections = read_jsonl(selections_path)
    _validate_candidate_inventory(candidates, design_lock)
    _validate_selection_table(selections, design_lock)
    development_scores = read_jsonl(scores_path)
    _validate_development_score_inventory(development_scores, design_lock)
    selected_adapters = _selected_adapter_inventory(candidates, selections)
    lock = self_hashed({
        "lock_schema_version": 1,
        "lock_kind": "postselection_retrospective_lock",
        "study_id": design_lock["study_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "retrospective_scoring_authorized",
        "parent_design_lock": {
            "path": output_path(design_lock_path).relative_to(output_path(".")).as_posix(),
            "sha256": sha256_file(output_path(design_lock_path)),
            "self_hash": design_lock["lock_sha256"],
        },
        "candidate_inventory": {
            "path": candidates_path.relative_to(output_path(".")).as_posix(),
            "sha256": sha256_file(candidates_path),
            "rows": len(candidates),
        },
        "development_scores": {
            "path": scores_path.relative_to(output_path(".")).as_posix(),
            "sha256": sha256_file(scores_path),
            "rows": len(development_scores),
        },
        "selection_table": {
            "path": selections_path.relative_to(output_path(".")).as_posix(),
            "sha256": sha256_file(selections_path),
            "rows": len(selections),
        },
        "selected_adapters": {
            "rows": 120,
            "aggregate_sha256": sha256_ordered(selected_adapters),
            "records": selected_adapters,
        },
        "retrospective_scoring_authorized": True,
        "confirmatory_scoring_authorized": False,
    })
    _write_new_lock(SELECTION_LOCK_PATH, lock)
    return lock


def create_prospective_lock(*_args, **_kwargs):
    raise ContractError(
        "prospective lock creation is disabled until a sealed, uninspected cohort and unsealing procedure exist"
    )


def validate_selection_lock(path: str | Path = SELECTION_LOCK_PATH) -> dict:
    lock_path = output_path(path)
    lock = read_json(lock_path)
    validate_self_hash(lock)
    if lock.get("lock_kind") != "postselection_retrospective_lock":
        raise ContractError("not a Paper C selection lock")
    parent = lock.get("parent_design_lock") or {}
    parent_path = output_path(str(parent.get("path", "")))
    if sha256_file(parent_path) != parent.get("sha256"):
        raise ContractError("selection-lock parent bytes drifted")
    design = validate_design_lock(parent_path)
    if design.get("lock_sha256") != parent.get("self_hash"):
        raise ContractError("selection-lock parent self-hash mismatch")
    for field in ("candidate_inventory", "development_scores", "selection_table"):
        record = lock.get(field) or {}
        artifact = output_path(str(record.get("path", "")))
        if not artifact.is_file() or sha256_file(artifact) != record.get("sha256"):
            raise ContractError(f"selection-lock artifact drifted: {field}")
    _validate_candidate_inventory(read_jsonl(output_path(lock["candidate_inventory"]["path"])), design)
    _validate_development_score_inventory(
        read_jsonl(output_path(lock["development_scores"]["path"])), design
    )
    _validate_selection_table(read_jsonl(output_path(lock["selection_table"]["path"])), design)
    expected_selected = _selected_adapter_inventory(
        read_jsonl(output_path(lock["candidate_inventory"]["path"])),
        read_jsonl(output_path(lock["selection_table"]["path"])),
    )
    selected_record = lock.get("selected_adapters") or {}
    if selected_record.get("rows") != 120 or selected_record.get("records") != expected_selected:
        raise ContractError("selection-lock selected adapter inventory drifted")
    if selected_record.get("aggregate_sha256") != sha256_ordered(expected_selected):
        raise ContractError("selection-lock selected adapter aggregate hash mismatch")
    return lock
