"""Candidate protocol lock for the scientifically new v2 study."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .contracts import (
    ContractError,
    HEX64,
    canonical_sha256,
    load_config,
    output_path,
    read_json,
    sha256_file,
    sha256_ordered,
    validate_config,
)


DEFAULT_LOCK = Path("locks/PROTOCOL_TAXONOMY_CANDIDATE.json")
LOCK_STATUS = "candidate_only_no_data_or_training_authorized"
REQUIRED_CHILDREN = (
    "pilot_data_policy_lock",
    "pilot_specialist_calibration_lock",
    "pilot_adjudicated_preference_lock",
    "post_pilot_prospective_primary_protocol_lock",
    "primary_data_policy_lock",
    "primary_adjudicated_preference_lock",
    "primary_aligned_candidate_lock",
    "primary_checkpoint_selection_lock",
    "sealed_confirmation_lock",
)


def source_inventory(*, root: Path | None = None) -> dict:
    root = (root or output_path(".")).resolve()
    records: list[dict[str, str]] = []
    top_level = (
        "README.md", "STATUS.md", "PROTOCOL.md", "DEVELOPMENT_PLAN.md",
        "pyproject.toml", "Makefile", ".gitignore",
    )
    for name in top_level:
        path = root / name
        if path.is_file():
            records.append({"path": name, "sha256": sha256_file(path)})
    for name in ("config", "schemas", "src", "tests", "manuscript", "provenance"):
        include = root / name
        if not include.exists():
            continue
        for path in sorted(item for item in include.rglob("*") if item.is_file()):
            if any(part in {"__pycache__", "build", ".pytest_cache"} for part in path.parts):
                continue
            records.append({
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            })
    if not records:
        raise ContractError("source inventory is empty")
    return {"files": records, "aggregate_sha256": sha256_ordered(records)}


def _self_hashed(payload: Mapping) -> dict:
    value = dict(payload)
    value["lock_sha256"] = canonical_sha256(value)
    return value


def _validate_self_hash(lock: Mapping) -> None:
    payload = dict(lock)
    observed = payload.pop("lock_sha256", None)
    if observed != canonical_sha256(payload):
        raise ContractError("lock self-hash mismatch")


def _write_exclusive_json(path: Path, value: object) -> None:
    """Create a lock exactly once; O_EXCL closes the check-then-replace race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ContractError(f"refusing to overwrite immutable lock: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def create_protocol_candidate(
    config_path: str | Path = "config/study.json",
    *,
    out_path: str | Path = DEFAULT_LOCK,
) -> dict:
    config = load_config(config_path)
    if config.get("profile") != "primary":
        raise ContractError("candidate protocol lock can bind only the primary profile")
    target = output_path(out_path)
    policy_path = output_path(config["mortgage_policy_path"])
    vintage_inventory_path = output_path(
        config["mortgage_policy_vintage_inventory_path"]
    )
    general_policy_path = output_path(config["general_safety_policy_path"])
    taxonomy_path = output_path(config["taxonomy_path"])
    predecessor_path = output_path("provenance/LEGACY_REFERENCE_CENTERING.json")
    lock = _self_hashed({
        "lock_schema_version": 1,
        "lock_kind": "protocol_taxonomy_candidate",
        "study_id": config["study_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": LOCK_STATUS,
        "config": {
            "path": output_path(config_path).relative_to(output_path(".")).as_posix(),
            "byte_sha256": sha256_file(output_path(config_path)),
            "object_sha256": canonical_sha256(config),
        },
        "taxonomy": {
            "path": taxonomy_path.relative_to(output_path(".")).as_posix(),
            "byte_sha256": sha256_file(taxonomy_path),
        },
        "general_safety_policy": {
            "path": general_policy_path.relative_to(output_path(".")).as_posix(),
            "byte_sha256": sha256_file(general_policy_path),
            "review_status": read_json(config["general_safety_policy_path"])[
                "review_status"
            ],
        },
        "mortgage_policy": {
            "path": policy_path.relative_to(output_path(".")).as_posix(),
            "byte_sha256": sha256_file(policy_path),
            "legal_review_status": read_json(config["mortgage_policy_path"])[
                "legal_review_status"
            ],
        },
        "mortgage_policy_vintage_inventory": {
            "path": vintage_inventory_path.relative_to(output_path(".")).as_posix(),
            "byte_sha256": sha256_file(vintage_inventory_path),
            "review_status": read_json(
                config["mortgage_policy_vintage_inventory_path"]
            )["review_status"],
            "complete": read_json(
                config["mortgage_policy_vintage_inventory_path"]
            )["complete"],
        },
        "historical_predecessor": {
            "relationship": "historical_only_not_parent_not_superseded",
            "path": predecessor_path.relative_to(output_path(".")).as_posix(),
            "byte_sha256": sha256_file(predecessor_path),
        },
        "execution_sources": source_inventory(),
        "data_build_authorized": False,
        "gpu_training_authorized": False,
        "claim_authorized": False,
        "required_children": list(REQUIRED_CHILDREN),
    })
    _write_exclusive_json(target, lock)
    return lock


def validate_protocol_candidate(
    path: str | Path = DEFAULT_LOCK,
    *,
    bind_live_sources: bool = True,
) -> dict:
    lock = read_json(path)
    _validate_self_hash(lock)
    if lock.get("lock_schema_version") != 1:
        raise ContractError("unsupported candidate-lock schema")
    if lock.get("lock_kind") != "protocol_taxonomy_candidate":
        raise ContractError("not a Paper C v2 protocol candidate lock")
    if lock.get("status") != LOCK_STATUS:
        raise ContractError("candidate-lock status drifted")
    try:
        created_at = datetime.fromisoformat(
            str(lock.get("created_utc", "")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ContractError("candidate lock has an invalid creation timestamp") from exc
    if created_at.tzinfo is None:
        raise ContractError("candidate lock creation timestamp lacks a timezone")
    if tuple(lock.get("required_children") or ()) != REQUIRED_CHILDREN:
        raise ContractError("candidate-lock child sequence drifted")
    config_record = lock.get("config") or {}
    config_relative = config_record.get("path")
    if not isinstance(config_relative, str) or not config_relative:
        raise ContractError("candidate lock has no config path")
    config_path = output_path(config_relative)
    if not config_path.is_file():
        raise ContractError("locked config path is not a file")
    if not HEX64.fullmatch(str(config_record.get("byte_sha256", ""))):
        raise ContractError("locked config byte hash is invalid")
    if not HEX64.fullmatch(str(config_record.get("object_sha256", ""))):
        raise ContractError("locked config object hash is invalid")
    if sha256_file(config_path) != config_record.get("byte_sha256"):
        raise ContractError("locked config bytes drifted")
    config = read_json(config_record["path"])
    validate_config(config)
    if config.get("profile") != "primary":
        raise ContractError("candidate protocol lock must bind the primary profile")
    if lock.get("study_id") != config.get("study_id"):
        raise ContractError("candidate lock and config study IDs disagree")
    if canonical_sha256(config) != config_record.get("object_sha256"):
        raise ContractError("locked config object drifted")
    expected_paths = {
        "taxonomy": config["taxonomy_path"],
        "general_safety_policy": config["general_safety_policy_path"],
        "mortgage_policy": config["mortgage_policy_path"],
        "mortgage_policy_vintage_inventory": config[
            "mortgage_policy_vintage_inventory_path"
        ],
        "historical_predecessor": "provenance/LEGACY_REFERENCE_CENTERING.json",
    }
    for field, expected_path in expected_paths.items():
        record = lock.get(field) or {}
        if record.get("path") != expected_path:
            raise ContractError(f"locked {field} path disagrees with config")
        if not HEX64.fullmatch(str(record.get("byte_sha256", ""))):
            raise ContractError(f"locked {field} hash is invalid")
        if not output_path(expected_path).is_file():
            raise ContractError(f"locked {field} path is not a file")
        if sha256_file(output_path(str(record.get("path", "")))) != record.get("byte_sha256"):
            raise ContractError(f"locked {field} drifted")
    mortgage_policy = read_json(config["mortgage_policy_path"])
    if lock["mortgage_policy"].get("legal_review_status") != mortgage_policy.get(
        "legal_review_status"
    ):
        raise ContractError("mortgage legal-review status drifted")
    general_policy = read_json(config["general_safety_policy_path"])
    if lock["general_safety_policy"].get("review_status") != general_policy.get(
        "review_status"
    ):
        raise ContractError("general-safety review status drifted")
    vintage_inventory = read_json(config["mortgage_policy_vintage_inventory_path"])
    if lock["mortgage_policy_vintage_inventory"].get(
        "review_status"
    ) != vintage_inventory.get("review_status") or lock[
        "mortgage_policy_vintage_inventory"
    ].get("complete") is not vintage_inventory.get("complete"):
        raise ContractError("mortgage policy-vintage inventory status drifted")
    inventory = lock.get("execution_sources") or {}
    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        raise ContractError("lock source inventory is empty")
    seen_paths: set[str] = set()
    for index, record in enumerate(files):
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ContractError(f"invalid source inventory record: {index}")
        relative = record["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen_paths
        ):
            raise ContractError(f"invalid source inventory path: {index}")
        seen_paths.add(relative)
        if not HEX64.fullmatch(str(record["sha256"])):
            raise ContractError(f"invalid source inventory hash: {relative}")
    if not HEX64.fullmatch(str(inventory.get("aggregate_sha256", ""))):
        raise ContractError("lock source aggregate hash is invalid")
    if sha256_ordered(files) != inventory.get("aggregate_sha256"):
        raise ContractError("lock source inventory hash mismatch")
    if bind_live_sources and source_inventory() != inventory:
        raise ContractError("live source tree differs from candidate lock")
    if any(lock.get(field) is not False for field in (
        "data_build_authorized", "gpu_training_authorized", "claim_authorized"
    )):
        raise ContractError("candidate lock grants unauthorized work")
    return lock
