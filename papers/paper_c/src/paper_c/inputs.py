"""Copy hash-verified parent inputs into the isolated Paper C workspace."""

from __future__ import annotations

from pathlib import Path
import shutil

from .contracts import (
    ContractError,
    REQUIRED_MANIFESTS,
    canonical_sha256,
    output_path,
    read_json,
    read_path,
    sha256_file,
    self_hashed,
    write_json,
)


def _manifest_records(parent_lock: dict) -> dict:
    records = ((parent_lock.get("manifests") or {}).get("splits") or {})
    if not isinstance(records, dict):
        raise ContractError("parent lock has no manifest split inventory")
    return records


def bootstrap_inputs(config: dict) -> dict:
    sources = config.get("parent_sources") or {}
    parent_lock_source = read_path(str(sources.get("lock", "")))
    manifest_source_dir = read_path(str(sources.get("manifests", "")))
    if not parent_lock_source.is_file():
        raise ContractError(f"parent lock does not exist: {parent_lock_source}")
    if not manifest_source_dir.is_dir():
        raise ContractError(f"parent manifest directory does not exist: {manifest_source_dir}")

    parent_lock = read_json(parent_lock_source)
    records = _manifest_records(parent_lock)
    local_parent = output_path(Path(config["input_root"]) / "parent" / "LOCK.json")
    local_manifests = output_path(Path(config["input_root"]) / "manifests")
    local_parent.parent.mkdir(parents=True, exist_ok=True)
    local_manifests.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent_lock_source, local_parent)

    copied = {}
    for name in REQUIRED_MANIFESTS:
        record = records.get(name)
        if not isinstance(record, dict) or not record.get("sha256"):
            raise ContractError(f"parent lock does not bind required manifest {name}")
        source = manifest_source_dir / name
        if not source.is_file():
            raise ContractError(f"required parent manifest is missing: {source}")
        observed = sha256_file(source)
        if observed != record["sha256"]:
            raise ContractError(f"parent manifest hash mismatch: {name}")
        target = local_manifests / name
        shutil.copy2(source, target)
        copied[name] = {
            "path": target.relative_to(output_path(".")).as_posix(),
            "sha256": observed,
            "rows": int(record.get("rows", 0)),
        }

    manifest = self_hashed({
        "schema_version": 1,
        "kind": "paper_c_isolated_input_manifest",
        "parent_lock": {
            "path": local_parent.relative_to(output_path(".")).as_posix(),
            "sha256": sha256_file(local_parent),
            "parent_self_hash": parent_lock.get("lock_sha256"),
        },
        "manifests": copied,
        "config_sha256": canonical_sha256(config),
        "credential_material_copied": False,
    }, hash_field="input_manifest_sha256")
    write_json(Path(config["input_root"]) / "INPUT_MANIFEST.json", manifest)
    return manifest


def validate_input_manifest(config: dict) -> dict:
    path = output_path(Path(config["input_root"]) / "INPUT_MANIFEST.json")
    manifest = read_json(path)
    expected = manifest.get("input_manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "input_manifest_sha256"}
    if canonical_sha256(payload) != expected:
        raise ContractError("isolated input manifest self-hash mismatch")
    parent = manifest.get("parent_lock") or {}
    parent_path = output_path(str(parent.get("path", "")))
    if sha256_file(parent_path) != parent.get("sha256"):
        raise ContractError("vendored parent lock drifted")
    records = manifest.get("manifests") or {}
    if set(records) != set(REQUIRED_MANIFESTS):
        raise ContractError("isolated input manifest has the wrong split inventory")
    for name, record in records.items():
        local = output_path(str(record.get("path", "")))
        if not local.is_file() or sha256_file(local) != record.get("sha256"):
            raise ContractError(f"vendored manifest drifted: {name}")
    return manifest
