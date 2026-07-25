#!/usr/bin/env python
"""Build and validate the Paper C v2 lock.

`init` creates a development lock and never authorizes claim-bearing training.
`finalize` additionally requires a clean git checkout and a complete, rehashed
20-cell Stage-1 adapter inventory.  Existing locks are never overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import os
import subprocess
import sys

_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE.parent), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import paper_a_common as A  # noqa: E402
import paper_c_dpo_common as P  # noqa: E402


LOCK_SCHEMA_VERSION = 2
SOURCE_PATHS = (
    ".gitignore",
    "configs/paper_c_dpo_v2.json",
    "docs/paper-c-prereg-v2.md",
    "docs/paper-c-development-plan.md",
    "docs/paper-c-code-design.md",
    "docs/paper-c-objective-axis-reward-and-design.md",
    "experiments/paper_c_dpo_common.py",
    "experiments/prepare_paper_c_dpo.py",
    "experiments/lock_paper_c_dpo.py",
    "experiments/train_paper_c_dpo.py",
    "experiments/eval_paper_c_dpo.py",
    "experiments/analyze_paper_c_dpo.py",
    "experiments/paper_a_common.py",
    "experiments/run_paper_a_sft.py",
    "experiments/eval_paper_a_sft.py",
    "guard_research/prompts.py",
    "guard_research/metrics.py",
    "guard_research/provenance.py",
    "guard_research/thresholds.py",
    "pyproject.toml",
    "requirements.txt",
    "tests/test_paper_c_dpo.py",
)


def _repo_path(relative: str) -> Path:
    return Path(A.REPO_ROOT) / relative


def _git_state() -> dict:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=A.REPO_ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=A.REPO_ROOT, check=True,
        text=True, capture_output=True,
    ).stdout
    return {"git_sha": sha, "clean": not bool(status.strip())}


def _source_inventory(*, require_all: bool) -> dict:
    entries = {}
    missing = []
    for relative in SOURCE_PATHS:
        path = _repo_path(relative)
        if not path.is_file():
            missing.append(relative)
            continue
        entries[relative] = P.sha256_file(path)
    if require_all and missing:
        raise P.PaperCContractError(
            f"final lock sources are missing: {missing}")
    return {
        "files": entries,
        "missing_development_files": missing,
        "aggregate_sha256": P.sha256_ordered(
            {"path": path, "sha256": digest}
            for path, digest in sorted(entries.items())
        ),
    }


def _manifest_inventory(parent_lock: dict) -> dict:
    manifests_dir = Path(A.abspath(A.artifact_paths(parent_lock)["manifests"]))
    names = (
        "train.jsonl", "calibration.jsonl", "id_test.jsonl", "transfer_test.jsonl",
        "orbench_safe_stress.jsonl", "harmbench_positive_stress.jsonl",
    )
    entries = {}
    for name in names:
        path = manifests_dir / name
        if not path.is_file():
            raise P.PaperCContractError(f"parent manifest is missing: {path}")
        entries[name] = {
            "path": os.path.relpath(path, A.REPO_ROOT),
            "sha256": P.sha256_file(path),
        }
    if entries["train.jsonl"]["sha256"] != parent_lock.get("train_manifest_sha256"):
        raise P.PaperCContractError("parent train manifest differs from parent LOCK.json")
    return entries


def _validate_stage1_inventory(path: str, config: dict, parent_lock: dict,
                               artifact_root: Path) -> dict:
    inventory = P.read_json(path)
    cells = inventory.get("cells")
    if not isinstance(cells, dict):
        raise P.PaperCContractError("Stage-1 inventory has no cells object")
    expected = {
        f"{model}/seed_{seed}"
        for model in config["models"] for seed in config["seeds"]
    }
    if set(cells) != expected:
        raise P.PaperCContractError(
            f"Stage-1 inventory grid mismatch: missing={sorted(expected-set(cells))[:5]}, "
            f"extra={sorted(set(cells)-expected)[:5]}")
    rebound = {}
    adapter_hashes = set()
    for key in sorted(cells):
        cell = cells[key]
        if cell.get("status") != "completed":
            raise P.PaperCContractError(f"Stage-1 cell is not completed: {key}")
        adapter_dir = Path(cell.get("adapter_dir", ""))
        run_meta = Path(cell.get("run_meta_path", ""))
        if not adapter_dir.is_absolute():
            adapter_dir = _repo_path(str(adapter_dir))
        if not run_meta.is_absolute():
            run_meta = _repo_path(str(run_meta))
        if (not A.path_is_within(adapter_dir, artifact_root)
                or not A.path_is_within(run_meta, artifact_root)):
            raise P.PaperCContractError(
                f"Stage-1 cell resolves outside the Paper C artifact root: {key}")
        if not A.adapter_is_present(str(adapter_dir)) or not run_meta.is_file():
            raise P.PaperCContractError(f"Stage-1 bytes are missing: {key}")
        observed_adapter = A.sha256_dir(adapter_dir)
        observed_meta = P.sha256_file(run_meta)
        if observed_adapter != cell.get("adapter_sha256"):
            raise P.PaperCContractError(f"Stage-1 adapter hash mismatch: {key}")
        if observed_meta != cell.get("run_meta_sha256"):
            raise P.PaperCContractError(f"Stage-1 run metadata hash mismatch: {key}")
        model_key, seed_text = key.split("/seed_", 1)
        validation = A.validate_run_artifact(
            parent_lock, model_key, int(seed_text), run_meta.parent,
            allow_legacy=False, recompute_adapter_hash=True,
        )
        if not validation["valid"]:
            raise P.PaperCContractError(
                f"Stage-1 cell fails the parent run contract: {key}: "
                f"{validation['issues']}")
        if Path(validation["adapter_dir"]).resolve() != adapter_dir.resolve():
            raise P.PaperCContractError(f"Stage-1 adapter/run directory mismatch: {key}")
        if observed_adapter in adapter_hashes:
            raise P.PaperCContractError("Stage-1 inventory reuses adapter bytes across cells")
        adapter_hashes.add(observed_adapter)
        rebound[key] = {
            "status": "completed",
            "adapter_dir": os.path.relpath(adapter_dir, A.REPO_ROOT),
            "adapter_sha256": observed_adapter,
            "run_meta_path": os.path.relpath(run_meta, A.REPO_ROOT),
            "run_meta_sha256": observed_meta,
        }
    return {
        "cells": rebound,
        "aggregate_sha256": P.sha256_ordered(
            {"cell": key, **value} for key, value in sorted(rebound.items())
        ),
    }


def _validate_stage2_inputs(path: str, config: dict, stage1: dict,
                            manifests: dict, artifact_root: Path) -> dict:
    """Rebind every model/seed split/reference/selection input by file bytes."""
    inventory = P.read_json(path)
    cells = inventory.get("cells")
    if not isinstance(cells, dict):
        raise P.PaperCContractError("Stage-2 input inventory has no cells object")
    expected = {
        f"{model}/seed_{seed}"
        for model in config["models"] for seed in config["seeds"]
    }
    if set(cells) != expected:
        raise P.PaperCContractError(
            f"Stage-2 input grid mismatch: missing={sorted(expected-set(cells))[:5]}, "
            f"extra={sorted(set(cells)-expected)[:5]}")
    train_path = _repo_path(manifests["train.jsonl"]["path"])
    train_rows = P.read_jsonl(train_path)
    expected_partition_rows = P.family_partition(
        train_rows,
        development_fraction=float(config["stage2"]["development_fraction"]),
        seed=int(config["stage2"]["development_split_seed"]),
    )
    partition_file_hashes = set()
    rebound = {}
    for key in sorted(cells):
        model_key, seed_text = key.split("/seed_", 1)
        seed = int(seed_text)
        raw = cells[key]
        files = {}
        for label in (
            "partition", "partition_metadata", "reference", "reference_metadata",
            "selection", "selection_metadata",
        ):
            record = raw.get(label)
            if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
                raise P.PaperCContractError(f"Stage-2 input {key} lacks {label} record")
            resolved = Path(record["path"])
            if not resolved.is_absolute():
                resolved = _repo_path(str(resolved))
            if not A.path_is_within(resolved, artifact_root):
                raise P.PaperCContractError(
                    f"Stage-2 input resolves outside the Paper C artifact root: {resolved}")
            if not resolved.is_file():
                raise P.PaperCContractError(f"Stage-2 input file is missing: {resolved}")
            observed = P.sha256_file(resolved)
            if observed != record["sha256"]:
                raise P.PaperCContractError(f"Stage-2 input hash mismatch: {key}/{label}")
            files[label] = {
                "path": os.path.relpath(resolved, A.REPO_ROOT),
                "sha256": observed,
            }

        partition_rows = P.read_jsonl(_repo_path(files["partition"]["path"]))
        reference_rows = P.read_jsonl(_repo_path(files["reference"]["path"]))
        selection_rows = P.read_jsonl(_repo_path(files["selection"]["path"]))
        P.validate_selections(selection_rows)
        if partition_rows != expected_partition_rows:
            raise P.PaperCContractError(
                f"Stage-2 partition does not reproduce from locked inputs: {key}")
        partition_file_hashes.add(files["partition"]["sha256"])
        partition_id_list = [str(row.get("sample_id", "")) for row in partition_rows]
        reference_id_list = [str(row.get("sample_id", "")) for row in reference_rows]
        partition_ids = set(partition_id_list)
        reference_ids = set(reference_id_list)
        if ("" in partition_ids or "" in reference_ids
                or len(partition_ids) != len(partition_id_list)
                or len(reference_ids) != len(reference_id_list)):
            raise P.PaperCContractError(f"duplicate or empty Stage-2 input identity: {key}")
        for row in reference_rows:
            try:
                safe_logit = float(row["safe_logit"])
                unsafe_logit = float(row["unsafe_logit"])
            except (KeyError, TypeError, ValueError) as exc:
                raise P.PaperCContractError(
                    f"reference row lacks numeric verdict logits: {key}") from exc
            if not math.isfinite(safe_logit) or not math.isfinite(unsafe_logit):
                raise P.PaperCContractError(f"reference row has non-finite logits: {key}")
        if not partition_ids or partition_ids != reference_ids:
            raise P.PaperCContractError(f"partition/reference identity mismatch: {key}")
        if any(str(row.get("sample_id", "")) not in partition_ids for row in selection_rows):
            raise P.PaperCContractError(f"selection contains a row outside the partition: {key}")
        expected_selection_rows = P.build_selections(
            partition_rows,
            reference_rows,
            uncertain_fraction=float(config["stage2"]["uncertain_fraction"]),
            seed=int(config["stage2"]["selection_seed"]),
        )
        if selection_rows != expected_selection_rows:
            raise P.PaperCContractError(
                f"Stage-2 selection does not reproduce from locked inputs: {key}")

        partition_meta = P.read_json(_repo_path(files["partition_metadata"]["path"]))
        reference_meta = P.read_json(_repo_path(files["reference_metadata"]["path"]))
        selection_meta = P.read_json(_repo_path(files["selection_metadata"]["path"]))
        stage1_sha = stage1["cells"][key]["adapter_sha256"]
        if partition_meta.get("algorithm_version") != P.PARTITION_ALGORITHM_VERSION:
            raise P.PaperCContractError(f"{key}/partition algorithm version mismatch")
        if selection_meta.get("algorithm_version") != P.SELECTION_ALGORITHM_VERSION:
            raise P.PaperCContractError(f"{key}/selection algorithm version mismatch")
        expected_common = {
            "config_sha256": P.canonical_sha256(config),
            "train_manifest_sha256": manifests["train.jsonl"]["sha256"],
        }
        for label, meta in (
            ("partition", partition_meta),
            ("reference", reference_meta),
            ("selection", selection_meta),
        ):
            for field, expected_value in expected_common.items():
                if meta.get(field) != expected_value:
                    raise P.PaperCContractError(
                        f"{key}/{label} metadata disagrees on {field}")
        for label, meta in (("reference", reference_meta), ("selection", selection_meta)):
            if meta.get("model_key") != model_key or int(meta.get("seed", -1)) != seed:
                raise P.PaperCContractError(f"{key}/{label} metadata model/seed mismatch")
            if meta.get("stage1_adapter_sha256") != stage1_sha:
                raise P.PaperCContractError(f"{key}/{label} metadata adapter mismatch")
        if reference_meta.get("reference_file_sha256") != files["reference"]["sha256"]:
            raise P.PaperCContractError(f"{key}/reference metadata file hash mismatch")
        if selection_meta.get("reference_sha256") != files["reference"]["sha256"]:
            raise P.PaperCContractError(f"{key}/selection metadata reference hash mismatch")
        if selection_meta.get("selection_file_sha256") != files["selection"]["sha256"]:
            raise P.PaperCContractError(f"{key}/selection metadata file hash mismatch")
        rebound[key] = {
            **files,
            "stage1_adapter_sha256": stage1_sha,
            "prompt_fingerprint_sha256": reference_meta.get("prompt_fingerprint_sha256"),
            "decision_tokens": reference_meta.get("decision_tokens"),
        }
        if not rebound[key]["prompt_fingerprint_sha256"] or not rebound[key]["decision_tokens"]:
            raise P.PaperCContractError(
                f"{key}/reference metadata lacks prompt or decision-token identity")
    if len(partition_file_hashes) != 1:
        raise P.PaperCContractError(
            "all model/seed cells must use the same global Stage-2 partition")
    return {
        "cells": rebound,
        "aggregate_sha256": P.sha256_ordered(
            {"cell": key, **value} for key, value in sorted(rebound.items())
        ),
    }


def _without_lock_hash(lock: dict) -> dict:
    return {key: value for key, value in lock.items() if key != "lock_sha256"}


def build_lock(args: argparse.Namespace, *, final: bool) -> dict:
    config = P.read_json(args.config)
    P.validate_config(config)
    artifact_root = _repo_path(config["artifact_root"]).resolve()
    configured_parent = _repo_path(str(config.get("parent_lock", ""))).resolve()
    if Path(args.parent_lock).resolve() != configured_parent:
        raise P.PaperCContractError(
            f"parent lock differs from config: expected {configured_parent}")
    if final and Path(args.out).resolve() != (artifact_root / "LOCK.json").resolve():
        raise P.PaperCContractError(
            f"final lock path is canonical: {artifact_root / 'LOCK.json'}")
    if Path(args.out).exists():
        raise P.PaperCContractError(
            f"refusing to overwrite an existing lock: {args.out}")
    parent_lock = A.load_lock(
        args.parent_lock, allow_legacy=False, verify_files=False)
    parent_sha = P.sha256_file(args.parent_lock)
    git = _git_state()
    if final and not git["clean"]:
        raise P.PaperCContractError("final Paper C lock requires a clean git checkout")
    sources = _source_inventory(require_all=final)
    stage1 = None
    stage2_inputs = None
    if final:
        if not args.stage1_inventory:
            raise P.PaperCContractError("finalize requires --stage1-inventory")
        stage1 = _validate_stage1_inventory(
            args.stage1_inventory, config, parent_lock, artifact_root)
        if not args.stage2_input_inventory:
            raise P.PaperCContractError("finalize requires --stage2-input-inventory")
        manifests = _manifest_inventory(parent_lock)
        stage2_inputs = _validate_stage2_inputs(
            args.stage2_input_inventory, config, stage1, manifests, artifact_root)
    else:
        manifests = _manifest_inventory(parent_lock)

    lock = {
        "lock_schema_version": LOCK_SCHEMA_VERSION,
        "study_id": config["study_id"],
        "finalization_status": "final" if final else "development",
        "created_utc": A.utcnow(),
        "artifact_root": config["artifact_root"],
        "parent": {
            "path": os.path.relpath(Path(args.parent_lock).resolve(), A.REPO_ROOT),
            "file_sha256": parent_sha,
            "lock_sha256": parent_lock.get("lock_sha256"),
        },
        "config": {
            "path": os.path.relpath(Path(args.config).resolve(), A.REPO_ROOT),
            "sha256": P.sha256_file(args.config),
            "object_sha256": P.canonical_sha256(config),
            "value": config,
        },
        "models": {key: A.lock_model_panel(parent_lock)[key] for key in config["models"]},
        "seeds": list(config["seeds"]),
        "manifests": manifests,
        "parent_data": parent_lock.get("data"),
        "parent_recipe": parent_lock.get("recipe"),
        "data": parent_lock.get("data"),
        "recipe": parent_lock.get("recipe"),
        "prompt": parent_lock.get("prompt"),
        "stage2": config["stage2"],
        "stage1_inventory": stage1,
        "stage2_inputs": stage2_inputs,
        "execution_sources": sources,
        "software_versions": A.software_versions(),
        "git": git,
        "loss_formula_version": P.LOSS_FORMULA_VERSION,
        "partition_algorithm_version": P.PARTITION_ALGORITHM_VERSION,
        "selection_algorithm_version": P.SELECTION_ALGORITHM_VERSION,
    }
    lock["lock_sha256"] = P.canonical_sha256(_without_lock_hash(lock))
    return lock


def cmd_init(args: argparse.Namespace) -> int:
    lock = build_lock(args, final=False)
    P.write_json(args.out, lock)
    print(f"[paper-c lock] development -> {args.out} ({lock['lock_sha256'][:16]})")
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    del args
    raise P.PaperCContractError(
        "finalization is intentionally disabled until lock-bound reference-logit and "
        "Stage-2-development score producers plus candidate-inventory validation are implemented")


def cmd_validate(args: argparse.Namespace) -> int:
    lock = P.read_json(args.lock)
    issues = []
    if int(lock.get("lock_schema_version", -1)) != LOCK_SCHEMA_VERSION:
        issues.append("lock_schema_version")
    observed = P.canonical_sha256(_without_lock_hash(lock))
    if observed != lock.get("lock_sha256"):
        issues.append("lock_sha256")
    try:
        P.validate_config((lock.get("config") or {}).get("value") or {})
        config_record = lock.get("config") or {}
        config_path = _repo_path(str(config_record.get("path", "")))
        if (not config_path.is_file()
                or P.sha256_file(config_path) != config_record.get("sha256")
                or P.canonical_sha256(config_record.get("value"))
                != config_record.get("object_sha256")):
            issues.append("config_binding")
        P.validate_execution_sources(lock, A.REPO_ROOT)
    except P.PaperCContractError as exc:
        issues.append(f"contract:{exc}")
    parent = lock.get("parent") or {}
    parent_path = _repo_path(str(parent.get("path", "")))
    if not parent_path.is_file() or P.sha256_file(parent_path) != parent.get("file_sha256"):
        issues.append("parent_lock_file")
    for relative, expected in (lock.get("execution_sources") or {}).get("files", {}).items():
        path = _repo_path(relative)
        if not path.is_file() or P.sha256_file(path) != expected:
            issues.append(f"execution_source:{relative}")
    for name, record in (lock.get("manifests") or {}).items():
        path = _repo_path(str(record.get("path", "")))
        if not path.is_file() or P.sha256_file(path) != record.get("sha256"):
            issues.append(f"manifest:{name}")
    if lock.get("finalization_status") == "final":
        try:
            config = (lock.get("config") or {}).get("value") or {}
            P.validate_config(config)
            if not lock.get("git", {}).get("clean"):
                issues.append("git_not_clean_at_finalization")
            stage1 = lock.get("stage1_inventory")
            if not stage1:
                issues.append("stage1_inventory")
            else:
                for key, cell in (stage1.get("cells") or {}).items():
                    adapter = _repo_path(cell["adapter_dir"])
                    run_meta = _repo_path(cell["run_meta_path"])
                    if (not A.adapter_is_present(str(adapter))
                            or A.sha256_dir(adapter) != cell.get("adapter_sha256")):
                        issues.append(f"stage1_adapter:{key}")
                    if (not run_meta.is_file()
                            or P.sha256_file(run_meta) != cell.get("run_meta_sha256")):
                        issues.append(f"stage1_run_meta:{key}")
            stage2_inputs = lock.get("stage2_inputs")
            if not stage2_inputs:
                issues.append("stage2_inputs")
            else:
                for key, cell in (stage2_inputs.get("cells") or {}).items():
                    for label in (
                        "partition", "partition_metadata", "reference", "reference_metadata",
                        "selection", "selection_metadata",
                    ):
                        record = cell.get(label) or {}
                        path = _repo_path(str(record.get("path", "")))
                        if not path.is_file() or P.sha256_file(path) != record.get("sha256"):
                            issues.append(f"stage2_input:{key}:{label}")
        except P.PaperCContractError as exc:
            issues.append(f"config:{exc}")
    if issues:
        print(f"[paper-c lock] INVALID: {issues}", file=sys.stderr)
        return 1
    print(f"[paper-c lock] valid {lock.get('finalization_status')} lock")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build/validate Paper C v2 locks")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, func in (("init", cmd_init), ("finalize", cmd_finalize)):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--parent-lock", required=True)
        command.add_argument("--stage1-inventory")
        command.add_argument("--stage2-input-inventory")
        command.add_argument("--out", required=True)
        command.set_defaults(func=func)
    validate = sub.add_parser("validate")
    validate.add_argument("--lock", required=True)
    validate.set_defaults(func=cmd_validate)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (P.PaperCContractError, A.ArtifactContractError, OSError,
            subprocess.CalledProcessError, ValueError) as exc:
        print(f"[paper-c lock] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
