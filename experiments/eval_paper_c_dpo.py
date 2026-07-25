#!/usr/bin/env python
"""Paper C v2 scorer for base, Stage-1, and selected Stage-2 adapters.

The scorer consumes an explicit immutable bundle inventory.  It reuses Paper A's
low-level prompt/logit/calibration helpers but owns its condition grid and
adapter validation so no non-base condition can silently score the base model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

import pandas as pd

_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE.parent), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import paper_a_common as A  # noqa: E402
import paper_c_dpo_common as P  # noqa: E402
import eval_paper_a_sft as E  # noqa: E402


SCORE_ARTIFACT_VERSION = 2


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(A.REPO_ROOT) / path


def _load_lock(path: str, allow_development: bool) -> dict:
    lock = P.read_json(path)
    expected = P.canonical_sha256(
        {key: value for key, value in lock.items() if key != "lock_sha256"})
    if expected != lock.get("lock_sha256"):
        raise P.PaperCContractError("Paper C lock hash mismatch")
    if lock.get("finalization_status") != "final" and not allow_development:
        raise P.PaperCContractError("final scoring requires a final Paper C lock")
    P.validate_config((lock.get("config") or {}).get("value") or {})
    if lock.get("finalization_status") == "final":
        P.validate_execution_sources(lock, A.REPO_ROOT)
        software_issues = A.protocol_software_issues(
            A.software_versions(), lock.get("software_versions"))
        if software_issues:
            raise P.PaperCContractError(
                f"software environment differs from final lock: {software_issues}")
    return lock


def _condition_fields(condition: str) -> tuple[str | None, str | None, str]:
    if condition == "base":
        return None, None, "base"
    if condition == "stage1_sft":
        return None, None, "stage1"
    for cell in P.condition_grid():
        if condition == cell["condition"]:
            return cell["objective"], cell["sampler"], "stage2"
    raise P.PaperCContractError(f"inventory has unknown condition: {condition}")


def _selection_lookup(path: str | None, lock: dict) -> dict:
    if not path:
        return {}
    artifact = P.read_json(path)
    if artifact.get("lock_sha256") != lock.get("lock_sha256"):
        raise P.PaperCContractError("checkpoint selection belongs to another lock")
    records = artifact.get("records")
    if not isinstance(records, list) or artifact.get("records_sha256") != P.canonical_sha256(records):
        raise P.PaperCContractError("checkpoint selection has an invalid record hash")
    lookup = {}
    for record in records:
        key = (str(record["model_key"]), str(record["condition"]), int(record["seed"]))
        if key in lookup:
            raise P.PaperCContractError(f"duplicate checkpoint-selection record: {key}")
        lookup[key] = record
    expected = {
        (model, cell["condition"], int(seed))
        for model in lock["models"] for seed in lock["seeds"]
        for cell in P.condition_grid()
    }
    if set(lookup) != expected:
        raise P.PaperCContractError("checkpoint selection has an incomplete condition grid")
    return lookup


def _inventory(path: str, lock: dict, *, synthetic: bool,
               allow_incomplete: bool, selection_lookup: dict | None = None) -> list[dict]:
    value = P.read_json(path)
    bundles = value.get("bundles")
    if not isinstance(bundles, list):
        raise P.PaperCContractError("bundle inventory must contain a bundles list")
    seen = set()
    normalized = []
    for raw in bundles:
        item = dict(raw)
        model_key = str(item.get("model_key", ""))
        condition = str(item.get("condition", ""))
        seed = int(item.get("seed", -1))
        if model_key not in lock["models"]:
            raise P.PaperCContractError(f"inventory model outside lock: {model_key}")
        objective, sampler, stage = _condition_fields(condition)
        key = (model_key, condition, seed)
        if key in seen:
            raise P.PaperCContractError(f"duplicate inventory bundle: {key}")
        seen.add(key)
        if stage == "base":
            if seed != -1 or item.get("adapter_dir") or item.get("adapter_sha256"):
                raise P.PaperCContractError(f"invalid base inventory record: {key}")
            adapter_path = None
            adapter_sha = None
            run_meta_sha = None
        else:
            if seed not in lock["seeds"]:
                raise P.PaperCContractError(f"inventory seed outside lock: {key}")
            adapter_path = _repo_path(str(item.get("adapter_dir", "")))
            if not synthetic:
                if not A.adapter_is_present(str(adapter_path)):
                    raise P.PaperCContractError(f"adapter missing for {key}: {adapter_path}")
                observed = A.sha256_dir(adapter_path)
                if observed != item.get("adapter_sha256"):
                    raise P.PaperCContractError(f"adapter hash mismatch for {key}")
                adapter_sha = observed
            else:
                adapter_sha = str(item.get("adapter_sha256") or f"synthetic::{key}")
            run_meta_path = item.get("run_meta_path")
            run_meta_sha = None
            if run_meta_path:
                resolved_meta = _repo_path(str(run_meta_path))
                if not synthetic and (not resolved_meta.is_file() or
                                      P.sha256_file(resolved_meta) != item.get("run_meta_sha256")):
                    raise P.PaperCContractError(f"run metadata hash mismatch for {key}")
                run_meta_sha = str(item.get("run_meta_sha256") or "")
            if not synthetic and stage == "stage1":
                locked = ((lock.get("stage1_inventory") or {}).get("cells") or {}).get(
                    f"{model_key}/seed_{seed}")
                if (not locked or locked.get("adapter_sha256") != adapter_sha
                        or not run_meta_path
                        or locked.get("run_meta_sha256") != run_meta_sha):
                    raise P.PaperCContractError(
                        f"Stage-1 inventory does not match the lock for {key}")
            if not synthetic and stage == "stage2":
                if not run_meta_path:
                    raise P.PaperCContractError(f"Stage-2 bundle has no run metadata: {key}")
                run_meta = P.read_json(resolved_meta)
                expected_core = {
                    "lock_sha256": lock["lock_sha256"],
                    "model_key": model_key,
                    "seed": seed,
                    "condition": condition,
                    "objective": objective,
                    "sampler": sampler,
                    "status": "completed",
                }
                for field, expected_value in expected_core.items():
                    if run_meta.get(field) != expected_value:
                        raise P.PaperCContractError(
                            f"Stage-2 run metadata disagrees on {field}: {key}")
                checkpoint = item.get("checkpoint_step")
                checkpoint_record = (run_meta.get("checkpoint_adapters") or {}).get(
                    str(int(checkpoint)) if checkpoint is not None else "")
                if not checkpoint_record or checkpoint_record.get("adapter_sha256") != adapter_sha:
                    raise P.PaperCContractError(
                        f"Stage-2 checkpoint adapter is not bound by run metadata: {key}")
                if selection_lookup is not None:
                    selected = selection_lookup.get(key)
                    if not selected:
                        raise P.PaperCContractError(
                            f"Stage-2 bundle absent from checkpoint selection: {key}")
                    expected_step = (selected.get("selected_checkpoint_step")
                                     if selected.get("target_feasible")
                                     else selected.get("descriptive_fallback_checkpoint_step"))
                    if expected_step is None or int(checkpoint) != int(expected_step):
                        raise P.PaperCContractError(
                            f"Stage-2 bundle uses an unselected checkpoint: {key}")
        normalized.append({
            "model_key": model_key,
            "condition": condition,
            "seed": seed,
            "stage": stage,
            "objective": objective,
            "sampler": sampler,
            "checkpoint_step": item.get("checkpoint_step"),
            "adapter_dir": None if adapter_path is None else str(adapter_path),
            "adapter_sha256": adapter_sha,
            "run_meta_sha256": run_meta_sha,
        })

    if not allow_incomplete:
        expected = {(model, "base", -1) for model in lock["models"]}
        expected |= {
            (model, "stage1_sft", int(seed))
            for model in lock["models"] for seed in lock["seeds"]
        }
        expected |= {
            (model, cell["condition"], int(seed))
            for model in lock["models"] for seed in lock["seeds"]
            for cell in P.condition_grid()
        }
        observed = {(row["model_key"], row["condition"], row["seed"])
                    for row in normalized}
        if observed != expected:
            raise P.PaperCContractError(
                f"bundle inventory grid mismatch: missing={len(expected-observed)}, "
                f"extra={len(observed-expected)}")
    return sorted(normalized, key=lambda row: (
        row["model_key"], row["stage"], row["condition"], row["seed"]))


def _manifests_dir(lock: dict) -> Path:
    records = lock.get("manifests") or {}
    for name, record in records.items():
        path = _repo_path(record["path"])
        if not path.is_file() or P.sha256_file(path) != record.get("sha256"):
            raise P.PaperCContractError(f"locked scoring manifest drifted: {name}")
    paths = [_repo_path(record["path"]).parent for record in records.values()]
    if not paths or len({path.resolve() for path in paths}) != 1:
        raise P.PaperCContractError("Paper C manifests do not share one directory")
    return paths[0]


def cmd_score(args: argparse.Namespace) -> int:
    if not args.development:
        raise P.PaperCContractError(
            "final scoring is intentionally disabled until final checkpoint-selection and "
            "candidate-inventory binding are implemented")
    lock = _load_lock(args.lock, args.development)
    diagnostic = bool(args.synthetic or args.limit is not None or args.allow_incomplete)
    if diagnostic and not args.development:
        raise P.PaperCContractError(
            "--synthetic, --limit, and --allow-incomplete require --development")
    expected_out = _repo_path(str(lock["artifact_root"])) / "scores" / "retrospective"
    if not args.development and Path(args.out).resolve() != expected_out.resolve():
        raise P.PaperCContractError(
            f"final retrospective scores must use the canonical path: {expected_out}")
    if not args.development and not args.checkpoint_selection:
        raise P.PaperCContractError("final scoring requires --checkpoint-selection")
    selection_lookup = _selection_lookup(args.checkpoint_selection, lock)
    bundles = _inventory(
        args.inventory, lock, synthetic=args.synthetic,
        allow_incomplete=args.allow_incomplete,
        selection_lookup=(selection_lookup if args.checkpoint_selection else None),
    )
    manifests_dir = _manifests_dir(lock)
    rows = E.load_scoring_rows(str(manifests_dir), args.limit)
    fingerprints = E.manifest_fingerprints(str(manifests_dir), rows)
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise P.PaperCContractError(f"refusing to overwrite nonempty score directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "synthetic" if args.synthetic else (args.device or E._default_device())
    target_fpr = float(lock["config"]["value"]["analysis"]["target_fpr"])

    records = []
    bundle_metadata = {}
    for index, bundle in enumerate(bundles, 1):
        model_key = bundle["model_key"]
        condition = bundle["condition"]
        seed = bundle["seed"]
        model = lock["models"][model_key]
        dtype = "synthetic" if args.synthetic else str(model.get("dtype", "bfloat16"))
        print(f"[paper-c eval] {index}/{len(bundles)} {model_key}:{condition}:seed_{seed}")
        logits, prompt_sha, decision = E.score_bundle(
            lock, rows, model_key, condition, seed,
            bundle["adapter_dir"], bundle["adapter_sha256"],
            device, dtype, args.batch_size, args.synthetic,
        )
        locked_prompt = (lock.get("prompt") or {}).get(
            "per_model_template_sha256", {}).get(model_key)
        if not args.synthetic and locked_prompt and prompt_sha != locked_prompt:
            raise P.PaperCContractError(
                f"scoring prompt identity drifted for {model_key}")
        assembled, calibration = E.assemble_bundle(
            lock, rows, logits, model_key, model["model_revision"],
            condition, seed, bundle["adapter_sha256"], prompt_sha,
            decision, target_fpr,
        )
        for row in assembled:
            row.update({
                "stage": bundle["stage"],
                "objective": bundle["objective"],
                "sampler": bundle["sampler"],
                "checkpoint_step": bundle["checkpoint_step"],
                "run_meta_sha256": bundle["run_meta_sha256"],
            })
        records.extend(assembled)
        key = f"{model_key}:{condition}:seed_{seed}"
        bundle_metadata[key] = {
            **bundle,
            "prompt_sha256": prompt_sha,
            "decision_tokens": decision,
            "calibration_and_threshold": calibration,
        }

    frame = pd.DataFrame.from_records(records)
    scores_path = out_dir / "scores.parquet"
    frame.to_parquet(scores_path, index=False)
    metadata = {
        "score_artifact_version": SCORE_ARTIFACT_VERSION,
        "finalization_status": "development" if args.development else "final",
        "lock_sha256": lock["lock_sha256"],
        "config_sha256": lock["config"]["object_sha256"],
        "inventory_file_sha256": P.sha256_file(args.inventory),
        "checkpoint_selection_file_sha256": (
            P.sha256_file(args.checkpoint_selection) if args.checkpoint_selection else None),
        "manifest_fingerprints": fingerprints,
        "n_bundles": len(bundles),
        "n_rows_per_bundle": len(rows),
        "n_score_rows": len(frame),
        "bundles": bundle_metadata,
        "scores_sha256": P.sha256_file(scores_path),
        "software_versions": A.software_versions(),
        "producer_runtime": A.runtime_environment(device),
    }
    P.write_json(out_dir / "metadata.json", metadata)
    print(f"[paper-c eval] rows={len(frame)} bundles={len(bundles)} -> {scores_path}")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Score Paper C v2 bundles")
    ap.add_argument("--lock", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--checkpoint-selection")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--development", action="store_true")
    ap.set_defaults(func=cmd_score)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (P.PaperCContractError, A.ArtifactContractError, OSError, ValueError) as exc:
        print(f"[paper-c eval] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
