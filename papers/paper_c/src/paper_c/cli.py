"""Command-line entry point for the isolated Paper C workspace."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import sys

from .analyze import select_checkpoints
from .contracts import (
    ContractError,
    canonical_sha256,
    load_config,
    output_path,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_ordered,
    validate_config,
    write_json,
    write_jsonl,
)
from .inputs import bootstrap_inputs
from .inventory import (
    build_candidate_inventory,
    build_development_score_inventory,
    build_stage1_inventory,
    build_stage2_input_inventory,
)
from .locks import (
    PROTOCOL_LOCK_PATH,
    create_design_lock,
    create_protocol_lock,
    create_prospective_lock,
    create_selection_lock,
    validate_design_lock,
    validate_protocol_lock,
    validate_selection_lock,
)
from .frontier import frontier_report
from .objectives import dpo_logratio_loss, dpo_loss, pair_ce_loss
from .power import (
    assert_design_powered,
    design_power_report,
    effective_learning_rate_ratio,
    seed_sd_by_model,
)
from .runtime import doctor, prepare_prompt_cache
from .sampling import (
    PARTITION_VERSION,
    SELECTION_VERSION,
    build_selections,
    family_partition,
)
from .score import score_adapter
from .train import train_stage1, train_stage2


DEFAULT_CONFIG = "config/study.json"


def _print_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False))


def cmd_validate_config(args) -> int:
    config = load_config(args.config)
    validate_config(config)
    _print_json({
        "status": "valid",
        "study_id": config["study_id"],
        "config_sha256": canonical_sha256(config),
        "stage1_cells": len(config["models"]) * len(config["seeds"]),
        "stage2_cells": len(config["models"]) * len(config["seeds"]) * 6,
        "candidate_checkpoints": len(config["models"]) * len(config["seeds"]) * 6
        * len(config["stage2"]["checkpoint_steps"]),
    })
    return 0


def cmd_loss_check(args) -> int:
    config = load_config(args.config)
    beta = float(config["stage2"]["beta"])
    margin = 0.73
    reference = -0.21
    canonical = dpo_logratio_loss(margin, 0.0, reference, 0.0, beta)
    reduced = dpo_loss(margin, reference, beta)
    if not math.isclose(canonical, reduced, abs_tol=1e-12):
        raise ContractError("DPO margin reduction self-check failed")
    initialization = dpo_loss(reference, reference, beta)
    if not math.isclose(initialization, math.log(2.0), abs_tol=1e-12):
        raise ContractError("DPO initialization identity failed")
    _print_json({
        "status": "valid",
        "beta": beta,
        "pair_ce_example": pair_ce_loss(margin, beta),
        "dpo_reduction_example": reduced,
        "dpo_step_zero": initialization,
    })
    return 0


def cmd_doctor(args) -> int:
    report = doctor(load_config(args.config))
    _print_json(report)
    return 0 if report["ready_for_local_contract_tests"] else 1


def cmd_bootstrap(args) -> int:
    manifest = bootstrap_inputs(load_config(args.config))
    _print_json({
        "status": "copied_and_verified",
        "input_manifest_sha256": manifest["input_manifest_sha256"],
        "credential_material_copied": False,
    })
    return 0


def cmd_prepare_prompts(args) -> int:
    metadata = prepare_prompt_cache(
        config=load_config(args.config),
        model_key=args.model_key,
        manifest_path=args.manifest,
        out_path=args.out,
    )
    _print_json(metadata)
    return 0


def cmd_partition(args) -> int:
    config = load_config(args.config)
    manifest_path = output_path(args.manifest)
    rows = read_jsonl(manifest_path)
    partition = family_partition(
        rows,
        development_fraction=float(config["stage2"]["development_fraction"]),
        seed=int(config["stage2"]["development_split_seed"]),
    )
    write_jsonl(args.out, partition)
    family_assignments = {}
    for row in partition:
        existing = family_assignments.setdefault(row["family_id"], row["stage2_partition"])
        if existing != row["stage2_partition"]:
            raise ContractError("a family crossed Stage-2 partitions")
    counts = Counter((row["source"], row["gold"], row["stage2_partition"]) for row in partition)
    write_json(f"{args.out}.metadata.json", {
        "kind": "paper_c_stage2_partition",
        "algorithm_version": PARTITION_VERSION,
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": canonical_sha256(config),
        "partition_sha256": sha256_ordered(partition),
        "rows": len(partition),
        "families": len(family_assignments),
        "counts": [
            {"source": key[0], "gold": key[1], "partition": key[2], "rows": value}
            for key, value in sorted(counts.items())
        ],
    })
    print(f"partitioned {len(partition)} rows -> {output_path(args.out)}")
    return 0


def cmd_select(args) -> int:
    config = load_config(args.config)
    partition_path = output_path(args.partition)
    reference_path = output_path(args.reference)
    partition = read_jsonl(partition_path)
    reference = read_jsonl(reference_path)
    selections = build_selections(
        partition,
        reference,
        uncertain_fraction=float(config["stage2"]["uncertain_fraction"]),
        seed=int(config["stage2"]["selection_seed"]),
    )
    write_jsonl(args.out, selections)
    write_json(f"{args.out}.metadata.json", {
        "kind": "paper_c_stage2_selection",
        "algorithm_version": SELECTION_VERSION,
        "model_key": args.model_key,
        "seed": args.seed,
        "partition_sha256": sha256_file(partition_path),
        "reference_sha256": sha256_file(reference_path),
        "selection_sha256": sha256_ordered(selections),
        "rows": len(selections),
    })
    print(f"selected {len(selections)} sampler rows -> {output_path(args.out)}")
    return 0


def cmd_protocol_lock(args) -> int:
    lock = create_protocol_lock(
        args.config, out_path=args.out, supersedes_path=args.supersedes,
    )
    _print_json({"status": lock["status"], "lock_sha256": lock["lock_sha256"]})
    return 0


def cmd_design_lock(args) -> int:
    lock = create_design_lock(
        protocol_lock_path=args.protocol_lock,
        stage1_inventory_path=args.stage1_inventory,
        stage2_input_inventory_path=args.stage2_input_inventory,
        partition_path=args.partition,
    )
    _print_json({"status": lock["status"], "lock_sha256": lock["lock_sha256"]})
    return 0


def cmd_validate_lock(args) -> int:
    lock = read_json(output_path(args.lock))
    kind = lock.get("lock_kind")
    if kind == "protocol_lock":
        validated = validate_protocol_lock(args.lock)
    elif kind == "stage2_design_lock":
        validated = validate_design_lock(args.lock)
    elif kind == "postselection_retrospective_lock":
        validated = validate_selection_lock(args.lock)
    else:
        raise ContractError(f"unknown lock kind: {kind}")
    _print_json({"status": "valid", "kind": kind, "lock_sha256": validated["lock_sha256"]})
    return 0


def cmd_selection_lock(args) -> int:
    lock = create_selection_lock(
        design_lock_path=args.design_lock,
        candidate_inventory_path=args.candidates,
        development_scores_path=args.development_scores,
        selection_table_path=args.selection_table,
    )
    _print_json({"status": lock["status"], "lock_sha256": lock["lock_sha256"]})
    return 0


def cmd_inventory_stage1(args) -> int:
    rows = build_stage1_inventory(
        config=load_config(args.config), runs_root=args.runs_root,
        prompt_root=args.prompt_root, out_path=args.out,
    )
    print(f"inventoried {len(rows)} Stage-1 cells -> {output_path(args.out)}")
    return 0


def cmd_inventory_stage2_inputs(args) -> int:
    rows = build_stage2_input_inventory(
        config=load_config(args.config), reference_root=args.reference_root,
        selection_root=args.selection_root, out_path=args.out,
    )
    print(f"inventoried {len(rows)} Stage-2 input cells -> {output_path(args.out)}")
    return 0


def cmd_inventory_candidates(args) -> int:
    rows = build_candidate_inventory(
        config=load_config(args.config), runs_root=args.runs_root, out_path=args.out,
    )
    print(f"inventoried {len(rows)} candidate checkpoints -> {output_path(args.out)}")
    return 0


def cmd_inventory_dev_scores(args) -> int:
    rows = build_development_score_inventory(
        config=load_config(args.config), stage1_root=args.stage1_root,
        candidate_root=args.candidate_root, out_path=args.out,
    )
    print(f"inventoried {len(rows)} development score bundles -> {output_path(args.out)}")
    return 0


def cmd_prospective_lock(args) -> int:
    create_prospective_lock()
    return 2


def cmd_train_stage1(args) -> int:
    config = load_config(args.config)
    if not args.dry_run and not config["study_id"].endswith("_smoke"):
        if not args.protocol_lock:
            raise ContractError("non-smoke Stage-1 training requires --protocol-lock")
        lock = validate_protocol_lock(args.protocol_lock)
        if lock["config"]["object_sha256"] != canonical_sha256(config):
            raise ContractError("Stage-1 config differs from the protocol lock")
    metadata = train_stage1(
        config=config, model_key=args.model_key, seed=args.seed,
        manifest_path=args.manifest, prompt_cache_path=args.prompt_cache,
        out_path=args.out, device=args.device, allow_cpu=args.allow_cpu,
        dry_run=args.dry_run,
    )
    _print_json({"status": metadata["status"], "output": str(output_path(args.out))})
    return 0 if metadata["status"] in {"completed", "dry_run"} else 1


def _partition_ids(path: str | None, role: str | None) -> set[str] | None:
    if not path:
        return None
    if role not in {"stage2_update", "stage2_dev"}:
        raise ContractError("partition role must be stage2_update or stage2_dev")
    return {
        str(row["sample_id"]) for row in read_jsonl(output_path(path))
        if row.get("stage2_partition") == role
    }


def cmd_score(args) -> int:
    config = load_config(args.config)
    is_smoke = config["study_id"].endswith("_smoke")
    if args.mode == "reference" and not is_smoke:
        if not args.protocol_lock:
            raise ContractError("reference scoring requires --protocol-lock")
        lock = validate_protocol_lock(args.protocol_lock)
        if lock["config"]["object_sha256"] != canonical_sha256(config):
            raise ContractError("reference-scoring config differs from the protocol lock")
    if args.mode == "stage2_dev":
        if not args.design_lock:
            raise ContractError("Stage-2 development scoring requires --design-lock")
        lock = validate_design_lock(args.design_lock)
        if lock["config"]["object_sha256"] != canonical_sha256(config):
            raise ContractError("development-scoring config differs from the design lock")
    if args.mode == "retrospective":
        if not args.selection_lock:
            raise ContractError("retrospective scoring requires --selection-lock")
        validate_selection_lock(args.selection_lock)
    metadata = score_adapter(
        config=config, mode=args.mode, model_key=args.model_key,
        manifest_path=args.manifest, prompt_cache_path=args.prompt_cache,
        out_path=args.out, condition=args.condition, adapter_path=args.adapter,
        partition_ids=_partition_ids(args.partition, args.partition_role),
        batch_size=args.batch_size, device=args.device, allow_cpu=args.allow_cpu,
    )
    _print_json({"status": metadata["status"], "output": str(output_path(args.out))})
    return 0 if metadata["status"] == "completed" else 1


def cmd_train_stage2(args) -> int:
    config = load_config(args.config)
    if not args.dry_run and not config["study_id"].endswith("_smoke"):
        if not args.design_lock:
            raise ContractError("non-smoke Stage-2 training requires --design-lock")
        lock = validate_design_lock(args.design_lock)
        if lock["config"]["object_sha256"] != canonical_sha256(config):
            raise ContractError("Stage-2 config differs from the design lock")
    metadata = train_stage2(
        config=config, model_key=args.model_key, seed=args.seed,
        objective=args.objective, sampler=args.sampler,
        manifest_path=args.manifest, prompt_cache_path=args.prompt_cache,
        selection_path=args.selection, reference_path=args.reference,
        stage1_adapter_path=args.stage1_adapter, out_path=args.out,
        device=args.device, allow_cpu=args.allow_cpu, dry_run=args.dry_run,
    )
    _print_json({"status": metadata["status"], "output": str(output_path(args.out))})
    return 0 if metadata["status"] in {"completed", "dry_run"} else 1


def cmd_select_checkpoints(args) -> int:
    output = select_checkpoints(
        config=load_config(args.config), stage1_scores_path=args.stage1_scores,
        candidate_scores_path=args.candidate_scores, out_path=args.out,
    )
    print(f"selected {len(output)} checkpoint cells -> {output_path(args.out)}")
    return 0


def cmd_smoke_audit(args) -> int:
    config = load_config(args.config)
    root = output_path(args.root)
    records = {}
    for objective in ("verdict_ce", "pair_ce", "dpo"):
        path = root / objective / "run_metadata.json"
        if not path.is_file():
            raise ContractError(f"smoke metadata is missing: {path}")
        record = read_json(path)
        if record.get("status") != "completed" or record.get("objective") != objective:
            raise ContractError(f"smoke objective did not complete cleanly: {objective}")
        if record.get("runtime", {}).get("device") != "cuda":
            raise ContractError(f"smoke objective was not executed on CUDA: {objective}")
        if int(record.get("completed_steps", -1)) != int(config["stage2"]["max_steps"]):
            raise ContractError(f"smoke objective has the wrong step count: {objective}")
        if set(record.get("checkpoints", {})) != {
            str(step) for step in config["stage2"]["checkpoint_steps"]
        }:
            raise ContractError(f"smoke checkpoint inventory is incomplete: {objective}")
        records[objective] = record
    equality_fields = (
        "model_key", "seed", "sampler", "manifest_sha256", "prompt_cache_sha256",
        "selection_sha256", "reference_sha256", "stage1_adapter_sha256",
        "ordered_sample_ids_sha256",
    )
    baseline = records["verdict_ce"]
    for objective, record in records.items():
        for field in equality_fields:
            if record.get(field) != baseline.get(field):
                raise ContractError(f"smoke cells differ on {field}: {objective}")
        step_zero = record.get("step_zero") or {}
        if abs(float(step_zero.get("mean_dpo_loss", -1)) - math.log(2.0)) > max(
            float(config["stage2"]["reference_margin_atol"]), 1e-6
        ):
            raise ContractError(f"smoke step-zero DPO identity failed: {objective}")
    audit = {
        "kind": "paper_c_three_objective_gpu_smoke_audit",
        "status": "passed",
        "model_key": baseline["model_key"],
        "seed": baseline["seed"],
        "sampler": baseline["sampler"],
        "shared_stage1_adapter_sha256": baseline["stage1_adapter_sha256"],
        "shared_ordered_sample_ids_sha256": baseline["ordered_sample_ids_sha256"],
        "objectives": {
            objective: {
                "wall_time_seconds": record.get("wall_time_seconds"),
                "peak_memory_bytes": record.get("peak_memory_bytes"),
                "final_loss": record.get("final_loss"),
                "step_zero": record.get("step_zero"),
            }
            for objective, record in records.items()
        },
    }
    out = root.parent / "SMOKE_AUDIT.json"
    write_json(out, audit)
    _print_json({"status": "passed", "audit": str(out)})
    return 0


def cmd_power(args) -> int:
    """Report what this design can detect, and fail closed if that is less than its target.

    Reads a seed-variance bundle of one record per (model_key, seed) with `transfer_ap`. The bundle
    is produced from the vendored parent scores by `bootstrap-inputs`; it is never a Paper C result,
    so this command is safe to run -- and must be run -- before Stage 1.
    """
    config = load_config(args.config)
    analysis = config["analysis"]
    power_config = analysis["power"]
    rows = read_jsonl(output_path(args.seed_variance))
    variance = seed_sd_by_model(rows, metric="transfer_ap")
    report = design_power_report(
        seed_variance=variance,
        n_models=len(config["models"]),
        n_seeds=len(config["seeds"]),
        target_effect=float(power_config["target_effect_transfer"]),
        pairing_variance_reduction=float(power_config["assumed_pairing_variance_reduction"]),
        ladder_points=len(config["stage2"]["checkpoint_steps"]),
    )
    if args.out:
        write_json(args.out, report)
    _print_json(report)
    if args.gate:
        assert_design_powered(report)
    return 0


def cmd_effective_lr(args) -> int:
    """Quantify how much of reference centering is a uniform gradient rescale at step zero.

    Consumes a reference-margin bundle (one record per row with `reference_margin`) and reports the
    DPO/PairCE effective-learning-rate ratio plus the dispersion of the PairCE weights. A large
    ratio with small dispersion is the confound that makes step-matched contrasts uninterpretable.
    """
    config = load_config(args.config)
    rows = read_jsonl(output_path(args.reference_margins))
    margins = [float(row["reference_margin"]) for row in rows]
    report = effective_learning_rate_ratio(margins, beta=float(config["stage2"]["beta"]))
    if args.out:
        write_json(args.out, report)
    _print_json(report)
    return 0


def cmd_frontier(args) -> int:
    """Compute the primary frontier estimands from scored candidate checkpoints.

    Input is one record per scored candidate: model_key, seed, sampler, objective, step,
    represented_ap, transfer_ap. The ladder is what makes the estimand possible, so this refuses to
    run on a single-checkpoint bundle.
    """
    config = load_config(args.config)
    analysis = config["analysis"]
    rows = read_jsonl(output_path(args.candidate_scores))
    steps = {int(row["step"]) for row in rows}
    if len(steps) < 2:
        raise ContractError(
            "the frontier estimand needs at least two checkpoint steps per cell; a "
            "single-checkpoint bundle can only support the secondary point estimands"
        )
    report = frontier_report(
        rows,
        replicates=int(analysis["bootstrap_replicates"]),
        seed=int(analysis["bootstrap_seed"]),
    )
    if args.out:
        write_json(args.out, report)
    _print_json(report)
    return 0


def cmd_clean_build(args) -> int:
    load_config(args.config)
    build = output_path("build")
    if build.exists():
        shutil.rmtree(build)
    print(f"removed generated build directory: {build}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper C isolated research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def config_arg(subparser):
        subparser.add_argument("--config", default=DEFAULT_CONFIG)

    validate = subparsers.add_parser("validate-config")
    config_arg(validate); validate.set_defaults(func=cmd_validate_config)
    loss = subparsers.add_parser("loss-check")
    config_arg(loss); loss.set_defaults(func=cmd_loss_check)
    doctor_parser = subparsers.add_parser("doctor")
    config_arg(doctor_parser); doctor_parser.set_defaults(func=cmd_doctor)
    bootstrap = subparsers.add_parser("bootstrap-inputs")
    config_arg(bootstrap); bootstrap.set_defaults(func=cmd_bootstrap)

    prompts = subparsers.add_parser("prepare-prompts")
    config_arg(prompts)
    prompts.add_argument("--model-key", required=True)
    prompts.add_argument("--manifest", required=True)
    prompts.add_argument("--out", required=True)
    prompts.set_defaults(func=cmd_prepare_prompts)

    partition = subparsers.add_parser("partition")
    config_arg(partition)
    partition.add_argument("--manifest", required=True)
    partition.add_argument("--out", required=True)
    partition.set_defaults(func=cmd_partition)

    select = subparsers.add_parser("select")
    config_arg(select)
    select.add_argument("--partition", required=True)
    select.add_argument("--reference", required=True)
    select.add_argument("--model-key", required=True)
    select.add_argument("--seed", required=True, type=int)
    select.add_argument("--out", required=True)
    select.set_defaults(func=cmd_select)

    protocol_lock = subparsers.add_parser("create-protocol-lock")
    config_arg(protocol_lock)
    protocol_lock.add_argument("--out", default=str(PROTOCOL_LOCK_PATH))
    protocol_lock.add_argument("--supersedes")
    protocol_lock.set_defaults(func=cmd_protocol_lock)
    design_lock = subparsers.add_parser("create-stage2-design-lock")
    design_lock.add_argument("--protocol-lock", required=True)
    design_lock.add_argument("--stage1-inventory", required=True)
    design_lock.add_argument("--stage2-input-inventory", required=True)
    design_lock.add_argument("--partition", required=True)
    design_lock.set_defaults(func=cmd_design_lock)
    validate_lock = subparsers.add_parser("validate-lock")
    validate_lock.add_argument("--lock", required=True)
    validate_lock.set_defaults(func=cmd_validate_lock)
    selection_lock = subparsers.add_parser("create-selection-lock")
    selection_lock.add_argument("--design-lock", required=True)
    selection_lock.add_argument("--candidates", required=True)
    selection_lock.add_argument("--development-scores", required=True)
    selection_lock.add_argument("--selection-table", required=True)
    selection_lock.set_defaults(func=cmd_selection_lock)

    inventory_stage1 = subparsers.add_parser("inventory-stage1")
    config_arg(inventory_stage1)
    inventory_stage1.add_argument("--runs-root", required=True)
    inventory_stage1.add_argument("--prompt-root", required=True)
    inventory_stage1.add_argument("--out", required=True)
    inventory_stage1.set_defaults(func=cmd_inventory_stage1)
    inventory_inputs = subparsers.add_parser("inventory-stage2-inputs")
    config_arg(inventory_inputs)
    inventory_inputs.add_argument("--reference-root", required=True)
    inventory_inputs.add_argument("--selection-root", required=True)
    inventory_inputs.add_argument("--out", required=True)
    inventory_inputs.set_defaults(func=cmd_inventory_stage2_inputs)
    inventory_candidates = subparsers.add_parser("inventory-candidates")
    config_arg(inventory_candidates)
    inventory_candidates.add_argument("--runs-root", required=True)
    inventory_candidates.add_argument("--out", required=True)
    inventory_candidates.set_defaults(func=cmd_inventory_candidates)
    inventory_scores = subparsers.add_parser("inventory-development-scores")
    config_arg(inventory_scores)
    inventory_scores.add_argument("--stage1-root", required=True)
    inventory_scores.add_argument("--candidate-root", required=True)
    inventory_scores.add_argument("--out", required=True)
    inventory_scores.set_defaults(func=cmd_inventory_dev_scores)

    prospective = subparsers.add_parser("create-prospective-lock")
    prospective.set_defaults(func=cmd_prospective_lock)

    stage1 = subparsers.add_parser("train-stage1")
    config_arg(stage1)
    stage1.add_argument("--model-key", required=True)
    stage1.add_argument("--seed", required=True, type=int)
    stage1.add_argument("--manifest", required=True)
    stage1.add_argument("--prompt-cache", required=True)
    stage1.add_argument("--out", required=True)
    stage1.add_argument("--protocol-lock")
    stage1.add_argument("--device")
    stage1.add_argument("--allow-cpu", action="store_true")
    stage1.add_argument("--dry-run", action="store_true")
    stage1.set_defaults(func=cmd_train_stage1)

    score = subparsers.add_parser("score")
    config_arg(score)
    score.add_argument("--mode", required=True, choices=("reference", "stage2_dev", "retrospective"))
    score.add_argument("--model-key", required=True)
    score.add_argument("--condition", required=True)
    score.add_argument("--manifest", required=True)
    score.add_argument("--prompt-cache", required=True)
    score.add_argument("--adapter")
    score.add_argument("--partition")
    score.add_argument("--partition-role")
    score.add_argument("--selection-lock")
    score.add_argument("--protocol-lock")
    score.add_argument("--design-lock")
    score.add_argument("--out", required=True)
    score.add_argument("--batch-size", type=int, default=4)
    score.add_argument("--device")
    score.add_argument("--allow-cpu", action="store_true")
    score.set_defaults(func=cmd_score)

    stage2 = subparsers.add_parser("train-stage2")
    config_arg(stage2)
    stage2.add_argument("--model-key", required=True)
    stage2.add_argument("--seed", required=True, type=int)
    stage2.add_argument("--objective", required=True, choices=("verdict_ce", "pair_ce", "dpo"))
    stage2.add_argument("--sampler", required=True, choices=("uncertain", "matched_random"))
    stage2.add_argument("--manifest", required=True)
    stage2.add_argument("--prompt-cache", required=True)
    stage2.add_argument("--selection", required=True)
    stage2.add_argument("--reference", required=True)
    stage2.add_argument("--stage1-adapter", required=True)
    stage2.add_argument("--out", required=True)
    stage2.add_argument("--design-lock")
    stage2.add_argument("--device")
    stage2.add_argument("--allow-cpu", action="store_true")
    stage2.add_argument("--dry-run", action="store_true")
    stage2.set_defaults(func=cmd_train_stage2)

    checkpoint = subparsers.add_parser("select-checkpoints")
    config_arg(checkpoint)
    checkpoint.add_argument("--stage1-scores", required=True)
    checkpoint.add_argument("--candidate-scores", required=True)
    checkpoint.add_argument("--out", required=True)
    checkpoint.set_defaults(func=cmd_select_checkpoints)

    smoke_audit = subparsers.add_parser("smoke-audit")
    config_arg(smoke_audit)
    smoke_audit.add_argument("--root", required=True)
    smoke_audit.set_defaults(func=cmd_smoke_audit)

    power = subparsers.add_parser("power")
    config_arg(power)
    power.add_argument("--seed-variance", required=True,
                       help="JSONL of one record per (model_key, seed) with transfer_ap")
    power.add_argument("--out")
    power.add_argument("--gate", action="store_true",
                       help="exit nonzero if the design cannot detect its own target effect")
    power.set_defaults(func=cmd_power)

    eff_lr = subparsers.add_parser("effective-lr")
    config_arg(eff_lr)
    eff_lr.add_argument("--reference-margins", required=True,
                        help="JSONL of one record per update row with reference_margin")
    eff_lr.add_argument("--out")
    eff_lr.set_defaults(func=cmd_effective_lr)

    frontier = subparsers.add_parser("frontier")
    config_arg(frontier)
    frontier.add_argument("--candidate-scores", required=True,
                          help="JSONL: model_key, seed, sampler, objective, step, "
                               "represented_ap, transfer_ap")
    frontier.add_argument("--out")
    frontier.set_defaults(func=cmd_frontier)

    clean = subparsers.add_parser("clean-build")
    config_arg(clean); clean.set_defaults(func=cmd_clean_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[paper-c] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
