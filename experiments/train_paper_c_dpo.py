#!/usr/bin/env python
"""Shared Stage-1/Stage-2 trainer for the Paper C v2 matched DPO study.

Stage 1 regenerates the parent Paper A SFT adapter bytes in the fresh Paper C
namespace.  Stage 2 loads one immutable Stage-1 adapter and changes only the
one-token scalar loss: verdict_ce, pair_ce, or dpo.

This script does not use TRL.  Generic sequence-DPO defaults would change
completion support, dropout, reference handling, and truncation across arms.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import uuid

_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE.parent), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import paper_a_common as A  # noqa: E402
import paper_c_dpo_common as P  # noqa: E402
from run_paper_a_sft import load_train_rows, train_one_cell  # noqa: E402


def _load_paper_c_lock(path: str, *, allow_development: bool) -> dict:
    lock = P.read_json(path)
    if int(lock.get("lock_schema_version", -1)) != 2:
        raise P.PaperCContractError("Paper C lock schema must be 2")
    expected = P.canonical_sha256(
        {key: value for key, value in lock.items() if key != "lock_sha256"})
    if expected != lock.get("lock_sha256"):
        raise P.PaperCContractError("Paper C lock hash mismatch")
    status = lock.get("finalization_status")
    if status != "final" and not allow_development:
        raise P.PaperCContractError(
            "claim-bearing training requires a final lock; use --development only for smoke work")
    config = (lock.get("config") or {}).get("value")
    P.validate_config(config or {})
    if status == "final":
        P.validate_execution_sources(lock, A.REPO_ROOT)
        software_issues = A.protocol_software_issues(
            A.software_versions(), lock.get("software_versions"))
        if software_issues:
            raise P.PaperCContractError(
                f"software environment differs from final lock: {software_issues}")
    return lock


def _path_from_repo(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(A.REPO_ROOT) / path


def _stage1_cell(lock: dict, model_key: str, seed: int,
                 explicit: str | None) -> tuple[Path, str, str | None]:
    if explicit:
        path = Path(explicit).resolve()
        if not A.adapter_is_present(str(path)):
            raise P.PaperCContractError(f"Stage-1 adapter is missing: {path}")
        run_meta = path.parent / "run_meta.json"
        return path, A.sha256_dir(path), (P.sha256_file(run_meta) if run_meta.is_file() else None)
    inventory = (lock.get("stage1_inventory") or {}).get("cells") or {}
    key = f"{model_key}/seed_{seed}"
    cell = inventory.get(key)
    if not cell:
        raise P.PaperCContractError(f"Paper C lock has no Stage-1 inventory cell {key}")
    path = _path_from_repo(cell["adapter_dir"])
    observed = A.sha256_dir(path)
    if observed != cell.get("adapter_sha256"):
        raise P.PaperCContractError(f"Stage-1 adapter bytes drifted for {key}")
    return path, observed, cell.get("run_meta_sha256")


def _manifest_path(lock: dict, explicit: str | None) -> Path:
    record = (lock.get("manifests") or {}).get("train.jsonl") or {}
    path = Path(explicit).resolve() if explicit else _path_from_repo(str(record.get("path", "")))
    if not path.is_file():
        raise P.PaperCContractError(f"train manifest is missing: {path}")
    expected = record.get("sha256")
    if expected and P.sha256_file(path) != expected:
        raise P.PaperCContractError("train manifest differs from Paper C lock")
    return path


def _stage2_input_paths(lock: dict, model_key: str, seed: int,
                        selection_override: str | None,
                        reference_override: str | None) -> tuple[Path, Path, dict | None]:
    key = f"{model_key}/seed_{seed}"
    cell = ((lock.get("stage2_inputs") or {}).get("cells") or {}).get(key)
    if lock.get("finalization_status") == "final":
        if not cell:
            raise P.PaperCContractError(f"final lock has no Stage-2 inputs for {key}")
        selection = _path_from_repo(cell["selection"]["path"])
        reference = _path_from_repo(cell["reference"]["path"])
        for label, path, override in (
            ("selection", selection, selection_override),
            ("reference", reference, reference_override),
        ):
            if override and Path(override).resolve() != path.resolve():
                raise P.PaperCContractError(
                    f"final {label} override differs from the lock-bound file")
            observed = P.sha256_file(path)
            if observed != cell[label]["sha256"]:
                raise P.PaperCContractError(f"lock-bound {label} bytes drifted for {key}")
        return selection, reference, cell
    if not selection_override or not reference_override:
        raise P.PaperCContractError(
            "development Stage-2 training requires --selection and --reference")
    return Path(selection_override).resolve(), Path(reference_override).resolve(), cell


def _selected_training_rows(
    train_path: Path,
    selection_path: Path,
    reference_path: Path,
    sampler: str,
) -> tuple[list[dict], list[dict], str, str]:
    train_rows = load_train_rows(str(train_path))
    raw_rows = P.read_jsonl(train_path)
    raw_by_id = {str(row["sample_id"]): row for row in raw_rows}
    text_by_id = {str(row["sample_id"]): row for row in train_rows}
    if len(raw_by_id) != len(raw_rows) or len(text_by_id) != len(train_rows):
        raise P.PaperCContractError("duplicate sample IDs in train manifest")

    selections = P.read_jsonl(selection_path)
    P.validate_selections(selections)
    selected_ids = P.selection_ids(selections, sampler)
    selected_records = {
        str(row["sample_id"]): row for row in selections
        if row.get("selection_role") == sampler
    }
    reference_rows = P.read_jsonl(reference_path)
    reference_by_id = {str(row.get("sample_id", "")): row for row in reference_rows}
    if len(reference_by_id) != len(reference_rows):
        raise P.PaperCContractError("duplicate sample IDs in reference artifact")

    out = []
    for sample_id in selected_ids:
        if sample_id not in text_by_id or sample_id not in raw_by_id:
            raise P.PaperCContractError(f"selected row absent from train manifest: {sample_id}")
        if sample_id not in reference_by_id:
            raise P.PaperCContractError(f"selected row absent from reference artifact: {sample_id}")
        identity = raw_by_id[sample_id]
        selection = selected_records[sample_id]
        for field in ("content_sha256", "family_id", "source"):
            if str(identity.get(field)) != str(selection.get(field)):
                raise P.PaperCContractError(
                    f"selection identity mismatch for {sample_id}: {field}")
        gold = P.normalize_gold(identity.get("gold", identity.get("label")))
        reference = reference_by_id[sample_id]
        ref_margin = P.signed_margin(
            float(reference["safe_logit"]), float(reference["unsafe_logit"]), gold)
        stored_margin = float(selection["reference_signed_margin"])
        if not math.isclose(ref_margin, stored_margin, rel_tol=0, abs_tol=1e-10):
            raise P.PaperCContractError(f"reference margin mismatch for {sample_id}")
        out.append({
            "sample_id": sample_id,
            "text": text_by_id[sample_id]["text"],
            "gold": gold,
            "reference_signed_margin": ref_margin,
        })
    if not out:
        raise P.PaperCContractError(f"selection contains no rows for sampler {sampler}")
    return out, selections, P.sha256_file(selection_path), P.sha256_file(reference_path)


def _expected_output(lock: dict, model_key: str, seed: int,
                     sampler: str, objective: str) -> Path:
    root = _path_from_repo(str(lock["artifact_root"]))
    return root / "runs" / model_key / f"seed_{seed}" / sampler / objective


def _run_meta(
    *, lock: dict, model_key: str, seed: int, objective: str, sampler: str,
    stage1_adapter: Path, stage1_sha: str, stage1_run_meta_sha: str | None,
    train_path: Path,
    selection_sha: str, reference_sha: str, out_dir: Path,
) -> dict:
    config = lock["config"]["value"]
    model = lock["models"][model_key]
    effective = dict(config["stage2"])
    effective.update({"objective": objective, "sampler": sampler})
    return {
        "run_id": f"{model_key}_{objective}_{sampler}_seed{seed}_{uuid.uuid4().hex[:8]}",
        "study_id": config["study_id"],
        "run_kind": "stage2",
        "model_key": model_key,
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "tokenizer_revision": model["tokenizer_revision"],
        "objective": objective,
        "sampler": sampler,
        "condition": P.run_condition(objective, sampler),
        "seed": int(seed),
        "training_seed": int(seed),
        "data_order_seed": int((lock.get("parent_data") or {}).get(
            "data_order_seed", A.DEFAULT_DATA_ORDER_SEED)),
        "lock_sha256": lock["lock_sha256"],
        "config_sha256": lock["config"]["object_sha256"],
        "execution_sources_sha256": lock["execution_sources"]["aggregate_sha256"],
        "loss_formula_version": P.LOSS_FORMULA_VERSION,
        "effective_stage2_config": effective,
        "effective_stage2_config_sha256": P.canonical_sha256(effective),
        "stage1_adapter_dir": str(stage1_adapter),
        "stage1_adapter_sha256": stage1_sha,
        "stage1_run_meta_sha256": stage1_run_meta_sha,
        "train_manifest": str(train_path),
        "train_manifest_sha256": P.sha256_file(train_path),
        "selection_sha256": selection_sha,
        "reference_margin_sha256": reference_sha,
        "out_dir": str(out_dir),
        "software_versions": A.software_versions(),
        "runtime_environment": None,
        "decision_tokens": None,
        "prompt_fingerprint_sha256": None,
        "ordered_sample_ids_sha256": None,
        "initial_adapter_sha256": stage1_sha,
        "final_adapter_sha256": None,
        "global_steps": int(config["stage2"]["max_steps"]),
        "completed_steps": 0,
        "examples_seen": 0,
        "prompt_tokens_seen_estimate": 0,
        "wall_time_s": None,
        "peak_mem_bytes": None,
        "status": "pending",
        "failure_reason": None,
    }


def cmd_stage1(args: argparse.Namespace) -> int:
    # The immutable parent lock/manifest/recipe are inputs; today's checkout is
    # independently rebound by the Paper C lock and need not reproduce every
    # historical Paper A analyzer byte.
    parent = A.load_lock(args.parent_lock, allow_legacy=False, verify_files=False)
    if args.model_key not in A.lock_model_panel(parent):
        raise P.PaperCContractError(f"unknown model key: {args.model_key}")
    if args.seed not in A.lock_seeds(parent):
        raise P.PaperCContractError(f"seed is outside the parent panel: {args.seed}")
    if args.steps is not None and not (args.development or args.dry_run):
        raise P.PaperCContractError("a final Stage-1 run may not override the parent step count")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise P.PaperCContractError(f"refusing to overwrite nonempty Stage-1 output: {out}")
    train_path = Path(args.train_manifest or (
        Path(A.abspath(A.artifact_paths(parent)["manifests"])) / "train.jsonl"))
    meta = train_one_cell(
        parent, args.model_key, args.seed, args.out, str(train_path),
        steps=args.steps, dry_run=args.dry_run, device=args.device,
        run_kind=("dry_run" if args.dry_run else
                  "nonfinal" if args.development else "final"),
        kl_beta=0.0,
    )
    print(f"[paper-c stage1] {args.model_key}/seed_{args.seed}: {meta['status']}")
    return 0 if meta["status"] in {"completed", "dry_run"} else 1


def cmd_stage2(args: argparse.Namespace) -> int:
    lock = _load_paper_c_lock(args.lock, allow_development=args.development)
    config = lock["config"]["value"]
    if args.model_key not in config["models"]:
        raise P.PaperCContractError(f"model is outside the Paper C panel: {args.model_key}")
    if args.seed not in config["seeds"]:
        raise P.PaperCContractError(f"seed is outside the Paper C panel: {args.seed}")
    if args.objective not in P.OBJECTIVES or args.sampler not in P.SAMPLERS:
        raise P.PaperCContractError("objective/sampler is outside the locked condition grid")
    if lock["finalization_status"] == "final" and args.stage1_adapter:
        raise P.PaperCContractError(
            "a final Stage-2 run must use the Stage-1 adapter bound by the lock")

    out_dir = Path(args.out).resolve() if args.out else _expected_output(
        lock, args.model_key, args.seed, args.sampler, args.objective)
    expected = _expected_output(lock, args.model_key, args.seed, args.sampler, args.objective)
    if lock["finalization_status"] == "final" and out_dir.resolve() != expected.resolve():
        raise P.PaperCContractError(
            f"final output is lock-authoritative: expected {expected}, got {out_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise P.PaperCContractError(f"refusing to overwrite nonempty Stage-2 output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_adapter, stage1_sha, stage1_run_meta_sha = _stage1_cell(
        lock, args.model_key, args.seed, args.stage1_adapter)
    train_path = _manifest_path(lock, args.train_manifest)
    selection_path, reference_path, locked_inputs = _stage2_input_paths(
        lock, args.model_key, args.seed, args.selection, args.reference)
    rows, selection_rows, selection_sha, reference_sha = _selected_training_rows(
        train_path, selection_path, reference_path, args.sampler)
    meta = _run_meta(
        lock=lock, model_key=args.model_key, seed=args.seed,
        objective=args.objective, sampler=args.sampler,
        stage1_adapter=stage1_adapter, stage1_sha=stage1_sha,
        stage1_run_meta_sha=stage1_run_meta_sha,
        train_path=train_path, selection_sha=selection_sha,
        reference_sha=reference_sha, out_dir=out_dir,
    )
    meta["dataset_rows"] = len(rows)
    if locked_inputs:
        meta["selection_metadata_sha256"] = locked_inputs["selection_metadata"]["sha256"]
        meta["reference_metadata_sha256"] = locked_inputs["reference_metadata"]["sha256"]
    meta["ordered_sample_ids_sha256"] = P.sha256_ordered(
        row["sample_id"] for row in rows)
    meta_path = out_dir / "run_meta.json"
    started = time.time()
    meta["start_utc"] = A.utcnow()

    if args.dry_run:
        meta["status"] = "dry_run"
        meta["wall_time_s"] = round(time.time() - started, 3)
        meta["completion_utc"] = A.utcnow()
        P.write_json(meta_path, meta)
        print(f"[paper-c stage2] dry-run {meta['condition']} rows={len(rows)} -> {out_dir}")
        return 0

    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torch.utils.data import Dataset, RandomSampler
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,
            TrainingArguments,
        )
        from peft import PeftModel

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model_record = lock["models"][args.model_key]
        device = args.device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if getattr(torch.backends, "mps", None)
                                 and torch.backends.mps.is_available() else "cpu")
        meta["runtime_environment"] = A.runtime_environment(device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_record["model_id"], revision=model_record["tokenizer_revision"],
            trust_remote_code=bool(model_record.get("trust_remote_code", True)),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"
        decision = A.resolve_decision_tokens(tokenizer)
        for label in ("safe", "unsafe"):
            ids = tokenizer.encode(decision[f"{label}_str"], add_special_tokens=False)
            if ids != [decision[f"{label}_id"]]:
                raise P.PaperCContractError(
                    f"{label} decision surface is not exactly one locked token")
        meta["decision_tokens"] = decision

        build_prompt, _ = A.require_prompts()
        max_length = int((lock.get("parent_recipe") or {}).get("max_length", 1024))

        class Stage2Dataset(Dataset):
            def __init__(self, records: list[dict]):
                self.examples = []
                self.prompt_fingerprints = []
                self.total_prompt_tokens = 0
                for record in records:
                    rendered, truncation = A.budgeted_prompt(
                        tokenizer, build_prompt, record["text"], max_length,
                        reserved_tokens=1,
                    )
                    if not truncation["wrapper_preserved"]:
                        raise P.PaperCContractError("Stage-2 prompt lost classifier wrapper")
                    input_ids = tokenizer(
                        rendered, add_special_tokens=False, truncation=False)["input_ids"]
                    if not input_ids or len(input_ids) + 1 > max_length:
                        raise P.PaperCContractError("Stage-2 prompt violates locked token budget")
                    target_id = (decision["unsafe_id"] if record["gold"] == 1
                                 else decision["safe_id"])
                    self.examples.append({
                        "sample_id": record["sample_id"],
                        "input_ids": input_ids,
                        "target_id": target_id,
                        "gold_sign": 1 if record["gold"] == 1 else -1,
                        "reference_signed_margin": record["reference_signed_margin"],
                    })
                    self.prompt_fingerprints.append(A.content_sha256(rendered))
                    self.total_prompt_tokens += len(input_ids)

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, index):
                return self.examples[index]

        dataset = Stage2Dataset(rows)
        meta["prompt_fingerprint_sha256"] = P.sha256_ordered(
            dataset.prompt_fingerprints)

        def collate(batch):
            width = max(len(item["input_ids"]) for item in batch)
            pad = tokenizer.pad_token_id
            input_ids, attention, targets, signs, references, sample_ids = [], [], [], [], [], []
            for item in batch:
                gap = width - len(item["input_ids"])
                input_ids.append(item["input_ids"] + [pad] * gap)
                attention.append([1] * len(item["input_ids"]) + [0] * gap)
                targets.append(item["target_id"])
                signs.append(item["gold_sign"])
                references.append(item["reference_signed_margin"])
                sample_ids.append(item["sample_id"])
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
                "paper_c_target_ids": torch.tensor(targets, dtype=torch.long),
                "paper_c_gold_signs": torch.tensor(signs, dtype=torch.float32),
                "paper_c_reference_margins": torch.tensor(references, dtype=torch.float32),
                "paper_c_sample_ids": sample_ids,
            }

        dtype_name = str(model_record.get("dtype", "bfloat16"))
        model_kwargs = {
            "revision": model_record["model_revision"],
            "dtype": A.torch_dtype_from_name(torch, dtype_name),
            "trust_remote_code": bool(model_record.get("trust_remote_code", True)),
        }
        if model_record.get("attn_implementation"):
            model_kwargs["attn_implementation"] = model_record["attn_implementation"]
        base = AutoModelForCausalLM.from_pretrained(model_record["model_id"], **model_kwargs)
        model = PeftModel.from_pretrained(base, stage1_adapter, is_trainable=True)
        model.config.use_cache = False
        model.enable_input_require_grads()
        locked_dropout = float(config["stage2"]["stage2_dropout"])
        dropout_modules = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = locked_dropout
                dropout_modules += 1
        meta["stage2_dropout"] = locked_dropout
        meta["dropout_modules_configured"] = dropout_modules
        model.to(device)

        data_order_seed = int(meta["data_order_seed"])
        objective = args.objective
        beta = float(config["stage2"]["pairwise_beta"])

        # Reference margins must describe the exact step-zero policy.  Check all
        # selected rows before the optimizer exists; this also catches prompt,
        # token, adapter, dtype, and reference-cache mismatches.
        model.eval()
        init_differences = []
        init_dpo_losses = []
        preflight_batch = max(1, int(config["stage2"]["per_device_batch"]))
        with torch.no_grad():
            for offset in range(0, len(dataset), preflight_batch):
                batch = collate([
                    dataset[index]
                    for index in range(offset, min(len(dataset), offset + preflight_batch))
                ])
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                signs = batch["paper_c_gold_signs"].to(device)
                references = batch["paper_c_reference_margins"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                last = attention_mask.sum(1) - 1
                batch_index = torch.arange(last.shape[0], device=last.device)
                logits = outputs.logits[batch_index, last]
                score = (logits[:, decision["unsafe_id"]]
                         - logits[:, decision["safe_id"]]).float()
                margins = signs * score
                init_differences.extend((margins - references).detach().cpu().tolist())
                init_dpo_losses.extend(
                    F.softplus(-beta * (margins - references)).detach().cpu().tolist())
        max_reference_error = max(abs(float(value)) for value in init_differences)
        mean_initial_dpo_loss = float(np.mean(init_dpo_losses))
        meta["step_zero_reference_margin_max_abs_error"] = max_reference_error
        meta["step_zero_dpo_loss"] = mean_initial_dpo_loss
        tolerance = float(config["stage2"]["reference_margin_atol"])
        if max_reference_error > tolerance:
            raise P.PaperCContractError(
                f"step-zero policy/reference margins differ by {max_reference_error:.6g} "
                f"(atol={tolerance})")
        if abs(mean_initial_dpo_loss - math.log(2.0)) > max(tolerance, 1e-6):
            raise P.PaperCContractError(
                "step-zero DPO loss is not log(2) within the locked tolerance")
        model.train()

        class SharedObjectiveTrainer(Trainer):
            def _get_train_sampler(self, *unused_args, **unused_kwargs):
                generator = torch.Generator()
                generator.manual_seed(data_order_seed)
                return RandomSampler(self.train_dataset, generator=generator)

            def compute_loss(self, model, inputs, return_outputs=False, **unused_kwargs):
                target_ids = inputs.pop("paper_c_target_ids")
                gold_signs = inputs.pop("paper_c_gold_signs")
                reference_margins = inputs.pop("paper_c_reference_margins")
                inputs.pop("paper_c_sample_ids", None)
                outputs = model(**inputs)
                last = inputs["attention_mask"].sum(1) - 1
                batch_index = torch.arange(last.shape[0], device=last.device)
                logits = outputs.logits[batch_index, last]
                if objective == "verdict_ce":
                    loss = F.cross_entropy(logits.float(), target_ids)
                else:
                    score = logits[:, decision["unsafe_id"]] - logits[:, decision["safe_id"]]
                    margin = gold_signs * score.float()
                    if objective == "pair_ce":
                        loss = F.softplus(-beta * margin).mean()
                    elif objective == "dpo":
                        loss = F.softplus(-beta * (margin - reference_margins)).mean()
                    else:  # locked parser makes this unreachable; retain fail-closed behavior
                        raise P.PaperCContractError(f"unknown objective: {objective}")
                self.paper_c_last_loss = float(loss.detach().cpu())
                return (loss, outputs) if return_outputs else loss

        checkpoint_steps = {int(step) for step in config["stage2"]["checkpoint_steps"]}

        class LockedCheckpointCallback(TrainerCallback):
            def on_step_end(self, args, state, control, model=None, **kwargs):
                if int(state.global_step) in checkpoint_steps:
                    checkpoint_dir = (
                        out_dir / "checkpoints" / f"step_{int(state.global_step)}" / "adapter")
                    if checkpoint_dir.exists():
                        raise P.PaperCContractError(
                            f"refusing to overwrite checkpoint adapter: {checkpoint_dir}")
                    if model is None:
                        raise P.PaperCContractError("Trainer callback did not receive the model")
                    model.save_pretrained(checkpoint_dir)
                return control

        stage2 = config["stage2"]
        arguments = TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=int(stage2["per_device_batch"]),
            gradient_accumulation_steps=int(stage2["gradient_accumulation"]),
            max_steps=int(stage2["max_steps"]),
            learning_rate=float(stage2["learning_rate"]),
            lr_scheduler_type=str(stage2["scheduler"]),
            warmup_ratio=float(stage2["warmup_ratio"]),
            bf16=(device == "cuda" and dtype_name in {"bfloat16", "bf16"}),
            fp16=(device == "cuda" and dtype_name in {"float16", "fp16", "half"}),
            gradient_checkpointing=(device == "cuda"),
            logging_steps=10,
            save_strategy="no",
            remove_unused_columns=False,
            report_to=[],
            seed=int(args.seed),
        )
        trainer = SharedObjectiveTrainer(
            model=model, args=arguments, train_dataset=dataset,
            data_collator=collate, callbacks=[LockedCheckpointCallback()],
        )
        trainer.train()
        meta["checkpoint_adapters"] = {}
        for checkpoint_step in sorted(checkpoint_steps):
            checkpoint_dir = out_dir / "checkpoints" / f"step_{checkpoint_step}" / "adapter"
            if not A.adapter_is_present(str(checkpoint_dir)):
                raise P.PaperCContractError(
                    f"locked checkpoint adapter was not saved: step {checkpoint_step}")
            meta["checkpoint_adapters"][str(checkpoint_step)] = {
                "adapter_dir": str(checkpoint_dir),
                "adapter_sha256": A.sha256_dir(checkpoint_dir),
            }
        adapter_dir = out_dir / "adapter"
        model.save_pretrained(adapter_dir)
        meta["final_adapter_sha256"] = A.sha256_dir(adapter_dir)
        meta["completed_steps"] = int(trainer.state.global_step)
        effective_batch = int(stage2["per_device_batch"]) * int(stage2["gradient_accumulation"])
        meta["examples_seen"] = meta["completed_steps"] * effective_batch
        mean_prompt_tokens = dataset.total_prompt_tokens / max(1, len(dataset))
        meta["prompt_tokens_seen_estimate"] = int(mean_prompt_tokens * meta["examples_seen"])
        meta["final_loss"] = getattr(trainer, "paper_c_last_loss", None)
        if device == "cuda":
            meta["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated())
            meta["device_name"] = torch.cuda.get_device_name(0)
        meta["status"] = "completed"
    except Exception as exc:
        meta["status"] = "failed"
        meta["failure_reason"] = f"{type(exc).__name__}: {exc}"
        meta["traceback"] = traceback.format_exc()
    finally:
        meta["wall_time_s"] = round(time.time() - started, 3)
        meta["completion_utc"] = A.utcnow()
        P.write_json(meta_path, meta)

    print(f"[paper-c stage2] {meta['condition']}: {meta['status']} -> {out_dir}")
    if meta["status"] == "failed":
        print(f"  {meta['failure_reason']}", file=sys.stderr)
    return 0 if meta["status"] == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Paper C v2 Stage-1/Stage-2 cells")
    sub = parser.add_subparsers(dest="command", required=True)

    stage1 = sub.add_parser("stage1", help="regenerate one immutable Paper A SFT starting adapter")
    stage1.add_argument("--parent-lock", required=True)
    stage1.add_argument("--model-key", required=True, choices=list(A.MODEL_KEYS))
    stage1.add_argument("--seed", required=True, type=int)
    stage1.add_argument("--out", required=True)
    stage1.add_argument("--train-manifest")
    stage1.add_argument("--steps", type=int)
    stage1.add_argument("--device")
    stage1.add_argument("--dry-run", action="store_true")
    stage1.add_argument("--development", action="store_true")
    stage1.set_defaults(func=cmd_stage1)

    stage2 = sub.add_parser("stage2", help="train one matched Stage-2 objective/sampler cell")
    stage2.add_argument("--lock", required=True)
    stage2.add_argument("--model-key", required=True)
    stage2.add_argument("--seed", required=True, type=int)
    stage2.add_argument("--objective", required=True, choices=list(P.OBJECTIVES))
    stage2.add_argument("--sampler", required=True, choices=list(P.SAMPLERS))
    stage2.add_argument("--selection")
    stage2.add_argument("--reference")
    stage2.add_argument("--stage1-adapter")
    stage2.add_argument("--train-manifest")
    stage2.add_argument("--out")
    stage2.add_argument("--device")
    stage2.add_argument("--dry-run", action="store_true")
    stage2.add_argument("--development", action="store_true")
    stage2.set_defaults(func=cmd_stage2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (P.PaperCContractError, A.ArtifactContractError, OSError, ValueError) as exc:
        print(f"[paper-c train] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
