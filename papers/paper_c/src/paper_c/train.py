"""Shared Stage-1 and Stage-2 training implementation.

The Stage-2 data/model path is identical across objectives. The only branch is
the scalar objective in `torch_objective_loss`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import time
import traceback

from .contracts import (
    ContractError,
    OBJECTIVES,
    SAMPLERS,
    output_path,
    read_jsonl,
    sha256_directory,
    sha256_file,
    sha256_ordered,
    write_json,
)
from .objectives import assert_step_zero_identity, torch_objective_loss
from .runtime import (
    load_base_model,
    load_tokenizer,
    seed_everything,
    software_versions,
    validate_prompt_cache,
)
from .sampling import selection_ids, validate_selections


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _device(requested: str | None, *, allow_cpu: bool) -> str:
    import torch
    if requested:
        device = requested
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if device != "cuda" and not allow_cpu:
        raise ContractError("claim-bearing training requires CUDA; use --allow-cpu only for development")
    return device


def _fresh_output(path: str | Path) -> Path:
    output = output_path(path)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ContractError(f"refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _runtime_record(device: str) -> dict:
    record = {"device": device, "software": software_versions()}
    try:
        import torch
        if device == "cuda":
            record.update({
                "device_name": torch.cuda.get_device_name(0),
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            })
    except Exception:
        pass
    return record


def train_stage1(
    *,
    config: dict,
    model_key: str,
    seed: int,
    manifest_path: str | Path,
    prompt_cache_path: str | Path,
    out_path: str | Path,
    device: str | None = None,
    allow_cpu: bool = False,
    dry_run: bool = False,
) -> dict:
    if model_key not in config["models"] or seed not in config["seeds"]:
        raise ContractError("Stage-1 cell is outside the locked model/seed panel")
    prompt_rows, prompt_metadata = validate_prompt_cache(
        config=config, model_key=model_key, manifest_path=manifest_path,
        cache_path=prompt_cache_path,
    )
    out = _fresh_output(out_path)
    metadata_path = out / "run_metadata.json"
    stage1 = config["stage1"]
    metadata = {
        "schema_version": 1,
        "kind": "paper_c_stage1",
        "study_id": config["study_id"],
        "model_key": model_key,
        "seed": int(seed),
        "status": "pending",
        "started_utc": _utcnow(),
        "manifest_sha256": sha256_file(manifest_path),
        "prompt_cache_sha256": sha256_file(output_path(prompt_cache_path)),
        "ordered_sample_ids_sha256": sha256_ordered(row["sample_id"] for row in prompt_rows),
        "recipe": stage1,
        "prompt_metadata": prompt_metadata,
    }
    if dry_run:
        metadata.update({"status": "dry_run", "rows": len(prompt_rows), "completed_utc": _utcnow()})
        write_json(metadata_path, metadata)
        return metadata

    started = time.time()
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import Dataset, RandomSampler
        from transformers import Trainer, TrainingArguments

        run_device = _device(device, allow_cpu=allow_cpu)
        seed_everything(seed)
        tokenizer, decision = load_tokenizer(config, model_key)
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ContractError("tokenizer has no EOS token")

        class Stage1Dataset(Dataset):
            def __init__(self, rows):
                self.rows = []
                for row in rows:
                    target = decision["unsafe_id"] if row["gold"] == 1 else decision["safe_id"]
                    input_ids = list(row["input_ids"]) + [target, int(eos_id)]
                    labels = [-100] * len(row["input_ids"]) + [target, int(eos_id)]
                    self.rows.append({"input_ids": input_ids, "labels": labels})

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                return self.rows[index]

        def collate(batch):
            width = max(len(row["input_ids"]) for row in batch)
            input_ids, labels, attention = [], [], []
            for row in batch:
                gap = width - len(row["input_ids"])
                input_ids.append(row["input_ids"] + [tokenizer.pad_token_id] * gap)
                labels.append(row["labels"] + [-100] * gap)
                attention.append([1] * len(row["input_ids"]) + [0] * gap)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        model = load_base_model(config, model_key, device=run_device)
        lora = stage1["lora"]
        model = get_peft_model(model, LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            task_type="CAUSAL_LM",
            target_modules=list(lora["target_modules"]),
        ))
        model.enable_input_require_grads()

        class FixedOrderTrainer(Trainer):
            def _get_train_sampler(self, *unused_args, **unused_kwargs):
                generator = torch.Generator()
                generator.manual_seed(int(stage1["data_order_seed"]))
                return RandomSampler(self.train_dataset, generator=generator)

        arguments = TrainingArguments(
            output_dir=str(out / "trainer"),
            per_device_train_batch_size=int(stage1["per_device_batch"]),
            gradient_accumulation_steps=int(stage1["gradient_accumulation"]),
            max_steps=int(stage1["max_steps"]),
            learning_rate=float(stage1["learning_rate"]),
            lr_scheduler_type=str(stage1["scheduler"]),
            warmup_ratio=float(stage1["warmup_ratio"]),
            bf16=(run_device == "cuda" and config["models"][model_key]["dtype"] == "bfloat16"),
            fp16=False,
            gradient_checkpointing=(run_device == "cuda"),
            logging_steps=10,
            save_strategy="no",
            remove_unused_columns=False,
            report_to=[],
            seed=int(seed),
        )
        trainer = FixedOrderTrainer(
            model=model,
            args=arguments,
            train_dataset=Stage1Dataset(prompt_rows),
            data_collator=collate,
        )
        trainer.train()
        adapter = out / "adapter"
        model.save_pretrained(adapter)
        metadata.update({
            "status": "completed",
            "completed_steps": int(trainer.state.global_step),
            "rows": len(prompt_rows),
            "adapter_path": adapter.relative_to(output_path(".")).as_posix(),
            "adapter_sha256": sha256_directory(adapter),
            "decision_tokens": decision,
            "runtime": _runtime_record(run_device),
        })
        if run_device == "cuda":
            metadata["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    except Exception as exc:
        metadata.update({
            "status": "failed",
            "failure": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    metadata["wall_time_seconds"] = round(time.time() - started, 3)
    metadata["completed_utc"] = _utcnow()
    write_json(metadata_path, metadata)
    return metadata


def _selected_rows(
    *,
    prompt_rows: list[dict],
    selection_path: str | Path,
    reference_path: str | Path,
    sampler: str,
) -> list[dict]:
    selection_path = output_path(selection_path)
    reference_path = output_path(reference_path)
    selections = read_jsonl(selection_path)
    validate_selections(selections)
    wanted_ids = selection_ids(selections, sampler)
    cache_by_id = {row["sample_id"]: row for row in prompt_rows}
    reference_rows = read_jsonl(reference_path)
    reference_by_id = {str(row.get("sample_id", "")): row for row in reference_rows}
    if len(reference_by_id) != len(reference_rows):
        raise ContractError("reference artifact contains duplicate sample IDs")
    selection_by_id = {
        row["sample_id"]: row for row in selections if row["selection_role"] == sampler
    }
    rows = []
    for sample_id in wanted_ids:
        if sample_id not in cache_by_id or sample_id not in reference_by_id:
            raise ContractError(f"selected sample is missing prompt/reference data: {sample_id}")
        selected = selection_by_id[sample_id]
        reference = reference_by_id[sample_id]
        if selected["prompt_sha256"] != cache_by_id[sample_id]["prompt_sha256"]:
            raise ContractError(f"selected prompt fingerprint drifted: {sample_id}")
        if selected["prompt_sha256"] != reference.get("prompt_sha256"):
            raise ContractError(f"reference prompt fingerprint drifted: {sample_id}")
        rows.append({
            **cache_by_id[sample_id],
            "reference_signed_margin": float(reference["reference_signed_margin"]),
        })
    return rows


def train_stage2(
    *,
    config: dict,
    model_key: str,
    seed: int,
    objective: str,
    sampler: str,
    manifest_path: str | Path,
    prompt_cache_path: str | Path,
    selection_path: str | Path,
    reference_path: str | Path,
    stage1_adapter_path: str | Path,
    out_path: str | Path,
    device: str | None = None,
    allow_cpu: bool = False,
    dry_run: bool = False,
) -> dict:
    if model_key not in config["models"] or seed not in config["seeds"]:
        raise ContractError("Stage-2 cell is outside the locked panel")
    if objective not in OBJECTIVES or sampler not in SAMPLERS:
        raise ContractError("Stage-2 objective/sampler is outside the locked grid")
    prompt_rows, prompt_metadata = validate_prompt_cache(
        config=config, model_key=model_key, manifest_path=manifest_path,
        cache_path=prompt_cache_path,
    )
    selected_rows = _selected_rows(
        prompt_rows=prompt_rows,
        selection_path=selection_path,
        reference_path=reference_path,
        sampler=sampler,
    )
    stage1_adapter = output_path(stage1_adapter_path)
    if not stage1_adapter.is_dir():
        raise ContractError(f"Stage-1 adapter does not exist: {stage1_adapter}")
    out = _fresh_output(out_path)
    metadata_path = out / "run_metadata.json"
    stage2 = config["stage2"]
    metadata = {
        "schema_version": 1,
        "kind": "paper_c_stage2",
        "study_id": config["study_id"],
        "model_key": model_key,
        "seed": int(seed),
        "objective": objective,
        "sampler": sampler,
        "condition": f"{objective}__{sampler}",
        "status": "pending",
        "started_utc": _utcnow(),
        "manifest_sha256": sha256_file(manifest_path),
        "prompt_cache_sha256": sha256_file(output_path(prompt_cache_path)),
        "selection_sha256": sha256_file(selection_path),
        "reference_sha256": sha256_file(reference_path),
        "stage1_adapter_sha256": sha256_directory(stage1_adapter),
        "ordered_sample_ids_sha256": sha256_ordered(row["sample_id"] for row in selected_rows),
        "recipe": stage2,
        "prompt_metadata": prompt_metadata,
    }
    if dry_run:
        metadata.update({"status": "dry_run", "rows": len(selected_rows), "completed_utc": _utcnow()})
        write_json(metadata_path, metadata)
        return metadata

    started = time.time()
    try:
        import torch
        from peft import PeftModel
        from torch.utils.data import Dataset, RandomSampler
        from transformers import Trainer, TrainerCallback, TrainingArguments

        run_device = _device(device, allow_cpu=allow_cpu)
        seed_everything(seed)
        tokenizer, decision = load_tokenizer(config, model_key)

        class Stage2Dataset(Dataset):
            def __init__(self, rows):
                self.rows = rows

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                row = self.rows[index]
                return {
                    "input_ids": list(row["input_ids"]),
                    "target_id": decision["unsafe_id"] if row["gold"] == 1 else decision["safe_id"],
                    "gold_sign": 1.0 if row["gold"] == 1 else -1.0,
                    "reference_margin": float(row["reference_signed_margin"]),
                }

        dataset = Stage2Dataset(selected_rows)

        def collate(batch):
            width = max(len(row["input_ids"]) for row in batch)
            ids, masks, targets, signs, references = [], [], [], [], []
            for row in batch:
                gap = width - len(row["input_ids"])
                ids.append(row["input_ids"] + [tokenizer.pad_token_id] * gap)
                masks.append([1] * len(row["input_ids"]) + [0] * gap)
                targets.append(row["target_id"])
                signs.append(row["gold_sign"])
                references.append(row["reference_margin"])
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
                "paper_c_target_ids": torch.tensor(targets, dtype=torch.long),
                "paper_c_gold_signs": torch.tensor(signs, dtype=torch.float32),
                "paper_c_reference_margins": torch.tensor(references, dtype=torch.float32),
            }

        base = load_base_model(config, model_key, device=run_device)
        model = PeftModel.from_pretrained(base, stage1_adapter, is_trainable=True)
        model.config.use_cache = False
        model.enable_input_require_grads()
        dropout_modules = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
                dropout_modules += 1
        model.to(run_device)

        policy_margins, reference_margins = [], []
        model.eval()
        with torch.no_grad():
            for offset in range(0, len(dataset), int(stage2["per_device_batch"])):
                batch = collate([
                    dataset[index] for index in range(
                        offset, min(len(dataset), offset + int(stage2["per_device_batch"]))
                    )
                ])
                input_ids = batch["input_ids"].to(run_device)
                attention = batch["attention_mask"].to(run_device)
                signs = batch["paper_c_gold_signs"].to(run_device)
                reference = batch["paper_c_reference_margins"].to(run_device)
                logits = model(input_ids=input_ids, attention_mask=attention).logits
                last = attention.sum(1) - 1
                indexes = torch.arange(last.shape[0], device=last.device)
                next_logits = logits[indexes, last]
                margins = signs * (
                    next_logits[:, decision["unsafe_id"]] - next_logits[:, decision["safe_id"]]
                ).float()
                policy_margins.extend(margins.cpu().tolist())
                reference_margins.extend(reference.cpu().tolist())
        preflight = assert_step_zero_identity(
            policy_margins,
            reference_margins,
            beta=float(stage2["beta"]),
            atol=float(stage2["reference_margin_atol"]),
        )
        model.train()

        class SharedObjectiveTrainer(Trainer):
            def _get_train_sampler(self, *unused_args, **unused_kwargs):
                generator = torch.Generator()
                generator.manual_seed(int(config["stage1"]["data_order_seed"]))
                return RandomSampler(self.train_dataset, generator=generator)

            def compute_loss(self, model, inputs, return_outputs=False, **unused_kwargs):
                targets = inputs.pop("paper_c_target_ids")
                signs = inputs.pop("paper_c_gold_signs")
                references = inputs.pop("paper_c_reference_margins")
                outputs = model(**inputs)
                last = inputs["attention_mask"].sum(1) - 1
                indexes = torch.arange(last.shape[0], device=last.device)
                next_logits = outputs.logits[indexes, last]
                loss = torch_objective_loss(
                    objective=objective,
                    logits=next_logits,
                    target_ids=targets,
                    gold_signs=signs,
                    reference_margins=references,
                    safe_token_id=decision["safe_id"],
                    unsafe_token_id=decision["unsafe_id"],
                    beta=float(stage2["beta"]),
                )
                self.paper_c_last_loss = float(loss.detach().cpu())
                return (loss, outputs) if return_outputs else loss

        checkpoint_steps = set(int(step) for step in stage2["checkpoint_steps"])

        class SaveLockedCheckpoints(TrainerCallback):
            def on_step_end(self, args, state, control, model=None, **kwargs):
                step = int(state.global_step)
                if step in checkpoint_steps:
                    target = out / "checkpoints" / f"step_{step}" / "adapter"
                    if target.exists():
                        raise ContractError(f"checkpoint already exists: {target}")
                    model.save_pretrained(target)
                return control

        arguments = TrainingArguments(
            output_dir=str(out / "trainer"),
            per_device_train_batch_size=int(stage2["per_device_batch"]),
            gradient_accumulation_steps=int(stage2["gradient_accumulation"]),
            max_steps=int(stage2["max_steps"]),
            learning_rate=float(stage2["learning_rate"]),
            lr_scheduler_type=str(stage2["scheduler"]),
            warmup_ratio=float(stage2["warmup_ratio"]),
            bf16=(run_device == "cuda" and config["models"][model_key]["dtype"] == "bfloat16"),
            fp16=False,
            gradient_checkpointing=(run_device == "cuda"),
            logging_steps=10,
            save_strategy="no",
            remove_unused_columns=False,
            report_to=[],
            seed=int(seed),
        )
        trainer = SharedObjectiveTrainer(
            model=model, args=arguments, train_dataset=dataset,
            data_collator=collate, callbacks=[SaveLockedCheckpoints()],
        )
        trainer.train()
        checkpoints = {}
        for step in sorted(checkpoint_steps):
            adapter = out / "checkpoints" / f"step_{step}" / "adapter"
            if not adapter.is_dir():
                raise ContractError(f"trainer did not save checkpoint step {step}")
            checkpoints[str(step)] = {
                "path": adapter.relative_to(output_path(".")).as_posix(),
                "sha256": sha256_directory(adapter),
            }
        metadata.update({
            "status": "completed",
            "rows": len(selected_rows),
            "completed_steps": int(trainer.state.global_step),
            "final_loss": getattr(trainer, "paper_c_last_loss", None),
            "checkpoints": checkpoints,
            "decision_tokens": decision,
            "step_zero": preflight,
            "dropout_modules_zeroed": dropout_modules,
            "runtime": _runtime_record(run_device),
        })
        if run_device == "cuda":
            metadata["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    except Exception as exc:
        metadata.update({
            "status": "failed",
            "failure": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    metadata["wall_time_seconds"] = round(time.time() - started, 3)
    metadata["completed_utc"] = _utcnow()
    write_json(metadata_path, metadata)
    return metadata
