"""Explicit reference, development, and locked retrospective scoring paths."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math
import time
import traceback

from .contracts import (
    ContractError,
    normalize_gold,
    output_path,
    sha256_directory,
    sha256_file,
    sha256_ordered,
    write_json,
    write_jsonl,
)
from .objectives import signed_margin, two_verdict_probability_unsafe
from .runtime import load_base_model, load_tokenizer, software_versions, validate_prompt_cache


SCORE_MODES = ("reference", "stage2_dev", "retrospective")


def _device(requested: str | None, allow_cpu: bool) -> str:
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
        raise ContractError("scoring requires CUDA unless --allow-cpu is explicitly set")
    return device


def score_adapter(
    *,
    config: dict,
    mode: str,
    model_key: str,
    manifest_path: str | Path,
    prompt_cache_path: str | Path,
    out_path: str | Path,
    condition: str,
    adapter_path: str | Path | None,
    partition_ids: set[str] | None = None,
    batch_size: int = 4,
    device: str | None = None,
    allow_cpu: bool = False,
) -> dict:
    if mode not in SCORE_MODES:
        raise ContractError(f"unknown score mode: {mode}")
    if model_key not in config["models"]:
        raise ContractError(f"unknown model key: {model_key}")
    if int(batch_size) <= 0:
        raise ContractError("batch size must be positive")
    prompt_rows, prompt_metadata = validate_prompt_cache(
        config=config, model_key=model_key, manifest_path=manifest_path,
        cache_path=prompt_cache_path,
    )
    if partition_ids is not None:
        prompt_rows = [row for row in prompt_rows if row["sample_id"] in partition_ids]
        if {row["sample_id"] for row in prompt_rows} != partition_ids:
            raise ContractError("requested scoring partition is not present in prompt cache")
    if not prompt_rows:
        raise ContractError("scoring set is empty")
    output = output_path(out_path)
    metadata_path = output_path(f"{out_path}.metadata.json")
    if output.exists() or metadata_path.exists():
        raise ContractError(f"refusing to overwrite score artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    adapter = output_path(adapter_path) if adapter_path is not None else None
    if adapter is not None and not adapter.is_dir():
        raise ContractError(f"adapter does not exist: {adapter}")
    metadata = {
        "schema_version": 1,
        "kind": "paper_c_text_free_scores",
        "study_id": config["study_id"],
        "mode": mode,
        "condition": condition,
        "model_key": model_key,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "manifest_sha256": sha256_file(manifest_path),
        "prompt_cache_sha256": sha256_file(output_path(prompt_cache_path)),
        "prompt_cache_metadata": prompt_metadata,
        "adapter_path": adapter.relative_to(output_path(".")).as_posix() if adapter else None,
        "adapter_sha256": sha256_directory(adapter) if adapter else None,
    }
    started = time.time()
    try:
        import torch
        from peft import PeftModel

        run_device = _device(device, allow_cpu)
        tokenizer, decision = load_tokenizer(config, model_key)
        base = load_base_model(config, model_key, device=run_device)
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False) if adapter else base
        model.to(run_device)
        model.eval()
        rows = []
        with torch.no_grad():
            for offset in range(0, len(prompt_rows), int(batch_size)):
                batch = prompt_rows[offset:offset + int(batch_size)]
                width = max(len(row["input_ids"]) for row in batch)
                input_ids, attention = [], []
                for row in batch:
                    gap = width - len(row["input_ids"])
                    input_ids.append(row["input_ids"] + [tokenizer.pad_token_id] * gap)
                    attention.append([1] * len(row["input_ids"]) + [0] * gap)
                ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=run_device)
                mask_tensor = torch.tensor(attention, dtype=torch.long, device=run_device)
                logits = model(input_ids=ids_tensor, attention_mask=mask_tensor).logits
                last = mask_tensor.sum(1) - 1
                indexes = torch.arange(last.shape[0], device=last.device)
                next_logits = logits[indexes, last].float()
                safe = next_logits[:, decision["safe_id"]].cpu().tolist()
                unsafe = next_logits[:, decision["unsafe_id"]].cpu().tolist()
                for source_row, safe_logit, unsafe_logit in zip(batch, safe, unsafe, strict=True):
                    gold = normalize_gold(source_row["gold"])
                    margin = signed_margin(safe_logit, unsafe_logit, gold)
                    record = {
                        "sample_id": source_row["sample_id"],
                        "source": source_row["source"],
                        "family_id": source_row["family_id"],
                        "content_sha256": source_row["content_sha256"],
                        "prompt_sha256": source_row["prompt_sha256"],
                        "gold": gold,
                        "model_key": model_key,
                        "condition": condition,
                        "safe_logit": float(safe_logit),
                        "unsafe_logit": float(unsafe_logit),
                        "score_unsafe_minus_safe": float(unsafe_logit) - float(safe_logit),
                        "probability_unsafe_two_verdict": two_verdict_probability_unsafe(
                            safe_logit, unsafe_logit
                        ),
                        "signed_margin": margin,
                    }
                    if mode == "reference":
                        record["reference_signed_margin"] = margin
                    rows.append(record)
        if any(not math.isfinite(row["signed_margin"]) for row in rows):
            raise ContractError("score artifact contains a non-finite margin")
        write_jsonl(output, rows)
        metadata.update({
            "status": "completed",
            "rows": len(rows),
            "decision_tokens": decision,
            "ordered_identity_sha256": sha256_ordered({
                "sample_id": row["sample_id"],
                "prompt_sha256": row["prompt_sha256"],
            } for row in rows),
            "score_sha256": sha256_file(output),
            "software": software_versions(),
            "device": run_device,
        })
        if run_device == "cuda":
            metadata.update({
                "device_name": torch.cuda.get_device_name(0),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            })
    except Exception as exc:
        metadata.update({
            "status": "failed",
            "failure": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    metadata["wall_time_seconds"] = round(time.time() - started, 3)
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(metadata_path, metadata)
    return metadata

