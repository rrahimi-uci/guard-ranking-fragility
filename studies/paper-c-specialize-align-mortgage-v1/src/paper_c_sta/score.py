"""Score a trained cell over an evaluation split into rows the analysis can consume.

The output row is deliberately minimal -- category, family, gold action, the three
action logits -- because that is exactly what :mod:`evaluate` needs to fit a
temperature and thresholds, and what :mod:`analysis` needs to form a paired
family contrast.  Nothing here applies a threshold or picks a checkpoint: scoring
must not know which operating point will later be chosen, or the split discipline
would be circular.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from .contracts import ContractError, output_path
from .modeling import ACTIONS, action_logits, load_backbone, render_prompt


def score_rows(config: Mapping, rows: Sequence[Mapping], *, backbone_key: str,
               adapter_path: str, device: str = "cpu", batch_size: int = 16,
               max_length: int = 1024) -> list[dict]:
    """Run one adapter over rows and return scoring records."""
    import torch

    if not rows:
        raise ContractError("scoring requires rows")
    spec = config["backbones"][backbone_key]
    model, tokenizer = load_backbone(
        spec["model_id"], spec["revision"], device=device, adapter_path=adapter_path
    )
    out: list[dict] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prompts = [render_prompt(r) for r in chunk]
        with torch.no_grad():
            logits = action_logits(model, tokenizer, prompts, device=device,
                                   max_length=max_length)
        for row, values in zip(chunk, logits):
            out.append({
                "sample_id": row["sample_id"],
                "family_id": row["family_id"],
                "content_family_id": row["content_family_id"],
                "category": row["category"],
                "split": row["split"],
                "gold_action": row["gold"]["action"],
                "action_logits": [float(v) for v in values],
            })
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def write_scores(records: Sequence[Mapping], path: str) -> dict:
    target = output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    by_split: dict[str, int] = {}
    for record in records:
        by_split[record["split"]] = by_split.get(record["split"], 0) + 1
    return {"path": str(target), "rows": len(records), "by_split": by_split}


def load_scores(path: str) -> list[dict]:
    target = output_path(path)
    if not target.is_file():
        raise ContractError(f"missing score file: {target}")
    return [json.loads(line) for line in target.open() if line.strip()]


def predictions(rows: Sequence[Mapping], *, temperature: float, t_intervene: float,
                t_review: float) -> list[dict]:
    """Attach a decided action, for the analysis layer's paired contrast."""
    from .evaluate import decide, softmax_t

    out = []
    for row in rows:
        probs = softmax_t(row["action_logits"], temperature)
        out.append({
            **row,
            "probabilities": probs,
            "predicted": decide(probs, t_intervene=t_intervene, t_review=t_review),
        })
    return out
