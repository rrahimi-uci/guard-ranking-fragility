"""Training cells: the joint reference, the category specialists, and the five arms.

One dispatcher, ``run_cell``, produces every one of the 110 cells in the program.
Which cell it is comes entirely from ``kind`` and the config, so the arms cannot
drift apart in optimizer, ladder, or accounting -- the property the matched-source
comparison manifest later machine-checks.

The three loss families:

``sft``
    Completion-only cross-entropy on the structured verdict.  Used for the joint
    multitask reference, for each category specialist, and for the ``gold_sft`` arm.

``pairs``
    ``cross_pairce`` or ``cm_dpo`` via :mod:`objectives`, over preference pairs whose
    log-probabilities are the masked response sums from :mod:`modeling`.  A pair with
    a truncated candidate is dropped before the optimizer sees it, per
    ``config.alignment.candidate_length_rule``.

``distill``
    Forward KL from the teacher's three calibrated action probabilities to the
    student's, and nothing else -- not tags, not policy ids, not rationales.

The alignment arms trained on pairs optimise a composite: a soft worst-category term
over the five focal categories, plus a gold anchor and a general-replay KL to the
frozen reference.  SFT-family cells optimise plain cross-entropy by definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import random
import time

from .contracts import ContractError, canonical_sha256, output_path, write_json
from .modeling import (
    ACTIONS,
    action_logits,
    batch_response_logprobs,
    load_backbone,
    render_prompt,
    render_response,
)
from .objectives import torch_composite_alignment_loss, torch_pair_loss

KINDS = ("reference", "specialist", "gold_sft", "soft_distill",
         "specialist_pairce", "generalist_cm_dpo", "specialist_cm_dpo")
ARM_LOSS = {
    "gold_sft": "sft",
    "soft_distill": "distill",
    "specialist_pairce": "cross_pairce",
    "generalist_cm_dpo": "cm_dpo",
    "specialist_cm_dpo": "cm_dpo",
}


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _lora(config: Mapping, section: str):
    from peft import LoraConfig

    spec = config[section]["lora"]
    return LoraConfig(
        r=spec["r"], lora_alpha=spec["alpha"], lora_dropout=spec["dropout"],
        bias="none", task_type="CAUSAL_LM",
        target_modules=list(spec.get("target_modules")
                            or ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]),
    )


def _sft_batch_loss(model, tokenizer, rows: Sequence[Mapping], *, device, max_length: int):
    """Completion-only cross-entropy: prompt tokens are masked to -100."""
    import torch

    input_ids, labels = [], []
    for row in rows:
        prompt = tokenizer(render_prompt(row), add_special_tokens=False)["input_ids"]
        response = tokenizer(render_response(row), add_special_tokens=False)["input_ids"]
        ids = (prompt + response)[-max_length:]
        n_response = min(len(response), len(ids))
        label = [-100] * (len(ids) - n_response) + ids[len(ids) - n_response:]
        input_ids.append(ids)
        labels.append(label)
    width = max(len(x) for x in input_ids)
    pad = tokenizer.pad_token_id
    padded = torch.tensor([[pad] * (width - len(x)) + x for x in input_ids], device=device)
    padded_labels = torch.tensor(
        [[-100] * (width - len(x)) + x for x in labels], device=device
    )
    attention = (padded != pad).long()
    out = model(input_ids=padded, attention_mask=attention, labels=padded_labels)
    return out.loss


def _distill_batch_loss(model, tokenizer, rows: Sequence[Mapping], *, device):
    """Forward KL from teacher action probabilities to the student's."""
    import torch

    targets = []
    for row in rows:
        teacher = row.get("teacher_action_probabilities")
        if not isinstance(teacher, Mapping) or set(teacher) != set(ACTIONS):
            raise ContractError(
                "soft_distill requires teacher_action_probabilities over exactly the "
                "three actions; tags, policy ids and rationales are never transferred"
            )
        targets.append([float(teacher[a]) for a in ACTIONS])
    target = torch.tensor(targets, device=device, dtype=torch.float32)
    target = target / target.sum(-1, keepdim=True)
    logits = action_logits(model, tokenizer, [render_prompt(r) for r in rows], device=device)
    student = torch.log_softmax(logits.float(), dim=-1)
    return torch.nn.functional.kl_div(student, target, reduction="batchmean")


def _pair_batch_loss(model, reference, tokenizer, rows: Sequence[Mapping], *, arm,
                     device, beta: float, max_length: int):
    """PairCE or CM-DPO over preference pairs, dropping truncated candidates."""
    import torch

    prompts = [render_prompt(r) for r in rows]
    chosen = [(p, r["chosen"]) for p, r in zip(prompts, rows)]
    rejected = [(p, r["rejected"]) for p, r in zip(prompts, rows)]
    chosen_lp, chosen_trunc = batch_response_logprobs(
        model, tokenizer, chosen, device=device, max_length=max_length)
    rejected_lp, rejected_trunc = batch_response_logprobs(
        model, tokenizer, rejected, device=device, max_length=max_length)
    keep = [i for i, (a, b) in enumerate(zip(chosen_trunc, rejected_trunc)) if not (a or b)]
    if not keep:
        raise ContractError("every pair in the batch had a truncated candidate")
    index = torch.tensor(keep, device=device)
    kwargs = {
        "arm": arm,
        "chosen_policy_logps": chosen_lp[index],
        "rejected_policy_logps": rejected_lp[index],
        "beta": beta,
    }
    if arm == "cm_dpo":
        with torch.no_grad():
            ref_chosen, _ = batch_response_logprobs(
                reference, tokenizer, chosen, device=device, max_length=max_length)
            ref_rejected, _ = batch_response_logprobs(
                reference, tokenizer, rejected, device=device, max_length=max_length)
        kwargs["chosen_reference_logps"] = ref_chosen[index]
        kwargs["rejected_reference_logps"] = ref_rejected[index]
    mean = torch_pair_loss(**kwargs)
    # per-example values for the worst-category term: recompute elementwise from the
    # same margins the mean was built from, so the two can never disagree.
    policy_margin = chosen_lp[index] - rejected_lp[index]
    if arm == "cm_dpo":
        policy_margin = policy_margin - (
            kwargs["chosen_reference_logps"] - kwargs["rejected_reference_logps"])
    per_example = torch.nn.functional.softplus(-beta * policy_margin)
    kept_rows = [rows[i] for i in keep]
    return mean, per_example, kept_rows, len(rows) - len(keep)


def _category_losses(rows: Sequence[Mapping], per_example, categories: Sequence[str]):
    """Group per-example losses by focal category, mean within each.

    Categories absent from a batch are given the batch mean rather than dropped, so
    the soft worst-category term always ranges over the full expected set and a
    category cannot be sacrificed simply by being unlucky in the sampler.
    """
    import torch

    buckets: dict[str, list] = {}
    for row, value in zip(rows, per_example):
        buckets.setdefault(row["category"], []).append(value)
    batch_mean = torch.stack(list(per_example)).mean()
    return {
        name: (torch.stack(buckets[name]).mean() if name in buckets else batch_mean)
        for name in categories
    }


def _retention_kl(model, reference, tokenizer, rows, *, device):
    """Forward KL from the frozen reference's action head to the policy's.

    This is the general-replay term: it keeps the student from forgetting the joint
    reference's behaviour while it chases preference signal.
    """
    import torch

    prompts = [render_prompt(r) for r in rows]
    with torch.no_grad():
        ref = torch.softmax(
            action_logits(reference, tokenizer, prompts, device=device).float(), -1)
    student = torch.log_softmax(
        action_logits(model, tokenizer, prompts, device=device).float(), -1)
    return torch.nn.functional.kl_div(student, ref, reduction="batchmean")


def run_cell(
    config: Mapping,
    *,
    kind: str,
    backbone_key: str,
    seed: int,
    rows: Sequence[Mapping],
    out_dir: str,
    category: str | None = None,
    reference_adapter: str | None = None,
    device: str = "cpu",
    max_steps: int | None = None,
    batch_size: int = 4,
    max_length: int = 1024,
    log_every: int = 10,
) -> dict:
    """Train one cell and save the checkpoint ladder. Returns the run record."""
    import torch
    from peft import get_peft_model

    if kind not in KINDS:
        raise ContractError(f"unknown cell kind: {kind}")
    if kind == "specialist" and not category:
        raise ContractError("a specialist cell requires its focal category")
    if not rows:
        raise ContractError("a training cell requires at least one row")

    backbone = config["backbones"][backbone_key]
    section = "specialists" if kind in ("reference", "specialist") else "alignment"
    settings = config[section]
    steps = int(max_steps if max_steps is not None else settings["max_steps"])
    ladder = [s for s in config["alignment"]["checkpoint_steps"] if s <= steps] \
        if section == "alignment" else [steps]
    if steps not in ladder:
        ladder.append(steps)

    _seed_everything(seed)
    model, tokenizer = load_backbone(
        backbone["model_id"], backbone["revision"], device=device,
        adapter_path=reference_adapter, trainable=bool(reference_adapter),
    )
    if reference_adapter is None:
        model = get_peft_model(model, _lora(config, section))
    model.train()

    loss_kind = "sft" if kind in ("reference", "specialist") else ARM_LOSS[kind]
    reference = None
    if loss_kind in ("cm_dpo", "cross_pairce"):
        if not reference_adapter:
            raise ContractError(
                "alignment arms require the frozen joint reference adapter: cm_dpo "
                "for its centering term, and every pair arm for the general-replay KL"
            )
        reference, _ = load_backbone(
            backbone["model_id"], backbone["revision"], device=device,
            adapter_path=reference_adapter, trainable=False,
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(settings["learning_rate"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))

    target = output_path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    beta = float(config["preferences"]["beta"])
    core_categories = list(config["core_categories"])
    # The composite applies to the alignment arms trained on pairs.  SFT-family cells
    # (reference, specialist, gold_sft) optimise plain cross-entropy by definition.
    use_composite = loss_kind in ("cross_pairce", "cm_dpo") and reference is not None
    history, dropped_total, cursor = [], 0, 0
    started = time.time()

    for step in range(1, steps + 1):
        batch = []
        while len(batch) < batch_size:
            if cursor >= len(order):
                cursor = 0
                random.Random(seed + step).shuffle(order)
            batch.append(rows[order[cursor]])
            cursor += 1
        parts = None
        if loss_kind == "sft":
            loss = _sft_batch_loss(model, tokenizer, batch, device=device,
                                   max_length=max_length)
        elif loss_kind == "distill":
            loss = _distill_batch_loss(model, tokenizer, batch, device=device)
        else:
            loss, per_example, kept, dropped = _pair_batch_loss(
                model, reference, tokenizer, batch, arm=loss_kind, device=device,
                beta=beta, max_length=max_length)
            dropped_total += dropped
            if use_composite:
                # The specified objective is NOT the bare pair loss: it is a soft
                # worst-category term over the five focal categories, plus a gold
                # anchor and a general-replay KL to the frozen reference.  Without
                # this the thinnest category can be traded away for aggregate gain,
                # which is exactly what the design forbids.
                cat_losses = _category_losses(kept, per_example, core_categories)
                gold_anchor = _sft_batch_loss(model, tokenizer, kept, device=device,
                                              max_length=max_length)
                retention = _retention_kl(model, reference, tokenizer, kept, device=device)
                parts = torch_composite_alignment_loss(
                    {k: v for k, v in cat_losses.items()},
                    gold_anchor_loss=gold_anchor,
                    retention_kl=retention,
                    temperature=float(config["alignment"]["category_dro_temperature"]),
                    lambda_gold=float(config["alignment"]["lambda_gold"]),
                    lambda_retain=float(config["alignment"]["lambda_retain"]),
                    expected_categories=core_categories,
                )
                loss = parts["total"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % log_every == 0 or step == 1:
            entry = {"step": step, "loss": float(loss.detach())}
            if parts is not None:
                entry.update({k: float(v.detach()) if hasattr(v, "detach") else float(v)
                          for k, v in parts.items() if k != "total"})
            history.append(entry)
            print(f"    step {step:4d}/{steps}  loss={float(loss.detach()):.4f}", flush=True)
        if step in ladder:
            model.save_pretrained(str(target / f"step{step:04d}"))

    record = {
        "kind": kind,
        "backbone_key": backbone_key,
        "backbone_revision": backbone["revision"],
        "seed": seed,
        "category": category,
        "loss_kind": loss_kind,
        "steps": steps,
        "checkpoint_ladder": sorted(ladder),
        "rows": len(rows),
        "batch_size": batch_size,
        "learning_rate": float(settings["learning_rate"]),
        "beta": beta if loss_kind in ("cross_pairce", "cm_dpo") else None,
        "pairs_dropped_truncated": dropped_total,
        "composite_objective": bool(use_composite),
        "category_dro_temperature": config["alignment"]["category_dro_temperature"],
        "lambda_gold": config["alignment"]["lambda_gold"],
        "lambda_retain": config["alignment"]["lambda_retain"],
        "reference_adapter": reference_adapter,
        "wall_seconds": round(time.time() - started, 1),
        "loss_history": history,
        "config_sha256": canonical_sha256(config),
        "device": device,
    }
    write_json(target / "run_record.json", record)
    return record
