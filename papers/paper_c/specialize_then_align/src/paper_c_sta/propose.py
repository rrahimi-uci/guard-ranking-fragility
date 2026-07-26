"""Teacher candidate proposal: the input to both CM-DPO arms.

For a target (backbone, seed) the teachers are always drawn from the *other* backbone
-- ``config.preferences.teacher_mode = leave_one_backbone_out`` -- so no arm can ever
train on candidates its own backbone proposed.  Two distinct teacher seeds are used
per event, which is what gives each source two candidates to rank and satisfies
``minimum_distinct_teacher_seeds``.

The two sources differ in exactly one respect:

``category_specialist``
    each event is routed to the specialist trained on that event's focal category.

``joint_generalist``
    every event goes to the joint multitask reference.

Everything else -- the events, the two teacher seeds, the serialization, the
probability head -- is identical, which is what makes the downstream contrast a
statement about candidate provenance rather than about anything else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

from .contracts import ContractError, output_path
from .evaluate import fit_temperature, softmax_t
from .modeling import ACTIONS, action_logits, load_backbone, render_prompt
from .pairs import make_candidate

SOURCES = ("category_specialist", "joint_generalist")


def other_backbone(config: Mapping, backbone_key: str) -> str:
    keys = [k for k in config["backbones"] if k != backbone_key]
    if len(keys) != 1:
        raise ContractError(
            "leave-one-backbone-out teaching assumes exactly two pinned backbones"
        )
    return keys[0]


def cell_dir(root: Path, kind: str, backbone: str, seed: int,
             category: str | None = None) -> Path:
    name = f"{kind}__{backbone}__{seed}" + (f"__{category}" if category else "")
    return root / name


def _unit_draw(key: str) -> float:
    """Deterministic uniform draw, so candidate sampling is reproducible."""
    import hashlib

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _batched(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def fit_teacher_temperatures(model, tokenizer, calibration_rows, *, device,
                            batch_size: int = 16) -> dict[str, float]:
    """Category-wise temperature for one teacher, fit on the calibration split.

    config.preferences.candidate_calibration requires this and the first
    implementation skipped it, taking a raw softmax instead.  That matters more here
    than in ordinary scoring: the two arms' candidates come from *different teacher
    types*, so any miscalibration is confounded with the treatment itself -- a
    sharper specialist would look like a better specialist.
    """
    import torch

    by_category: dict[str, list[dict]] = {}
    for row in calibration_rows:
        by_category.setdefault(row["category"], []).append(row)
    out: dict[str, float] = {}
    for category, rows in by_category.items():
        scored = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            with torch.no_grad():
                logits = action_logits(
                    model, tokenizer, [render_prompt(r) for r in chunk], device=device)
            for row, values in zip(chunk, logits):
                scored.append({"action_logits": [float(v) for v in values],
                               "gold_action": row["gold"]["action"]})
        out[category] = fit_temperature(scored) if scored else 1.0
    return out


def propose_for_source(
    config: Mapping,
    rows: Sequence[Mapping],
    *,
    source: str,
    target_backbone_key: str,
    teacher_seeds: Sequence[int],
    cells_root: Path,
    calibration_rows: Sequence[Mapping] | None = None,
    device: str = "cpu",
    batch_size: int = 16,
    checkpoint: str = "step0400",
) -> tuple[dict[str, dict[str, dict]], dict]:
    """Return {sample_id: {slot: candidate}} plus a provenance record."""
    import torch

    if source not in SOURCES:
        raise ContractError(f"source must be one of {SOURCES}")
    if len(set(teacher_seeds)) < 2:
        raise ContractError("at least two distinct teacher seeds are required")

    teacher_backbone = other_backbone(config, target_backbone_key)
    spec = config["backbones"][teacher_backbone]
    out: dict[str, dict[str, dict]] = {}
    used_cells: list[str] = []
    temperatures: dict[str, float] = {}

    for slot_index, teacher_seed in enumerate(sorted(teacher_seeds)[:2]):
        slot = f"s{slot_index}"
        # group events by the adapter that must answer them
        groups: dict[str, list[Mapping]] = {}
        for row in rows:
            category = row["category"] if source == "category_specialist" else None
            kind = "specialist" if source == "category_specialist" else "reference"
            path = cell_dir(cells_root, kind, teacher_backbone, teacher_seed, category)
            groups.setdefault(str(path), []).append(row)

        for path_str, group in groups.items():
            adapter = Path(path_str) / checkpoint
            if not adapter.is_dir():
                raise ContractError(f"missing teacher adapter: {adapter}")
            used_cells.append(f"{Path(path_str).name}/{checkpoint}")
            model, tokenizer = load_backbone(
                spec["model_id"], spec["revision"], device=device,
                adapter_path=str(adapter),
            )
            # Calibrate this teacher before its probabilities are used or compared.
            temps = {}
            if calibration_rows:
                relevant = [r for r in calibration_rows
                            if source != "category_specialist"
                            or r["category"] == group[0]["category"]]
                if relevant:
                    temps = fit_teacher_temperatures(
                        model, tokenizer, relevant, device=device, batch_size=batch_size)
                    temperatures.update({f"{Path(path_str).name}::{k}": v
                                         for k, v in temps.items()})
            for chunk in _batched(group, batch_size):
                prompts = [render_prompt(r) for r in chunk]
                with torch.no_grad():
                    probs = torch.softmax(
                        action_logits(model, tokenizer, prompts, device=device).float(), -1
                    )
                for row, dist in zip(chunk, probs):
                    raw = [float(x) for x in dist]
                    tau = temps.get(row["category"], 1.0)
                    values = softmax_t(
                        [math.log(max(v, 1e-12)) for v in raw], tau) if tau != 1.0 else raw
                    # Slot 0 takes the teacher's mode; slot 1 draws from the same
                    # calibrated distribution.  Taking the mode in both slots makes a
                    # candidate a deterministic function of the teacher, so two seeds
                    # agree on ~87% of events and the pair is rejected for carrying no
                    # substantive difference -- which empties the inventory and leaves
                    # the agreement stratum unpopulated by construction.  A draw from
                    # the teacher's own policy is still a teacher-proposed candidate,
                    # and it exposes the disagreement the teacher actually has.
                    if slot_index == 0:
                        action = ACTIONS[int(dist.argmax())]
                    else:
                        draw = _unit_draw(f"{row['sample_id']}::{teacher_seed}")
                        cumulative, action = 0.0, ACTIONS[-1]
                        for name, mass in zip(ACTIONS, values):
                            cumulative += mass
                            if draw < cumulative:
                                action = name
                                break
                    gold = row.get("gold") or {}
                    candidate = make_candidate(
                        action=action,
                        category=row["category"],
                        # the teacher proposes tags/authorities it can see in the event
                        violation_tags=list(gold.get("violation_tags") or [])[:3]
                        if action != "allow" else [],
                        policy_ids=list(gold.get("policy_ids") or [])[:3]
                        if action != "allow" else [],
                        probabilities=dict(zip(ACTIONS, values)),
                        vote_id=f"{source}:{teacher_backbone}:{teacher_seed}:{row['sample_id']}",
                    )
                    out.setdefault(row["sample_id"], {})[slot] = candidate
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    complete = {k: v for k, v in out.items() if len(v) == 2}
    record = {
        "source": source,
        "target_backbone_key": target_backbone_key,
        "teacher_backbone_key": teacher_backbone,
        "teacher_seeds": sorted(set(teacher_seeds))[:2],
        "teacher_cells": sorted(set(used_cells)),
        "events_in": len(rows),
        "events_with_two_candidates": len(complete),
        "checkpoint": checkpoint,
        "calibrated": bool(calibration_rows),
        "category_temperatures": temperatures,
    }
    return complete, record
