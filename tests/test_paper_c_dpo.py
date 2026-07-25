from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import analyze_paper_c_dpo as Analyze  # noqa: E402
import paper_c_dpo_common as P  # noqa: E402


def _config() -> dict:
    return P.read_json(REPO / "configs" / "paper_c_dpo_v2.json")


def _lock(models=("model_a",), seeds=(42,)) -> dict:
    config = _config()
    config["models"] = list(models)
    config["seeds"] = list(seeds)
    return {
        "models": {model: {"model_id": model} for model in models},
        "seeds": list(seeds),
        "config": {"value": config},
    }


def _score_record(*, model: str, condition: str, seed: int, split: str,
                  sample_id: str, source: str, gold: int, score: float,
                  checkpoint=None, stage=None, objective=None, sampler=None) -> dict:
    if condition == "base":
        stage = "base"
        adapter = None
    elif condition == "stage1_sft":
        stage = "stage1"
        adapter = f"adapter::{model}::{seed}::stage1"
    else:
        stage = stage or "stage2"
        adapter = f"adapter::{model}::{seed}::{condition}::{checkpoint}"
    probability = 1.0 / (1.0 + math.exp(-score))
    return {
        "sample_id": sample_id,
        "content_sha256": f"content::{sample_id}",
        "source": source,
        "split": split,
        "gold": gold,
        "family_id": f"family::{split}::{sample_id}",
        "model_key": model,
        "condition": condition,
        "seed": seed,
        "stage": stage,
        "objective": objective,
        "sampler": sampler,
        "checkpoint_step": checkpoint,
        "adapter_sha256": adapter,
        "safe_logit": -score / 2.0,
        "unsafe_logit": score / 2.0,
        "score_raw": score,
        "probability_raw": probability,
        "probability_calibrated": probability,
    }


def test_scalar_loss_reduction_and_initialization_identity():
    policy_margin = 1.3
    reference_margin = -0.2
    beta = 0.4
    assert P.pair_ce_loss(policy_margin, beta) == pytest.approx(
        -math.log(1.0 / (1.0 + math.exp(-beta * policy_margin))))
    assert P.dpo_loss(policy_margin, reference_margin, beta) == pytest.approx(
        P.dpo_logratio_loss(policy_margin, 0.0, reference_margin, 0.0, beta))
    assert P.dpo_loss(reference_margin, reference_margin, beta) == pytest.approx(math.log(2.0))

    epsilon = 1e-6
    derivative = (
        P.dpo_loss(policy_margin + epsilon, reference_margin, beta)
        - P.dpo_loss(policy_margin - epsilon, reference_margin, beta)
    ) / (2 * epsilon)
    assert derivative < 0
    assert P.dpo_loss(policy_margin, reference_margin, 0.8) < P.dpo_loss(
        policy_margin, reference_margin, 0.2)


def test_config_and_condition_grid_are_exact():
    config = _config()
    P.validate_config(config)
    assert [row["condition"] for row in P.condition_grid()] == [
        "verdict_ce__uncertain", "pair_ce__uncertain", "dpo__uncertain",
        "verdict_ce__matched_random", "pair_ce__matched_random",
        "dpo__matched_random",
    ]
    broken = {**config, "stage2": {**config["stage2"], "objectives": ["dpo"]}}
    with pytest.raises(P.PaperCContractError):
        P.validate_config(broken)


def test_family_partition_is_deterministic_and_never_splits_family():
    rows = []
    for source in ("source_a", "source_b"):
        for gold in (0, 1):
            for family_number in range(4):
                family = f"{source}-{gold}-family-{family_number}"
                for member in range(2):
                    sample = f"{family}-row-{member}"
                    rows.append({
                        "sample_id": sample,
                        "source": source,
                        "label": gold,
                        "family_id": family,
                        "content_sha256": f"sha::{sample}",
                    })
    first = P.family_partition(rows, development_fraction=0.25, seed=7)
    second = P.family_partition(rows, development_fraction=0.25, seed=7)
    assert first == second
    by_family = {}
    for row in first:
        by_family.setdefault(row["family_id"], set()).add(row["stage2_partition"])
    assert all(len(partitions) == 1 for partitions in by_family.values())
    assert {row["stage2_partition"] for row in first} == set(P.PARTITIONS)


def test_family_crossing_sources_is_assigned_globally():
    rows = []
    for source in ("s1", "s2"):
        for gold in (0, 1):
            for index in range(5):
                family = "shared" if gold == 1 and index == 0 else f"{source}-{gold}-{index}"
                sample = f"{source}-{gold}-{index}"
                rows.append({"sample_id": sample, "source": source, "gold": gold,
                             "family_id": family, "content_sha256": sample})
    partition = P.family_partition(rows, development_fraction=0.2, seed=1)
    shared = {row["stage2_partition"] for row in partition
              if row["family_id"] == "shared"}
    assert len(shared) == 1


def test_uncertain_and_random_selections_are_disjoint_and_matched():
    partition = []
    reference = []
    for source in ("a", "b"):
        for gold in (0, 1):
            for index in range(8):
                sample = f"{source}-{gold}-{index}"
                partition.append({
                    "sample_id": sample,
                    "content_sha256": f"sha::{sample}",
                    "family_id": f"family::{sample}",
                    "source": source,
                    "gold": gold,
                    "stage2_partition": "stage2_update",
                })
                signed_margin = 0.05 + index
                raw_score = signed_margin if gold == 1 else -signed_margin
                reference.append({
                    "sample_id": sample,
                    "safe_logit": -raw_score / 2,
                    "unsafe_logit": raw_score / 2,
                })
    selected = P.build_selections(
        partition, reference, uncertain_fraction=0.25, seed=99)
    assert selected == P.build_selections(
        partition, reference, uncertain_fraction=0.25, seed=99)
    uncertain = {row["sample_id"] for row in selected
                 if row["selection_role"] == "uncertain"}
    random = {row["sample_id"] for row in selected
              if row["selection_role"] == "matched_random"}
    assert uncertain.isdisjoint(random)
    assert len(uncertain) == len(random) == 8
    P.validate_selections(selected)


def _dev_frame() -> pd.DataFrame:
    rows = []
    examples = [
        ("n1", 0, -2.0), ("p1", 1, 2.0),
        ("n2", 0, -1.0), ("p2", 1, 1.0),
    ]
    for sample, gold, score in examples:
        rows.append(_score_record(
            model="model_a", condition="stage1_sft", seed=42,
            split="stage2_dev", sample_id=sample, source="source", gold=gold,
            score=score,
        ))
    for cell in P.condition_grid():
        for step in (25, 50, 100, 200):
            for index, (sample, gold, _) in enumerate(examples):
                score = ([0.4, 0.3, 0.2, 0.1][index] if step == 25
                         else (-2.0 if gold == 0 else 2.0))
                rows.append(_score_record(
                    model="model_a", condition=cell["condition"], seed=42,
                    split="stage2_dev", sample_id=sample, source="source", gold=gold,
                    score=score, checkpoint=step, objective=cell["objective"],
                    sampler=cell["sampler"],
                ))
    return pd.DataFrame(rows)


def test_checkpoint_selection_uses_dev_and_chooses_earliest_feasible():
    frame = _dev_frame()
    records = Analyze.select_checkpoints(frame, _lock())
    assert len(records) == 6
    assert all(record["target_feasible"] for record in records)
    assert {record["selected_checkpoint_step"] for record in records} == {50}
    wrong_split = frame.copy()
    wrong_split["split"] = "id_test"
    with pytest.raises(P.PaperCContractError, match="stage2_dev"):
        Analyze.select_checkpoints(wrong_split, _lock())


def _retrospective_frame(models=("model_a",), seeds=(42,)) -> pd.DataFrame:
    rows = []
    examples = []
    for split in ("id_test", "transfer_test"):
        for source in ("source_a", "source_b"):
            for index, gold in enumerate((0, 1, 0, 1, 0, 1)):
                examples.append((split, source, f"{split}-{source}-{index}", gold))
    for model in models:
        for split, source, sample, gold in examples:
            rows.append(_score_record(
                model=model, condition="base", seed=-1, split=split,
                sample_id=sample, source=source, gold=gold,
                score=(-0.1 if gold == 0 else 0.1),
            ))
        for seed in seeds:
            for split, source, sample, gold in examples:
                rows.append(_score_record(
                    model=model, condition="stage1_sft", seed=seed, split=split,
                    sample_id=sample, source=source, gold=gold,
                    score=(-0.2 if gold == 0 else 0.2),
                ))
            for cell in P.condition_grid():
                for index, (split, source, sample, gold) in enumerate(examples):
                    if cell["objective"] == "verdict_ce":
                        pattern = (0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
                    elif cell["objective"] == "pair_ce":
                        pattern = (-0.2, 0.4, 0.1, 0.3, 0.2, 0.25)
                    else:
                        pattern = (-0.6, 0.6, -0.4, 0.5, -0.2, 0.4)
                    local_index = int(sample.rsplit("-", 1)[1])
                    score = pattern[local_index]
                    if cell["sampler"] == "matched_random" and cell["objective"] == "dpo":
                        score = (-0.3 if gold == 0 else 0.3)
                    rows.append(_score_record(
                        model=model, condition=cell["condition"], seed=seed,
                        split=split, sample_id=sample, source=source, gold=gold,
                        score=score, checkpoint=50, objective=cell["objective"],
                        sampler=cell["sampler"],
                    ))
    return pd.DataFrame(rows)


def test_retrospective_grid_and_known_contrasts():
    frame = _retrospective_frame()
    lock = _lock()
    Analyze.validate_retrospective_grid(frame, lock)
    cells = Analyze.cell_metrics(frame)
    contrasts = Analyze.contrast_table(cells)
    assert (contrasts[contrasts.contrast == "C_pair"]["estimate"] > 0).all()
    assert (contrasts[contrasts.contrast == "C_ref"]["estimate"] >= 0).all()
    assert set(contrasts["contrast"]) == {
        "C_pair", "C_ref", "C_total", "C_selection_interaction",
    }
    missing = frame[~(
        (frame.condition == "dpo__uncertain") & (frame.seed == 42)
    )]
    with pytest.raises(P.PaperCContractError, match="grid mismatch"):
        Analyze.validate_retrospective_grid(missing, lock)


def test_hierarchical_bootstrap_is_deterministic():
    frame = _retrospective_frame(seeds=(42, 43))
    lock = _lock(seeds=(42, 43))
    first = Analyze.hierarchical_bootstrap(frame, lock, reps=20, seed=123)
    second = Analyze.hierarchical_bootstrap(frame, lock, reps=20, seed=123)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["interval_method"]) == {
        "paired_poisson_family_and_within_model_seed_bootstrap"
    }
    assert np.isfinite(first[["estimate", "ci95_low", "ci95_high"]].to_numpy()).all()


def test_hashes_bind_order_and_values():
    assert P.sha256_ordered(["a", "b"]) != P.sha256_ordered(["b", "a"])
    assert P.canonical_sha256({"a": 1, "b": 2}) == P.canonical_sha256({"b": 2, "a": 1})
