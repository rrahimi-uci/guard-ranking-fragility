#!/usr/bin/env python
"""Offline checkpoint selection and retrospective analysis for Paper C v2.

The two commands intentionally consume different artifacts:

``select-checkpoints``
    consumes only scores for the family-held-out ``stage2_dev`` rows;

``retrospective``
    consumes the Paper A calibration/test/stress score artifact plus the frozen
    checkpoint-selection table.

This separation is load bearing.  The represented and transfer test rows are
never an admissible input to checkpoint selection.

Both commands are currently development-only.  Final analysis remains disabled
until candidate-adapter inventory binding and every preregistered diagnostic and
sensitivity output are implemented.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE.parent), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from guard_research.metrics import brier, log_loss_  # noqa: E402
import paper_c_dpo_common as P  # noqa: E402
import paper_a_common as A  # noqa: E402


ANALYSIS_ARTIFACT_VERSION = 2
REGIME_SPLITS = {"represented": "id_test", "transfer": "transfer_test"}
IDENTITY_COLUMNS = (
    "sample_id", "content_sha256", "source", "split", "gold", "family_id",
)
REQUIRED_COLUMNS = set(IDENTITY_COLUMNS) | {
    "model_key", "condition", "seed", "stage", "objective", "sampler",
    "checkpoint_step", "safe_logit", "unsafe_logit", "score_raw",
    "probability_raw", "probability_calibrated", "adapter_sha256",
}
CONTRASTS = {
    "C_pair": ("pair_ce", "verdict_ce"),
    "C_ref": ("dpo", "pair_ce"),
    "C_total": ("dpo", "verdict_ce"),
}


def _load_lock(path: str, *, development: bool) -> dict:
    lock = P.read_json(path)
    expected = P.canonical_sha256(
        {key: value for key, value in lock.items() if key != "lock_sha256"})
    if expected != lock.get("lock_sha256"):
        raise P.PaperCContractError("Paper C lock hash mismatch")
    if lock.get("finalization_status") != "final" and not development:
        raise P.PaperCContractError(
            "claim-bearing analysis requires a final lock; use --development for plumbing")
    P.validate_config((lock.get("config") or {}).get("value") or {})
    if lock.get("finalization_status") == "final":
        P.validate_execution_sources(lock, A.REPO_ROOT)
        software_issues = A.protocol_software_issues(
            A.software_versions(), lock.get("software_versions"))
        if software_issues:
            raise P.PaperCContractError(
                f"software environment differs from final lock: {software_issues}")
    return lock


def _read_scores(scores_path: str, metadata_path: str, lock: Mapping,
                 *, development: bool) -> tuple[pd.DataFrame, dict]:
    metadata = P.read_json(metadata_path)
    if P.sha256_file(scores_path) != metadata.get("scores_sha256"):
        raise P.PaperCContractError("score parquet hash differs from score metadata")
    if metadata.get("lock_sha256") != lock.get("lock_sha256"):
        raise P.PaperCContractError("score artifact was produced under another lock")
    if metadata.get("finalization_status") != "final" and not development:
        raise P.PaperCContractError("nonfinal score artifact requires --development")
    frame = pd.read_parquet(scores_path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise P.PaperCContractError(f"score artifact lacks required columns: {missing}")
    if frame.empty:
        raise P.PaperCContractError("score artifact is empty")
    numeric = (
        "gold", "seed", "safe_logit", "unsafe_logit", "score_raw",
        "probability_raw", "probability_calibrated",
    )
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise P.PaperCContractError(f"score column has non-finite values: {column}")
    if not frame["gold"].isin([0, 1]).all():
        raise P.PaperCContractError("gold must be binary")
    if not np.allclose(
        frame["score_raw"].to_numpy(float),
        frame["unsafe_logit"].to_numpy(float) - frame["safe_logit"].to_numpy(float),
        rtol=0, atol=1e-7,
    ):
        raise P.PaperCContractError("score_raw is not unsafe_logit-safe_logit")
    expected_probability = 1.0 / (1.0 + np.exp(-frame["score_raw"].to_numpy(float)))
    if not np.allclose(
        frame["probability_raw"].to_numpy(float), expected_probability,
        rtol=0, atol=1e-7,
    ):
        raise P.PaperCContractError("probability_raw is not sigmoid(score_raw)")
    return frame, metadata


def _bundle_key_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["model_key", "condition", "seed"]
    if frame["checkpoint_step"].notna().any():
        columns.append("checkpoint_step")
    return columns


def _identity_digest(group: pd.DataFrame) -> str:
    rows = group.loc[:, IDENTITY_COLUMNS].sort_values(
        ["split", "source", "sample_id"], kind="stable")
    return P.sha256_ordered(rows.to_dict("records"))


def _validate_common_rows(frame: pd.DataFrame) -> None:
    keys = _bundle_key_columns(frame)
    digests = {
        _identity_digest(group)
        for _, group in frame.groupby(keys, dropna=False, sort=False)
    }
    if len(digests) != 1:
        raise P.PaperCContractError(
            "score bundles do not contain the same ordered row identities")
    row_key = keys + ["split", "source", "sample_id"]
    if frame.duplicated(row_key).any():
        raise P.PaperCContractError("duplicate bundle/sample row in score artifact")


def validate_retrospective_grid(frame: pd.DataFrame, lock: Mapping) -> None:
    """Validate the exact base + Stage-1 + six selected Stage-2 bundle grid."""
    forbidden = {"train", "stage2_update", "stage2_dev"} & set(frame["split"].astype(str))
    if forbidden:
        raise P.PaperCContractError(
            f"retrospective score artifact contains selection/training rows: {sorted(forbidden)}")
    _validate_condition_semantics(frame)
    models = tuple(lock["models"])
    seeds = tuple(int(seed) for seed in lock["seeds"])
    expected = {(model, "base", -1) for model in models}
    expected |= {(model, "stage1_sft", seed) for model in models for seed in seeds}
    expected |= {
        (model, cell["condition"], seed)
        for model in models for seed in seeds for cell in P.condition_grid()
    }
    bundles = frame[["model_key", "condition", "seed"]].drop_duplicates()
    observed = {
        (str(row.model_key), str(row.condition), int(row.seed))
        for row in bundles.itertuples(index=False)
    }
    if observed != expected:
        raise P.PaperCContractError(
            f"retrospective bundle grid mismatch: missing={len(expected-observed)}, "
            f"extra={len(observed-expected)}")
    per_bundle_steps = frame.groupby(
        ["model_key", "condition", "seed"], dropna=False)["checkpoint_step"].nunique(
            dropna=False)
    if (per_bundle_steps != 1).any():
        raise P.PaperCContractError("a retrospective bundle mixes checkpoint identities")
    _validate_common_rows(frame)


def validate_dev_grid(frame: pd.DataFrame, lock: Mapping) -> None:
    """Validate Stage-1 plus every predeclared Stage-2 checkpoint on dev rows."""
    if set(frame["split"].astype(str)) != {"stage2_dev"}:
        raise P.PaperCContractError(
            "checkpoint selection accepts only the stage2_dev score artifact")
    _validate_condition_semantics(frame)
    models = tuple(lock["models"])
    seeds = tuple(int(seed) for seed in lock["seeds"])
    checkpoints = tuple(int(v) for v in lock["config"]["value"]["stage2"]["checkpoint_steps"])
    expected = {(model, "stage1_sft", seed, -1) for model in models for seed in seeds}
    expected |= {
        (model, cell["condition"], seed, checkpoint)
        for model in models for seed in seeds for cell in P.condition_grid()
        for checkpoint in checkpoints
    }
    observed = set()
    bundles = frame[["model_key", "condition", "seed", "checkpoint_step"]].drop_duplicates()
    for row in bundles.itertuples(index=False):
        checkpoint = -1 if pd.isna(row.checkpoint_step) else int(row.checkpoint_step)
        observed.add((str(row.model_key), str(row.condition), int(row.seed), checkpoint))
    if observed != expected:
        raise P.PaperCContractError(
            f"development bundle grid mismatch: missing={len(expected-observed)}, "
            f"extra={len(observed-expected)}")
    _validate_common_rows(frame)


def _validate_condition_semantics(frame: pd.DataFrame) -> None:
    for row in frame[[
        "condition", "stage", "objective", "sampler", "seed", "adapter_sha256",
    ]].drop_duplicates().itertuples(index=False):
        if row.condition == "base":
            if row.stage != "base" or int(row.seed) != -1 or pd.notna(row.adapter_sha256):
                raise P.PaperCContractError("invalid base bundle semantics")
        elif row.condition == "stage1_sft":
            if (row.stage != "stage1" or pd.isna(row.adapter_sha256)
                    or pd.notna(row.objective) or pd.notna(row.sampler)):
                raise P.PaperCContractError("invalid Stage-1 bundle semantics")
        else:
            if row.stage != "stage2" or row.objective not in P.OBJECTIVES \
                    or row.sampler not in P.SAMPLERS:
                raise P.PaperCContractError("invalid Stage-2 bundle semantics")
            if row.condition != P.run_condition(str(row.objective), str(row.sampler)):
                raise P.PaperCContractError("condition disagrees with objective/sampler")
            if pd.isna(row.adapter_sha256):
                raise P.PaperCContractError("Stage-2 bundle has no adapter hash")


def _macro_ap(group: pd.DataFrame, *, weights: np.ndarray | None = None) -> float:
    values: list[float] = []
    for _, source_rows in group.groupby("source", sort=True):
        labels = source_rows["gold"].to_numpy(int)
        scores = source_rows["score_raw"].to_numpy(float)
        if np.unique(labels).size < 2:
            continue
        source_weights = None if weights is None else weights[source_rows.index.to_numpy(int)]
        if source_weights is not None:
            pos = source_weights[labels == 1].sum()
            neg = source_weights[labels == 0].sum()
            if pos <= 0 or neg <= 0:
                return float("nan")
        values.append(float(average_precision_score(
            labels, scores, sample_weight=source_weights)))
    return float(np.mean(values)) if values else float("nan")


def _bernoulli_kl(policy_probability: np.ndarray,
                  reference_probability: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(np.asarray(policy_probability, float), eps, 1.0 - eps)
    q = np.clip(np.asarray(reference_probability, float), eps, 1.0 - eps)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def select_checkpoints(frame: pd.DataFrame, lock: Mapping) -> list[dict]:
    """Apply the frozen development-only target rule to every Stage-2 arm."""
    validate_dev_grid(frame, lock)
    stage2 = lock["config"]["value"]["stage2"]
    margin = float(stage2["represented_noninferiority_margin"])
    max_step = int(stage2["max_steps"])
    records: list[dict] = []
    for model in lock["models"]:
        for seed in lock["seeds"]:
            baseline = frame[
                (frame.model_key == model) & (frame.seed == int(seed))
                & (frame.condition == "stage1_sft")
            ].copy()
            baseline_ap = _macro_ap(baseline)
            if not math.isfinite(baseline_ap):
                raise P.PaperCContractError(
                    f"Stage-1 development AP is undefined for {model}/seed_{seed}")
            reference = baseline.set_index("sample_id")["probability_raw"]
            for cell in P.condition_grid():
                candidates = []
                condition_rows = frame[
                    (frame.model_key == model) & (frame.seed == int(seed))
                    & (frame.condition == cell["condition"])
                ]
                for checkpoint, scored in condition_rows.groupby("checkpoint_step", sort=True):
                    aligned = scored.set_index("sample_id").sort_index()
                    ref = reference.reindex(aligned.index)
                    if ref.isna().any():
                        raise P.PaperCContractError("Stage-1/dev checkpoint identities differ")
                    development_ap = _macro_ap(scored)
                    if not math.isfinite(development_ap):
                        raise P.PaperCContractError(
                            f"development AP is undefined for {model}/{cell['condition']}/"
                            f"seed_{seed}/step_{int(checkpoint)}")
                    candidates.append({
                        "checkpoint_step": int(checkpoint),
                        "development_macro_ap": development_ap,
                        "mean_kl_policy_to_stage1": float(np.mean(_bernoulli_kl(
                            aligned["probability_raw"].to_numpy(float),
                            ref.to_numpy(float),
                        ))),
                    })
                target = baseline_ap - margin
                feasible = [item for item in candidates
                            if item["development_macro_ap"] >= target]
                selected = min(feasible, key=lambda item: (
                    item["checkpoint_step"], item["mean_kl_policy_to_stage1"])) if feasible else None
                fallback = next(item for item in candidates
                                if item["checkpoint_step"] == max_step)
                records.append({
                    "model_key": model,
                    "seed": int(seed),
                    "condition": cell["condition"],
                    "objective": cell["objective"],
                    "sampler": cell["sampler"],
                    "stage1_development_macro_ap": baseline_ap,
                    "target_macro_ap": target,
                    "target_feasible": selected is not None,
                    "selected_checkpoint_step": (
                        None if selected is None else selected["checkpoint_step"]),
                    "descriptive_fallback_checkpoint_step": (
                        None if selected is not None else fallback["checkpoint_step"]),
                    "selected_development_macro_ap": (
                        None if selected is None else selected["development_macro_ap"]),
                    "selected_mean_kl_policy_to_stage1": (
                        None if selected is None else selected["mean_kl_policy_to_stage1"]),
                    "candidates": candidates,
                })
    return records


def _selection_lookup(path: str, lock: Mapping) -> tuple[dict, dict]:
    artifact = P.read_json(path)
    if artifact.get("lock_sha256") != lock.get("lock_sha256"):
        raise P.PaperCContractError("checkpoint table was produced under another lock")
    records = artifact.get("records")
    if not isinstance(records, list):
        raise P.PaperCContractError("checkpoint table has no records list")
    expected_hash = P.canonical_sha256(records)
    if artifact.get("records_sha256") != expected_hash:
        raise P.PaperCContractError("checkpoint-selection record hash mismatch")
    lookup = {}
    for record in records:
        key = (str(record["model_key"]), str(record["condition"]), int(record["seed"]))
        if key in lookup:
            raise P.PaperCContractError(f"duplicate checkpoint-selection row: {key}")
        lookup[key] = record
    return artifact, lookup


def validate_selected_checkpoints(frame: pd.DataFrame, selection_path: str,
                                  lock: Mapping) -> dict:
    artifact, lookup = _selection_lookup(selection_path, lock)
    expected_keys = {
        (model, cell["condition"], int(seed))
        for model in lock["models"] for seed in lock["seeds"]
        for cell in P.condition_grid()
    }
    if set(lookup) != expected_keys:
        raise P.PaperCContractError("checkpoint-selection table has an incomplete condition grid")
    stage2 = frame[frame["stage"] == "stage2"]
    for key, group in stage2.groupby(["model_key", "condition", "seed"]):
        record = lookup[(str(key[0]), str(key[1]), int(key[2]))]
        expected_step = (record.get("selected_checkpoint_step")
                         if record.get("target_feasible")
                         else record.get("descriptive_fallback_checkpoint_step"))
        observed = {int(value) for value in group["checkpoint_step"].dropna().unique()}
        if expected_step is None or observed != {int(expected_step)}:
            raise P.PaperCContractError(
                f"retrospective bundle uses the wrong selected checkpoint for {key}")
    return artifact


def cell_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    conditions = frame[frame["condition"] != "base"]
    for (model, seed, condition, regime), group in conditions.assign(
        regime=conditions["split"].map(
            {value: key for key, value in REGIME_SPLITS.items()})
    ).dropna(subset=["regime"]).groupby(
        ["model_key", "seed", "condition", "regime"], sort=True
    ):
        first = group.iloc[0]
        source_brier = []
        source_nll = []
        for _, source_rows in group.groupby("source", sort=True):
            source_brier.append(brier(
                source_rows["probability_calibrated"], source_rows["gold"]))
            source_nll.append(log_loss_(
                source_rows["probability_calibrated"], source_rows["gold"]))
        rows.append({
            "model_key": model,
            "seed": int(seed),
            "condition": condition,
            "stage": first["stage"],
            "objective": first["objective"],
            "sampler": first["sampler"],
            "checkpoint_step": first["checkpoint_step"],
            "regime": regime,
            "macro_ap": _macro_ap(group),
            "macro_calibrated_brier": float(np.mean(source_brier)),
            "macro_calibrated_nll": float(np.mean(source_nll)),
            "n_rows": len(group),
            "n_sources": group["source"].nunique(),
        })
    result = pd.DataFrame(rows)
    if result.empty or not np.isfinite(result["macro_ap"]).all():
        raise P.PaperCContractError("one or more primary macro-AP cells are undefined")
    return result


def contrast_table(cells: pd.DataFrame) -> pd.DataFrame:
    stage2 = cells[cells["stage"] == "stage2"]
    lookup = stage2.set_index(
        ["model_key", "seed", "regime", "sampler", "objective"])["macro_ap"]
    records: list[dict] = []
    groups = stage2[["model_key", "seed", "regime", "sampler"]].drop_duplicates()
    for row in groups.itertuples(index=False):
        prefix = (row.model_key, int(row.seed), row.regime, row.sampler)
        for name, (left, right) in CONTRASTS.items():
            records.append({
                "model_key": row.model_key,
                "seed": int(row.seed),
                "regime": row.regime,
                "sampler": row.sampler,
                "contrast": name,
                "estimate": float(lookup[prefix + (left,)] - lookup[prefix + (right,)]),
            })
    table = pd.DataFrame(records)
    for (model, seed, regime), group in table[
        table["contrast"] == "C_total"
    ].groupby(["model_key", "seed", "regime"]):
        by_sampler = group.set_index("sampler")["estimate"]
        table.loc[len(table)] = {
            "model_key": model, "seed": int(seed), "regime": regime,
            "sampler": "interaction", "contrast": "C_selection_interaction",
            "estimate": float(by_sampler["uncertain"] - by_sampler["matched_random"]),
        }
    return table


def hierarchical_bootstrap(frame: pd.DataFrame, lock: Mapping, *, reps: int,
                           seed: int, max_redraw: int = 2000) -> pd.DataFrame:
    """Paired Poisson-family/seed bootstrap for all primary contrasts.

    The model panel is fixed, matching Paper A.  One family-weight vector is
    shared across every objective and sampler.  Seeds are resampled within each
    model with replacement, and the same draws are reused for every contrast.
    """
    if reps < 1:
        raise P.PaperCContractError("bootstrap replicate count must be positive")
    primary = frame[
        (frame["stage"] == "stage2")
        & (frame["split"].isin(REGIME_SPLITS.values()))
    ].copy().reset_index(drop=True)
    if primary.empty:
        raise P.PaperCContractError("no Stage-2 primary rows are available for bootstrap")
    rng = np.random.default_rng(seed)
    models = [str(model) for model in lock["models"]]
    seeds = [int(value) for value in lock["seeds"]]
    family_keys = sorted(str(value) for value in primary["family_id"].unique())
    family_index = {key: index for index, key in enumerate(family_keys)}
    primary["_family_index"] = [
        family_index[str(family)] for family in primary["family_id"]
    ]
    # A single representative bundle is sufficient for zero-class validation,
    # because the scorer contract already proved row identity across bundles.
    representative_condition = P.condition_grid()[0]["condition"]
    representative = primary[
        (primary.model_key == models[0]) & (primary.seed == seeds[0])
        & (primary.condition == representative_condition)
    ]
    validity_entries = []
    for _, group in representative.groupby(["split", "source"], sort=True):
        validity_entries.append((
            group["gold"].to_numpy(int), group["_family_index"].to_numpy(int)))

    entries: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, group in primary.groupby(
        ["model_key", "seed", "split", "sampler", "objective", "source"], sort=True
    ):
        entries[tuple(key)] = (
            group["score_raw"].to_numpy(float),
            group["gold"].to_numpy(int),
            group["_family_index"].to_numpy(int),
        )
    strata = [
        (regime, sampler, contrast)
        for regime in REGIME_SPLITS for sampler in P.SAMPLERS for contrast in CONTRASTS
    ] + [
        (regime, "interaction", "C_selection_interaction")
        for regime in REGIME_SPLITS
    ]
    samples = {stratum: np.empty(reps, float) for stratum in strata}
    redraws = 0
    for replicate in range(reps):
        attempts = 0
        while True:
            weights = rng.poisson(1.0, size=len(family_keys)).astype(float)
            valid = all(
                weights[indexes[labels == 1]].sum() > 0
                and weights[indexes[labels == 0]].sum() > 0
                for labels, indexes in validity_entries
            )
            if valid:
                break
            attempts += 1
            redraws += 1
            if attempts > max_redraw:
                raise P.PaperCContractError(
                    "hierarchical bootstrap exceeded the zero-class redraw cap")
        seed_pick = {
            model: rng.integers(0, len(seeds), size=len(seeds)) for model in models
        }
        ap_values: dict[tuple, float] = {}
        for regime, split in REGIME_SPLITS.items():
            sources = sorted(primary[primary.split == split]["source"].unique())
            for model in models:
                for run_seed in seeds:
                    for sampler in P.SAMPLERS:
                        for objective in P.OBJECTIVES:
                            source_values = []
                            for source in sources:
                                scores, labels, indexes = entries[
                                    (model, run_seed, split, sampler, objective, source)]
                                source_values.append(float(average_precision_score(
                                    labels, scores, sample_weight=weights[indexes])))
                            ap_values[(regime, model, run_seed, sampler, objective)] = float(
                                np.mean(source_values))
        for regime in REGIME_SPLITS:
            for sampler in P.SAMPLERS:
                for contrast, (left, right) in CONTRASTS.items():
                    per_model = []
                    for model in models:
                        per_seed = {
                            run_seed: (
                                ap_values[(regime, model, run_seed, sampler, left)]
                                - ap_values[(regime, model, run_seed, sampler, right)]
                            ) for run_seed in seeds
                        }
                        per_model.append(float(np.mean([
                            per_seed[seeds[index]] for index in seed_pick[model]
                        ])))
                    samples[(regime, sampler, contrast)][replicate] = float(
                        np.mean(per_model))
            per_model = []
            for model in models:
                per_seed = {
                    run_seed: (
                        (ap_values[(regime, model, run_seed, "uncertain", "dpo")]
                         - ap_values[(regime, model, run_seed, "uncertain", "verdict_ce")])
                        - (ap_values[(regime, model, run_seed, "matched_random", "dpo")]
                           - ap_values[(regime, model, run_seed, "matched_random", "verdict_ce")])
                    ) for run_seed in seeds
                }
                per_model.append(float(np.mean([
                    per_seed[seeds[index]] for index in seed_pick[model]
                ])))
            samples[(regime, "interaction", "C_selection_interaction")][replicate] = float(
                np.mean(per_model))
        if reps >= 1000 and ((replicate + 1) % max(1, reps // 10) == 0):
            print(f"[paper-c bootstrap] {replicate + 1}/{reps}", flush=True)

    point = contrast_table(cell_metrics(frame))
    records = []
    for (regime, sampler, contrast), draws in samples.items():
        subset = point[
            (point.regime == regime) & (point.sampler == sampler)
            & (point.contrast == contrast)
        ]
        records.append({
            "regime": regime,
            "sampler": sampler,
            "contrast": contrast,
            "estimate": float(subset["estimate"].mean()),
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "bootstrap_replicates": int(reps),
            "bootstrap_family_count": len(family_keys),
            "bootstrap_zero_class_redraws": int(redraws),
            "interval_method": "paired_poisson_family_and_within_model_seed_bootstrap",
        })
    return pd.DataFrame(records)


def movement_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Descriptive KL(policy || Stage-1), separately on each primary regime."""
    records = []
    stage1 = frame[frame.condition == "stage1_sft"]
    policy_rows = frame[
        (frame.stage == "stage2") & frame["split"].isin(REGIME_SPLITS.values())
    ].assign(regime=lambda value: value["split"].map(
        {split: regime for regime, split in REGIME_SPLITS.items()}))
    for (model, seed, condition, regime), policy in policy_rows.groupby(
        ["model_key", "seed", "condition", "regime"], sort=True
    ):
        reference = stage1[
            (stage1.model_key == model) & (stage1.seed == int(seed))
            & (stage1["split"] == REGIME_SPLITS[regime])
        ].set_index(["split", "sample_id"])
        aligned = policy.set_index(["split", "sample_id"]).sort_index()
        ref = reference.reindex(aligned.index)
        if ref["probability_raw"].isna().any():
            raise P.PaperCContractError("Stage-1 and Stage-2 score identities differ")
        records.append({
            "model_key": model,
            "seed": int(seed),
            "condition": condition,
            "regime": regime,
            "objective": aligned["objective"].iloc[0],
            "sampler": aligned["sampler"].iloc[0],
            "mean_kl_policy_to_stage1": float(np.mean(_bernoulli_kl(
                aligned["probability_raw"].to_numpy(float),
                ref["probability_raw"].to_numpy(float),
            ))),
            "mean_signed_margin_change": float(np.mean(
                (2 * aligned["gold"].to_numpy(float) - 1)
                * (aligned["score_raw"].to_numpy(float)
                   - ref["score_raw"].to_numpy(float))
            )),
            "n_rows": len(aligned),
        })
    return pd.DataFrame(records)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")


def cmd_select(args: argparse.Namespace) -> int:
    if not args.development:
        raise P.PaperCContractError(
            "final checkpoint selection is disabled until candidate-adapter inventory binding is implemented")
    lock = _load_lock(args.lock, development=args.development)
    frame, metadata = _read_scores(
        args.dev_scores, args.dev_metadata, lock, development=args.development)
    records = select_checkpoints(frame, lock)
    artifact = {
        "analysis_artifact_version": ANALYSIS_ARTIFACT_VERSION,
        "artifact_kind": "paper_c_stage2_checkpoint_selection",
        "finalization_status": "development" if args.development else "final",
        "lock_sha256": lock["lock_sha256"],
        "dev_scores_sha256": metadata["scores_sha256"],
        "selection_rule": lock["config"]["value"]["stage2"]["checkpoint_rule"],
        "records": records,
        "records_sha256": P.canonical_sha256(records),
    }
    P.write_json(args.out, artifact)
    print(f"[paper-c analyze] selected checkpoints for {len(records)} cells -> {args.out}")
    return 0


def cmd_retrospective(args: argparse.Namespace) -> int:
    if not args.development:
        raise P.PaperCContractError(
            "final retrospective analysis is disabled until all preregistered diagnostics and sensitivities are implemented")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise P.PaperCContractError(f"refusing to overwrite nonempty analysis directory: {out}")
    lock = _load_lock(args.lock, development=args.development)
    frame, score_metadata = _read_scores(
        args.scores, args.metadata, lock, development=args.development)
    validate_retrospective_grid(frame, lock)
    selection = validate_selected_checkpoints(frame, args.checkpoint_selection, lock)
    cells = cell_metrics(frame)
    contrasts = contrast_table(cells)
    configured_reps = int(lock["config"]["value"]["analysis"]["bootstrap_replicates"])
    reps = int(args.bootstrap_replicates or configured_reps)
    if not args.development and reps != configured_reps:
        raise P.PaperCContractError("final analysis may not override bootstrap_replicates")
    intervals = hierarchical_bootstrap(
        frame, lock, reps=reps,
        seed=int(lock["config"]["value"]["analysis"]["bootstrap_seed"]),
    )
    movement = movement_table(frame)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "cell_metrics.csv", cells)
    _write_csv(out / "paired_contrasts.csv", contrasts)
    _write_csv(out / "development_intervals.csv", intervals)
    _write_csv(out / "movement.csv", movement)
    infeasible = sum(not bool(record["target_feasible"])
                     for record in selection["records"])
    summary = {
        "analysis_artifact_version": ANALYSIS_ARTIFACT_VERSION,
        "analysis_scope": "retrospective_estimation_only",
        "finalization_status": "development" if args.development else "final",
        "lock_sha256": lock["lock_sha256"],
        "scores_sha256": score_metadata["scores_sha256"],
        "checkpoint_selection_file_sha256": P.sha256_file(args.checkpoint_selection),
        "target_infeasible_cells": int(infeasible),
        "formal_claim_authorized": False,
        "interval_status": "retrospective paired hierarchical intervals; estimation only",
        "diagnostic_availability": {
            "two_verdict_kl": "available_as_descriptive_kl_policy_to_stage1_by_primary_regime",
            "full_vocabulary_kl": "unavailable_score_schema_has_only_two_verdict_logits",
            "compute_and_lora": "unavailable_until_run_metadata_is_retained_by_inventory",
        },
        "outputs": {},
    }
    for name in ("cell_metrics.csv", "paired_contrasts.csv",
                 "development_intervals.csv", "movement.csv"):
        summary["outputs"][name] = P.sha256_file(out / name)
    P.write_json(out / "summary.json", summary)
    print(f"[paper-c analyze] retrospective development analysis -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Paper C v2 score artifacts")
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser(
        "select-checkpoints", help="select using only the Stage-2 development score artifact")
    select.add_argument("--lock", required=True)
    select.add_argument("--dev-scores", required=True)
    select.add_argument("--dev-metadata", required=True)
    select.add_argument("--out", required=True)
    select.add_argument("--development", action="store_true")
    select.set_defaults(func=cmd_select)

    retrospective = sub.add_parser(
        "retrospective", help="analyze selected checkpoints on retrospective test rows")
    retrospective.add_argument("--lock", required=True)
    retrospective.add_argument("--scores", required=True)
    retrospective.add_argument("--metadata", required=True)
    retrospective.add_argument("--checkpoint-selection", required=True)
    retrospective.add_argument("--out", required=True)
    retrospective.add_argument("--bootstrap-replicates", type=int)
    retrospective.add_argument("--development", action="store_true")
    retrospective.set_defaults(func=cmd_retrospective)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (P.PaperCContractError, OSError, ValueError, KeyError) as exc:
        print(f"[paper-c analyze] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
