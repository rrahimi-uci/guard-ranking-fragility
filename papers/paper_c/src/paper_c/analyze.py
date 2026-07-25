"""Checkpoint selection and primary point-estimate analysis.

Checkpoint selection and retrospective analysis are intentionally separate
functions and artifacts. Test scores are never accepted by the selector.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from .contracts import ContractError, OBJECTIVES, SAMPLERS, output_path, read_jsonl, write_json, write_jsonl


def tie_aware_average_precision(gold: Sequence[int], scores: Sequence[float]) -> float:
    if len(gold) != len(scores) or not gold:
        raise ContractError("AP inputs must be nonempty and aligned")
    total_positive = sum(int(value) for value in gold)
    if total_positive == 0 or total_positive == len(gold):
        raise ContractError("AP requires both classes")
    groups: dict[float, list[int]] = defaultdict(list)
    for label, score in zip(gold, scores, strict=True):
        if int(label) not in (0, 1):
            raise ContractError("AP labels must be binary")
        groups[float(score)].append(int(label))
    cumulative_positive = 0
    cumulative_total = 0
    ap = 0.0
    for score in sorted(groups, reverse=True):
        labels = groups[score]
        positives = sum(labels)
        cumulative_positive += positives
        cumulative_total += len(labels)
        precision = cumulative_positive / cumulative_total
        ap += (positives / total_positive) * precision
    return ap


def source_macro_ap(rows: Sequence[Mapping]) -> float:
    by_source: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)
    if not by_source:
        raise ContractError("cannot compute macro AP on empty rows")
    values = []
    for source, source_rows in sorted(by_source.items()):
        try:
            values.append(tie_aware_average_precision(
                [int(row["gold"]) for row in source_rows],
                [float(row["score_unsafe_minus_safe"]) for row in source_rows],
            ))
        except ContractError as exc:
            raise ContractError(f"invalid AP source {source}: {exc}") from exc
    return sum(values) / len(values)


def select_checkpoints(
    *,
    config: dict,
    stage1_scores_path: str | Path,
    candidate_scores_path: str | Path,
    out_path: str | Path,
) -> list[dict]:
    baseline_rows = read_jsonl(output_path(stage1_scores_path))
    candidate_rows = read_jsonl(output_path(candidate_scores_path))
    if any(row.get("score_role") != "stage2_dev" for row in baseline_rows + candidate_rows):
        raise ContractError("checkpoint selection accepts Stage-2 development scores only")
    baseline_groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in baseline_rows:
        baseline_groups[(str(row["model_key"]), int(row["seed"]))].append(row)
    candidate_groups: dict[tuple[str, int, str, str, int], list[dict]] = defaultdict(list)
    for row in candidate_rows:
        key = (
            str(row["model_key"]), int(row["seed"]), str(row["sampler"]),
            str(row["objective"]), int(row["step"]),
        )
        candidate_groups[key].append(row)
    margin = float(config["stage2"]["represented_noninferiority_margin"])
    output = []
    for model_key in config["models"]:
        for seed in config["seeds"]:
            baseline_key = (model_key, seed)
            if baseline_key not in baseline_groups:
                raise ContractError(f"missing Stage-1 development score bundle: {baseline_key}")
            baseline_ap = source_macro_ap(baseline_groups[baseline_key])
            threshold = baseline_ap - margin
            for sampler in SAMPLERS:
                for objective in OBJECTIVES:
                    candidates = []
                    for step in config["stage2"]["checkpoint_steps"]:
                        key = (model_key, seed, sampler, objective, int(step))
                        if key not in candidate_groups:
                            raise ContractError(f"missing candidate development scores: {key}")
                        candidates.append((int(step), source_macro_ap(candidate_groups[key])))
                    feasible = [(step, ap) for step, ap in candidates if ap >= threshold]
                    if feasible:
                        selected_step, selected_ap = feasible[0]
                        status = "target_reached"
                    else:
                        selected_step, selected_ap = candidates[-1]
                        status = "target_infeasible"
                    output.append({
                        "model_key": model_key,
                        "seed": seed,
                        "sampler": sampler,
                        "objective": objective,
                        "stage1_dev_macro_ap": baseline_ap,
                        "target_macro_ap": threshold,
                        "selected_step": selected_step,
                        "selected_dev_macro_ap": selected_ap,
                        "selection_status": status,
                        "eligible_for_primary_target_matched_contrast": status == "target_reached",
                    })
    if len(output) != 120:
        raise ContractError("checkpoint selector did not produce the exact 120-cell table")
    write_jsonl(out_path, output)
    write_json(f"{out_path}.metadata.json", {
        "kind": "paper_c_checkpoint_selection",
        "input_role": "stage2_dev_only",
        "rows": len(output),
        "primary_panel_complete": all(
            row["eligible_for_primary_target_matched_contrast"] for row in output
        ),
    })
    return output


def factorial_contrasts(metric_rows: Sequence[Mapping]) -> dict:
    """Compute paired point estimates; interval implementation is a later gate."""
    values = {}
    for row in metric_rows:
        key = (
            str(row["model_key"]), int(row["seed"]), str(row["regime"]),
            str(row["sampler"]), str(row["objective"]),
        )
        if key in values:
            raise ContractError(f"duplicate metric cell: {key}")
        values[key] = float(row["macro_ap"])
    output = {"per_cell": [], "factorial_marginal": []}
    panel = sorted({(key[0], key[1], key[2]) for key in values})
    for model_key, seed, regime in panel:
        sampler_effects = {}
        for sampler in SAMPLERS:
            def value(objective):
                key = (model_key, seed, regime, sampler, objective)
                if key not in values:
                    raise ContractError(f"incomplete primary metric grid: {key}")
                return values[key]
            contrasts = {
                "c_pair": value("pair_ce") - value("verdict_ce"),
                "c_ref": value("dpo") - value("pair_ce"),
                "c_total": value("dpo") - value("verdict_ce"),
            }
            sampler_effects[sampler] = contrasts
            output["per_cell"].append({
                "model_key": model_key, "seed": seed, "regime": regime,
                "sampler": sampler, **contrasts,
            })
        output["factorial_marginal"].append({
            "model_key": model_key,
            "seed": seed,
            "regime": regime,
            "c_pair": sum(sampler_effects[s]["c_pair"] for s in SAMPLERS) / 2.0,
            "c_ref": sum(sampler_effects[s]["c_ref"] for s in SAMPLERS) / 2.0,
            "c_total": sum(sampler_effects[s]["c_total"] for s in SAMPLERS) / 2.0,
            "selection_interaction": (
                sampler_effects["uncertain"]["c_total"]
                - sampler_effects["matched_random"]["c_total"]
            ),
        })
    return output
