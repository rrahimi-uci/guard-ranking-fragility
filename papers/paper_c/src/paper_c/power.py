"""Design power: what effect size can this panel actually detect, and is that plausible?

WHY THIS MODULE EXISTS
----------------------
The study's decision rule requires a one-sided 97.5% lower bound above zero. Whether that is
reachable is a property of the design, not of the result, so it must be settled BEFORE Stage 1 --
otherwise a null is uninterpretable: we would not know whether reference centering does nothing or
whether the panel could never have seen it.

The noise floor is not the row count. Seeds inside one checkpoint share the model, the frozen
manifest, the LoRA recipe and the data order, so they are not independent replicates of the effect;
the checkpoint is. A four-model panel therefore has roughly four independent units for a panel
mean, and the interval scales as `sd / sqrt(4)`, not `sd / sqrt(20)`.

`seed_sd` is measured from the vendored parent scores rather than assumed. On the parent SFT runs
the within-checkpoint seed SD of transfer macro-AP is ~0.033, which puts the smallest detectable
one-sided effect at roughly:

    n_effective = 4   ->  MDE ~ 0.033 / 2  * 1.96  ~  0.033
    n_effective = 20  ->  MDE ~ 0.033 / 4.5 * 1.96 ~  0.015

For scale, an *explicit* KL anchor moved parent transfer by +0.061, and reference centering starts
as an approximately uniform ~1.3x gradient rescale. Expecting a single-point `C_ref` above +0.03 is
therefore optimistic, which is the quantitative reason the primary estimand moved to the frontier
form (see `frontier.py`): pairing at matched represented AP over a four-point ladder removes the
checkpoint-level variance component that dominates this SD, and the module reports the variance
reduction it needs in order to be adequately powered.

Nothing here consumes a result. It reads parent score bundles and the design constants only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import defaultdict
import math

from .contracts import ContractError

POWER_VERSION = "power_v1"

#: Normal deviate for a one-sided 97.5% bound (the study's per-question level after the
#: Bonferroni split across the two research questions).
Z_ONE_SIDED_975 = 1.959963985


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ContractError("standard deviation needs at least two observations")
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def seed_sd_by_model(rows: Sequence[Mapping], *, metric: str = "transfer_ap") -> dict:
    """Within-checkpoint seed SD of a per-cell metric, plus the pooled value.

    `rows` carries one record per (model, seed) with the metric already aggregated to macro-AP.
    This is the honest noise floor for a same-recipe rerun: it includes initialisation, dropout and
    execution nondeterminism, and excludes anything that differs between objectives.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if "model_key" not in row or metric not in row:
            raise ContractError(f"seed-variance rows need model_key and {metric}")
        grouped[str(row["model_key"])].append(float(row[metric]))
    if not grouped:
        raise ContractError("no rows supplied for seed-variance estimation")
    per_model = {}
    for model_key, values in sorted(grouped.items()):
        if len(values) < 2:
            continue
        per_model[model_key] = {"n_seeds": len(values), "sd": _sample_sd(values),
                                "mean": _mean(values)}
    if not per_model:
        raise ContractError("seed-variance estimation needs at least one model with two seeds")
    pooled = math.sqrt(_mean([record["sd"] ** 2 for record in per_model.values()]))
    return {"metric": metric, "per_model": per_model, "pooled_sd": pooled,
            "n_models": len(per_model)}


def minimum_detectable_effect(
    *,
    sd: float,
    n_effective: int,
    pairing_variance_reduction: float = 1.0,
    z: float = Z_ONE_SIDED_975,
) -> float:
    """Smallest true effect whose one-sided lower bound would clear zero in expectation.

    `pairing_variance_reduction` is the fraction of the raw variance that survives pairing. It is 1.0
    for an unpaired contrast. A paired contrast that shares model, seed, sampler, initialisation and
    scored rows removes the checkpoint-level component, which is the dominant one here; the design
    must *measure* this factor on the pilot rather than assume it.
    """
    if not math.isfinite(sd) or sd <= 0:
        raise ContractError("sd must be finite and positive")
    if int(n_effective) < 1:
        raise ContractError("n_effective must be at least 1")
    if not 0 < float(pairing_variance_reduction) <= 1:
        raise ContractError("pairing_variance_reduction must lie in (0, 1]")
    standard_error = sd * math.sqrt(float(pairing_variance_reduction)) / math.sqrt(int(n_effective))
    return float(z) * standard_error


def design_power_report(
    *,
    seed_variance: Mapping,
    n_models: int,
    n_seeds: int,
    target_effect: float,
    pairing_variance_reduction: float,
    ladder_points: int,
) -> dict:
    """Compare the design's MDE against the effect the study is willing to call meaningful.

    Two unit counts are reported deliberately, because the difference between them is the single
    largest design decision in the study:

    * `clustered`  -- one independent unit per checkpoint (the defensible reading).
    * `optimistic` -- one per (checkpoint, seed) cell, which assumes seeds are independent draws of
      the effect. Reported only so the gap is explicit; it is not the basis of the gate.
    """
    sd = float(seed_variance["pooled_sd"])
    target = float(target_effect)
    if not math.isfinite(target) or target <= 0:
        raise ContractError("target_effect must be finite and positive")
    scenarios = {}
    for label, n_effective, reduction in (
        ("point_estimand_clustered", int(n_models), 1.0),
        ("point_estimand_optimistic", int(n_models) * int(n_seeds), 1.0),
        ("frontier_estimand_clustered", int(n_models), float(pairing_variance_reduction)),
    ):
        mde = minimum_detectable_effect(sd=sd, n_effective=n_effective,
                                        pairing_variance_reduction=reduction)
        scenarios[label] = {
            "n_effective": n_effective,
            "pairing_variance_reduction": reduction,
            "standard_error": sd * math.sqrt(reduction) / math.sqrt(n_effective),
            "minimum_detectable_effect": mde,
            "powered_for_target": bool(mde <= target),
        }
    primary = scenarios["frontier_estimand_clustered"]
    return {
        "power_version": POWER_VERSION,
        "pooled_seed_sd": sd,
        "per_model_seed_sd": {key: value["sd"] for key, value in
                              sorted(seed_variance["per_model"].items())},
        "n_models": int(n_models),
        "n_seeds": int(n_seeds),
        "ladder_points_per_cell": int(ladder_points),
        "target_effect": target,
        "scenarios": scenarios,
        "primary_scenario": "frontier_estimand_clustered",
        "primary_minimum_detectable_effect": primary["minimum_detectable_effect"],
        "required_pairing_variance_reduction": _required_reduction(
            sd=sd, n_effective=int(n_models), target=target),
        "verdict": "powered" if primary["powered_for_target"] else "underpowered",
    }


def _required_reduction(*, sd: float, n_effective: int, target: float) -> float:
    """Variance-reduction factor pairing must achieve for the design to detect `target`."""
    standard_error_needed = target / Z_ONE_SIDED_975
    raw_standard_error = sd / math.sqrt(n_effective)
    if raw_standard_error <= 0:
        raise ContractError("degenerate standard error")
    ratio = (standard_error_needed / raw_standard_error) ** 2
    return min(1.0, ratio)


def assert_design_powered(report: Mapping) -> None:
    """Fail closed when the design cannot detect the effect it is committed to calling meaningful.

    The pilot exists to replace the assumed `pairing_variance_reduction` with a measured one. Until
    that measurement exists, an underpowered verdict must block the full panel rather than be
    discovered afterwards in the discussion section.
    """
    if report.get("verdict") != "powered":
        required = report.get("required_pairing_variance_reduction")
        raise ContractError(
            "design is underpowered for its own target effect: primary MDE "
            f"{report.get('primary_minimum_detectable_effect'):.4f} exceeds target "
            f"{report.get('target_effect'):.4f}. Pairing must remove at least "
            f"{1.0 - float(required):.1%} of the variance, or the target effect, seed count or "
            "panel size must change before Stage 2 is authorised."
        )


def effective_learning_rate_ratio(
    reference_margins: Sequence[float],
    *,
    beta: float,
) -> dict:
    """How much of reference centering is just a larger effective learning rate at step zero.

    At initialisation the policy margin equals the reference margin, so the DPO per-example gradient
    weight is `sigma(0) = 0.5` for every row while PairCE's is `sigma(-beta*m_ref)`. The ratio of
    the means is the effective-LR multiplier that DPO enjoys for free; the coefficient of variation
    of the PairCE weights says how much of the reweighting is *differential* rather than uniform.

    A high ratio with a low coefficient of variation is the confound this study must neutralise --
    and is precisely what the frontier estimand is designed to be invariant to.
    """
    if not reference_margins:
        raise ContractError("effective-LR diagnostic needs reference margins")
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0:
        raise ContractError("beta must be finite and positive")
    weights = []
    for margin in reference_margins:
        exponent = beta * float(margin)
        exponent = max(min(exponent, 60.0), -60.0)
        weights.append(1.0 / (1.0 + math.exp(exponent)))
    mean_weight = _mean(weights)
    sd_weight = _sample_sd(weights) if len(weights) > 1 else 0.0
    positive = sum(1 for margin in reference_margins if float(margin) > 0)
    return {
        "beta": beta,
        "n_rows": len(weights),
        "dpo_weight_at_init": 0.5,
        "pair_ce_mean_weight": mean_weight,
        "pair_ce_weight_sd": sd_weight,
        "pair_ce_weight_cv": (sd_weight / mean_weight) if mean_weight > 0 else float("inf"),
        "effective_lr_ratio": 0.5 / mean_weight if mean_weight > 0 else float("inf"),
        "fraction_positive_margin": positive / len(reference_margins),
        "interpretation": (
            "A ratio well above 1 with a small coefficient of variation means reference centering "
            "acts mostly as a uniform gradient rescale at step zero, so step-matched contrasts are "
            "confounded with effective learning rate and only the frontier estimand is safe."
        ),
    }
