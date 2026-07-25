"""Frontier-based estimands: the LR-invariant test of what reference centering buys.

WHY THIS MODULE EXISTS
----------------------
The obvious estimand -- `C_ref = AP(DPO) - AP(PairCE)` at one selected checkpoint -- is both
underpowered and confounded, for one shared reason.

At Stage-2 initialisation the policy margin equals the reference margin by construction, so DPO
weights every training example at exactly `sigma(0) = 0.5`, while PairCE weights it at
`sigma(-beta*m_ref)`. On this panel's Stage-1 margins that is a *near-uniform* reweighting: 93% of
rows sit at positive margin, and the PairCE weight has interquartile range ~0.05 around a mean of
~0.385. Reference centering therefore begins life as an approximately constant ~1.3x gradient
rescale -- i.e. as a larger effective learning rate, not as a differential re-emphasis.

A single-point AP difference cannot distinguish "reference centering is a better objective" from
"reference centering is the same objective at a higher effective learning rate". Both raise AP at a
fixed step count. That is exactly the confound class this study was redesigned to eliminate, and it
is invisible to `C_ref`.

The frontier estimand removes it. A pure learning-rate rescale moves a run *along* its own
represented/transfer trade-off curve; it does not move the curve. So we compare curves, not points:

  1. Each (model, seed, sampler, objective) cell contributes a trajectory of
     (represented AP, transfer AP) pairs over the checkpoint ladder.
  2. Trajectories are compared at *matched represented AP*, by interpolating each one onto a shared
     represented-AP grid and reading off transfer.
  3. The statistic is the mean vertical (transfer) gap over the overlapping represented range.

If reference centering is only an effective-LR change, the gap is zero at every matched represented
level even though the raw per-step APs differ. If it is a genuine anti-forgetting mechanism -- the
same family as an explicit KL anchor or an output-space average -- it buys transfer *at equal
represented ranking*, and the gap is positive. The confound becomes the null hypothesis.

The ladder also multiplies the evidence per GPU-hour: four checkpoints per cell instead of one, and
the paired construction (same model, same seed, same sampler, same rows) cancels the dominant
variance component, which is the checkpoint-level effect.

Everything here is pure Python on already-computed AP values. No torch, no GPU, no network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import defaultdict
import math

from .contracts import ContractError, OBJECTIVES, SAMPLERS

FRONTIER_VERSION = "frontier_v1"

#: Trajectories are compared on this many equally spaced represented-AP levels inside the
#: overlapping range of the two cells being contrasted. Fixed in advance so the grid can never be
#: chosen after seeing results.
GRID_POINTS = 21

#: A contrast is only formed when the two trajectories overlap in represented AP by at least this
#: much. Narrower overlap means the comparison would be an extrapolation.
MIN_OVERLAP = 0.01


def _finite(value: object, label: str) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ContractError(f"{label} must be finite")
    return number


def trajectory(points: Sequence[Mapping]) -> list[tuple[float, float]]:
    """Order one cell's checkpoint ladder into a monotone represented-AP trajectory.

    `points` carries one record per saved checkpoint with `step`, `represented_ap`, `transfer_ap`.
    Points are ordered by step (the training order, never by outcome). Represented AP is not
    guaranteed monotone in step, so we take the running maximum: the trajectory records, for each
    represented level actually reached, the transfer AP of the earliest checkpoint that reached it.
    That is the same "earliest checkpoint meeting a target" logic the selection rule uses, applied
    at every level instead of one.
    """
    if not points:
        raise ContractError("trajectory requires at least one checkpoint")
    ordered = sorted(points, key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in ordered]
    if len(set(steps)) != len(steps):
        raise ContractError("duplicate checkpoint steps in one cell")
    curve: list[tuple[float, float]] = []
    best_represented = -math.inf
    for row in ordered:
        represented = _finite(row["represented_ap"], "represented_ap")
        transfer = _finite(row["transfer_ap"], "transfer_ap")
        if represented > best_represented:
            best_represented = represented
            curve.append((represented, transfer))
    return curve


def interpolate_transfer(curve: Sequence[tuple[float, float]], level: float) -> float | None:
    """Transfer AP at a given represented-AP level, by linear interpolation inside the curve.

    Returns None outside the curve's represented range: we never extrapolate a frontier.
    """
    if not curve:
        return None
    if len(curve) == 1:
        return curve[0][1] if abs(curve[0][0] - level) < 1e-12 else None
    lo, hi = curve[0][0], curve[-1][0]
    if level < lo - 1e-12 or level > hi + 1e-12:
        return None
    for (x0, y0), (x1, y1) in zip(curve, curve[1:], strict=False):
        if x0 - 1e-12 <= level <= x1 + 1e-12:
            if x1 - x0 <= 1e-12:
                return (y0 + y1) / 2.0
            weight = (level - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    return None


def matched_transfer_gap(
    treatment: Sequence[tuple[float, float]],
    control: Sequence[tuple[float, float]],
    *,
    grid_points: int = GRID_POINTS,
    min_overlap: float = MIN_OVERLAP,
) -> dict:
    """Mean transfer gap between two trajectories at matched represented AP.

    This is the LR-invariant primitive. A uniform gradient rescale slides a cell along its own
    trajectory, so both cells' curves are unchanged and the gap is zero.
    """
    if not treatment or not control:
        raise ContractError("both trajectories must be nonempty")
    low = max(treatment[0][0], control[0][0])
    high = min(treatment[-1][0], control[-1][0])
    overlap = high - low
    if overlap < min_overlap:
        return {"status": "insufficient_overlap", "overlap": overlap, "gap": None,
                "n_levels": 0, "low": low, "high": high}
    if grid_points < 2:
        raise ContractError("grid_points must be at least 2")
    gaps: list[float] = []
    levels: list[float] = []
    for index in range(grid_points):
        level = low + (high - low) * index / (grid_points - 1)
        a = interpolate_transfer(treatment, level)
        b = interpolate_transfer(control, level)
        if a is None or b is None:
            continue
        gaps.append(a - b)
        levels.append(level)
    if not gaps:
        return {"status": "no_common_levels", "overlap": overlap, "gap": None,
                "n_levels": 0, "low": low, "high": high}
    return {
        "status": "ok",
        "overlap": overlap,
        "low": low,
        "high": high,
        "n_levels": len(gaps),
        "gap": sum(gaps) / len(gaps),
        "gap_min": min(gaps),
        "gap_max": max(gaps),
        "dominates_everywhere": all(value > 0 for value in gaps),
        "dominated_everywhere": all(value < 0 for value in gaps),
    }


def cell_key(row: Mapping) -> tuple[str, int, str, str]:
    return (str(row["model_key"]), int(row["seed"]), str(row["sampler"]), str(row["objective"]))


def build_trajectories(rows: Sequence[Mapping]) -> dict[tuple[str, int, str, str], list[tuple[float, float]]]:
    """Group scored candidate checkpoints into one trajectory per factorial cell."""
    grouped: dict[tuple[str, int, str, str], list[Mapping]] = defaultdict(list)
    for row in rows:
        for field in ("model_key", "seed", "sampler", "objective", "step",
                      "represented_ap", "transfer_ap"):
            if field not in row:
                raise ContractError(f"candidate score row missing {field}")
        if str(row["objective"]) not in OBJECTIVES:
            raise ContractError(f"unknown objective {row['objective']!r}")
        if str(row["sampler"]) not in SAMPLERS:
            raise ContractError(f"unknown sampler {row['sampler']!r}")
        grouped[cell_key(row)].append(row)
    return {key: trajectory(points) for key, points in grouped.items()}


def paired_frontier_gaps(
    trajectories: Mapping[tuple[str, int, str, str], Sequence[tuple[float, float]]],
    *,
    treatment: str,
    control: str,
) -> list[dict]:
    """One matched-represented transfer gap per (model, seed, sampler).

    Pairing is exact: the two objectives being contrasted share the model, the seed, the sampler,
    the Stage-1 initialisation and the scored rows. Only the objective differs.
    """
    if treatment not in OBJECTIVES or control not in OBJECTIVES:
        raise ContractError("treatment and control must be locked objectives")
    out: list[dict] = []
    for (model_key, seed, sampler, objective), curve in sorted(trajectories.items()):
        if objective != treatment:
            continue
        control_curve = trajectories.get((model_key, seed, sampler, control))
        if control_curve is None:
            out.append({"model_key": model_key, "seed": seed, "sampler": sampler,
                        "status": "missing_control", "gap": None})
            continue
        result = matched_transfer_gap(curve, control_curve)
        out.append({"model_key": model_key, "seed": seed, "sampler": sampler,
                    "treatment": treatment, "control": control, **result})
    return out


def factorial_marginal(gaps: Sequence[Mapping]) -> dict:
    """Equal-weight marginal over samplers, then over models -- the preregistered aggregation.

    Samplers are averaged first with equal weight so the selector cannot be chosen after the fact.
    Models are averaged second with equal weight so a single checkpoint cannot dominate the panel.
    Seeds are averaged inside a model, which is why the effective independent unit is the model.
    """
    usable = [row for row in gaps if row.get("status") == "ok" and row.get("gap") is not None]
    if not usable:
        return {"status": "unavailable", "value": None, "n_models": 0, "n_cells": 0}
    by_model_sampler: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in usable:
        by_model_sampler[(str(row["model_key"]), str(row["sampler"]))].append(float(row["gap"]))
    by_model: dict[str, list[float]] = defaultdict(list)
    for (model_key, _sampler), values in by_model_sampler.items():
        by_model[model_key].append(sum(values) / len(values))
    per_model = {model: sum(values) / len(values) for model, values in by_model.items()}
    value = sum(per_model.values()) / len(per_model)
    return {
        "status": "ok",
        "value": value,
        "per_model": dict(sorted(per_model.items())),
        "n_models": len(per_model),
        "n_cells": len(usable),
        "n_excluded": len(gaps) - len(usable),
    }


def cluster_bootstrap(
    gaps: Sequence[Mapping],
    *,
    replicates: int,
    seed: int,
) -> dict:
    """Paired bootstrap that resamples MODELS, not cells.

    Seeds inside a checkpoint share the model, the manifest and the data order, so they are not
    independent draws. Resampling cells would understate the interval by roughly sqrt(5). We
    therefore resample the four model clusters with replacement and recompute the marginal.
    """
    usable = [row for row in gaps if row.get("status") == "ok" and row.get("gap") is not None]
    if not usable:
        return {"status": "unavailable", "point": None}
    models = sorted({str(row["model_key"]) for row in usable})
    if len(models) < 2:
        return {"status": "too_few_clusters", "point": factorial_marginal(usable)["value"],
                "n_models": len(models)}
    by_model: dict[str, list[Mapping]] = defaultdict(list)
    for row in usable:
        by_model[str(row["model_key"])].append(row)
    state = int(seed) & 0xFFFFFFFF
    def _next_index(bound: int) -> int:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF  # deterministic, stdlib-free
        return state % bound
    draws: list[float] = []
    for _ in range(int(replicates)):
        picked: list[Mapping] = []
        for _ in models:
            picked.extend(by_model[models[_next_index(len(models))]])
        marginal = factorial_marginal(picked)
        if marginal["status"] == "ok":
            draws.append(float(marginal["value"]))
    if not draws:
        return {"status": "unavailable", "point": None}
    draws.sort()
    def _quantile(fraction: float) -> float:
        position = fraction * (len(draws) - 1)
        low = int(math.floor(position))
        high = min(low + 1, len(draws) - 1)
        return draws[low] + (position - low) * (draws[high] - draws[low])
    point = factorial_marginal(usable)
    return {
        "status": "ok",
        "point": point["value"],
        "per_model": point["per_model"],
        "n_models": len(models),
        "n_cells": len(usable),
        "replicates": len(draws),
        "ci_lower_95": _quantile(0.025),
        "ci_upper_95": _quantile(0.975),
        "lcb_one_sided_975": _quantile(0.025),
        "fraction_positive": sum(1 for value in draws if value > 0) / len(draws),
    }


def frontier_report(
    rows: Sequence[Mapping],
    *,
    replicates: int,
    seed: int,
) -> dict:
    """The full frontier analysis: F_ref (centering), F_pair (pairing), F_total.

    F_ref is primary. It is the transfer gap at matched represented AP between DPO and PairCE, so
    it is invariant to the effective-learning-rate difference that reference centering induces at
    initialisation. F_pair and F_total are reported for the same decomposition the point estimands
    use.
    """
    trajectories = build_trajectories(rows)
    contrasts = {
        "F_ref": ("dpo", "pair_ce"),
        "F_pair": ("pair_ce", "verdict_ce"),
        "F_total": ("dpo", "verdict_ce"),
    }
    out: dict = {"frontier_version": FRONTIER_VERSION, "grid_points": GRID_POINTS,
                 "min_overlap": MIN_OVERLAP, "n_cells": len(trajectories), "contrasts": {}}
    for name, (treatment, control) in contrasts.items():
        gaps = paired_frontier_gaps(trajectories, treatment=treatment, control=control)
        out["contrasts"][name] = {
            "treatment": treatment,
            "control": control,
            "per_cell": gaps,
            "bootstrap": cluster_bootstrap(gaps, replicates=replicates, seed=seed),
        }
    return out
