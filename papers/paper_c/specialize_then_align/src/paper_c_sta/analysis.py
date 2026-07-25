"""The primary contrast, its interval, and the Holm-corrected secondaries.

The estimand is

    specialist_cm_dpo  -  generalist_cm_dpo

on worst-category accuracy over the five focal categories, evaluated once on the
sealed cohort at each arm's own selected checkpoint.  Because the objective, the
reference, the source events, the optimizer, the ladder and the token budget are all
matched, what remains is the incremental value of category-specialist candidate
generation -- not of DPO, and not of preference learning in general.

Three properties this module enforces because they are the ones easiest to get
quietly wrong:

*Pairing.*  Arms are compared on identical scenario families, so family difficulty
cancels.  ``paired_family_contrast`` refuses inputs whose family sets differ.

*The resampling unit is the family, not the row.*  Rows within a family are
near-duplicates; resampling rows would understate the interval.  Inference is also
conditional on the two named backbones and three named seeds -- across-seed spread is
reported as descriptive dispersion, never as a population variance.

*Multiplicity.*  Secondary contrasts get Holm; the primary does not, because it was
named in advance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from .contracts import ContractError

PRIMARY = ("specialist_cm_dpo", "generalist_cm_dpo")
SECONDARY = (
    ("specialist_cm_dpo", "specialist_pairce"),    # reference centering
    ("specialist_pairce", "gold_sft"),             # pairwise learning
    ("specialist_cm_dpo", "gold_sft"),             # total method effect
    ("soft_distill", "gold_sft"),                  # distillation vs gold
)


def _lcg(seed: int):
    """Deterministic PRNG: results must not depend on interpreter RNG state."""
    state = seed & 0xFFFFFFFF

    def draw(bound: int) -> int:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state % bound

    return draw


def worst_category_accuracy(rows: Sequence[Mapping]) -> float:
    per: dict[str, list[int]] = {}
    for row in rows:
        per.setdefault(row["category"], []).append(int(row["predicted"] == row["gold_action"]))
    if not per:
        raise ContractError("no rows to score")
    return min(sum(v) / len(v) for v in per.values())


def _family_index(rows: Sequence[Mapping]) -> dict[str, list[Mapping]]:
    out: dict[str, list[Mapping]] = {}
    for row in rows:
        out.setdefault(row["family_id"], []).append(row)
    return out


def paired_family_contrast(left: Sequence[Mapping], right: Sequence[Mapping], *,
                           resamples: int = 2000, seed: int = 20260725,
                           alpha: float = 0.05) -> dict:
    """Paired difference in worst-category accuracy, resampling families."""
    left_by, right_by = _family_index(left), _family_index(right)
    if set(left_by) != set(right_by):
        only_left = sorted(set(left_by) - set(right_by))[:3]
        only_right = sorted(set(right_by) - set(left_by))[:3]
        raise ContractError(
            f"paired contrast requires identical family sets; "
            f"only-left={only_left} only-right={only_right}"
        )
    families = sorted(left_by)
    if len(families) < 2:
        raise ContractError("a family bootstrap needs at least two families")
    observed = worst_category_accuracy(left) - worst_category_accuracy(right)

    draw = _lcg(seed)
    deltas = []
    for _ in range(resamples):
        picked = [families[draw(len(families))] for _ in range(len(families))]
        l_rows = [r for f in picked for r in left_by[f]]
        r_rows = [r for f in picked for r in right_by[f]]
        try:
            deltas.append(worst_category_accuracy(l_rows) - worst_category_accuracy(r_rows))
        except ContractError:
            continue
    deltas.sort()
    if not deltas:
        raise ContractError("every bootstrap resample was degenerate")
    lo = deltas[max(0, int((alpha / 2) * len(deltas)) - 1)]
    hi = deltas[min(len(deltas) - 1, int((1 - alpha / 2) * len(deltas)))]
    # two-sided bootstrap p: how often the resampled difference crosses zero
    crossings = sum(1 for d in deltas if (d <= 0) if observed > 0) or \
        sum(1 for d in deltas if (d >= 0) if observed < 0)
    p_value = min(1.0, 2 * (crossings + 1) / (len(deltas) + 1))
    return {
        "observed": observed,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_value,
        "families": len(families),
        "resamples": len(deltas),
        "excludes_zero": (lo > 0) or (hi < 0),
        "resampling_unit": "scenario_family",
    }


def holm(p_values: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down: strong familywise control without assuming independence."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    n = len(ordered)
    out: dict[str, dict] = {}
    still_rejecting = True
    for rank, (name, p) in enumerate(ordered):
        threshold = alpha / (n - rank)
        if still_rejecting and p <= threshold:
            out[name] = {"p_value": p, "holm_threshold": threshold, "reject": True}
        else:
            still_rejecting = False
            out[name] = {"p_value": p, "holm_threshold": threshold, "reject": False}
    return out


def seed_dispersion(per_seed: Mapping[int, float]) -> dict:
    """Descriptive spread across the named seeds. Never a population variance."""
    values = list(per_seed.values())
    if len(values) < 2:
        return {"n_seeds": len(values), "mean": values[0] if values else None,
                "sd": None, "note": "fewer than two seeds; no dispersion"}
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return {
        "n_seeds": len(values), "mean": mean, "sd": sd,
        "min": min(values), "max": max(values),
        "note": "descriptive dispersion over the named seeds; not a seed-population estimate",
    }


def analyse_panel(scored: Mapping[str, Sequence[Mapping]], *, resamples: int = 2000,
                  seed: int = 20260725, alpha: float = 0.05) -> dict:
    """Primary contrast plus Holm-corrected secondaries over a scored panel."""
    missing = [a for a in PRIMARY if a not in scored]
    if missing:
        raise ContractError(f"the primary contrast needs both CM-DPO arms; missing {missing}")
    primary = paired_family_contrast(
        scored[PRIMARY[0]], scored[PRIMARY[1]],
        resamples=resamples, seed=seed, alpha=alpha,
    )
    secondaries, p_values = {}, {}
    for left, right in SECONDARY:
        if left not in scored or right not in scored:
            secondaries[f"{left}-{right}"] = {"status": "unavailable", "reason": "arm not scored"}
            continue
        try:
            result = paired_family_contrast(
                scored[left], scored[right], resamples=resamples, seed=seed, alpha=alpha)
        except ContractError as exc:
            secondaries[f"{left}-{right}"] = {"status": "refused", "reason": str(exc)}
            continue
        secondaries[f"{left}-{right}"] = result
        p_values[f"{left}-{right}"] = result["p_value"]
    return {
        "primary": {
            "contrast": f"{PRIMARY[0]} - {PRIMARY[1]}",
            "metric": "worst_category_accuracy",
            **primary,
            "multiplicity": "none; named in advance",
        },
        "secondary": secondaries,
        "secondary_holm": holm(p_values, alpha=alpha) if p_values else {},
        "inference_scope": (
            "conditional on the two named backbones and the named seeds; "
            "families are the resampling unit; no architecture- or seed-population claim"
        ),
    }
