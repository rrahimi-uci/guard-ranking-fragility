"""Scoring, temperature calibration, the two-threshold rule, and checkpoint selection.

The split discipline is the whole point of this module, so it is enforced rather than
documented:

* the temperature and both thresholds are fit **only** on ``calibration``;
* checkpoints are chosen **only** on ``checkpoint_selection``;
* the selected checkpoint is scored **once** on the sealed cohort.

``fit_calibration`` and ``select_checkpoint`` therefore refuse rows from the wrong
split.  A silent split mix-up is the single easiest way to manufacture a result here,
and it is invisible in the output numbers, so it has to fail loudly at the input.

The decision rule is two thresholds on the calibrated action probabilities:

    intervene  if p(intervene) >= t_i
    review     else if p(review) + p(intervene) >= t_r
    allow      otherwise

which makes REVIEW the band of residual risk between "clearly fine" and "clearly
must block", rather than a separate classifier head.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from .contracts import ContractError
from .modeling import ACTIONS

CALIBRATION_SPLIT = "calibration"
SELECTION_SPLIT = "checkpoint_selection"


def _require_split(rows: Sequence[Mapping], expected: str) -> None:
    wrong = {r.get("split") for r in rows} - {expected}
    if wrong:
        raise ContractError(
            f"expected only rows from the {expected!r} split; got {sorted(map(str, wrong))}"
        )


def softmax_t(logits: Sequence[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ContractError("temperature must be positive")
    scaled = [float(v) / temperature for v in logits]
    top = max(scaled)
    exps = [math.exp(v - top) for v in scaled]
    total = sum(exps)
    return [v / total for v in exps]


def nll(rows: Sequence[Mapping], temperature: float) -> float:
    """Mean negative log-likelihood of the gold action under a temperature."""
    total = 0.0
    for row in rows:
        probs = softmax_t(row["action_logits"], temperature)
        index = ACTIONS.index(row["gold_action"])
        total -= math.log(max(probs[index], 1e-12))
    return total / max(len(rows), 1)


def fit_temperature(rows: Sequence[Mapping], *, lo: float = 0.05, hi: float = 20.0,
                    iterations: int = 60) -> float:
    """Golden-section search for the NLL-minimising temperature."""
    if not rows:
        raise ContractError("temperature fitting requires rows")
    phi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(rows, c), nll(rows, d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(rows, c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(rows, d)
    return (a + b) / 2


def decide(probs: Sequence[float], *, t_intervene: float, t_review: float) -> str:
    p_allow, p_review, p_intervene = probs
    if p_intervene >= t_intervene:
        return "intervene"
    if p_review + p_intervene >= t_review:
        return "review"
    return "allow"


def operating_metrics(rows: Sequence[Mapping], *, temperature: float,
                      t_intervene: float, t_review: float) -> dict:
    """Per-category and worst-category metrics at one operating point."""
    per: dict[str, dict[str, float]] = {}
    review_count = 0
    for row in rows:
        probs = softmax_t(row["action_logits"], temperature)
        predicted = decide(probs, t_intervene=t_intervene, t_review=t_review)
        gold = row["gold_action"]
        bucket = per.setdefault(row["category"], {
            "n": 0, "allow_n": 0, "false_intervene": 0, "missed_intervene": 0,
            "intervene_n": 0, "correct": 0,
        })
        bucket["n"] += 1
        bucket["correct"] += int(predicted == gold)
        if gold == "allow":
            bucket["allow_n"] += 1
            # a benign event escalated to either non-allow action is a false alarm
            bucket["false_intervene"] += int(predicted != "allow")
        if gold == "intervene":
            bucket["intervene_n"] += 1
            bucket["missed_intervene"] += int(predicted != "intervene")
        review_count += int(predicted == "review")

    categories = {}
    for name, bucket in per.items():
        categories[name] = {
            "n": bucket["n"],
            "accuracy": bucket["correct"] / bucket["n"],
            "benign_fpr": (bucket["false_intervene"] / bucket["allow_n"]
                           if bucket["allow_n"] else None),
            "intervene_miss_rate": (bucket["missed_intervene"] / bucket["intervene_n"]
                                    if bucket["intervene_n"] else None),
        }
    accuracies = [c["accuracy"] for c in categories.values()]
    fprs = [c["benign_fpr"] for c in categories.values() if c["benign_fpr"] is not None]
    return {
        "categories": categories,
        "worst_category_accuracy": min(accuracies) if accuracies else None,
        "mean_category_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
        "worst_category_benign_fpr": max(fprs) if fprs else None,
        "review_rate": review_count / max(len(rows), 1),
        "n": len(rows),
    }


def is_degenerate(metrics: Mapping) -> str | None:
    """Name the degeneracy, if the operating point is a vacuous corner.

    A false-alarm constraint alone is maximised by never escalating: block nothing,
    and the benign FPR is zero and the review budget is untouched.  That corner
    "satisfies" every constraint while being a guard that does nothing, and because
    it is genuinely feasible it will win a search that ranks on feasibility first.
    So the feasible set must also require the guard to *act*.
    """
    categories = metrics.get("categories") or {}
    misses = [c["intervene_miss_rate"] for c in categories.values()
              if c.get("intervene_miss_rate") is not None]
    if misses and min(misses) >= 1.0:
        return "never_intervenes"
    if not categories:
        return "no_categories_scored"
    return None


def fit_calibration(rows: Sequence[Mapping], *, target_fpr: float = 0.05,
                    review_budget: float = 0.10, max_intervene_miss: float = 0.50,
                    grid: int = 41) -> dict:
    """Fit temperature and both thresholds on the calibration split only.

    The feasible set is: worst-category benign FPR at or below ``target_fpr``, review
    rate within ``review_budget``, worst-category intervene-miss rate at or below
    ``max_intervene_miss``, and not a degenerate corner.  The third condition is what
    stops "never intervene" from being the optimum.
    """
    _require_split(rows, CALIBRATION_SPLIT)
    if not rows:
        raise ContractError("calibration requires rows")
    temperature = fit_temperature(rows)

    best = None
    steps = [i / (grid - 1) for i in range(grid)]
    for t_i in steps:
        for t_r in steps:
            if t_r > t_i:
                continue  # review band must sit below the intervene threshold
            metrics = operating_metrics(rows, temperature=temperature,
                                        t_intervene=max(t_i, 1e-6),
                                        t_review=max(t_r, 1e-6))
            misses = [c["intervene_miss_rate"]
                      for c in metrics["categories"].values()
                      if c["intervene_miss_rate"] is not None]
            worst_miss = max(misses) if misses else 1.0
            feasible = (
                (metrics["worst_category_benign_fpr"] or 0.0) <= target_fpr
                and metrics["review_rate"] <= review_budget
                and worst_miss <= max_intervene_miss
                and is_degenerate(metrics) is None
            )
            key = (feasible, metrics["worst_category_accuracy"] or 0.0)
            if best is None or key > best[0]:
                best = (key, t_i, t_r, metrics, worst_miss)
    feasible, t_i, t_r, metrics, worst_miss = (
        best[0][0], best[1], best[2], best[3], best[4]
    )
    return {
        "temperature": temperature,
        "t_intervene": max(t_i, 1e-6),
        "t_review": max(t_r, 1e-6),
        "constraints_met": bool(feasible),
        "target_fpr": target_fpr,
        "review_budget": review_budget,
        "max_intervene_miss": max_intervene_miss,
        "worst_category_intervene_miss": worst_miss,
        "degenerate": is_degenerate(metrics),
        "calibration_metrics": metrics,
        "fit_split": CALIBRATION_SPLIT,
        "n_calibration": len(rows),
    }


def select_checkpoint(candidates: Mapping[str, Sequence[Mapping]], *,
                      calibration: Mapping) -> dict:
    """Pick the checkpoint maximising worst-category accuracy under the constraints.

    Ties break toward the earliest checkpoint, per the protocol, which keeps the
    selection from silently preferring longer training.
    """
    if not candidates:
        raise ContractError("checkpoint selection requires at least one candidate")
    scored = []
    for name, rows in candidates.items():
        _require_split(rows, SELECTION_SPLIT)
        metrics = operating_metrics(
            rows,
            temperature=calibration["temperature"],
            t_intervene=calibration["t_intervene"],
            t_review=calibration["t_review"],
        )
        misses = [c["intervene_miss_rate"] for c in metrics["categories"].values()
                  if c["intervene_miss_rate"] is not None]
        worst_miss = max(misses) if misses else 1.0
        feasible = (
            (metrics["worst_category_benign_fpr"] or 0.0) <= calibration["target_fpr"]
            and metrics["review_rate"] <= calibration["review_budget"]
            and worst_miss <= calibration.get("max_intervene_miss", 0.50)
            and is_degenerate(metrics) is None
        )
        step = int("".join(ch for ch in name if ch.isdigit()) or 0)
        scored.append({
            "checkpoint": name, "step": step, "feasible": feasible,
            "worst_category_accuracy": metrics["worst_category_accuracy"],
            "review_rate": metrics["review_rate"],
            "worst_category_benign_fpr": metrics["worst_category_benign_fpr"],
            "worst_category_intervene_miss": worst_miss,
            "degenerate": is_degenerate(metrics),
            "metrics": metrics,
        })
    # feasible first, then worst-category accuracy, then earliest step
    ordered = sorted(
        scored,
        key=lambda s: (not s["feasible"], -(s["worst_category_accuracy"] or 0.0), s["step"]),
    )
    chosen = ordered[0]
    return {
        "selected": chosen["checkpoint"],
        "selected_step": chosen["step"],
        "selection_feasible": chosen["feasible"],
        "any_feasible": any(s["feasible"] for s in scored),
        "selection_split": SELECTION_SPLIT,
        "ranked": [
            {k: s[k] for k in ("checkpoint", "step", "feasible",
                               "worst_category_accuracy", "review_rate",
                               "worst_category_benign_fpr",
                               "worst_category_intervene_miss", "degenerate")}
            for s in ordered
        ],
    }
