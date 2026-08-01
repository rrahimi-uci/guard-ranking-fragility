"""Operating-region metrics: partial AUC and TPR at a false-alarm budget.

A deliberately SEPARATE module from ``guard_research.metrics``, and the separation is a
provenance constraint rather than a style choice. ``metrics.py`` is one of the six files
hashed into Paper A's release contract (``RELEASE_CACHE_SOURCE_FILES`` in
``experiments/paper_a_common.py``): its exact bytes are committed inside
``artifacts/paper_a_sft_v2/`` as the definition of "the metrics this release was computed
with". Adding a function to it -- even a purely additive one that changes no existing value --
invalidates that commitment and makes the release analyzer fail closed, which is the lock
doing its job. New metrics therefore live beside the sealed module, not inside it.

Why these metrics exist. The report's primary quantity is macro-AP, an average of precision
over the *whole* ranking. Every deployment sentence it writes is about a guard that fires on a
few percent of traffic. AP rewards ordering deep in the negative mass, below any threshold an
inline guard would use, so a change in AP need not imply the same change where the guard is
actually placed -- and on this repository's own panel the two disagree by a factor of two to
three (see ``papers/unified-report/low_fpr.py``).

Both functions take 0/1 ``labels`` and scores where higher means more "unsafe", coerce like
``guard_research.metrics``, and return ``float('nan')`` on single-class input.
"""

from __future__ import annotations

import numpy as np

__all__ = ["tpr_at_fpr", "partial_auc", "LOW_FPR_MAX"]

# The deployment operating region this report argues about. Every low-FPR metric here defaults
# to it so a caller cannot silently evaluate a different budget than the matched-FPR tables use.
LOW_FPR_MAX = 0.05


def _coerce(scores, labels):
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)


def _single_class(y: np.ndarray) -> bool:
    return y.size == 0 or float(y.min()) == float(y.max())


def _roc(scores, labels):
    """Tie-correct ROC points, ascending in FPR, anchored at (0,0) and (1,1).

    ``sklearn.metrics.roc_curve`` already groups tied scores into one threshold, which is the
    property that matters here: a guard whose negatives pile up on a handful of distinct values
    (the hosted frontier's integer risk, or a saturating small guard) must not have those ties
    silently broken in its favour.
    """
    from sklearn.metrics import roc_curve

    s, y = _coerce(scores, labels)
    ok = ~(np.isnan(s) | np.isnan(y))
    s, y = s[ok], y[ok]
    if _single_class(y):
        return None
    fpr, tpr, _ = roc_curve(y, s)
    return np.asarray(fpr, float), np.asarray(tpr, float)


def tpr_at_fpr(scores, labels, max_fpr: float = LOW_FPR_MAX) -> float:
    """Recall at the largest ROC point whose false-alarm rate does not exceed ``max_fpr``.

    This is the *conservative* reading of an alarm budget: with tied scores the ROC is a step
    function, so demanding FPR <= budget can land strictly below it. Interpolating between ROC
    vertices instead would credit the guard with an operating point it cannot actually realise
    on these scores, which is precisely the error the report's tie discussion warns about for
    the frontier's coarse integer risk.

    Returns nan when the labels are single-class. Undefined budgets (outside (0, 1]) raise.
    """
    if not 0.0 < float(max_fpr) <= 1.0:
        raise ValueError(f"max_fpr must be in (0, 1], got {max_fpr!r}")
    roc = _roc(scores, labels)
    if roc is None:
        return float("nan")
    fpr, tpr = roc
    within = fpr <= float(max_fpr) + 1e-12
    return float(tpr[within].max()) if within.any() else 0.0


def partial_auc(scores, labels, max_fpr: float = LOW_FPR_MAX, normalize: bool = True) -> float:
    """One-way partial AUC over the FPR range ``[0, max_fpr]``.

    Why this exists, stated once. The report's primary metric is macro-AP, an average over the
    whole ranking, while every deployment claim it makes is about recall at a 5% false-alarm
    budget. Those are different quantities, and a method can move them in opposite directions:
    AP rewards ordering deep in the negative mass, where a guard that fires on 5% of traffic
    never operates. pAUC integrates the ROC only over the region an inline guard can actually
    be placed in.

    The curve is truncated exactly at ``max_fpr`` by linear interpolation between the two ROC
    vertices that straddle it -- the standard McClish construction, and the same one
    ``sklearn.roc_auc_score(max_fpr=...)`` uses -- so the value does not jump when a single
    negative crosses the budget.

    ``normalize=True`` (default) rescales by dividing by ``max_fpr``, which makes the value the
    *mean TPR over the region* -- comparable across budgets and on the same scale as a recall.

    Read it against the right floor. A random ranker has TPR = FPR, so its unnormalized pAUC is
    ``max_fpr**2 / 2`` and its normalized pAUC is ``max_fpr / 2`` -- **0.025** at the default 5%
    budget, not 0.5. This deliberately is NOT McClish standardization (which rescales chance to
    0.5): McClish's map inflates small differences near the floor, and the report's convention
    everywhere else is to state a metric beside its own chance level rather than to rescale it.

    Returns nan when the labels are single-class.
    """
    if not 0.0 < float(max_fpr) <= 1.0:
        raise ValueError(f"max_fpr must be in (0, 1], got {max_fpr!r}")
    roc = _roc(scores, labels)
    if roc is None:
        return float("nan")
    fpr, tpr = roc
    m = float(max_fpr)

    keep = fpr <= m
    x, y = fpr[keep], tpr[keep]
    if x.size == 0 or x[0] > 0.0:                      # always anchor the origin
        x, y = np.concatenate([[0.0], x]), np.concatenate([[0.0], y])
    if x[-1] < m:                                      # interpolate the truncation point
        nxt = np.searchsorted(fpr, m, side="left")
        if nxt < fpr.size and fpr[nxt] > x[-1]:
            frac = (m - x[-1]) / (fpr[nxt] - x[-1])
            y_at_m = y[-1] + frac * (tpr[nxt] - y[-1])
        else:
            y_at_m = y[-1]                             # curve ends before the budget
        x, y = np.concatenate([x, [m]]), np.concatenate([y, [y_at_m]])

    area = float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))
    return area / m if normalize else area
