"""Canonical evaluation metrics for Paper A.

THESE ARE THE SINGLE CANONICAL IMPLEMENTATIONS. Every script, notebook, test,
and analysis step must import average precision and AUROC from this module.

Rationale (paper's #1 correctness requirement): average precision must be
*tie-aware* and *non-interpolated*. An order-dependent custom AP loop -- one
that sorts by score and silently breaks ties by row position -- produces
metrics that change when tied rows are permuted. A reviewer found several
legacy scripts using such loops. `sklearn.metrics.average_precision_score`
groups equal scores correctly, so we wrap it and forbid hand-rolled AP loops
anywhere else in the codebase.

All functions accept python lists or numpy arrays and coerce with
``np.asarray(..., dtype=float)``. Single-class inputs (all 0 or all 1 labels,
or empty) return ``float('nan')`` for ranking metrics because AP/AUROC are
undefined without both classes present.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = ["average_precision", "auroc", "brier", "log_loss_"]


def _coerce(scores, labels):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    return s, y


def _single_class(y: np.ndarray) -> bool:
    """True when AP/AUROC are undefined: empty, or only one label present."""
    return y.size == 0 or float(y.min()) == float(y.max())


def average_precision(scores, labels) -> float:
    """Tie-aware, non-interpolated average precision.

    Thin wrapper over ``sklearn.metrics.average_precision_score`` (the canonical
    tie-aware AP). ``labels`` are 0/1; higher ``scores`` mean more "unsafe".
    Returns ``float('nan')`` if the labels are single-class (all 0 or all 1).
    """
    s, y = _coerce(scores, labels)
    if _single_class(y):
        return float("nan")
    return float(average_precision_score(y, s))


def auroc(scores, labels) -> float:
    """Area under the ROC curve.

    Thin wrapper over ``sklearn.metrics.roc_auc_score`` (tie-correct: it uses
    the trapezoidal/mid-rank treatment of ties). Returns ``float('nan')`` if the
    labels are single-class.
    """
    s, y = _coerce(scores, labels)
    if _single_class(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def brier(probs, labels) -> float:
    """Brier score: mean squared error between calibrated probabilities and
    0/1 labels. Defined for any label composition; returns nan for empty input."""
    p, y = _coerce(probs, labels)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def log_loss_(probs, labels, eps: float = 1e-12) -> float:
    """Binary log loss (negative log-likelihood) with probability clipping.

    Trailing underscore avoids shadowing ``sklearn.metrics.log_loss``. Computed
    directly so it is well-defined for single-class calibration slices (which
    sklearn's label inference otherwise complains about). Returns nan for empty
    input."""
    p, y = _coerce(probs, labels)
    if p.size == 0:
        return float("nan")
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
