"""Tests for the operating-region metrics (partial AUC, TPR at a false-alarm budget).

Separate file mirroring the separate module: `guard_research/metrics.py` is sealed into Paper
A's release contract, so these live beside it rather than inside it.
"""

import numpy as np
import pytest

from guard_research.operating_point import LOW_FPR_MAX, partial_auc, tpr_at_fpr


def _perm_within_ties(scores, labels, rng):
    """Permute row order within each tied-score group (same helper as test_metrics)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, float).copy()
    for v in np.unique(s):
        idx = np.flatnonzero(s == v)
        y[idx] = y[rng.permutation(idx)]
    return s, y


# --------------------------------------------------------------- low-FPR region metrics
# These back the report's deployment claims, which are all stated at a false-alarm budget.
# macro-AP is an average over the whole ranking; pAUC and TPR@FPR are the region a guard is
# actually placed in, so they need their own correctness tests -- especially around ties,
# which is where a guard with clustered scores can be silently credited with an operating
# point it cannot realise.

def test_partial_auc_matches_sklearn_area_before_standardization():
    """Our unnormalized pAUC must equal the area sklearn standardizes.

    sklearn.roc_auc_score(max_fpr=m) returns McClish's *standardized* value; inverting that
    map recovers the raw area, which is what we compute. Agreement pins our truncation and
    interpolation to a reference implementation rather than to our own arithmetic.
    """
    from sklearn.metrics import roc_auc_score


    rng = np.random.default_rng(11)
    y = np.r_[np.ones(300), np.zeros(1200)]
    for shift in (0.5, 1.5, 3.0):
        s = np.r_[rng.normal(shift, 1, 300), rng.normal(0, 1, 1200)]
        for m in (0.01, 0.05, 0.2):
            std = roc_auc_score(y, s, max_fpr=m)          # McClish: 0.5*(1 + (A - Amin)/(Amax - Amin))
            a_min, a_max = m * m / 2.0, m
            area = (2.0 * std - 1.0) * (a_max - a_min) + a_min
            assert partial_auc(s, y, max_fpr=m, normalize=False) == pytest.approx(area, abs=1e-9)


def test_partial_auc_floor_and_ceiling():
    """Chance is max_fpr/2 normalized (NOT 0.5); a perfect ranker is 1.0."""

    rng = np.random.default_rng(3)
    y = np.r_[np.ones(400), np.zeros(4000)]
    perfect = np.r_[np.full(400, 10.0), rng.normal(0, 1, 4000)]
    assert partial_auc(perfect, y, max_fpr=0.05) == pytest.approx(1.0, abs=1e-9)

    chance = np.concatenate([rng.normal(0, 1, 400), rng.normal(0, 1, 4000)])
    assert partial_auc(chance, y, max_fpr=0.05) == pytest.approx(0.025, abs=0.02)


def test_tpr_at_fpr_is_conservative_under_ties():
    """A tied score block must not be split in the guard's favour.

    Ten negatives and ten positives share one score. Any threshold that admits the positives
    admits all ten negatives too -- FPR 1.0 -- so at a 5% budget the realisable recall is 0.
    An implementation that interpolated inside the tie block would report ~0.5 here, which is
    an operating point these scores cannot produce. This is the frontier model's coarse
    integer risk in miniature.
    """

    y = np.r_[np.ones(10), np.zeros(10)]
    s = np.zeros(20)
    assert tpr_at_fpr(s, y, max_fpr=0.05) == 0.0


def test_tpr_at_fpr_exact_step():
    """With clean separation on a fraction of positives, recall is that exact fraction."""

    # 100 negatives at 0; 30 positives at +1 (clearly above), 70 positives at -1 (below).
    y = np.r_[np.ones(100), np.zeros(100)]
    s = np.r_[np.full(30, 1.0), np.full(70, -1.0), np.zeros(100)]
    assert tpr_at_fpr(s, y, max_fpr=0.05) == pytest.approx(0.30, abs=1e-9)


def test_low_fpr_metrics_are_permutation_invariant_within_ties():
    """Same regression guard as AP: tied rows must not depend on row order."""

    rng = np.random.default_rng(7)
    s = rng.integers(0, 6, 600).astype(float)      # heavy ties, like an integer risk score
    y = (rng.random(600) < 0.3).astype(float)
    p0, t0 = partial_auc(s, y), tpr_at_fpr(s, y)
    for _ in range(20):
        s2, y2 = _perm_within_ties(s, y, rng)
        assert partial_auc(s2, y2) == pytest.approx(p0, abs=1e-12)
        assert tpr_at_fpr(s2, y2) == pytest.approx(t0, abs=1e-12)


def test_low_fpr_metrics_single_class_and_bad_budget():

    assert np.isnan(partial_auc([1.0, 2.0], [1, 1]))
    assert np.isnan(tpr_at_fpr([1.0, 2.0], [0, 0]))
    assert np.isnan(partial_auc([], []))
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            partial_auc([1.0, 0.0], [1, 0], max_fpr=bad)
        with pytest.raises(ValueError):
            tpr_at_fpr([1.0, 0.0], [1, 0], max_fpr=bad)


def test_partial_auc_monotone_in_budget():
    """Widening the budget cannot reduce the unnormalized area."""

    rng = np.random.default_rng(5)
    y = np.r_[np.ones(200), np.zeros(800)]
    s = np.r_[rng.normal(1.2, 1, 200), rng.normal(0, 1, 800)]
    areas = [partial_auc(s, y, max_fpr=m, normalize=False) for m in (0.01, 0.02, 0.05, 0.1, 0.5)]
    assert all(b >= a - 1e-12 for a, b in zip(areas, areas[1:]))
