"""Tests for the conservative Clopper-Pearson operating point (plan sec 13.4
test_thresholds).

Covers: test labels cannot change the calibration threshold; the selected
threshold is optimal among candidates satisfying the conservative bound;
infeasible targets return NO_FEASIBLE_THRESHOLD; tied-score thresholds are
handled consistently.
"""

import inspect

import numpy as np
import pytest
from scipy.stats import beta

from guard_research.thresholds import clopper_pearson_upper, select_threshold


def test_clopper_pearson_edge_cases():
    # tn == 0 -> 1.0 (no informative negatives, or every negative is an FP)
    assert clopper_pearson_upper(0, 0) == 1.0
    assert clopper_pearson_upper(5, 0) == 1.0
    # fp == 0 -> closed form 1 - (1-conf)^(1/tn); equals the Beta(1, tn) quantile
    for tn in (1, 10, 59, 100, 1000):
        cf = clopper_pearson_upper(0, tn)
        assert cf == pytest.approx(1.0 - 0.05 ** (1.0 / tn))
        assert cf == pytest.approx(beta.ppf(0.95, 1, tn), rel=1e-9)
    # fp > 0 -> matches scipy Beta quantile exactly
    for fp, tn in [(1, 50), (3, 97), (10, 200)]:
        assert clopper_pearson_upper(fp, tn) == pytest.approx(beta.ppf(0.95, fp + 1, tn))
    # monotone in FP (fixed tn) and tighter with larger n at fp=0
    ub = [clopper_pearson_upper(fp, 100) for fp in range(0, 20)]
    assert all(ub[i] <= ub[i + 1] + 1e-12 for i in range(len(ub) - 1))
    assert clopper_pearson_upper(0, 60) < clopper_pearson_upper(0, 30)


def _feasible_candidates(scores, labels, target):
    """Independent re-derivation of the sec 10.5 candidate table."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    neg = y == 0
    pos = y == 1
    n_neg = int(neg.sum())
    n_pos = int(pos.sum())
    rows = []
    for thr in list(np.unique(s)) + [np.inf]:
        fp = int(np.count_nonzero(s[neg] >= thr))
        tp = int(np.count_nonzero(s[pos] >= thr))
        tn = n_neg - fp
        fn = n_pos - tp
        cp = clopper_pearson_upper(fp, tn)
        rows.append(
            {
                "thr": float(thr),
                "fp": fp,
                "tn": tn,
                "tp": tp,
                "fn": fn,
                "cp": cp,
                "recall": tp / n_pos if n_pos else 0.0,
                "fpr": fp / n_neg if n_neg else 0.0,
                "feasible": cp <= target,
            }
        )
    return rows


def test_selected_threshold_is_optimal():
    rng = np.random.default_rng(0)
    checked_feasible = 0
    checked_infeasible = 0
    for _ in range(80):
        n = int(rng.integers(120, 400))
        y = rng.integers(0, 2, size=n)
        s = rng.normal(loc=y * 1.2, scale=1.0)  # positives shifted higher
        if y.min() == y.max():
            continue
        out = select_threshold(s, y, target_fpr=0.05)
        feas = [c for c in _feasible_candidates(s, y, 0.05) if c["feasible"]]
        if not feas:
            assert out == {"status": "NO_FEASIBLE_THRESHOLD"}
            checked_infeasible += 1
            continue
        checked_feasible += 1
        assert "status" not in out
        # the returned point satisfies the conservative bound
        assert out["cal_fpr_upper"] <= 0.05 + 1e-12
        # optimality: the selected (recall, -fpr, thr) key is the maximum over
        # all feasible candidates (thresholds are unique -> unique maximizer)
        best = max((c["recall"], -c["fpr"], c["thr"]) for c in feas)
        sel = (out["cal_recall"], -out["cal_fpr_point"], out["threshold"])
        assert sel == best
        # confusion counts agree with the brute-force winner
        winner = max(feas, key=lambda c: (c["recall"], -c["fpr"], c["thr"]))
        assert (out["fp"], out["tn"], out["cal_tp"], out["fn"]) == (
            winner["fp"],
            winner["tn"],
            winner["tp"],
            winner["fn"],
        )
        if np.isfinite(winner["thr"]):
            assert out["threshold"] == pytest.approx(winner["thr"])
        else:
            assert np.isinf(out["threshold"])
    assert checked_feasible > 0  # the optimality path actually executed


def test_infeasible_target_returns_status():
    # (a) target 0.0 is unattainable: even fp=0 gives a strictly positive bound
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert select_threshold(s, y, target_fpr=0.0) == {"status": "NO_FEASIBLE_THRESHOLD"}
    # (b) too few negatives (10) to certify 5% even with zero observed FP
    s = np.concatenate([np.linspace(0.0, 1.0, 10), np.linspace(0.5, 1.5, 20)])
    y = np.concatenate([np.zeros(10), np.ones(20)])
    assert select_threshold(s, y, target_fpr=0.05) == {
        "status": "NO_FEASIBLE_THRESHOLD"
    }


def test_tied_scores_handled_consistently():
    """Rows tied at a score are classified together, so the result is invariant
    to permutations that only reorder tied rows (indeed any permutation)."""
    scores = np.concatenate([np.zeros(80), np.full(60, 0.5), np.ones(80)])
    # both labels present within every tie group; balanced -> feasible at 5%
    labels = np.concatenate(
        [np.tile([0, 1], 40), np.tile([0, 1], 30), np.tile([0, 1], 40)]
    )
    base = select_threshold(scores, labels, target_fpr=0.05)
    assert "status" not in base  # a real finite/inf threshold was selected
    for k in range(50):
        p = np.random.default_rng(300 + k).permutation(scores.size)
        assert select_threshold(scores[p], labels[p], target_fpr=0.05) == base


def test_only_calibration_inputs_used():
    """Test labels cannot change the threshold: the API structurally has no
    test-data parameter, and selection is a pure function of its inputs."""
    params = list(inspect.signature(select_threshold).parameters)
    assert params == ["cal_scores", "cal_labels", "target_fpr"]
    s = np.concatenate([np.linspace(0.0, 0.4, 70), np.linspace(0.6, 1.0, 70)])
    y = np.concatenate([np.zeros(70), np.ones(70)])
    a = select_threshold(s, y, 0.05)
    b = select_threshold(s, y, 0.05)
    assert a == b
