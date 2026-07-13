"""Tests for the canonical tie-aware metrics (plan sec 13.4 test_metrics).

The headline test is permutation invariance within tied-score groups: this is
the regression guard for the order-dependent-AP bug a reviewer found in legacy
scripts.
"""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from guard_research.metrics import auroc, average_precision, brier, log_loss_


def _perm_within_ties(scores, labels, rng):
    """Permute row order within each tied-score group.

    All rows in a group share a score, so reordering them is equivalent to
    permuting the labels attached to those tied positions.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).copy()
    for v in np.unique(scores):
        idx = np.where(scores == v)[0]
        perm = idx.copy()
        rng.shuffle(perm)
        labels[idx] = labels[perm]
    return scores, labels


def test_average_precision_matches_sklearn_random():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(20, 200))
        scores = rng.normal(size=n)
        labels = rng.integers(0, 2, size=n)
        if labels.min() == labels.max():
            continue
        assert average_precision(scores, labels) == pytest.approx(
            average_precision_score(labels, scores)
        )


def test_average_precision_matches_sklearn_many_ties():
    rng = np.random.default_rng(1)
    for _ in range(25):
        n = int(rng.integers(50, 300))
        scores = rng.integers(0, 4, size=n).astype(float)  # heavy ties
        labels = rng.integers(0, 2, size=n)
        if labels.min() == labels.max():
            continue
        assert average_precision(scores, labels) == pytest.approx(
            average_precision_score(labels, scores)
        )


def test_auroc_matches_sklearn():
    rng = np.random.default_rng(2)
    for _ in range(25):
        n = int(rng.integers(30, 250))
        scores = rng.integers(0, 5, size=n).astype(float)  # ties present
        labels = rng.integers(0, 2, size=n)
        if labels.min() == labels.max():
            continue
        assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))


def test_accepts_python_lists():
    scores = [0.1, 0.1, 0.9, 0.9, 0.5]
    labels = [0, 1, 1, 0, 1]
    assert average_precision(scores, labels) == pytest.approx(
        average_precision_score(labels, scores)
    )
    assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))


def test_permutation_within_ties_invariant():
    """KEY regression test: permuting rows within tied-score groups must not
    change AP or AUROC. An order-dependent AP loop fails this."""
    rng = np.random.default_rng(3)
    scores = np.repeat(np.arange(6, dtype=float), 40)  # 6 tie groups of 40
    labels = rng.integers(0, 2, size=scores.size)
    ap0 = average_precision(scores, labels)
    au0 = auroc(scores, labels)
    for k in range(300):
        s2, l2 = _perm_within_ties(scores, labels, np.random.default_rng(100 + k))
        assert average_precision(s2, l2) == pytest.approx(ap0, abs=1e-12)
        assert auroc(s2, l2) == pytest.approx(au0, abs=1e-12)


def test_full_permutation_invariant():
    """AP/AUROC are functions of the (score, label) multiset, so any full
    row permutation leaves them unchanged."""
    rng = np.random.default_rng(4)
    scores = rng.integers(0, 4, size=150).astype(float)
    labels = rng.integers(0, 2, size=150)
    ap0 = average_precision(scores, labels)
    au0 = auroc(scores, labels)
    for k in range(100):
        p = np.random.default_rng(200 + k).permutation(scores.size)
        assert average_precision(scores[p], labels[p]) == pytest.approx(ap0, abs=1e-12)
        assert auroc(scores[p], labels[p]) == pytest.approx(au0, abs=1e-12)


def test_single_class_returns_nan():
    scores = [0.2, 0.7, 0.4, 0.9]
    assert np.isnan(average_precision(scores, [0, 0, 0, 0]))
    assert np.isnan(average_precision(scores, [1, 1, 1, 1]))
    assert np.isnan(auroc(scores, [0, 0, 0, 0]))
    assert np.isnan(auroc(scores, [1, 1, 1, 1]))
    assert np.isnan(average_precision([], []))
    assert np.isnan(auroc([], []))


def test_brier_and_log_loss():
    probs = np.array([0.1, 0.8, 0.6, 0.3])
    labels = np.array([0, 1, 1, 0])
    assert brier(probs, labels) == pytest.approx(float(np.mean((probs - labels) ** 2)))
    expected_ll = float(
        -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    )
    assert log_loss_(probs, labels) == pytest.approx(expected_ll, rel=1e-6)
    # near-perfect predictions -> ~0 loss, finite thanks to clipping
    assert log_loss_([1.0, 0.0], [1, 0]) == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(brier([], []))
    assert np.isnan(log_loss_([], []))
