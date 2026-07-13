"""Conservative operating-point selection (plan sec 10.5).

Secondary deployment diagnostic (RQ4). NOT a distribution-free production
guarantee. The threshold is chosen on *calibration* data only:

    maximize calibration recall subject to the one-sided 95% Clopper-Pearson
    upper bound on the pooled calibration-negative FPR being at most target_fpr.

Test labels never enter this module -- :func:`select_threshold` takes only
calibration scores and labels, so a threshold can never be tuned on test data.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta

__all__ = ["clopper_pearson_upper", "select_threshold"]


def clopper_pearson_upper(fp, tn, conf: float = 0.95) -> float:
    """One-sided upper confidence bound on the FPR from FP false positives out of
    (FP + TN) negatives.

        U_CP = Beta^{-1}(conf; FP + 1, TN)      (plan sec 10.5 step 4)

    Exact edge cases:
      * ``tn <= 0``: no informative negatives (or all negatives are FP) -> 1.0.
      * ``fp == 0``: the Beta(1, TN) quantile has the closed form
        ``1 - (1 - conf)**(1/TN)``, used directly for numerical robustness.
    """
    fp = int(fp)
    tn = int(tn)
    if tn <= 0:
        return 1.0
    if fp == 0:
        return float(1.0 - (1.0 - conf) ** (1.0 / tn))
    return float(beta.ppf(conf, fp + 1, tn))


def select_threshold(cal_scores, cal_labels, target_fpr: float = 0.05) -> dict:
    """Select the conservative operating point on calibration data (plan sec 10.5).

    Steps 1-8:
      1. candidate set = +inf plus every unique calibrated score;
      2. predict unsafe when score >= candidate threshold;
      3. count FP / TN over the calibration NEGATIVES;
      4. compute the one-sided 95% Clopper-Pearson upper bound on FPR;
      5. keep candidates whose CP upper bound <= target_fpr;
      6. among the feasible set, pick maximum calibration recall;
      7. break equal-recall ties by lower empirical FPR, then higher threshold;
      8. if none is feasible, return {'status': 'NO_FEASIBLE_THRESHOLD'}.

    Returns, on success::

        {threshold, cal_tp, fp, tn, fn, cal_fpr_point, cal_fpr_upper, cal_recall}

    ``threshold`` may be ``float('inf')`` (the candidate above the maximum
    observed score, which predicts no calibration positives).
    """
    s = np.asarray(cal_scores, dtype=float)
    y = np.asarray(cal_labels, dtype=float)

    neg = y == 0
    pos = y == 1
    n_neg = int(neg.sum())
    n_pos = int(pos.sum())
    neg_scores = s[neg]
    pos_scores = s[pos]

    # Candidate set: every unique score, plus +inf (predict nothing unsafe).
    candidates = np.concatenate([np.unique(s), np.array([np.inf])])

    best_key = None
    best = None
    for thr in candidates:
        # Predict unsafe when score >= thr. Exact float equality here means all
        # rows tied at exactly `thr` are classified together (consistent ties).
        fp = int(np.count_nonzero(neg_scores >= thr))
        tp = int(np.count_nonzero(pos_scores >= thr))
        tn = n_neg - fp
        fn = n_pos - tp

        cp_upper = clopper_pearson_upper(fp, tn, conf=0.95)
        if cp_upper > target_fpr:
            continue  # step 5: fails the conservative gate

        recall = tp / n_pos if n_pos else 0.0
        fpr_point = fp / n_neg if n_neg else 0.0
        # step 6/7: max recall, then min FPR, then max threshold. Encoding all
        # three as a single tuple to maximize gives exactly that ordering.
        key = (recall, -fpr_point, thr)
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "threshold": float(thr),
                "cal_tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "cal_fpr_point": float(fpr_point),
                "cal_fpr_upper": float(cp_upper),
                "cal_recall": float(recall),
            }

    if best is None:
        return {"status": "NO_FEASIBLE_THRESHOLD"}
    return best
