#!/usr/bin/env python
"""Compute ROC / precision-recall curves + ROC-AUC for **every** guard in
outputs/benchmark_results.json, and write:

- outputs/curves.json — {benchmark: {guard: {auc, kind, roc, pr, n}}}
- the same `roc_auc` merged back into each cell of benchmark_results.json

Single source of truth: the per-sample predictions the scoreboard was built from
(``outputs/predictions/<guard>.json``, rows ``[y, u, score, ms]``). Curves and AUC are
derived from those exact scores, so curves.json and benchmark_results.json can never
disagree, and every guard — including ensembles — is treated identically:

- **Continuous score** (e.g. the fine-tuned encoder) → a real threshold-swept ROC/PR
  curve and rank ROC-AUC.
- **Hard decision** (keyword · decoders · OpenAI · hard-vote ensembles) → the score is
  binary, so the swept curve collapses to a single operating point and the rank-AUC
  provably equals ``(recall + 1 - FPR) / 2`` — the same number, computed the same way.

A guard×bench with NO dumped predictions falls back to a single operating point derived
from its stored metrics (with the PR endpoint at the true class prevalence, not 0.5).

Usage:
    python scripts/report/compute_curves.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402

from agent_bouncer.evaluation.curves import downsample, pr_curve, roc_auc, roc_curve  # noqa: E402
from agent_bouncer.evaluation.ensembles import load_predictions  # noqa: E402

RESULTS_JSON = "outputs/benchmark_results.json"
OUT = "outputs/curves.json"


def _prevalence(m: dict, meta_cell: dict | None) -> float:
    """Positive (unsafe) prevalence for a cell scored WITHOUT dumped predictions.

    Prefer the benchmark's stored class counts; else recover it algebraically from
    recall/FPR/precision; else assume balanced (0.5)."""
    if meta_cell:
        ns, nu = meta_cell.get("n_safe"), meta_cell.get("n_unsafe")
        if ns is not None and nu is not None and (ns + nu) > 0:
            return nu / (ns + nu)
    # recall*P / (recall*P + fpr*N) = precision, with P+N = n  ->  P/N = prec*fpr / (rec*(1-prec))
    rec, fpr, prec = m.get("recall", 0.0), m.get("fpr_on_benign", 0.0), m.get("precision", 0.0)
    denom = rec * (1.0 - prec)
    if denom > 0 and prec > 0:
        ratio = prec * fpr / denom
        return ratio / (1.0 + ratio)
    return 0.5


def _derive_point(m: dict, meta_cell: dict | None) -> dict:
    """Single-operating-point ROC/PR/AUC from a hard classifier's stored metrics (fallback
    for cells with no dumped predictions). The PR endpoint at recall=1 is the class
    prevalence (predict-all-positive precision), NOT a hardcoded 0.5."""
    tpr = float(m.get("recall", 0.0))
    fpr = float(m.get("fpr_on_benign", 0.0))
    prec = float(m.get("precision", 0.0))
    prev = _prevalence(m, meta_cell)
    return {
        "auc": (tpr + 1.0 - fpr) / 2.0, "kind": "point", "n": int(m.get("n", 0)),
        "roc": [[0.0, 0.0], [fpr, tpr], [1.0, 1.0]],
        "pr": [[0.0, 1.0], [tpr, prec], [1.0, prev]],
    }


def _from_predictions(rows: list) -> dict:
    """Real ROC/PR curve + rank ROC-AUC from a guard's per-sample ``[y, u, score, ms]`` rows.
    Collapses to a single operating point automatically when the score is binary."""
    labels = [r[0] for r in rows]
    scores = [r[2] for r in rows]
    auc = roc_auc(labels, scores)          # None only when a single class is present
    swept = len(set(scores)) > 2           # more than {0,1} distinct scores -> a real curve
    return {
        "auc": auc, "kind": "swept" if swept else "point", "n": len(rows),
        "roc": [list(p) for p in downsample(roc_curve(labels, scores))],
        "pr": [list(p) for p in downsample(pr_curve(labels, scores))],
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    if not os.path.exists(RESULTS_JSON):
        raise SystemExit(f"{RESULTS_JSON} not found — run the benchmark suite first")
    blob = json.load(open(RESULTS_JSON))
    results = blob["results"]
    all_meta = blob.get("meta", {})
    preds = load_predictions()             # {guard: {bench: rows}}

    curves: dict[str, dict] = {}
    for bench, guard_map in results.items():
        curves[bench] = {}
        for gname, m in guard_map.items():
            rows = preds.get(gname, {}).get(bench)
            if rows:
                entry = _from_predictions(rows)
                if entry["auc"] is None:   # single class in this cell -> operating-point AUC
                    entry = _derive_point(m, all_meta.get(bench))
            else:
                entry = _derive_point(m, all_meta.get(bench))
            curves[bench][gname] = entry
            m["roc_auc"] = entry["auc"]     # keep the scoreboard identical to the curve
            print(f"  [{bench}] {gname}: AUC={entry['auc']:.3f} ({entry['kind']}, n={entry['n']})")

    os.makedirs("outputs", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(curves, fh, indent=2)
    # merge roc_auc back into the results file (atomic)
    fd, tmp = tempfile.mkstemp(dir="outputs", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(blob, fh, indent=2)
    os.replace(tmp, RESULTS_JSON)
    print(f"\nwrote {OUT} and merged roc_auc into {RESULTS_JSON}")


if __name__ == "__main__":
    main()
