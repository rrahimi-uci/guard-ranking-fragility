#!/usr/bin/env python
"""Frontier vs. local guards on the general-safety benchmarks both sides already scored.

\\Cref{tab:frontier} compares the panel against gpt-5.4 on ExpGuard only, because ExpGuard
was the one source deliberately row-aligned between the two runs. But the GPT baseline also
scored five of the corpora the Act~I panel scores, and those rows *can* be joined -- the two
sides simply hash their row ids differently:

    gpt-baseline/datasets.py  rid = sha256(text)[:16]                 (raw UTF-8)
    guard_research.provenance content_sha256 = sha256(normalize_text(text))  (NFC-normalized)

Re-deriving both digests from the local corpus recovers the mapping exactly (100% of the
panel's rows join on all five sources). That is the whole content of this script.

The comparison matters because ExpGuard is an *external breadth probe* -- a source the panel
never trained on -- so it measures only the transfer regime. These five sources span both
regimes, and Act~I's own manifest fixes which is which:

    train.jsonl         = {toxicchat, prompt_injections, jailbreak_classification}
    transfer_test.jsonl = {jailbreakbench, xstest, wildguardtest, wildjailbreak}

so `id_test` rows are held-out *rows* from a *represented* source, and `transfer_test` rows
are held out at the source level. Reporting the frontier gap separately per regime is the
point; pooling them would average the two answers into neither.

Text-free by construction. All five sources are `text_free_only` in
benchmarks/registry/distribution.yaml, so nothing here persists prompt text: the corpus is
read only to derive the two digests and a binary label, and the artifact stores digests,
scores and metrics. Run it from the repo root.

Writes:
  artifacts/frontier_general_h2h/h2h.json          -- every cell, plus provenance
  artifacts/frontier_general_h2h/join_audit.json   -- id-join coverage per source
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from guard_research.metrics import auroc, average_precision  # noqa: E402
from guard_research.provenance import content_sha256  # noqa: E402

SCORES = os.path.join(ROOT, "artifacts", "paper_a_sft_v2", "scores", "scores.parquet")
MANIFESTS = os.path.join(ROOT, "artifacts", "paper_a_sft_v2", "manifests")
RAW = os.path.join(ROOT, "gpt-baseline", "raw")
CORPUS = os.path.join(ROOT, "data", "benchmarks", "full")
OUT = os.path.join(ROOT, "artifacts", "frontier_general_h2h")

# Matches papers/unified-report/frontier.py: recall at a common false-alarm budget, because
# the two sides sit at very different self-chosen operating points.
FPR_BUDGET = 0.05
N_BOOT = 2000
BOOT_SEED = 20260730

# The five sources both runs scored. Regime is read from the manifests, not asserted here.
SOURCES = ["prompt_injections", "toxicchat", "jailbreak_classification", "jailbreakbench", "xstest"]

# gpt-5.4 (low) is the pre-specified comparator: it is the frontier config that leads
# \Cref{tab:frontier} on ExpGuard. The other configs are reported but not used for deltas,
# so no per-benchmark argmax over frontier configs enters a headline number.
FRONTIER_REF = "gpt-5.4__low"
FRONTIER = [FRONTIER_REF, "gpt-5.4__medium", "gpt-5.4__high",
            "gpt-5.4-mini__low", "gpt-5.4-mini__medium", "gpt-5.4-mini__high"]

# gpt-baseline/datasets.py label vocabulary, reproduced so the labels agree row-for-row.
SAFE_LABELS = {"safe", "0", "false", "benign", "no"}
UNSAFE_LABELS = {"unsafe", "1", "true", "harmful", "jailbreak", "yes"}


# ──────────────────────────────────────────────────────────────── metrics


def tpr_at_fpr(scores, y, budget: float = FPR_BUDGET) -> float:
    """Recall once the score is re-thresholded to `budget` FPR on these rows.

    Same convention as frontier.py: the threshold is the (1-budget) quantile of the
    negatives with `method="higher"`, and positives must beat it strictly.
    """
    s, yy = np.asarray(scores, float), np.asarray(y, int)
    ok = ~np.isnan(s)
    s, yy = s[ok], yy[ok]
    if (yy == 0).sum() == 0 or (yy == 1).sum() == 0:
        return float("nan")
    thr = np.quantile(s[yy == 0], 1 - budget, method="higher")
    return float((s[yy == 1] > thr).mean())


def cell(scores, y) -> dict:
    """One guard on one row set. `n_unique` is the tie diagnostic -- see `degenerate`."""
    s = np.asarray(scores, float)
    ok = ~np.isnan(s)
    n_unique = int(len(np.unique(s[ok]))) if ok.any() else 0
    out = {
        "n": int(ok.sum()),
        "tpr": tpr_at_fpr(s, y),
        "ap": average_precision(s[ok], np.asarray(y, int)[ok]) if ok.any() else float("nan"),
        "auroc": auroc(s[ok], np.asarray(y, int)[ok]) if ok.any() else float("nan"),
        "n_unique_scores": n_unique,
    }
    # A coarse integer score can tie so heavily that the FPR-budget quantile lands inside one
    # tie block, which collapses TPR while leaving AUROC intact (gpt-5.4/low on jailbreakbench
    # is the live example: 35 unique values over 200 rows). Flag it rather than let the reader
    # mistake a tie artifact for behaviour.
    out["degenerate_tpr"] = bool(
        ok.any() and not np.isnan(out["tpr"]) and out["tpr"] < 0.5 <= (out["auroc"] or 0) - 0.35
    )
    return out


def mean_over_seeds(per_seed: list[dict]) -> dict:
    """Metric per seed, then averaged -- Paper A's convention (frontier.py:158).

    Averaging metrics rather than margins is deliberate: the seeds are five draws of one
    recipe, and pooling their margins would blend five score scales into a ranking that
    belongs to no guard.
    """
    keys = ["tpr", "ap", "auroc"]
    out = {k: float(np.mean([p[k] for p in per_seed])) for k in keys}
    out["n"] = per_seed[0]["n"]
    out["n_seeds"] = len(per_seed)
    out["tpr_min"] = float(min(p["tpr"] for p in per_seed))
    out["tpr_max"] = float(max(p["tpr"] for p in per_seed))
    return out


def paired_delta(a, b, y, n_boot: int = N_BOOT) -> dict:
    """Paired row bootstrap of (a - b) in TPR@FPR and AP, on rows both sides scored."""
    a, b, y = np.asarray(a, float), np.asarray(b, float), np.asarray(y, int)
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b, y = a[ok], b[ok], y[ok]
    obs = (tpr_at_fpr(a, y) - tpr_at_fpr(b, y), average_precision(a, y) - average_precision(b, y))
    rng = np.random.default_rng(BOOT_SEED)
    bt, bap = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        bt.append(tpr_at_fpr(a[idx], ys) - tpr_at_fpr(b[idx], ys))
        bap.append(average_precision(a[idx], ys) - average_precision(b[idx], ys))

    def pct(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2

    return {"n": int(len(y)), "d_tpr": float(obs[0]), "ci_tpr": pct(bt),
            "d_ap": float(obs[1]), "ci_ap": pct(bap), "n_boot": int(len(bt))}


# ──────────────────────────────────────────────────────────────── joining


def manifest_regime() -> dict:
    """Which sources Act~I trained on, read from the committed manifests."""
    seen = {}
    for split in ("train", "id_test", "calibration", "transfer_test"):
        path = os.path.join(MANIFESTS, f"{split}.jsonl")
        for line in open(path):
            row = json.loads(line)
            src = row.get("source") or row.get("dataset")
            seen.setdefault(split, set()).add(src)
    trained = seen.get("train", set())
    return {"trained_sources": sorted(trained),
            "regime": {s: ("represented" if s in trained else "transfer")
                       for s in sorted(set().union(*seen.values()))}}


def corpus_index(source: str) -> tuple[dict, dict]:
    """(gpt rid -> content_sha256, content_sha256 -> label) derived from the local corpus.

    Neither map holds text. Duplicate prompts are dropped on first-seen, matching
    gpt-baseline/datasets.py so the two sides agree on which row a digest denotes.
    """
    path = os.path.join(CORPUS, f"{source}.jsonl")
    rid2norm, norm2label, seen = {}, {}, set()
    for line in open(path):
        row = json.loads(line)
        text = str(row.get("text") or row.get("prompt") or row.get("input") or "")
        if not text.strip():
            continue
        token = str(row.get("label", row.get("gold"))).strip().lower()
        if token in SAFE_LABELS:
            label = 0
        elif token in UNSAFE_LABELS:
            label = 1
        else:
            continue
        rid = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if rid in seen:
            continue
        seen.add(rid)
        norm = content_sha256(text)
        rid2norm[rid] = norm
        norm2label[norm] = label
    return rid2norm, norm2label


def frontier_scores(config: str, source: str, rid2norm: dict) -> dict:
    """content_sha256 -> self-reported 0-100 risk, for the rows the provider answered."""
    path = os.path.join(RAW, f"{config}__{source}.jsonl")
    if not os.path.isfile(path):
        return {}
    out = {}
    for line in open(path):
        rec = json.loads(line)
        if not rec.get("ok"):
            continue  # transport failure or a provider 400: never imputed as a negative
        norm = rid2norm.get(rec["rid"])
        if norm is not None:
            out[norm] = float(rec["raw"]["risk"])
    return out


# ──────────────────────────────────────────────────────────────── main


def compute(splits: tuple[str, ...]) -> tuple[dict, dict]:
    reg = manifest_regime()
    local = pd.read_parquet(SCORES)
    local = local[local.source.isin(SOURCES)]
    results, audit = {}, {}

    for source in SOURCES:
        rid2norm, norm2label = corpus_index(source)
        loc = local[(local.source == source) & (local.split.isin(splits))]
        if loc.empty:
            continue
        ids = sorted(set(loc.content_sha256))
        y = np.array([norm2label[i] for i in ids], int)
        gpt_raw = {c: frontier_scores(c, source, rid2norm) for c in FRONTIER}

        audit[source] = {
            "corpus_rows_labelled": len(rid2norm),
            "panel_rows": len(ids),
            "panel_splits": sorted(loc.split.unique()),
            "joined": {c: int(sum(i in s for i in ids)) for c, s in gpt_raw.items()},
            "prevalence": float(y.mean()),
        }

        guards = {}
        for mk in sorted(loc.model_key.unique()):
            base = loc[(loc.model_key == mk) & (loc.condition == "base")]
            if not base.empty:
                v = base.set_index("content_sha256").score_raw.reindex(ids).to_numpy(float)
                guards[f"{mk}__base"] = cell(v, y)
            per_seed, vecs = [], []
            for seed in sorted(loc[(loc.model_key == mk) & (loc.condition == "sft")].seed.unique()):
                sub = loc[(loc.model_key == mk) & (loc.condition == "sft") & (loc.seed == seed)]
                v = sub.set_index("content_sha256").score_raw.reindex(ids).to_numpy(float)
                if np.isnan(v).all():
                    continue
                per_seed.append(cell(v, y))
                vecs.append(v)
            if per_seed:
                guards[f"{mk}__sft"] = mean_over_seeds(per_seed)

        for config, sc in gpt_raw.items():
            v = np.array([sc.get(i, np.nan) for i in ids], float)
            guards[config] = cell(v, y)

        # Deltas: every checkpoint against the one pre-specified frontier config, on the rows
        # both scored. Per-seed SFT margins are averaged here (a single ranking is needed for a
        # paired bootstrap); the tabulated SFT metric stays the mean-over-seeds above.
        ref = np.array([gpt_raw[FRONTIER_REF].get(i, np.nan) for i in ids], float)
        # If the reference's own TPR is tie-collapsed, every TPR delta against it is an artifact
        # of where its tie block fell, not a comparison. Mark the whole source rather than
        # publish deltas that cannot be read (jailbreakbench is the live case).
        ref_degenerate = bool(guards[FRONTIER_REF].get("degenerate_tpr"))
        deltas = {}
        for mk in sorted(loc.model_key.unique()):
            for cond in ("base", "sft"):
                sub = loc[(loc.model_key == mk) & (loc.condition == cond)]
                if sub.empty:
                    continue
                mats = [sub[sub.seed == s].set_index("content_sha256").score_raw
                        .reindex(ids).to_numpy(float) for s in sorted(sub.seed.unique())]
                v = np.nanmean(np.vstack(mats), axis=0)
                deltas[f"{mk}__{cond}"] = paired_delta(v, ref, y)

        results[source] = {
            "regime": reg["regime"].get(source, "unknown"),
            "n": len(ids),
            "prevalence": float(y.mean()),
            "guards": guards,
            "deltas_vs_frontier_ref": deltas,
            "reference_tpr_degenerate": ref_degenerate,
            "deltas_interpretable": not ref_degenerate,
        }

    meta = {
        "fpr_budget": FPR_BUDGET,
        "frontier_reference": FRONTIER_REF.replace("__", " / "),
        "splits": list(splits),
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "local_scores": os.path.relpath(SCORES, ROOT),
        "frontier_predictions": os.path.relpath(RAW, ROOT),
        "trained_sources": reg["trained_sources"],
        "join": "gpt rid=sha256(text)[:16] -> content_sha256=sha256(normalize_text(text))",
        "text_free": "all five sources are text_free_only in the distribution ledger; "
                     "this artifact stores digests, scores and metrics only",
        "flavor": "retrospective, estimation-only -- these rows and this panel were inspected "
                  "during development; not a preregistered or confirmatory comparison",
    }
    return {"meta": meta, "sources": results}, audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["id_test", "transfer_test"],
                    help="panel splits to score (default excludes calibration)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    payload, audit = compute(tuple(args.splits))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "h2h.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "join_audit.json"), "w") as fh:
        json.dump(audit, fh, indent=2, sort_keys=True)

    for source, r in payload["sources"].items():
        print(f"\n=== {source}  [{r['regime']}]  n={r['n']}  prev={r['prevalence']:.3f} ===")
        rows = sorted(r["guards"].items(), key=lambda kv: -(kv[1]["tpr"] if kv[1]["tpr"] == kv[1]["tpr"] else -1))
        for name, c in rows:
            flag = "  <- tie-collapsed TPR" if c.get("degenerate_tpr") else ""
            print(f"  {name:28s} TPR={c['tpr']:.3f}  AP={c['ap']:.4f}  AUROC={c['auroc']:.4f}{flag}")
    print(f"\nwrote {os.path.join(args.out, 'h2h.json')}")


if __name__ == "__main__":
    main()
