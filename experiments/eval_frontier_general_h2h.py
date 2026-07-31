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


def paired_delta(arms, b, y, n_boot: int = N_BOOT) -> dict:
    """Paired bootstrap of (guard - reference), consistent with the tabulated metric.

    `arms` is a LIST of score vectors -- one per training seed for an SFT guard, or a single
    vector for a base guard.

    Why a list, and not the mean of those vectors: the tables report Paper A's convention,
    metric-per-seed then averaged (`mean_over_seeds`). An earlier version of this function was
    handed `np.nanmean` of the seed vectors instead, which computes the metric of a five-run
    SCORE ENSEMBLE -- a different estimand, and not the expected performance of one trained
    guard. The two disagreed visibly: the table printed .948 and .741 while the delta printed
    +.185 rather than the .207 those two values imply. Averaging the per-seed deltas instead
    makes the point estimate identically equal to (tabulated guard metric - reference metric),
    so a reader can always check the arithmetic on the page.

    The interval resamples BOTH rows and seeds. Seeds are resampled with replacement because
    the five adapters are draws of one recipe, so training-seed variance belongs inside the
    interval; the old version held the averaged vector fixed and so reported row uncertainty
    only. For a single-arm guard the seed dimension is degenerate and only rows are resampled.
    """
    arms = [np.asarray(a, float) for a in arms]
    b, y = np.asarray(b, float), np.asarray(y, int)
    ok = ~np.isnan(b)
    for a in arms:
        ok &= ~np.isnan(a)
    arms = [a[ok] for a in arms]
    b, y = b[ok], y[ok]
    n_arm = len(arms)

    def point(idx, seed_pick):
        """mean over seeds of (metric(seed) - metric(reference)), on rows `idx`."""
        yy = y[idx]
        ref_t, ref_a = tpr_at_fpr(b[idx], yy), average_precision(b[idx], yy)
        dt = [tpr_at_fpr(arms[s][idx], yy) - ref_t for s in seed_pick]
        da = [average_precision(arms[s][idx], yy) - ref_a for s in seed_pick]
        return float(np.mean(dt)), float(np.mean(da))

    all_rows = np.arange(len(y))
    obs = point(all_rows, range(n_arm))

    rng = np.random.default_rng(BOOT_SEED)
    bt, bap = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        picks = rng.integers(0, n_arm, n_arm) if n_arm > 1 else [0]
        t, a = point(idx, picks)
        bt.append(t)
        bap.append(a)

    def pct(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2

    return {"n": int(len(y)), "n_arms": n_arm, "d_tpr": float(obs[0]), "ci_tpr": pct(bt),
            "d_ap": float(obs[1]), "ci_ap": pct(bap), "n_boot": int(len(bt)),
            "estimand": "mean over seeds of (guard - reference); rows and seeds both resampled"}


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
    arm_store: dict = {}   # source -> cell -> (arms, ref vector, y) for the aggregate boot

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
        # both scored. The seed arms are passed through INDIVIDUALLY -- see paired_delta on why
        # pre-averaging them silently changed the estimand and broke the displayed arithmetic.
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
                deltas[f"{mk}__{cond}"] = paired_delta(mats, ref, y)
                # keep the aligned arrays: the aggregate estimand needs to resample the same
                # rows across every cell of a source, which per-cell summaries cannot support
                okm = ~np.isnan(ref)
                for a in mats:
                    okm &= ~np.isnan(a)
                arm_store.setdefault(source, {})[f"{mk}__{cond}"] = (
                    [a[okm] for a in mats], ref[okm], y[okm])

        results[source] = {
            "regime": reg["regime"].get(source, "unknown"),
            "n": len(ids),
            "prevalence": float(y.mean()),
            "guards": guards,
            "deltas_vs_frontier_ref": deltas,
            "reference_tpr_degenerate": ref_degenerate,
            "deltas_interpretable": not ref_degenerate,
        }

    # ── multiplicity, and one aggregate that is not chosen by its own result ────────────
    # The headline used to be the LARGEST significant per-cell delta out of the represented
    # SFT cells, with its nominal interval printed unadjusted. That is a post-selection
    # interval: scanning twelve cells and reporting the winner's 95% CI overstates it however
    # many of the twelve are individually significant. Two things fix it. (1) An aggregate
    # estimand -- the equal-source, equal-checkpoint mean delta over represented sources --
    # which is fixed in advance and does not depend on which cell wins. (2) Holm-adjusted
    # per-cell intervals, so any single cell that is still quoted is quoted honestly.
    rep = [s for s, r in results.items()
           if r["regime"] == "represented" and r["deltas_interpretable"]]
    sft_cells = [(s, g, results[s]["deltas_vs_frontier_ref"][g])
                 for s in rep for g in sorted(results[s]["deltas_vs_frontier_ref"])
                 if g.endswith("__sft")]

    # Holm step-down on the bootstrap two-sided p-value proxy: the smallest alpha at which the
    # interval would still exclude zero is bounded by how far the nearer bound sits from zero,
    # so rank by |delta| / half-width and adjust the family of m tests.
    m = len(sft_cells)
    scored = []
    for s, g, d in sft_cells:
        lo, hi = d["ci_tpr"]
        half = (hi - lo) / 2 if hi == hi and lo == lo else float("nan")
        z = abs(d["d_tpr"]) / half if half and half == half and half > 0 else 0.0
        scored.append((z, s, g, d))
    scored.sort(key=lambda t: -t[0])
    for rank, (z, s, g, d) in enumerate(scored):
        # Holm: the k-th largest test is compared at alpha/(m-k); widen the interval by the
        # ratio of the adjusted to the nominal critical value (normal approximation).
        alpha_adj = 0.05 / max(m - rank, 1)
        from math import sqrt
        # z_{1-alpha/2} for the adjusted vs nominal level, via a rational approximation good
        # to ~1e-4 over the range we need (alpha in [0.05/12, 0.05]).
        def zcrit(alpha):
            p = 1 - alpha / 2
            t = sqrt(-2.0 * __import__("math").log(1 - p))
            return t - ((0.010328 * t + 0.802853) * t + 2.515517) / \
                       (((0.001308 * t + 0.189269) * t + 1.432788) * t + 1.0)
        widen = zcrit(alpha_adj) / zcrit(0.05)
        lo, hi = d["ci_tpr"]
        mid = d["d_tpr"]
        d["ci_tpr_holm"] = [mid - (mid - lo) * widen, mid + (hi - mid) * widen]
        d["holm_rank"] = rank + 1
        d["holm_alpha"] = alpha_adj
        d["holm_significant"] = bool(d["ci_tpr_holm"][0] > 0)
        d["holm_family_size"] = m

    # The aggregate needs its own interval, not just a point estimate, or it cannot carry a
    # claim either. Hierarchical bootstrap: within each represented source resample rows and
    # seeds, average over that source's checkpoints, then average over sources -- one draw of
    # the whole estimand per replicate, with sources drawn jointly so the mean is coherent.
    def aggregate_boot(key):
        rng2 = np.random.default_rng(BOOT_SEED + 1)
        obs_per_source, draws = [], []
        for s in rep:
            cells = [g for g in sorted(arm_store[s]) if g.endswith("__sft")]
            vals = []
            for g in cells:
                arms, ref_v, yv = arm_store[s][g]
                r_t, r_a = tpr_at_fpr(ref_v, yv), average_precision(ref_v, yv)
                per = [(tpr_at_fpr(a, yv) - r_t) if key == "d_tpr"
                       else (average_precision(a, yv) - r_a) for a in arms]
                vals.append(float(np.mean(per)))
            obs_per_source.append(float(np.mean(vals)))
        obs = float(np.mean(obs_per_source))

        for _ in range(N_BOOT):
            per_source = []
            ok = True
            for s in rep:
                cells = [g for g in sorted(arm_store[s]) if g.endswith("__sft")]
                arms0, ref0, y0 = arm_store[s][cells[0]]
                idx = rng2.integers(0, len(y0), len(y0))
                if y0[idx].min() == y0[idx].max():
                    ok = False
                    break
                vals = []
                for g in cells:
                    arms, ref_v, yv = arm_store[s][g]
                    yy = yv[idx]
                    r_t, r_a = tpr_at_fpr(ref_v[idx], yy), average_precision(ref_v[idx], yy)
                    picks = rng2.integers(0, len(arms), len(arms)) if len(arms) > 1 else [0]
                    per = [(tpr_at_fpr(arms[p][idx], yy) - r_t) if key == "d_tpr"
                           else (average_precision(arms[p][idx], yy) - r_a) for p in picks]
                    vals.append(float(np.mean(per)))
                per_source.append(float(np.mean(vals)))
            if ok:
                draws.append(float(np.mean(per_source)))
        ci = ([float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
              if draws else [float("nan")] * 2)
        return obs, ci, len(draws)

    d_tpr_obs, d_tpr_ci, nb = aggregate_boot("d_tpr")
    d_ap_obs, d_ap_ci, _ = aggregate_boot("d_ap")

    aggregate = {
        "estimand": "equal-source, equal-checkpoint mean paired delta (panel SFT - "
                    f"{FRONTIER_REF.replace('__', ' / ')}) over represented sources",
        "selection_status": "POST HOC. This aggregate did not exist before the per-cell headline "
                            "failed multiplicity; it was added in the same revision that discovered "
                            "the failure. It is a better summary than the maximum of twelve because "
                            "it does not depend on which cell wins, but it is not pre-specified, and "
                            "it must not be described as such. Only a summary frozen before a fresh "
                            "cohort is scored can carry a confirmatory frontier claim.",
        "sources": rep,
        "n_cells": m,
        "n_boot": nb,
        "d_tpr": d_tpr_obs, "ci_tpr": d_tpr_ci,
        "d_ap": d_ap_obs, "ci_ap": d_ap_ci,
        "significant": bool(d_tpr_ci[0] > 0),
        "n_significant_holm": sum(1 for _, _, _, d in scored if d["holm_significant"]),
        "n_significant_nominal": sum(1 for _, _, _, d in scored if d["ci_tpr"][0] > 0),
    }

    meta = {
        "fpr_budget": FPR_BUDGET,
        "frontier_reference": FRONTIER_REF.replace("__", " / "),
        "splits": list(splits),
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "delta_estimand": "mean over seeds of (guard - reference), rows and seeds resampled; "
                          "equals (tabulated guard metric - reference metric) by construction",
        "multiplicity": f"Holm across the {m} represented-source SFT cells; per-cell nominal "
                        "intervals are exploratory, ci_tpr_holm is the adjusted interval",
        "local_scores": os.path.relpath(SCORES, ROOT),
        "frontier_predictions": os.path.relpath(RAW, ROOT),
        "trained_sources": reg["trained_sources"],
        "join": "gpt rid=sha256(text)[:16] -> content_sha256=sha256(normalize_text(text))",
        "text_free": "all five sources are text_free_only in the distribution ledger; "
                     "this artifact stores digests, scores and metrics only",
        "flavor": "retrospective, estimation-only -- these rows and this panel were inspected "
                  "during development; not a preregistered or confirmatory comparison",
    }
    return {"meta": meta, "aggregate": aggregate, "sources": results}, audit


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
