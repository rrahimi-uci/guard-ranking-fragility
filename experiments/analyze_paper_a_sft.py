#!/usr/bin/env python
"""Paper A analysis (plan sec 12, 15.5, 15.6, 16).

From the keyed score table it computes, using the canonical metric module
(guard_research.metrics -- never a custom AP):

  * per (model_key, condition, seed, benchmark) tie-aware AP + AUROC;
  * macro-AP over represented-source and over transfer benchmarks;
  * base->SFT deltas per checkpoint (mean over seeds) and the fixed-panel
    aggregate (mean over 4 checkpoints);
  * the hierarchical PAIRED bootstrap of plan sec 12.4 (10000 reps, seed
    20260712: 4 checkpoints fixed; resample 5 seed indices within each
    checkpoint; one Poisson(1) weight per GLOBAL family_id applied to all its
    rows across datasets; weighted tie-aware AP per benchmark; macro-average;
    one-sided 95% LCB/UCB and two-sided 95%);
  * leave-one-benchmark-out and leave-one-base-out sensitivity;
  * secondary 5% FPR operating point (TPR + realized FPR, represented/transfer);
  * OR-Bench benign FPR + HarmBench recall (one-class, NO AP);
  * claim gates (plan sec 16) with precision_focused estimation language.

Emits results.json, seed_values.csv, per_benchmark.csv, sensitivity.json,
claim_checks.json, LaTeX Table 3/4 fragments, and the specialization-plane
figure -- all generated from artifacts (no hand-entered numbers).

Usage:
  python experiments/analyze_paper_a_sft.py --lock LOCK.json \
    --scores artifacts/paper_a_sft/scores/scores.parquet \
    --out artifacts/paper_a_sft/analysis
  python experiments/analyze_paper_a_sft.py --self-test   # synthetic end-to-end check
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_common as C  # noqa: E402

ANALYSIS_CODE_VERSION = "paper_a_sft_analysis_v1"
AP_SPLITS = {"represented": "id_test", "transfer": "transfer_test"}


# --------------------------------------------------------------------------------------
# build per-benchmark aligned arrays from the score table
# --------------------------------------------------------------------------------------
def build_bench_data(df, regimes, model_keys, seeds):
    """Return (data, families) where
       data[mk][bench] = {regime, split, gold, fam_idx, base(scores), sft{seed:scores},
                          base_pred, sft_pred{seed}}
       families = ordered list of global family_ids over AP rows."""
    ap_rows = df[df["split"].isin(list(AP_SPLITS.values()))]
    families = sorted(map(str, ap_rows["family_id"].dropna().unique()))
    fam_index = {f: i for i, f in enumerate(families)}

    # which benchmarks are present in each regime
    present = {}
    for regime, split in AP_SPLITS.items():
        srcs = set(df[df["split"] == split]["source"].unique())
        present[regime] = [b for b in regimes[regime] if b in srcs]

    data = {}
    for mk in model_keys:
        data[mk] = {}
        for regime, benches in present.items():
            split = AP_SPLITS[regime]
            for bench in benches:
                base = df[(df.model_key == mk) & (df.condition == "base")
                          & (df.split == split) & (df.source == bench)].sort_values("sample_id")
                if base.empty:
                    continue
                order = base["sample_id"].tolist()
                gold = base["gold"].to_numpy(int)
                fam_idx = np.array([fam_index[str(f)] for f in base["family_id"]], int)
                entry = {
                    "regime": regime, "split": split, "n": len(order),
                    "gold": gold, "fam_idx": fam_idx,
                    "base": base["score_raw"].to_numpy(float),
                    "base_pred": base["prediction"].to_numpy(int),
                    "sft": {}, "sft_pred": {},
                }
                for s in seeds:
                    sf = df[(df.model_key == mk) & (df.condition == "sft") & (df.seed == s)
                            & (df.split == split) & (df.source == bench)]
                    sf = sf.set_index("sample_id").reindex(order)
                    entry["sft"][s] = sf["score_raw"].to_numpy(float)
                    entry["sft_pred"][s] = sf["prediction"].to_numpy(float)
                data[mk][bench] = entry
    return data, families, present


# --------------------------------------------------------------------------------------
# point estimates (plan sec 12.1-12.3)
# --------------------------------------------------------------------------------------
def macro_ap(data, mk, benches, ap_fn, condition, seed=None, weights=None):
    vals = []
    for b in benches:
        e = data[mk].get(b)
        if e is None:
            continue
        scores = e["base"] if condition == "base" else e["sft"][seed]
        w = None if weights is None else weights[e["fam_idx"]]
        vals.append(C.weighted_metric(ap_fn, scores, e["gold"], w))
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def point_estimates(data, present, model_keys, seeds, ap_fn):
    out = {"per_checkpoint": {}, "aggregate": {}, "seed_values": []}
    for regime, benches in present.items():
        ck = {}
        for mk in model_keys:
            base = macro_ap(data, mk, benches, ap_fn, "base")
            sft_seed = {s: macro_ap(data, mk, benches, ap_fn, "sft", s) for s in seeds}
            sft_mean = float(np.mean(list(sft_seed.values())))
            ck[mk] = {"base": base, "sft_mean": sft_mean,
                      "sft_by_seed": sft_seed, "delta": sft_mean - base,
                      "seed_deltas": {s: sft_seed[s] - base for s in seeds}}
        out["per_checkpoint"][regime] = ck
        out["aggregate"][regime] = float(np.mean([ck[mk]["delta"] for mk in model_keys]))
    # tidy seed_values rows
    for mk in model_keys:
        for s in seeds:
            row = {"model_key": mk, "seed": s}
            for regime in present:
                ck = out["per_checkpoint"][regime][mk]
                row[f"{regime}_base"] = ck["base"]
                row[f"{regime}_sft"] = ck["sft_by_seed"][s]
                row[f"{regime}_delta"] = ck["seed_deltas"][s]
            out["seed_values"].append(row)
    return out


# --------------------------------------------------------------------------------------
# hierarchical paired bootstrap (plan sec 12.4)
# --------------------------------------------------------------------------------------
def hierarchical_bootstrap(data, present, model_keys, seeds, ap_fn, reps, rng_seed,
                           max_redraw=2000):
    rng = np.random.default_rng(rng_seed)
    n_fam = 1 + max((e["fam_idx"].max() for mk in data for e in data[mk].values()
                     if e["fam_idx"].size), default=-1)
    n_seeds = len(seeds)
    regimes = list(present.keys())

    # precompute per (regime,mk) bench list; per bench pos/neg fam-membership for validity
    bench_of = {r: present[r] for r in regimes}
    # collect the set of (mk,bench) entries and, per bench, index arrays for validity test
    entries = [(mk, b, data[mk][b]) for mk in model_keys for r in regimes
               for b in bench_of[r] if b in data[mk]]

    agg_samples = {r: np.empty(reps) for r in regimes}
    ck_samples = {r: {mk: np.empty(reps) for mk in model_keys} for r in regimes}
    total_redraw = 0

    def weights_valid(w):
        for _mk, _b, e in entries:
            g = e["gold"]; fi = e["fam_idx"]
            if w[fi[g == 1]].sum() <= 0 or w[fi[g == 0]].sum() <= 0:
                return False
        return True

    for rep in range(reps):
        # 1. valid Poisson(1) family weights (redraw whole vector on zero-class)
        redraws = 0
        while True:
            w = rng.poisson(1.0, size=n_fam).astype(float)
            if weights_valid(w):
                break
            redraws += 1; total_redraw += 1
            if redraws > max_redraw:
                raise RuntimeError("bootstrap: exceeded redraw cap (data too sparse?)")
        # 2. seed resample indices per checkpoint (with replacement)
        seed_pick = {mk: rng.integers(0, n_seeds, size=n_seeds) for mk in model_keys}
        # 3-9. weighted macro AP -> per-ckpt delta -> aggregate
        for regime in regimes:
            benches = bench_of[regime]
            per_ck = []
            for mk in model_keys:
                base_M = macro_ap(data, mk, benches, ap_fn, "base", weights=w)
                seed_M = {s: macro_ap(data, mk, benches, ap_fn, "sft", s, weights=w) for s in seeds}
                picked = [seed_M[seeds[j]] for j in seed_pick[mk]]
                delta = float(np.mean(picked)) - base_M
                per_ck.append(delta)
                ck_samples[regime][mk][rep] = delta
            agg_samples[regime][rep] = float(np.mean(per_ck))

    def summarize(arr):
        return {
            "mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)),
            "lcb95_one_sided": float(np.percentile(arr, 5)),
            "ucb95_one_sided": float(np.percentile(arr, 95)),
            "ci95_two_sided": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "p_le_0": float(np.mean(arr <= 0)), "p_ge_0": float(np.mean(arr >= 0)),
        }

    result = {"aggregate": {r: summarize(agg_samples[r]) for r in regimes},
              "per_checkpoint": {r: {mk: summarize(ck_samples[r][mk]) for mk in model_keys}
                                 for r in regimes},
              "reps": reps, "rng_seed": rng_seed, "n_families": int(n_fam),
              "redraws": int(total_redraw),
              "rejected_fraction": float(total_redraw / (reps + total_redraw)) if reps else 0.0}
    return result


# --------------------------------------------------------------------------------------
# sensitivity: leave-one-benchmark-out, leave-one-base-out, heterogeneity (plan sec 12.4)
# --------------------------------------------------------------------------------------
def sensitivity(data, present, model_keys, seeds, ap_fn, point):
    out = {"leave_one_benchmark_out": {}, "leave_one_base_out": {},
           "per_base_delta": {}, "per_benchmark_delta": {}, "sign_table": {},
           "range_across_bases": {}, "range_across_benchmarks": {}}

    def agg_delta(regime, benches, mks):
        vals = []
        for mk in mks:
            base = macro_ap(data, mk, benches, ap_fn, "base")
            sft = float(np.mean([macro_ap(data, mk, benches, ap_fn, "sft", s) for s in seeds]))
            vals.append(sft - base)
        return float(np.mean(vals)) if vals else float("nan")

    for regime, benches in present.items():
        full = point["aggregate"][regime]
        # leave-one-benchmark-out
        loo_b = {}
        for b in benches:
            rem = [x for x in benches if x != b]
            loo_b[b] = agg_delta(regime, rem, model_keys) if rem else float("nan")
        out["leave_one_benchmark_out"][regime] = {
            "full": full, "loo": loo_b,
            "sign_stable": _sign_stable(full, list(loo_b.values()))}
        # leave-one-base-out
        loo_base = {}
        for mk in model_keys:
            rem = [x for x in model_keys if x != mk]
            loo_base[mk] = agg_delta(regime, benches, rem)
        out["leave_one_base_out"][regime] = {
            "full": full, "loo": loo_base,
            "sign_stable": _sign_stable(full, list(loo_base.values()))}
        # per-base delta + range/std
        pbd = {mk: point["per_checkpoint"][regime][mk]["delta"] for mk in model_keys}
        out["per_base_delta"][regime] = pbd
        out["range_across_bases"][regime] = {
            "min": float(min(pbd.values())), "max": float(max(pbd.values())),
            "range": float(max(pbd.values()) - min(pbd.values())),
            "std": float(np.std(list(pbd.values()), ddof=1)) if len(pbd) > 1 else 0.0}
        # per-benchmark delta (fixed-panel mean over bases) + range/std + sign row
        pbench = {}
        sign_row = {}
        for b in benches:
            deltas = []
            for mk in model_keys:
                e = data[mk].get(b)
                if e is None:
                    continue
                base = C.weighted_metric(ap_fn, e["base"], e["gold"])
                sft = float(np.mean([C.weighted_metric(ap_fn, e["sft"][s], e["gold"]) for s in seeds]))
                deltas.append(sft - base)
                sign_row[f"{mk}:{b}"] = int(np.sign(sft - base))
            pbench[b] = float(np.mean(deltas)) if deltas else float("nan")
        out["per_benchmark_delta"][regime] = pbench
        vv = [v for v in pbench.values() if not math.isnan(v)]
        out["range_across_benchmarks"][regime] = {
            "min": float(min(vv)) if vv else float("nan"),
            "max": float(max(vv)) if vv else float("nan"),
            "range": float(max(vv) - min(vv)) if vv else float("nan"),
            "std": float(np.std(vv, ddof=1)) if len(vv) > 1 else 0.0}
        out["sign_table"][regime] = sign_row
    return out


def _sign_stable(full, loos):
    fs = np.sign(full)
    if fs == 0:
        return False
    return all(np.sign(v) == fs for v in loos if not (isinstance(v, float) and math.isnan(v)))


# --------------------------------------------------------------------------------------
# secondary 5% FPR operating point (RQ4) + one-class stress (plan sec 10.5, 11.2)
# --------------------------------------------------------------------------------------
def operating_point(df, present, model_keys, seeds):
    def tpr_fpr(sub):
        g = sub["gold"].to_numpy(int); p = sub["prediction"].to_numpy(int)
        pos = g == 1; neg = g == 0
        tpr = float(p[pos].mean()) if pos.any() else float("nan")
        fpr = float(p[neg].mean()) if neg.any() else float("nan")
        return tpr, fpr

    def macro(mk, condition, seed, benches, split):
        tprs, fprs = [], []
        for b in benches:
            sub = df[(df.model_key == mk) & (df.condition == condition) & (df.split == split)
                     & (df.source == b) & ((df.seed == seed) if condition == "sft" else True)]
            if sub.empty:
                continue
            t, f = tpr_fpr(sub)
            tprs.append(t); fprs.append(f)
        return (float(np.nanmean(tprs)) if tprs else float("nan"),
                float(np.nanmean(fprs)) if fprs else float("nan"))

    out = {}
    for regime, benches in present.items():
        split = AP_SPLITS[regime]
        base_t, base_f, sft_t, sft_f = [], [], [], []
        for mk in model_keys:
            bt, bf = macro(mk, "base", None, benches, split)
            base_t.append(bt); base_f.append(bf)
            st = [macro(mk, "sft", s, benches, split) for s in seeds]
            sft_t.append(float(np.nanmean([x[0] for x in st])))
            sft_f.append(float(np.nanmean([x[1] for x in st])))
        out[regime] = {
            "base_macro_tpr": float(np.nanmean(base_t)), "base_macro_fpr": float(np.nanmean(base_f)),
            "sft_macro_tpr": float(np.nanmean(sft_t)), "sft_macro_fpr": float(np.nanmean(sft_f)),
            "delta_tpr": float(np.nanmean(sft_t) - np.nanmean(base_t)),
            "realized_fpr_note": "calibration-targeted; realized test FPR is reported, not assumed",
        }
    return out


def stress_metrics(df, model_keys, seeds):
    def rate(split, mk, condition, seed):
        sub = df[(df.model_key == mk) & (df.condition == condition) & (df.split == split)
                 & ((df.seed == seed) if condition == "sft" else True)]
        return float(sub["prediction"].mean()) if len(sub) else float("nan")
    out = {}
    for name, split in (("orbench_benign_fpr", "stress_orbench"),
                        ("harmbench_recall", "stress_harmbench")):
        base = float(np.nanmean([rate(split, mk, "base", None) for mk in model_keys]))
        sft = float(np.nanmean([np.nanmean([rate(split, mk, "sft", s) for s in seeds])
                                for mk in model_keys]))
        out[name] = {"base_panel_mean": base, "sft_panel_mean": sft, "one_class": True,
                     "note": "single-class stress set; NO AP/AUROC computed"}
    return out


# --------------------------------------------------------------------------------------
# claim gates (plan sec 16) + Holm multiplicity
# --------------------------------------------------------------------------------------
def claim_checks(boot, sens, analysis_mode):
    rep = boot["aggregate"]["represented"]
    tr = boot["aggregate"]["transfer"]
    precision = (analysis_mode == "precision_focused")

    gate_a = rep["lcb95_one_sided"] > 0
    loo_b = sens["leave_one_benchmark_out"]["transfer"]["sign_stable"]
    loo_base = sens["leave_one_base_out"]["transfer"]["sign_stable"]
    full_tr_neg = sens["leave_one_benchmark_out"]["transfer"]["full"] < 0
    gate_b = (tr["ucb95_one_sided"] < 0) and loo_b and loo_base and full_tr_neg
    specialization = gate_a and gate_b

    # Holm across the two one-sided components (informational proxy p-values)
    p_a = rep["p_le_0"]       # H0: represented delta <= 0
    p_b = tr["p_ge_0"]        # H0: transfer delta >= 0
    holm = _holm({"gate_a": p_a, "gate_b": p_b}, alpha=0.05)

    def word(passed, est, ci, kind):
        if precision:
            if kind == "represented":
                return (f"the estimated represented-source macro-AP change was {est:+.4f} "
                        f"(95% one-sided LCB {ci:+.4f})")
            return (f"the estimated transfer macro-AP change was {est:+.4f} "
                    f"(95% one-sided UCB {ci:+.4f})")
        if kind == "represented":
            return ("For this fixed panel and recipe, SFT improved represented-source macro AP."
                    if passed else "The study did not establish a panel-wide represented-source improvement.")
        return ("For this fixed panel and recipe, SFT reduced dataset-held-out transfer macro AP."
                if passed else "The study reports the measured transfer heterogeneity (Gate B not met).")

    if precision:
        spec_word = ("estimation-only mode: report the joint (represented, transfer) estimate "
                     "and intervals; no formal specialization rejection is claimed")
    elif specialization:
        spec_word = ("The fixed panel exhibits an in-source/transfer specialization trade-off "
                     "under the frozen LoRA-SFT recipe.")
    else:
        spec_word = "Specialization trade-off not established (intersection-union not satisfied)."

    return {
        "analysis_mode": analysis_mode,
        "precision_focused_language": precision,
        "gate_a": {"criterion": "LCB95(mean_delta_represented) > 0",
                   "represented_delta_mean": rep["mean"],
                   "lcb95": rep["lcb95_one_sided"], "passed": bool(gate_a),
                   "wording": word(gate_a, rep["mean"], rep["lcb95_one_sided"], "represented")},
        "gate_b": {"criterion": "UCB95(mean_delta_transfer) < 0 AND LOO sign-stable",
                   "transfer_delta_mean": tr["mean"], "ucb95": tr["ucb95_one_sided"],
                   "full_transfer_negative": bool(full_tr_neg),
                   "loo_benchmark_sign_stable": bool(loo_b),
                   "loo_base_sign_stable": bool(loo_base), "passed": bool(gate_b),
                   "wording": word(gate_b, tr["mean"], tr["ucb95_one_sided"], "transfer")},
        "specialization": {"criterion": "gate_a AND gate_b (intersection-union, alpha 0.05)",
                           "passed": bool(specialization), "wording": spec_word},
        "holm_two_component": holm,
        "rq4": {"status": "descriptive_only",
                "note": "operating-point TPR/FPR is a deployment diagnostic, not a confirmatory test"},
    }


def _holm(pvals: dict, alpha: float):
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out = {}; reject_further = True
    for i, (name, p) in enumerate(ordered):
        thresh = alpha / (m - i)
        rej = reject_further and (p <= thresh)
        if not rej:
            reject_further = False
        out[name] = {"p": float(p), "holm_threshold": float(thresh), "reject": bool(rej)}
    return {"alpha": alpha, "components": out,
            "note": "bootstrap crossing-fraction proxies for one-sided component nulls"}


# --------------------------------------------------------------------------------------
# emit: csv / json / latex tables / figure
# --------------------------------------------------------------------------------------
def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def write_seed_values_csv(path, point):
    import pandas as pd
    pd.DataFrame(point["seed_values"]).to_csv(path, index=False)


def write_per_benchmark_csv(path, data, present, model_keys, seeds, ap_fn, auroc_fn):
    import pandas as pd
    rows = []
    for regime, benches in present.items():
        for mk in model_keys:
            for b in benches:
                e = data[mk].get(b)
                if e is None:
                    continue
                for condition, seed in ([("base", None)] + [("sft", s) for s in seeds]):
                    scores = e["base"] if condition == "base" else e["sft"][seed]
                    rows.append({
                        "model_key": mk, "condition": condition,
                        "seed": (seed if seed is not None else -1),
                        "benchmark": b, "regime": regime,
                        "ap": C.weighted_metric(ap_fn, scores, e["gold"]),
                        "auroc": auroc_fn(scores, e["gold"]),
                        "n": e["n"], "n_pos": int(e["gold"].sum()),
                        "n_neg": int((e["gold"] == 0).sum())})
    pd.DataFrame(rows).to_csv(path, index=False)


def write_table3(path, point, boot, model_keys):
    lines = [r"% Auto-generated by analyze_paper_a_sft.py -- do not edit by hand.",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Checkpoint & Rep base & Rep SFT & $\Delta$ Rep [95\% CI] "
             r"& Tr base & Tr SFT & $\Delta$ Tr [95\% CI] \\", r"\midrule"]
    for mk in model_keys:
        rc = point["per_checkpoint"]["represented"][mk]
        tc = point["per_checkpoint"]["transfer"][mk]
        rb = boot["per_checkpoint"]["represented"][mk]
        tb = boot["per_checkpoint"]["transfer"][mk]
        lines.append(
            f"{_tex(mk)} & {_fmt(rc['base'])} & {_fmt(rc['sft_mean'])} & "
            f"{_fmt(rc['delta'])} [{_fmt(rb['ci95_two_sided'][0])}, {_fmt(rb['ci95_two_sided'][1])}] & "
            f"{_fmt(tc['base'])} & {_fmt(tc['sft_mean'])} & "
            f"{_fmt(tc['delta'])} [{_fmt(tb['ci95_two_sided'][0])}, {_fmt(tb['ci95_two_sided'][1])}] \\\\")
    lines.append(r"\midrule")
    ar = boot["aggregate"]["represented"]; at = boot["aggregate"]["transfer"]
    lines.append(
        f"Fixed-panel aggregate & -- & -- & {_fmt(ar['mean'])} "
        f"[{_fmt(ar['ci95_two_sided'][0])}, {_fmt(ar['ci95_two_sided'][1])}] & -- & -- & "
        f"{_fmt(at['mean'])} [{_fmt(at['ci95_two_sided'][0])}, {_fmt(at['ci95_two_sided'][1])}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_text(path, "\n".join(lines))


def write_table4(path, sens, opr, stress, present):
    lines = [r"% Auto-generated by analyze_paper_a_sft.py -- do not edit by hand.",
             r"\begin{tabular}{llr}", r"\toprule",
             r"Regime / benchmark & Metric & Value \\", r"\midrule"]
    for regime in present:
        for b, d in sens["per_benchmark_delta"][regime].items():
            lines.append(f"{_tex(regime)} / {_tex(b)} & paired $\\Delta$AP & {_fmt(d)} \\\\")
    lines.append(r"\midrule")
    for regime in present:
        o = opr[regime]
        lines.append(f"{_tex(regime)} & SFT TPR@target FPR & {_fmt(o['sft_macro_tpr'])} \\\\")
        lines.append(f"{_tex(regime)} & realized FPR (SFT) & {_fmt(o['sft_macro_fpr'])} \\\\")
        lines.append(f"{_tex(regime)} & realized FPR (base) & {_fmt(o['base_macro_fpr'])} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Stress & OR-Bench benign FPR (SFT) & {_fmt(stress['orbench_benign_fpr']['sft_panel_mean'])} \\\\")
    lines.append(f"Stress & HarmBench recall (SFT) & {_fmt(stress['harmbench_recall']['sft_panel_mean'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_text(path, "\n".join(lines))


def write_specialization_figure(path, point, model_keys, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    cmap = plt.get_cmap("tab10")
    for i, mk in enumerate(model_keys):
        xs = [point["per_checkpoint"]["represented"][mk]["seed_deltas"][s] for s in seeds]
        ys = [point["per_checkpoint"]["transfer"][mk]["seed_deltas"][s] for s in seeds]
        ax.scatter(xs, ys, color=cmap(i), s=36, alpha=0.75, label=mk, edgecolors="none")
    ax.scatter([point["aggregate"]["represented"]], [point["aggregate"]["transfer"]],
               marker="X", s=180, color="black", label="fixed-panel mean", zorder=5)
    ax.axhline(0, color="0.5", lw=0.8); ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel(r"represented-source macro-AP $\Delta$")
    ax.set_ylabel(r"transfer macro-AP $\Delta$")
    ax.set_title("Specialization plane (per seed)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(path, format="pdf"); plt.close(fig)


def _tex(s): return str(s).replace("_", r"\_")


def _write_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def run_analysis(df, lock, out_dir, ap_fn, auroc_fn, reps=None, rng_seed=None):
    regimes = lock.get("regime_benchmarks", C.REGIME_BENCHMARKS)
    model_keys = [mk for mk in C.MODEL_KEYS if mk in set(df["model_key"].unique())]
    seeds = C.lock_seeds(lock)
    reps = int(reps if reps is not None else lock.get("resampling_rules", {}).get("replicates",
                                                                                 C.DEFAULT_BOOTSTRAP_REPLICATES))
    rng_seed = int(rng_seed if rng_seed is not None else lock.get("resampling_rules", {}).get(
        "rng_seed", C.DEFAULT_BOOTSTRAP_SEED))
    analysis_mode = lock.get("analysis_mode", "precision_focused")

    data, families, present = build_bench_data(df, regimes, model_keys, seeds)
    point = point_estimates(data, present, model_keys, seeds, ap_fn)
    boot = hierarchical_bootstrap(data, present, model_keys, seeds, ap_fn, reps, rng_seed)
    sens = sensitivity(data, present, model_keys, seeds, ap_fn, point)
    opr = operating_point(df, present, model_keys, seeds)
    stress = stress_metrics(df, model_keys, seeds)
    checks = claim_checks(boot, sens, analysis_mode)

    os.makedirs(out_dir, exist_ok=True)
    tables = os.path.join(out_dir, "tables"); figures = os.path.join(out_dir, "figures")
    os.makedirs(tables, exist_ok=True); os.makedirs(figures, exist_ok=True)

    results = {
        "analysis_code_version": ANALYSIS_CODE_VERSION, "created_utc": C.utcnow(),
        "lock_sha256": lock.get("lock_sha256"), "analysis_mode": analysis_mode,
        "model_keys": model_keys, "seeds": seeds,
        "benchmarks_present": present, "n_families": len(families),
        "point_estimates": {"per_checkpoint": point["per_checkpoint"],
                            "aggregate": point["aggregate"]},
        "bootstrap": boot, "operating_point": opr, "stress": stress,
    }
    C.write_json(os.path.join(out_dir, "results.json"), results)
    C.write_json(os.path.join(out_dir, "sensitivity.json"), sens)
    C.write_json(os.path.join(out_dir, "claim_checks.json"), checks)
    write_seed_values_csv(os.path.join(out_dir, "seed_values.csv"), point)
    write_per_benchmark_csv(os.path.join(out_dir, "per_benchmark.csv"),
                            data, present, model_keys, seeds, ap_fn, auroc_fn)
    write_table3(os.path.join(tables, "table3_primary.tex"), point, boot, model_keys)
    write_table4(os.path.join(tables, "table4_per_benchmark.tex"), sens, opr, stress, present)
    write_specialization_figure(os.path.join(figures, "specialization_plane.pdf"),
                                point, model_keys, seeds)
    return results, checks, sens, point, boot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Paper A analysis (plan sec 12/15/16).")
    ap.add_argument("--lock", default=None)
    ap.add_argument("--scores", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bootstrap-reps", type=int, default=None)
    ap.add_argument("--bootstrap-seed", type=int, default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic end-to-end check of bootstrap + gates + emitters")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test(args)

    if not (args.lock and args.scores and args.out):
        ap.error("--lock, --scores and --out are required (or use --self-test)")
    import pandas as pd
    lock = C.load_lock(args.lock)
    df = pd.read_parquet(args.scores)
    ap_fn, auroc_fn = C.require_metrics()
    out_dir = C.abspath(args.out) if not os.path.isabs(args.out) else args.out
    _, checks, _, _, boot = run_analysis(df, lock, out_dir, ap_fn, auroc_fn,
                                         reps=args.bootstrap_reps, rng_seed=args.bootstrap_seed)
    print(f"[analyze] wrote results/seed_values/per_benchmark/sensitivity/claim_checks + tables + figure to {out_dir}")
    print(f"[analyze] represented delta mean={boot['aggregate']['represented']['mean']:+.4f} "
          f"LCB={boot['aggregate']['represented']['lcb95_one_sided']:+.4f}")
    print(f"[analyze] transfer   delta mean={boot['aggregate']['transfer']['mean']:+.4f} "
          f"UCB={boot['aggregate']['transfer']['ucb95_one_sided']:+.4f}")
    print(f"[analyze] gate_a={checks['gate_a']['passed']} gate_b={checks['gate_b']['passed']} "
          f"specialization={checks['specialization']['passed']} (mode={checks['analysis_mode']})")
    return 0


# --------------------------------------------------------------------------------------
# self-test: fabricate a scores DataFrame with a KNOWN effect and check the machinery
# --------------------------------------------------------------------------------------
def _synthetic_scores_df(effect_rep=+0.9, effect_tr=-0.9, seeds=(42, 43, 44, 45, 46),
                         n_per=60, rng_seed=0):
    import pandas as pd
    rng = np.random.default_rng(rng_seed)
    regimes = {"represented": (["toxicchat", "prompt_injections", "jailbreak_classification"], "id_test"),
               "transfer": (["jailbreakbench", "xstest", "wildguardtest", "wildjailbreak"], "transfer_test")}
    rows = []
    for mk in C.MODEL_KEYS:
        base_shift = rng.normal(0, 0.05)
        for regime, (benches, split) in regimes.items():
            eff = effect_rep if regime == "represented" else effect_tr
            for b in benches:
                for i in range(n_per):
                    gold = i % 2
                    fam = f"{b}_fam_{i % (n_per // 3)}"
                    sid = f"{b}_{i}"
                    csha = f"{b}:{i}"
                    base_score = (2 * gold - 1) * (1.0 + base_shift) + rng.normal(0, 1.0)
                    rows.append(_row(sid, csha, b, split, gold, fam, mk, "base", -1, base_score))
                    for s in seeds:
                        sft_score = base_score + eff * (2 * gold - 1) + rng.normal(0, 0.15)
                        rows.append(_row(sid, csha, b, split, gold, fam, mk, "sft", s, sft_score))
    # stress rows (one-class) for RQ4/stress emitters
    for mk in C.MODEL_KEYS:
        for i in range(20):
            rows.append(_row(f"orb_{i}", f"orb:{i}", "orbench", "stress_orbench", 0,
                             f"orb_{i}", mk, "base", -1, rng.normal(-1, 1), pred=0))
            rows.append(_row(f"hb_{i}", f"hb:{i}", "harmbench", "stress_harmbench", 1,
                             f"hb_{i}", mk, "base", -1, rng.normal(1, 1), pred=1))
            for s in seeds:
                rows.append(_row(f"orb_{i}", f"orb:{i}", "orbench", "stress_orbench", 0,
                                 f"orb_{i}", mk, "sft", s, rng.normal(-1, 1), pred=0))
                rows.append(_row(f"hb_{i}", f"hb:{i}", "harmbench", "stress_harmbench", 1,
                                 f"hb_{i}", mk, "sft", s, rng.normal(1, 1), pred=1))
    return pd.DataFrame(rows)


def _row(sid, csha, src, split, gold, fam, mk, cond, seed, score, pred=None):
    prob = 1.0 / (1.0 + math.exp(-score))
    return {"sample_id": sid, "content_sha256": csha, "source": src, "split": split,
            "gold": gold, "family_id": fam, "model_key": mk, "model_revision": "rev",
            "condition": cond, "seed": seed, "adapter_sha256": None, "prompt_sha256": "p",
            "safe_token_id": 0, "unsafe_token_id": 1, "safe_logit": -score / 2,
            "unsafe_logit": score / 2, "score_raw": score, "probability_raw": prob,
            "probability_calibrated": prob, "threshold_id": "t", "prediction": (int(prob >= 0.5) if pred is None else pred),
            "original_token_count": 5, "scored_token_count": 5, "truncated": False, "latency_ms": 0.0}


def _self_test(args) -> int:
    import tempfile, json
    ap_fn, auroc_fn = C.require_metrics()
    lock = {"analysis_mode": "precision_focused", "seeds": [42, 43, 44, 45, 46],
            "regime_benchmarks": C.REGIME_BENCHMARKS,
            "resampling_rules": {"replicates": args.bootstrap_reps or 300, "rng_seed": 20260712},
            "lock_sha256": "selftest"}
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"); ok = ok and bool(cond)

    print("== self-test A: positive represented, negative transfer (specialization) ==")
    df = _synthetic_scores_df(+0.9, -0.9)
    out = tempfile.mkdtemp()
    results, checks, sens, point, boot = run_analysis(df, lock, out, ap_fn, auroc_fn,
                                                      reps=args.bootstrap_reps or 300)
    check("represented aggregate > 0", boot["aggregate"]["represented"]["mean"] > 0)
    check("transfer aggregate < 0", boot["aggregate"]["transfer"]["mean"] < 0)
    check("gate_a passed (LCB>0)", checks["gate_a"]["passed"])
    check("gate_b passed (UCB<0 + sign-stable)", checks["gate_b"]["passed"])
    check("specialization passed", checks["specialization"]["passed"])
    check("transfer LOO-benchmark sign-stable", sens["leave_one_benchmark_out"]["transfer"]["sign_stable"])
    check("precision_focused estimation language",
          "estimated" in checks["gate_a"]["wording"])
    for fn in ("results.json", "seed_values.csv", "per_benchmark.csv", "sensitivity.json",
               "claim_checks.json", "tables/table3_primary.tex", "tables/table4_per_benchmark.tex",
               "figures/specialization_plane.pdf"):
        check(f"emitted {fn}", os.path.exists(os.path.join(out, fn)))

    print("== self-test B: null effect (no gates) ==")
    df0 = _synthetic_scores_df(0.0, 0.0, rng_seed=7)
    out0 = tempfile.mkdtemp()
    _, checks0, _, _, boot0 = run_analysis(df0, lock, out0, ap_fn, auroc_fn,
                                           reps=args.bootstrap_reps or 300)
    check("null: gate_a not passed", not checks0["gate_a"]["passed"])
    check("null: gate_b not passed", not checks0["gate_b"]["passed"])
    check("null: specialization not passed", not checks0["specialization"]["passed"])

    print("== self-test C: reproducibility (same seed -> identical aggregate) ==")
    _, _, _, _, boot_r = run_analysis(_synthetic_scores_df(+0.9, -0.9), lock,
                                      tempfile.mkdtemp(), ap_fn, auroc_fn, reps=300)
    check("bootstrap deterministic w/ fixed rng_seed",
          abs(boot_r["aggregate"]["represented"]["lcb95_one_sided"]
              - boot["aggregate"]["represented"]["lcb95_one_sided"]) < 1e-9)

    print("== self-test D: weighted (replicated) AP == canonical AP when weights all 1 ==")
    s = np.array([0.1, 0.9, 0.4, 0.4, 0.8, 0.2]); y = np.array([0, 1, 0, 1, 1, 0])
    a1 = ap_fn(s, y); a2 = C.weighted_metric(ap_fn, s, y, np.ones_like(y))
    check("weighted_metric(all-ones) == average_precision", abs(a1 - a2) < 1e-12)

    print(f"\nSELF-TEST {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
