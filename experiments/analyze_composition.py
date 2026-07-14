#!/usr/bin/env python
"""Composition ("Compose, Don't Tune") analysis: base + tuned guard ensembles.

Reproducibly regenerates the numbers behind the composition result from the committed
row-keyed score table. For each of the four checkpoints it evaluates the untuned base,
the SFT adapter (seed-mean), and several *composed* guards (combine the base and SFT
per-row scores), on the represented and dataset-held-out transfer regimes.

Outputs (into --out):
  - composition.json     : point estimates (all combiners), bootstrap CIs (primary
                           combiner), leave-one-benchmark-out, matched-FPR operating
                           point, and the complementarity shuffle-null control;
  - composition.md       : human-readable summary tables.

Metric: benchmark-macro tie-aware Average Precision (guard_research.metrics), macro over
a regime's benchmarks then mean over the 4-checkpoint panel (SFT/ensemble also over seeds).
Uncertainty: the same hierarchical PAIRED bootstrap as analyze_paper_a_sft.py -- 4
checkpoints fixed; resample the 5 SFT seed indices within each checkpoint; one Poisson(1)
weight per GLOBAL family_id; weighted tie-aware AP -> macro -> panel; percentile CIs.

IMPORTANT: this reads the LEGACY score artifact, so results are ESTIMATION-ONLY (a clean
rerun is required for confirmatory use). WiSE-FT weight interpolation is out of scope --
it needs the adapter weights, which are not present in the score table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score  # canonical tie-aware AP (weighted form)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from guard_research.metrics import average_precision  # noqa: E402  (scores, labels)
from guard_research.thresholds import select_threshold  # noqa: E402

MODELS = ["qwen25_15b", "smollm2_17b", "smollm3_3b", "qwen3_4b"]
REP = ["toxicchat", "prompt_injections", "jailbreak_classification"]
TR = ["jailbreakbench", "xstest", "wildguardtest", "wildjailbreak"]
REGIMES = {"represented": ("id_test", REP), "transfer": ("transfer_test", TR)}
CAL_SPLIT = "calibration"


# ----------------------------------------------------------------------------- weighted AP
def wap(scores, labels, weights=None):
    """Weighted tie-aware AP. Unweighted path uses the canonical guard_research wrapper;
    the weighted path calls the same sklearn function with sample_weight (Poisson family
    weights). NaN if single-class."""
    y = np.asarray(labels, float)
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    if weights is None:
        return average_precision(scores, labels)
    return float(average_precision_score(y, np.asarray(scores, float), sample_weight=np.asarray(weights, float)))


# ----------------------------------------------------------------------------- data
def load(scores_path):
    df = pd.read_parquet(scores_path)
    df = df[df["gold"].isin([0, 1])].copy()
    df["gold"] = df["gold"].astype(int)
    df["seed"] = pd.to_numeric(df["seed"]).astype(int)
    return df


def build(df, seeds):
    """data[model][split][source] = dict(gold, fam(str[]), base{cal,raw,logit},
    sft{seed:{cal,raw,logit}}), all aligned to the base row order (by sample_id)."""
    data = {}
    for mk in MODELS:
        data[mk] = {}
        for split in ("id_test", "transfer_test", CAL_SPLIT):
            data[mk][split] = {}
            srcs = REP if split in ("id_test", CAL_SPLIT) else TR
            for src in srcs:
                b = df[(df.model_key == mk) & (df.condition == "base") & (df.split == split) & (df.source == src)]
                b = b.sort_values("sample_id")
                if b.empty:
                    continue
                order = b["sample_id"].tolist()
                entry = {
                    "gold": b["gold"].to_numpy(int),
                    "fam": [str(f) for f in b["family_id"]],
                    "base": {"cal": b["probability_calibrated"].to_numpy(float),
                             "raw": b["probability_raw"].to_numpy(float),
                             "logit": b["score_raw"].to_numpy(float)},
                    "sft": {},
                }
                for s in seeds:
                    sf = df[(df.model_key == mk) & (df.condition == "sft") & (df.seed == s)
                            & (df.split == split) & (df.source == src)].set_index("sample_id").reindex(order)
                    entry["sft"][s] = {"cal": sf["probability_calibrated"].to_numpy(float),
                                       "raw": sf["probability_raw"].to_numpy(float),
                                       "logit": sf["score_raw"].to_numpy(float)}
                data[mk][split][src] = entry
    return data


# ----------------------------------------------------------------------------- combiners
def sft_seedmean(entry, field, seeds):
    return np.mean([entry["sft"][s][field] for s in seeds], axis=0)


def combiner_score(entry, s, name, pit=None):
    """Composed per-row score for combiner `name` on this benchmark entry, for ONE SFT
    seed `s`. (Seed averaging happens at the AP level in `macro`, matching Paper A's
    mean-of-per-seed-AP estimand -- i.e. base composed with a *single* adapter.)"""
    b = entry["base"]
    if name == "base":
        return b["cal"]
    sft = entry["sft"][s]
    if name == "sft":
        return sft["cal"]
    if name == "calibrated_avg":
        return 0.5 * (b["cal"] + sft["cal"])
    if name == "raw_avg":
        return 0.5 * (b["raw"] + sft["raw"])
    if name == "logit_avg":
        return 0.5 * (b["logit"] + sft["logit"])
    if name == "max_cal":
        return np.maximum(b["cal"], sft["cal"])
    if name == "pit_avg":
        fb, fs = pit
        return 0.5 * (fb(b["logit"]) + fs(sft["logit"]))
    if name.startswith("convex:"):
        w = float(name.split(":")[1])
        return w * sft["cal"] + (1.0 - w) * b["cal"]
    raise ValueError(name)


def macro(data, mk, split, sources, seeds, name, pit=None):
    """Benchmark-macro AP for guard `name`: per source, mean-of-per-seed AP (base has no
    seeds); then mean over sources. Equivalent to Paper A's seed-mean-of-macro-AP."""
    vals = []
    for src in sources:
        e = data[mk][split].get(src)
        if e is None:
            continue
        p = pit.get((mk, src)) if pit else None
        if name == "base":
            ap = wap(e["base"]["cal"], e["gold"])
        else:
            aps = [wap(combiner_score(e, s, name, pit=p), e["gold"]) for s in seeds]
            aps = [a for a in aps if not math.isnan(a)]
            ap = float(np.mean(aps)) if aps else float("nan")
        if not math.isnan(ap):
            vals.append(ap)
    return float(np.mean(vals)) if vals else float("nan")


def panel(data, split, sources, seeds, name, pit=None):
    return float(np.mean([macro(data, mk, split, sources, seeds, name, pit=pit) for mk in MODELS]))


# ----------------------------------------------------------------------------- PIT (leak-free)
def fit_pit(data, seeds):
    """Empirical-CDF maps fit on the CALIBRATION split only (leak-free): for each model,
    one map for the base logit and one for the seed-mean SFT logit."""
    pit = {}
    for mk in MODELS:
        base_c, sft_c = [], []
        for src in REP:
            e = data[mk][CAL_SPLIT].get(src)
            if e is None:
                continue
            base_c.append(e["base"]["logit"])
            sft_c.append(sft_seedmean(e, "logit", seeds))
        if not base_c:
            continue
        bs = np.sort(np.concatenate(base_c))
        ss = np.sort(np.concatenate(sft_c))
        fb = lambda x, bs=bs: np.searchsorted(bs, np.asarray(x, float), side="right") / max(len(bs), 1)
        fs = lambda x, ss=ss: np.searchsorted(ss, np.asarray(x, float), side="right") / max(len(ss), 1)
        for src in REP + TR:
            pit[(mk, src)] = (fb, fs)
    return pit


# ----------------------------------------------------------------------------- point estimates
def point_estimates(data, seeds, combiners, pit):
    out = {}
    for name in combiners:
        out[name] = {}
        for regime, (split, srcs) in REGIMES.items():
            per_model = {mk: macro(data, mk, split, srcs, seeds, name, pit=pit) for mk in MODELS}
            out[name][regime] = {"per_model": per_model, "panel": float(np.mean(list(per_model.values())))}
    return out


def select_convex_w(data, seeds):
    """Transfer-blind: pick w maximizing REPRESENTED panel macro-AP (never touches transfer)."""
    best_w, best = 0.0, -1.0
    for w in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        m = panel(data, "id_test", REP, seeds, f"convex:{w}")
        if m > best:
            best, best_w = m, float(w)
    return best_w


# ----------------------------------------------------------------------------- bootstrap
def bootstrap(data, seeds, reps, rng_seed, name="calibrated_avg"):
    """Paired hierarchical bootstrap of the composed guard's advantage. Poisson(1) weight
    per global family + resample seed indices within each checkpoint. Reports, per regime,
    per-model and panel percentile CIs for (ensemble - SFT) and (ensemble - base)."""
    rng = np.random.default_rng(rng_seed)
    fams = sorted({f for mk in MODELS for split in ("id_test", "transfer_test")
                   for e in data[mk][split].values() for f in e["fam"]})
    fam_idx = {f: i for i, f in enumerate(fams)}
    n_fam = len(fams)
    for mk in MODELS:
        for split in ("id_test", "transfer_test"):
            for e in data[mk][split].values():
                e["_fi"] = np.array([fam_idx[f] for f in e["fam"]], int)
    ns = len(seeds)

    entries = [(mk, split, e) for mk in MODELS for split in ("id_test", "transfer_test")
               for e in data[mk][split].values()]

    def valid(w):
        # only benchmarks that actually contain both classes must keep both classes weighted;
        # a genuinely single-class benchmark (none here) is skipped rather than hanging forever.
        for _mk, _sp, e in entries:
            g, fi = e["gold"], e["_fi"]
            has_pos, has_neg = (g == 1).any(), (g == 0).any()
            if has_pos and has_neg and (w[fi[g == 1]].sum() <= 0 or w[fi[g == 0]].sum() <= 0):
                return False
        return True

    keys = ["ens_minus_sft", "ens_minus_base"]
    samp = {r: {k: {**{mk: np.empty(reps) for mk in MODELS}, "panel": np.empty(reps)} for k in keys}
            for r in REGIMES}

    def bench_macro_weighted(mk, split, srcs, scfn, w):
        vals = []
        for src in srcs:
            e = data[mk][split].get(src)
            if e is None:
                continue
            vals.append(wap(scfn(e), e["gold"], weights=w[e["_fi"]]))
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    redraws = 0
    for rep in range(reps):
        tries = 0
        while True:
            w = rng.poisson(1.0, size=n_fam).astype(float)
            if valid(w):
                break
            redraws += 1
            tries += 1
            if tries > 2000:
                raise RuntimeError("bootstrap: exceeded redraw cap (data too sparse?)")
        pick = {mk: rng.integers(0, ns, size=ns) for mk in MODELS}
        for regime, (split, srcs) in REGIMES.items():
            d_es, d_eb = [], []
            for mk in MODELS:
                base_M = bench_macro_weighted(mk, split, srcs, lambda e: e["base"]["cal"], w)
                sft_seed, ens_seed = {}, {}
                for s in seeds:
                    sft_seed[s] = bench_macro_weighted(mk, split, srcs, lambda e, s=s: e["sft"][s]["cal"], w)
                    ens_seed[s] = bench_macro_weighted(
                        mk, split, srcs, lambda e, s=s: 0.5 * (e["base"]["cal"] + e["sft"][s]["cal"]), w)
                sft_M = float(np.mean([sft_seed[seeds[j]] for j in pick[mk]]))
                ens_M = float(np.mean([ens_seed[seeds[j]] for j in pick[mk]]))
                samp[regime]["ens_minus_sft"][mk][rep] = ens_M - sft_M
                samp[regime]["ens_minus_base"][mk][rep] = ens_M - base_M
                d_es.append(ens_M - sft_M)
                d_eb.append(ens_M - base_M)
            samp[regime]["ens_minus_sft"]["panel"][rep] = float(np.mean(d_es))
            samp[regime]["ens_minus_base"]["panel"][rep] = float(np.mean(d_eb))

    def summ(a):
        return {"mean": float(np.mean(a)), "std": float(np.std(a, ddof=1)),
                "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                "lcb95": float(np.percentile(a, 5)), "ucb95": float(np.percentile(a, 95))}

    out = {"reps": reps, "rng_seed": rng_seed, "n_families": n_fam, "redraws": redraws, "combiner": name}
    for regime in REGIMES:
        out[regime] = {k: {"panel": summ(samp[regime][k]["panel"]),
                           "per_model": {mk: summ(samp[regime][k][mk]) for mk in MODELS}} for k in keys}
    return out


# ----------------------------------------------------------------------------- leave-one-benchmark-out
def loo_benchmark(data, seeds, name="calibrated_avg", pit=None):
    out = {}
    for regime, (split, srcs) in REGIMES.items():
        full_eb = panel(data, split, srcs, seeds, name, pit) - panel(data, split, srcs, seeds, "base", pit)
        full_es = panel(data, split, srcs, seeds, name, pit) - panel(data, split, srcs, seeds, "sft", pit)
        loo = {}
        for drop in srcs:
            keep = [s for s in srcs if s != drop]
            eb = panel(data, split, keep, seeds, name, pit) - panel(data, split, keep, seeds, "base", pit)
            es = panel(data, split, keep, seeds, name, pit) - panel(data, split, keep, seeds, "sft", pit)
            loo[drop] = {"ens_minus_base": eb, "ens_minus_sft": es}
        out[regime] = {"full": {"ens_minus_base": full_eb, "ens_minus_sft": full_es}, "loo": loo,
                       "ens_minus_base_sign_stable": all(np.sign(v["ens_minus_base"]) == np.sign(full_eb) for v in loo.values()),
                       "ens_minus_sft_sign_stable": all(np.sign(v["ens_minus_sft"]) == np.sign(full_es) for v in loo.values())}
    return out


# ----------------------------------------------------------------------------- matched-FPR operating point
def operating_point(data, seeds, target_fpr, name="calibrated_avg"):
    def cal_scores(mk, scfn):
        s, y = [], []
        for src in REP:
            e = data[mk][CAL_SPLIT].get(src)
            if e is None:
                continue
            s.append(scfn(e)); y.append(e["gold"])
        return np.concatenate(s), np.concatenate(y)

    scfns = {
        "base": lambda e: e["base"]["cal"],
        "sft": lambda e: sft_seedmean(e, "cal", seeds),
        name: lambda e: 0.5 * (e["base"]["cal"] + sft_seedmean(e, "cal", seeds)),
    }
    out = {"target_fpr": target_fpr}
    for guard, scfn in scfns.items():
        thr = {mk: select_threshold(*cal_scores(mk, scfn), target_fpr=target_fpr) for mk in MODELS}
        g = {}
        for regime, (split, srcs) in REGIMES.items():
            tpr_m, fpr_m = [], []
            fp = tp = nneg = npos = 0
            for mk in MODELS:
                t = thr[mk].get("threshold")
                if t is None:
                    continue
                for src in srcs:
                    e = data[mk][split].get(src)
                    if e is None:
                        continue
                    sc = scfn(e); gd = e["gold"]
                    pos, neg = gd == 1, gd == 0
                    if pos.sum():
                        tpr_m.append(float((sc[pos] >= t).mean()))
                    if neg.sum():
                        fpr_m.append(float((sc[neg] >= t).mean()))
                    tp += int((sc[pos] >= t).sum()); npos += int(pos.sum())
                    fp += int((sc[neg] >= t).sum()); nneg += int(neg.sum())
            g[regime] = {"macro_tpr": float(np.mean(tpr_m)) if tpr_m else float("nan"),
                         "macro_fpr": float(np.mean(fpr_m)) if fpr_m else float("nan"),
                         "pooled_tpr": tp / npos if npos else float("nan"),
                         "pooled_fpr": fp / nneg if nneg else float("nan")}
        out[guard] = g
    return out


# ----------------------------------------------------------------------------- shuffle-null (complementarity)
def shuffle_null(data, seeds, rng_seed=20260714, name="calibrated_avg"):
    """Two null controls on the panel ens-base delta (seed-mean SFT; coarse diagnostic):

    - 'signal_null' permutes the SFT scores across ALL rows within each (model, benchmark),
      destroying the SFT guard's discriminative signal entirely. If the ensemble gain requires
      the SFT guard to actually carry information, this collapses it (>= 0 -> <= 0).
    - 'complementarity_null' permutes the SFT scores WITHIN each gold class separately,
      preserving the SFT guard's marginal AP but breaking its per-row co-location with the base.
      If the gain came from base<->SFT agreeing on the same *individual* rows, this removes it;
      if the gain survives, it is a combination of two informative *rankings*, not per-row teamwork.

    Interpreting both: a gain that dies under signal_null but survives complementarity_null is a
    genuine ensemble effect driven by the SFT guard's marginal signal (not an averaging artifact,
    and not dependent on the two guards being right on the same prompts)."""
    def panel_eb(split, srcs, mode, rng):
        vals = []
        for mk in MODELS:
            mv = []
            for src in srcs:
                e = data[mk][split].get(src)
                if e is None:
                    continue
                ps = sft_seedmean(e, "cal", seeds)
                if mode == "signal":
                    ps = ps[rng.permutation(len(ps))]
                elif mode == "complementarity":
                    ps = ps.copy(); g = e["gold"]
                    for cls in (0, 1):
                        idx = np.where(g == cls)[0]
                        ps[idx] = ps[idx][rng.permutation(len(idx))]
                ens = 0.5 * (e["base"]["cal"] + ps)
                mv.append(wap(ens, e["gold"]) - wap(e["base"]["cal"], e["gold"]))
            mv = [v for v in mv if not math.isnan(v)]
            if mv:
                vals.append(float(np.mean(mv)))
        return float(np.mean(vals))

    out = {}
    for regime, (split, srcs) in REGIMES.items():
        out[regime] = {
            "real_ens_minus_base": panel_eb(split, srcs, "real", np.random.default_rng(rng_seed)),
            "signal_null_ens_minus_base": panel_eb(split, srcs, "signal", np.random.default_rng(rng_seed)),
            "complementarity_null_ens_minus_base": panel_eb(split, srcs, "complementarity", np.random.default_rng(rng_seed + 1)),
        }
    return out


# ----------------------------------------------------------------------------- render
def render_md(res):
    L = ["# Composition analysis — Compose, Don't Tune (legacy scores, estimation-only)", ""]
    L.append(f"Scores: `{res['scores_sha256'][:16]}…`  ·  seeds {res['seeds']}  ·  "
             f"bootstrap reps {res['bootstrap']['reps']} (rng {res['bootstrap']['rng_seed']}).")
    L += ["", "## Panel macro-AP by combiner (represented / transfer)", "",
          "| Combiner | represented | transfer |", "|---|---:|---:|"]
    pe = res["point_estimates"]
    for name in res["combiner_order"]:
        r = pe[name]["represented"]["panel"]; t = pe[name]["transfer"]["panel"]
        L.append(f"| {name} | {r:.3f} | {t:.3f} |")
    L += ["", "## Per-model transfer macro-AP (base / SFT / composed calibrated_avg)", "",
          "| Model | base | SFT | composed |", "|---|---:|---:|---:|"]
    for mk in MODELS:
        b = pe["base"]["transfer"]["per_model"][mk]
        s = pe["sft"]["transfer"]["per_model"][mk]
        c = pe["calibrated_avg"]["transfer"]["per_model"][mk]
        L.append(f"| {mk} | {b:.3f} | {s:.3f} | {c:.3f} |")
    bt = res["bootstrap"]
    L += ["", "## Bootstrap CIs — composed(calibrated_avg) advantage (panel)", "",
          "| Regime | ens − SFT [95% CI] | ens − base [95% CI] |", "|---|---|---|"]
    for regime in REGIMES:
        es = bt[regime]["ens_minus_sft"]["panel"]; eb = bt[regime]["ens_minus_base"]["panel"]
        L.append(f"| {regime} | {es['mean']:+.3f} [{es['ci95'][0]:+.3f}, {es['ci95'][1]:+.3f}] | "
                 f"{eb['mean']:+.3f} [{eb['ci95'][0]:+.3f}, {eb['ci95'][1]:+.3f}] |")
    L += ["", "### Per-model transfer ens − base [95% CI]", ""]
    for mk in MODELS:
        eb = bt["transfer"]["ens_minus_base"]["per_model"][mk]
        L.append(f"- {mk}: {eb['mean']:+.3f} [{eb['ci95'][0]:+.3f}, {eb['ci95'][1]:+.3f}]")
    op = res["operating_point"]
    L += ["", f"## Matched-FPR operating point (target {op['target_fpr']:.0%}) — realized rates", "",
          "| Guard | regime | macro TPR | macro FPR | pooled FPR |", "|---|---|---:|---:|---:|"]
    for guard in ("base", "sft", "calibrated_avg"):
        for regime in REGIMES:
            g = op[guard][regime]
            L.append(f"| {guard} | {regime} | {g['macro_tpr']:.3f} | {g['macro_fpr']:.3f} | {g['pooled_fpr']:.3f} |")
    sn = res["shuffle_null"]
    L += ["", "## Shuffle-null controls (panel ens − base)", "",
          "| Regime | real | signal-null (SFT signal destroyed) | complementarity-null (per-row broken) |",
          "|---|---:|---:|---:|"]
    for regime in REGIMES:
        L.append(f"| {regime} | {sn[regime]['real_ens_minus_base']:+.3f} | "
                 f"{sn[regime]['signal_null_ens_minus_base']:+.3f} | "
                 f"{sn[regime]['complementarity_null_ens_minus_base']:+.3f} |")
    L += ["", "*Legacy artifact → estimation-only; a clean rerun is required for confirmatory use. "
          "WiSE-FT weight interpolation is out of scope (adapter weights absent).*", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=os.path.join(_ROOT, "artifacts/paper_a_sft/scores/scores.parquet"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "artifacts/paper_a_sft/analysis/composition"))
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--rng-seed", type=int, default=20260712)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    args = ap.parse_args()

    import hashlib
    sha = hashlib.sha256(open(args.scores, "rb").read()).hexdigest()
    df = load(args.scores)
    seeds = sorted(int(s) for s in df.loc[df.condition == "sft", "seed"].unique())
    data = build(df, seeds)
    pit = fit_pit(data, seeds)

    w = select_convex_w(data, seeds)
    combiners = ["base", "sft", "calibrated_avg", "raw_avg", "logit_avg", "max_cal", "pit_avg", f"convex:{w}"]
    pe = point_estimates(data, seeds, combiners, pit)
    # normalize the convex key name for reporting
    pe["convex_blind"] = pe.pop(f"convex:{w}"); pe["convex_blind"]["selected_w_on_represented"] = w
    order = ["base", "sft", "calibrated_avg", "raw_avg", "logit_avg", "max_cal", "pit_avg", "convex_blind"]

    print("[composition] point estimates done; running bootstrap…", flush=True)
    bt = bootstrap(data, seeds, args.reps, args.rng_seed)
    loo = loo_benchmark(data, seeds, pit=pit)
    op = operating_point(data, seeds, args.target_fpr)
    sn = shuffle_null(data, seeds)

    res = {"analysis": "composition_v1", "scores_sha256": sha, "seeds": seeds, "legacy": True,
           "combiner_order": order, "convex_selected_w": w,
           "point_estimates": pe, "bootstrap": bt, "leave_one_benchmark_out": loo,
           "operating_point": op, "shuffle_null": sn,
           "note": "LEGACY scores -> estimation-only; WiSE-FT (weight interpolation) out of scope."}

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "composition.json"), "w") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "composition.md"), "w") as f:
        f.write(render_md(res))
    print(render_md(res))
    print(f"\n[composition] wrote {args.out}/composition.json and composition.md")


if __name__ == "__main__":
    main()
