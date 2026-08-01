"""The Act I effect re-read in the deployment operating region, from committed per-row scores.

Every deployment sentence in this report is about a false-alarm budget -- "at a matched 5%
alarm budget", "an inline guard runs on every request". Every *headline* number is macro-AP,
which averages precision over the whole ranking, including the deep negative mass where a
guard that fires on 5% of traffic is never placed. Those are different quantities, and nothing
in the report so far establishes that a change in one implies the same change in the other.

This module answers that empirically. For each checkpoint and regime it recomputes the paired
base->SFT change under three metrics on the *identical* committed rows:

  * macro-AP                -- the report's primary metric (whole ranking);
  * macro pAUC over FPR [0, .05] -- mean TPR inside the budget, chance floor .025;
  * macro TPR at 5% FPR     -- the single operating point, conservative under ties.

The finding is not a sign flip: on all eight cells the low-FPR metrics move the same direction
as AP, so Act I's qualitative claim survives being read at the operating point. The finding is
one of *magnitude*. macro-AP systematically understates both halves of the specialization
trade: represented gains are roughly two to five times larger inside the budget, and transfer
losses two to three times larger. Read only on AP, Act I's transfer cost looks like a few
points; read where a guard is actually placed, it is a third of the recall.

This needs no GPU and no pinned environment. It is ranking arithmetic on the same committed
`score_raw`/`gold`/`family_id` columns as every other covered artifact, so the emitted table
and macros are byte-checked by `make verify`.

Uncertainty follows the report's protocol rather than inventing a second one: a paired
bootstrap that resamples near-duplicate evaluation `family_id` clusters *and* training seeds,
holding checkpoint identity fixed (see `experiments/eval_frontier_general_h2h.py`, which
documents why families rather than bare rows are the resampling unit).
"""
from __future__ import annotations

import numpy as np

from guard_research.metrics import average_precision
from guard_research.operating_point import LOW_FPR_MAX, partial_auc, tpr_at_fpr

PRETTY = {"qwen25_15b": "Qwen2.5-1.5B", "smollm2_17b": "SmolLM2-1.7B",
          "smollm3_3b": "SmolLM3-3B", "qwen3_4b": "Qwen3-4B"}
ORDER = ["qwen25_15b", "smollm2_17b", "smollm3_3b", "qwen3_4b"]

# Metric registry. Keys are stable and used in the emitted macro names.
METRICS = (("ap", "macro-AP", average_precision),
           ("pauc", f"pAUC[0,{LOW_FPR_MAX:g}]", partial_auc),
           ("tpr", f"TPR@{LOW_FPR_MAX:g}", tpr_at_fpr))

N_BOOT = 2000
BOOT_SEED = 20260801


def _macro(sub, fn) -> float:
    """Benchmark-macro of `fn`: metric per source, then equal-weight mean.

    Equal weight per source is the report's convention everywhere; it is what stops one large
    corpus from owning the panel number.
    """
    vals = []
    for _, g in sub.groupby("source", sort=True):
        v = fn(g["score_raw"].to_numpy(float), g["gold"].to_numpy(float))
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def _cells(df, split):
    """{model_key: (base_frame, [sft_frame per seed])} for one evaluation split."""
    f = df[df["split"] == split]
    out = {}
    for mk in ORDER:
        m = f[f["model_key"] == mk]
        base = m[m["condition"] == "base"]
        sft = m[m["condition"] == "sft"]
        seeds = sorted(sft["seed"].unique())
        out[mk] = (base, [sft[sft["seed"] == s] for s in seeds])
    return out


def _point(base, sfts, fn) -> tuple[float, float]:
    """(base metric, mean over seeds of the SFT metric) -- Paper A's metric-per-seed convention.

    Averaging the per-seed metric, not the metric of an averaged score vector: the two are
    different estimands, and the report already records the confusion that mistake caused in
    the head-to-head table.
    """
    return _macro(base, fn), float(np.mean([_macro(a, fn) for a in sfts]))


def _boot_delta(base, sfts, fn, n_boot=N_BOOT, seed=BOOT_SEED) -> tuple[float, float]:
    """Two-sided 95% interval on (SFT - base), resampling family clusters and seeds.

    Clusters are formed per source so the macro structure is preserved inside every replicate;
    a bootstrap that pooled sources would quietly turn the equal-weight macro into a row-weighted
    mean.
    """
    rng = np.random.default_rng(seed)
    sources = sorted(set(base["source"].unique()))
    # Pre-extract per-source arrays once; the inner loop is hot.
    pre = {}
    for s in sources:
        b = base[base["source"] == s]
        fams = {}
        for i, f in enumerate(b["family_id"].astype(str).to_numpy()):
            fams.setdefault(f, []).append(i)
        pre[s] = {
            "clusters": [np.asarray(v, int) for _, v in sorted(fams.items())],
            "b_s": b["score_raw"].to_numpy(float), "b_y": b["gold"].to_numpy(float),
            "a_s": [a[a["source"] == s]["score_raw"].to_numpy(float) for a in sfts],
        }
        # Seed arms must be row-aligned with the base arm for a paired resample to be valid.
        for arr in pre[s]["a_s"]:
            if arr.shape != pre[s]["b_s"].shape:
                return float("nan"), float("nan")

    n_seeds = len(sfts)
    deltas = np.empty(n_boot, float)
    for r in range(n_boot):
        seed_pick = rng.integers(0, n_seeds, n_seeds)
        db, da = [], []
        for s in sources:
            p = pre[s]
            idx = np.concatenate([p["clusters"][t] for t in
                                  rng.integers(0, len(p["clusters"]), len(p["clusters"]))])
            y = p["b_y"][idx]
            if y.min() == y.max():
                continue
            db.append(fn(p["b_s"][idx], y))
            da.append(float(np.mean([fn(p["a_s"][k][idx], y) for k in seed_pick])))
        deltas[r] = (np.mean(da) - np.mean(db)) if db else np.nan
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def compute(df, n_boot: int = N_BOOT) -> dict:
    """Per-checkpoint base->SFT deltas under all three metrics, both regimes, with intervals."""
    out = {"max_fpr": LOW_FPR_MAX, "n_boot": n_boot, "regimes": {}}
    for regime, split in (("represented", "id_test"), ("transfer", "transfer_test")):
        cells = _cells(df, split)
        rows = []
        for mk in ORDER:
            base, sfts = cells[mk]
            row = {"key": mk, "name": PRETTY[mk], "n_seeds": len(sfts)}
            for key, _, fn in METRICS:
                b, a = _point(base, sfts, fn)
                lo, hi = _boot_delta(base, sfts, fn, n_boot=n_boot)
                row[key] = {"base": b, "sft": a, "delta": a - b, "lo": lo, "hi": hi}
            rows.append(row)
        panel = {}
        for key, _, _ in METRICS:
            panel[key] = {
                "base": float(np.mean([r[key]["base"] for r in rows])),
                "sft": float(np.mean([r[key]["sft"] for r in rows])),
                "delta": float(np.mean([r[key]["delta"] for r in rows])),
            }
        out["regimes"][regime] = {"rows": rows, "panel": panel}
    return out


def amplification(res) -> dict:
    """How much larger the low-FPR effect is than the macro-AP effect, per regime.

    Reported as a ratio of panel-mean absolute deltas. This is the number the section turns on:
    a ratio near 1 would mean AP is an adequate proxy for the operating region, and it is not.
    """
    out = {}
    for regime, r in res["regimes"].items():
        ap = abs(r["panel"]["ap"]["delta"])
        out[regime] = {k: (abs(r["panel"][k]["delta"]) / ap if ap > 0 else float("nan"))
                       for k in ("pauc", "tpr")}
    return out


def _fmt(d) -> str:
    return f"{d['delta']:+.3f}~[{d['lo']:+.3f},\\,{d['hi']:+.3f}]"


def emit_table(df, res=None) -> str:
    r = res or compute(df)
    amp = amplification(r)
    body = []
    for regime, label in (("represented", "Represented (\\code{id\\_test})"),
                          ("transfer", "Transfer (\\code{transfer\\_test})")):
        body.append("\\multicolumn{4}{@{}l}{\\emph{" + label + "}} \\\\")
        for row in r["regimes"][regime]["rows"]:
            body.append("\\quad {n} & {ap} & {pa} & {tp} \\\\".format(
                n=row["name"], ap=_fmt(row["ap"]), pa=_fmt(row["pauc"]), tp=_fmt(row["tpr"])))
        p = r["regimes"][regime]["panel"]
        body.append("\\quad \\textbf{{Panel mean}} & \\textbf{{{a:+.3f}}} & \\textbf{{{b:+.3f}}} "
                    "& \\textbf{{{c:+.3f}}} \\\\".format(
                        a=p["ap"]["delta"], b=p["pauc"]["delta"], c=p["tpr"]["delta"]))
        if regime == "represented":
            body.append("\\midrule")

    v = {
        "rap": f"{r['regimes']['represented']['panel']['ap']['delta']:+.3f}",
        "rpa": f"{r['regimes']['represented']['panel']['pauc']['delta']:+.3f}",
        "tap": f"{r['regimes']['transfer']['panel']['ap']['delta']:+.3f}",
        "tpa": f"{r['regimes']['transfer']['panel']['pauc']['delta']:+.3f}",
        "ttp": f"{r['regimes']['transfer']['panel']['tpr']['delta']:+.3f}",
        "ra": f"{amp['represented']['pauc']:.1f}", "ta": f"{amp['transfer']['pauc']:.1f}",
        "mf": f"{LOW_FPR_MAX:g}", "nb": f"{r['n_boot']:,}",
    }
    caption = (
        "\\textbf{The same eight cells, read in the operating region instead of over the whole "
        "ranking.} Paired base$\\to$SFT change on identical committed rows under three metrics: "
        "benchmark-macro AP (the report's primary metric), benchmark-macro one-way partial AUC "
        "over FPR $[0,"
        + v["mf"] + "]$ (mean TPR inside the alarm budget; chance floor $0.025$, not $0.5$), and "
        "benchmark-macro TPR at that budget. Brackets are two-sided $95\\%$ paired bootstrap "
        "intervals over " + v["nb"] + " replicates resampling evaluation \\code{family\\_id} "
        "clusters and training seeds, the same protocol as the rest of the report. "
        "\\textbf{No sign flips}: every cell moves the same way under all three metrics, so "
        "Act~I's direction survives being read at a deployable operating point. What does not "
        "survive is the \\emph{size}. macro-AP understates both halves of the trade --- the "
        "represented gain is " + v["rap"] + " on AP against " + v["rpa"] + " on pAUC "
        "($" + v["ra"] + "\\times$), and the transfer cost is " + v["tap"] + " against "
        + v["tpa"] + " ($" + v["ta"] + "\\times$), with panel-mean budget recall moving "
        + v["ttp"] + ". Averaging precision over the whole ranking credits ordering in the deep "
        "negative mass, where an inline guard never operates. Same rows, same scorer, no GPU: "
        "only the metric changes, and it is byte-checked by \\code{make verify}."
    )
    return ("% GENERATED by reproduce.py (low_fpr.py) from "
            "artifacts/paper_a_sft_v2/scores/scores.parquet\n"
            "\\begin{table}[htbp]\\centering\\small\\setlength{\\tabcolsep}{5pt}\n"
            "\\caption{" + caption + "}\n"
            "\\label{tab:lowfpr}\n"
            "\\begin{tabular}{@{}lccc@{}}\\toprule\n"
            "Checkpoint & $\\Delta$ macro-AP & $\\Delta$ pAUC$[0,"
            + v["mf"] + "]$ & $\\Delta$ TPR@" + v["mf"] + " \\\\\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def emit_macros(df, res=None) -> str:
    r = res or compute(df)
    amp = amplification(r)
    L = ["% GENERATED by reproduce.py (low_fpr.py) -- do not hand-edit.",
         f"\\newcommand{{\\LowFprBudget}}{{{LOW_FPR_MAX:g}}}",
         f"\\newcommand{{\\LowFprChance}}{{{LOW_FPR_MAX / 2:.3f}}}",
         f"\\newcommand{{\\LowFprNBoot}}{{{r['n_boot']:,}}}"]
    for regime, tag in (("represented", "Rep"), ("transfer", "Trans")):
        p = r["regimes"][regime]["panel"]
        for key, mac in (("ap", "Ap"), ("pauc", "PAuc"), ("tpr", "Tpr")):
            L.append(f"\\newcommand{{\\LowFpr{tag}{mac}Delta}}{{{p[key]['delta']:+.3f}}}")
            L.append(f"\\newcommand{{\\LowFpr{tag}{mac}Base}}{{{p[key]['base']:.3f}}}")
            L.append(f"\\newcommand{{\\LowFpr{tag}{mac}Sft}}{{{p[key]['sft']:.3f}}}")
        L.append(f"\\newcommand{{\\LowFpr{tag}Amplification}}{{{amp[regime]['pauc']:.1f}}}")
    # The single worst cell, which is the sentence the section leads with.
    worst = min(r["regimes"]["transfer"]["rows"], key=lambda x: x["pauc"]["delta"])
    L += [f"\\newcommand{{\\LowFprWorstName}}{{{worst['name']}}}",
          f"\\newcommand{{\\LowFprWorstAp}}{{{worst['ap']['delta']:+.3f}}}",
          f"\\newcommand{{\\LowFprWorstPAuc}}{{{worst['pauc']['delta']:+.3f}}}",
          f"\\newcommand{{\\LowFprWorstPAucCI}}"
          f"{{[{worst['pauc']['lo']:+.3f}, {worst['pauc']['hi']:+.3f}]}}",
          f"\\newcommand{{\\LowFprWorstTpr}}{{{worst['tpr']['delta']:+.3f}}}"]
    # Does any cell change sign between AP and pAUC? The section's honesty hinges on this being 0.
    flips = sum(1 for reg in r["regimes"].values() for row in reg["rows"]
                if np.sign(row["ap"]["delta"]) != np.sign(row["pauc"]["delta"]))
    L.append(f"\\newcommand{{\\LowFprSignFlips}}{{{flips}}}")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------
# The KL-SFT control, read in the same region.
#
# Act I's recipe control (Table 7) is reported as a macro-AP trade: +0.061 transfer bought
# for -0.035 represented, and the guidelines table turns that into "a tradeoff dial, not a
# free upgrade". Both halves of that dial are measured over the whole ranking. Recomputing
# them inside the alarm budget does not reverse the direction -- KL still buys transfer and
# still charges represented -- but it changes the exchange rate enough to change the advice,
# and it changes how badly the preregistered non-inferiority margin is missed.
# --------------------------------------------------------------------------------------

KL_BETAS = (0.5, 1.0)
KL_PRIMARY_BETA = 0.5


def compute_kl(frames: dict, n_boot: int = N_BOOT) -> dict:
    """Per-checkpoint KL(beta) - KL(beta=0) deltas under all three metrics, both regimes.

    `frames` maps model_key -> the committed KL parquet frame for that checkpoint. beta=0 is
    the in-environment vanilla-SFT arm, which is the correct reference: it isolates the KL
    term from the execution-environment difference the report documents elsewhere.
    """
    out = {"max_fpr": LOW_FPR_MAX, "n_boot": n_boot, "betas": list(KL_BETAS), "regimes": {}}
    for regime, split in (("represented", "id_test"), ("transfer", "transfer_test")):
        per_beta = {}
        for beta in KL_BETAS:
            rows = []
            for mk in ORDER:
                if mk not in frames:
                    continue
                f = frames[mk][frames[mk]["split"] == split]
                ref = [f[(f["kl_beta"] == 0.0) & (f["seed"] == s)]
                       for s in sorted(f[f["kl_beta"] == 0.0]["seed"].unique())]
                arm = [f[(f["kl_beta"] == beta) & (f["seed"] == s)]
                       for s in sorted(f[f["kl_beta"] == beta]["seed"].unique())]
                row = {"key": mk, "name": PRETTY[mk]}
                for key, _, fn in METRICS:
                    r = float(np.mean([_macro(x, fn) for x in ref]))
                    a = float(np.mean([_macro(x, fn) for x in arm]))
                    row[key] = {"ref": r, "arm": a, "delta": a - r}
                rows.append(row)
            per_beta[beta] = {
                "rows": rows,
                "panel": {k: {"delta": float(np.mean([r[k]["delta"] for r in rows]))}
                          for k, _, _ in METRICS},
            }
        out["regimes"][regime] = per_beta
    return out


def kl_amplification(res_kl, beta: float = KL_PRIMARY_BETA) -> dict:
    """|low-FPR delta| / |AP delta| per regime -- how much the AP headline understates the dial."""
    out = {}
    for regime, per_beta in res_kl["regimes"].items():
        p = per_beta[beta]["panel"]
        ap = abs(p["ap"]["delta"])
        out[regime] = {k: (abs(p[k]["delta"]) / ap if ap > 0 else float("nan"))
                       for k in ("pauc", "tpr")}
    return out


def emit_kl_macros(res_kl) -> str:
    amp = kl_amplification(res_kl)
    L = ["% GENERATED by reproduce.py (low_fpr.py) -- do not hand-edit.",
         f"\\newcommand{{\\LowFprKlBeta}}{{{KL_PRIMARY_BETA:g}}}"]
    for regime, tag in (("represented", "Rep"), ("transfer", "Trans")):
        p = res_kl["regimes"][regime][KL_PRIMARY_BETA]["panel"]
        for key, mac in (("ap", "Ap"), ("pauc", "PAuc"), ("tpr", "Tpr")):
            L.append(f"\\newcommand{{\\LowFprKl{tag}{mac}Delta}}{{{p[key]['delta']:+.3f}}}")
        L.append(f"\\newcommand{{\\LowFprKl{tag}Amplification}}{{{amp[regime]['pauc']:.1f}}}")
    # How many multiples of the registered -0.02 non-inferiority margin the represented cost is,
    # read in the operating region. On AP the margin is missed by under 2x; this is the sharper
    # statement the preregistered study could have made.
    rep = res_kl["regimes"]["represented"][KL_PRIMARY_BETA]["panel"]
    L += [f"\\newcommand{{\\LowFprKlMarginMultipleAp}}{{{abs(rep['ap']['delta']) / 0.02:.1f}}}",
          f"\\newcommand{{\\LowFprKlMarginMultiplePAuc}}{{{abs(rep['pauc']['delta']) / 0.02:.1f}}}"]
    return "\n".join(L) + "\n"


def emit_kl_table(res_kl) -> str:
    body = []
    for regime, label in (("transfer", "Transfer (\\code{transfer\\_test})"),
                          ("represented", "Represented (\\code{id\\_test})")):
        body.append("\\multicolumn{4}{@{}l}{\\emph{" + label + "}} \\\\")
        for beta in KL_BETAS:
            blk = res_kl["regimes"][regime][beta]
            for row in blk["rows"]:
                body.append("\\quad {n} ($\\beta{{=}}{b:g}$) & {a:+.3f} & {p:+.3f} & {t:+.3f} \\\\".format(
                    n=row["name"], b=beta, a=row["ap"]["delta"],
                    p=row["pauc"]["delta"], t=row["tpr"]["delta"]))
            pn = blk["panel"]
            body.append("\\quad \\textbf{{Panel mean}} ($\\beta{{=}}{b:g}$) & \\textbf{{{a:+.3f}}} & "
                        "\\textbf{{{p:+.3f}}} & \\textbf{{{t:+.3f}}} \\\\".format(
                            b=beta, a=pn["ap"]["delta"], p=pn["pauc"]["delta"], t=pn["tpr"]["delta"]))
        if regime == "transfer":
            body.append("\\midrule")

    amp = kl_amplification(res_kl)
    tp = res_kl["regimes"]["transfer"][KL_PRIMARY_BETA]["panel"]
    rp = res_kl["regimes"]["represented"][KL_PRIMARY_BETA]["panel"]
    v = {"tap": f"{tp['ap']['delta']:+.3f}", "tpa": f"{tp['pauc']['delta']:+.3f}",
         "ttp": f"{tp['tpr']['delta']:+.3f}", "rap": f"{rp['ap']['delta']:+.3f}",
         "rpa": f"{rp['pauc']['delta']:+.3f}", "rtp": f"{rp['tpr']['delta']:+.3f}",
         "ta": f"{amp['transfer']['pauc']:.1f}", "ra": f"{amp['represented']['pauc']:.1f}",
         "mult": f"{abs(rp['pauc']['delta']) / 0.02:.0f}", "mf": f"{LOW_FPR_MAX:g}"}
    caption = (
        "\\textbf{The KL dial, priced in the operating region.} Change from the in-environment "
        "$\\beta{=}0$ arm, on identical committed rows, under the same three metrics as "
        "\\Cref{tab:lowfpr}. The direction of \\Cref{tab:klsft} is unchanged --- KL buys transfer "
        "and charges represented ranking at both $\\beta$ --- but the exchange rate is not what "
        "macro-AP shows. At $\\beta{=}0.5$ the transfer gain is " + v["tap"] + " on AP against "
        + v["tpa"] + " on pAUC ($" + v["ta"] + "\\times$) and " + v["ttp"] + " in budget recall, "
        "while the represented cost is " + v["rap"] + " on AP against " + v["rpa"] + " on pAUC "
        "($" + v["ra"] + "\\times$). Both halves of the dial are larger where a guard is placed, "
        "and the represented half grows faster than the transfer half. This also sharpens "
        "\\Cref{sec:adaptation}'s registered failure: read on AP the represented cost misses the "
        "$-0.02$ non-inferiority margin by under a factor of two, but read at the budget it misses "
        "by roughly " + v["mult"] + "$\\times$. Point estimates only --- these are seed means over "
        "the same four checkpoints, not a re-run of the registered study's family bootstrap."
    )
    return ("% GENERATED by reproduce.py (low_fpr.py) from artifacts/klsft_v1/scores/\n"
            "\\begin{table}[htbp]\\centering\\small\\setlength{\\tabcolsep}{5pt}\n"
            "\\caption{" + caption + "}\n"
            "\\label{tab:lowfpr-kl}\n"
            "\\begin{tabular}{@{}lccc@{}}\\toprule\n"
            "Checkpoint & $\\Delta$ macro-AP & $\\Delta$ pAUC$[0," + v["mf"] + "]$ & "
            "$\\Delta$ TPR@" + v["mf"] + " \\\\\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
