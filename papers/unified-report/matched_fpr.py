"""Matched false-alarm-budget (iso-FPR) operating point, from committed per-row scores.

Table 3 compares base and SFT recall at each guard's *own* calibrated threshold, where the
tuned guard realises a far higher false-alarm rate (17.0% vs 4.3% pooled transfer FPR). A
recall comparison at unequal alarm rates is not a comparison of discriminative power: the
tuned guard buys recall by alarming more. Act I says so in prose and then reports the
unequal-rate row anyway, describing the matched-rate reconstruction as a "direction"
requiring the lock-pinned environment.

It does not require that environment. Matching false-alarm rates is ranking arithmetic on
`score_raw` and `gold` -- the same committed inputs every other covered artifact uses, no
GPU and no network. This module computes it, and the emitted table is byte-checked by
`reproduce.py` like any other covered artifact.

Threshold rule (fixed here, not tuned): for each checkpoint the *budget* is the base's
realised pooled FPR on transfer negatives at its own calibrated 5%-target threshold. Each
SFT seed is then given the threshold whose pooled transfer FPR is at most that budget
(`method="higher"`, the conservative side). Recalls are macro-averaged over the four
transfer sources, matching the report's primary convention. HarmBench is a positives-only
stress set, so it is scored at the same matched threshold.
"""
from __future__ import annotations

PRETTY = {"qwen25_15b": "Qwen2.5-1.5B", "smollm2_17b": "SmolLM2-1.7B",
          "smollm3_3b": "SmolLM3-3B", "qwen3_4b": "Qwen3-4B"}
ORDER = ["qwen25_15b", "smollm2_17b", "smollm3_3b", "qwen3_4b"]

# Quantile conventions checked for stability; the emitted table uses QUANTILE_METHOD.
QUANTILE_METHOD = "higher"
ALT_METHODS = ("lower", "linear")


def _macro(frame, col, sources):
    import numpy as np
    return float(np.mean([frame[frame.source == s][col].mean()
                          for s in sources if (frame.source == s).any()]))


def compute(df, method: str = QUANTILE_METHOD) -> dict:
    """Per-checkpoint rows plus panel means. Pure function of the committed score matrix."""
    import numpy as np

    tr = df[df.split == "transfer_test"]
    hb = df[df.split == "stress_harmbench"]
    sources = sorted(tr.source.unique())

    rows = []
    for mk in ORDER:
        b = tr[(tr.model_key == mk) & (tr.condition == "base")]
        budget = float(b[b.gold == 0].prediction.mean())
        base_tpr = _macro(b[b.gold == 1], "prediction", sources)
        base_hb = float(hb[(hb.model_key == mk) & (hb.condition == "base")].prediction.mean())

        own_tpr, mat_tpr, own_hb, mat_hb = [], [], [], []
        for sd in sorted(tr[(tr.model_key == mk) & (tr.condition == "sft")].seed.unique()):
            a = tr[(tr.model_key == mk) & (tr.condition == "sft") & (tr.seed == sd)]
            thr = np.quantile(a[a.gold == 0].score_raw.values, 1.0 - budget, method=method)
            pos = a[a.gold == 1]
            own_tpr.append(_macro(pos, "prediction", sources))
            mat_tpr.append(_macro(pos.assign(hit=(pos.score_raw.values >= thr).astype(float)),
                                  "hit", sources))
            h = hb[(hb.model_key == mk) & (hb.condition == "sft") & (hb.seed == sd)]
            own_hb.append(float(h.prediction.mean()))
            mat_hb.append(float((h.score_raw.values >= thr).mean()))

        rows.append({
            "name": PRETTY[mk], "budget": budget,
            "base_tpr": base_tpr, "own_tpr": float(np.mean(own_tpr)),
            "mat_tpr": float(np.mean(mat_tpr)),
            "base_hb": base_hb, "own_hb": float(np.mean(own_hb)),
            "mat_hb": float(np.mean(mat_hb)),
        })

    keys = ("budget", "base_tpr", "own_tpr", "mat_tpr", "base_hb", "own_hb", "mat_hb")
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    worse_tpr = sum(r["mat_tpr"] < r["base_tpr"] for r in rows)
    worse_hb = sum(r["mat_hb"] < r["base_hb"] for r in rows)
    return {"rows": rows, "mean": mean, "worse_tpr": worse_tpr, "worse_hb": worse_hb}


def stability(df) -> tuple[float, float]:
    """Min and max panel-mean matched-FPR transfer delta across quantile conventions."""
    deltas = []
    for m in (QUANTILE_METHOD, *ALT_METHODS):
        r = compute(df, method=m)["mean"]
        deltas.append(r["mat_tpr"] - r["base_tpr"])
    return min(deltas), max(deltas)


def emit_table(df) -> str:
    r = compute(df)
    lo, hi = stability(df)
    body = "\n".join(
        "{name} & {b:.1f}\\% & {bt:.3f} & {ot:.3f} & {mt:.3f} & {bh:.3f} & {oh:.3f} & {mh:.3f} \\\\".format(
            name=x["name"], b=x["budget"] * 100, bt=x["base_tpr"], ot=x["own_tpr"],
            mt=x["mat_tpr"], bh=x["base_hb"], oh=x["own_hb"], mh=x["mat_hb"])
        for x in r["rows"])
    m = r["mean"]
    mean_row = (
        "\\midrule\n"
        "Panel mean & {b:.1f}\\% & {bt:.3f} & {ot:.3f} & \\textbf{{{mt:.3f}}} & "
        "{bh:.3f} & {oh:.3f} & \\textbf{{{mh:.3f}}} \\\\".format(
            b=m["budget"] * 100, bt=m["base_tpr"], ot=m["own_tpr"], mt=m["mat_tpr"],
            bh=m["base_hb"], oh=m["own_hb"], mh=m["mat_hb"]))

    # Numbers first, LaTeX second: the caption is full of literal braces, so it must never be
    # passed through .format()/f-string interpolation.
    n_worse = "all four" if r["worse_tpr"] == 4 else f"{r['worse_tpr']} of the four"
    v = {
        "bt": f"{m['base_tpr']:.3f}", "mt": f"{m['mat_tpr']:.3f}",
        "dt": f"{m['mat_tpr'] - m['base_tpr']:+.3f}",
        "bh": f"{m['base_hb']:.3f}", "mh": f"{m['mat_hb']:.3f}",
        "dh": f"{m['mat_hb'] - m['base_hb']:+.3f}",
        "lo": f"{lo:+.3f}", "hi": f"{hi:+.3f}", "n": n_worse,
    }
    caption = (
        "\\textbf{The same operating point read at an equal false-alarm budget.} "
        "\\Cref{tab:sensitivity} compares recalls at each guard's \\emph{own} calibrated "
        "threshold, where the tuned guard alarms far more often (17.0\\% vs 4.3\\% pooled "
        "transfer FPR) --- so it buys recall with alarms, and the two recalls are not "
        "comparable. Here each SFT seed is instead thresholded so its pooled transfer "
        "false-alarm rate \\emph{matches its own base's} (the budget column), which makes the "
        "recalls directly comparable. At an equal budget the tuned guard is worse on "
        "\\textbf{" + v["n"] + "} checkpoints and on both instruments: transfer recall "
        + v["bt"] + "$\\rightarrow$" + v["mt"] + " (" + v["dt"] + ") and \\texttt{HarmBench} "
        "recall " + v["bh"] + "$\\rightarrow$" + v["mh"] + " (" + v["dh"] + "). The direction "
        "is stable across the three quantile conventions we tried (panel-mean transfer delta "
        + v["lo"] + " to " + v["hi"] + "). Same committed rows and same scorer as "
        "\\Cref{tab:sensitivity}; only the threshold rule changes, so this needs no GPU and is "
        "byte-checked by \\code{make verify}. \\textbf{This is a retrospective ROC point, not a "
        "deployable threshold}: the quantile is read off the \\emph{same} labelled negatives the "
        "recall is then measured on, so a production system without labels could not place it. "
        "Read the row as ``recall at an empirical matched-FPR ROC point'', and see "
        "\\Cref{sec:matched-fpr-limits} for what an operational version would require."
    )

    return ("% GENERATED by reproduce.py (matched_fpr.py) from "
            "artifacts/paper_a_sft_v2/scores/scores.parquet\n"
            # tabcolsep: eight columns of numbers ran 5.5pt past the text block at the
            # default 6pt, so the emitter sets it rather than leaving the caller to.
            "\\begin{table}[htbp]\\centering\\small\\setlength{\\tabcolsep}{4.5pt}\n"
            f"\\caption{{{caption}}}\n"
            "\\label{tab:matchedfpr}\n"
            "\\begin{tabular}{lrrrrrrr}\\toprule\n"
            "& FPR & \\multicolumn{3}{c}{transfer recall (macro)} & "
            "\\multicolumn{3}{c}{\\texttt{HarmBench} recall} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
            "Checkpoint & budget & base & SFT own thr. & SFT matched & "
            "base & SFT own thr. & SFT matched \\\\\n"
            "\\midrule\n"
            f"{body}\n{mean_row}\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def emit_macros(df) -> str:
    r = compute(df)
    m = r["mean"]
    lo, hi = stability(df)
    return ("% GENERATED by reproduce.py (matched_fpr.py) from "
            "artifacts/paper_a_sft_v2/scores/scores.parquet\n"
            f"\\newcommand{{\\MatchedTransferBase}}{{{m['base_tpr']:.3f}}}\n"
            f"\\newcommand{{\\MatchedTransferSft}}{{{m['mat_tpr']:.3f}}}\n"
            f"\\newcommand{{\\MatchedTransferDelta}}{{{m['mat_tpr'] - m['base_tpr']:+.3f}}}\n"
            f"\\newcommand{{\\MatchedHarmBase}}{{{m['base_hb']:.3f}}}\n"
            f"\\newcommand{{\\MatchedHarmSft}}{{{m['mat_hb']:.3f}}}\n"
            f"\\newcommand{{\\MatchedHarmDelta}}{{{m['mat_hb'] - m['base_hb']:+.3f}}}\n"
            f"\\newcommand{{\\MatchedWorseTransfer}}{{{r['worse_tpr']}}}\n"
            f"\\newcommand{{\\MatchedWorseHarm}}{{{r['worse_hb']}}}\n"
            f"\\newcommand{{\\MatchedStabilityLo}}{{{lo:+.3f}}}\n"
            f"\\newcommand{{\\MatchedStabilityHi}}{{{hi:+.3f}}}\n")
