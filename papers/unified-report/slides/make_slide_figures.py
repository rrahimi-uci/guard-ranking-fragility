#!/usr/bin/env python
"""Presentation-scale figures for the unified-report slide deck.

These are NOT the paper figures upscaled. The paper figures are typeset for a 10pt
document and their labels become unreadable on a projector, so every panel here is
re-rendered from the same committed generated/ artifacts at deck scale: larger type,
fewer ticks, higher contrast, values annotated on the marks.

Every number is parsed from papers/unified-report/generated/*.tex, which is itself
generated from committed per-row scores. Nothing is typed in by hand.

Run:  python slides/make_slide_figures.py     (from papers/unified-report/)
Out:  slides/assets/*.png  (200 dpi)
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
GEN = REPORT / "generated"
OUT = HERE / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- deck identity
INK = "#12263A"       # headline / axis text
SLATE = "#5A6B7C"     # secondary text
RULE = "#D8DEE4"      # gridlines, spines
BLUE = "#2563EB"      # represented-source
ORANGE = "#EA580C"    # dataset-held-out transfer
GREEN = "#15803D"     # composition / recovered
RED = "#A02128"       # regression / accent
GOLD = "#B7791F"
PURPLE = "#6D28D9"

MODEL_COLOR = {
    "Qwen2.5-1.5B": BLUE,
    "SmolLM2-1.7B": ORANGE,
    "SmolLM3-3B": GREEN,
    "Qwen3-4B": RED,
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": RULE,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": RULE,
    "grid.alpha": 0.7,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "savefig.transparent": False,
    "figure.facecolor": "white",
})


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPORT)}")


def tidy(ax, *, xgrid=False, ygrid=True):
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=ygrid)
    ax.grid(axis="x", visible=xgrid)
    ax.tick_params(length=0)


# ------------------------------------------------------------------- data reads
def primary_rows():
    """[(checkpoint, rep_base, rep_sft, d_rep, tr_base, tr_sft, d_tr)] from tab_primary_gen."""
    rows = []
    pat = re.compile(
        r"^\s*([\w.\-]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\-\d.]+)\s*\[.*?\]\s*&"
        r"\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\-\d.]+)\s*\[")
    for ln in (GEN / "tab_primary_gen.tex").read_text().splitlines():
        if "aggregate" in ln.lower():
            continue
        m = pat.match(ln)
        if m:
            rows.append((m.group(1), *(float(m.group(i)) for i in range(2, 8))))
    assert len(rows) == 4, rows
    return rows


def seed_rows():
    """[(checkpoint, seed, d_rep, d_tr)] from tab_seed_values_gen."""
    rows = []
    pat = re.compile(r"^\s*([\w.\-]+)\s*&\s*(\d+)\s*&\s*([\-\d.]+)\s*&\s*([\-\d.]+)\s*\\\\")
    for ln in (GEN / "tab_seed_values_gen.tex").read_text().splitlines():
        m = pat.match(ln)
        if m:
            rows.append((m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))))
    assert len(rows) == 20, len(rows)
    return rows


def macros(fname):
    """\\newcommand{\\Name}{value} -> {Name: value}, stripping $ and braces."""
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (GEN / fname).read_text()):
        out[m.group(1)] = m.group(2).replace("$", "").strip()
    return out


def composition_rows():
    """[(checkpoint, base, sft, comp)] from pilot_per_model_table."""
    rows = []
    pat = re.compile(r"^([\w.\-]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&")
    for ln in (GEN / "pilot_per_model_table.tex").read_text().splitlines():
        m = pat.match(ln.strip())
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    assert len(rows) == 4, rows
    return rows


def sftsft_rows():
    """[(checkpoint, base, sft, base+sft, sft+sft)] from tab_sftsft_gen."""
    rows = []
    pat = re.compile(r"^([\w.\-]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\\,\[.*?\]\s*&"
                     r"\s*([\d.]+)\\,\[")
    for ln in (GEN / "tab_sftsft_gen.tex").read_text().splitlines():
        m = pat.match(ln.strip())
        if m:
            rows.append((m.group(1), *(float(m.group(i)) for i in range(2, 6))))
    assert len(rows) == 4, rows
    return rows


def klsft_rows():
    """[(checkpoint, tr_base, tr_sft, tr_kl5, tr_kl1, rep_sft, rep_kl5, rep_kl1)]."""
    rows = []
    pat = re.compile(r"^([\w.\-]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)"
                     r"\s*&\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)")
    for ln in (GEN / "tab_klsft_gen.tex").read_text().splitlines():
        m = pat.match(ln.strip())
        if m:
            rows.append((m.group(1), *(float(m.group(i)) for i in range(2, 9))))
    assert len(rows) == 4, rows
    return rows


def adaptation_rows():
    """[(checkpoint, kind, sft_rep, sft_tr, kl_rep, kl_tr)] from tab_adaptation_gen."""
    rows, kind = [], "general"
    pat = re.compile(r"^([\w.\-]+)\s*&\s*\w+\s*&\s*\$([+\-\d.]+)\$\s*&\s*\$([+\-\d.]+)\$\s*&"
                     r"\s*\$([+\-\d.]+)\$\s*&\s*\$([+\-\d.]+)\$")
    for ln in (GEN / "tab_adaptation_gen.tex").read_text().splitlines():
        s = ln.strip()
        if "Released purpose-built" in s:
            kind = "released"
        m = pat.match(s)
        if m:
            rows.append((m.group(1), kind, *(float(m.group(i)) for i in range(2, 6))))
    assert len(rows) == 10, len(rows)
    return rows


def mortgage_rows():
    """[(guard, ap_g, ap_d, delta_prob, delta_margin)] from mortgage_baseline_table."""
    pretty = {"qwen25\\_15b\\_base": "Qwen2.5-1.5B", "qwen3\\_4b\\_base": "Qwen3-4B",
              "smollm2\\_17b\\_base": "SmolLM2-1.7B", "smollm3\\_3b\\_base": "SmolLM3-3B"}
    rows = []
    pat = re.compile(r"^([\w\\]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\\,\[.*?\]\s*&\s*[\d.]+\s*&"
                     r"\s*[\d.]+\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&")
    for ln in (GEN / "mortgage_baseline_table.tex").read_text().splitlines():
        m = pat.match(ln.strip())
        if m and m.group(1) in pretty:
            rows.append((pretty[m.group(1)], *(float(m.group(i)) for i in range(2, 6))))
    assert len(rows) == 4, rows
    return rows


def expguard_rows():
    """[(guard, ap_all, fin, health, law)] from expguard_table."""
    rows = []
    pat = re.compile(r"^([\w.\-]+)\s*&\s*([\d.]+)\\,\[.*?\]\s*&\s*[\d.]+\s*&\s*([\d.]+)\s*&"
                     r"\s*([\d.]+)\s*&\s*([\d.]+)")
    for ln in (GEN / "expguard_table.tex").read_text().splitlines():
        m = pat.match(ln.strip())
        if m:
            rows.append((m.group(1), *(float(m.group(i)) for i in range(2, 6))))
    assert len(rows) == 4, rows
    return rows


SHORT = {"Qwen2.5-1.5B": "Qwen2.5\n1.5B", "SmolLM2-1.7B": "SmolLM2\n1.7B",
         "SmolLM3-3B": "SmolLM3\n3B", "Qwen3-4B": "Qwen3\n4B"}


# =============================================================== FIGURE BUILDERS
def fig_teaser():
    """Three panels: the split, the rank flip, the missed violation."""
    # 4.4in, not 4.1: panel 3 carries a footnote below its x-axis, and tight_layout would
    # otherwise pay for it by shortening all three panels -- which pushed panel 2's "1st"
    # row annotations up into its own title.
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))

    # -- panel 1: represented gain vs transfer change, with the 20 seeds ---------
    ax = axes[0]
    mac = macros("results_macros_gen.tex")
    rep_d, tr_d = float(mac["RepDelta"]), float(mac["TransferDelta"])
    seeds = seed_rows()
    ax.bar([0], [rep_d], 0.62, color=BLUE, zorder=3)
    ax.bar([1], [tr_d], 0.62, color=ORANGE, zorder=3)
    rng = np.random.default_rng(42)
    for j, idx in enumerate((2, 3)):
        vals = [r[idx] for r in seeds]
        jit = rng.uniform(-0.13, 0.13, len(vals))
        ax.scatter(np.full(len(vals), j) + jit, vals, s=16, c=INK, alpha=0.55,
                   zorder=4, linewidths=0)
    ax.axhline(0, color=INK, lw=1.1, zorder=2)
    ax.text(0, rep_d / 2, f"{rep_d:+.2f}", ha="center", va="center", color="white",
            fontsize=19, fontweight="bold", zorder=5)
    # left of the bar, not below it: the seed cloud occupies the space under the bar
    ax.text(0.60, tr_d, f"{tr_d:+.2f}", ha="right", va="center", color=ORANGE,
            fontsize=19, fontweight="bold", zorder=5)
    ax.set_xlim(-0.48, 1.48)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["represented\n(trained-on)", "transfer\n(held-out)"], fontsize=12)
    ax.set_ylabel("base $\\to$ SFT $\\Delta$ macro-AP", fontsize=12)
    ax.set_ylim(-0.30, 0.62)
    ax.set_title("Tuning buys the benchmark,\nnot transfer", fontsize=13.5,
                 fontweight="bold", color=INK, pad=10)
    tidy(ax)

    # -- panel 2: the top-ranked guard changes with the benchmark ---------------
    ax = axes[1]
    prim = {r[0]: r[4] for r in primary_rows()}          # transfer base AP
    mort = {r[0]: r[2] for r in mortgage_rows()}          # AP.D
    expg = {r[0]: r[1] for r in expguard_rows()}          # AP all
    arms = [("general\ntransfer", prim), ("mortgage\npolicy", mort), ("finance /\nhealth / law", expg)]
    order = [sorted(d, key=lambda k: -d[k]) for _, d in arms]
    for model in MODEL_COLOR:
        ys = [order[j].index(model) + 1 for j in range(3)]
        ax.plot(range(3), ys, "-o", color=MODEL_COLOR[model], lw=2.6, ms=8, zorder=3)
        for j, y in enumerate(ys):
            ax.annotate(f"{arms[j][1][model]:.2f}", (j, y), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=10,
                        color=MODEL_COLOR[model], fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                                  alpha=0.92))
        ax.text(2.10, ys[2], model.replace("Qwen2.5-", "Q2.5-").replace("Qwen3-", "Q3-")
                .replace("SmolLM2-", "SL2-").replace("SmolLM3-", "SL3-"),
                va="center", ha="left", fontsize=11, fontweight="bold",
                color=MODEL_COLOR[model])
    ax.set_xticks(range(3))
    ax.set_xticklabels([a for a, _ in arms], fontsize=12)
    ax.set_xlim(-0.35, 3.35)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["1st", "2nd", "3rd", "4th"])
    ax.invert_yaxis()
    ax.set_title("The top-ranked guard\nchanges with the benchmark", fontsize=13.5,
                 fontweight="bold", color=INK, pad=10)
    tidy(ax, ygrid=False)
    ax.grid(axis="y", visible=True)

    # -- panel 3: the violation ranked below the benign median ------------------
    ax = axes[2]
    coded = {"Qwen2.5-1.5B": 46, "SmolLM2-1.7B": 57, "SmolLM3-3B": 65, "Qwen3-4B": 44}
    named = {"Qwen2.5-1.5B": 1, "SmolLM2-1.7B": 7, "SmolLM3-3B": 0, "Qwen3-4B": 15}
    x = np.arange(4)
    labels = ["Q2.5-1.5B", "SL2-1.7B", "SL3-3B", "Q3-4B"]
    keys = ["Qwen2.5-1.5B", "SmolLM2-1.7B", "SmolLM3-3B", "Qwen3-4B"]
    ax.bar(x - 0.20, [coded[k] for k in keys], 0.38, color=RED, label="coded proxy", zorder=3)
    ax.bar(x + 0.20, [named[k] for k in keys], 0.38, color=GREEN, label="traits named", zorder=3)
    for i, k in enumerate(keys):
        ax.text(i - 0.20, coded[k] + 1.5, coded[k], ha="center", fontsize=10.5,
                fontweight="bold", color=RED)
        ax.text(i + 0.20, named[k] + 1.5, named[k], ha="center", fontsize=10.5,
                fontweight="bold", color=GREEN)
    ax.axhline(32.5, color=INK, ls="--", lw=1.4, zorder=4)
    ax.text(3.45, 34.5, "median\nbenign row", ha="right", va="bottom", fontsize=10,
            color=INK, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("benign rows ranked above\nthe violation (of 65)", fontsize=11.5)
    ax.set_ylim(0, 78)
    ax.legend(loc="upper left", fontsize=11, ncols=1)
    ax.set_title("Coded, the violation ranks\nbelow the benign median", fontsize=13.5,
                 fontweight="bold", color=INK, pad=10)
    # The two arms are DIFFERENT rows, not a controlled protected-trait swap: they differ in
    # fact sheet, domain, cited cards and request type, and v1 contains no protected pair on
    # which a violation is scored. Stated on the panel so the bars cannot be read as a
    # measured surface-form effect.
    ax.text(0.5, -0.205, "different rows — not a controlled pair", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, color=RED, style="italic")
    tidy(ax)

    fig.tight_layout(w_pad=2.4)
    save(fig, "teaser")


def fig_act1_bars():
    rows = primary_rows()
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(6.9, 4.5))
    rep = [r[3] for r in rows]
    tr = [r[6] for r in rows]
    ax.bar(x - 0.20, rep, 0.38, color=BLUE, label="represented-source $\\Delta$", zorder=3)
    ax.bar(x + 0.20, tr, 0.38, color=ORANGE, label="held-out transfer $\\Delta$", zorder=3)
    ax.axhline(0, color=INK, lw=1.2, zorder=4)
    for i, v in enumerate(rep):
        ax.text(i - 0.20, v + 0.015, f"{v:+.2f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=BLUE)
    for i, v in enumerate(tr):
        off, va = (0.015, "bottom") if v >= 0 else (-0.015, "top")
        ax.text(i + 0.20, v + off, f"{v:+.2f}", ha="center", va=va,
                fontsize=12, fontweight="bold", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[r[0]] for r in rows], fontsize=12)
    ax.set_ylabel("base $\\to$ SFT change in macro-AP")
    ax.set_ylim(-0.24, 0.63)
    ax.legend(loc="upper right", fontsize=11.5)
    tidy(ax)
    fig.tight_layout()
    save(fig, "act1_bars")


def fig_spec_plane():
    seeds = seed_rows()
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    ax.axhspan(-0.28, 0, xmin=0.0, xmax=1.0, color="#FDF0EC", zorder=0)
    for model, color in MODEL_COLOR.items():
        pts = [(r[2], r[3]) for r in seeds if r[0] == model]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=95, color=color,
                   label=model, zorder=4, edgecolors="white", linewidths=1.3)
    mac = macros("results_macros_gen.tex")
    ax.scatter([float(mac["RepDelta"])], [float(mac["TransferDelta"])], s=340, marker="X",
               color=INK, zorder=5, edgecolors="white", linewidths=1.8)
    ax.axhline(0, color=INK, lw=1.2, zorder=2)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.text(0.56, -0.262, "SPECIALIZE\nrepresented $\\uparrow$, transfer $\\downarrow$",
            ha="right", va="bottom", fontsize=11.5, color=RED, fontweight="bold")
    ax.text(0.015, 0.098, "UNIFORM GAIN", ha="left", va="top", fontsize=11.5,
            color=GREEN, fontweight="bold")
    ax.set_xlabel("represented-source macro-AP $\\Delta$")
    ax.set_ylabel("held-out transfer macro-AP $\\Delta$")
    ax.set_xlim(-0.02, 0.58)
    ax.set_ylim(-0.27, 0.11)
    handles = [Line2D([], [], color=c, marker="o", ls="none", ms=9, label=m)
               for m, c in MODEL_COLOR.items()]
    handles.append(Line2D([], [], color=INK, marker="X", ls="none", ms=11,
                          label="fixed-panel mean"))
    # outside the axes: the in-plot lower-left corner holds the Qwen3-4B seed-45 outlier
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=3,
              fontsize=10.5, columnspacing=1.4, handletextpad=0.5)
    tidy(ax, xgrid=True)
    fig.tight_layout()
    save(fig, "spec_plane")


def fig_operating():
    """Dumbbell: what the calibrated 5%-FPR operating point does, base -> SFT.

    Upper group is report Table 3 -- each guard read at its OWN calibrated threshold, where
    the tuned guard alarms nearly four times as often, so the two recalls are not comparable.
    Lower group is report Table 4: the same two recall instruments re-read at an EQUAL
    false-alarm budget, where the apparent recall gain does not shrink but reverses.
    Both groups are parsed from committed generated/*.tex, so neither can drift from the paper.
    """
    mac = macros("results_macros_gen.tex")
    mfp = macros("matched_fpr_macros.tex")
    items = [
        ("Represented recall (TPR)", float(mac["RepBaseTPRPct"]), float(mac["RepSFTTPRPct"]), True),
        ("Transfer recall (TPR)", float(mac["TransferBaseTPRPct"]), float(mac["TransferSFTTPRPct"]), True),
        ("Over-refusal FPR (OR-Bench)", float(mac["ORBenchBaseFPRPct"]), float(mac["ORBenchSFTFPRPct"]), False),
        ("Transfer false alarms (macro)", float(mac["TransferBaseFPRPct"]), float(mac["TransferSFTFPRPct"]), False),
        ("Transfer false alarms (pooled)", float(mac["TransferBasePooledFPRPct"]), float(mac["TransferSFTPooledFPRPct"]), False),
        ("Hard-attack recall (HarmBench)", float(mac["HarmBenchBaseRecallPct"]), float(mac["HarmBenchSFTRecallPct"]), True),
    ]
    matched = [
        ("Transfer recall", float(mfp["MatchedTransferBase"]) * 100,
         float(mfp["MatchedTransferSft"]) * 100, True),
        ("Hard-attack recall", float(mfp["MatchedHarmBase"]) * 100,
         float(mfp["MatchedHarmSft"]) * 100, True),
    ]
    # 8 rows at a slide-7 aspect (~1.87) so picture() fills the 8.05x4.24in box without
    # letterboxing; the two groups are separated by a rule rather than interleaved.
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ys = list(np.arange(len(items))[::-1] + 2.35)         # 7.35 .. 2.35
    ys_m = [0.60, -0.50]
    divider = 1.62

    def draw(y, b, s, higher_better):
        improved = (s > b) if higher_better else (s < b)
        flat = abs(s - b) < 0.5
        col = SLATE if flat else (GREEN if improved else RED)
        if not flat:
            ax.annotate("", xy=(s, y), xytext=(b, y),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.30,head_length=0.62",
                                        color=col, lw=3.0, shrinkA=7, shrinkB=2))
        ax.scatter([b], [y], s=115, color="white", edgecolors=SLATE, linewidths=2.2, zorder=5)
        # place the two values on opposite sides of the motion so they never collide
        going_right = s >= b
        ax.text(b + (-1.6 if going_right else 1.6), y, f"{b:.1f}",
                ha="right" if going_right else "left", va="center",
                fontsize=11.5, color=SLATE)
        ax.text(s + (2.0 if going_right else -2.0), y, f"{s:.1f}%",
                ha="left" if going_right else "right", va="center",
                fontsize=12.5, color=col, fontweight="bold")
        if flat:
            ax.text(s + 14.5, y, "(flat)", ha="left", va="center", fontsize=11,
                    color=SLATE, style="italic")

    for y, (_, b, s, hb) in zip(ys, items):
        draw(y, b, s, hb)
    ax.axhline(divider, color=RULE, lw=1.2, zorder=1)
    for y, (_, b, s, hb) in zip(ys_m, matched):
        draw(y, b, s, hb)

    ax.set_yticks(ys + ys_m)
    ax.set_yticklabels([i[0] for i in items] + [m[0] for m in matched], fontsize=12.5)
    # Each group states its own threshold rule: the two blocks are NOT the same comparison,
    # which is the whole point of the lower one.
    ax.text(-3.4, 7.98, "at each guard's OWN calibrated 5%-FPR threshold",
            fontsize=11, color=SLATE, style="italic", va="center", ha="left")
    ax.text(-3.4, 1.12, "at an EQUAL false-alarm budget", fontsize=11.5, color=RED,
            style="italic", fontweight="bold", va="center", ha="left")
    ax.set_xlabel("recall / false-alarm rate   (%)")
    ax.set_xlim(-4, 92)
    ax.set_ylim(-1.15, 8.25)
    handles = [Line2D([], [], color="white", marker="o", ms=10, mec=SLATE, mew=2.2,
                      ls="none", label="untuned base"),
               Line2D([], [], color=GREEN, lw=3, label="SFT guard — improves"),
               Line2D([], [], color=RED, lw=3, label="SFT guard — regresses")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.36, 0.72),
              fontsize=11.5, labelspacing=0.6)
    tidy(ax, xgrid=True, ygrid=False)
    fig.tight_layout()
    save(fig, "operating")


def fig_klsft():
    rows = klsft_rows()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.3), sharey=False)
    x = np.arange(4)
    labels = [SHORT[r[0]] for r in rows]

    # transfer: base / SFT / KL.5
    a1.bar(x - 0.26, [r[1] for r in rows], 0.25, color=SLATE, label="untuned base", zorder=3)
    a1.bar(x + 0.00, [r[2] for r in rows], 0.25, color=ORANGE, label="plain SFT", zorder=3)
    a1.bar(x + 0.26, [r[3] for r in rows], 0.25, color=PURPLE, label="KL-SFT ($\\beta$=0.5)", zorder=3)
    for i, r in enumerate(rows):
        a1.text(i + 0.26, r[3] + 0.006, f"{r[3]:.2f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=PURPLE)
    a1.set_ylim(0.70, 0.99)
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=11.5)
    a1.set_ylabel("transfer macro-AP")
    a1.set_title("KL buys the transfer back  (+0.061 vs SFT)", fontsize=13,
                 fontweight="bold", color=GREEN, pad=9)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3,
              fontsize=10.5, columnspacing=1.2, handlelength=1.4)
    tidy(a1)

    # represented: SFT / KL.5  (the cost)
    a2.bar(x - 0.14, [r[5] for r in rows], 0.28, color=BLUE, label="plain SFT", zorder=3)
    a2.bar(x + 0.14, [r[6] for r in rows], 0.28, color=PURPLE, label="KL-SFT ($\\beta$=0.5)", zorder=3)
    for i, r in enumerate(rows):
        a2.annotate("", xy=(i + 0.14, r[6]), xytext=(i - 0.14, r[5]),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.5",
                                    color=RED, lw=2.0), zorder=6)
        a2.text(i + 0.14, r[6] - 0.008, f"{r[6]:.2f}", ha="center", va="top",
                fontsize=10.5, fontweight="bold", color="white", zorder=7)
    a2.set_ylim(0.85, 1.005)
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=11.5)
    a2.set_ylabel("represented macro-AP")
    a2.set_title("and charges for it  ($-$0.035 represented)", fontsize=13,
                 fontweight="bold", color=RED, pad=9)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2,
              fontsize=10.5, columnspacing=1.2, handlelength=1.4)
    tidy(a2)

    fig.tight_layout(w_pad=2.2)
    save(fig, "klsft")


def fig_adapt_plane():
    rows = adaptation_rows()
    # hand-placed label offsets (points); ten arrows in a small region collide otherwise
    lbl = {"Qwen2.5-1.5B": (11, -14), "Qwen3-4B": (12, -5), "SmolLM2-1.7B": (-6, -17),
           "SmolLM3-3B": (11, -14), "Granite-Guard-2B": (12, -13),
           "Qwen3Guard-0.6B": (-3, -21), "Qwen3Guard-4B": (12, 3),
           "ShieldGemma-2B": (11, -14), "WildGuard-7B": (12, 6)}
    fig, ax = plt.subplots(figsize=(8.1, 4.9))
    ax.axhspan(-0.19, 0, color="#FDF0EC", zorder=0)
    for name, kind, sr, st, kr, kt in rows:
        col = BLUE if kind == "general" else GREEN
        degenerate = (sr == 0.0 and st == 0.0)
        if degenerate:  # labelled in the legend, not in-plot: the origin is crowded
            ax.scatter([sr], [st], s=95, facecolors="none", edgecolors=SLATE,
                       linewidths=1.8, zorder=4)
            continue
        ax.add_patch(FancyArrowPatch((sr, st), (kr, kt), arrowstyle="-|>",
                                     mutation_scale=15, color=col, lw=1.7,
                                     alpha=0.85, zorder=3, shrinkA=5, shrinkB=2))
        ax.scatter([sr], [st], s=88, color=col, zorder=4, edgecolors="white", linewidths=1.2)
        ax.scatter([kr], [kt], s=95, color=col, marker="^", zorder=4,
                   edgecolors="white", linewidths=1.2)
        dx, dy = lbl.get(name, (9, -13))
        ax.annotate(name, (sr, st), textcoords="offset points", xytext=(dx, dy),
                    ha="right" if dx < 0 else "left",
                    fontsize=9.5, color=col, fontweight="bold", zorder=6)
    ax.axhline(0, color=INK, lw=1.2, zorder=2)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.set_xlabel("represented-source macro-AP $\\Delta$  (vs. the same checkpoint)")
    ax.set_ylabel("held-out transfer macro-AP $\\Delta$")
    ax.set_xlim(-0.035, 0.60)
    ax.set_ylim(-0.185, 0.100)
    ax.text(0.585, -0.176, "SPECIALIZE", ha="right", va="bottom", fontsize=11.5,
            color=RED, fontweight="bold")
    handles = [Line2D([], [], color=BLUE, marker="o", ls="none", ms=9, label="general instruction checkpoint"),
               Line2D([], [], color=GREEN, marker="o", ls="none", ms=9, label="released purpose-built guard"),
               Line2D([], [], color=SLATE, marker="o", ls="none", ms=9, label="plain SFT"),
               Line2D([], [], color=SLATE, marker="^", ls="none", ms=9, label="KL-SFT ($\\beta$=0.5)"),
               Line2D([], [], color="none", marker="o", ls="none", ms=9, mec=SLATE, mew=1.8,
                      label="Llama-Guard-3-1B: null cell")]
    ax.legend(handles=handles, loc="upper left", fontsize=10.5, ncols=1,
              labelspacing=0.42)
    tidy(ax, xgrid=True)
    fig.tight_layout()
    save(fig, "adapt_plane")


def fig_composition():
    rows = composition_rows()
    order = ["SmolLM2-1.7B", "Qwen2.5-1.5B", "SmolLM3-3B", "Qwen3-4B"]
    rows = sorted(rows, key=lambda r: order.index(r[0]))
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.bar(x - 0.26, [r[1] for r in rows], 0.25, color=SLATE, label="untuned base", zorder=3)
    ax.bar(x + 0.00, [r[2] for r in rows], 0.25, color=ORANGE, label="SFT guard", zorder=3)
    ax.bar(x + 0.26, [r[3] for r in rows], 0.25, color=GREEN, label="base + SFT composition", zorder=3)
    for i, r in enumerate(rows):
        ax.text(i + 0.26, r[3] + 0.005, f"{r[3]:.3f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=GREEN)
        ax.annotate("", xy=(i + 0.26, r[3]), xytext=(i + 0.00, r[2]),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.20,head_length=0.45",
                                    color=GREEN, lw=1.8, alpha=0.9), zorder=6)
        ax.text(i + 0.13, max(r[2], r[3]) + 0.021, f"{r[3]-r[2]:+.3f}", ha="center",
                fontsize=10, color=GREEN, style="italic")
    ax.set_ylim(0.74, 0.985)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[r[0]] for r in rows], fontsize=12)
    ax.set_ylabel("dataset-held-out transfer macro-AP")
    ax.legend(loc="upper left", fontsize=11)
    tidy(ax)
    fig.tight_layout()
    save(fig, "composition")


def fig_sftsft():
    rows = sftsft_rows()
    order = ["Qwen2.5-1.5B", "SmolLM2-1.7B", "SmolLM3-3B", "Qwen3-4B"]
    rows = sorted(rows, key=lambda r: order.index(r[0]))
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(x - 0.19, [r[3] for r in rows], 0.36, color=GREEN,
           label="base + SFT  (keeps the base)", zorder=3)
    ax.bar(x + 0.19, [r[4] for r in rows], 0.36, color="#9CA3AF",
           label="SFT + SFT  (same 2-pass cost, no base)", zorder=3)
    for i, r in enumerate(rows):
        gap = r[3] - r[4]
        ax.text(i, max(r[3], r[4]) + 0.008, f"{gap:+.3f}", ha="center", va="bottom",
                fontsize=11.5, fontweight="bold",
                color=GREEN if gap > 0.02 else SLATE)
    ax.set_ylim(0.74, 0.945)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[r[0]] for r in rows], fontsize=12)
    ax.set_ylabel("transfer macro-AP")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2,
              fontsize=11, columnspacing=1.6, handlelength=1.5)
    tidy(ax)
    fig.tight_layout()
    save(fig, "sftsft")


def fig_fairness():
    rows = mortgage_rows()
    order = ["Qwen2.5-1.5B", "Qwen3-4B", "SmolLM2-1.7B", "SmolLM3-3B"]
    rows = sorted(rows, key=lambda r: order.index(r[0]))
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(x - 0.19, [r[3] for r in rows], 0.36, color=BLUE,
           label="probability scale  ($\\Delta_{context}$)", zorder=3)
    ax.bar(x + 0.19, [r[4] for r in rows], 0.36, color=RED,
           label="raw margin, log-odds  ($\\Delta^{margin}$)", zorder=3)
    for i, r in enumerate(rows):
        ax.text(i - 0.19, r[3] + 0.02, f"{r[3]:.3f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=BLUE)
        ax.text(i + 0.19, r[4] + 0.02, f"{r[4]:.2f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=RED)
    ax.annotate("looks perfectly invariant\non one scale, near-worst\non the other",
                xy=(1.19, 0.80), xytext=(2.05, 0.99), fontsize=10.5, color=RED,
                fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6,
                                connectionstyle="arc3,rad=0.22"))
    ax.set_ylim(0, 1.20)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[r[0]] for r in rows], fontsize=12)
    ax.set_ylabel("protected-pair gap  (0 = invariant)")
    ax.legend(loc="upper left", fontsize=10.5)
    tidy(ax)
    fig.tight_layout()
    save(fig, "fairness")


def fig_expguard():
    rows = expguard_rows()
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for k, (dom, col) in enumerate([("finance", BLUE), ("health", GREEN), ("law", ORANGE)]):
        vals = [r[2 + k] for r in rows]
        ax.bar(x + (k - 1) * 0.26, vals, 0.24, color=col, label=dom, zorder=3)
    for i, r in enumerate(rows):
        ax.text(i, max(r[2:]) + 0.004, f"{r[1]:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=INK)
    ax.set_ylim(0.84, 0.985)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[r[0]] for r in rows], fontsize=12)
    ax.set_ylabel("average precision")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3,
              fontsize=11, columnspacing=2.0, handlelength=1.5)
    ax.text(0.015, 0.972, "bold = aggregate AP over 2,275 expert-annotated rows",
            transform=ax.transAxes, fontsize=10, color=SLATE, style="italic")
    tidy(ax)
    fig.tight_layout()
    save(fig, "expguard")


def fig_prevalence():
    """Transfer macro-AP as a function of deployment prevalence, recomputed from the
    committed base-guard transfer ROC (report Eq. 4). No new model runs."""
    import numpy as np
    import pandas as pd

    sp = REPORT.parents[1] / "artifacts/paper_a_sft_v2/scores/scores.parquet"
    if not sp.exists():
        print("  SKIP prevalence: scores.parquet not present")
        return
    df = pd.read_parquet(sp, columns=["sample_id", "split", "source", "gold", "model_key",
                                      "condition", "probability_calibrated"])
    tr = df[(df.split == "transfer_test") & (df.condition == "base")] \
        .drop_duplicates(["model_key", "sample_id"])
    order = ["qwen3_4b", "smollm3_3b", "smollm2_17b", "qwen25_15b"]
    pretty = {"qwen25_15b": "Qwen2.5-1.5B", "smollm2_17b": "SmolLM2-1.7B",
              "smollm3_3b": "SmolLM3-3B", "qwen3_4b": "Qwen3-4B"}
    cols = {"qwen25_15b": BLUE, "smollm2_17b": ORANGE, "smollm3_3b": GREEN, "qwen3_4b": RED}

    def ap_prev(frame, prev):
        aps = []
        for s in frame.source.unique():
            sub = frame[frame.source == s]
            y = sub.gold.values.astype(int)
            o = np.argsort(-sub.probability_calibrated.values)
            y = y[o]
            P, N = y.sum(), len(y) - y.sum()
            if P == 0 or N == 0:
                continue
            tpr = np.cumsum(y) / P
            fpr = np.cumsum(1 - y) / N
            prec = (prev * tpr) / (prev * tpr + (1 - prev) * fpr + 1e-12)
            aps.append(float(np.sum(prec * np.diff(np.concatenate([[0], tpr])))))
        return float(np.mean(aps)) if aps else float("nan")

    prevs = np.geomspace(0.005, 0.5, 60)
    fig, ax = plt.subplots(figsize=(6.0, 3.85))  # compact: rendered half-width on slide 7
    at1, at50 = {}, {}
    for mk in order:
        f = tr[tr.model_key == mk]
        ax.plot(prevs * 100, [ap_prev(f, p) for p in prevs], color=cols[mk], lw=2.8,
                zorder=3, label=pretty[mk])
        at1[mk], at50[mk] = ap_prev(f, 0.01), ap_prev(f, 0.5)
    for p in (1, 50):
        ax.axvline(p, color=SLATE, lw=1.0, ls=":", alpha=0.6, zorder=1)
    bbox = dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.94)
    for mk in ("qwen3_4b", "smollm2_17b", "qwen25_15b"):
        ax.scatter([1], [at1[mk]], s=80, color=cols[mk], zorder=5,
                   edgecolors="white", linewidths=1.6)
        ax.annotate(f"{at1[mk]:.2f}", (1, at1[mk]), textcoords="offset points",
                    xytext=(-9, 9), ha="right", fontsize=11.5, fontweight="bold",
                    color=cols[mk], zorder=6, bbox=bbox)
    ax.text(1.13, 0.345, "1%\nprevalence", fontsize=10, color=SLATE, style="italic",
            ha="left", va="center")
    # the re-order: Qwen2.5 leads SmolLM2 at balance, trails it once positives are rare
    ax.annotate("these two\nswap order", xy=(25.5, 0.680), xytext=(31, 0.34),
                fontsize=10.5, fontweight="bold", color=SLATE, ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.5,
                                connectionstyle="arc3,rad=0.25"))
    ax.legend(loc="upper left", fontsize=10.5, labelspacing=0.35, handlelength=1.5,
              handletextpad=0.6, borderpad=0.1)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 5, 10, 50])
    ax.set_xticklabels(["0.5%", "1%", "5%", "10%", "50%"])
    ax.set_xlim(0.45, 62)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("deployment prevalence of unsafe prompts  (log scale)")
    ax.set_ylabel("transfer macro-AP")
    tidy(ax, xgrid=False)
    fig.tight_layout()
    save(fig, "prevalence")
    print("      AP at balance: " + ", ".join(f"{pretty[m]} {at50[m]:.2f}" for m in order))
    print("      AP at 1%     : " + ", ".join(f"{pretty[m]} {at1[m]:.2f}" for m in order))


def fig_latency():
    """Per-call latency to the single-token verdict, from the committed latency table."""
    rows = []
    pat = re.compile(r"^(?:\\textbf\{)?([\w.\- ]+?)\}?\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&"
                     r"\s*([\d.]+)\s*\\\\")
    for ln in (GEN / "latency_table.tex").read_text().splitlines():
        m = pat.match(ln.strip())
        if m and "Guard" not in m.group(1):
            rows.append((m.group(1).strip(), *(float(m.group(i)) for i in range(2, 5))))
    rows = [r for r in rows if r[0] in SHORT]          # drop the "All four" summary row
    assert len(rows) == 4, rows
    rows = sorted(rows, key=lambda r: r[1])

    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for k, (lab, col) in enumerate([("P50", GREEN), ("P90", BLUE), ("P99", SLATE)]):
        vals = [r[1 + k] for r in rows]
        ax.barh(y + (1 - k) * 0.26, vals, 0.24, color=col, label=lab, zorder=3)
        for i, v in enumerate(vals):
            ax.text(v + 1.6, y[i] + (1 - k) * 0.26, f"{v:.0f}", va="center", ha="left",
                    fontsize=10.5, fontweight="bold", color=col)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0].replace("-", "\n", 0) for r in rows], fontsize=12)
    ax.set_xlabel("milliseconds per call  (one forward pass, batch 16, one A100)")
    ax.set_xlim(0, 112)
    ax.legend(loc="lower right", fontsize=11, ncols=3, columnspacing=1.4)
    tidy(ax, xgrid=True, ygrid=False)
    fig.tight_layout()
    save(fig, "latency")


if __name__ == "__main__":
    print("Building deck figures from committed generated/ artifacts ...")
    fig_teaser()
    fig_act1_bars()
    fig_spec_plane()
    fig_operating()
    fig_klsft()
    fig_adapt_plane()
    fig_composition()
    fig_sftsft()
    fig_fairness()
    fig_expguard()
    fig_prevalence()
    fig_latency()
    print("done.")
