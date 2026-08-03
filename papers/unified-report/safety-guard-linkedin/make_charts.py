"""
LinkedIn article charts — all values taken verbatim from
"Safety Benchmark Gains Do Not Guarantee Safety Transfer" (unified_report.pdf).
Source table for every number is noted in each function's docstring.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import os

BG      = "#0b1220"
PANEL   = "#131d33"
TEXT    = "#eaf0f8"
MUTED   = "#8ea4c2"
BLUE    = "#9ec9f0"
BLUE_D  = "#4b86c6"
RED     = "#e8615a"
GREEN   = "#6fd6ab"
GREY    = "#5a6b87"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": "DejaVu Sans", "font.size": 15,
    "axes.edgecolor": "#26344f", "axes.linewidth": 1.0,
})

OUT = "charts"
os.makedirs(OUT, exist_ok=True)
FIGSIZE = (10, 5.625)   # 16:9 -> 1600x900 at dpi 160
DPI = 160


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=DPI, bbox_inches="tight",
                pad_inches=0.35, facecolor=BG)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. HERO
def hero():
    """Table 3 (+0.3234 represented) and Table 6 (transfer recall 0.517 -> 0.217)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    ax.text(50, 95, "SAME MODELS.   SAME ROWS.   SAME FALSE-ALARM BUDGET.",
            ha="center", va="top", fontsize=13.5, color=MUTED)

    ax.add_patch(Rectangle((3, 24), 45, 56, facecolor=PANEL, edgecolor="#20304d", lw=1.2))
    ax.text(25.5, 73, "THE BENCHMARK SCORE", ha="center", va="center", fontsize=12.5, color=MUTED)
    ax.text(25.5, 55, "+0.32", ha="center", va="center", fontsize=54, color=GREEN, fontweight="bold")
    ax.text(25.5, 35, "macro-AP on the sources\nit was trained on", ha="center", va="center",
            fontsize=13.5, color=TEXT, linespacing=1.6)

    ax.add_patch(Rectangle((52, 24), 45, 56, facecolor=PANEL, edgecolor="#20304d", lw=1.2))
    ax.text(74.5, 73, "WHAT IT ACTUALLY CATCHES", ha="center", va="center", fontsize=12.5, color=MUTED)
    ax.text(74.5, 55, "52% " + chr(8594) + " 22%", ha="center", va="center", fontsize=40,
            color=RED, fontweight="bold")
    ax.text(74.5, 35, "recall on unfamiliar attacks,\nat an equal false-alarm budget", ha="center",
            va="center", fontsize=13.5, color=TEXT, linespacing=1.6)

    ax.text(50, 14, "The benchmark went up. The protection went down.",
            ha="center", va="center", fontsize=19.5, color=TEXT, fontweight="bold")
    ax.text(50, 5, "4 checkpoints " + chr(183) + " 5 seeds each " + chr(183) +
            " every guard compared only with its own pre-tuning base",
            ha="center", va="center", fontsize=12, color=MUTED)
    save(fig, "1_hero")


# ------------------------------------------------- 2. PER-CHECKPOINT SPLIT
def split():
    """Table 3: paired base->SFT macro-AP change, represented and transfer."""
    names = ["SmolLM2-1.7B", "Qwen2.5-1.5B", "SmolLM3-3B", "Qwen3-4B"]
    base  = [".452", ".633", ".662", ".885"]
    rep   = [0.528, 0.354, 0.313, 0.098]
    tra   = [0.040, -0.039, -0.087, -0.150]
    y = range(len(names))[::-1]
    y = list(y)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    h = 0.34
    for i, yy in enumerate(y):
        ax.barh(yy + h/1.9, rep[i], height=h, color=BLUE, zorder=3)
        ax.barh(yy - h/1.9, tra[i], height=h, color=RED if tra[i] < 0 else GREEN, zorder=3)
        ax.text(rep[i] + 0.012, yy + h/1.9, f"+{rep[i]:.3f}", va="center", fontsize=14,
                color=BLUE, fontweight="bold")
        off = -0.012 if tra[i] < 0 else 0.012
        ha = "right" if tra[i] < 0 else "left"
        ax.text(tra[i] + off, yy - h/1.9, f"{tra[i]:+.3f}", va="center", ha=ha, fontsize=14,
                color=RED if tra[i] < 0 else GREEN, fontweight="bold")

    ax.axvline(0, color="#3d4f70", lw=1.4, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\nstarting score {b}" for n, b in zip(names, base)], fontsize=13.5,
                       color=TEXT, linespacing=1.5)
    ax.set_xlim(-0.26, 0.62)
    ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel("change in ranking accuracy after fine-tuning  (macro-AP)", fontsize=13.5, labelpad=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#1d2942", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=10)

    ax.set_ylim(-0.62, 4.25)
    ax.text(-0.245, 3.80, "■ traffic it has never seen", color=RED, fontsize=14, fontweight="bold")
    ax.text(0.30, 3.80, "■ traffic it was trained on", color=BLUE, fontsize=14, fontweight="bold")
    ax.set_title("The gain is real. It just doesn't travel.",
                 fontsize=19, color=TEXT, loc="left", pad=52, fontweight="bold")
    ax.text(-0.245, 4.50, "Rows ordered weakest starting model (top) to strongest (bottom)",
            fontsize=13.5, color=MUTED, clip_on=False)
    save(fig, "2_split")


# --------------------------------------------- 3. MATCHED-BUDGET COLLAPSE
def matched():
    """Table 5 (own-threshold) and Table 6 (equal budget)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    rows = [
        ("Recall on\nunfamiliar attacks", 0.517, 0.581, 0.217),
        ("Recall on the\nhardest attacks", 0.780, 0.600, 0.203),
    ]
    ys = [1.55, 0.35]
    for (label, b, own, mt), yy in zip(rows, ys):
        ax.plot([min(b, own), max(b, own)], [yy + 0.20, yy + 0.20], color=GREY, lw=2.2,
                zorder=2, solid_capstyle="round")
        ax.scatter([b], [yy + 0.20], s=200, color=TEXT, zorder=5, edgecolor=BG, linewidth=2)
        ax.scatter([own], [yy + 0.20], s=160, color=GREY, zorder=5, edgecolor=BG, linewidth=2)
        lo = min(b, own)
        for val, col, fw, fs in ((b, TEXT, "bold", 15), (own, MUTED, "normal", 13.5)):
            if val == lo:
                ax.text(val - 0.016, yy + 0.20, str(round(val*100, 1)) + "%", ha="right",
                        va="center", fontsize=fs, color=col, fontweight=fw)
            else:
                ax.text(val + 0.016, yy + 0.20, str(round(val*100, 1)) + "%", ha="left",
                        va="center", fontsize=fs, color=col, fontweight=fw)
        ax.add_patch(FancyArrowPatch((b, yy - 0.16), (mt, yy - 0.16), arrowstyle="-|>",
                                     mutation_scale=24, color=RED, lw=3.0, zorder=4))
        ax.text(mt - 0.022, yy - 0.16, str(round(mt*100, 1)) + "%", ha="right", va="center",
                fontsize=17, color=RED, fontweight="bold")

    ax.scatter([], [], s=140, color=TEXT, label="untuned base")
    ax.scatter([], [], s=120, color=GREY, label="fine-tuned, at its own calibrated threshold")
    ax.plot([], [], color=RED, lw=3, label="fine-tuned, re-read at the base's own alarm budget")
    ax.legend(loc="upper right", frameon=False, fontsize=12.5,
              labelcolor=[TEXT, MUTED, RED], handletextpad=0.9, borderpad=0.2)

    ax.set_xlim(0.02, 0.98); ax.set_ylim(-0.35, 2.75)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=14, color=TEXT, linespacing=1.5)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xticklabels(["20%", "40%", "60%", "80%"])
    ax.set_xlabel("share of unsafe prompts caught", fontsize=13.5, labelpad=12)
    ax.tick_params(axis="y", length=0, pad=12)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="x", color="#1d2942", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Let the tuned guard alarm more often and it looks better.\n"
                 "Hold the alarm rate fixed and it catches less than half of what its own base caught.",
                 fontsize=16.5, color=TEXT, loc="left", pad=22, fontweight="bold", linespacing=1.6)
    save(fig, "3_matched_budget")


# ------------------------------------------- 4. THE METRIC UNDERSTATES IT
def metric():
    """Table 7: same eight cells under macro-AP vs partial AUC over FPR [0, 0.05]."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels = ["Gain on familiar traffic", "Cost on unfamiliar traffic"]
    ap     = [0.323, -0.059]
    pauc   = [0.686, -0.174]
    x = [0, 1]
    w = 0.30
    for i in x:
        c = BLUE if ap[i] > 0 else RED
        ax.bar(i - w/1.8, ap[i], width=w, color=c, alpha=0.42, zorder=3)
        ax.bar(i + w/1.8, pauc[i], width=w, color=c, zorder=3)
        va1 = "bottom" if ap[i] > 0 else "top"
        o = 0.018 if ap[i] > 0 else -0.018
        ax.text(i - w/1.8, ap[i] + o, f"{ap[i]:+.3f}", ha="center", va=va1, fontsize=15, color=c)
        ax.text(i + w/1.8, pauc[i] + o, f"{pauc[i]:+.3f}", ha="center", va=va1, fontsize=17,
                color=c, fontweight="bold")
        ymid = (ap[i] + pauc[i]) / 2
        ax.text(i + 0.42, ymid, ["2.1x bigger", "3.0x bigger"][i].replace("x", chr(215)),
                fontsize=16, color=TEXT, fontweight="bold", va="center")

    ax.axhline(0, color="#3d4f70", lw=1.4, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=15, color=TEXT)
    ax.set_xlim(-0.60, 1.78); ax.set_ylim(-0.44, 0.80)
    ax.set_ylabel("size of the effect", fontsize=13.5, labelpad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#1d2942", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.text(-0.55, -0.335, chr(9608) + "  averaged over the whole ranking " +
            chr(183) + " macro-AP, the metric everyone publishes", fontsize=13, color="#6d7f9c")
    ax.text(-0.55, -0.405, chr(9608) + "  read only inside a 5% false-alarm budget " +
            chr(183) + " where a guard is actually placed", fontsize=13, color=TEXT)
    ax.set_title("The metric everyone reports understates both halves of the trade",
                 fontsize=18, color=TEXT, loc="left", pad=22, fontweight="bold")
    save(fig, "4_metric")


# ------------------------------------------------- 5. MORTGAGE QUADRANT
def quadrant():
    """Table 15 / Figure 9: frozen v1_hmda2022, 994 rows."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    cells = [
        # x, y, w, h, n, label, sub, color, emphasis
        (32, 52, 29, 26, "450", "benign", "must not be flagged", GREY, False),
        (64, 52, 29, 26, "0",   "general harm only", "empty in v1", GREY, False),
        (32, 22, 29, 26, "502", "READS SAFE\nIS A VIOLATION", "the payload", RED, True),
        (64, 22, 29, 26, "42",  "bad on both counts", "", GREY, False),
    ]
    for (x, y, w, h, n, lab, sub, col, emph) in cells:
        face = "#3a1f26" if emph else PANEL
        edge = RED if emph else "#24324e"
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, lw=2.2 if emph else 1.2))
        ax.text(x + w/2, y + h - 7, n, ha="center", va="center",
                fontsize=40 if emph else 32, color=RED if emph else TEXT, fontweight="bold")
        ax.text(x + w/2, y + 10.5 if emph else y + 9.5, lab, ha="center", va="center",
                fontsize=12 if emph else 13, color=RED if emph else MUTED,
                fontweight="bold" if emph else "normal", linespacing=1.35)
        if sub:
            ax.text(x + w/2, y + 3.6, sub, ha="center", va="center", fontsize=11.5, color=MUTED)

    ax.text(46.5, 81, "a general safety guard\ncalls it SAFE", ha="center", fontsize=13,
            color=MUTED, linespacing=1.4)
    ax.text(78.5, 81, "a general safety guard\ncalls it UNSAFE", ha="center", fontsize=13,
            color=MUTED, linespacing=1.4)
    ax.text(29, 65, "mortgage policy:\nALLOW", ha="right", va="center", fontsize=13,
            color=MUTED, linespacing=1.4)
    ax.text(29, 35, "mortgage policy:\nINTERVENE", ha="right", va="center", fontsize=13,
            color=MUTED, linespacing=1.4)

    ax.set_title("", pad=0)
    fig.text(0.06, 0.955, "502 of 994 rows read as perfectly safe — and are compliance violations",
             fontsize=18.5, color=TEXT, fontweight="bold")
    fig.text(0.06, 0.905, "A frozen, HMDA-grounded mortgage benchmark. Every row carries two independent labels.",
             fontsize=13.5, color=MUTED)
    fig.text(0.06, 0.045,
             "Zero-shot guards scored 0.67–0.85 detecting the policy label — against a 0.555 chance floor. "
             "0.12–0.30 above guessing.",
             fontsize=13, color=RED)
    save(fig, "5_quadrant")


# ------------------------------------------------- 6. THE REGIME REVERSAL
def regime():
    """Table 22 (+0.083 represented) and Table 18 (+0.109 hosted on ExpGuard)."""
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(left=0.30, right=0.955, top=0.74, bottom=0.32)

    rows = [
        (1.0, 0.083, "Traffic your training\nmanifest names", GREEN),
        (0.0, -0.109, "Traffic it\ndoes not name", RED),
    ]
    for yy, v, ylab, col in rows:
        ax.barh(yy, v, height=0.34, color=col, zorder=3)
        if v > 0:
            ax.text(v + 0.010, yy, "+" + format(v, ".3f"), va="center", ha="left",
                    fontsize=19, color=col, fontweight="bold")
        else:
            ax.text(v - 0.010, yy, format(abs(v), ".3f"), va="center", ha="right",
                    fontsize=19, color=col, fontweight="bold")

    ax.axvline(0, color="#42557a", lw=1.6, zorder=4)
    ax.text(-0.012, 1.78, chr(9664) + "  hosted frontier model wins", ha="right", va="center",
            fontsize=14, color=RED, fontweight="bold")
    ax.text(0.012, 1.78, "small self-hosted guard wins  " + chr(9654), ha="left", va="center",
            fontsize=14, color=GREEN, fontweight="bold")

    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels([r[2] for r in rows], fontsize=15, color=TEXT, linespacing=1.5)
    ax.tick_params(axis="y", length=0, pad=14)
    ax.set_xlim(-0.24, 0.24)
    ax.set_ylim(-0.62, 2.05)
    ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax.set_xlabel("difference in share of unsafe prompts caught, at a matched 5% false-alarm budget",
                  fontsize=12.5, labelpad=12)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="x", color="#1d2942", lw=0.9, zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.045, 0.90, "Which guard is better reverses with the traffic",
             fontsize=21, color=TEXT, fontweight="bold", va="top")
    fig.text(0.045, 0.145,
             "Two different comparisons, on different data, reported side by side and never pooled.\n"
             "The upper bar is a post-hoc summary over three purposively chosen sources; resampling\n"
             "that source set widens it to [-0.019, +0.220]. Read it as directional.",
             fontsize=11.5, color=MUTED, linespacing=1.8, va="top")
    fig.savefig(f"{OUT}/6_regime.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("wrote 6_regime")


if __name__ == "__main__":
    hero(); split(); matched(); metric(); quadrant(); regime()
