#!/usr/bin/env python
"""Build the executive deck's figures from the committed scores, not from typed numbers.

    python slides/make_exec_figures.py     (from papers/unified-report/)
    -> slides/assets/exec_gap.png, exec_tax.png, exec_scale.png

Every value plotted here is computed by `frontier.py` from the same text-free per-row
scores the report's tables are built from, so a figure cannot disagree with the paper any
more than a table can.

FORM CHOICES, and why each is the form rather than a default:

  exec_gap    The story is "one of these is materially better", not "here are eight
              guards". That is *emphasis*: the hosted bar carries the accent, the best
              open-weights bar carries green, everything else recedes to gray. A
              categorical palette here would bury the one bar that matters.

  exec_tax    Two measures, same unit (AP points), one signed axis -- so one chart with a
              zero rule, never a dual axis. The reader's job is polarity (which way did
              tuning move each regime) and the trend across base strength, so bars run
              signed from a shared zero and the rows are ordered by how strong the base
              already was. That ordering IS the finding.

  exec_scale  Magnitude across an ordered scale (4B -> 8B -> 32B) against a fixed
              reference. Columns for the ladder, a single hairline rule for the hosted
              model: the gap is the empty space between them, which is the point.

Palette comes from `deck_theme`, shared with both decks, and the surface is the DARK
content-slide background so a figure sits on the slide with no visible panel. The earlier
contrast validation in this docstring was run against a white surface and no longer applies;
the dark palette has not been re-run through the dataviz checker, so treat the colour choices
as inherited from the redesign rather than independently validated. Direct labels on every
mark remain the secondary encoding that carries the reading if a hue pair is hard to separate.
Sans throughout -- no serif on a numeral.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import json  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

import frontier as FR  # noqa: E402

# ── brand ink (from deck_theme, shared with both decks) ─────────────────────────
# SURFACE is the CONTENT-SLIDE background, so these figures sit on the slide with no visible
# panel. savefig.facecolor is set explicitly: it does not inherit from figure.facecolor, and
# omitting it is exactly how a dark figure ends up saved on a white canvas.
import deck_theme as T  # noqa: E402

ACCENT = T.hexc(T.ACCENT)              # hosted / the cost side
BLUE = T.hexc(T.DATA_REPRESENTED)      # the gain side
GREEN = T.hexc(T.DATA_COMPOSITION)     # best open-weights
INK = T.hexc(T.TEXT)
SECOND = T.hexc(T.BODY)
MUTED = T.hexc(T.DIM)
GRID = T.hexc(T.CARD_LINE)
GRAY = T.hexc(T.PANEL_LINE)            # de-emphasis fill, readable on the dark surface
SURFACE = T.hexc(T.BG_SLIDE)
SANS = T.FIG_FONT_STACK

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": SANS,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.edgecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": SECOND, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": SECOND,
    "xtick.labelcolor": SECOND, "ytick.labelcolor": SECOND,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.dpi": 220,
})


def _finish(ax):
    """Recessive chrome: hairline axes, no grid, no top/right spines."""
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=0, pad=6)


# ── figure 1 · the gap (emphasis) ───────────────────────────────────────────────
def fig_gap(rows):
    """Every guard we measured, on one axis, with the hosted model emphasised.

    Purpose-built guards are included because "why not just run Llama Guard?" is the first
    question asked, and the answer belongs in the same picture rather than a footnote. A
    guard whose score column collapsed is skipped outright -- plotting a constant as a bar
    would present a harness artifact as a model result.
    """
    both = {**FR.PRETTY, **FR.SCALE_PRETTY, **FR.GUARD_PRETTY}
    items = []
    for grp, suffix in (("base", ""), ("scale", ""), ("sft", " tuned"),
                        ("scale_sft", " tuned")):
        for k, m in rows[grp].items():
            items.append((both[k] + suffix, m["tpr"], "open"))
    for k, m in rows.get("guard", {}).items():
        if not m.get("degenerate"):
            items.append((both[k], m["tpr"], "guard"))
    best_hosted = max(rows["frontier"], key=lambda k: rows["frontier"][k]["tpr"])
    items.append((FR.FRONTIER_PRETTY[best_hosted].replace(" (low)", ""),
                  rows["frontier"][best_hosted]["tpr"], "hosted"))
    items.sort(key=lambda t: t[1])
    best_open = max((t for t in items if t[2] in ("open", "guard")),
                    key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    for i, (name, val, kind) in enumerate(items):
        if kind == "hosted":
            c, tc, w = ACCENT, ACCENT, "bold"
        elif name == best_open[0]:
            c, tc, w = GREEN, GREEN, "bold"
        elif kind == "guard":
            c, tc, w = BLUE, BLUE, "normal"     # released purpose-built guards
        else:
            c, tc, w = GRAY, SECOND, "normal"
        ax.barh(i, val, height=0.62, color=c, zorder=3)
        ax.text(val + 0.012, i, f"{val*100:.0f}%", va="center", ha="left",
                fontsize=11, color=tc, fontweight=w)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(
        [n for n, _, _ in items],
        fontsize=10.5,
        color=INK)
    for lbl, (n, _v, kind) in zip(ax.get_yticklabels(), items):
        if kind == "hosted":
            lbl.set_color(ACCENT)
            lbl.set_fontweight("bold")
        elif n == best_open[0]:
            lbl.set_color(GREEN)
            lbl.set_fontweight("bold")
        elif kind == "guard":
            lbl.set_color(BLUE)
        else:
            lbl.set_color(SECOND)
    ax.set_xlim(0, 1.02)
    ax.set_xticks([])
    ax.set_xlabel("Share of unsafe prompts caught, at a matched 5% false-alarm rate",
                  fontsize=10, color=SECOND, labelpad=10)
    # Three colours now carry family identity, so a legend is mandatory -- the colour-coded
    # row labels are secondary encoding, not a substitute for saying what blue means.
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=ACCENT, label="hosted frontier"),
        Patch(facecolor=BLUE, label="released purpose-built guard"),
        Patch(facecolor=GRAY, label="general open-weights model (ours)"),
    ], loc="lower right", frameon=False, fontsize=10,
        labelcolor=[ACCENT, BLUE, SECOND], handlelength=1.1, handleheight=1.1,
        borderpad=0.2, labelspacing=0.45, bbox_to_anchor=(1.0, -0.015))
    _finish(ax)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    out = ASSETS / "exec_gap.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out


# ── figure 2 · the specialization tax (signed, one axis) ────────────────────────
def fig_tax(rows):
    """Δ represented and Δ transfer per checkpoint, ordered by base strength."""
    reg = FR.compute_scale_regimes()
    summary = json.loads((FR.KLSFT_SUMMARY).read_text())
    pts = []
    for d in summary:
        pts.append((d["pretty"], d["base_represented"],
                    d["sft_committed_represented"] - d["base_represented"],
                    d["sft_committed_transfer"] - d["base_transfer"]))
    for k, s in reg["ladder_sft"].items():
        b = reg["ladder"][k]
        pts.append((FR.SCALE_PRETTY[k], b["represented"],
                    s["represented"] - b["represented"],
                    s["transfer"] - b["transfer"]))
    # Descending by base strength: barh puts index 0 at the BOTTOM, so sorting
    # strongest-first makes the weakest base the top row. Reading top-down then follows
    # the sentence the chart is making -- as the base gets stronger, the gain shrinks and
    # the held-out effect turns from a gain into a loss. Sorting the other way tells the
    # story backwards. Note the LOSS is not monotone in base strength (Qwen3-4B gives back
    # more than either larger model), which is why neither deck claims that it is.
    pts.sort(key=lambda t: -t[1])

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    h = 0.32
    for i, (_name, _bs, drep, dtr) in enumerate(pts):
        ax.barh(i + h / 2 + 0.02, drep, height=h, color=BLUE, zorder=3)
        ax.barh(i - h / 2 - 0.02, dtr, height=h, color=ACCENT, zorder=3)
        ax.text(drep + (0.012 if drep >= 0 else -0.012), i + h / 2 + 0.02,
                f"{drep:+.3f}", va="center", ha="left" if drep >= 0 else "right",
                fontsize=9.5, color=BLUE)
        ax.text(dtr + (0.012 if dtr >= 0 else -0.012), i - h / 2 - 0.02,
                f"{dtr:+.3f}", va="center", ha="left" if dtr >= 0 else "right",
                fontsize=9.5, color=ACCENT)
    ax.axvline(0, color=INK, linewidth=1.0, zorder=4)
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels([n for n, _, _, _ in pts], fontsize=10.5, color=SECOND)
    ax.set_ylim(-0.75, len(pts) - 0.25)
    ax.set_xlim(-0.32, 0.70)
    ax.set_xticks([])
    ax.set_xlabel("Change in accuracy after fine-tuning  (AP points)\n"
                  "rows ordered weakest base (top) to strongest base (bottom)",
                  fontsize=10, color=SECOND, labelpad=10)
    # A real legend, not spatial labels: colour carries which measure, the side of the
    # zero rule carries the sign. Spatial labels broke on SmolLM2, whose unfamiliar-traffic
    # bar is positive and therefore sits on the right. The ordering cue lives in the axis
    # label rather than as annotations beside the rows, which collided with the row names.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=BLUE, label="familiar traffic  (trained on)"),
                       Patch(facecolor=ACCENT, label="unfamiliar traffic  (held out)")],
              loc="lower right", frameon=False, fontsize=10.5,
              labelcolor=[BLUE, ACCENT], handlelength=1.1, handleheight=1.1,
              borderpad=0.2, labelspacing=0.5)
    _finish(ax)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = ASSETS / "exec_tax.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out


# ── figure 3 · the ladder against a fixed reference ─────────────────────────────
def fig_scale(rows):
    """A dot plot, not columns.

    The whole story lives between 75% and 90%, and a zero-baseline column chart squeezes
    that range into the top sixth of the panel -- the differences that matter become
    invisible. Truncating a *bar* axis would be dishonest (bar length encodes magnitude
    from zero), but a dot encodes position only, so a zoomed axis is legitimate here and
    lets the two intervals that carry the argument be read directly.
    """
    fam = [("Qwen3-4B", 4, rows["base"]["qwen3_4b"]["tpr"]),
           ("Qwen3-8B", 8, rows["scale"]["qwen3_8b"]["tpr"]),
           ("Qwen3-32B", 32, rows["scale"]["qwen3_32b"]["tpr"])]
    best_hosted = max(rows["frontier"], key=lambda k: rows["frontier"][k]["tpr"])
    hosted = rows["frontier"][best_hosted]["tpr"]
    items = [(n, v, "open") for n, _p, v in fam]   # the size is already in the name
    items.append(("hosted frontier", hosted, "hosted"))

    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    lo, hi = 0.70, 0.95
    for i, (name, v, kind) in enumerate(items):
        y = len(items) - 1 - i                      # top-down in listed order
        ax.plot([lo, hi], [y, y], color=GRID, lw=0.9, zorder=1)
        if kind == "hosted":
            col, w = ACCENT, "bold"
        elif name.startswith("Qwen3-32B"):
            col, w = GREEN, "bold"
        else:
            col, w = GRAY, "normal"
        ax.plot([v], [y], "o", ms=13, color=col, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(v + 0.008, y + 0.30, f"{v*100:.0f}%", fontsize=12, color=col,
                fontweight=w, ha="left", va="center")
        ax.text(lo - 0.008, y, name, fontsize=11, ha="right", va="center",
                color=col if kind == "hosted" or w == "bold" else SECOND,
                fontweight=w)

    # The two intervals that carry the argument, drawn under the dots.
    ax.annotate("", xy=(fam[-1][2], -0.62), xytext=(fam[0][2], -0.62),
                arrowprops=dict(arrowstyle="|-|,widthA=0.35,widthB=0.35",
                                color=GREEN, lw=1.2))
    ax.text((fam[0][2] + fam[-1][2]) / 2, -0.86,
            f"8× parameters  →  +{(fam[-1][2]-fam[0][2])*100:.0f} pts",
            fontsize=10.5, color=GREEN, ha="center", va="top", fontweight="bold")
    ax.annotate("", xy=(hosted, -0.62), xytext=(fam[-1][2], -0.62),
                arrowprops=dict(arrowstyle="|-|,widthA=0.35,widthB=0.35",
                                color=ACCENT, lw=1.2))
    ax.text((hosted + fam[-1][2]) / 2, -0.86,
            f"still {(hosted-fam[-1][2])*100:.0f} pts short",
            fontsize=10.5, color=ACCENT, ha="center", va="top", fontweight="bold")

    ax.set_xlim(lo - 0.075, hi)
    ax.set_ylim(-1.55, len(items) - 0.45)
    ax.set_yticks([])
    ax.set_xticks([0.75, 0.80, 0.85, 0.90])
    ax.set_xticklabels(["75%", "80%", "85%", "90%"], fontsize=10, color=MUTED)
    ax.set_xlabel("Share of unsafe prompts caught, at a matched 5% false-alarm rate",
                  fontsize=10, color=SECOND, labelpad=8)
    _finish(ax)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = ASSETS / "exec_scale.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out


# ── figure 4 · the cascade: you do not have to choose ───────────────────────────
INHOUSE_KEY = "smollm3_3b"        # 3B, ~20 ms self-hosted: the realistic inline choice
INHOUSE_NAME = "SmolLM3-3B"


def cascade_curve(local_key: str = INHOUSE_KEY, budget: float = FR.FPR_BUDGET):
    """Recall at a single global false-alarm budget as a function of escalated share.

    The inline in-house guard scores every request. The least-confident slice -- the rows
    nearest its own decision threshold -- is re-scored by the hosted model, and ONE global
    threshold is then set on the fused ranking so the alarm budget is held fixed across the
    whole comparison. Ranks rather than raw margins because the two guards' score scales are
    unrelated (a logit margin against a self-reported 0-100 risk); fusing raw values would
    let one scale dominate for reasons that have nothing to do with accuracy.

    A per-band threshold was tried first and rejected: with a small deferred slice there are
    too few deferred negatives to place a stable quantile, which made the curve non-monotone
    (the 32B row fell between 0% and 10%) for purely numerical reasons.
    """
    import numpy as np
    from scipy.stats import rankdata
    labels = FR.load_labels()
    loc = FR._read(f"scores_{local_key}_base.json")
    hosted = FR._read("scores_gpt54_low.json")
    ids = [i for i in labels if i in loc and i in hosted]
    sl = np.asarray([loc[i] for i in ids], float)
    sh = np.asarray([hosted[i] for i in ids], float)
    y = np.asarray([labels[i]["label"] for i in ids], int)
    n = len(ids)
    rl, rh = rankdata(sl) / n, rankdata(sh) / n
    thr_r = np.quantile(rl[y == 0], 1 - budget, method="higher")
    order = np.argsort(np.abs(rl - thr_r))

    def recall(frac):
        fused = rl.copy()
        k = int(round(frac * n))
        if k:
            fused[order[:k]] = rh[order[:k]]
        t = np.quantile(fused[y == 0], 1 - budget, method="higher")
        return float((fused[y == 1] > t).mean())

    fracs = [i / 100 for i in range(0, 101, 5)]
    return fracs, [recall(f) for f in fracs]


def fig_cascade(rows):
    fracs, rec = cascade_curve()
    floor, ceil = rec[0], rec[-1]
    # Plot the decision-relevant range. Past ~50% you have outsourced the majority of
    # traffic, so the choice is no longer "in-house with escalation"; the ceiling line
    # carries the full-escalation endpoint, so nothing is hidden by stopping here.
    XMAX = 50
    keep = [(f, r) for f, r in zip(fracs, rec) if f * 100 <= XMAX]
    xs = [f * 100 for f, _ in keep]
    ys = [r for _, r in keep]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axhline(ceil, color=ACCENT, lw=1.5, zorder=2)
    ax.axhline(floor, color=GREEN, lw=1.5, zorder=2)
    ax.fill_between(xs, floor, ys, color=BLUE, alpha=0.10, zorder=1)
    ax.plot(xs, ys, color=BLUE, lw=2.6, zorder=4)
    # Labels sit ABOVE their rule, left-aligned inside the plot -- centring them on the line
    # struck the text through.
    ax.text(0.6, ceil + 0.004, f"send every request to the hosted model — {ceil*100:.0f}%",
            fontsize=10.5, color=ACCENT, va="bottom", fontweight="bold")
    # Right-hand side: near x=0 the curve leaves the green rule and would cross this label.
    ax.text(XMAX - 0.6, floor + 0.004,
            f"keep every request in-house — {floor*100:.0f}%",
            fontsize=10.5, color=GREEN, va="bottom", ha="right", fontweight="bold")
    # Annotations go BELOW-right, into the shaded area, which is empty. Placing them above
    # the point puts them in the path of the rising curve, which strikes the text through.
    for f in (0.10, 0.20, 0.30):
        i = fracs.index(f)
        share = (rec[i] - floor) / (ceil - floor)
        ax.plot([f * 100], [rec[i]], "o", ms=11, color=BLUE, zorder=5,
                markeredgecolor=SURFACE, markeredgewidth=2.2)
        ax.annotate(f"{rec[i]*100:.0f}%\n{share*100:.0f}% of the gap closed",
                    xy=(f * 100, rec[i]), xytext=(f * 100 + 1.4, rec[i] - 0.008),
                    fontsize=10.5, color=BLUE, va="top", fontweight="bold",
                    linespacing=1.35)
    ax.set_xlim(-0.6, XMAX + 1)
    ax.set_ylim(floor - 0.012, ceil + 0.020)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%", "50%"], fontsize=11, color=MUTED)
    ax.set_yticks([])
    ax.set_xlabel(f"Share of requests escalated to the hosted model\n"
                  f"inline guard {INHOUSE_NAME} (~20 ms, self-hosted) · recall held at a "
                  f"fixed 5% false-alarm budget throughout",
                  fontsize=10, color=SECOND, labelpad=10)
    _finish(ax)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    out = ASSETS / "exec_cascade.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out


def main() -> int:
    print(T.figure_font_report())   # see make_slide_figures: a silent fallback is a defect
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = FR.compute()["rows"]
    for fn in (fig_gap, fig_tax, fig_scale, fig_cascade):
        p = fn(rows)
        print(f"  wrote {p.relative_to(HERE)}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
