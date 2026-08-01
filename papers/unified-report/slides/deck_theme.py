"""The deck design system, shared by every generator.

Single source of truth for both PowerPoint generators (`make_deck.py`, `make_exec_deck.py`)
and both figure generators (`make_slide_figures.py`, `make_exec_figures.py`). The tokens below
were originally read out of two hand-designed decks (`*_redesigned.pptx`) which used an
identical palette and type scale, so they belonged in one module rather than being duplicated
four times and drifting. Those two files have since been deleted: this module IS the redesign
now, and keeping hand-edited copies beside code-generated ones meant shipping decks with a
withdrawn title and pre-correction numbers. Recover them from git history if the tokens ever
need re-deriving.

The design is a dark system. Two properties are load-bearing and easy to break:

  1. Figures are rendered on `FIG_BG`, which is the SAME colour as the content-slide
     background. A figure therefore has no visible panel; it sits directly on the slide. Any
     figure regenerated on white will read as a bright rectangle and break the whole deck.
  2. The palette is deliberately restrained to two accents -- a light blue and a red/salmon --
     against a navy structure. The redesign has no green, purple or gold.

`DATA_*` below preserves the report's five semantic series because several slides need more
than two categories. Only `DATA_REPRESENTED` and `DATA_TRANSFER` are taken directly from the
redesign; the other three are derived in the same key (light-on-dark, desaturated) and are
marked as such, because the redesign does not specify them.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────── surfaces
BG_TITLE = "060E1A"      # title / section-break slides, one step darker
BG_SLIDE = "0C1424"      # every content slide, and every figure facecolor
CARD = "18263D"          # standard card / panel fill
CARD_LINE = "1E3050"     # standard card border, 0.75pt
PANEL_LINE = "26374F"    # neutral divider / secondary outline, 0.75pt
WARN_CARD = "331A1C"     # red-tinted panel: caveats, costs, "what this does not show"
WARN_LINE = "4A2427"     # red-tinted panel border, 0.75pt
DECO_LINE = "15223A"     # decorative concentric rings on the title slide, 1.25pt
DECO_LINE_WARM = "3A2226"

# ─────────────────────────────────────────────────────────────── ink
TEXT = "EEF2F8"          # headings and emphasis
BODY = "9DAFC7"          # body copy
DIM = "6B7F9F"           # sub-headings, captions
FAINT = "7488A8"         # running footer
DATA = "BFD8F5"          # data labels, table headers, pill text
ACCENT = "E0564F"        # kickers, slide numbers, the one solid accent
ACCENT_SOFT = "F08A7F"   # salmon: secondary accent, warning-panel labels

# ─────────────────────────────────────────────── semantic data series
DATA_REPRESENTED = DATA          # from the redesign
DATA_TRANSFER = ACCENT_SOFT      # from the redesign
DATA_COMPOSITION = "8FD3B6"      # derived, not in the redesign
DATA_KL = "C3B0EF"               # derived, not in the redesign
DATA_GOLD = "E3C285"             # derived, not in the redesign

# ─────────────────────────────────────────────────────────────── type
DISPLAY = "Cambria"      # titles and large numerals
UI = "Calibri"           # everything else

# PowerPoint resolves DISPLAY/UI on the viewer's machine, so the two names above are all the
# .pptx needs. Matplotlib is different: figures are rasterised at generation time, so whatever
# font is installed *here* gets baked into the PNG. Calibri and Cambria ship with Office and
# are frequently absent on a build machine, so the stack below degrades deliberately:
#   Carlito is metric-compatible with Calibri (same widths, so layout does not shift);
#   Arial and DejaVu Sans both cover U+2192 (->), which the deck uses in axis labels.
# Helvetica and Helvetica Neue are deliberately EXCLUDED: both are missing U+2192, and
# matplotlib does not fall back per glyph -- it renders a missing-glyph box and warns. Verified
# with fontTools against the installed faces; do not add them back as "closer to Calibri".
# If a figure looks off-brand, check `deck_theme.figure_font_report()` before anything else.
FIG_FONT_STACK = [UI, "Carlito", "Arial", "DejaVu Sans"]

# Glyphs the figures actually use that a candidate fallback must cover.
FIG_REQUIRED_GLYPHS = [0x2192, 0x2191, 0x2193, 0x0394, 0x00B7]


def figure_font_report() -> str:
    """Which of FIG_FONT_STACK matplotlib can see, which it will use, and its glyph coverage.

    Worth running whenever figures are regenerated on a new machine: Calibri ships with
    Office, so a build box often lacks it and the figures silently render in a fallback.
    """
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    lines = [f"  {n:14s} {'available' if n in have else 'MISSING'}" for n in FIG_FONT_STACK]
    chosen = next((n for n in FIG_FONT_STACK if n in have), None)
    out = "figure font stack:\n" + "\n".join(lines)
    out += f"\n  -> rendering with: {chosen or 'matplotlib default'}"
    if chosen:
        try:
            from fontTools.ttLib import TTFont
            path = fm.findfont(fm.FontProperties(family=chosen), fallback_to_default=False)
            font = TTFont(path, fontNumber=0, lazy=True)
            cmap = {cp for t in font["cmap"].tables for cp in t.cmap}
            missing = [hex(g) for g in FIG_REQUIRED_GLYPHS if g not in cmap]
            out += ("\n  glyph coverage: OK" if not missing
                    else f"\n  glyph coverage: MISSING {missing} -- figures will show boxes")
        except Exception as exc:  # fontTools is optional; never fail a build over a report
            out += f"\n  glyph coverage: not checked ({type(exc).__name__})"
    if chosen != UI:
        out += (f"\n  NOTE: {UI} is unavailable here, so figures are NOT in the deck's own"
                f" typeface. The .pptx chrome still specifies {UI} and resolves on the viewer.")
    return out

SZ_TITLE_HERO = 38.0     # title slide h1
SZ_TITLE = 25.0          # content-slide title
SZ_STAT = 22.0           # large stat inside a card
SZ_STAT_SM = 19.0
SZ_SUBTITLE = 13.5       # title-slide subtitle
SZ_LEAD = 11.5           # content-slide sub-headline, pill text
SZ_KICKER = 10.0         # uppercase eyebrow
SZ_KICKER_SM = 9.6
SZ_BODY = 10.2           # card body copy
SZ_BODY_SM = 9.8
SZ_LABEL = 9.6           # data labels
SZ_TABLE_HEAD = 8.6
SZ_PAGENUM = 9.5
SZ_FOOTER = 8.6

# ─────────────────────────────────────────────────────────── geometry (inches)
SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.72
CONTENT_W = SLIDE_W - 2 * MARGIN          # 11.893
CARD_INSET = 0.26                          # text inset inside a card
KICKER_Y = 0.36
TITLE_Y = 0.64
LEAD_Y = 1.26
BODY_Y = 1.80                              # first row of content
FOOTER_Y = 7.00
COL3_W = 3.85                              # three-up card width
COL3_PITCH = 3.99                          # left edges 0.72 / 4.71 / 8.70

LW_CARD = 0.75
LW_ACCENT = 1.5
LW_ACCENT_THIN = 1.0
LW_DECO = 1.25

# There is no top accent bar and no rule under the header: the redesign removed both, and the
# kicker's colour carries the separation instead. Re-adding either breaks the look.
HAS_TOP_BAR = False
HAS_HEADER_RULE = False


def rgb(hex_str: str):
    """Hex string -> python-pptx RGBColor, so callers never hand-write byte triples."""
    from pptx.dml.color import RGBColor
    return RGBColor.from_string(hex_str)


def hexc(hex_str: str) -> str:
    """Hex string -> matplotlib colour."""
    return f"#{hex_str}"


def apply_matplotlib_theme(plt) -> None:
    """Put matplotlib on the deck's dark surface.

    Called by both figure generators before any figure is built. `savefig.facecolor` is set
    explicitly because it does NOT inherit from `figure.facecolor`, and a figure saved with a
    white canvas is the single most visible way to break this design.
    """
    plt.rcParams.update({
        "figure.facecolor": hexc(BG_SLIDE),
        "axes.facecolor": hexc(BG_SLIDE),
        "savefig.facecolor": hexc(BG_SLIDE),
        "savefig.edgecolor": hexc(BG_SLIDE),
        "text.color": hexc(TEXT),
        "axes.labelcolor": hexc(BODY),
        "axes.edgecolor": hexc(PANEL_LINE),
        "axes.titlecolor": hexc(TEXT),
        "xtick.color": hexc(BODY),
        "ytick.color": hexc(BODY),
        "xtick.labelcolor": hexc(BODY),
        "ytick.labelcolor": hexc(BODY),
        "grid.color": hexc(CARD_LINE),
        "grid.alpha": 1.0,
        "legend.facecolor": hexc(CARD),
        "legend.edgecolor": hexc(CARD_LINE),
        "legend.labelcolor": hexc(BODY),
        "font.family": "sans-serif",
        "font.sans-serif": [UI, "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


SERIES = {           # convenience for figure code that keys off the report's series names
    "represented": hexc(DATA_REPRESENTED),
    "transfer": hexc(DATA_TRANSFER),
    "composition": hexc(DATA_COMPOSITION),
    "kl": hexc(DATA_KL),
    "gold": hexc(DATA_GOLD),
    "accent": hexc(ACCENT),
    "text": hexc(TEXT),
    "body": hexc(BODY),
    "dim": hexc(DIM),
    "card": hexc(CARD),
    "card_line": hexc(CARD_LINE),
    "bg": hexc(BG_SLIDE),
}


# ------------------------------------------------------------ document properties
# Both generators start from python-pptx's default template, whose docProps say the deck has
# no title, no author, "Steve Canny" as last modifier, and an on-screen 4:3 format. None of
# that is true of these decks, and all of it is visible in PowerPoint's File > Properties and
# to anything that indexes the file. The two helpers below correct it at save time.
#
# Timestamps are deliberately left alone. Writing a fresh `modified` date on every build would
# make the output non-deterministic, and the README's contract is that re-running the
# generators is idempotent.

AUTHOR = "Reza Rahimi"
COMPANY = "JazzX AI"
REPO_URL = "https://github.com/rrahimi-uci/safety-guard-dynamics"


def stamp_properties(prs, title: str, subject: str = "") -> None:
    """Replace the default-template document properties with this deck's own."""
    cp = prs.core_properties
    cp.title = title
    cp.author = AUTHOR
    cp.last_modified_by = AUTHOR
    cp.subject = subject
    cp.category = "Research presentation"
    cp.comments = f"Generated from code; sources and data at {REPO_URL}"
    cp.keywords = ("AI safety; safety guard; LLM guardrail; small language model; "
                   "fine-tuning; specialization; out-of-distribution transfer")


def fix_presentation_format(path) -> None:
    """Rewrite docProps/app.xml's inherited `<PresentationFormat>` to match the slide size.

    python-pptx copies app.xml from its template verbatim and never revises it, so a 16:9
    deck shipped claiming "On-screen Show (4:3)". Rewriting one element means rebuilding the
    zip, which is why this runs after `Presentation.save` rather than through python-pptx.
    """
    import re
    import shutil
    import zipfile
    from pathlib import Path

    path = Path(path)
    part = "docProps/app.xml"
    with zipfile.ZipFile(path) as z:
        if part not in z.namelist():
            return
        items = [(i, z.read(i.filename)) for i in z.infolist()]

    def patch(data: bytes) -> bytes:
        text = data.decode("utf-8")
        new = re.sub(r"<PresentationFormat>[^<]*</PresentationFormat>",
                     "<PresentationFormat>Widescreen</PresentationFormat>", text)
        return new.encode("utf-8")

    tmp = path.with_suffix(".tmp.pptx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for info, data in items:
            out.writestr(info, patch(data) if info.filename == part else data)
    shutil.move(str(tmp), str(path))
