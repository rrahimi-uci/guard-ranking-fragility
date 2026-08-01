"""Read the frontier / scale-ladder figures straight out of the generated LaTeX macros.

`slides/README.md` states the deck's rule: a number in a slide must not be able to drift
from the number in the paper. The existing deck honours that by transcription plus review.
This module makes it structural for the frontier material instead: both decks read
`generated/frontier_macros.tex` -- the same file `unified_report.tex` `\\input`s -- so if an
analysis is rerun and a figure moves, the slides move with it or the build fails loudly.

Returns strings, deliberately, in exactly the form the paper prints them (".896", "+0.066",
"[+0.043, +0.089]"). Reformatting here would reintroduce the drift this module exists to
prevent.
"""
from __future__ import annotations

import re
from pathlib import Path

GENERATED = Path(__file__).resolve().parents[1] / "generated"
MACROS = GENERATED / "frontier_macros.tex"
H2H_MACROS = GENERATED / "h2h_macros.tex"
CASCADE_MACROS = GENERATED / "cascade_macros.tex"

ADAPTATION_MACROS = GENERATED / "adaptation_macros.tex"

_MACRO = re.compile(r"\\newcommand\{\\Frontier([A-Za-z]+)\}\{(.*)\}\s*$")
_H2H_MACRO = re.compile(r"\\newcommand\{\\Htwo([A-Za-z]+)\}\{(.*)\}\s*$")
_CASCADE_MACRO = re.compile(r"\\newcommand\{\\Cascade([A-Za-z]+)\}\{(.*)\}\s*$")
_ADA_MACRO = re.compile(r"\\newcommand\{\\Ada([A-Za-z]+)\}\{(.*)\}\s*$")


class MissingFigure(KeyError):
    """Raised when a slide asks for a figure the analysis did not emit."""


class Figures(dict):
    def __missing__(self, key):
        raise MissingFigure(
            f"\\Frontier{key} is not in {MACROS.name}. Either the analysis has not been "
            f"rerun (papers/unified-report/reproduce.py) or the slide is asking for a "
            f"figure that no longer exists. Available: {', '.join(sorted(self))}"
        )


def load() -> Figures:
    if not MACROS.is_file():
        raise FileNotFoundError(
            f"{MACROS} missing. Run reproduce.py to emit the frontier artifacts first."
        )
    out = Figures()
    for line in MACROS.read_text().splitlines():
        m = _MACRO.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    if not out:
        raise ValueError(f"{MACROS} parsed to zero macros; the emitter format changed")
    return out


def load_h2h() -> Figures:
    """The represented-vs-transfer head-to-head figures, same anti-drift contract as load().

    Reads generated/h2h_macros.tex -- the file unified_report.tex also inputs -- so a slide
    quoting the regime result cannot drift from Table 17. Values come back in the paper's own
    printed form (".948", "+0.185", "$[+0.065, +0.423]$").
    """
    if not H2H_MACROS.is_file():
        raise FileNotFoundError(
            f"{H2H_MACROS} missing. Run experiments/emit_frontier_general_h2h_tex.py first."
        )
    out = Figures()
    for line in H2H_MACROS.read_text().splitlines():
        m = _H2H_MACRO.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    if not out:
        raise ValueError(f"{H2H_MACROS} parsed to zero macros; the emitter format changed")
    return out


def load_cascade() -> Figures:
    """The selective-cascade curve, read from generated/cascade_macros.tex.

    A slide quoting an escalation point must not hardcode it: the exec deck previously printed
    87% for the 20% point when the artifact says .842, and used that inflated figure to claim
    escalation beats the 18-model blend (.850), which reverses the actual ordering.
    """
    if not CASCADE_MACROS.is_file():
        raise FileNotFoundError(
            f"{CASCADE_MACROS} missing. Run experiments/emit_cascade_tex.py first.")
    out = Figures()
    for line in CASCADE_MACROS.read_text().splitlines():
        m = _CASCADE_MACRO.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    if not out:
        raise ValueError(f"{CASCADE_MACROS} parsed to zero macros; the emitter format changed")
    return out


def load_adaptation() -> Figures:
    """The analysis-preregistered adaptation figures, from generated/adaptation_macros.tex.

    These were the last hand-typed numbers in either deck, and they went stale the moment the
    analyzer was repaired to compute the REGISTERED purpose-built-panel estimand instead of a
    six-family mixed panel: the slide said +0.174 / LCB +0.129 for a quantity that had changed.
    Reading them here makes that class of drift impossible.
    """
    if not ADAPTATION_MACROS.is_file():
        raise FileNotFoundError(
            f"{ADAPTATION_MACROS} missing. Run experiments/emit_adaptation_tex.py first.")
    out = Figures()
    for line in ADAPTATION_MACROS.read_text().splitlines():
        m = _ADA_MACRO.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    if not out:
        raise ValueError(f"{ADAPTATION_MACROS} parsed to zero macros; the emitter format changed")
    return out


def load_named(fname: str) -> Figures:
    """Any generated/*.tex macro file -> {macro name without its prefix stripped: body}.

    load()/load_h2h()/load_cascade()/load_adaptation() each strip a known prefix, which is
    convenient at the call site but only works for files whose macros share one. This returns the
    full macro name, for files like results_macros_gen.tex and pilot_macros.tex whose names do not.
    """
    path = GENERATED / fname
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing. Run papers/unified-report/reproduce.py first.")
    out = Figures()
    for line in path.read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    if not out:
        raise ValueError(f"{path} parsed to zero macros; the emitter format changed")
    return out


def bare(value: str) -> str:
    r"""Strip the LaTeX the macros carry for the paper, so a slide can print the value.

    Handles the three forms these macros actually use: math mode ($+0.185$), the \code{...}
    wrapper around identifiers, and the escaped underscore inside them. Order matters --
    unwrap \code{} before removing braces, or the command name survives and the braces do not.
    """
    v = value.strip()
    v = re.sub(r"\\code\{([^}]*)\}", r"\1", v)   # \code{prompt\_injections} -> prompt\_injections
    v = v.replace("\\_", "_").replace("\\%", "%").replace("$", "")
    return v.replace("{", "").replace("}", "").strip()


def signed(value: str) -> str:
    """Normalise a leading ASCII hyphen to a real minus sign (U+2212).

    The macro files write `$-0.034$`; the slide prose around it is hand-typed with U+2212. In
    Arial the two glyphs differ visibly in width, so one sentence carrying both looks like a
    typo. Only a leading sign is touched: `gpt-5.4` and `Qwen2.5-1.5B` must keep their hyphens.
    """
    v = value.strip()
    return "\u2212" + v[1:] if v[:1] == "-" and v[1:2].isdigit() else v


def pct(value: str) -> str:
    """'.896' -> '90%'. For executive slides, where two decimals are noise.

    Rounds half away from zero rather than banker's rounding, because a reader comparing
    the slide to the paper should see the digit they would get by hand.
    """
    from decimal import ROUND_HALF_UP, Decimal
    v = Decimal(value.strip().lstrip("+"))
    if v < 1:
        v = v * 100
    return f"{v.quantize(Decimal('1'), rounding=ROUND_HALF_UP)}%"


def points(value: str) -> str:
    """'+0.066' -> '+7 pts' (percentage points, rounded)."""
    from decimal import ROUND_HALF_UP, Decimal
    v = Decimal(value.strip()) * 100
    q = v.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return f"{'+' if q > 0 else ''}{q} pts"


if __name__ == "__main__":
    f = load()
    for k in sorted(f):
        print(f"  \\Frontier{k:20s} {f[k]}")
    print(f"\n{len(f)} figures available")
