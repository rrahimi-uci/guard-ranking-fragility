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

_MACRO = re.compile(r"\\newcommand\{\\Frontier([A-Za-z]+)\}\{(.*)\}\s*$")


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
