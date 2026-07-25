# Presentation deck

`safety_guard_benchmark_deck.pptx` — 17 slides, 16:9, speaker notes on every slide.

Built for conference talks and internal briefings from the same committed artifacts the
report itself `\input`s, so a number in the deck cannot drift from a number in the paper.

## Build

```bash
cd papers/unified-report
python slides/make_slide_figures.py   # generated/*.tex  ->  slides/assets/*.png
python slides/make_deck.py            # assets + text    ->  the .pptx
```

Requires `python-pptx`, `matplotlib`, `pillow`, and — for the prevalence curve only —
`pandas` + `pyarrow`, which read `artifacts/paper_a_sft_v2/scores/scores.parquet`. If
that file is absent the prevalence panel is skipped with a printed notice rather than
failing the build. Both scripts are idempotent; re-running overwrites in place.

## Why the figures are regenerated rather than lifted from the PDF

The paper figures are typeset for a 10pt document — their tick labels are unreadable on
a projector. `make_slide_figures.py` re-renders each panel at deck scale from the same
parsed `generated/*.tex` tables: larger type, fewer ticks, values annotated on the marks,
one shared palette. It parses the tables rather than restating them, so if an analysis is
rerun and a table changes, the slide figure changes with it.

## Fonts

Georgia (headings) + Arial (body). Both ship with Office on macOS and Windows, so the
deck renders identically off this machine — no embedded-font surprises at a venue.

Georgia's old-style figures are used deliberately for standalone statistics (`+0.3234`),
but Arial is forced wherever digits sit next to letters: in Georgia, `G0 / D0` reads as
`Go / Do`.

## Structure

| # | Slide | Source |
| --- | --- | --- |
| 1 | Title | — |
| 2 | The problem — three failure modes | Table 3 |
| 3 | The whole study in one figure | Figure 1 |
| 4 | Method: the paired estimand | §2, Figure 2 |
| 5 | Act I — represented gain, no transfer | Table 1 |
| 6 | Act I — the specialization plane | Table 2, Figure 4 |
| 7 | Act I — the deployable operating point | Table 3 |
| 8 | Act I — the deployment base rate | Figure 5, Eq. 4 |
| 9 | Act I — KL-SFT is a dial | Table 4 |
| 10 | Preregistered adaptation study | Table 5, Figure 6 |
| 11 | Act II — composition recovers transfer | Table 7 |
| 12 | Act II — it is the base, not ensembling | Tables 8, 9 |
| 13 | Act III — the dual-label design | Table 10, Figure 9 |
| 14 | Act III — one row, end to end | Figure 8 |
| 15 | Act III — two negatives | Tables 11, 12 |
| 16 | Deployment economics: why self-host | Tables 14, 15 |
| 17 | The decision guide | Table 13, Figure 12 |

## Scope discipline

The deck carries the report's evidence tiers rather than smoothing them: Acts I–II are
labelled retrospective, the adaptation study is labelled preregistered, ExpGuard is
labelled the one expert-annotated tier, and the mortgage labels are labelled LLM-judge
and not counsel-reviewed. Slide 15 restates the scope boundary in full. Speaker notes
carry the caveats that do not fit on the slides — read them before presenting.
