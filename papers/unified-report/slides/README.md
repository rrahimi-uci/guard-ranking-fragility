# Presentation deck

`safety_guard_benchmark_deck.pptx` — 18 slides, 16:9, speaker notes on every slide.

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
| 7 | Act I — the operating point, both threshold rules | Tables 3, 4 |
| 8 | Act I — the deployment base rate | Figure 5, Eq. 4 |
| 9 | Act I — KL-SFT is a dial | Table 5 |
| 10 | Preregistered adaptation study | Table 6, Figure 6 |
| 11 | Act II — composition recovers transfer | Table 8 |
| 12 | Act II — it is the base, not ensembling | Tables 9, 10 |
| 13 | Act III — the dual-label design | Table 11, Figure 9 |
| 14 | Act III — one row, end to end | Figure 8 |
| 15 | Act III — two negatives | Tables 12, 13 |
| 16 | Deployment economics: why self-host | Tables 15, 16 |
| 17 | The decision guide | Table 14, Figure 12 |
| 18 | What this contributes, and what would make it evidence | §9 |

Table numbers above are the *rendered* numbers in the current PDF. Adding
`tab:matchedfpr` as Table 4 shifted every later table by one; the deck itself cites
sources by content rather than by number (only slide 2's speaker notes name a table), so
this column is the thing that goes stale when a float is inserted. Re-derive it with
`pdftotext unified_report.pdf - | grep -oE '^Table [0-9]+:.{0,60}'`.

## Scope discipline

The deck carries the report's evidence tiers rather than smoothing them: Acts I–II are
labelled retrospective, the adaptation study is labelled preregistered, ExpGuard is
labelled the one expert-annotated tier, and the mortgage labels are labelled LLM-judge
and not counsel-reviewed. Slide 15 restates the scope boundary in full. Speaker notes
carry the caveats that do not fit on the slides — read them before presenting.

Four limits are load-bearing enough that the deck states them on the slide, not only in
the notes. Slide 7 shows the operating point under **both** threshold rules, because a
recall comparison at unequal false-alarm rates is not a comparison of discriminative
power — at an equal budget Act I's apparent transfer-recall gain reverses on all four
checkpoints. Slide 3's third panel is annotated *"different rows — not a controlled
pair"*, since v1 contains no protected pair on which a violation is scored. Slide 15
reports health as *leaning* against a tie rather than resolved: four unadjusted paired
comparisons, the interval clearing zero by +0.0026. And slide 9's notes carry the
≈0.015 mean / 0.029 worst-case reproduction noise floor measured by the KL control's
β = 0 arm, which is what bounds every small effect quoted anywhere in the deck.
