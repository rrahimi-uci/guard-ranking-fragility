# Presentation deck

Two decks are built from code, and only these two are current:

- `safety_guard_benchmark_deck.pptx` — 21 slides, 16:9, speaker notes on every slide.
  The research talk.
- `safety_guard_exec_deck.pptx` — 13 slides, 16:9. The guardrail-sourcing decision,
  for a non-research audience.

Both track the report *Benchmark Gains Do Not Guarantee Transfer: Fine-Tuning Small Language Model
Safety Guards*. Earlier hand-edited
`_redesigned` and `- Repaired` copies were deleted: the redesign now lives in
`deck_theme.py`, so the generated decks *are* the redesigned decks, and the stale copies
still carried the withdrawn title and pre-correction numbers.

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

The deck requests Cambria for display text and Calibri for body text. Figure generation
uses the first available face in the declared Calibri → Carlito → Arial → DejaVu Sans
fallback stack and prints the selected face, so a fallback is visible rather than silent.

## Structure

| # | Slide | Source |
| --- | --- | --- |
| 1 | Title | — |
| 2 | The problem — a guard's score is not a property of the guard | Table 1 (claim ledger) |
| 3 | One figure — three acts, one thesis | Figure 1 |
| 4 | How we measure — compare each guard to its own base | §2.6, Figure 2 |
| 5 | Act I — a large represented gain that does not transfer | Table 2 |
| 6 | Act I — 15 of 20 guards specialize | Table 3, Figure 4 |
| 7 | Act I — the operating point, under both threshold rules | Tables 4, 5 |
| 8 | Act I — the deployment base rate | Figure 5, Eq. 4 |
| 9 | Act I — KL is a tradeoff dial | Table 6 |
| 10 | Preregistered — released guards specialize too | Table 7, Figure 6 |
| 11 | Act II — repair without retraining | Table 9 |
| 12 | Act II — it is the base, not ensembling | Tables 10, 11 |
| 13 | Act III — general safety ≠ domain compliance | Table 12, Figure 9 |
| 14 | Act III — one row, end to end | Figure 8 |
| 15 | Act III — two results we did not want | Tables 13, 14 |
| 16 | Why self-host | Tables 22, 23 |
| 17 | External reference point — the hosted frontier model | Table 16 |
| 18 | Two routes that do not work: tuning and scale | Table 17, Figure 12 |
| 19 | The gap is a regime, not a size | Table 18, Figure 13 |
| 20 | What to do — gate candidates, not leaderboards | Table 21, Figure 14 |
| 21 | What this contributes, and what would make it evidence | §10 |

Table and figure numbers above are the *rendered* numbers in the current PDF, and they move whenever
a float is inserted — adding the claim ledger as Table 1 shifted every later table by one. The deck
itself cites sources by content rather than by number (only slide 2's speaker notes name a table), so
this column is the thing that goes stale. Re-derive it with:

```bash
pdftotext ../unified_report.pdf - | grep -oE '^(Table|Figure) [0-9]+:.{0,60}'
```

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
