# Presentation deck

Two decks are built from code, and only these two are current:

- `safety_guard_benchmark_deck.pptx` — 21 slides, 16:9, speaker notes on every slide.
  The research talk.
- `safety_guard_exec_deck.pptx` — 13 slides, 16:9. The guardrail-sourcing decision,
  for a non-research audience.

Both track the report *Safety Benchmark Gains Do Not Guarantee Safety Transfer: A Comprehensive Study of Fine-Tuning Small Language Model Safety Guards for High-Compliance and General Safety Domains*. Earlier hand-edited
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
fallback stack, and both figure scripts print `deck_theme.figure_font_report()` before they
render, so a fallback is visible rather than silent. (An earlier version of this README
promised that report while nothing called it; the safeguard existed only on paper.)

Read that report: Calibri ships with Office, so most build boxes lack it. The committed
figures were rendered with **Arial**, the third entry in the stack, while the `.pptx` chrome
still asks for Calibri and resolves it on the viewer's machine. That is a deliberate
metric-compatible fallback, not a defect — but it means the figures and the slide text are
not guaranteed to be the same face.

## Structure

| # | Slide | Source in the report |
| --- | --- | --- |
| 1 | Title | — |
| 2 | The problem — a guard's score is not a property of the guard | Act I per-benchmark + operating-point table; worked G0/D1 case-study figure |
| 3 | One figure — four questions, one thesis | the four-panel front figure |
| 4 | How we measure — compare each guard to its own base | Background §"Estimands, the fixed panel, and the paired hierarchical bootstrap"; study-at-a-glance figure |
| 5 | Act I — a large represented gain that does not transfer | Act I fixed-panel result table |
| 6 | Act I — 15 of 20 guards specialize | per-seed value table; specialization plane |
| 7 | Act I — the operating point, under both threshold rules | Act I per-benchmark/operating-point table; matched-false-alarm-budget table; low-FPR re-reading table |
| 8 | Act I — the deployment base rate | prevalence curve; the AP(π₊) equation |
| 9 | Act I — KL is a tradeoff dial | anti-forgetting (KL-SFT) control table |
| 10 | Preregistered — released guards specialize too | adaptation movement-vector table; adaptation plane |
| 11 | Act II — repair without retraining | composition fixed-panel summary table |
| 12 | Act II — it is the base, not ensembling | per-checkpoint composition table; SFT+SFT equal-cost control |
| 13 | Act III — general safety ≠ domain compliance | mortgage zero-shot baseline table; fairness-gate figure |
| 14 | Act III — one row, end to end | worked G0/D1 case-study figure |
| 15 | Act III — two results we did not want | mortgage baseline and ExpGuard tables |
| 16 | Why self-host | guard-latency and deployment-economics tables |
| 17 | External reference point — the hosted frontier model | frontier-vs-local table |
| 18 | Two routes that do not work: tuning and scale | scale-versus-tuning table; gap-ladder figure |
| 19 | The gap is a regime, not a size | represented-vs-transfer head-to-head table; regime map |
| 20 | What to do — gate candidates, not leaderboards | guidelines table; gating workflow figure |
| 21 | What this contributes, and what would make it evidence | Conclusion |

Sources are named by **content, not by rendered float number**. Rendered numbers move whenever a
float is inserted — adding the claim ledger as Table 1 shifted every later table by one, and this
column then sat stale in 8 of its 21 rows while claiming to be current. The speaker notes follow the
same rule. If you do want the numbers for a particular build:

```bash
pdftotext ../unified_report.pdf - | grep -oE '^(Table|Figure) [0-9]+:.{0,60}'
```

## Scope discipline

The deck carries the report's evidence tiers rather than smoothing them: Acts I–II are
labelled retrospective, the adaptation study is labelled preregistered, ExpGuard is
labelled the one expert-annotated tier, and the mortgage labels are labelled LLM-judge
and not counsel-reviewed. Slide 15 restates the scope boundary in full. Speaker notes
carry the caveats that do not fit on the slides — read them before presenting.

Six limits are load-bearing enough that the deck states them on the slide, not only in
the notes. Slide 7 shows the operating point under **both** threshold rules, because a
recall comparison at unequal false-alarm rates is not a comparison of discriminative
power — at an equal budget Act I's apparent transfer-recall gain reverses on all four
checkpoints. Slides 7, 9 and 20 also carry the FPR $[0,.05]$ re-read of that trade
(`lowfpr_macros.tex`, `lowfpr_kl_macros.tex`): macro-AP understates **both** halves, by
2.1× on the represented gain and 3.0× on the transfer cost, and understates the KL dial
asymmetrically (2.4× on what it buys, 6.2× on what it charges). Slide 3 carries the
qualifier under its four panels: the domain-arm CIs
overlap and Q4's left bar is post hoc, so the claim is that the leaderboard's answer
moves, not that the ordering is resolved. Slide 19 and exec slide 5 carry the source-set
conditional on that post-hoc aggregate — resampling the three corpora widens $+0.083$ to
an interval that includes zero, which is the honest statement of how far it travels. Slide 15
reports health as *leaning* against a tie rather than resolved: four unadjusted paired
comparisons, the interval clearing zero by +0.0026. And slide 9's notes carry the
≈0.015 mean / 0.029 worst-case reproduction noise floor measured by the KL control's
β = 0 arm, which is what bounds every small effect quoted anywhere in the deck.

Five claims the decks deliberately do **not** make, because the report withdrew or scoped
them. That the hosted–local gap is *conservative* (coarse integer ties bound how finely the
hosted ranking resolves, but they do not fix a direction). That closing the gap by scale alone
would take another order of magnitude (three points, one non-monotonic, identify no scaling
law). That the specialization tax *scales with the base* — the transfer cost is not monotone in
base strength (Qwen3-4B pays more from a weaker base than either larger model), so slide 18 and
exec slide 6 say the tax is the distance to a benchmark-fixed **endpoint**, which is arithmetic
rather than a behavioural law. That ExpGuard finance and law are *ties* — an interval containing
zero is not evidence of no difference and no equivalence margin was registered, so they are
differences too small to sign. And that SmolLM2's positive transfer delta is a deployment result:
it is +0.040 on macro-AP and straddles zero inside the alarm budget, so slides 5–7 scope it to
average ranking. Earlier builds of the decks asserted the first four.
