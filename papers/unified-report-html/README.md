# Unified report — HTML edition

A single-file, offline-capable HTML edition of
[the unified report](../unified-report/unified_report.pdf), built for reading on a screen:
sticky section navigation, tables that are wider than the prose column, MathJax formulas,
vector figures, and a light/dark theme that follows the OS.

Open [`index.html`](index.html) in any browser. No server required.

## Build

```bash
python build.py           # regenerate index.html + assets/fig/
python build.py --check   # fail if index.html differs from a fresh build
```

Requires `pandoc`, `pdftocairo` (poppler), and `beautifulsoup4`. MathJax loads from a CDN at
view time; everything else is local, so the page renders offline apart from formula typesetting.

## Why generated rather than hand-written

The same rule the rest of this repository follows: **no claim-bearing number is retyped.**
`build.py` reads the identical LaTeX sources and the identical committed
[`generated/*.tex`](../unified-report/generated/) artifacts that `unified_report.tex` itself
`\input`s. Rerun an analysis, and this edition changes with it on the next build. There is no
second copy of any figure to keep in sync.

The build also **asserts its float numbering against the built PDF**: it extracts every
`Table N:` and `Figure N:` caption from `unified_report.pdf` and fails if the counts disagree,
so `Table 4` in the HTML is `Table 4` in the paper. That check caught four tables that pandoc
had silently dropped, and it is the reason the two editions can be cited interchangeably.

Current state: **22 tables, 16 figures, 10 numbered equations, 224 cross-references, 44
references** — all resolving, zero mismatches against the PDF.

## Pipeline

| Stage | What it does |
|---|---|
| `figures()` | PDF figures → SVG via `pdftocairo`; PNGs copied as-is |
| `flatten()` | expands `\input`, neutralizes print-only LaTeX, marks the four tcolorbox callouts, rewrites equations with sentinels |
| `pandoc` | flattened body → HTML fragment, math left for MathJax |
| `postprocess()` | numbers sections/floats/equations in document order, resolves `\Cref`, renders citations from `refs.bib`, builds the callouts and the TOC |
| `verify()` | float numbering vs. the built PDF |

Cross-references and citations survive pandoc as sentinels (`⟦REF:tab:x⟧`) and are resolved
against numbering derived from document order, rather than being hand-maintained.

### Things that needed special handling, and why

- **`@{}` column padding** — `\begin{tabular}{@{}l cc@{}}` and `\multicolumn{5}{@{}l}{…}` make
  pandoc abandon its table reader and emit `<br>`-separated lines. Four tables (adaptation,
  datasets, both ensembling tables) vanished silently until these were stripped.
- **`<embed>` for PDF graphics** — pandoc emits `<embed>`, not `<img>`, for a `.pdf` image, since
  browsers cannot render PDF in an `<img>`. Eleven of fourteen figures were invisible until
  these were rewritten to the SVG conversions.
- **Numeric macros in math mode** — the generated macros wrap values in math (`{$+0.129$}`), and
  the prose also writes `$\AdaHGainLCB>0$`. LaTeX tolerates the nesting; pandoc does not. Bodies
  that are just a signed number are unwrapped; `\KLTakeaway`'s real KL term is left alone.
- **Multi-line `\citep{a,\n b}`** — the sentinel spans a newline, so every scan is `re.S`.
- **`\paragraph`** — pandoc maps it to `h4`. The PDF leaves it unnumbered, so it renders as a
  run-in heading rather than joining the section numbering.
- **The tikz workflow flowchart** (Figure 12) is redrawn as semantic HTML/CSS rather than
  rasterized: it is selectable, accessible, and reflows on a phone.

## What differs from the PDF, deliberately

- **Wide floats break out of the text column.** Prose keeps a 43rem measure; tables and figures
  get up to 58rem. A letter page forces an 8-column table to shrink its type; here it does not.
- **Editorial `edbox` notes are hidden** — they are print-draft annotations.
- **Citations are numbered by first-author alphabetical order** (`plainnat`'s scheme, recomputed
  here from `refs.bib`) rather than lifted from the PDF's `.bbl`. Numbers may differ from the
  PDF's if `natbib`'s compression differs; the linked target is always correct.
- Page-dependent constructs — page breaks at every section, running heads, float placement —
  have no HTML meaning and are dropped.

## Distribution gate

This file is a publishable web page, so it sits inside
[`tests/test_no_unlicensed_publication.py`](../../tests/test_no_unlicensed_publication.py)
with a **declared quotation budget** of 11 restricted-vocabulary hits — the paper's own policy
vocabulary plus one worked G0/D1 row from the frozen `v1_hmda2022` benchmark, whose
redistribution decision is still unresolved in the ledger. The content is identical to the
committed PDF; the budget records the exposure rather than waving it through, and **the build
fails if that count grows**, forcing a fresh review. For calibration: a restricted benchmark row
carries ≈2.7 hits, so the withdrawn 2,000-row export carried on the order of 5,400.

Nothing here authorizes publication. Until a source is approved for `publish_text` in
[`benchmarks/registry/distribution.yaml`](../../benchmarks/registry/distribution.yaml), no
Pages or release build of this page is authorized.

### Publishing this page

A GitHub Pages workflow exists and **refuses to deploy**:

```bash
make pages-authorized    # exit 1 today, naming the source that blocks it
```

[`PUBLICATION_REQUIREMENTS.json`](PUBLICATION_REQUIREMENTS.json) declares the one source this
page needs approved — `mortgage_benchmark_v1_hmda2022`, because the worked G0/D1 case study
quotes a row of it. The other eight restricted sources are *not* required: the report carries
their row hashes and scores, never their text, so publishing it redistributes none of them.

Record the licensing decision in the ledger and the gate opens on its own; see
[the root README](../../README.md#github-pages-wired-tested-and-refusing) for what that
involves. Do not enable Pages through GitHub's Settings UI instead — "Deploy from a branch"
bypasses the gate entirely.

## Scope

Same scope and same caveats as the PDF: Acts I–II are retrospective estimation on a fixed
four-checkpoint panel, the adaptation study is the one analysis-preregistered piece, ExpGuard is
the one expert-annotated tier, and the mortgage labels are LLM-judge, not SME-adjudicated. This
edition changes the typography, not the evidence.
