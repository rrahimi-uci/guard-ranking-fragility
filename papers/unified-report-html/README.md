# Unified report — HTML edition

A single-file, offline-capable HTML edition of
[the unified report](../unified-report/unified_report.pdf), built for reading on a screen:
sticky section navigation, tables that are wider than the prose column, MathJax formulas,
vector figures, and a light/dark theme that follows the OS.

Open [`index.html`](index.html) in any browser. No server required.

## Build

```bash
python build.py           # regenerate index.html, robots.txt, sitemap.xml + assets/fig/
python build.py --check   # fail if any of the three differs from a fresh build
```

The page carries discoverability metadata the predecessor site got from `jekyll-seo-tag` and
`jekyll-sitemap`: a canonical URL, keywords, Open Graph and Twitter card tags, a
`ScholarlyArticle` JSON-LD block, and a social preview card rendered from the teaser figure
(`assets/fig/og_teaser.png`). This site is a static upload with no Jekyll, so `build.py` emits
those fields and both SEO files directly from one `SITE_URL` constant — a test asserts the
canonical, `og:url`, JSON-LD, `robots.txt` and `sitemap.xml` all agree, because a canonical
that disagrees with where the page lives fails silently. The sitemap carries no `<lastmod>`:
it would have to come from a clock or an mtime, which defeats `--check` for no crawler benefit
on a one-page site.

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

The builder emits the current **table and figure** counts on each run and fails if they disagree
with the PDF; it also reports how many abstract cross-references it resolved. Equations,
cross-references and the bibliography are numbered and resolved but are not cross-checked against
the PDF — an earlier version of this line claimed they were. No hand-maintained count is kept here.

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

## Licence and attribution

This edition publishes the worked G0/D1 case study **in full**, including the two quoted rows of
the frozen `v1_hmda2022` benchmark. That source was licensed **CC BY 4.0** on 2026-07-27, so the
page carries the required notice in its colophon:

> MortgageGuardBench `v1_hmda2022`, Reza Rahimi, PhD (JazzX AI), licensed CC BY 4.0.

The notice is emitted by the build rather than added by hand, and a test fails if the page ever
quotes the rows without it — CC BY 4.0 is permissive but conditional.

Three boundaries survive the licence and travel with the rows: the prompts are synthetic and the
labels are LLM-judge and policy-card-consistent, **not SME-adjudicated**; the release checksums
cover **release bytes only**, not the generator, judge, configuration, or code; and the prompts
solicit violations by design, so reuse should treat them as harmful-content samples.

**Before the licence, this edition withheld those rows.** `redact_restricted_rows()` held back the
prompt text and kept the row id, gold labels, cited policy cards, scores, and ranks. That
machinery is retained and still fails closed — it is now opt-in:

```bash
python build.py --redact-case-study    # withhold the rows again
```

It was kept rather than deleted because the situation recurs with the next source whose licence is
unresolved, and rebuilding it under time pressure is how prompt text ends up published by accident.

## Distribution gate

The page sits inside
[`tests/test_no_unlicensed_publication.py`](../../tests/test_no_unlicensed_publication.py) with
a **declared quotation budget** of 11 restricted-vocabulary hits, re-baselined from 8 when the
redaction was lifted. The budget is kept rather than retired because the probes are mortgage
vocabulary and one mortgage source is still closed — `mortgage_guard_bench_2k_v0_1_0` remains
`local_only` — so it still catches 2K draft text reaching a published page. **The build fails if
the count grows.** For
calibration: a restricted benchmark row carries ≈2.7 hits, so the withdrawn 2,000-row export
carried on the order of 5,400.

[`PUBLICATION_REQUIREMENTS.json`](PUBLICATION_REQUIREMENTS.json) declares the one source this
page depends on and records all three states it has passed through: refused while unresolved,
authorized on an empty dependency set while the rows were withheld, and now authorized on the
licence itself. `make pages-authorized` exits 0 and names the source, and
[a fixture](../../tests/fixtures/pages_artifact_unapproved/) naming a still-closed source keeps
the refusal path under test.

Do not enable Pages through GitHub's Settings UI instead: "Deploy from a branch" bypasses the
gate and would serve the un-redacted sources.

## Scope

Same scope and same caveats as the PDF: Acts I–II are retrospective estimation on a fixed
four-checkpoint panel, the adaptation study is the one analysis-preregistered piece, ExpGuard is
the one expert-annotated tier, and the mortgage labels are LLM-judge, not SME-adjudicated. This
edition changes the typography and **withholds nothing**: the redaction path exists
(`build.py --redact-case-study`) but has been off since v1_hmda2022 was licensed CC BY 4.0 on
2026-07-27, so the worked case study is published here in full, exactly as in the PDF. It changes
no number, interval, or verdict. (An earlier version of this section still said one row was
withheld, contradicting the Licence section above and the page itself.)
