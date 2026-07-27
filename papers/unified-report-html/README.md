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

## What this edition withholds, and why

**This is the published edition, and it is not the whole paper.** The worked G0/D1 case study
in the PDF quotes two rows of the frozen `v1_hmda2022` benchmark — one in full, one in part.
This edition holds that prompt text back and keeps everything that is ours: the row id, the
gold labels, the cited policy cards, the per-guard scores, and the ranks that carry the claim.

The reason is not caution for its own sake. That source is `local_only` with
`permits_redistribution: unknown`; its own
[`DATA_CARD.md`](../../mortgage-benchmark/benchmark/v1_hmda2022/DATA_CARD.md) reads *"LICENSE
NOT YET SELECTED"* and names an FFIEC/CFPB terms-of-use check as a precondition; no reviewer is
on record; and the data card is checksum-frozen. Approving the source in order to publish the
page would have meant writing a licensing conclusion nobody reached into the provenance record.
Removing the dependency was the honest route, so the ledger entry is untouched.

For the full text, read [the PDF](../unified-report/unified_report.pdf) or build locally:

```bash
python build.py --with-restricted-text    # full text; NOT publishable
```

`redact_restricted_rows()` raises `RedactionError` if either quotation anchor stops matching, so
a regenerated case study stops the build rather than quietly publishing the prompt.

## Distribution gate

The page sits inside
[`tests/test_no_unlicensed_publication.py`](../../tests/test_no_unlicensed_publication.py) with
a **declared quotation budget** of 8 restricted-vocabulary hits, down from 11 before the
redaction. What remains is the paper's own policy vocabulary — a background box defining
"underwriting" and "adverse-action notice" — plus prose describing what the withheld row does.
No verbatim run of either row survives; that is checked, not assumed. **The build fails if the
count grows**, which makes the budget the tripwire on a redaction that stops working. For
calibration: a restricted benchmark row carries ≈2.7 hits, so the withdrawn 2,000-row export
carried on the order of 5,400.

[`PUBLICATION_REQUIREMENTS.json`](PUBLICATION_REQUIREMENTS.json) declares that the page now
needs **no** source approved, and records what the requirement used to be and why it went — so
"needs nothing approved" cannot be asserted from a blank slate. `make pages-authorized` exits 0
on that basis, and [a fixture](../../tests/fixtures/pages_artifact_unapproved/) keeps the
refusal path under test.

Do not enable Pages through GitHub's Settings UI instead: "Deploy from a branch" bypasses the
gate and would serve the un-redacted sources.

## Scope

Same scope and same caveats as the PDF: Acts I–II are retrospective estimation on a fixed
four-checkpoint panel, the adaptation study is the one analysis-preregistered piece, ExpGuard is
the one expert-annotated tier, and the mortgage labels are LLM-judge, not SME-adjudicated. This
edition changes the typography and withholds one quoted benchmark row (above); it changes no
number, interval, or verdict.
