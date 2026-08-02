# August writing and presentation improvement plan

## Purpose

This plan addresses the **Writing and presentation: 12/15** portion of the review of
`unified_report.pdf`. It does not re-score the science and does not authorize changing a
result merely to make the narrative cleaner. The goal is to make the paper substantially
shorter, easier to navigate, more precise, and visually calmer while preserving the full
audit trail.

The practical target is **14/15 after the August revision**, with **15/15 as a stretch target
after an independent cold read**. A score cannot be guaranteed by editing alone; the exit
criteria below define what must be true before requesting a re-score.

The requested filename is intentionally retained as `improvment-plan-august.md`.

## Scope and evidence boundary

In scope:

- narrative architecture, ordering, and section balance;
- abstract, headings, transitions, paragraph construction, and terminology;
- separation of the main argument from the audit and revision history;
- figure, table, caption, callout, and page-layout design;
- navigation, PDF metadata, and feasible accessibility improvements;
- editorial and rendered-PDF quality assurance.

Out of scope unless separately approved:

- adding experiments, changing estimands, or changing evidence tiers;
- promoting retrospective or development evidence to confirmatory evidence;
- rewriting generated result tables or macros by hand;
- claiming production readiness, legal validity, or full external validity;
- promising PDF/UA conformance without a toolchain that can produce and validate it.

All claim-bearing numbers must continue to come from generated inputs. If a presentation
edit exposes a scientific inconsistency, stop editing that claim, correct and regenerate the
underlying analysis through its source workflow, and only then resume the prose pass.

## Measured baseline on 1 August 2026

These measurements make the plan testable. They describe the current checked-out PDF and
LaTeX sources, not a hypothetical submission version.

| Measure | Current state | August target |
|---|---:|---:|
| Total PDF length | 96 pages | Preserve the full report; add a 30--32-page main-reading edition |
| Pages through the conclusion | 67 pages | 30--32 pages, excluding references and appendices |
| PDF-extracted words | approximately 51,345 | Report both editions; main-reading edition at most 18,000 |
| PDF-extracted words through the conclusion | approximately 36,483 | Main-reading edition at most 18,000 |
| PDF-extracted abstract words | approximately 410 | 180--220 words |
| Targeted revision-history phrases in current main sources | 21 line hits | At most 2, with history moved to one provenance note |
| Background callouts | 13 | At most 3 in the main-reading edition |
| Takeaway callouts | 16 | At most 5 in the main-reading edition |
| Evidence/design/boundary callouts | 4 | At most 4; retain only when they prevent misreading |
| Figures | 17 environments | Each retained main figure must support one indispensable claim |
| Tables | 11 environments | No split rows, missing repeated headers, or unreadably reduced type |
| `\resizebox` uses | 2 | 0 in the main-reading edition |
| Layout-warning lines in the build log | 23 | Inspect all; 0 overfull boxes and 0 oversized-float warnings |
| PDF metadata/tagging | no custom metadata; untagged | Add metadata and bookmarks; treat tagging as a validated stretch goal |

Two known rendered defects are non-negotiable fixes:

1. the build log reports an overfull vertical box of about 55.6 pt;
2. it reports a float about 52.8 pt too large, and the evidence-ledger table visibly breaks
   across pages 89--90 without a clean repeated header or intact final row.

The current design has real strengths that should survive the edit: the restrained maroon
visual identity, the opening synthesis graphic, the evidence/design/boundary distinction,
and the claim ledger. The problem is cumulative density, not the absence of a design system.

### Why the current presentation stops at 12/15

The missing three points are not attributable to one cosmetic defect:

- the main argument runs for 67 pages before the appendices and asks readers to retain too
  many parallel results;
- current results are repeatedly interrupted by revision history and superseded values;
- "Act" labels and research-question labels form two competing navigation systems;
- 33 teaching/evidence callouts across the sources dilute rather than sharpen hierarchy;
- several tables and captions carry prose that should be in the argument or appendix;
- one oversized float, one overfull vertical box, and a broken multi-page evidence ledger are
  visible production defects;
- absent PDF metadata and an untagged output leave navigation/accessibility work unfinished.

The August edit should therefore improve architecture first, objects and sentences second,
and typography last. Copyediting the current 96-page sequence without restructuring it would
not be enough for a 14/15 target.

## Editorial strategy: one source, two reading depths

A single 96-page artifact is being asked to serve two incompatible jobs: concise research
argument and exhaustive audit record. Preserve both jobs by producing two editions from
shared LaTeX components:

1. **Main-reading edition** (proposed `unified_report_main.tex` and PDF). A 30--32-page
   argument for reviewers and readers who need the
   contribution, methods, main results, boundaries, and decision implications.
2. **Full technical report** (`unified_report.tex` and `unified_report.pdf`). The existing
   comprehensive artifact, tightened and repaired,
   retaining detailed diagnostics, correction provenance, per-source results, and
   reproducibility material.

Do not copy result prose or numbers into a second independent source tree. Use shared section
files, generated macros, and explicit inclusion switches or thin wrapper files. The full
report remains the durable audit artifact; the main-reading edition becomes the recommended
entry point. If a venue later imposes a stricter limit, its limit replaces the 30--32-page
target without changing the editorial principles.

## Target narrative

The paper should read as a single argument rather than a sequence of loosely connected acts:

> Benchmark composition changes the apparent winner; specialization is conditional, score
> composition can recover some capability, general-safety scores do not by themselves cover
> a regulated-domain policy, and local deployment decisions must be made on a matched
> quality--latency--cost frontier.

Use the research-question labels consistently. Retire the parallel "Act I/II/III" naming
from navigation in the main-reading edition because two naming systems make the hierarchy
harder to learn. Historical source filenames can remain unchanged.

### Main-reading edition page budget

| Part | Page budget | Required content |
|---|---:|---|
| Abstract and opening synthesis | 2 | Problem, design, headline result, boundaries, one synthesis visual |
| Introduction and closest work | 3 | Gap, contributions, closest comparisons, report map |
| Shared methods and evidence rules | 3 | Data units, metrics, pairing, evidence tiers, operating-point rule |
| Q1: benchmark-conditioned specialization | 5 | Primary result, matched-budget view, one stability diagnostic |
| Q1b and Q2: guard adaptation and composition | 5 | What failed, what recovered, equal-cost control, limits |
| Q3: regulated-domain boundary case | 4 | Case-study framing, quadrant result, error structure, limits |
| Q4: local-guard frontier | 4 | External gap, latency/cost frontier, explicitly post-hoc sensitivity |
| Synthesis, limitations, reproducibility | 3 | Decision guide, strongest limits, exact artifact pointer |
| Conclusion | 1 | Answer the research questions without introducing new numbers |
| **Total** | **30** | Leaves up to 2 pages of contingency |

References and technical appendices sit outside this budget. A table of contents is useful in
the full report but should normally be omitted from the main-reading edition.

## Work plan

### Phase 0: freeze correctness before polishing dependent claims

**Objective:** ensure that line editing does not make a known analysis issue look settled.

1. Resolve the threshold-tie semantics in `matched_fpr.py`. The current quantile plus `>=`
   rule can produce a realized false-positive rate different from the stated target when
   scores tie. Decide and document whether the estimand is:
   - a deterministic threshold with the closest attainable realized FPR;
   - a conservative threshold that never exceeds the budget; or
   - randomized tie-breaking, if that is scientifically and operationally justified.
2. Regenerate every dependent table, macro, and sentence through the source workflow.
3. Until that decision is complete, do not call the current row "exactly matched" and do not
   call the corresponding point "deployable." Use a neutral placeholder such as "evaluated
   near the target false-alarm budget."
4. Run the study verification commands and record pinned-environment limitations rather than
   treating unavailable checks as passes.
5. Freeze the post-correction artifact commit or content hash for the editorial pass.

**Exit criterion:** the realized operating point, table label, caption, and prose all describe
the same rule, and the generated artifacts verify under the environments that are available.

### Phase 1: build the structural edit map

**Objective:** decide what the main argument needs before rewriting sentences.

1. Create a paragraph-level reverse outline. For every paragraph, record:
   - its single job;
   - the claim it supports;
   - its evidence tier;
   - whether it stays in the main edition, moves to the full report/appendix, merges, or goes.
2. Build one claim-control sheet with these fields:
   - stable claim identifier;
   - current wording;
   - source artifact or generated macro;
   - population and benchmark scope;
   - confirmatory, retrospective, development, or descriptive tier;
   - allowed conclusion;
   - prohibited overreach;
   - main-text location.
3. Require each major section to answer one research question in its first and last
   paragraphs. Any subsection that does not change the answer moves out of the main edition.
4. Move historical corrections and superseded values into a single, dated "Corrections and
   provenance" appendix or repository change log. Main prose states the current truth; it
   should not force every new reader through the history of discovering it.
5. Retain a short correction in the main text only when omitting the history would make the
   current evidence misleading or prevent a reader from reconciling a previously released
   artifact.

**Deliverables:** reverse outline, claim-control sheet, cut/move map, and main-edition wrapper
design.

**Exit criterion:** every main-edition paragraph has a unique purpose and every headline
claim has an evidence pointer and a stated boundary.

### Phase 2: rewrite the front matter

**Objective:** let a reader understand the contribution and evidence boundary in two pages.

#### Abstract

Replace the approximately 410-word PDF-extracted abstract with 180--220 words in five moves:

1. one sentence for the practical research problem;
2. one sentence for the paired design, datasets, and evaluation regime;
3. one or two sentences for the strongest Q1/Q2 numerical result;
4. one sentence for the regulated-domain and frontier boundaries;
5. one sentence for the qualified takeaway.

Do not place reproduction-status counts, correction history, fragile post-hoc Q4 values, or a
miniature discussion section in the abstract. Use no more than three headline numbers, and
ensure each is generated and repeated consistently in the results.

#### Title and opening spread

- Remove broad adjectives such as "comprehensive" unless the title needs them for
  disambiguation.
- Keep a subtitle only if it states the comparison or evidence boundary in at most 12 words.
- Retain one opening synthesis visual. Do not make the figure and claim ledger repeat the
  same prose.
- Reduce the claim ledger to question, direct answer, strongest evidence, and boundary.
- Replace categorical question answers with evidence-calibrated wording. For example:
  - Q3: "Not by itself on this regulated-domain benchmark," not an unqualified "No."
  - Q4: "When is a small local guard preferable on the measured frontier?" not a universal
    deployment recommendation.
- Rename "At a deployable operating point" to "At a matched false-alarm budget" or the exact
  phrase licensed by the corrected operating-point analysis.

**Exit criterion:** a cold reader can state the problem, principal finding, and strongest
boundary after reading only the abstract and opening spread.

### Phase 3: compress each section around its decision-relevant result

#### Introduction and related work

- Merge the essential closest-work comparison into the introduction; reviewers should not
  wait until after methods to learn what is new.
- Keep one contribution list with three or four falsifiable contributions.
- Delete duplicate roadmaps, repeated thesis statements, and prose that narrates the paper's
  revision process.
- Reduce the related-work taxonomy to the comparisons that constrain novelty. Move the full
  landscape table and teaching explanation to the full report.
- End with the four research questions, each using the exact wording used in later headings.

**Target:** 3 pages combined.

#### Shared background and methods

- Assume a technically literate ML reader. Keep only definitions required to interpret the
  results: AP, matched false-alarm evaluation, pairing, uncertainty unit, evidence tiers, and
  the regulated-domain label structure.
- Move worked AP examples, elementary mortgage vocabulary, extended policy-card teaching
  material, and procedural detail to appendices or the full report.
- Put the evidence hierarchy before the first result and use it unchanged thereafter.
- Present the common design once. Later sections should state only deviations.

**Target:** 3 pages in the main edition.

#### Q1: benchmark-conditioned specialization

- Lead with the primary paired result and its uncertainty, then explain what changed and what
  did not.
- Keep one primary table, one matched-budget result, and at most one stability or mechanism
  diagnostic.
- Move per-benchmark matrices, attractor details, prevalence decompositions, seed-level
  values, and extended KL diagnostics to the technical appendix unless they overturn the
  main answer.
- Describe KL regularization as a retention trade-off, not as a demonstrated generalization
  solution.
- Remove revision archaeology; state the current panel and current estimand directly.

**Target:** 5 pages.

#### Q1b: adaptation of released guards

- State panel eligibility and preflight exclusions before reporting an effect.
- Separate "the guard did not move" from "the instrument could not observe movement."
- Keep the registered contrast and its boundary. Move superseded mixed-panel values, null
  harness details, and correction history to provenance material.
- Do not turn a sparse or failed preflight into a broad claim that released guards cannot be
  adapted.

**Target:** 2 pages, sharing a 5-page block with Q2.

#### Q2: score composition

- Start with the operator and the exact deployment question it answers.
- Keep the primary recovery result, the equal-cost control, and one failure case.
- Move secondary ablations and tutorial detail out of the main edition.
- Use "recovery" when that is what the estimator supports; reserve "dominance" for a tested
  Pareto statement.

**Target:** 3 pages, sharing a 5-page block with Q1b.

#### Q3: regulated-domain case study

- Frame this as a measured case study, not proof about every regulated domain.
- Keep the policy/knowledge quadrant, the strongest baseline comparison, and the error type
  that most affects the practical conclusion.
- Move dataset-construction chronology, extended vocabulary, full pair examples, and release
  corrections to the full report.
- Preserve the statement that the guard supports triage and audit; it does not make lending,
  legal, or compliance decisions.

**Target:** 4 pages.

#### Q4: quality--latency--cost frontier

- Separate the supported external gap from represented-panel and post-hoc sensitivity
  analyses. Do not let the latter carry the section headline.
- Keep one frontier figure and one compact sensitivity table. Move the full model ladder,
  released-guard catalog, ensemble details, and exploratory cells to appendices.
- State hardware, batch regime, quantization state, threshold rule, and comparison population
  beside the frontier claim.
- Say "preferable on the measured frontier under these constraints," not "better for local
  deployment" without qualification.

**Target:** 4 pages.

#### Synthesis, limitations, reproducibility, and conclusion

- Replace repeated result summaries with one decision table: deployment need, supported
  evidence, unsupported inference, next measurement.
- Rank limitations by their power to reverse the conclusion. Do not give equal visual weight
  to cosmetic and thesis-threatening limitations.
- Keep a one-page reproduction contract in the main edition: artifact, command, expected
  coverage, pinned-environment boundary, and where to find full instructions.
- Put detailed coverage ledgers and command transcripts in the appendix or repository.
- Make the conclusion answer the four questions using no new result and no stronger verb than
  the evidence tier permits.

**Target:** 4 pages total.

### Phase 4: perform a claim-safe line edit

**Objective:** improve clarity without flattening scientific qualifications.

Apply the following rules consistently:

1. **Result-first paragraphs:** claim, quantitative evidence, scope/boundary, implication.
2. **One term per concept:** define and standardize "represented," "transfer," "purpose-built,"
   "general-safety," "matched FPR," "development," and "retrospective."
3. **One qualification in the right place:** state a caveat next to the claim and in the claim
   ledger; do not repeat it in every transition.
4. **Current truth in present prose:** replace "an earlier revision was wrong" with the current
   definition or result, with a pointer to the provenance note when needed.
5. **Calibrated verbs:** use "shows" for direct measurements, "supports" for bounded
   inference, "suggests" for exploratory evidence, and "does not establish" for boundaries.
6. **Controlled precision:** use consistent decimal precision by metric. Avoid printing more
   digits in prose than affect the decision. Preserve higher precision in generated tables
   only when analytically necessary.
7. **Short navigation:** delete phrases that repeatedly announce what the next paragraph or
   section will do. Use headings and transitions to carry navigation.
8. **Plain syntax:** prefer active voice, concrete subjects, and one main assertion per
   sentence. Split sentences whose caveat stack obscures the result.
9. **No self-certifying rhetoric:** cut words such as "honest," "clearly," "actually," and
   "definitive" unless they are part of a defined technical distinction.
10. **Stable cross-references:** refer to research questions and descriptive section names,
    not page numbers or the obsolete act hierarchy.

Run three separate passes rather than mixing them:

- Pass A: claims, evidence tier, and scope;
- Pass B: paragraph order, redundancy, and transitions;
- Pass C: sentence clarity, terminology, grammar, and punctuation.

### Phase 5: redesign tables, figures, and callouts

**Objective:** make visual objects faster to interpret than the prose they replace.

#### Tables

- Replace the broken evidence-ledger float with `longtable` or intentionally split tables.
  Repeat column headers on every page and never allow a row to appear as detached prose.
- Remove `\resizebox` from the main edition. Reduce columns, split panels, transpose the
  comparison, or use a planned landscape page instead of shrinking text.
- Set a minimum table type size of `\footnotesize` and verify it at 100% print scale.
- Put the inferential unit, interval type, evidence tier, and sample size in a concise note,
  not scattered across body prose.
- Use em dashes, "not evaluated," and "not applicable" distinctly; never let an empty cell
  carry ambiguous meaning.
- Align decimals and keep metric order identical across related tables.

#### Figures

- Give every main figure a one-sentence claim. If two figures make the same claim, retain the
  clearer one.
- Use a colorblind-safe palette that remains distinguishable in grayscale.
- Standardize panel lettering, axis typography, line weights, legend order, and model-family
  colors.
- Put direct labels near the data when they reduce legend lookup.
- Export Graphviz/process diagrams as vector PDF or SVG when the build path supports it.
  Otherwise use a documented high-resolution raster and inspect it at 200--300% zoom.
- Do not imply a continuous frontier where only a few measured configurations exist.

#### Captions

Use a common caption pattern:

1. the result in one sentence;
2. what is plotted or tabulated;
3. estimator, uncertainty, and evaluation unit;
4. the single boundary most likely to change interpretation.

Aim for 60--90 words. A caption must support standalone interpretation, but it should not
reproduce the entire methods section.

#### Callouts

- Use at most one takeaway callout per research question in the main edition.
- Reserve background boxes for prerequisites genuinely needed by the target reader.
- Keep evidence/design/boundary boxes only when the distinction prevents a plausible
  overclaim.
- Convert decorative or repetitive boxes to ordinary paragraphs. The target is no more than
  8 callouts across the main edition.
- Never place two callouts back-to-back.

### Phase 6: repair typography, navigation, and accessibility

**Objective:** deliver a professionally finished PDF, not only clean source prose.

- Add `pdftitle`, `pdfauthor`, `pdfsubject`, and `pdfkeywords` through `\hypersetup`.
- Verify bookmarks, link targets, table-of-contents entries, and visible link styling.
- Eliminate orphan headings, widowed single lines, clipped rules, split callouts, and caption
  separation from their objects.
- Harmonize spacing before/after headings, tables, figures, equations, and callouts.
- Keep running headers short and stable; avoid exposing internal source names.
- Provide meaningful textual descriptions around essential figures. Where the toolchain
  supports actual alternative text, include and validate it.
- Investigate tagged-PDF support only after the layout is stable. Mark it complete only if an
  accessibility checker validates the produced file; otherwise document it as remaining work.
- Test the PDF on screen, in grayscale, and at 100% print scale.

### Phase 7: run editorial and artifact QA

**Objective:** verify the edited artifact rather than trusting source inspection.

#### Automated checks

Run from the repository root unless noted:

```bash
make -C papers/unified-report verify
make -C papers/unified-report verify-heavy
make -C papers/unified-report pdf
make report-html
make check-report-html
git diff --check
rg -ni "Overfull|Underfull|Float too large|undefined references|multiply defined" \
  papers/unified-report/build/unified_report.log
pdfinfo papers/unified-report/unified_report.pdf
pdftotext -layout papers/unified-report/unified_report.pdf /tmp/unified-report.txt
```

Interpret `verify` correctly: pinned-environment items that cannot run are incomplete checks,
not failures and not passes. Because `verify-heavy` depends on `verify`, its head-to-head step
will not start while the standard check is incomplete; either satisfy the pinned environment
or run and compare the documented offline head-to-head command separately. Run the heavy
bootstrap only when its cost is acceptable. `report-html` intentionally rewrites the HTML
edition, so inspect that diff before running the read-only `check-report-html`. Writing-only
changes still require a PDF build and artifact-drift check.

Add a small editorial audit script or documented command set that reports:

- page and extracted-word counts for both editions;
- abstract word count;
- callout, figure, table, and `\resizebox` counts;
- remaining correction-history phrases in the main text;
- all build warnings;
- missing PDF metadata;
- unresolved or multiply defined references.

#### Rendered review

1. Render every PDF page to images and inspect a contact sheet for rhythm and density.
2. Inspect every table and figure at 100% and 200% zoom.
3. Check all multi-page tables for repeated headers and intact rows.
4. Check that no page has a stranded heading, one-line paragraph, or unexplained blank area.
5. Open every internal and external link in a representative PDF viewer.
6. Compare the abstract, opening claim ledger, results headings, and conclusion for identical
   claim scope.
7. Verify that moving material did not delete the only definition of a term used in the main
   edition.

#### Human review

Use two readers with different jobs:

- **Claim auditor:** checks every headline statement against the claim-control sheet and
  generated evidence.
- **Cold reader:** has not followed the revision history and marks every point where the
  argument, terminology, visual hierarchy, or practical implication is unclear.

The author then performs a final read in the built PDF, not only in LaTeX. Any issue found in
the PDF reopens the relevant phase.

## August schedule

### Week 1: correctness freeze and architecture

- resolve the matched-FPR definition and regenerate dependent artifacts;
- create the claim-control sheet and reverse outline;
- decide the shared-source mechanism for the two editions;
- approve the 30--32-page outline and move/cut map.

**Milestone:** no unresolved claim-definition issue blocks prose work.

### Week 2: front matter, Q1, Q1b, and Q2

- rewrite the abstract, opening spread, introduction, and closest-work section;
- compress common methods;
- rewrite Q1/Q1b/Q2 around their principal estimands;
- move diagnostics and revision history without losing provenance.

**Milestone:** the first half of the main edition builds and is within its page budget.

### Week 3: Q3, Q4, synthesis, and visual system

- rewrite Q3 as a bounded regulated-domain case study;
- separate supported and post-hoc Q4 evidence;
- replace repeated summaries with the decision table;
- repair the evidence ledger, eliminate `\resizebox`, consolidate callouts, and standardize
  figures/captions.

**Milestone:** the complete main edition builds at no more than 32 pages.

### Week 4: line edit, rendered QA, and cold review

- complete the three line-edit passes;
- add metadata and navigation improvements;
- clear layout defects and inspect every rendered page;
- run reproduction, build, and diff checks;
- obtain claim-auditor and cold-reader feedback;
- make only evidence-safe final corrections and request a re-score.

**Milestone:** all release gates below pass, or remaining exceptions are explicitly recorded.

## Release gates and re-score rubric

### Mandatory release gates

- [ ] The abstract is 180--220 words and contains no more than three headline numbers.
- [ ] The main-reading edition is at most 32 pages and 18,000 PDF-extracted words, excluding
      references and appendices.
- [ ] The full technical report remains available and retains necessary provenance.
- [ ] Every headline number is generated or has an explicit source artifact.
- [ ] The abstract, claim ledger, section verdicts, and conclusion use the same claim scope.
- [ ] The matched-FPR wording matches the implemented and realized operating-point rule.
- [ ] Retrospective, development, and post-hoc evidence are labeled at first use and never
      promoted by the rewrite.
- [ ] Targeted revision-history phrases are reduced from 21 current main-source line hits to
      at most 2 in the main-reading edition.
- [ ] The main edition has at most 8 callouts and no back-to-back callouts.
- [ ] No main-edition table uses `\resizebox` or type smaller than `\footnotesize`.
- [ ] Multi-page tables repeat headers and preserve complete rows.
- [ ] The build log contains no overfull box or oversized-float warning.
- [ ] All remaining layout warnings have been inspected and recorded as harmless or fixed.
- [ ] PDF title, author, subject, and keywords are present; bookmarks and links work.
- [ ] Every page has been inspected from a rendered image and every visual at 200% zoom.
- [ ] Reproduction checks report passes, failures, and unavailable pinned-environment checks
      separately.
- [ ] `git diff --check` passes and the final worktree diff contains only intended changes.
- [ ] A claim auditor and a cold reader have reviewed the built main-reading PDF.

### What earns 14/15

All mandatory gates pass; the main argument is understandable on one read; no visual defect
blocks interpretation; and the cold reader finds no section-level restructuring issue. Minor
sentence edits or optional accessibility enhancements may remain.

### What earns the 15/15 stretch score

The 14/15 conditions pass, the independent reader finds no material ambiguity or navigation
failure, the visual system works in grayscale and at print scale, and all feasible metadata
and accessibility checks supported by the chosen toolchain pass. Full PDF tagging is not a
requirement unless the project adopts a toolchain that claims and validates it.

## Risks and controls

| Risk | Control |
|---|---|
| Cutting length removes necessary caveats | Keep the claim ledger and claim-control sheet; move evidence, do not erase its boundary |
| Two editions drift | Share sections and generated macros; prohibit copied numerical prose |
| Prose freezes before results | Complete Phase 0 and freeze the artifact hash first |
| Revision history overwhelms the argument | Centralize it in one dated provenance location |
| Simplified headings overclaim | Review every heading against the evidence tier and allowed conclusion |
| Tables become smaller instead of clearer | Ban main-edition `\resizebox`; split or redesign |
| Accessibility is claimed but not delivered | Validate produced files; record unsupported tagging honestly |
| Page target conflicts with a later venue | Treat the venue limit as authoritative and retain the full report |
| A writing edit changes a number | Numbers remain generated; rerun verification after every result-bearing merge |

## Self-review record for this plan

Reviewed on 1 August 2026 against the checked-out repository:

- `pdfinfo` confirmed 96 pages, no custom metadata, and an untagged PDF;
- `pdftotext -layout` confirmed approximately 51,345 total extracted words, 36,483 through
  page 67, and approximately 410 extracted words in the abstract;
- source searches confirmed 13 background, 16 takeaway, and 4 evidence/design/boundary
  callouts; 17 figure environments, 11 table environments, and 2 `\resizebox` uses;
- a targeted search for revision-history language found 21 line hits in current main-source
  material; broad uses of the word "correction" were rejected because they also count valid
  statistical phrases such as multiplicity correction;
- the build log confirmed 23 overfull/underfull/oversized-float warning lines, including the
  approximately 55.6 pt overfull vertical box and 52.8 pt oversized float;
- `matched_fpr.py` was inspected directly: it selects a quantile and scores positives with
  `>=`, so tied negative scores can make the realized FPR differ from the nominal budget;
- the named Make targets were dry-run to confirm that they exist, and the dependency of
  `verify-heavy` on `verify` was reflected in the plan;
- whitespace checks passed for this new Markdown file, referenced current paths exist, and
  the repository Markdown-link checker reported no broken links.

This review validates the plan's current baselines and command names. It does **not** claim
that the paper has already passed the future release gates, that the matched-FPR issue has
been fixed, or that a 30--32-page edition has already been built. The page and word limits
are editorial targets pending a venue-specific limit; the PDF-tagging recommendation remains
conditional on toolchain support and independent validation.

## Definition of done

The plan is complete only when both editions build from shared evidence, the main edition
meets the page and word budgets, all mandatory gates pass, the full report preserves the
audit trail, and the final re-score is performed on the rendered PDF. Completion is not "the
LaTeX compiles"; it is a concise, evidence-calibrated paper whose visual and verbal hierarchy
lets a new reader recover the contribution, result, and boundary without prior knowledge of
the project's revision history.
