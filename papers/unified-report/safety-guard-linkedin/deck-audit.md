# Deck Audit — Technical & Executive Decks vs. `unified_report.pdf`

**Ground truth:** *Safety Benchmark Gains Do Not Guarantee Safety Transfer* (96 pp., Aug 1 2026)
**Audited:** `safety_guard_benchmark_deck - Repaired.pptx` (21 slides) · `safety_guard_exec_deck.pptx` (13 slides)
**Date:** 2 Aug 2026

---

## 1. Verdict

Both decks are in unusually good shape. I checked roughly 190 discrete numeric and factual claims across slide bodies, tables, embedded charts and speaker notes against the paper's tables, figures and appendices. **I found zero fabricated numbers and no claim that contradicts the paper's direction.** Every table cell, chart bar and interval I could trace resolved to a committed artifact in the report.

What I did find falls into three buckets:

| Bucket | Tech deck | Exec deck |
|---|---|---|
| **Numbers wrong** | 0 material · 2 trivial | 1 trivial (rounding) |
| **Numbers right, framing loses a caveat the paper insists on** | 5 | 4 |
| **Right number, wrong label / wrong baseline attached** | 3 | 3 |

The single highest-value fix in each deck:

- **Technical:** the headline `0.517 → 0.217` matched-budget collapse is quoted five times and never carries the paper's own qualifier — *it is an ROC point at a common alarm rate, not a deployable threshold* (Abstract, Table 1 preamble, Table 6 caption, §3.5, Appendix B.6). This is the one number the deck tells the presenter to lead with; it is also the one the paper hedges most carefully.
- **Executive:** slide 6's chart is labeled "change in **catch rate**" and its axis says "**AP points**", while the speaker notes quote a third quantity (ExpGuard matched-budget recall) and describe the chart as "external expert-annotated data" when the plotted values are the internal, inspected panel. Three measurements, one slide.

Neither is a correctness failure. Both are the kind of thing that costs credibility in the Q&A rather than in the deck.

---

## 2. What I checked and how

1. Extracted the full 96-page PDF to text (`pdfplumber`), including all 31 tables and figure captions.
2. Extracted every text frame, table cell and speaker-note block from both `.pptx` files (`python-pptx`).
3. Extracted all 16 embedded chart images and read each one visually, comparing plotted values and labels to the source tables.
4. Traced each deck claim to a numbered table/figure/section in the report.
5. Re-derived every arithmetic claim (ratios, "×", "% of the gap closed", "nearly four times") from the underlying values rather than accepting the deck's arithmetic.

Charts verified bar-by-bar against source tables: exec slides 4, 6, 7, 8; tech slides 7, 8, 9, 10, 11, 12, 15b, 16. **Exec slide 4's 19-bar leaderboard is exact against Table 18 on all 19 rows.**

---

## 3. Technical deck — findings

### T1 · HIGH — The matched-budget result is presented without the paper's central qualifier

**Where:** slide 2 (notes), slide 3 (Q1 bullet), slide 7 (bullet 1 + notes), slide 20 (bullet 2), slide 21 (bullet 2, "if you only get one sentence, use that one").

**Deck says:** "Match the budget and transfer recall goes 51.7% → 21.7%, HarmBench 78.0% → 20.3% — worse on all four." Slide 21: "The matched-alarm-budget read. Cheap, almost never done, and it reverses a headline… Ranking arithmetic on committed scores — no GPU, no retraining."

**Paper says (Table 6 caption, verbatim):** *"This is a retrospective ROC point, not a deployable threshold: the quantile is read off the same labelled negatives the recall is then measured on, so a production system without labels could not place it. Read the row as 'recall at an empirical matched-FPR ROC point', and see Appendix B.6 for what an operational version would require."* The abstract and Table 1's preamble repeat it.

**Why it matters:** the deck's framing — "no GPU, no retraining, ranking arithmetic on committed scores" — is true and is precisely what invites a listener to hear it as something they could ship. The paper anticipated exactly that reading and blocked it. A reviewer who has read the report will ask, and the presenter currently has no prepared answer.

**Fix:** one clause on slide 7 ("…an ROC point at a common alarm budget, not a threshold a production system could place") and one in slide 21's bullet. It costs nothing and makes the slide unattackable, which is the deck's own stated philosophy ("prefer the measurement that can be wrong in a detectable way").

---

### T2 · MEDIUM — Slide 10's header names a different panel from the one its numbers come from

**Deck says:** header `PREREGISTERED · 10 CHECKPOINTS, 6 MODEL FAMILIES`, then quotes RQ1 `+0.111 (LCB +0.070)`, `H_conc +0.183 (LCB +0.137)`, RQ2 `+0.047 (LCB +0.032)` / `−0.034 (LCB −0.062)`.

**Paper (§4.1, "Which panel each statistic is computed over, and a correction"):** those four statistics are the **registered purpose-built panel — 6 released guards spanning 5 model families**. The general panel (4 checkpoints, 2 families) is a separate block; the 10-checkpoint / 6-family grid is the full design, not the estimand. The paper devotes a subsection to this because an earlier revision computed it on the mixed six-family panel and got different numbers (`H_gain +0.174`, `H_conc +0.239`).

**Fix:** `PREREGISTERED · 6 RELEASED GUARDS, 5 FAMILIES (within a 10-checkpoint grid)`. The "not a general-model artifact" bullet already carries the 10-checkpoint story.

---

### T3 · MEDIUM — Slide 10 omits two of the four reasons the paper refuses a confirmatory reading

**Paper (§4.2):** *"A confirmatory verdict would require a protocol that was actually followed, and four things went wrong with this one… the claim registry is dev_nonfinal and no lock binds it; **no checkpoint has a passing preflight, so the eligibility gate never ran**; a degenerate cell was retained against that gate; and **the panel split reported here was written after the outcomes were known**, to repair an analyzer computing the wrong estimand. Any one of those is enough to disqualify a confirmatory reading."*

**Deck notes cover:** registry `dev_nonfinal`, and not-data-blind. **Missing:** no passing preflight, and the post-hoc panel repair.

**Why it matters:** the notes tell the presenter this is "the strongest evidence tier for the specialization claim, so lean on it." Leaning on it while two of four disqualifiers are unspoken is the highest-risk moment in the deck. The slide's `RQ1 · CRITERION MET` wording is exactly right — the paper's own phrasing — so this is a notes fix only.

---

### T4 · MEDIUM — Mortgage `AP·D` values appear without their chance floor

**Where:** slide 3's Q3 panel shows `0.85 / 0.79 / 0.73 / 0.67` under "mortgage policy."

**Paper (§6.4, Table 16, Figure 10, Evidence box, Appendix E.1 — five places):** *"D-positives are 81/146, so a random ranker already scores AP·D ≈ 0.555 and the observed band is only 0.12–0.30 above chance."*

**Why it matters:** unqualified, `0.85` reads as a competent guard and quietly undercuts the deck's own Act III thesis. With the floor stated, the same number becomes evidence *for* the argument. This is the cheapest strengthening move in the deck.

**Fix:** add `chance floor 0.555` to the Q3 panel caption or as a bullet on slide 13.

---

### T5 · MEDIUM — Slide 2 mislabels the case-study row's domain and over-generalizes the "below all 65"

**Deck (slide 2):** *"**below all 65** — A coded **fair-lending** violation ranked below every benign mortgage inquiry in the split."*

**Paper (Figure 8):** row `MGB-UD-00020` is `udaap / deceptive`, difficulty hard. Its cited cards span fair-lending concepts (D02 redlining, D07 disparate impact) plus D01, D12, D13, D14 — six cards, not three. And *"All four guards rank this violation below the **median** benign inquiry… and **SmolLM3-3B** ranks it below every one of the 65."* Only one of four guards produces "below all 65" (Qwen2.5 46/65, SmolLM2 57/65, Qwen3-4B 44/65).

Slide 14 gets both right. Slide 2 is the compressed version that drifts.

**Fix:** "a coded redlining-by-proxy violation" and "one of the four guards ranked it below every benign inquiry in the split; all four ranked it below the median."

---

### T6 · LOW — Slide 21: "the 39 unscored protected pairs"

3 of the 39 pairs sit in public-test and **were** scored — they are the `n=3` behind slide 15's `Δcontext` numbers. The paper's roadmap item is that *all* 39 are legitimate evaluation data for zero-shot guards and scoring them is a zero-training recompute, reported separately from the public-test three. Say **"36 remaining protected pairs"** or **"rescore all 39 as one comparable set."**

---

### T7 · LOW — Slide 12's operating-point table doesn't say which regime it is

Column headers read `Macro TPR | Macro FPR | Pooled FPR`. Paper Table 14 is explicitly *"realized **transfer** operating points."* Since the same slide's callout compares against "the 5% target," a reader can take these for represented-source rates. Add `(transfer)` to the table title.

---

### T8 · LOW — Slide 11 drops the qualifier on "best worst-regime scorer"

**Deck:** "min(represented, transfer) = 0.883, beating the base's 0.658 and SFT's 0.807."
**Paper (Table 11 + §5.5):** the logit-average ablation reaches **0.891** — nominally better — and is non-promotable only because it was dev-visible. The paper is careful to write *"the most balanced **promotable** operator"* and repeats it in the Evidence box.

Add "promotable" (one word) and, in notes, the 0.891 line. It's a self-imposed handicap that reads as rigor.

---

### T9 · LOW — β=1.0 SmolLM2 transfer change: deck says `+0.004`, Table 9 says `+0.005`

The deck matches the paper's §3.7 **prose**, which itself disagrees with its own Table 9 by 0.001. Worth reconciling in the paper; adopt the table value in the deck.

---

### T10 · LOW — Slides 17/18 tuned rows are the in-env re-execution, unlabelled

**Paper (§7.2, Appendix E.1):** *"One provenance caveat governs every tuned row and is stated wherever they appear.* The Act I release adapters were lost with an ephemeral runner, so all tuned rows in §7 are the KL-SFT `β=0` arm — same LOCK contract and manifest, different `adapter_sha256`. Labelled `SFT (in-env)` throughout." Slide 9's notes explain the mechanism beautifully; slides 17/18 just need the label in the table.

---

### T11 · LOW (design) — Slide 8's prevalence chart uses two near-identical reds

Qwen3-4B and SmolLM2-1.7B are plotted in nearly the same red at opposite ends of the chart. On a projector these read as one series, which destroys the "the lower two swap" point the slide is built around. Recolor SmolLM2.

---

## 4. Executive deck — findings

### E1 · HIGH — Slide 6 labels one measurement, plots a second, and narrates a third

| Element | Quantity it actually is |
|---|---|
| Subtitle: "Change in **catch rate** after fine-tuning, across 6 models · 5 training runs each" | implies ExpGuard TPR@5%FPR |
| Chart axis: "Change in accuracy after fine-tuning (**AP points**)"; values `+0.528/+0.040 … +0.037/−0.117` | **Table 19 macro-AP** on the internal Act I panel (represented / transfer) |
| Notes: "swings from −0.059 to +0.122", "mean +0.005" | **ExpGuard matched-budget recall** (§7.2) |
| Notes: "reproduced here on **external expert-annotated data** rather than on our own panel" | false of the chart — the chart *is* the internal, inspected panel |

All three sets of numbers are individually correct. The slide asserts they are the same thing.

**Why it matters:** this is the slide that kills the "just fine-tune ours" proposal. If anyone in the room has the report open, the mismatch between the subtitle and the axis is visible in five seconds, and the notes' "external data" claim is the one that will get challenged. The paper is emphatic that internal-panel and ExpGuard numbers *"are never pooled"* (Appendix E.2 — "the three flavors are never pooled" is called "the spine of the whole report").

**Fix, cheapest version:** change the subtitle to "Change in ranking accuracy after fine-tuning (macro-AP), our internal 6-model panel · 5 training runs each"; move the ExpGuard sign-split numbers into a separate labelled bullet ("and on external expert-annotated prompts, the same pattern: 4 of 6 got worse"). Keep the chart.

---

### E2 · HIGH — Slide 9's table mixes two different in-house baselines

- Row 1: *"Best single in-house guard — **83%** — 1 model call — the baseline"* = Qwen3-32B base, `.830`.
- Row 5: *"Escalate the uncertain 20% instead — **84%** — 1.2 model calls"* = `.842`, which the paper measures with a **SmolLM3-3B** inline guard starting at `.787`. Slide 8's own chart states this: *"inline guard SmolLM3-3B (~20 ms, self-hosted) · keep every request in-house — 79%."*

Both figures are correct absolute TPR@5%FPR on the same 2,275 rows, so nothing is false. But the table layout says *escalation adds one point to your best in-house guard*, when what was measured is *escalation adds five points to a weaker 3B guard*. And escalation on top of a 32B inline guard **was never measured** — that combination doesn't exist in the report.

An exec reading rows 1 and 5 together will plan around "83% → 84%." An engineer implementing it will get 79% → 84% and a different serving cost.

**Fix:** name the inline guard in row 5 ("Escalate the uncertain 20% *from the 3B inline guard*"), restate the baseline column so 79% and 83% both appear, and footnote that the cascade over a larger inline guard is unmeasured.

---

### E3 · HIGH — Slide 9 row 2 attaches the wrong baseline *and* the wrong cost to seed ensembling

**Deck:** *"Average the 5 tuning runs of one guard — **+0.026 on top** — cost: **already paid for** — verdict: worth doing."*

Two problems:

1. **Baseline.** The paper's `+0.026` (§7.2) is the mean gain of a seed ensemble over a **single SFT seed of the same checkpoint** — not over the 83% base row it sits directly beneath. For Qwen3-32B concretely: base `.830` → seed ensemble `.834`, i.e. **+0.004 over the base**, +0.025 over its own tuned seed. Reading "+0.026 on top" of 83% produces ~85.6%, which appears nowhere in the report and would beat the fitted 18-guard stack.
2. **Cost.** "Already paid for" is true of **training** — the paper says *"the adapters already exist"* and *"costs no new training."* It is false of **inference**: averaging five adapters is **five forward passes per request**, in a column headed "Cost per request", in a deck whose whole argument is per-request latency and cost.

**Fix:** `+0.004 over the base / +0.026 over a single tuned seed` and `5 model calls (no new training)`. This also removes the row's apparent conflict with row 3 ("18 model calls — worse than one").

*Note:* the paper contributes to this — §7.2 calls the `.834` seed ensemble *"the strongest single open guard,"* which is loose. Worth fixing upstream too.

---

### E4 · MEDIUM — Slide 9: callout says "a fifteenth of the compute", notes say "a sixteenth"

18 calls ÷ 1.2 calls = 15. Make both "a fifteenth."

---

### E5 · MEDIUM — Slide 5 omits the sensitivity the paper calls "the one to keep in view"

The `+0.083` represented-source advantage is the load-bearing evidence for lane 1 and lane 3 ("self-host the enumerable share"). Slide 5's notes correctly cover two of the four weightings in Table 21 — row-count weighting (`+0.049 [−0.031,+0.112]`, straddles zero) and source resampling (`[−0.019,+0.220]`).

**Missing:** the fourth weighting. Including the base arms alongside the tuned ones gives **−0.264 [−0.321,−0.179] — excluding zero in the opposite direction.** The paper: *"That last row is the one to keep in view, because it says what the result is about: the represented-source advantage is a property of **tuned guards specifically** — Act I's specialization seen from the other side — and not a general statement that small guards beat hosted ones."*

**Why it matters commercially:** it converts "run a small guard in-house on traffic we can describe" into "run a small guard *that we have tuned on a manifest of that traffic*." That is a funded project with a maintenance burden, not a default configuration. The room should price that before approving lane 1.

---

### E6 · MEDIUM — The residency lane has the weakest evidence in the study, and the deck doesn't say so

Lane 1 — *"prompt contains borrower or patient data → in-house only → cannot lawfully leave"* — is the lane the whole architecture is built around, and the one where escalation is unavailable by construction. The report has a purpose-built instrument for exactly that traffic, and its findings are unflattering:

- All four zero-shot guards rank the worked `G0/D1` violation **below the median benign inquiry** in the same split; one ranks it below **all 65** benign rows (Figure 8).
- `AP·D` tops out at **0.85 against a 0.555 chance floor** — 0.12–0.30 above chance (Table 16).
- The fixed-threshold operating point is **not reportable**: its `G0/D1` catch count swung by >50 rows across library versions of one quantile routine (§6.4).
- The `G0/D1` stratum — reads safe, is a violation — is **502 of 994 rows**, by design the payload.

Slide 12 gestures at this ("not the harder mortgage-compliance question") as a scope note. It is not a scope note; it is a finding about the exact lane the deck recommends.

**Fix:** one slide or one strong callout before slide 13: *"On the one lane where we cannot escalate, we have the least evidence the in-house guard works — and the instrument we built to check says it misses the payload."* This does not weaken the recommendation (there is no lawful alternative for that traffic); it makes slide 13's "build the domain instrument" ask a consequence rather than an aspiration, and it protects the presenter from discovering this in Q&A.

---

### E7 · MEDIUM — Slide 3's speaker notes do not derive slide 3's three numbers

Slide shows **20% / 51% / 80%**. Notes give "+0.066 catch-rate gap [+0.043,+0.089]", "77×", "$0.80/1k" — none of which is where 51% comes from. The 51% derives from `.787 → .842` against `.896`, i.e. against the **+0.109** gap, not the +0.066 one.

A presenter asked "where does 51% come from?" will quote the wrong interval and be wrong about which model pair it refers to.

**Fix:** replace the notes' first bullet with: *"51% = (.842 − .787) / (.896 − .787), SmolLM3-3B inline vs. gpt-5.4 (low) at a matched 5% budget. The +0.066 [+0.043,+0.089] figure is a different pair — hosted vs. Qwen3-32B — and belongs on slide 4."*

---

### E8 · LOW — Slide 8 chart: "64% of the gap closed" at 30% escalation

`(.856 − .787) / (.896 − .787) = 63.3%`. The paper's own generated Figure 12 prints **+63%**. Match the artifact — the deck's other two points (30% and 51%) match exactly, so this reads as a transcription slip rather than a different convention.

---

### E9 · LOW — Slide 2's "SO WE MEASURED ON OUR OWN TERMS" risks implying internal data

ExpGuard is a **third-party, expert-annotated** set — which is a strength, and slide 12 says so correctly ("external and expert-annotated, which is the best tier available to us"). "Our own terms" invites a legal or risk stakeholder to assume the evaluation ran on company traffic. Reword to "so we measured on regulated-domain terms" or "…on external, expert-annotated finance/health/law prompts."

---

### E10 · LOW — Slide 5 headline generalizes past the result

Headline: *"On traffic we can describe in advance, our own guard already wins."* Body text correctly says "our **tuned** guards." Given E5, the tuned qualifier is load-bearing. Align the headline.

---

### E11 · LOW — Slide 2's notes misname the title finding

*"guard rankings reorder when the benchmark changes — that is the report's title finding."* The title finding is that a **represented-benchmark gain does not establish transfer**. Ranking reorder across benchmarks is the Q3 corollary. Minor, but the presenter is quoting the title.

---

## 5. Cross-cutting observations

**X1 — Noise-floor discipline is applied unevenly.** Tech slide 9's notes handle the reproduction envelope (mean 0.015, worst 0.029) exactly as the paper does, and correctly name composition's `+0.017` over base and KL β=1.0's `+0.004` as unresolved. Slide 11 then quotes composition's headline against SFT (`+0.076` — safely above the floor, fine) but its "RECOVERY, NOT DOMINANCE" callout could add one clause: the vs.-base edge is *inside* the envelope and therefore unresolved, not merely small. Paper, Appendix E.1: *"an edge inside the 0.015–0.029 reproduction envelope… and therefore unresolved."*

**X2 — The exec deck has no composition slide.** Base+adapter averaging is the paper's cheapest in-house repair (`+0.076` transfer for one extra pass, `+0.058,+0.093`, positive on all four checkpoints) and belongs beside slide 9's ensembling rows. **Caveat before you add it:** it is measured in macro-AP on the internal Act I panel, *not* in matched-budget recall on ExpGuard, so it cannot be dropped into slide 9's table as a fifth row without violating the deck's own "same rows, same budget" premise. Either give it its own framing or commission the ExpGuard measurement.

**X3 — Reproducibility claim is under-sold.** Tech slide 21 says "one command that regenerates the covered tables and prints the coverage it did NOT achieve" — accurate and well-phrased. The abstract's actual numbers are stronger and worth printing: **31 of 35 generated artifacts byte-checked in any environment, 4 requiring the pinned lock, 0 uncovered.** Also worth one line: Figure 4 (the specialization plane) ships without a generator and is neither regenerated nor compared — the paper names this itself, and naming it first is the deck's established style.

---

## 6. Recommendations to make each deck stronger

### Technical deck

1. **Fix T1 and the deck's best slide becomes its most defensible.** Slide 7 already does the hardest thing in the report — it retracts an earlier hedge and reports a *larger* effect. Adding "ROC point, not a deployable threshold" completes the move.
2. **Retitle slide 18 around its own conclusion.** The title is "Tuning does not close it, and neither does scale," but the slide's actual news — and the thing the paper explicitly retracted an earlier claim to say — is *"what is regular is the endpoint."* That's the sentence a technical audience will remember, and it's currently buried in the bottom callout.
3. **Promote one negative result to the front.** Slide 15 ("We built a fairness gate. It does not survive its own audit") is the deck's trust-purchasing moment and it arrives at slide 15 of 21. One line on slide 1 or 2 — "we built a fairness probe and it failed its own audit; that's in here too" — buys the room's attention for the other twenty slides.
4. **Slide 3 (Figure 1) will not read on a projector.** Four panels with rotated axis labels at that density is a handout figure. Either split it across two slides or annotate the four takeaways in slide-native type and keep the figure as a backup.
5. **Add the chance floor wherever `AP·D` appears** (T4). One number, large payoff.
6. **Consider a "noise floor" callout on slide 21.** The accidental reproduction envelope is genuinely the most interesting methodological byproduct in the report — an effect-size ceiling discovered by accident and then turned against the authors' own small effects. It's currently a notes-only item on slide 9 and a half-bullet on slide 21.

### Executive deck

1. **E1 first** — relabel slide 6. Highest credibility-per-minute fix in either deck.
2. **Rebuild slide 9 on one consistent baseline** (E2 + E3). As it stands, the most-quoted table in the deck has two baselines and one cost column that contradicts the deck's thesis.
3. **Add the "advantage is a property of tuned guards" sensitivity to slide 5** (E5). This is a budget consequence, not a statistical footnote.
4. **Add the regulated-domain evidence** (E6). Biggest *content* gap. It converts slide 13's fourth ask from a nice-to-have into the direct consequence of a measured finding, which is a much easier thing to fund.
5. **Make the provider-refusal finding more visible.** It is the strongest non-obvious point in the deck — a compliance control that intermittently declines to answer, non-reproducibly, for reasons the operator cannot inspect or appeal — and it is currently a callout on slide 10. It's also already coupled to the cheapest action item on slide 13. Consider making it slide 10's headline rather than its footnote.
6. **Price the self-hosted side, or say explicitly that you haven't.** `$0.80/1k` is billed tokens at list prices and *excludes the GPU you already own*. Both the paper and slide 12 say so. An executive will ask "so what does ours cost?" and the honest answer — "we have not amortized it; here is the A100-hours figure" — is better prepared than improvised.
7. **Consider merging slides 6 and 7.** "Two routes that don't work" is one idea occupying two slides in a 13-slide deck. Merging frees the room for E6, which is the thing the deck is currently missing.
8. **Slide 13's "one decision from this room" is well-chosen.** Consider adding a second: who owns the escalation-share dial once it exists, since slide 8 correctly frames it as a standing tradeoff between legal ceiling and budget rather than a one-time architecture choice.

---

## 7. Appendix — slide-by-slide alignment

### Technical deck (21 slides)

| # | Slide | Status | Note |
|---|---|---|---|
| 1 | Title | ✅ | |
| 2 | The problem | ⚠️ | 4.3→17.0%, 78.0→60.0% exact (Table 5). T5: domain mislabel + "below all 65" over-generalized |
| 3 | One figure (Fig. 1) | ⚠️ | All four panel values trace to Tables 3/16/17/22. T4: no chance floor on Q3. Legibility risk |
| 4 | How we measure | ✅ | Manifest, recipe, regimes, bootstrap all match §3.1 |
| 5 | What SFT buys (Fig. 3) | ✅ | +0.3234 / −0.0589 with intervals; per-benchmark deltas exact (Table 5) |
| 6 | Specialization plane (Fig. 4) | ✅ | 15/5/0/0 quadrants; "4 SmolLM2 + 1 Qwen2.5" exact (§3.4) |
| 7 | Operating point | ⚠️ | Chart exact vs. Tables 5+6. **T1** |
| 8 | Base rate (Fig. 5) | ⚠️ | Table + Eq. 4 exact. T11: color collision |
| 9 | KL mitigation | ⚠️ | Chart exact vs. Table 8; all pAUC ratios correct. T9 (±0.001) |
| 10 | Preregistered (Fig. 6) | ⚠️ | Chart exact vs. Table 10. **T2**, **T3** |
| 11 | Composition (Fig. 7) | ⚠️ | All deltas exact vs. Table 12. T8 |
| 12 | Equal-cost control | ⚠️ | Chart exact vs. Table 13; op-point table exact vs. Table 14. T7 |
| 13 | Dual-label design | ✅ | 450/0/502/42, 994, 24 cards, nesting limitation all exact |
| 14 | One row end-to-end | ✅ | All 12 table values exact vs. Figure 8; MGB-FL-00028 inversion correctly hedged |
| 15 | Two negatives | ✅ | Δcontext chart exact vs. Table 16; health +0.017 / +0.0026 / "not demonstrated ties" all exact |
| 16 | Why self-host | ✅ | Latency chart exact vs. Table 24; 77×, $0.80, concurrency 200 all correct |
| 17 | External reference | ⚠️ | All 4 rows + both paired intervals exact vs. Table 18. T10 |
| 18 | Two routes | ⚠️ | All 12 table cells exact vs. Table 19; +0.062/+0.066/−0.020 exact. T10; retitle (rec. 2) |
| 19 | Regime, not size | ✅ | +0.083, +0.039, .967/.917, [−0.019,+0.220], −0.264, 67–451, 1.6–5.0% — all exact |
| 20 | What to do | ⚠️ | Workflow matches Figure 14 exactly. **T1** (bullet 2) |
| 21 | Contributions | ⚠️ | **T1**, T6; X3 opportunity |

### Executive deck (13 slides)

| # | Slide | Status | Note |
|---|---|---|---|
| 1 | Title | ✅ | |
| 2 | The decision | ⚠️ | E9, E11 (both notes/wording) |
| 3 | The answer | ⚠️ | 20/51/80 all correct. **E7** (notes don't derive them) |
| 4 | What hosted buys | ✅ | **19/19 chart bars exact vs. Table 18**; "about half the misses" exact (Table 20 caption) |
| 5 | Better at what | ⚠️ | +0.083, [+0.013,+0.157], +0.039, .967/.917, .948/.741, 3-of-12, 67–451 all exact. **E5**, E10 |
| 6 | Route one: tune | ❌ | Chart values exact vs. Table 19; notes exact vs. §7.2. **E1 — three quantities, one label** |
| 7 | Route two: scale | ✅ | Chart + all three intervals exact vs. Table 18 |
| 8 | Why the rule works | ⚠️ | 79/82/84/86 and 30/51% exact. **E8** (64% vs. paper's 63%) |
| 9 | If nothing may leave | ❌ | .791/.850/.842 all exact. **E2**, **E3**, E4 |
| 10 | What hosted costs | ✅ | Table exact; refusal finding faithful to §7.1 |
| 11 | The useful finding | ✅ | Supported by §7.3; deployment-choice caveat correctly on-slide |
| 12 | Confidence and limits | ✅ | Six limits all traceable; 3.0× and "<51%" both correct |
| 13 | Proposal | ⚠️ | ~20 ms correct (SmolLM3-3B P50 20.1). **E6** gap lands here |

**Legend:** ✅ verified against source · ⚠️ numbers correct, framing/labeling issue · ❌ needs rework before presenting

---

## 8. One-paragraph summary for your own use

The decks do not misreport the paper. Every number I traced is real, and the hedging discipline — matched budgets, evidence tiers never pooled, negative results kept — survives the translation to slides better than it usually does. The corrections that matter are four: the technical deck's headline matched-budget number needs the paper's own "ROC point, not a deployable threshold" clause; slide 10's preregistration framing names the wrong panel and drops two of four disqualifiers; the exec deck's slide 6 labels an internal macro-AP chart as external catch rates; and the exec deck's slide 9 mixes two in-house baselines and prices a five-pass ensemble as free. Beyond corrections, the largest opportunity is in the exec deck: the recommendation rests on a residency lane where the report's own instrument shows the in-house guard missing the payload, and saying so out loud would make the ask on the final slide far easier to fund.
