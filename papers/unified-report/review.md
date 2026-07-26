# Deep review of *The Safety-Guard Benchmark Chooses the Winner*

**Artifact reviewed:** `unified_report.pdf` (71 pages, generated 2026-07-25)  
**Review date:** 2026-07-25  
**Recommendation:** **Major revision; not ready for claim-bearing submission in its current form.**

## Executive assessment

This report has a strong paper inside it. The paired base-to-tune design is substantially more informative than an ordinary leaderboard; the represented-versus-transfer split is useful; the paper usually states its fixed-panel scope; and the decision to report the protected-pair probe as a negative methodological result is unusually candid. The committed per-row evidence and the Paper A release contract are also a serious foundation rather than a cosmetic reproducibility appendix.

The present manuscript nevertheless has two central blockers. First, the adaptation extension called “preregistered confirmatory” is neither release-locked nor analyzed according to the purpose-built-panel estimand advertised in its protocol. Second, the abstract's one-command reproducibility claim is materially stronger than the current harness: only part of the claim-bearing surface is byte-checked, figures are regenerated but not compared, and stale outputs can be accepted after failed subprocesses. These are not wording nits; they affect the manuscript's strongest evidence label and its reproducibility contribution.

The mortgage work is promising as a benchmark prototype, but it is not yet a high-compliance validation instrument. Its two labels are empirically nested, not independent; the executable build path uses a single target-filtered judge vote despite the PDF claiming a three-sample majority; one protected/reference pair crosses train and public test; its “sealed” test is committed in plaintext; and its license remains unresolved. The policy-card labels also lack SME adjudication and a legally effective, versioned source snapshot. The domain experiments are base-only zero-shot probes, so the title and synthesis must not imply that tuning or composition was validated in regulated domains.

My publication-level verdict is therefore **major revision** (or weak reject if this were a conference submission with no revision cycle). The core retrospective finding can remain. The “confirmatory,” “all tables/figures,” “sealed,” “independent labels,” and deployment-prescriptive claims cannot.

## Claim-by-claim verdict

| Manuscript claim | Review verdict | What the evidence currently supports |
|---|---|---|
| Ordinary LoRA-SFT raises represented-source AP and has heterogeneous transfer effects on four compact checkpoints. | **Supported, conditional on the fixed panel and inspected data.** | A retrospective paired characterization of this recipe, manifest, and four checkpoints; not a model-population or causal law. |
| Calibrated base+adapter composition recovers transfer relative to SFT. | **Supported as a retrospective fixed-panel comparison.** | Recovery relative to SFT, with heterogeneous performance relative to the untuned base and an extra inference pass. The equal-cost control is useful but does not prove a general mechanism. |
| A preregistered confirmatory study establishes that released purpose-built guards specialize under SFT. | **Not supported as written.** | The per-checkpoint movements and a diagnostic purpose-built-only recomputation point in the same direction, but the printed bounds use the wrong pooled panel and the study has no final lock. |
| KL-SFT preserves transfer but incurs a represented-source cost. | **Descriptively supported on the scored panel; not confirmatory.** | The direction is present, but achieved KL, matched-achieved-KL sensitivity, full protocol binding, and native-contract retention are missing. |
| Mortgage and ExpGuard show that the benchmark/domain changes the numerical leader. | **Supported as a base-only, zero-shot descriptive observation.** | A benchmark-dependent numerical ordering. It does not establish domain tuning, compliance readiness, or an equivalence/tie between unresolved models. |
| The protected-pair probe ranks fairness behavior. | **Correctly rejected by the manuscript itself.** | A useful negative result: three evaluated pairs and scale saturation cannot rank guards. |
| Every claim-bearing table is regenerated from committed scores by one command. | **False for the current harness.** | Partial analysis reproducibility with strong Paper A coverage and several uncovered or weakly checked surfaces. |
| The mortgage benchmark is frozen and has a sealed private test. | **Checksummed: yes. Sealed and provenance-complete: no.** | A committed static benchmark with a public-by-repository 95-row split, an unbound judge configuration, and a protected pair split across train and test. |

## Critical findings

### 1. The adaptation study's “confirmatory” verdict is not bound to the advertised estimand

This is the most important scientific issue.

The abstract and synthesis call the ten-checkpoint extension “preregistered confirmatory” and say it confirms specialization in purpose-built guards ([`unified_report.tex:120-123`](unified_report.tex#L120-L123), [`unified_report.tex:377-382`](unified_report.tex#L377-L382); PDF pp. 1, 38). The adaptation section itself discloses that the registry is `dev_nonfinal`, no release lock exists, and the same inspected 3,308 rows are reused ([`sections/act-adaptation.tex:7-16`](sections/act-adaptation.tex#L7-L16); PDF p. 19). The underlying normative contract is even more explicit: revisions remain `PIN_AT_PHASE0`, and “no field here is claim-bearing until ... `LOCK.json` binds it” ([`primary_contract.json:1-10`](../../artifacts/starting_type_adaptation_v1/protocol/primary_contract.json#L1-L10)). No such lock is present.

The absence of a lock is compounded by a different estimand in code:

- The protocol's RQ1 is over the complete **purpose-built** panel ([`proposal.md:128-150`](proposal.md#L128-L150)).
- `build_panel()` retains every checkpoint, and `compute_hypotheses()` averages all model families without filtering `starting_type` ([`analyze_starting_type_adaptation.py:135-205`](../../experiments/analyze_starting_type_adaptation.py#L135-L205), [`:258-305`](../../experiments/analyze_starting_type_adaptation.py#L258-L305)).
- Consequently, the printed `H_gain=0.17359`, `H_conc=0.23861`, and `H_held_sft=-0.06502` mix four general checkpoints with six purpose-built guards. The Qwen family even averages checkpoints from both starting types.
- A direct purpose-built-only recomputation from the committed scores changes the point estimates to `H_gain=0.11109`, `H_conc=0.18297`, and `H_held_sft=-0.07188`. A 1,000-resample diagnostic gave approximate lower bounds `0.0705` and `0.1359` for the first two, so the qualitative direction appears likely to survive. That diagnostic is reassuring, but it is not the registered 10,000-resample result and cannot validate the currently printed bounds.

The executable claim logic also does not consume the committed claim registry. The registry selects RQ1 wording using the held-out **point-estimate sign** ([`claim_registry.json:80-103`](../../artifacts/starting_type_adaptation_v1/protocol/claim_registry.json#L80-L103)); the analyzer substitutes an unregistered `UCB(H_held_sft)<0` predicate and different “bound-confirmed loss” wording ([`analyze_starting_type_adaptation.py:398-470`](../../experiments/analyze_starting_type_adaptation.py#L398-L470)). The registry and analyzer were both committed before the score commit, which is encouraging chronology, but two conflicting pre-result specifications are not an executable preregistration.

Other protocol-completion failures reinforce the problem:

- Every committed preflight record has `eligible:false`, because required beta-zero and trained-adapter smoke checks were skipped; see, for example, [`preflight_llama_guard_3_1b.json:91-134`](../../artifacts/starting_type_adaptation_v1/preflight/preflight_llama_guard_3_1b.json#L91-L134).
- Llama Guard is retained as a zero-valued family after its score proved constant, even though the protocol's missing-cell rule says a failed cell blocks the aggregate and the stop/narrow rules include unusable tied outputs ([`primary_contract.json:80-88`](../../artifacts/starting_type_adaptation_v1/protocol/primary_contract.json#L80-L88), [`claim_registry.json:171-180`](../../artifacts/starting_type_adaptation_v1/protocol/claim_registry.json#L171-L180)). Calling the inclusion conservative does not make the post-result treatment preregistered.
- The proposal requires achieved train/evaluation KL and a matched-achieved-KL sensitivity, because a fixed `beta=0.5` is not an equal intervention across checkpoints ([`proposal.md:12-17`](proposal.md#L12-L17), [`primary_contract.json:44-53`](../../artifacts/starting_type_adaptation_v1/protocol/primary_contract.json#L44-L53)). Those quantities are absent from the score schema and manuscript.
- The registered beta `1.0` sensitivity and full official-native-contract retention are not reported. Qwen3Guard's `Controversial` logit is explicitly discarded, so this is a harmonized binary-margin study, not validation that the released guard's native contract was retained.
- The manuscript says the bootstrap uses 3,170 row families over 3,308 rows. The actual analyzed `id_test + transfer_test` surface is 2,257 rows in 2,140 families, exactly as recorded in [`results.json:120-137`](../../artifacts/starting_type_adaptation_v1/analysis/results.json#L120-L137). The larger count includes calibration and stress rows that do not enter these H statistics.

**Required correction:** For this manuscript, relabel the extension everywhere as an **analysis-preregistered retrospective fixed-panel study** and replace the printed statistics with the purpose-built-only analysis. Programmatically load and hash-validate one finalized registry, resolve eligibility before analysis, report all protocol deviations, achieved KL, and native-contract retention, and regenerate every dependent sentence/table/figure. “Confirmatory” would require a new, genuinely uninspected cohort evaluated under a final lock; a retroactive lock cannot upgrade these already inspected results.

### 2. The one-command reproducibility headline is materially false

The abstract says every claim-bearing table is regenerated from committed per-row scores by one command ([`unified_report.tex:132-136`](unified_report.tex#L132-L136)). The reproducibility section says every table and figure is bound to a lock and implies figure byte identity ([`unified_report.tex:470-488`](unified_report.tex#L470-L488); PDF p. 41). The code and the section's own coverage count contradict that headline.

In an isolated checkout, `reproduce.py --check` exited nonzero. Ten TeX artifacts were genuinely byte-identical, four Paper A artifacts correctly required the lock-pinned Python environment, and eight claim-bearing TeX artifacts were not covered: adaptation, KL-SFT, ensembling, and mortgage composition. The script printed `11/22` as “byte-checked” only because it counted `figures='regenerated'` as verified ([`reproduce.py:384-422`](reproduce.py#L384-L422)).

The figure path is not a check: it overwrites figures and never compares them to committed bytes ([`reproduce.py:384-386`](reproduce.py#L384-L386)). The figure builder catches missing-input exceptions, prints `[skip]`, and still exits zero ([`figures/make_figures.py:510-518`](figures/make_figures.py#L510-L518)). Two included graphics are not built by that script at all. The mortgage and ExpGuard paths also ignore subprocess return codes and can compare stale pre-existing files after a producer fails ([`reproduce.py:85-126`](reproduce.py#L85-L126)). Finally, all in-prose numerical claims are explicitly outside the byte check.

This does not erase the genuine strengths: the Paper A release cache, hashes, lock, scores, analyzer source, and generated tables verify cleanly; mortgage checksums pass; and an independent Tectonic build produced the same 71-page text. It does mean the current manuscript must say **partial one-command coverage**, not “every claim-bearing table” or “every table and figure.”

**Required correction:** Make reproduction non-mutating in `--check` mode; write every candidate output to a temporary directory; check every subprocess return code; fail on skipped generators, stale outputs, uncovered claim artifacts, missing locks, and figure drift; hash the complete input/output graph; and generate prose numbers from macros or a machine-checked claim registry. Offer explicit `partial` and `full` modes if some pinned environments cannot be unified. The abstract should report the actual full-mode status, not the best covered subset.

### 3. The mortgage release is neither sealed nor provenance-complete

The label-construction account in PDF p. 57 does not match the committed executable path. The paper says the rubric-bound judge runs at temperature zero with a **three-sample majority vote** ([`sections/appendix-detail.tex:180-188`](sections/appendix-detail.tex#L180-L188)). The real-build entry point instead sets both generator and judge to `gpt-5.4-mini` and `judge_samples=1` ([`build_real_benchmark.py:23-40`](../../mortgage-benchmark/scripts/build_real_benchmark.py#L23-L40)). The judge does use temperature zero, but one sample is necessarily accepted as “consensus” ([`judge.py:100-110`](../../mortgage-benchmark/magen/judge.py#L100-L110), [`:130-141`](../../mortgage-benchmark/magen/judge.py#L130-L141)). The generator is explicitly told the target quadrant and policy card ([`generate.py:116-136`](../../mortgage-benchmark/magen/generate.py#L116-L136)), and a row is retained only when the judge reproduces the planner's target `G` and `D` labels ([`pipeline.py:66-86`](../../mortgage-benchmark/magen/pipeline.py#L66-L86)). The resulting golds are therefore **planner-target-consistent, single-judge-filtered labels**, not independently produced three-vote adjudications. Because the release manifest records only version, seed, counts, and an index path—not model snapshots, prompt/code hashes, vote count, or raw votes—it cannot establish that a different build produced the frozen rows ([`MANIFEST.json:1-11`](../../mortgage-benchmark/benchmark/v1_hmda2022/MANIFEST.json#L1-L11)).

The split contract also fails for one protected pair. `PAIR-0000#1` has its reference member in train and protected member in public test ([`train.jsonl:549`](../../mortgage-benchmark/benchmark/v1_hmda2022/train.jsonl#L549), [`public_test.jsonl:140`](../../mortgage-benchmark/benchmark/v1_hmda2022/public_test.jsonl#L140)). Splitting groups only by `content_family`, which differs across the two arms ([`split.py:15-57`](../../mortgage-benchmark/magen/split.py#L15-L57)); validation checks variants and benign labels but not co-location ([`validate.py:54-67`](../../mortgage-benchmark/magen/validate.py#L54-L67)); evaluation silently uses only pairs complete within the evaluated split ([`evaluate.py:132-142`](../../mortgage-benchmark/magen/evaluate.py#L132-L142)). This does not change the current zero-shot AP-D calculation, but it invalidates the advertised family-isolated release for future training/tuning and silently removes the disability pair from the public fairness probe.

PDF p. 29 also calls the 95-row `private_test` sealed and held back. It is tracked in full, including plaintext `user_prompt`, at [`private_test.jsonl`](../../mortgage-benchmark/benchmark/v1_hmda2022/private_test.jsonl). The data card says the file is not in the bundle and only a text-free index is distributed ([`DATA_CARD.md:14-25`](../../mortgage-benchmark/benchmark/v1_hmda2022/DATA_CARD.md#L14-L25)), while the adjacent README admits that it is committed and merely asks readers not to tune on it ([`benchmark/README.md:10-15`](../../mortgage-benchmark/benchmark/README.md#L10-L15)). It is unused by the present public-test result, but it cannot be a future sealed or confirmatory cohort. The same data card states `LICENSE NOT YET SELECTED` ([`DATA_CARD.md:27-28`](../../mortgage-benchmark/benchmark/v1_hmda2022/DATA_CARD.md#L27-L28)), contradicting the manuscript's unqualified statement that the benchmark and data are public and ready to ship.

**Required correction:** Rebuild under a release-bound construction manifest that records separate versioned generator/judge identities, exact prompts and code hashes, the promised vote count, raw votes, and acceptance statistics. Group split assignment by both content family and protected-pair identity, then validate pair co-location. Relabel the current private split as an additional public holdout and obtain a genuinely custodian-held future cohort. Select a redistribution license and reconcile the data card, README, generated table, and paper before claiming a public benchmark release.

### 4. The prevalence analysis is neither tie-aware nor deployment-measured

The prevalence section calls its curves “measured,” “exact,” and suitable for deployment precision (PDF pp. 17-18; [`sections/act1.tex:248-284`](sections/act1.tex#L248-L284)). Reweighting an empirical ROC to a new class prior is exact only under prior-probability shift: the class-conditional score distributions, `P(score | Y)`, must remain unchanged between evaluation and deployment. That assumption is not stated and is especially strong in a paper whose thesis is distribution dependence. The figure is a **counterfactual label-shift sensitivity**, not measured deployed AP.

The implementation also violates the repository's canonical tie-aware metric. It sorts individual rows and integrates each positive in row order ([`figures/make_figures.py:135-145`](figures/make_figures.py#L135-L145)), while [`guard_research/metrics.py:6-12`](../../guard_research/metrics.py#L6-L12) explicitly warns that this makes AP depend on arbitrary ordering inside score ties. At 1% prevalence, the current code gives Qwen2.5 `0.109856`; grouping ties through weighted canonical AP gives `0.099441`. Across random within-tie orderings, the value ranges roughly `0.0995-0.1167`. The paper-wide tie-aware contract therefore does not cover Figure 5.

**Required correction:** Compute target-prior AP with the canonical tie-aware metric using class weights (`pi/P` for positives and `(1-pi)/N` for negatives), add a unit test with tied scores and row permutations, label the curve a prior-shift sensitivity, and state the stable-class-conditional assumption. Update the Qwen2.5 1% value from approximately `0.11` to approximately `0.10` under the canonical method.

## Major scientific and statistical findings

### 5. The mortgage construct is useful, but “independent” and “orthogonal” are incorrect

The manuscript repeatedly calls `G` and `D` independent labels (PDF pp. 28, 34, 51; [`sections/act4-mortgage.tex:20-32`](sections/act4-mortgage.tex#L20-L32), [`unified_report.tex:256-259`](unified_report.tex#L256-L259), [`sections/appendix-methods.tex:263-280`](sections/appendix-methods.tex#L263-L280)). Yet `G1/D0` is empty, `G` is nested in `D`, and `final = G OR D` equals `D` row-for-row. “Orthogonality ... demonstrated on three quadrants” ([`sections/act4-mortgage.tex:66-73`](sections/act4-mortgage.tex#L66-L73)) is mathematically wrong: a missing quadrant is evidence that orthogonality has not been demonstrated.

The accurate phrase is **“separately assigned labels that are empirically nested in v1.”** The dual annotation still adds value because it isolates G0/D1, but it does not yet supply a crossed specificity/generality design, and the derived `final` label adds no new target.

Other construct-validity limits should move from the appendix into the main result:

- The appendix says all 39 protected pairs are identical except for one token, but the main text correctly says only 21 are single-token swaps and 18 use a multiword placeholder. Only three complete pairs occur in public test—two race and one religion—and all release pairs are benign G0/D0 by construction ([`sections/act4-mortgage.tex:78-97`](sections/act4-mortgage.tex#L78-L97), [`validate.py:54-67`](../../mortgage-benchmark/magen/validate.py#L54-L67)). This is a protected-attribute **score-sensitivity probe**, not a fairness gate, and it says nothing about sensitivity on rows where intervention is appropriate.
- The build draws 3,000 source rows from 12 selected small states and samples fields independently from their marginals ([`build_real_benchmark.py:31-35`](../../mortgage-benchmark/scripts/build_real_benchmark.py#L31-L35), [`sections/appendix-detail.tex:96-114`](sections/appendix-detail.tex#L96-L114)). Call these **HMDA-marginal-informed synthetic scenarios**. They preserve selected univariate frequencies, not real joint applicant profiles, traffic prevalence, or empirical relationships among state, income, DTI, LTV, and outcome.
- `G` is defined partly as what an ordinary guard “would call unsafe,” while an LLM judge assigns both labels. This is a benchmark-local construct, not an independently observed general-safety truth.
- At least one D-positive row, `MGB-FL-00037`, carries `policy_context:["None"]` while its rationale cites D02, D03, D05, and D07 ([`train.jsonl:270`](../../mortgage-benchmark/benchmark/v1_hmda2022/train.jsonl#L270)). The schema requires only a nonempty list, so the sentinel string passes validation ([`schema.py:139-141`](../../mortgage-benchmark/magen/schema.py#L139-L141)). This is a concrete failure of policy-label provenance, not merely a missing documentation field.
- Statements such as “honoring it would commit ... mortgage-law violations” exceed the policy cards' own honesty contract, which says a row violates a benchmark card and is never asserted to be illegal ([`cards.yaml:1-10`](../../mortgage-benchmark/policy_cards/cards.yaml#L1-L10)). Use “would violate the benchmark policy rubric” unless counsel has adjudicated the row.

The legal source layer also needs temporal versioning. The policy cards were reviewed on 2026-07-14. A CFPB Regulation B final rule effective 2026-07-21 states that ECOA does not authorize effects-test liability and narrows the discouragement standard; the current Fair Housing Act regulation separately retains discriminatory-effects liability. See the [CFPB final rule](https://www.federalregister.gov/documents/2026/04/22/2026-07804/equal-credit-opportunity-act-regulation-b) and [24 CFR 100.500](https://www.ecfr.gov/current/title-24/subtitle-B/chapter-I/part-100/subpart-G/section-100.500). Card D07 appropriately points to FHA and marks the doctrine contested, but the release must freeze dated source text/effective dates and receive SME/counsel signoff before any “high-compliance” claim.

**Required correction:** Replace independent/orthogonal language, remove the redundant final-target framing, fix the appendix's pair description, validate actual policy-card IDs, and rename the current construction as HMDA-marginal-informed synthetic data. Freeze effective legal-source snapshots with hashes and jurisdiction/effective dates, then SME-adjudicate a stratified subset with agreement reporting. Populate G1/D0 and protected D=1 counterfactual pairs before claiming a crossed compliance instrument.

### 6. The title and abstract merge three different evidence blocks into one stronger story

The abstract says “four findings on one fixed panel,” but the report combines:

1. a retrospective four-checkpoint base/SFT/composition study on inspected rows;
2. an unlocked, analysis-preregistered ten-checkpoint adaptation extension on the same rows; and
3. separate four-base, zero-shot mortgage and ExpGuard probes with different label provenance.

Those are not one panel or one evidence tier. The evidence ledger does not include the adaptation extension at all ([`sections/limitations-validation.tex:47-78`](sections/limitations-validation.tex#L47-L78)), even though it carries the strongest headline. The ledger also says base-only ExpGuard tests whether the specialization/transfer pattern recurs; without tuned scores, ExpGuard cannot measure a base-to-SFT specialization effect.

The subtitle “Measuring, Tuning, and Composing ... in High-Compliance Regulated Domains” implies that tuning and composition were tested in those domains. They were not: the mortgage and ExpGuard arms are base-only zero-shot, and the tuned domain comparisons remain future work.

**Recommended title:**

> **Choosing Small Safety Guards Under Distribution Shift: Paired Fine-Tuning Evidence, Composition, and Regulated-Domain Probes**

A defensible abstract nucleus would be:

> On a fixed four-checkpoint panel, ordinary LoRA-SFT strongly improves ranking on represented sources but has heterogeneous transfer effects; calibrated base-plus-adapter averaging partially recovers transfer relative to SFT. An analysis-preregistered but unlocked extension shows similar descriptive movement in released guards, while separate base-only domain probes show that numerical rankings depend on the evaluation construct.

Rewrite the abstract, scope box, synthesis, conclusion, and ledger around these three explicit evidence blocks. Do not let the external expert label tier lend confirmatory status to the retrospective analyses.

### 7. Several statistical statements turn non-rejection into equivalence or description into mechanism

The ExpGuard paired intervals support a health difference, but they do not establish “genuine ties” in finance and law ([`generated/expguard_table.tex:13`](generated/expguard_table.tex#L13), [`unified_report.tex:285-298`](unified_report.tex#L285-L298)). Finance `[-0.0075, 0.0097]` and law `[-0.0118, 0.0140]` are unresolved differences around zero. To claim a tie, preregister a substantively justified equivalence margin and use an equivalence test or show the appropriate CI lies wholly within that margin.

The reported health contrast is also exploratory: the observed top two overall guards are selected first, then overall plus three domain intervals are inspected without post-selection or multiplicity correction. ExpGuard has no committed family/dependence graph, so the row bootstrap additionally assumes independent rows. State those limitations and define the contrast family before treating any domain-specific interval as confirmatory.

The mortgage section makes a related error when it concludes that only Qwen3-4B versus SmolLM2-1.7B separates because five of six **marginal** AP-D intervals overlap ([`sections/act4-mortgage.tex:176-181`](sections/act4-mortgage.tex#L176-L181)). Marginal-interval overlap is not a paired contrast on identical rows. A diagnostic 2,000-resample paired row bootstrap from the committed scores found three unadjusted contrasts whose 95% intervals excluded zero: Qwen2.5-SmolLM2 `[0.0317, 0.2052]`, Qwen3-SmolLM2 `[0.0737, 0.2779]`, and Qwen3-SmolLM3 `[0.0485, 0.1893]`. Those intervals are diagnostic, not publication-ready—they still need the benchmark's dependence unit and multiplicity control—but they prove that the present “only one pair” inference does not follow from the data. Replace marginal-overlap reasoning with prespecified paired, family-aware, multiplicity-adjusted contrasts.

The paper also repeatedly explains observed patterns with unmeasured mechanisms:

- Act I says SFT learns surface cues and “forgets the hardest cases first” ([`sections/act1.tex:70-92`](sections/act1.tex#L70-L92), [`:118-125`](sections/act1.tex#L118-L125)). The experiment measures score movement by benchmark; it does not identify learned cues or forgetting dynamics.
- Act II says correlated errors explain why a strong base benefits least and that both acts are one mechanism ([`sections/act3.tex:177-198`](sections/act3.tex#L177-L198)). Error correlations and the SFT+SFT control are consistent with a diversity account, but AP is nonlinear, adapter distance is not measured, and four checkpoints do not identify the claimed mechanism.
- The equal-cost control supports “base+SFT outperforms SFT+SFT on this panel,” not the causal wording “the recovery is attributable to keeping the base” in a general sense.
- The appendix correctly notes that regressing `SFT-base` on `base` mechanically induces a slope near -1 when SFT endpoints are narrow, but then promotes the shared endpoint into “the benchmark and manifest largely choose the score” ([`sections/appendix-detail.tex:27-49`](sections/appendix-detail.tex#L27-L49)). Four checkpoints from two lineages under one optimizer/recipe cannot assign that endpoint to the benchmark rather than the training algorithm, hyperparameters, or sampled models. Treat the “attractor” as a hypothesis and vary manifests, recipes, and model lineages before making that attribution.

This conflicts with the limitations section's correct statement that no mechanism is isolated ([`sections/limitations-validation.tex:13-18`](sections/limitations-validation.tex#L13-L18)). Keep mechanistic passages explicitly as hypotheses and add direct tests—representation/score-shift analyses, controlled diversity interventions, or ablations—before promoting them.

### 8. The composition result is useful but should remain a retrospective operator study

The composition implementation has strong score provenance, and its comparisons are valuable. However, its metadata says the primary combiner and bootstrap settings are `fixed_prototype_constants_not_paper_a_lock` ([`composition_metadata.json:1-10`](../../artifacts/paper_a_sft_v2/analysis/composition/composition_metadata.json#L1-L10)). The text's “fixed before looking at any transfer result” claim is therefore not independently lock-verifiable. Call it a fixed retrospective prototype, which is already compatible with the paper's overall evidence tier.

Two technical clarifications are needed:

- The actual calibrator is positive temperature scaling, which is strictly monotone. The background's isotonic example does not justify “AP is unchanged by a monotone calibrator,” because non-strict isotonic mapping can create ties and change tie-aware AP ([`sections/act3.tex:42-60`](sections/act3.tex#L42-L60)). State the property for the calibrator actually used.
- “Roughly doubling inference cost” is plausible but not measured for the composed serving path; the latency table measures batched per-row single-pass A100 timings. Report composition latency directly before making service-level claims.

### 9. Practitioner and deployment prescriptions exceed the measured scope

The paper says it is not a deployment recommendation ([`unified_report.tex:238-240`](unified_report.tex#L238-L240)) but provides a professional “what to do” table and recommends self-hosting over hosted APIs ([`unified_report.tex:320-384`](unified_report.tex#L320-L384), [`:424-463`](unified_report.tex#L424-L463)). The hosted-API latency, pricing, data-residency, rate-limit, and silent-update columns are not measured or cited. “Commodity GPU (or CPU)” is unsupported by the A100 batch-16 measurement.

Recast these as candidate validation practices rather than conclusions established by this experiment. Remove the hosted comparison or support every time-sensitive claim with dated primary vendor/system evidence and a controlled benchmark. Data residency depends on deployment and contract, not simply “hosted” versus “self-hosted.”

### 10. Novelty and citation claims need tighter boundaries

The paired evaluation discipline is the most defensible novelty. Absolute claims such as “almost without exception” and “none” in the related-work section require a systematic search protocol, not a selective narrative review ([`sections/related-work.tex:43-55`](sections/related-work.tex#L43-L55)). Output-space calibrated ensembles are already established; the contribution is their targeted application and controlled comparison in this guard setting, not the operator itself.

Use “we did not identify prior work combining ...” and document databases, query dates, and inclusion criteria if novelty is load-bearing. Normalize model-card URLs into `url` fields with access dates, add stable locators/DOIs where available, and cite or remove the hosted-API comparison. No unresolved LaTeX citations were visible, which is a positive baseline.

## Internal consistency issues to fix

- The report calls the adaptation study confirmatory while its own limitation says `dev_nonfinal`, unlocked, and not data-blind.
- The reproducibility section calls the eight uncovered artifacts outputs of “locked analyses,” but adaptation has no lock.
- The introduction says the manifest is decontaminated; the limitations say the formal v2 overlap audit is still pending. Use “family-isolated under the builder; formal cross-suite audit pending.”
- The shared-methods appendix says every study uses the same four checkpoints, training manifest, scorer, and fail-closed locks; adaptation uses ten checkpoints, the domain probes do not use the training manifest, native output contracts differ, and several analyses have no lock ([`sections/appendix-methods.tex:5-14`](sections/appendix-methods.tex#L5-L14)).
- The methods appendix describes all 39 protected pairs as one-token swaps; the main result says only 21 qualify.
- The mortgage appendix says three judge votes; the committed real-build path requests one, and the manifest cannot resolve the contradiction.
- The mortgage splitter claims family isolation while one protected/reference pair crosses train and public test.
- The mortgage data card says the private split is absent; the file and README say it is committed.
- The mortgage README says no guard has been scored, while the manuscript reports four baselines. Update stale release documentation.
- The generated mortgage table calls raw log-odds “scale-free,” while the main text correctly says cross-model logit scales are not guaranteed comparable ([`generated/mortgage_baseline_table.tex:5`](generated/mortgage_baseline_table.tex#L5), [`sections/act4-mortgage.tex:118-126`](sections/act4-mortgage.tex#L118-L126)).
- The report uses 3,170 adaptation bootstrap families where the result artifact records 2,140 analyzed families.
- “Finance and law are tied” is stronger than the reported intervals.
- “Every number” and “every figure” are stronger than the reproduction graph.

## Presentation and document engineering

The compiled PDF is readable, visually consistent, and free of unresolved references, but it is not shaped for a target venue.

- At 71 pages, with related work beginning after the conclusion on p. 43, this is a technical report rather than a conventional article. Either label it explicitly as such or choose a venue and move tutorials, per-seed tables, the full evidence ledger, construction details, and operational guidance into a supplement.
- Forced page breaks produce visibly underfilled pages, especially pp. 2, 40, 42, and 46. Remove global section-clearing behavior and let floats settle naturally.
- PDF p. 23 begins with an orphaned one-line continuation; Figure 1 is crowded; the p. 34 mortgage table caption overwhelms its table; and a heading breaks awkwardly around p. 16.
- The five-line conclusion does not synthesize adaptation status, composition limits, domain label tiers, or the negative fairness result. It should mirror the three evidence blocks used in the revised abstract.
- Two embedded DejaVuSans fonts are Type 3, and `pdfinfo` reports `Tagged: no`. Replace the Type 3 figure fonts, add document metadata/accessible tagging where the toolchain permits, add alt-text equivalents in the source, and fix `\today` to a release date for stable archival builds.

## What is already strong

1. **Paired estimand.** Comparing each tune with its own base on identical rows directly answers a better question than a cross-model leaderboard.
2. **Regime separation.** Represented, transfer, calibration, and stress surfaces are kept conceptually distinct.
3. **Conditional uncertainty.** The Paper A family-aware bootstrap and fixed-panel caveats are unusually explicit.
4. **Ranking versus thresholding.** The manuscript correctly emphasizes that AP recovery does not imply threshold transfer.
5. **Negative-result honesty.** The protected-pair analysis retracts its own tempting guard ranking after scale and sample checks.
6. **Evidence artifacts.** Paper A's release cache and the mortgage checksums are substantive, verifiable assets.
7. **Useful controls.** SFT+SFT and the base-relative composition breakdown materially improve the composition story, even though they do not prove the proposed mechanism.

These strengths should become the paper's center. The current grander framing hides them behind claims that are easier to challenge.

## Required revision sequence

### P0 — before any submission

1. Remove “confirmatory” from the current adaptation evidence; rerun the stated purpose-built estimand and regenerate all dependent claims.
2. Reconcile and executable-bind the adaptation contract, registry, eligibility rule, analyzer, revisions, adapter hashes, score hashes, and output hashes. Document every deviation.
3. Replace the abstract's complete-reproducibility claim with the exact current coverage, or make a fail-closed full harness that truly covers every claim-bearing table, figure, and prose value.
4. Rebuild or relabel the mortgage release: reconcile one-versus-three judge votes, bind model/prompt/vote provenance, keep protected pairs within splits, relabel the committed `private_test`, select a license, and correct the data card/README.
5. Fix prevalence reweighting and its label-shift interpretation.

### P1 — scientific revision

6. Rewrite the mortgage construct as separately assigned but nested; remove independent/orthogonal language and legally authoritative wording; validate policy-card identifiers and narrow the HMDA-grounding claim.
7. Add the adaptation extension to the evidence ledger and stop saying base-only ExpGuard replicates specialization.
8. Replace “ties” with unresolved differences unless an equivalence analysis is added, and replace mortgage marginal-CI overlap with paired family-aware contrasts.
9. Demote mechanism language to hypotheses; keep the measured comparisons as results.
10. Narrow the title, abstract, synthesis, recommendations, and conclusion to the three actual evidence blocks.

### P2 — release and presentation

11. SME/counsel-review a stratified mortgage subset and version the effective legal sources.
12. Add a genuinely sealed acceptance cohort if future confirmation is intended.
13. Choose a venue/technical-report format, shorten the main narrative, repair sparse pages and dense captions, remove Type 3 fonts, and improve PDF accessibility.
14. Normalize bibliography metadata and add evidence for any retained hosted-service claims.

## Verification record for this review

- Inspected the 71-page PDF and extracted text page by page.
- Audited the TeX sources, generated tables/macros, analysis code, locks/contracts, score metadata, mortgage data card, policy cards, and release documentation.
- Independently rebuilt the PDF with Tectonic in an isolated copy; the extracted text matched the committed PDF.
- Ran the unified reproduction check in an isolated copy: 10 TeX artifacts byte-matched, four required the Paper A pinned environment, eight were uncovered, and the check exited nonzero.
- Verified the Paper A release/hash contract and the mortgage benchmark checksums/counts.
- Audited the committed mortgage generator/judge configuration and acceptance rule, enumerated complete and cross-split protected pairs, and checked row-level policy-card provenance.
- Recomputed canonical tie-aware AP under 1% prior shift, the purpose-built-only adaptation point estimand, and paired mortgage AP-D contrasts; diagnostic bootstrap results are reported above only to assess whether identified bugs change the interpretation.
- Checked the current mortgage-law framing against official 2026 federal sources. This review is a research/engineering audit, not legal advice, and it does not substitute for SME or counsel adjudication.

## Final recommendation

Do not submit this version with the current title, confirmatory label, or reproducibility headline. Preserve the retrospective paired study, the composition comparison, and the negative fairness-instrument result; they are the manuscript's strongest contributions. After the P0 corrections, the work could become a credible technical report and, with substantial compression and a genuinely locked prospective extension, a strong empirical paper.
