# Technical Re-Review: *The Benchmark Chooses the Winner*

Review date: 2026-07-12  
Current revision: `1d00f9d` (`main`, also `paper/compliance-reframe-and-review-fixes`)  
Prior reviewed revision: `3123fd4`  
Scope: current manuscript source and rendered PDF, included TeX files, experiment and training scripts, local ignored outputs, a clean `git archive`, benchmark bundle, reproduction notebook, and current related work.

## Recommendation

**Reject in the current form; invite a focused major revision.**

The revision makes several important and correct changes. The primary SmolLM3, novel-benchmark, and mortgage AP values are now tie-aware; the mortgage base-versus-SFT winner claim has been retracted; OR-Bench's family overlap is disclosed in several places; the GPT result is usually described as inconclusive rather than equivalent; and the related-work discussion is more candid about limited novelty.

Those improvements do not yet close the earlier review. The manuscript now makes new clean-split, three-seed, objective, ExpGuard, and name-fairness claims without versioned per-run artifacts that allow them to be checked. The current workspace contains **zero** `summary_*clean-*-s*.json` files and zero clean adapters, `aggregate_clean_sweep.py` finds no completed cells, and a clean `git archive` contains no outputs or adapters. The clean trainer still retains four exact Jailbreak-Classification-to-WildJailbreak overlaps with conflicting labels. HPO still evaluates the final novel set in every trial and still uses the order-dependent AP helper. The new fairness bootstrap is not a valid bootstrap. The paper also remains much broader than its strongest evidence: approximately 16,500 source words, a 355-word abstract, 39 section-level headings, 21 table environments, nine figures, and at least seven distinct research stories.

The publishable core remains valuable but should be narrower:

> **A controlled, checkpoint-matched estimate of what LoRA-SFT adds on represented benchmark sources and what transfers to dataset-held-out safety benchmarks.**

The operating-point study should become supporting methodology, not the claimed novelty. Mortgage compliance should move to Paper B. DPO/GRPO/HPO, ensembling, the name audit, GPT parity, and the guardrail landscape should not remain co-equal main-paper contributions.

## Overall verification result

| Area | Status | Current conclusion |
|---|---|---|
| Primary tie-aware metrics | **Partially resolved** | Main paths and headline values were corrected, but several active producers still use order-dependent AP/AUROC. |
| Clean held-out design | **Unresolved** | OR-Bench can be disabled, but four conflicting WildJailbreak overlaps remain and no executed near-duplicate audit or immutable manifest exists. |
| HPO/test isolation | **Unresolved** | The limitation is disclosed, but the final novel set is still scored every trial and the paper still draws substantive conclusions from it. |
| Operating-point claim | **Partially reframed** | The scoring-rule confound is disclosed, but the title, contribution, and figure still describe a cross-metric/cross-score reversal as an operating-point result. |
| Multi-seed evidence | **Not verifiable** | Driver scaffolding exists; the claimed clean per-seed outputs, scores, adapters, and generated table artifact do not. |
| Statistical units | **Unresolved** | Most intervals still resample rows rather than source/family clusters and do not combine seed and evaluation uncertainty. |
| Pooling/prevalence | **Partially resolved** | In-house macro AP and limitations were added; transfer macro AP and prevalence/cost analysis are still absent. |
| Mortgage construct validity | **Partially corrected** | The result is narrowed to an illustrative case study, but it remains abstract/contribution/conclusion-level evidence. |
| GPT comparison | **Partially corrected** | Non-equivalence is acknowledged, but the run remains unpinned and prose still uses “tie” and “parity.” |
| Dataset/training documentation | **Partially corrected** | ToxicChat, BF16, and verdict-plus-EOS descriptions improved; revisions, hashes, exact manifests, and prompt-label validation remain absent. |
| Fresh-clone reproduction | **Unresolved** | No result artifacts/adapters are tracked or immutably released; the checked-in notebook and expected values are stale. |
| Narrative focus | **Unresolved; worsened** | New ExpGuard and fairness arcs were added without removing older side stories. |
| Novelty positioning | **Improved but incomplete** | The paper now says novelty is limited-to-moderate, but omits several closest 2026 guard-specialization and preservation papers. |

## Verification performed

The following checks were performed against the current worktree:

- Compared `3123fd4..1d00f9d` by commit and file diff.
- Read the complete current manuscript and included measured-guard tables.
- Inspected every modified metric, training, HPO, sweep, external-domain, and fairness script.
- Searched every Python producer for custom AP/AUROC implementations.
- Parsed all 31 experiment Python files with `ast.parse`: no syntax errors.
- Ran `bash -n` over the shell scripts: no shell syntax errors.
- Ran `git diff --check`: no whitespace errors.
- Ran `experiments/aggregate_clean_sweep.py`: it found no clean sweep or HPO cells.
- Built a clean `git archive`: it contained zero output files, zero adapters, and no tests.
- Recomputed the Jailbreak-Classification/WildJailbreak exact intersection from the current Hugging Face training split and frozen evaluation rows: four exact overlaps, all four with conflicting labels.
- Recomputed the hardened mortgage tie-aware AP from current cached scores: base `0.8900`, Mortgage-SFT `0.8950`, general guard `0.8203`.
- Compiled the current TeX with Tectonic 0.15.0: exit status 0, 23 pages, but with a six-pass rerun/internal-consistency warning and a 31.8 pt overfull line in the ExpGuard paragraph.
- Visually inspected the rendered first page and the affected ExpGuard/cross-family pages.

The checked-in PDF is stale relative to the source: `paper/benchmark_chooses_the_winner.pdf` is 22 pages, while a fresh compile of the current TeX is 23 pages and has a different SHA-256.

## What the revision implemented correctly

### 1. The primary tie bug was recognized and its largest consequence was corrected

The following paths now use `sklearn.metrics.average_precision_score` and tie-correct AUROC:

- `experiments/eval_corrected.py:118-131`
- `experiments/guard_eval_pipeline.py:55-64`
- `experiments/eval_mortgage_hard.py:39-48`
- `experiments/verify_novel.py:34-38`
- `experiments/ensemble_deployable.py:33-37`
- `experiments/expguard_eval.py:18`

The paper now reports the corrected primary values:

- in-house guard AP: `0.833`, not `0.844`;
- novel base/guard/Llama AP: `0.884/0.780/0.685`, not `0.886/0.781/0.701`;
- mortgage base/SFT/general AP: `0.890/0.895/0.820`, not `0.892/0.924/0.824`.

Most importantly, `paper/benchmark_chooses_the_winner.tex:829-848` retracts the unsupported mortgage winner change and reports the paired interval crossing zero. This is a substantive and correct scientific correction.

### 2. Several terminology and disclosure problems were improved

The manuscript now:

- distinguishes ordinary single-system bootstrap intervals from paired effect intervals at `paper/benchmark_chooses_the_winner.tex:364-369`;
- says McNemar tests prediction discordance rather than equality of F1;
- discloses realized test FPR after dev matching;
- states that OR-Bench-Hard is a held-out configuration rather than a source-family-held-out dataset in several sections;
- describes GPT as a prompted fixed rule and explicitly says the confidence interval is not an equivalence test;
- correctly describes SFT loss as verdict plus EOS;
- correctly says ToxicChat is used in training;
- correctly describes BF16 model weights;
- moves the rank-preservation argument to an appendix and labels it a rationale rather than empirical proof.

### 3. The related-work tone is more credible

`paper/benchmark_chooses_the_winner.tex:125-138` now acknowledges that calibration, matched-FPR evaluation, regime dependence, base-model competence, KL-regularized preservation, and calibrated ensembling have prior art. Calling the current novelty “limited-to-moderate” is more defensible than the previous “two genuinely new findings” framing.

### 4. Some repository path problems were fixed

- `experiments/build_paper_reproduction_notebook.py` now invokes `experiments/` rather than the removed `scripts/` directory.
- `experiments/run_extra_models.sh` is now repo-relative.
- `paper/Makefile` and the paper/notebook READMEs now use `experiments/` paths.
- Optuna was added to the requirements.

These are useful repairs, although they do not yet make the numerical artifact reproducible.

## Remaining blocking correctness and validity issues

### 1. “All metrics are tie-aware” is still false

The revision corrected several prominent scripts but did not centralize the metric implementation. Active or cited producers still contain the original order-dependent AP implementation:

- `experiments/emit_inhouse_auprc_poolings.py:17-25` — direct producer for the in-house pooling table;
- `experiments/hpo_guard.py:54-57` — producer for the HPO result that the paper calls tie-aware;
- `experiments/ensemble_probe.py:31-37` — still cited for the light-blend result at manuscript line 779;
- `experiments/eval_novel_gaps.py:94-99`;
- `experiments/eval_expanded_heldout.py:107-114`;
- `experiments/eval_llamaguard_logprob.py:37-43`;
- `experiments/eval_mortgage.py:44-52`;
- `experiments/eval_mortgage_tuned.py:33-40`;
- `experiments/diag_base_id_ood.py:12-15`.

This directly contradicts `paper/benchmark_chooses_the_winner.tex:855`, which says all AP/AUROC values are now tie-aware. There is no canonical `metrics.py`, no tie-permutation unit test, and no test directory.

The checked-in reproduction notebook is also stale. It still contains the old `scripts/` invocation even though its generator was fixed, and its validation cell expects `0.886/0.781/0.701` and mortgage `0.924`. The ±0.03 tolerance is so broad that the corrected mortgage result `0.895` would falsely pass against the obsolete `0.924` expectation.

**Required change:** create one canonical metric module; replace every custom implementation; add permutation tests proving that row order within score ties cannot change AP/AUROC; regenerate every summary, table, notebook, figure, and prose value from that module.

### 2. The new “clean, source-family-disjoint, uncontaminated” claim is not true

`experiments/stage2.sh` sets `OR_BENCH_CAP=0`, which removes the obvious OR-Bench family overlap. That does not make the complete training/evaluation pair decontaminated.

Both trainers build `eval_keys()` only from the in-house sources:

- `experiments/train_guard.py:48-56`
- `experiments/train_guard_pref.py:58-65`

They never compare training rows against WildGuardTest, WildJailbreak, OR-Bench-Hard, or HarmBench. Comparing the current `jackhhao/jailbreak-classification` train split with the frozen WildJailbreak rows finds four exact prompts. Every one is labeled `unsafe/jailbreak` in training and `safe` in WildJailbreak evaluation. Setting `OR_BENCH_CAP=0` does not remove them because the other source selection is unchanged.

The paper is also internally contradictory:

- line 317 says the novel sets are source-family held out “with near-duplicate filtering”;
- line 383 correctly admits only normalized exact filtering and the OR-Bench exception;
- lines 694, 698, 731, 754, and 855 then call the new sweep “clean,” “source-family-disjoint,” “uncontaminated,” and regenerated from scratch.

No executed near-duplicate report, immutable row manifest, source revision, or train/evaluation hash ledger exists.

**Required change:** construct one immutable training manifest, remove all exact and global near-duplicate families before any training, publish the four removed rows and all audit counts, hash every row, and fail the run if any train/calibration/test family intersects. Until then, use “dataset-held-out with a known overlap audit pending,” not “uncontaminated source-family OOD.”

### 3. The clean three-seed tables are unsupported by auditable artifacts

The paper makes completed-result claims at `paper/benchmark_chooses_the_winner.tex:694-770`:

- every SFT and objective cell is the mean of seeds 42/43/44;
- the split is clean and source-family-disjoint;
- seed spread is tight;
- training stochasticity has been measured directly.

The current evidence does not allow any of those numerical claims to be verified:

- local `summary_*clean-*-s*.json` count: **0**;
- tracked clean summary count: **0**;
- tracked clean adapter count: **0**;
- `experiments/aggregate_clean_sweep.py` reports no cells for every model/objective;
- the clean archive contains no `notebooks/outputs`, `outputs`, adapters, or compact score artifacts;
- the aggregator only prints values; it does not emit a versioned table/analysis artifact.

`experiments/stage2.sh` is scaffolding, not a reproducible completed-run record:

- line 7 hard-codes `cd ~/guard`, which is not this checkout;
- its base list omits the primary SmolLM3 checkpoint entirely;
- its comment says the default skips Qwen3-8B, while line 18 includes it;
- it lacks `set -euo pipefail`;
- piping training/evaluation through `tail` can mask a failing producer;
- it can print `SUMMARY_DONE` and `STAGE2_ALL_DONE` without proving the expected summary exists;
- it records no data/model/tokenizer/prompt/config hashes.

This does not prove the remote runs never occurred. It proves that the current paper's main new evidence is not auditable from the repository or workspace supplied for review.

**Required change:** release, at minimum, one manifest and one compact artifact per `(checkpoint, objective, seed)` containing source/model revisions, row hashes, config, adapter hash, per-row scores, and summary metrics. Generate manuscript tables directly from these files and make missing cells a hard failure.

### 4. HPO leakage is acknowledged but not fixed

`experiments/hpo_guard.py` still loads the final novel set at lines 33-38 and scores it in every trial at line 98. It stores and prints the result for every trial, deletes the adapter at line 99, and never performs the promised full-step locked retrain. Its AP implementation at lines 54-57 is tie-incorrect.

The HPO invocation also does not set `OR_BENCH_CAP=0` in the trial environment at line 90, so the documented default still includes OR-Bench training rows even though Table 14 is described as clean.

The manuscript candidly calls HPO exploratory, but still uses it to argue that:

- learning rate “is the OOD knob”;
- ordinary in-distribution selection “steers SFT straight” into collapse;
- DPO is Pareto-competitive;
- KL regularization makes OOD robustness the default.

Those are substantive findings drawn from the repeatedly inspected test set, not harmless diagnostics.

**Required change:** either remove HPO from Paper A or use nested partitions: optimize only on represented-source development data, lock the winner, retrain at the final step budget across seeds, then touch an untouched transfer test exactly once. Preserve the selected adapters and configs.

### 5. The objective mechanism is not identified

The paper says “only the objective varies” and concludes that the objective controls OOD convergence. The code varies much more:

- SFT: right padding, batch 1 × accumulation 4, learning rate `2e-4`;
- DPO: left padding, batch 2 × accumulation 4, default learning rate `5e-6`;
- GRPO: online groups, group batch 8 × accumulation 2, default learning rate `1e-6`, sampled completions and reward parsing.

Thus SFT uses a learning rate 40× DPO's and 200× GRPO's, while compute, data exposure, batch structure, and loss geometry also differ. GRPO's result is almost identical to the base both in-house and OOD, which is at least as consistent with a near-no-op/reward-starved update as with successful robustness preservation.

There is no matched-update norm, matched token exposure, beta-zero control, direct pair-rank-drift measure, or mediation analysis connecting reference anchoring to preserved OOD ranking.

**Required change:** describe these as three training **recipes**, not an isolated objective comparison. For Paper A, the cleaner choice is to remove DPO/GRPO from the main paper. If retained later, match update budget/exposure, include beta and update-norm controls, report base-distance/rank-reversal measures, and use locked HPO.

### 6. Statistical inference still uses the wrong units

Most reported intervals resample individual rows. That answers conditional sampling variation for fixed weights and fixed benchmark domains; it does not support claims across training runs, benchmark families, semantic twins, or deployment domains.

Remaining problems include:

- novel aggregate inference treats 2,020 prompts as IID despite only three balanced benchmark datasets;
- ExpGuard inference treats 2,275 rows as IID despite only three domains and a law-domain reversal;
- mortgage inference ignores pair/trap clusters and multi-turn families;
- the multi-seed tables report only means, with no seed values or combined seed/evaluation intervals;
- no leave-one-benchmark-out sensitivity is reported;
- multiple model, objective, ensemble, benchmark, and subgroup comparisons are not multiplicity controlled;
- several statements still call marginal intervals “paired” or use marginal-CI non-overlap despite paired effect estimates being available.

The limitation at line 877 still says “one seed per model,” directly contradicting the three-seed claim at line 770.

**Required change:** make benchmark-macro effects primary, retain per-benchmark effects, and use a hierarchical paired analysis that resamples global semantic families within benchmark and training seeds while holding the fixed model panel fixed. Report leave-one-benchmark-out results. Keep row-only intervals explicitly conditional and secondary.

### 7. The new pooled-versus-macro decomposition is mathematically invalid

At manuscript line 626, the paper subtracts the macro AP gap (`0.092`) from the pooled AP gap (`0.136`) and attributes the remainder (`0.044`) to cross-benchmark score alignment.

That difference is not an additive decomposition. Pooled AP and the mean of per-benchmark APs differ simultaneously in:

- benchmark weights;
- prevalence and sample counts;
- cross-benchmark score calibration/alignment;
- the nonlinear ordering functional used by average precision.

One cannot identify an “alignment contribution” by subtracting the two gaps.

**Required change:** remove the `0.092 + 0.044` attribution. If score alignment is scientifically important, measure it directly with within-benchmark rank normalization, source-specific monotone calibration, or a formal decomposition/simulation that holds weights and within-source rankings fixed.

### 8. Balanced challenge sets still do not support deployment claims

The paper added a prevalence limitation, but the main text continues to use “deployed point,” “real-time production,” “high compliance,” and “parity accuracy” language. No production prevalence, utility/cost matrix, partial AUROC, precision at realistic unsafe base rates, or low-FPR confidence analysis is provided.

Pooled AP remains the primary headline even though BeaverTails supplies more than half the in-house rows. Macro AP was added for the in-house comparison but not made co-primary for the transfer panel or clean multi-model study.

**Required change:** use benchmark-macro AP as the primary outcome; report pooled AP secondarily; add calibration-only recall at a 5% FPR target with realized test FPR; and label prevalence/cost curves as scenarios unless actual traffic weights exist.

### 9. The mortgage section remains too prominent for its evidence

The numerical correction is welcome, but the construct-validity problems remain:

- the 334-row set is single-annotator;
- no domain-expert agreement/adjudication exists;
- no producing pipeline regenerates the current 334-row file;
- no fixed jurisdiction or versioned regulatory control registry grounds all labels;
- there is no calibration split;
- FPR/accuracy use oracle thresholds selected on the same test labels;
- row bootstrap ignores semantic pairs and trap families;
- `experiments/eval_mortgage_hard.py` scores only `row["text"]`, so the 15 `conversation` contexts are discarded even though the paper presents multi-turn items;
- many source `flag` labels denote a need for a compliant/educational response, not that the user prompt itself should be blocked.

Yet mortgage and a “high-compliance stack” remain in the abstract, motivation, contributions, conclusion, and recommendation sections.

**Required change:** move the mortgage material to Paper B or an explicitly exploratory appendix. It should not support Paper A's title, abstract, contributions, or novelty claim. A feasible, evidence-gated design for that follow-on study is specified in [`docs/paper-b-joint-compliance-stack-plan.md`](docs/paper-b-joint-compliance-stack-plan.md); crucially, current separately labeled general and mortgage rows are diagnostic inputs, not a valid jointly labeled development/test set.

### 10. The GPT claim is better worded but still inconsistent

The table and nearby prose correctly say the study did not establish equivalence. Elsewhere, however, the paper still says:

- “GPT is a tie” (`paper/benchmark_chooses_the_winner.tex:863`);
- “parity accuracy” and “reaches parity” (`paper/benchmark_chooses_the_winner.tex:561-583`);
- the abstract presents lower latency alongside the inconclusive accuracy result.

The run still uses an unversioned alias, does not archive raw responses/system fingerprint, changed 131/2,018 predictions between nominally repeated runs, and compares local model compute with remote network/API latency.

**Required change:** remove GPT from the main claim. If retained as context, pin the dated snapshot and decoding settings, archive raw responses, and say only “no difference detected under this prompted rule.” Do not use “tie,” “parity,” or equivalence language.

### 11. The new name-fairness result is not statistically valid

`experiments/name_fairness_probe.py:114-121` samples template IDs with replacement, then uses `np.isin(tid, samp)`. `np.isin` collapses repeated template draws, so the procedure becomes random subset selection rather than a bootstrap with multiplicity. The reported interval is therefore not the claimed template bootstrap.

Additional problems:

- only 12 hand-authored templates and four hand-picked names per perceived-origin group;
- the reported max-minus-min group gap selects extreme groups without adjustment;
- no paired confidence interval on the base-to-tuned reduction;
- hard-coded calibrations come from a different primary run;
- perceived name origin is not demographic ground truth;
- no result artifact exists;
- “tuning removes bias” and ECOA/Reg-B implications are much stronger than this probe supports.

**Required change:** remove the fairness claim from Paper A. If developed separately, use a preregistered name resource, many templates/names, multiplicity-aware paired inference, a correct cluster bootstrap, and careful language about perceived-name effects rather than protected-class outcomes.

### 12. ExpGuard is useful external evidence but not yet a replication

The ExpGuard addition is directionally helpful because it uses a stronger domain benchmark. However:

- no summary or per-row score artifact exists;
- only one unspecified adapter/seed is evaluated;
- dataset/model revisions are unpinned;
- the row bootstrap ignores the three-domain structure;
- law reverses the aggregate direction;
- `paired_ci()` reports the mean of bootstrap deltas as `mean` rather than the observed paired delta;
- cache acceptance is length-only;
- the manuscript uses ExpGuard without citing the [ExpGuard paper](https://arxiv.org/abs/2603.02588).

**Required change:** move ExpGuard to external-validation appendix material unless the fixed model panel and seeds are all evaluated. Report the observed effect, domain-macro effect, per-domain effects, and a domain/seed sensitivity analysis. Cite and clearly credit ExpGuard.

## Reproducibility assessment

**Current rating: not reproducible or independently auditable from a fresh clone.**

### What improved

- Several repo-relative paths were repaired.
- The manuscript compiles from source.
- A multi-seed runner and aggregation scaffold now exist.
- Primary metric code can be recomputed locally when the ignored caches and adapters are present.

### What remains blocking

1. `git ls-files notebooks/outputs outputs` returns zero result artifacts.
2. A clean archive contains no frozen row manifest, per-row scores, summaries, adapters, or immutable model IDs for the adapters.
3. `verify_cached` is the notebook default but requires absent ignored JSON.
4. `recompute` evaluates missing adapters without first training or downloading them.
5. The checked-in notebook was not regenerated after its builder path fix and still embeds stale old values.
6. The notebook omits the new clean sweep, objective, HPO, ensemble, measured-guard, ExpGuard, and fairness result families while the paper says the released notebook reproduces the study.
7. `stage2.sh` is not repo-relative, omits primary SmolLM3, may mask failures, and emits no immutable run manifest.
8. Caches are usually validated only by length, not sample IDs/content hashes, model revision, tokenizer, prompt, adapter SHA, or code SHA.
9. No dataset or model revisions are pinned.
10. `experiments/make_figures.py` contains hand-entered values; figures and TeX tables are not generated from result artifacts.
11. There is no test suite or CI check for metrics, contamination, threshold isolation, cache joins, prompt/token parity, run completeness, or claim values.
12. The tracked PDF is stale relative to current TeX.
13. The current local notebook execution also fails because its Jupyter kernelspec points to the former `agent-bouncer/.venv`; this is workspace-specific but confirms that the advertised entrypoint has not been exercised end-to-end after the repo move.

### Required artifact contract

For each final run, publish:

```text
run_id
code_git_sha
model_id + revision
tokenizer_id + revision
adapter_sha256
objective + full optimizer/training config
training_seed + fixed data-order seed
training_manifest_sha256
evaluation_manifest_sha256
prompt_template_sha256
row_id + content_sha256 + source + family_id + split
raw safe_logit + unsafe_logit or their difference
calibration artifact and fit-row hash
summary metrics generated from the row-keyed scores
```

No cache should be accepted on length alone. Every table and figure should be generated from these artifacts, and CI should exercise `git archive` rather than the developer's dirty checkout.

## Compilation and presentation assessment

The current TeX compiles successfully with Tectonic 0.15.0, but the source is not publication-clean:

- fresh output is 23 pages while the tracked PDF is 22 pages;
- Tectonic stops after six passes with an internal consistency/rerun warning;
- line 683-684 produces a 31.8 pt overfull box;
- visual inspection confirms the `6rightjade/expguardmix` text spills into the adjacent column on page 12;
- page 1 is extremely dense: the 355-word abstract occupies most of the first column and immediately repeats the same three findings in the introduction;
- captions frequently carry provenance and rebuttal paragraphs that should be in body text or an artifact appendix.

The paper is approximately 16,500 source words with 39 headings, 21 table environments, and nine figures. This is not a focused presentation of one research question.

## Research narrative and novelty assessment

### Current narrative coherence

The current manuscript contains at least these independent arcs:

1. native-threshold/cross-metric guard ranking;
2. SmolLM3 base-versus-SFT specialization;
3. six-checkpoint convergence;
4. SFT versus DPO versus GRPO and HPO;
5. ExpGuard external validation;
6. name-origin fairness;
7. base-plus-tuned ensembling;
8. mortgage compliance;
9. open-guard/GPT leaderboard and latency;
10. a broad guardrail landscape;
11. rank-preservation theory.

They do not form one causal chain. Several are exploratory, several use different runs/splits, and several have no accessible artifacts. Adding limitations does not make the narrative focused; it makes the reader carry many caveats at once.

The paper also presents two different SmolLM3 headline experiments without cleanly choosing between them:

- original primary run: novel AP `0.884 -> 0.780`;
- claimed clean three-seed sweep: novel AP `0.883 -> 0.696`.

These need not be mathematically incompatible, because the training recipe/data differ, but using the first in the abstract and the second as evidence of stability makes the main estimand unclear. Choose one locked experiment as authoritative.

### Novelty verdict

**Current demonstrated novelty: low-to-moderate. Potential novelty after a clean focused rerun: moderate and publishable as a controlled measurement study.**

Operating-point sensitivity, post-hoc calibration, domain drift, policy overfitting, fine-tuning degradation, preservation regularization, and base/fine-tuned interpolation all have close prior art. The closest current boundaries include:

- [Domain Generalizable AI Guardrails with Augmented Policy Training](https://aclanthology.org/2026.acl-long.748/) — guard overfitting to training policies and unseen-policy/domain generalization;
- [FlexGuard](https://aclanthology.org/2026.acl-long.263/) — continuous guard risk scores and deployment strictness/threshold selection;
- [Challenges and Remedies of Domain-Specific Classifiers as LLM Guardrails](https://aclanthology.org/2025.naacl-industry.15/) — domain drift and false-refusal degradation;
- [ExpGuard](https://arxiv.org/abs/2603.02588) — expert-annotated domain guard evaluation in finance, healthcare, and law;
- [Guardrails in Logit Space / Safety Token Regularization](https://arxiv.org/abs/2604.17210), [RefusalGuard](https://arxiv.org/abs/2605.01913), and [When Safety Geometry Collapses](https://arxiv.org/abs/2605.02914) — preserving safety behavior/representations during fine-tuning;
- [WiSE-FT](https://arxiv.org/abs/2109.01903) — combining zero-shot and fine-tuned behavior under distribution shift.

The paper should not claim a new metric, a new calibration protocol, a new preservation principle, a new ensemble, or a general theorem that fine-tuning harms OOD performance.

### Defensible novelty after revision

> We contribute a controlled same-checkpoint, same-training-manifest, multi-seed estimate of how LoRA-SFT moves compact prompt guards across represented-source and dataset-held-out benchmark panels. The contribution is the attribution design and its heterogeneous empirical map, not a universal law of fine-tuning.

That is clear, useful, and differentiated enough if the clean artifacts and analysis are released.

## Recommended focused Paper A

### Title

> **The Benchmark Chooses the Winner: Measuring Fine-Tuning Specialization Across Safety-Guard Benchmarks**

Avoid “fair evaluation.” “Fair” is subjective, collides with fair-lending terminology, and currently describes a comparison that changes both metric and scoring rule.

### One-sentence message

> A compact guard's apparent gain after SFT is concentrated on benchmark sources represented during adaptation and does not imply transferable prompt-moderation competence on dataset-held-out sources.

### Research questions

1. **RQ1 — Represented-source effect:** How does SFT change benchmark-macro AP on held-out rows from sources represented in training?
2. **RQ2 — Dataset-held-out transfer:** How does SFT change benchmark-macro AP on sources excluded from training and calibration?
3. **RQ3 — Heterogeneity:** How do those paired effects vary across checkpoints, benchmarks, and training seeds?
4. **RQ4 — Secondary operating point:** At a calibration-derived 5% FPR target, how do recall and realized FPR transfer?

### Contributions

1. **Controlled attribution design.** Checkpoint-matched base-versus-SFT comparisons using identical prompts, scoring heads, training rows, LoRA capacity, update budget, and evaluation rows.
2. **Auditable benchmark provenance.** An explicit separation between represented-source evaluation and dataset-held-out transfer, with global family decontamination and benchmark-macro reporting.
3. **Heterogeneity map.** Multi-seed, per-checkpoint and per-benchmark estimates showing where adaptation adds competence and where it preserves, improves, or degrades transfer.
4. **Reproducible artifact.** Frozen manifests, row-keyed scores, hashes, configs, seeds, and generated tables/figures.

Do not list operating-point matching, DPO, ensemble recovery, mortgage compliance, GPT latency, or fairness as main contributions.

### Abstract template

Use this only after the clean artifacts have been regenerated and replace brackets with locked results:

> Fine-tuned safety guards are often evaluated on benchmark sources related to their adaptation data, making it difficult to separate inherited capability from specialization. We measure how LoRA supervised fine-tuning changes prompt-level safety classification relative to the same untuned instruction checkpoint. Using a frozen training manifest, [three or five] seeds, and four 1.5–4B checkpoints from two model lineages, we score identical rows with the same two-token log-probability head. We separate represented-source benchmarks from dataset-held-out transfer benchmarks and use benchmark-macro average precision as the primary outcome, complemented by per-benchmark effects and calibration-derived recall at 5% false-positive rate. Fine-tuning changes represented-source macro AP by [X] but transfer macro AP by [Y], with [direction/count] varying across checkpoints and benchmarks. These results show that an apparent guard improvement can reflect specialization to represented benchmark distributions rather than a general moderation upgrade. We contribute a controlled base-to-adapter attribution design, an auditable benchmark-provenance protocol, and versioned row-level artifacts. We do not claim that fine-tuning universally harms out-of-distribution performance; we identify when adaptation adds represented-source competence without reliable transfer.

This is approximately 175–200 words after values are inserted, rather than the current 355 words.

## Recommended model and benchmark scope

The user-requested narrow scope can be achieved entirely with models and data already present.

### Primary model panel

Use a complete four-checkpoint panel:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `HuggingFaceTB/SmolLM3-3B`
- `Qwen/Qwen3-4B`

Describe these as **four checkpoints from two model lineages**, not broad cross-family evidence.

Move `DeepSeek-R1-Distill-Qwen-1.5B` to an appendix because its chain-of-thought/verdict behavior creates a separate reward-starvation and interface confound. Move Qwen3-8B to an appendix because it adds compute and an incomplete DPO cell without materially broadening the SFT claim.

### Confirmatory benchmark roles

| Role | Benchmarks | Treatment |
|---|---|---|
| Represented-source evaluation | ToxicChat, Prompt Injections, Jailbreak Classification | Train on fixed source-disjoint rows; evaluate on immutable held-out families from the same source. |
| Dataset-held-out transfer | JailbreakBench, XSTest, WildGuardTest, WildJailbreak | Zero train/calibration rows; remove exact and near-duplicate global families before training. |
| Over-refusal stress | OR-Bench-Hard benign portion | Report safe-side FPR/score distribution only; do not construct AP from benign and toxic rows drawn from different configurations. |
| All-positive stress | HarmBench | Report recall/score sensitivity only; exclude from AP aggregates. |
| External validation | ExpGuardTest | Appendix unless the full checkpoint/seed panel is scored and artifacted. |

Remove BeaverTails from the confirmatory prompt-classification panel because its `is_safe` field labels a response-in-context pair while the pipeline discards the response and reuses the label as prompt gold. Keeping it as the largest pooled source makes a known construct transformation load-bearing.

Move the mortgage benchmark entirely to Paper B.

### Training matrix

The clean minimal experiment is feasible:

- one deterministic 1,200-row training manifest;
- 400 rows per represented source;
- 200 safe and 200 unsafe per source;
- one fixed data-order seed;
- four checkpoints;
- preferably five training seeds (`4 × 5 = 20` adapters), with three as the minimum if compute is constrained;
- identical LoRA modules, rank, alpha, dropout, steps, batch exposure, and completion format.

Base models require scoring only, not training.

## Required analysis

### Primary estimands

- Per-checkpoint paired change in represented-source benchmark-macro AP.
- Per-checkpoint paired change in dataset-held-out benchmark-macro AP.
- Panel mean of those paired changes, explicitly conditional on the fixed four-checkpoint panel.
- Per-benchmark and per-seed effects.

### Secondary estimands

- Recall and realized FPR at a 5% FPR threshold chosen on calibration only.
- AUROC and pooled AP as supporting sensitivity metrics.
- OR-Bench benign FPR and HarmBench recall as one-class stress tests.

### Inference

- Use row-keyed paired scores.
- Resample global semantic families within benchmark.
- Resample training seeds for adaptation effects.
- Keep model IDs fixed; do not pretend four checkpoints are a random sample of all SLMs.
- Report leave-one-benchmark-out transfer effects.
- Report exact per-seed values rather than mean-only tables.
- Make multiplicity rules explicit if claiming checkpoint- or benchmark-specific significance.

## Recommended figures

Keep at most three main figures:

1. **Provenance pipeline.** Exact sources entering training, calibration, represented-source evaluation, and dataset-held-out transfer; include removed-overlap counts.
2. **Paired specialization plot.** Two panels—represented-source macro AP and transfer macro AP—with base and SFT connected for every checkpoint/seed.
3. **Per-benchmark delta heatmap.** Rows are checkpoints, columns are benchmarks, cells are mean `SFT - base` AP with seed intervals or uncertainty markers.

Optionally add a compact fourth panel for calibration-derived recall at 5% FPR.

Move or remove:

- open-guard leaderboard bars;
- native-F1/AUPRC ranking-flip plot;
- GPT latency Pareto plot;
- mortgage ranking diagram;
- fairness prose-only result;
- theorem/rank-preservation appendix unless it is connected to measured pair reversals;
- guardrail landscape tables;
- HPO and objective tables;
- ensemble table.

## Exact manuscript refactor

### Keep and rewrite

- Single-token score formulation.
- Model and LoRA description, reduced to the four-checkpoint fixed panel.
- One provenance/split diagram.
- Canonical AP/AUROC definitions.
- Clean base-versus-SFT results.
- Per-benchmark heterogeneity.
- Honest limitations about fixed models, prompt-only labels, benchmark scope, and prevalence.

### Move to appendix

- ExpGuard external validation after it is rerun across the locked panel or clearly labeled single-checkpoint descriptive evidence.
- OR-Bench and HarmBench one-class stress analyses.
- Open guards as sanity references, without a “best guard” recommendation.
- Full training/environment details.

### Remove from Paper A

- Mortgage/high-compliance stack framing and case study.
- Name-origin fairness claim.
- GPT equivalence/parity and cross-substrate latency story.
- DPO/GRPO/HPO causal objective claims.
- Base-plus-tuned ensemble as a contribution.
- Broad guardrail landscape and product recommendations.
- Rank-preservation theorem unless pair-reversal quantities become measured outcomes.
- Revision-history prose explaining every earlier bug.

### Recommended section order

1. Introduction and four RQs.
2. Closest work and novelty boundary.
3. Controlled design: checkpoints, manifests, training, scoring, decontamination.
4. Represented-source and dataset-held-out benchmark panels.
5. Primary base-versus-SFT results.
6. Heterogeneity and low-FPR sensitivity.
7. Limitations and artifact availability.
8. Conclusion.

## Additional experiments worth doing

These are ordered by scientific value per unit cost.

1. **Mandatory: clean exact/near-duplicate rerun.** Remove the four known WildJailbreak conflicts and all global near-duplicate families, freeze hashes, and retrain the final SFT panel.
2. **Mandatory: seed-complete fixed panel.** Produce three to five valid SFT seeds for each of the four checkpoints with all artifacts present.
3. **Mandatory: macro and per-benchmark analysis.** Replace pooled-primary claims with represented/transfer macro AP and per-benchmark deltas.
4. **High value: leave-one-benchmark-out sensitivity.** Show that the panel conclusion is not chosen by one transfer benchmark.
5. **High value: low-FPR transfer.** Use a calibration-only 5% FPR threshold and report realized FPR/recall for every benchmark.
6. **Useful appendix: ExpGuard external validation.** Score the fixed panel and seeds, cite the dataset paper, and report domain-macro rather than only pooled row inference.
7. **Optional mechanism analysis:** measure unsafe-safe pair reversals and base-to-SFT logit drift. This would connect the empirical result to the rank-preservation discussion without claiming a new theorem.

Do not spend the next compute budget on more objectives, more guard leaderboards, or more domain case studies until the clean SFT attribution experiment is complete and auditable.

## Acceptance gates for the next revision

### Correctness

- [ ] One canonical, tested tie-aware metric module is used everywhere.
- [ ] Tie-order permutation tests pass for AP and AUROC.
- [ ] No test/calibration labels enter training, HPO, threshold fitting, or model selection.
- [ ] Exact and near-duplicate train/evaluation family intersections are zero after documented removals.
- [ ] Every claim value is generated from row-keyed artifacts rather than hand transcription.
- [ ] The macro-versus-pooled additive decomposition is removed.
- [ ] Fairness bootstrap/result is removed or correctly redesigned.

### Evidence

- [ ] Every primary checkpoint has the required complete seeds.
- [ ] Per-seed scores/configs/hashes are available.
- [ ] Primary results use represented-source and dataset-held-out benchmark-macro AP.
- [ ] Family/seed-aware paired intervals and leave-one-benchmark-out results are reported.
- [ ] The final test is touched only by locked final runs.
- [ ] Stress sets are not mixed into two-class aggregates they cannot support.

### Reproducibility

- [ ] `git archive` plus documented downloads can run `verify` without private ignored files.
- [ ] Immutable adapters or public adapter IDs are provided.
- [ ] Model, tokenizer, and dataset revisions are pinned.
- [ ] Cache keys include row, model, adapter, tokenizer, prompt, and code hashes.
- [ ] Stage runner is repo-relative, fail-fast, and validates every expected artifact.
- [ ] Tables and figures are generated from versioned analysis artifacts.
- [ ] A CI smoke test covers manifests, metrics, claim gates, and manuscript build.
- [ ] Checked-in notebook and PDF are regenerated from the final source/artifacts.

### Narrative

- [ ] One title, one thesis, four aligned RQs, and at most four contributions.
- [ ] Abstract is approximately 200 words and contains only locked confirmatory findings.
- [ ] Mortgage, fairness, GPT parity, objective HPO, ensemble, and landscape are removed from Paper A's main narrative.
- [ ] “Fair,” “source-family-disjoint,” “uncontaminated,” “parity,” “tie,” and “only objective varies” are used only where literally supported.
- [ ] Closest 2025–2026 guard specialization, calibration, and preservation work is cited.

## Updated scorecard

| Criterion | Score | Assessment |
|---|---:|---|
| Technical correctness | 2/5 | Primary metric bug corrected, but active producers, HPO, fairness inference, and several claims remain incorrect. |
| Experimental design | 2/5 | A promising same-checkpoint design exists, but contamination, mixed run regimes, HPO exposure, and objective confounds remain. |
| Statistical support | 1.5/5 | Paired row effects improved, but benchmark/family/seed units and multiplicity are not handled. |
| Reproducibility | 1/5 | Source builds, but the numerical artifact and new clean multi-seed evidence are absent from a clean archive. |
| Novelty | 2.5/5 | Controlled specialization attribution could be publishable; the current broad claims overlap substantial prior art and are not fully evidenced. |
| Readability and flow | 1.5/5 | More honest locally, but longer and less focused globally; too many tables, stories, and caveat-heavy captions. |

## Final verdict

The authors have responded seriously to the earlier review, particularly on tie handling, mortgage result correction, terminology, and literature positioning. However, the revision has not yet demonstrated that the earlier recommendations were implemented **correctly and consistently end-to-end**. Several issues were disclosed rather than fixed, and the most important new empirical claim—the clean, multi-seed sweep—is unsupported by accessible run artifacts and still contaminated by known exact overlaps.

The right next move is not another expansion. It is a contraction: one versioned clean base-versus-SFT experiment, four current checkpoints, represented-source versus dataset-held-out benchmark-macro AP, family/seed-aware inference, and a complete artifact chain. That paper would have a clear message and a defensible moderate novelty claim.
