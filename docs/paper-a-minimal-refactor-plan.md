# Paper A Minimal Refactor — Complete Research and Implementation Plan

Date: 2026-07-12  
Status: implementation specification  
Target manuscript: **paper/benchmark_chooses_the_winner.tex**  
Primary goal: produce a focused, defensible measurement paper with the smallest scientifically adequate change to the current research program.

---

## 1. Direct decision

Paper A will become a fixed-panel, multi-seed study of one question:

> What does LoRA-SFT add beyond the untuned base model, and does that gain transfer to evaluation datasets whose rows were not used for SFT?

Recommended title:

> **The Benchmark Chooses the Winner: Measuring Fine-Tuning Specialization Across Safety-Guard Benchmarks**

Fallback title:

> **Fine-Tuning Small Prompt-Safety Classifiers: In-Source Gains and Held-Out Benchmark Transfer**

The paper will not introduce or claim a new:

- metric;
- calibration algorithm;
- preservation loss;
- preference-learning method;
- ensemble;
- compliance architecture;
- theorem.

Its defensible novelty is:

> a controlled, same-checkpoint, same-training-manifest, multi-seed estimate of how compact binary prompt guards move between represented-source discrimination and dataset-held-out benchmark transfer.

This is a measurement paper. Operating-point analysis is a secondary deployment diagnostic. Mortgage compliance belongs to Paper B.

---

## 2. Why this is the minimum defensible refactor

The current manuscript combines at least seven stories:

1. operating-point ranking instability;
2. an open-guard leaderboard;
3. base-versus-tuned specialization;
4. a six-base convergence claim;
5. SFT/DPO/GRPO objective comparisons;
6. a base-plus-tuned ensemble;
7. a mortgage-compliance case study.

Those stories compete for attention and rely on evidence with different validity levels. The base-versus-tuned observation is the cleanest conceptual center because it:

- uses the same underlying checkpoint before and after adaptation;
- measures the actual effect of converting an instruction model into a guard;
- directly reuses the existing guard wrapper, LoRA recipe, benchmarks, and scoring interface;
- can be made multi-seed with 20 SFT runs rather than the 40-adapter SFT/DPO/KL matrix;
- does not require a new training algorithm;
- creates a clear practical conclusion without claiming a universal best guard.

The refactor preserves most of the existing methodology but removes secondary claims that would otherwise require extensive reruns.

---

## 3. Central thesis and claim boundary

### 3.1 Thesis

Use this result-neutral thesis before final scoring:

> For a fixed four-checkpoint panel and one frozen LoRA-SFT recipe, guard adaptation changes performance differently on represented-source and dataset-held-out tests. Comparing every guard with its own untuned base reveals whether adaptation adds transferable competence or redistributes performance toward sources represented during training.

After final scoring, the conclusion must follow the preregistered decision rules in Section 16. Do not force a negative transfer conclusion if the effect is heterogeneous or inconclusive.

### 3.2 Scope

The paper covers:

- English input-prompt classification;
- binary safe/unsafe decision-token scoring;
- small instruction models from 1.5B to 4B parameters;
- one fixed LoRA-SFT recipe;
- three represented-source datasets;
- four dataset-held-out transfer benchmarks;
- two single-class stress sets;
- a fixed, purposively chosen four-checkpoint panel.

The paper does not cover:

- response moderation;
- tool-call or agent-action moderation;
- full-dialogue policy compliance;
- arbitrary unseen policies;
- production traffic;
- legal or regulatory compliance;
- model-population inference;
- adversarially adaptive attacks;
- universal claims about fine-tuning.

### 3.3 Terminology

| Avoid | Use |
|---|---|
| fixed-policy source shift | dataset-held-out benchmark transfer |
| source-family OOD, unless lineage is proved | dataset-held-out transfer |
| fine-tuning hurts OOD | this LoRA-SFT recipe changed transfer performance by a measured amount |
| cross-family model result | fixed four-checkpoint panel |
| frontier, with one tuned point | joint represented-source/transfer movement |
| guard improvement | represented-source gain, transfer gain, or transfer loss |
| matched test FPR | calibration-targeted FPR with realized test FPR |
| production PPV | prevalence sensitivity under stated assumptions |

---

## 4. Contributions

The introduction will state only these three contributions.

### Contribution 1 — controlled base-to-guard attribution

For each checkpoint, compare the untuned base with five independently trained SFT adapters on identical examples, prompts, score definitions, and metrics.

### Contribution 2 — fixed-panel transfer characterization

Estimate represented-source and dataset-held-out changes separately, report per-base and per-benchmark heterogeneity, and restrict conclusions to the named panel.

### Contribution 3 — auditable measurement package

Release immutable manifests or redistributable identifiers, source revisions, content hashes, seed-level results, keyed scores, direct paired intervals, generated tables and figures, and an executable cached-analysis path.

Operating-point evaluation is supporting methodology, not a contribution. The paper will explicitly say this.

---

## 5. Research questions

### RQ1 — represented-source effect

For the fixed four-checkpoint panel, how does a common LoRA-SFT recipe change macro average precision on benchmark sources represented during training?

### RQ2 — benchmark-transfer effect

How does the same adaptation change macro average precision on dataset-held-out transfer benchmarks?

### RQ3 — heterogeneity

Are the represented-source and transfer changes stable across checkpoints and evaluation datasets, or are they concentrated in particular bases and benchmarks?

### RQ4 — deployment sensitivity

What TPR/FPR trade-off do the base and SFT systems realize when thresholds are selected on calibration data at a nominal 5% pooled-negative FPR target?

RQ4 is descriptive and secondary. It does not provide a distribution-free production guarantee. Failure to obtain a feasible threshold under the prespecified bound is reportable and is not a reason to select a threshold on test.

---

## 6. Data design

### 6.1 Primary training pool

Start from the realized 4,279-row legacy pool, but do not reuse its file as the final manifest.

| Source | Legacy rows | Primary training decision | Reason |
|---|---:|---|---|
| ToxicChat | 655 | include in non-commercial research branch | prompt-level toxicity label; license restrictions must be explicit |
| Prompt Injections | 406 | include | prompt-level binary injection label |
| Jailbreak Classification | 1,018 | include after overlap removal | prompt-level jailbreak label |
| BeaverTails | 1,200 | exclude from primary | is_safe labels the prompt-response interaction, not the prompt alone |
| OR-Bench | 1,000 | exclude | evaluation-family contamination and safe-only training contribution |

Before near-duplicate filtering:

\[
655+406+1{,}018=2{,}079.
\]

Remove the four known normalized exact overlaps between Jailbreak Classification training rows and WildJailbreak evaluation rows:

\[
2{,}079-4=2{,}075
\]

maximum candidate rows before additional audit removals.

The manifest builder must recompute exact safe/unsafe counts. Do not hard-code 2,075 as the final count.

To preserve the current 300-update recipe while ensuring identical row exposure, the final train manifest will contain exactly 1,200 rows selected after the audit:

- 400 rows per included source;
- within each source, 200 safe and 200 unsafe rows;
- deterministic selection by a frozen hash-ranking rule salted with data_seed;
- one fixed row order shared across every checkpoint and training seed.

All included sources currently have at least 200 examples per label before final near-duplicate adjudication. If a stratum falls below 200 after audit, stop and revise the manifest rule before LOCK.json; do not silently sample with replacement or change the source weights.

At effective batch size 4, 1,200 rows produce exactly 300 optimizer updates for one complete exposure of the frozen manifest.

### 6.2 Licensing fork

The configuration must choose one branch before final training.

#### Branch A — recommended minimal academic study

- Include ToxicChat.
- Mark the study and any resulting adapter release as non-commercial unless a qualified review permits a different interpretation.
- Do not place third-party raw text under the repository's Apache-2.0 license.
- Release source identifiers, revisions, hashes, scripts, score artifacts where permitted, and a clear third-party data notice.
- Do not promise unrestricted adapter redistribution.

#### Branch B — permissive-release study

- Exclude ToxicChat as well.
- Recompute the training pool and represented-source evaluation roles.
- Add a permissively licensed prompt-safety source only if it can be incorporated before the lock without inspecting final results.
- Retrain every cell under the new manifest.

Never mix results across licensing branches.

Dataset terms to record:

- BeaverTails: CC BY-NC 4.0.
- ToxicChat: CC BY-NC 4.0.
- WildGuardMix: ODC-BY plus AI2 access conditions.
- WildJailbreak: ODC-BY plus AI2 access conditions.
- Prompt Injections: verify and snapshot the authoritative license metadata.
- Jailbreak Classification, JailbreakBench, XSTest, OR-Bench, and HarmBench: record exact source license and redistribution terms.

### 6.3 Evaluation regimes

#### Represented-source ID

| Benchmark | Current test rows | Use |
|---|---:|---|
| ToxicChat | 401 | AP/AUROC and operating point |
| Prompt Injections | 68 | AP/AUROC; report wide uncertainty |
| Jailbreak Classification | 148 | AP/AUROC |
| **Total** | **617** | macro-average over three sources |

The manifest builder must verify these counts against pinned source snapshots.

#### Dataset-held-out benchmark transfer

| Benchmark | Current rows | Use |
|---|---:|---|
| JailbreakBench | 120 | transfer AP/AUROC |
| XSTest | 240 | transfer and over-refusal behavior |
| WildGuardTest | 800 | external dataset-held-out transfer |
| WildJailbreak | 420 | external dataset-held-out transfer |
| **Total** | **1,580** | macro-average over four sources |

These benchmarks use heterogeneous native label constructs and may have conceptual or data-generation lineage relationships. The paper therefore studies dataset-held-out benchmark transfer, not source-family independence or a single fixed policy.

#### Stress-only sets

| Benchmark | Rows | Use |
|---|---:|---|
| OR-Bench-Hard benign portion | approximately 400 | false-positive/over-refusal stress only |
| HarmBench | 200 unsafe-only | recall stress only |

Do not compute AP or AUROC on single-class stress data.

Do not use the current OR-Bench-Hard hybrid for AP: its benign and unsafe classes come from different configurations and are class-by-source confounded.

### 6.4 Calibration data

Rebuild calibration rows from pinned represented-source development material after BeaverTails exclusion.

Requirements:

- no training row overlap;
- no final-test row overlap;
- source-stratified;
- class counts reported;
- family IDs reported;
- large enough to support a conservative 5% FPR target;
- never used for model training or early stopping.

There is no HPO development set in the minimal paper because the training recipe is frozen before final scoring.

#### 6.4.1 Mechanical source construction

All source revisions are explicit values in configs/paper_a_sft.yaml. The builder must never resolve a moving revision. The following rules are part of the lock:

| Role | Dataset/config/split | Text field | Native-to-binary label map | Deterministic target |
|---|---|---|---|---:|
| train candidate | lmsys/toxic-chat / toxicchat0124 / train | user_input | toxicity == 1 → unsafe; otherwise safe | eligible pool |
| train candidate | deepset/prompt-injections / default / train | text | label == 1 → unsafe; otherwise safe | eligible pool |
| train candidate | jackhhao/jailbreak-classification / default / train | prompt | type starts with jailbreak, case-insensitive → unsafe; otherwise safe | eligible pool |
| calibration + ID | lmsys/toxic-chat / toxicchat0124 / test | user_input | toxicity == 1 → unsafe; otherwise safe | at most 400 per label before 40/60 split |
| calibration + ID | deepset/prompt-injections / default / test | text | label == 1 → unsafe; otherwise safe | all eligible, balanced to minority label |
| calibration + ID | jackhhao/jailbreak-classification / default / test | prompt | type starts with jailbreak, case-insensitive → unsafe; otherwise safe | at most 400 per label before 40/60 split |
| transfer | JailbreakBench/JBB-Behaviors / behaviors / harmful and benign splits | Goal, fallback goal | harmful split → unsafe; benign split → safe | 60 per label |
| transfer | natolambert/xstest-v2-copy / default / prompts | prompt | type starts with contrast, case-insensitive → unsafe; otherwise safe | 120 per label |
| transfer | allenai/wildguardmix / wildguardtest / test | prompt | prompt_harm_label harmful → unsafe; unharmful → safe; discard other labels | 400 per label |
| transfer | allenai/wildjailbreak / eval / train | adversarial | integer label 1 → unsafe; 0 → safe | all eligible, expected 210 per label |
| benign stress | bench-llm/or-bench / or-bench-hard-1k / train | prompt | safe | 400 rows |
| positive stress | walledai/HarmBench / standard / train | prompt | unsafe | 200 rows |

For each source:

1. require a nonempty text field;
2. preserve the upstream row identifier; if none exists, create source_row_id from the source revision, config, split, and original integer position;
3. normalize text by the frozen NFKC/lowercase/whitespace rule for hashing while retaining the original text for scoring;
4. deduplicate exact normalized text within a source, keeping the lexicographically smallest source_row_id when labels agree;
5. quarantine same-text conflicting labels rather than applying unsafe-wins;
6. rank eligible rows within each source-label stratum by SHA-256 of data_seed, source_row_id, and content_sha256;
7. take rows in rank order until the target is met;
8. if audit removal creates a shortfall, continue down the already frozen ranked candidate list;
9. never regenerate, paraphrase, or hand-select a replacement.

#### 6.4.2 Removal precedence and calibration/ID split

Apply operations in this order:

1. load pinned upstream rows and apply the label map;
2. filter missing/unsupported labels;
3. exact-deduplicate within source;
4. construct all evaluation candidate pools;
5. remove exact evaluation overlaps from training candidates;
6. construct union family clusters across all candidate rows;
7. adjudicate cross-split near-duplicate clusters and remove the train-side row when unresolved;
8. select the final 1,200-row training manifest by frozen hash rank;
9. select each transfer/stress target by frozen hash rank;
10. split represented-source test candidates into calibration and ID at the family level.

Calibration/ID split algorithm:

- operate independently within each represented source;
- target 40% calibration and 60% ID by row count;
- sort family_id values by SHA-256 of data_seed, source, and family_id;
- greedily assign whole families to calibration until adding the next family would move the calibration row count farther from the 40% target than leaving it for ID;
- assign all remaining families to ID;
- require both labels in both partitions;
- require at least 60 pooled calibration negatives and at least 10 calibration negatives from each represented source;
- if an assertion fails, stop before lock and revise the design explicitly rather than moving individual rows manually.

The current counts of 617 represented-source ID rows and 1,580 transfer rows are legacy sizing anchors, not guaranteed final counts. The manifest report must publish final post-audit counts and explain every difference.

### 6.5 Canonical row schema

Every manifest row must contain:

~~~text
sample_id
source
source_config
source_revision
source_row_id
split
label
label_provenance
text_or_download_reference
content_sha256
family_id
license_id
redistribution_class
known_overlap_disposition
~~~

If raw text cannot be redistributed, publish the upstream identifier, content hash, reconstruction procedure, and access requirements instead.

### 6.6 Required manifests

~~~text
artifacts/paper_a_sft/manifests/train.jsonl
artifacts/paper_a_sft/manifests/calibration.jsonl
artifacts/paper_a_sft/manifests/id_test.jsonl
artifacts/paper_a_sft/manifests/transfer_test.jsonl
artifacts/paper_a_sft/manifests/orbench_safe_stress.jsonl
artifacts/paper_a_sft/manifests/harmbench_positive_stress.jsonl
artifacts/paper_a_sft/manifests/manifest.json
~~~

### 6.7 Data audit

The audit must perform:

1. normalized exact-text overlap;
2. conflicting-label exact overlap, reported separately from same-label overlap;
3. character n-gram MinHash or an equivalent near-duplicate search;
4. source-family membership validation;
5. class and row counts;
6. source revision validation;
7. license inventory;
8. family/cluster construction validation;
9. train/calibration/test role validation;
10. proof that OR-Bench and BeaverTails counts in primary training are zero.

Family construction must be mechanical and frozen:

1. add an edge between rows sharing an authoritative upstream family, conversation, pair, or scenario ID;
2. independently normalize every row across every source and split using Unicode NFKC, lowercase, collapsed whitespace, and preserved punctuation;
3. compute character 5-gram MinHash signatures with 256 permutations for every row, including rows that already have upstream family IDs;
4. add an undirected edge when estimated Jaccard similarity is at least 0.85;
5. take the union of upstream-family edges and MinHash edges across all rows and splits;
6. define connected components by union-find;
7. set family_id to the SHA-256 of the lexicographically smallest content hash in the component;
8. manually adjudicate every component crossing train and any evaluation split;
9. remove the train-side member of unresolved cross-split components;
10. freeze the implementation version, normalization rules, threshold, edge provenance, and adjudication file before locking.

The 0.85 threshold is a prespecified candidate-generation threshold, not proof of semantic equivalence. The audit report must include sensitivity counts at 0.80 and 0.90 without changing dispositions after final scores are viewed.

Hard assertions:

~~~text
or_bench_train_count == 0
beavertails_train_count == 0
exact_train_vs_eval_overlap == 0
conflicting_label_overlap == 0
every_row_has_source_revision == true
every_row_has_content_hash == true
every_near_duplicate_candidate_has_disposition == true
~~~

Outputs:

~~~text
artifacts/paper_a_sft/audit/audit.json
artifacts/paper_a_sft/audit/audit.md
~~~

---

## 7. Model panel

Use the four locally available checkpoints below, but verify each revision before locking.

| Key | Model | Current local revision candidate |
|---|---|---|
| qwen25_15b | Qwen/Qwen2.5-1.5B-Instruct | 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 |
| smollm2_17b | HuggingFaceTB/SmolLM2-1.7B-Instruct | 31b70e2e869a7173562077fd711b654946d38674 |
| smollm3_3b | HuggingFaceTB/SmolLM3-3B | a07cc9a04f16550a088caea529712d1d335b0ac1 |
| qwen3_4b | Qwen/Qwen3-4B | 1cfa9a7208912126459214e8b04321603b3df60c |

For every model:

- pin model and tokenizer revisions;
- record tokenizer file hashes;
- verify that safe and unsafe are distinct single tokens under the actual decision-token convention;
- hash the rendered system/user prompt template;
- verify that the chat template is deterministic;
- record trust_remote_code state;
- record dtype and attention implementation;
- fail rather than fall back to moving main.

Decision-token convention:

- first test the leading-space strings " safe" and " unsafe";
- if both are distinct single tokens, use those exact target strings;
- otherwise test the no-leading-space strings "safe" and "unsafe";
- fail if neither convention produces distinct single tokens;
- record the selected strings and token IDs per checkpoint;
- training completion loss includes the selected verdict token and the appended EOS token;
- scoring, training, and tests must import the same prompt/token specification.

The statistical estimand is this fixed panel. Do not bootstrap model identities or infer over all model families.

---

## 8. Guard formulation

Retain the existing one-forward-pass binary guard.

For prompt x and final prompt position t:

\[
s(x)=z_{unsafe}(x,t)-z_{safe}(x,t)
\]

and:

\[
p(unsafe\mid x)=
\frac{\exp(z_{unsafe})}
{\exp(z_{unsafe})+\exp(z_{safe})}.
\]

Requirements:

- use raw safe/unsafe logits as the canonical stored values;
- derive probabilities from the stored logits;
- do not use generated verdict text as the primary score;
- store the token IDs and token strings;
- record truncation, original token count, and scored token count;
- fail if the decision tokens are not distinct single tokens;
- use one versioned prompt template across every base and seed, with documented model-specific chat rendering only where unavoidable.

---

## 9. Training design

### 9.1 Fixed SFT recipe

To minimize change, retain the current SFT recipe rather than introducing objective-parity changes that are irrelevant in an SFT-only paper.

| Parameter | Locked value |
|---|---|
| objective | completion-only SFT |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| target modules | q, k, v, o, gate, up, down projections |
| per-device batch | 1 |
| gradient accumulation | 4 |
| effective batch | 4 |
| optimizer updates | 300 |
| learning rate | 2e-4 |
| scheduler | cosine |
| warmup ratio | 0.03 |
| maximum sequence length | 1,024 |
| seeds | 42, 43, 44, 45, 46 |
| data-order seed | 42, fixed for every run |
| final training rows | 1,200, identical across every run |
| completion loss | verdict token plus EOS |

If one of these values changes after smoke testing, change it before generating LOCK.json. Smoke precedes the final lock. No final cell may exist when the final lock is created, and no hyperparameter may be tuned on final evaluation results.

training_seed controls LoRA initialization and stochastic layers. data_order_seed remains fixed. Record both separately.

### 9.2 Training matrix

| Model | 42 | 43 | 44 | 45 | 46 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | SFT | SFT | SFT | SFT | SFT | 5 |
| SmolLM2-1.7B | SFT | SFT | SFT | SFT | SFT | 5 |
| SmolLM3-3B | SFT | SFT | SFT | SFT | SFT | 5 |
| Qwen3-4B | SFT | SFT | SFT | SFT | SFT | 5 |
| **Total** | 4 | 4 | 4 | 4 | 4 | **20** |

No DPO, GRPO, KL-SFT, ensemble training, mortgage training, or open-guard training belongs in the minimal paper.

### 9.3 Staged execution

#### Stage 0 — pre-lock smoke

- run one short smoke per base with a separate smoke-only output path and a provisional smoke configuration;
- validate adapter loading, hashes, prompt/token parity, and scoring on synthetic plus calibration-only fixtures;
- do not score ID, transfer, or stress manifests;
- revise the recipe only from engineering failures or calibration-only diagnostics;
- after smoke passes, freeze the recipe and create final LOCK.json.

#### Stage 1 — first twelve final cells

- under final LOCK.json, train seeds 42, 43, and 44 for all four bases;
- validate adapters, run metadata, hashes, and completeness only;
- do not score ID, transfer, or stress manifests;
- do not compute or expose provisional headline metrics.

#### Stage 2 — complete and score the final matrix

- after Stage 1 engineering validation, run seeds 45 and 46;
- require 20/20 adapters;
- score ID, transfer, and stress manifests once, after 20/20 runs are complete;
- a failed cell remains visible and blocks the fixed-panel aggregate until rerun or explicitly removed by a preregistered failure rule.

### 9.4 Run metadata

Every run writes:

~~~text
run_id
model_key
model_revision
tokenizer_revision
train_manifest_sha256
config_sha256
prompt_sha256
seed
adapter_sha256
LoRA configuration
optimizer configuration
global steps
examples seen
tokens seen
wall time
device
software versions
git SHA
start and completion timestamps
status
failure reason
~~~

---

## 10. Scoring and calibration

### 10.1 Score matrix

Score:

- four untuned bases once each;
- twenty SFT adapters;
- calibration, represented-source ID, transfer, OR-Bench benign stress, and HarmBench positive stress manifests.

Total model bundles:

\[
4\text{ bases}+20\text{ adapters}=24.
\]

Base scores must be reused across comparisons rather than recomputed independently for each seed.

### 10.2 Per-row score schema

~~~text
sample_id
content_sha256
source
split
gold
family_id
model_key
model_revision
condition
seed
adapter_sha256
prompt_sha256
safe_token_id
unsafe_token_id
safe_logit
unsafe_logit
score_raw
probability_raw
probability_calibrated
threshold_id
prediction
original_token_count
scored_token_count
truncated
latency_ms
~~~

Store Parquet plus immutable metadata JSON. If Parquet is used, pin pyarrow.

### 10.3 Cache validity

A cache is valid only when all of these match:

- manifest hash;
- ordered sample IDs;
- content hashes;
- model revision;
- tokenizer revision;
- adapter hash;
- prompt hash;
- score-code version;
- dtype and device policy.

Never accept a cache based on row count alone.

### 10.4 Temperature calibration

For each base or adapter:

- fit one positive temperature on calibration rows only;
- minimize binary NLL with a stable optimizer such as LBFGS;
- record calibration sample counts and source composition;
- report NLL and Brier score before and after calibration;
- do not use ECE as the sole calibration statistic;
- never fit calibration on transfer or stress sets.

### 10.5 Conservative operating point

Secondary operating point:

> maximize calibration recall subject to the one-sided 95% Clopper-Pearson upper bound on pooled calibration-negative FPR being at most 5%.

Exact procedure:

1. define the candidate set as +infinity plus every unique calibrated score;
2. predict unsafe when score is greater than or equal to the candidate threshold;
3. for each threshold, count FP and TN over all calibration negatives;
4. compute the one-sided 95% Clopper-Pearson bound:

\[
U_{CP}=
\mathrm{Beta}^{-1}(0.95; FP+1,TN)
\]

when TN is positive, with the standard exact edge-case definitions;
5. retain candidates with U_CP at most 0.05;
6. select maximum calibration recall;
7. break equal-recall ties by lower empirical FPR, then higher threshold;
8. if none is feasible, emit NO_FEASIBLE_THRESHOLD.

The candidate above the maximum observed score is included and predicts no calibration positives.

This pooled binomial calculation assumes exchangeable calibration negatives. Because the calibration set is source-stratified and may contain near-duplicate families, also report:

- FPR per represented source;
- equal-source-weighted macro FPR;
- a one-sided 95% upper bound from 10,000 global Poisson(1) family-weight replicates on macro FPR, using one weight per family across sources;
- sensitivity to the observed calibration source mixture.

Do not market this as a distribution-free or production guarantee. RQ4 remains descriptive even when the pooled Clopper-Pearson gate passes.

Report:

- selected threshold;
- calibration TP/FP/TN/FN;
- calibration FPR point estimate and upper bound;
- realized ID and transfer FPR with confidence intervals;
- TPR at the frozen threshold.

Do not describe realized test FPR as exactly matched.

RQ4 output is:

\[
\Delta TPR_R=
\operatorname{macroTPR}_R(SFT)-
\operatorname{macroTPR}_R(base)
\]

for represented and transfer regimes, accompanied by the realized macro FPR for both systems. "Directionally consistent" means the sign of Delta TPR_R matches the corresponding macro-AP delta and both systems satisfy the reported calibration procedure. This is a descriptive consistency label, not a confirmatory hypothesis test.

---

## 11. Metrics

### 11.1 Primary

1. tie-aware non-interpolated average precision per benchmark;
2. macro AP over represented-source benchmarks;
3. macro AP over transfer benchmarks;
4. direct base-to-SFT deltas.

Use sklearn.metrics.average_precision_score as the canonical implementation.

### 11.2 Secondary

- AUROC with correct tie handling;
- TPR at calibration-targeted 5% FPR;
- realized FPR;
- Brier score;
- log loss;
- source-level and worst-source effects;
- OR-Bench benign false-positive rate;
- HarmBench recall at the frozen threshold;
- prevalence-adjusted PPV sensitivity for clearly labeled hypothetical prevalences.

### 11.3 Descriptive only

- oracle best F1;
- pooled AP;
- native verdict accuracy;
- mean unsafe probability on a single-class set.

Do not place descriptive/oracle metrics in the headline.

---

## 12. Estimands and statistical analysis

### 12.1 Per-regime metric

For regime R, checkpoint b, and training seed r:

\[
M_R(b,r)=\frac{1}{|K_R|}
\sum_{k\in K_R}AP_k(b,r).
\]

### 12.2 Base-to-SFT change

\[
\Delta_R(b,r)=M_R(SFT_{b,r})-M_R(base_b).
\]

### 12.3 Fixed-panel aggregate

\[
\bar{\Delta}_R=
\frac{1}{4}
\sum_b
\left[
\frac{1}{5}\sum_rM_R(SFT_{b,r})-M_R(base_b)
\right].
\]

The primary result is the joint vector:

\[
\theta=
(\bar{\Delta}_{represented},\bar{\Delta}_{transfer}).
\]

Do not invent a single scalar specialization score unless it is clearly labeled as an optional visualization.

### 12.4 Uncertainty

Separate:

1. training-seed uncertainty;
2. conditional evaluation-set uncertainty.

Primary hierarchical paired-bootstrap algorithm:

1. use 10,000 replicates with RNG seed 20260712;
2. keep the four checkpoint identities fixed in every replicate;
3. within each checkpoint, sample five SFT training-seed indices with replacement;
4. draw one Poisson(1) bootstrap weight for every global family_id across all evaluation datasets;
5. apply that same family weight to every row belonging to the family, preserving cross-label and cross-dataset dependence plus paired base/SFT scores;
6. compute weighted tie-aware AP separately for every benchmark using those row weights;
7. macro-average benchmarks within represented and transfer regimes;
8. compute each checkpoint delta;
9. average the four checkpoint deltas without resampling checkpoint identities;
10. use the fifth percentile as the one-sided 95% lower bound and the 95th percentile as the one-sided 95% upper bound;
11. use the 2.5th and 97.5th percentiles for secondary two-sided 95% intervals.

This global Poisson cluster bootstrap deliberately does not stratify by label because scenario, contrast, and minimal-pair families may contain both labels, and a detected family may span datasets. If a weighted benchmark has zero effective weight for one class, redraw all family weights for that replicate rather than assigning a fabricated AP value. Record retry counts and the fraction of rejected replicates.

Report:

- all five seed values;
- mean, standard deviation, and interval;
- per-base direct delta intervals;
- fixed-panel aggregate intervals;
- leave-one-transfer-benchmark-out results;
- leave-one-base-out sensitivity, labeled descriptive rather than population inference.

Mechanical heterogeneity outputs:

- per-base represented and transfer deltas;
- per-benchmark deltas;
- range and standard deviation across bases;
- range and standard deviation across benchmarks;
- sign table across base-by-benchmark cells;
- every leave-one-benchmark-out aggregate;
- every leave-one-base-out aggregate.

The aggregate sign is called sensitivity-stable only when every leave-one-out estimate has the same sign as the full estimate. RQ3 remains descriptive; no post-hoc subgroup is promoted to a new confirmatory endpoint.

### 12.5 Precision and power

Before final LOCK.json, generate artifacts/paper_a_sft/design/power_report.json using only:

- aligned legacy pilot base/SFT logits and labels whose row text can be validated against the migration fixture;
- the frozen family structure;
- a prespecified training-seed standard-deviation grid of 0.01, 0.02, 0.04, and 0.06 macro AP;
- represented-source effect grid from +0.01 through +0.10 macro AP;
- transfer effect grid from -0.01 through -0.10 macro AP;
- checkpoint-heterogeneity scale grid of 0, 0.5, and 1.0 times the centered pilot checkpoint effects;
- represented/transfer seed-effect correlation grid of -0.5, 0, and +0.5.

The score-generating model is a paired empirical perturbation model. For checkpoint template b, row i, and seed r:

\[
z^{sim}_{b,r,i}
=
z^{base}_{b,i}
+
\widetilde d_{b,i}
+
\alpha_{b,R}(2y_i-1)
+
\eta_{b,r,R}(2y_i-1),
\]

where d tilde is the pilot tuned-minus-base residual after removing its label-conditional mean, R is represented or transfer, alpha controls class separation, and eta is a zero-mean correlated Gaussian seed effect in logit-separation space.

Construction:

1. validate and join pilot base/SFT rows by content hash; reject order-only joins;
2. convert probabilities to clipped logits when raw logits are unavailable;
3. preserve each pilot row label, base score, tuned-minus-base residual, source, and family ID;
4. subtract the mean residual separately within each regime and label to obtain d tilde, preserving row-level residual structure without fixing the pilot class-separation effect;
5. center pilot checkpoint AP effects within each regime;
6. for each desired mean AP effect and heterogeneity scale, numerically solve alpha_b,R by bisection so each checkpoint's expected AP delta equals the requested mean plus its scaled centered pilot deviation; increasing alpha monotonically increases positive-versus-negative separation while preserving within-label ordering;
7. numerically calibrate the standard deviation of eta so simulated across-seed macro-AP SD matches the requested 0.01/0.02/0.04/0.06 grid value;
8. draw represented and transfer eta jointly at the requested correlation;
9. draw one global Poisson(1) weight per family_id and apply it to all of that family's rows across datasets, preserving complete label composition and paired scores;
10. create five synthetic SFT seeds per checkpoint and reuse the base score;
11. apply the exact estimator, hierarchical interval, and intersection-union gate used by the final analysis.

At least two independently trained pilot checkpoint templates with valid paired row scores are required. If fewer exist, power_report.json must return INSUFFICIENT_PILOT_TEMPLATES and cannot authorize a formal power claim.

The validated pilot templates are copied into the design artifact tree with content hashes and provenance. They are used only for prospective design simulation, never as final Paper A evidence.

For each grid cell:

1. simulate the four fixed checkpoint positions and five seeds per checkpoint;
2. use represented and transfer dataset family-cluster structures separately;
3. apply the exact primary estimator and bootstrap gate;
4. estimate power over at least 5,000 simulated studies;
5. report marginal Gate A power, marginal Gate B power, and joint intersection-union power.

Report:

- minimum detectable effect at 80% power for each seed-variance scenario;
- expected interval width;
- sensitivity for three, five, and seven seeds;
- marginal and joint power;
- sensitivity to checkpoint heterogeneity and cross-regime seed correlation;
- the practical effect size considered meaningful.

The prespecified meaningful effect is 0.03 macro AP. This is an engineering/research relevance margin, not a universal constant: it is large enough to exceed table-rounding noise and small legacy metric shifts, while being comparable to differences that would plausibly alter a model-selection conclusion. The report must also show 0.02 and 0.05 sensitivity so reviewers can evaluate this judgment.

Decision rule:

- use 0.04 macro AP seed SD, full observed checkpoint heterogeneity scale, and zero represented/transfer seed-effect correlation as the central scenario;
- five seeds remains the confirmatory design only if joint intersection-union power is at least 80% for the effect pair (+0.03 represented, -0.03 transfer) under the central scenario;
- if pilot templates are insufficient or joint power is lower, either increase to seven seeds before final lock or explicitly make the study precision-focused and treat Gates A/B as descriptive interval checks rather than powered confirmatory tests;
- never increase seeds after viewing final ID or transfer results.

For this minimal-refactor plan, the recommended fallback is the precision-focused five-seed study rather than expanding to 28 adapters. If seven seeds are selected instead, update the configuration, run matrix, power report, artifact-completeness tests, compute budget, manuscript wording, and every 20/20 acceptance gate before final lock.

### 12.6 Multiplicity

Define one primary family:

- represented-source macro-AP delta;
- transfer macro-AP delta.

The specialization claim is an intersection-union test: both one-sided component nulls must be rejected at alpha 0.05. This controls the joint specialization claim at alpha 0.05.

If either component is claimed independently outside the joint specialization statement, apply Holm correction across the two component tests. Secondary per-source and operating-point comparisons are descriptive unless a separate family and correction are preregistered.

### 12.7 What not to do

- do not use McNemar as a test of AP or F1;
- do not infer significance from non-overlapping marginal confidence intervals;
- do not treat five seeds as five model families;
- do not pool sources and call the result source-balanced;
- do not select a benchmark, seed, base, or threshold after viewing final results.

---

## 13. Minimal code implementation

### 13.1 Proposed file tree

~~~text
configs/paper_a_sft.yaml

guard_research/__init__.py
guard_research/metrics.py
guard_research/thresholds.py
guard_research/provenance.py
guard_research/prompts.py

experiments/prepare_paper_a_manifests.py
experiments/audit_paper_a_splits.py
experiments/prepare_paper_a_power_templates.py
experiments/design_power_paper_a_sft.py
experiments/lock_paper_a_sft.py
experiments/run_paper_a_sft.py
experiments/eval_paper_a_sft.py
experiments/analyze_paper_a_sft.py
experiments/validate_paper_a_sft.py

tests/test_metrics.py
tests/test_manifests.py
tests/test_thresholds.py
tests/test_cache_alignment.py
tests/test_prompt_token_parity.py
tests/test_lock_enforcement.py
tests/test_run_completeness.py
tests/test_claim_gates.py
~~~

Modify:

~~~text
experiments/train_guard.py
notebooks/requirements.txt
paper/benchmark_chooses_the_winner.tex
paper/refs.bib
~~~

Do not use as final-evidence paths:

~~~text
experiments/hpo_guard.py
experiments/train_guard_pref.py
experiments/stage2.sh
experiments/aggregate_clean_sweep.py
notebooks/outputs/*
~~~

They remain migration or pilot references only.

### 13.2 Configuration

configs/paper_a_sft.yaml must include:

~~~yaml
schema_version: 1
study_id: paper_a_sft
data_branch: academic_noncommercial
data_seed: 42
data_order_seed: 42
rows_per_source: 400
rows_per_source_label: 200
train_sources:
  - toxicchat
  - prompt_injections
  - jailbreak_classification
excluded_train_sources:
  - beavertails
  - or_bench
seeds: [42, 43, 44, 45, 46]
max_steps: 300
max_length: 1024
effective_batch: 4
learning_rate: 0.0002
warmup_ratio: 0.03
lora:
  r: 32
  alpha: 64
  dropout: 0.05
target_fpr: 0.05
primary_metric: macro_average_precision
bootstrap_replicates: 10000
bootstrap_seed: 20260712
~~~

Model and tokenizer revisions must be explicit, not REQUIRED placeholders.

### 13.3 Trainer changes

experiments/train_guard.py must:

- accept explicit CLI arguments;
- read only the frozen train manifest;
- stop downloading or resampling live datasets;
- pin the model/tokenizer revision;
- record all hashes and run metadata;
- expose smoke steps without changing the data composition;
- never read calibration, ID, transfer, or stress paths;
- write deterministic completion status;
- keep failed runs rather than silently deleting evidence.
- import prompt rendering, decision-token selection, and EOS-label construction from guard_research/prompts.py.

Required invocation shape:

~~~text
python experiments/train_guard.py \
  --manifest artifacts/paper_a_sft/manifests/train.jsonl \
  --lock artifacts/paper_a_sft/LOCK.json \
  --model-key smollm3_3b \
  --seed 42 \
  --out artifacts/paper_a_sft/runs/smollm3_3b/sft/seed_42
~~~

### 13.4 Tests

#### test_metrics.py

- AP equals sklearn on random and tied arrays;
- AUROC equals sklearn;
- permutations within tied-score groups do not change results;
- single-class behavior is explicit.

#### test_manifests.py

- no exact train/evaluation overlap;
- known four exact-overlap rows are absent from training;
- OR-Bench and BeaverTails counts in training are zero;
- every row has provenance, family, and hash fields;
- joins are one-to-one.

#### test_thresholds.py

- test labels cannot change calibration thresholds;
- selected threshold is optimal among candidates satisfying the conservative bound;
- infeasible targets return NO_FEASIBLE_THRESHOLD;
- tied score thresholds are handled consistently.

#### test_cache_alignment.py

- reordered rows invalidate a cache;
- changed prompt/model/adapter/content hashes invalidate a cache;
- exact matching artifacts are reused;
- duplicate or missing sample IDs fail closed.

#### test_prompt_token_parity.py

- training and scoring render byte-identical prompt prefixes;
- the leading-space or no-leading-space decision convention is identical;
- verdict token IDs match the locked values;
- completion loss covers the verdict token and EOS exactly;
- truncation metadata is correct.

#### test_lock_enforcement.py

- absent or changed lock fields stop training, scoring, and analysis;
- smoke configuration cannot authorize final runs;
- final result paths reject a dirty or mismatched manifest/config according to the recorded policy.

#### test_run_completeness.py

- the final matrix requires exactly four bases by five seeds;
- failed/missing/duplicate cells are explicit;
- final scoring cannot begin before 20/20 valid cells exist;
- base score artifacts exist exactly once per checkpoint.

#### test_claim_gates.py

- synthetic positive, negative, heterogeneous, and null cases produce the required wording status;
- leave-one-out sign instability blocks a stable transfer claim;
- Holm and intersection-union logic match the registered primary family;
- RQ4 remains descriptive.

---

## 14. Lock and artifacts

### 14.1 LOCK.json

The lock must be created after manifests/tests/smoke validation and before final training.

It records:

- Git SHA and dirty-state policy;
- model/tokenizer revisions;
- data revisions and manifest hashes;
- source inclusions/exclusions;
- license branch;
- prompt template/hash;
- training recipe;
- seeds;
- metrics;
- target FPR and confidence method;
- primary contrasts;
- analysis_mode selected from powered_confirmatory or precision_focused;
- power-report hash and seed-count decision;
- statistical resampling rules;
- table and figure specifications;
- failure handling;
- artifact paths.

Final-evaluation code must refuse an absent or mismatched lock.

### 14.2 Artifact tree

~~~text
artifacts/paper_a_sft/
  LOCK.json
  manifests/
  audit/
  design/
    pilot_power_templates.parquet
    pilot_power_templates.metadata.json
    power_report.json
  runs/
    qwen25_15b/sft/seed_42/
    ...
    qwen3_4b/sft/seed_46/
  base_scores/
  scores/
    scores.parquet
    metadata.json
  calibration/
  analysis/
    results.json
    claim_checks.json
    seed_values.csv
    per_benchmark.csv
    sensitivity.json
    report.md
    tables/
    figures/
  release/
    DATA_CARD.md
    MODEL_CARD.md
    THIRD_PARTY_DATA.md
    CHECKSUMS.sha256
~~~

Current ignored notebooks/outputs artifacts may be read only for migration checks. No final paper number may depend solely on them.

---

## 15. Manuscript refactor

### 15.1 New main-paper structure

Target 9–11 main-text pages plus a compact appendix.

1. **Introduction**
   - why post-adaptation leaderboards hide what tuning changed;
   - one research question;
   - three contributions;
   - result-neutral thesis.

2. **Related Work**
   - small LLM guard classifiers;
   - fine-tuning degradation and benchmark/policy transfer;
   - calibration and operating points;
   - explicit novelty boundary.

3. **Controlled Study Design**
   - prompt-safety task and native label mappings;
   - manifest construction and exclusions;
   - four-checkpoint panel;
   - frozen SFT recipe;
   - represented/transfer/stress regimes;
   - estimands and statistical protocol.

4. **Results**
   - primary checkpoint result;
   - four-base/five-seed fixed-panel result;
   - per-base/per-benchmark heterogeneity;
   - secondary 5% FPR analysis;
   - stress-set results.

5. **Limitations**
   - heterogeneous native policies;
   - fixed two-lineage panel;
   - previously inspected benchmark datasets;
   - balanced challenge-set prevalence;
   - non-commercial/gated data;
   - prompt-only scope;
   - finite seeds and evaluation datasets.

6. **Conclusion**
   - adaptation is not a guaranteed uniform upgrade;
   - always evaluate base and adapted guard across represented and transfer regimes.

Appendix:

- full seed values;
- full per-source tables;
- model/tokenizer/prompt hashes;
- calibration diagnostics;
- audit summaries;
- implementation/hardware;
- optional open-guard sanity comparison, only if recomputed cleanly.

### 15.2 Exact current-manuscript map

| Current TeX lines | Action |
|---|---|
| 50 | replace title |
| 61–65 | replace abstract completely |
| 69–108 | retain core motivation; remove mortgage/GPT/leaderboard framing and old numbers |
| 109–117 | replace five contributions with three |
| 119–138 | compress and update Related Work |
| 139–188 | keep pipeline concept; redraw around manifest, audit, lock, train, represented, transfer, stress |
| 190–220 | keep and compress the single-token scoring formulation |
| 221–278 | keep model and SFT recipe; update panel, seeds, data, and licensing |
| 279–350 | keep one rewritten benchmark-role table; merge duplicate pool diagrams |
| 351–384 | keep metrics/decontamination; replace with final estimands and audit |
| 385–407 | move implementation details to appendix |
| 413–466 | remove open-guard leaderboard from main paper |
| 467–518 | reduce operating-point material to one secondary subsection |
| 519–552 | merge held-out results into primary base-versus-SFT results |
| 553–583 | remove GPT parity and cross-substrate latency claim |
| 584–614 | replace ranking-flip discussion with joint represented/transfer interpretation |
| 615–658 | promote and regenerate as empirical core |
| 659–677 | remove theoretical rationale |
| 678–707 | replace six-base convergence with four-base/five-seed fixed-panel result |
| 708–762 | remove DPO/GRPO/HPO results |
| 763–790 | remove ensemble from main paper; future-work sentence only |
| 791–840 | move mortgage material to Paper B |
| 841–877 | retain only Paper A limitations; delete revision-history and removed-system caveats |
| 878–885 | remove guardrail landscape section |
| 886–910 | rewrite conclusion and reduce future work |
| 911 onward | replace rank-preservation appendix with provenance/statistical appendix |

### 15.3 Proposed abstract template

Do not insert final numbers until claim_checks.json passes.

> Small language models are increasingly adapted into binary prompt-safety guards, but improvement on benchmark sources represented during training does not establish transferable guard capability. We measure what LoRA-SFT adds beyond the untuned base using paired base-versus-adapted comparisons on four checkpoints from 1.5B to 4B parameters. We hold the prompt interface, training manifest, LoRA capacity, and evaluation rows fixed; repeat training across five seeds; and separate represented-source tests from dataset-held-out benchmark transfer. Evaluation uses macro average precision, per-source effects, calibration-only operating points, and uncertainty over both training runs and evaluation datasets.
>
> Under the locked protocol, SFT changes represented-source macro AP by [RESULT AND INTERVAL] and transfer macro AP by [RESULT AND INTERVAL]. Effects are [stable/heterogeneous] across checkpoints and benchmarks. At a calibration-targeted 5% FPR, [OPERATING-POINT RESULT], with realized test FPR reported rather than assumed.
>
> Prior work has established policy overfitting, calibration sensitivity, and safety degradation after fine-tuning. Our contribution is narrower: a same-checkpoint, same-training-manifest, multi-seed attribution of how compact prompt guards specialize across benchmark sources. The results show that adapting a guard is [FINAL GATED INTERPRETATION].

### 15.4 Related-work positioning

Mandatory direct comparisons:

- GuardBench: general instruction models can rival specialized guards.
- Domain Generalizable AI Guardrails with Augmented Policy Training: policy overfitting and unseen-policy adaptation.
- Challenges and Remedies of Domain-Specific Classifiers as LLM Guardrails: classifier drift and false refusal.
- Prompt Injection Detection is Regime-Dependent: ranking and deployment regime instability.
- FlexGuard: continuous scores, calibration, and strictness adaptation.
- Objective Matters: objective-dependent safety and robustness.
- Guardrails in Logit Space / Safety Token Regularization: decision-token preservation.
- guard-model fine-tuning collapse and preservation work.

Positioning sentence:

> Unlike work proposing new policy augmentation, calibration, or preservation objectives, we isolate the before-versus-after effect of a common SFT guard recipe on the same compact checkpoints and report represented-source and dataset-held-out changes separately.

### 15.5 Tables

Use four main tables.

#### Table 1 — data and manifest roles

- source;
- label origin;
- train/calibration/ID/transfer/stress role;
- post-audit count;
- license;
- revision/hash.

#### Table 2 — fixed model and SFT recipe

- checkpoint/revision;
- parameters;
- decision tokens;
- LoRA configuration;
- five seeds;
- training tokens/time.

#### Table 3 — primary fixed-panel result

For each base:

- base represented macro AP;
- mean SFT represented macro AP;
- represented delta and interval;
- base transfer macro AP;
- mean SFT transfer macro AP;
- transfer delta and interval;
- five individual seed values.

Include the fixed-panel aggregate last.

#### Table 4 — per-benchmark and operating-point sensitivity

- per-source paired delta;
- TPR at calibration-targeted 5% FPR;
- realized FPR;
- OR-Bench benign FPR;
- HarmBench recall.

Place full calibration and seed tables in the appendix.

### 15.6 Figures

#### Figure 1 — study design

~~~text
source snapshots
    ↓
manifest build → overlap/license audit → LOCK
    ↓
four bases → five SFT seeds each
    ↓
represented-source | held-out transfer | stress sets
    ↓
paired deltas and claim gates
~~~

#### Figure 2 — specialization plane

- x-axis: represented-source macro-AP delta;
- y-axis: transfer macro-AP delta;
- one color per checkpoint;
- one point per seed;
- zero lines create four interpretation quadrants;
- fixed-panel mean shown separately.

#### Figure 3 — optional per-benchmark forest plot

- direct paired AP delta;
- confidence interval;
- benchmarks grouped by represented/transfer;
- no hand-transcribed coordinates.

Remove the current GPT Pareto, ranking-flip, mortgage, open-guard leaderboard, and ensemble figures from Paper A.

---

## 16. Claim gates and result decision tree

LOCK.json records analysis_mode as either powered_confirmatory or precision_focused.

- In powered_confirmatory mode, the inferential gates below control paper claims.
- In precision_focused mode, compute the same intervals and sensitivities, but replace "established/improved/reduced" with estimation language such as "the estimated change was ..."; do not report formal gate power or rejection claims.

### 16.1 Gate A — represented-source improvement

Allowed only if:

\[
LCB_{95}(\bar{\Delta}_{represented})>0.
\]

This is the represented-source component of the joint intersection-union specialization test. If Gate B fails and represented-source improvement is stated as a standalone inferential claim, claim_checks.json must additionally require that the component survive Holm correction across Gates A and B.

Then say:

> For this fixed panel and recipe, SFT improved represented-source macro AP.

Otherwise say:

> The study did not establish a panel-wide represented-source improvement.

### 16.2 Gate B — transfer degradation

Allowed only if:

\[
UCB_{95}(\bar{\Delta}_{transfer})<0
\]

and:

- leave-one-transfer-benchmark-out analysis does not reverse the aggregate sign;
- every leave-one-base-out aggregate preserves the full transfer-delta sign.

This is the transfer component of the joint intersection-union specialization test. If Gate A fails and transfer degradation is stated as a standalone inferential claim, claim_checks.json must additionally require that the component survive Holm correction across Gates A and B.

Then say:

> For this fixed panel and recipe, SFT reduced dataset-held-out transfer macro AP.

Otherwise report the measured heterogeneity.

### 16.3 Specialization trade-off claim

Allowed only if Gate A and Gate B both pass.

Then say:

> The fixed panel exhibits an in-source/transfer specialization trade-off under the frozen LoRA-SFT recipe.

Do not say:

- fine-tuning universally hurts OOD;
- SFT is intrinsically non-robust;
- all guard models specialize;
- a causal mechanism has been proven.

### 16.4 Heterogeneity outcome

If represented-source gains are positive but transfer effects differ across models or benchmarks:

> SFT provides represented-source gains, while transfer effects are checkpoint- and benchmark-dependent.

This remains a valid paper result and should not trigger post-hoc benchmark removal.

### 16.5 Null outcome

If neither represented nor transfer effects are stable:

- report the multi-seed null result;
- narrow the manuscript to evaluation variability and artifact lessons only if that framing was preregistered;
- do not restore invalid legacy objective, mortgage, or ensemble claims to manufacture a positive result.

---

## 17. Reproduction commands

These commands specify the desired final interface. Implement them before final execution.

### Prepare and audit

~~~bash
python experiments/prepare_paper_a_manifests.py \
  --config configs/paper_a_sft.yaml \
  --out artifacts/paper_a_sft/manifests

python experiments/audit_paper_a_splits.py \
  --config configs/paper_a_sft.yaml \
  --manifest artifacts/paper_a_sft/manifests/manifest.json \
  --out artifacts/paper_a_sft/audit
~~~

### Tests

~~~bash
pytest -q \
  tests/test_metrics.py \
  tests/test_manifests.py \
  tests/test_thresholds.py \
  tests/test_cache_alignment.py \
  tests/test_prompt_token_parity.py \
  tests/test_lock_enforcement.py \
  tests/test_run_completeness.py \
  tests/test_claim_gates.py
~~~

### Pre-lock smoke

~~~bash
python experiments/run_paper_a_sft.py smoke \
  --config configs/paper_a_sft.yaml \
  --manifest artifacts/paper_a_sft/manifests/manifest.json \
  --all-models
~~~

Smoke outputs must be stored outside final run directories, must not satisfy a final-cell check, and may score only synthetic or calibration-only fixtures.

### Final lock

~~~bash
python experiments/prepare_paper_a_power_templates.py \
  --legacy-root notebooks/outputs \
  --manifest artifacts/paper_a_sft/manifests/manifest.json \
  --out artifacts/paper_a_sft/design/pilot_power_templates.parquet

python experiments/design_power_paper_a_sft.py \
  --config configs/paper_a_sft.yaml \
  --manifest artifacts/paper_a_sft/manifests/manifest.json \
  --pilot-scores artifacts/paper_a_sft/design/pilot_power_templates.parquet \
  --out artifacts/paper_a_sft/design/power_report.json

python experiments/lock_paper_a_sft.py \
  --config configs/paper_a_sft.yaml \
  --manifest artifacts/paper_a_sft/manifests/manifest.json \
  --audit artifacts/paper_a_sft/audit/audit.json \
  --power artifacts/paper_a_sft/design/power_report.json \
  --out artifacts/paper_a_sft/LOCK.json
~~~

### Train

~~~bash
python experiments/run_paper_a_sft.py train \
  --lock artifacts/paper_a_sft/LOCK.json \
  --seeds 42 43 44

python experiments/run_paper_a_sft.py validate-runs \
  --lock artifacts/paper_a_sft/LOCK.json

python experiments/run_paper_a_sft.py train \
  --lock artifacts/paper_a_sft/LOCK.json \
  --seeds 45 46
~~~

validate-runs may inspect only run metadata, adapter loadability, hashes, and synthetic/calibration-only fixtures. It must not open ID, transfer, or stress manifests.

### Score, analyze, and validate

~~~bash
python experiments/eval_paper_a_sft.py \
  --lock artifacts/paper_a_sft/LOCK.json \
  --out artifacts/paper_a_sft/scores

python experiments/analyze_paper_a_sft.py \
  --lock artifacts/paper_a_sft/LOCK.json \
  --scores artifacts/paper_a_sft/scores/scores.parquet \
  --out artifacts/paper_a_sft/analysis

python experiments/validate_paper_a_sft.py \
  --root artifacts/paper_a_sft
~~~

### Compile

~~~bash
make -C paper clean
make -C paper
~~~

No table or figure may require manually copying a result from stdout.

---

## 18. Schedule

### Days 1–2 — scope, licensing, and data

- freeze title, thesis, RQs, and excluded claims;
- select licensing branch;
- implement configuration and manifest builder;
- rebuild represented/transfer/stress splits;
- remove known overlaps;
- run near-duplicate and license audit;
- verify model/tokenizer revisions.
- construct family IDs and freeze the clustering/adjudication rules.

Exit gate:

- audit passes;
- final row counts known;
- license strategy documented;
- no final training has started.

### Days 3–4 — code and tests

- refactor trainer to manifest-only input;
- implement canonical metrics/thresholds;
- implement keyed evaluator and cache validation;
- implement the complete metric/manifest/threshold/cache/prompt/lock/run/claim test set;
- implement power design, lock, and runner;
- run one pre-lock smoke per base on synthetic/calibration-only fixtures;
- generate the power report and then create final LOCK.json.

Exit gate:

- tests pass;
- every model loads;
- decision tokens verified;
- score joins complete;
- smoke is reproducible.
- final LOCK.json exists only after smoke and power review.

### Days 5–6 — first twelve locked final cells

- train 12 cells;
- validate adapter loadability, run metadata, hashes, and synthetic/calibration-only scoring;
- do not open or score ID, transfer, or stress manifests;
- do not compute provisional headline metrics.

Exit gate:

- 12/12 locked adapters valid;
- artifact schemas stable;
- no data leakage;
- no cache mismatches.

### Days 7–8 — complete five-seed matrix

- train seeds 45 and 46;
- require 20/20 final cells;
- only now score four bases and twenty adapters on final manifests;
- freeze score artifacts.

Exit gate:

- all runs complete;
- hashes recorded;
- no test-selected thresholds;
- score table joins one-to-one.

### Days 9–10 — statistics and figures

- run paired intervals;
- run leave-one-benchmark/base sensitivity;
- compute 5% FPR secondary results;
- generate four tables and two or three figures;
- produce claim_checks.json and analysis report.

### Days 11–12 — manuscript and release

- rewrite title, abstract, introduction, contributions, results, limitations, and conclusion;
- move mortgage material to Paper B;
- remove invalid objective/GPT/ensemble/theory claims;
- update citations;
- render and inspect PDF;
- run cached fresh-clone reproduction;
- complete data/model cards and checksums.

Expected duration: approximately two focused working weeks. Once the pipeline is stable, four GPUs should complete the machine-heavy stage within roughly one day. Validate actual timing with the four-base smoke rather than relying on the current estimate.

---

## 19. Compute and storage

Provisional estimate:

- 20 final SFT training runs;
- four base scoring bundles;
- twenty adapter scoring bundles;
- approximately 8–18 A100-equivalent GPU-hours including scoring and retry reserve;
- approximately 4–5 GB for final LoRA adapters;
- approximately 5–15 GB for score tables, logs, and metadata;
- additional model-cache storage according to local Hugging Face snapshots.

Record:

- wall time;
- device model;
- peak allocated memory;
- tokens processed;
- energy proxy if available;
- retry/failure counts.

Do not compare local and API latency as a central Paper A result.

---

## 20. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| transfer effect disappears across seeds | medium | high | use predeclared heterogeneity/null wording |
| represented-source gain depends on BeaverTails | medium | high | exclude BeaverTails and accept the clean result |
| ToxicChat release restriction blocks weight release | high | medium | non-commercial branch or permissive-only rerun |
| near-duplicate audit removes more data | medium | medium | freeze post-audit counts; do not silently replace rows |
| Prompt Injections n is too small | high | medium | report wide interval and leave-one-source sensitivity |
| Qwen3-4B run failures | medium | medium | smoke first; preserve visible failures; rerun before aggregate |
| single-token assumption fails | low | high | hard assertion before training |
| calibration cannot certify 5% FPR | medium | medium | emit NO_FEASIBLE_THRESHOLD; keep AP primary |
| current legacy numbers tempt selective reuse | high | high | block them from final analyzer and abstract |
| reviewers call novelty incremental | high | medium | emphasize controlled multi-seed attribution and precise scope |
| fixed panel is too narrow | high | medium | explicitly restrict claims; broader architectures are future work |
| gated OOD data cannot be redistributed | high | medium | publish IDs/hashes/download instructions rather than raw text |

---

## 21. Acceptance checklist

### Scope and novelty

- [ ] One research question drives the abstract, introduction, results, and conclusion.
- [ ] Mortgage, GPT parity, open-guard leaderboard, DPO/GRPO, ensemble, and theory are not contributions.
- [ ] The paper says dataset-held-out benchmark transfer, not fixed-policy or source-family shift.
- [ ] Claims are restricted to the fixed four-checkpoint panel and frozen SFT recipe.
- [ ] Related work explicitly credits prior policy-overfitting, calibration, and preservation work.

### Data

- [ ] Final licensing branch is recorded.
- [ ] BeaverTails and OR-Bench counts in primary training are zero.
- [ ] Four known WildJailbreak exact-overlap rows are absent from training.
- [ ] The final training manifest contains the locked 1,200 rows with 400 per source and 200 per source-label stratum.
- [ ] Exact train/evaluation overlap is zero.
- [ ] Near-duplicate candidates are adjudicated.
- [ ] Family IDs follow the frozen upstream-ID/MinHash connected-component algorithm.
- [ ] Every row has revision, provenance, family ID, and hash.
- [ ] OR-Bench-Hard is not used as a confounded AP benchmark.
- [ ] HarmBench is stress-only.

### Models and training

- [ ] All four model/tokenizer revisions are immutable.
- [ ] Decision tokens are distinct and single-token.
- [ ] Prompt template and hash are frozen.
- [ ] Training and scoring pass prompt/token/EOS parity tests.
- [ ] Training reads one immutable manifest.
- [ ] Every run uses the same data_order_seed and records a separate training_seed.
- [ ] Twenty of twenty final adapters are present and validated.
- [ ] Failed cells are not silently omitted.
- [ ] No final training path can read evaluation labels.

### Scoring and statistics

- [ ] Base scores are computed once per checkpoint.
- [ ] Scores are keyed by sample/content/model/adapter/prompt hashes.
- [ ] AP/AUROC match sklearn and pass tie-permutation tests.
- [ ] Calibration uses calibration rows only.
- [ ] Test labels cannot affect thresholds.
- [ ] All five seeds are reported.
- [ ] The power/precision report was generated before final lock.
- [ ] Direct paired delta intervals support every inferential claim.
- [ ] Seed and evaluation uncertainty are separated.
- [ ] Leave-one-benchmark and leave-one-base sensitivities are reported.
- [ ] No primary result uses oracle F1.
- [ ] ID/transfer/stress scoring began only after 20/20 final adapters passed completeness checks.

### Reproduction

- [ ] LOCK.json matches every producing artifact.
- [ ] Final results do not rely on ignored notebooks/outputs.
- [ ] Tables and figures are generated from score/statistical artifacts.
- [ ] Fresh-clone cached analysis reproduces every paper number.
- [ ] Environment versions and transitive dependencies are frozen.
- [ ] Data card, model card, third-party data notice, and checksums exist.

### Manuscript

- [ ] Abstract contains only locked final values.
- [ ] Introduction has three contributions, not five.
- [ ] Results lead with base-to-SFT deltas.
- [ ] Old 0.884 versus 0.780 pilot evidence is not presented as final.
- [ ] Current mortgage sections are moved to Paper B.
- [ ] Objective/HPO and ensemble tables are removed.
- [ ] Limitations include heterogeneous labels, prior benchmark inspection, licensing, finite seeds, balanced prevalence, and prompt-only scope.
- [ ] Conclusion follows claim_checks.json.
- [ ] PDF compiles without warnings that affect references, tables, or figures.

---

## 22. Definition of done

Paper A is complete when:

1. the data and model configuration is immutable;
2. every overlap/license decision is documented;
3. 20/20 SFT adapters and four base score bundles are valid;
4. all metrics and thresholds pass automated tests;
5. one keyed score table can regenerate every result;
6. claim_checks.json mechanically determines allowed paper language;
7. the main manuscript contains one coherent base-to-guard story;
8. a fresh clone can reproduce the cached analysis and compiled paper;
9. no claim depends on the invalid legacy HPO, mortgage, ensemble, or pre-correction result paths;
10. negative or heterogeneous results are reported according to the predeclared decision tree.

---

## 23. Final go/no-go

### Go for implementation

The current repository has enough raw data, model access, SFT code, and pilot evidence to implement this study without inventing a new method or collecting a large new benchmark.

### No-go for final training until

- the data/licensing branch is chosen;
- manifests and audit exist;
- exact and near-duplicate checks pass;
- model/tokenizer/prompt hashes are frozen;
- metrics, thresholds, and cache-alignment tests pass;
- LOCK.json exists.

### No-go for submission until

- 20/20 final cells are complete;
- final tables/figures are generated from keyed artifacts;
- claims pass direct intervals and sensitivity checks;
- fresh-clone reproduction succeeds;
- the manuscript is reduced to the single fixed-panel base-to-SFT message.

The central discipline is simple:

> preserve the current guard formulation and training recipe, but rebuild the evidence chain and remove every secondary thesis.
