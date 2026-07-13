# Paper A — Measuring the Specialization Frontier of Small LLM Safety Guards

## Document purpose

This is an executable research and implementation specification for the general-guard paper. It is deliberately stricter than the current manuscript: every proposed claim has a named dataset, producing program, statistical test, artifact, and go/no-go criterion.

The existing results are development evidence only. They are useful for sizing the study and detecting failure modes, but they are not the final evidence because some runs used OR-Bench in training, some metrics were historically tie-incorrect, the objective HPO inspected the OOD tests, thresholds were sometimes selected on test labels, and most training cells have one seed.

Recommended title:

> **Measuring the Specialization Frontier of Small LLM Safety Guards under Source Shift**

Recommended thesis:

> Adapting an instruction model into a fixed-policy safety guard does not uniformly improve detection. It moves the model along an in-distribution versus source-shift frontier; the movement depends on the base model and objective, and can be moderated by an explicit decision-head preservation penalty.

The paper is about fixed-policy, input-prompt moderation under source-family shift. It is not a mortgage paper, a frontier-model parity paper, or a claim about every form of domain/policy generalization.

---

## 1. Direct experimental decision

### 1.1 Main matrix

Use four checkpoints that the current code can already load and score:

| Key | Immutable model identifier | Role |
|---|---|---|
| `qwen25_15b` | `Qwen/Qwen2.5-1.5B-Instruct` | small cross-family base |
| `smollm2_17b` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | small same-vendor predecessor |
| `smollm3_3b` | `HuggingFaceTB/SmolLM3-3B` | primary checkpoint |
| `qwen3_4b` | `Qwen/Qwen3-4B` | larger small-model checkpoint |

Pin the exact model and tokenizer revision in the experiment lockfile. Do not use moving `main` revisions.

Treatments:

1. Untuned base: score once per checkpoint; it has no training seed.
2. LoRA-SFT: five seeds on all four checkpoints.
3. LoRA-DPO: five seeds on the two anchor checkpoints, SmolLM3-3B and Qwen3-4B.
4. Decision-head-preserving KL-SFT: five seeds on the same two anchor checkpoints.

This is 40 final adapters:

\[
4\text{ bases}\times 5\text{ SFT seeds}
+2\text{ anchors}\times 5\text{ DPO seeds}
+2\text{ anchors}\times 5\text{ KL-SFT seeds}=40.
\]

This matrix is preferable to six bases × three objectives × three seeds. It gives stronger seed inference, includes the proposed optimization, and avoids incomplete 8B-DPO and qualitatively different reasoning-distilled cells.

Minimum defensible fallback if compute is constrained: seeds 42, 43, and 44, for 24 adapters. Any three-seed result must be described as lower-power.

### 1.2 Excluded from the main matrix

- DeepSeek-R1-Distill-Qwen-1.5B: reasoning-token behavior changes the task and prompt path.
- Qwen3-8B: DPO does not fit the current 40 GB execution path, leaving an incomplete factorial design.
- GRPO: a single binary verdict offers weak within-group reward variation, so the current GRPO formulation is not a credible optimization baseline.
- GPT parity: a prompted API decision is not required to answer the paper's causal question.
- Mortgage experiments: move to Paper B.
- The current rank-preservation theorem: it does not prove the empirical specialization effect.

---

## 2. What is already available

The following inventory was verified in this checkout.

### 2.1 General training data

The existing shared training pool contains 4,279 rows:

| Source | Safe | Unsafe | Total | Paper A use |
|---|---:|---:|---:|---|
| BeaverTails | 600 | 600 | 1,200 | train, after prompt-label audit |
| Jailbreak Classification | 515 | 503 | 1,018 | train |
| Prompt Injections | 203 | 203 | 406 | train |
| ToxicChat | 330 | 325 | 655 | train only if license/release policy permits |
| OR-Bench | 1,000 | 0 | 1,000 | **remove from train** |

Removing OR-Bench produces a clean candidate pool of 3,279 rows, almost exactly label-balanced: 1,648 safe and 1,631 unsafe. Therefore, removing OR-Bench does not create a class-balance problem and requires no replacement dataset.

Existing pool examples are stored under `notebooks/outputs/*-guard/train_pool.json`. These files establish feasibility, but a new manifest must be rebuilt from pinned source revisions because the primary SmolLM3 train pool is not committed as a canonical artifact.

### 2.2 Existing frozen evaluation material

[`frozen_eval_rows.json`](../notebooks/outputs/frozen_eval_rows.json) contains:

| Role | Rows | Positive | Notes |
|---|---:|---:|---|
| current ID development | 1,102 | 563 | must be split into tuning and calibration |
| current combined test | 2,018 | 998 | source names are available |
| WildGuardTest | 800 | 400 | candidate source-family OOD |
| WildJailbreak | 420 | 210 | candidate source-family OOD |
| OR-Bench-Hard hybrid | 800 | 400 | source-family OOD only after OR-Bench is removed from train |
| HarmBench | 200 | 200 | recall stress only; AP/AUROC undefined |

The 2,018-row test decomposes into:

- seen-source ID test: BeaverTails 1,041, ToxicChat 401, Prompt Injections 68, and Jailbreak Classification 148; total 1,658;
- transfer diagnostic: JailbreakBench 120 and XSTest 240; total 360.

The frozen file lacks complete row provenance and family IDs. It is a regression fixture, not the final manifest.

### 2.3 Existing models, scripts, and pilot results

Usable implementation foundations:

- [`train_guard.py`](../experiments/train_guard.py): completion-only LoRA-SFT, parameterized base, seed, rank, learning rate, warmup, and OR-Bench cap.
- [`train_guard_pref.py`](../experiments/train_guard_pref.py): DPO and other preference objectives using the same broad prompt formulation.
- [`guard_eval_pipeline.py`](../experiments/guard_eval_pipeline.py): base/tuned scoring and current tie-aware AP/AUROC implementation.
- [`stage2.sh`](../experiments/stage2.sh): in-progress clean, multi-seed orchestration scaffolding.
- [`aggregate_clean_sweep.py`](../experiments/aggregate_clean_sweep.py): in-progress result collection scaffolding.
- [`expguard_eval.py`](../experiments/expguard_eval.py): optional locked external confirmation on ExpGuardTest.

Pilot numerical anchors, not publication claims:

- current tie-aware OOD aggregate: base approximately 0.884 AP versus SFT approximately 0.780;
- current SFT ID result is substantially above the untuned base;
- current single-run DPO result suggests that an objective can preserve more OOD ranking than SFT;
- existing 120-step HPO trials take roughly 6–9 minutes each on an A100, including scoring.

No `summary_*-clean-*-s*.json` files currently exist. The clean multi-seed matrix has not yet produced final evidence.

---

## 3. Research questions, estimands, and claim gates

### RQ1 — What does SFT change?

For the fixed four-checkpoint panel, how does a common LoRA-SFT recipe change discrimination on seen-source ID data and genuinely training-family-unseen data?

For base \(b\), objective \(o\), seed \(r\), and benchmark set \(k\), define:

\[
M_k(b,o,r)=\operatorname{AP}_k(s_{b,o,r}).
\]

Define macro performance in regime \(R\):

\[
M_R(b,o,r)=\frac{1}{|\mathcal B_R|}
\sum_{k\in\mathcal B_R}M_k(b,o,r).
\]

Then:

\[
\Delta^{ID}_{b,o,r}=M_{ID}(b,o,r)-M_{ID}(b,\text{base}),
\]

\[
\Delta^{OOD}_{b,o,r}=M_{OOD}(b,o,r)-M_{OOD}(b,\text{base}).
\]

Primary hypotheses:

- H1a: mean \(\Delta^{ID}_{b,SFT,r}>0\) over the fixed panel.
- H1b: estimate the sign and heterogeneity of \(\Delta^{OOD}_{b,SFT,r}\); do not preregister that SFT must always hurt.

Allowed claim:

> For these four checkpoints and this fixed recipe, SFT improves ID discrimination while its source-shift effect is negative on average / heterogeneous.

The phrase “degrades OOD” is allowed only if the upper 95% fixed-panel paired interval for mean \(\Delta^{OOD}\) is below zero and leave-one-benchmark-out sensitivity does not reverse the sign.

### RQ2 — Does objective choice change the frontier?

Compare DPO and SFT on the same anchor base and seed.

\[
d^{ID}_{b,r}=M_{ID}(b,DPO,r)-M_{ID}(b,SFT,r),
\]

\[
d^{OOD}_{b,r}=M_{OOD}(b,DPO,r)-M_{OOD}(b,SFT,r).
\]

Preregister:

- ID noninferiority margin: \(-0.020\) macro-AP;
- OOD superiority margin: 0.

The joint Pareto claim passes only when:

\[
LCB_{95}(E[d^{ID}])>-0.020
\quad\text{and}\quad
LCB_{95}(E[d^{OOD}])>0.
\]

If either condition fails, report the measured tradeoff rather than saying DPO dominates.

### RQ3 — Can explicit preservation reduce specialization?

Compare KL-SFT with ordinary SFT on the two anchor bases.

The trainable distribution is restricted to the two decision tokens:

\[
q_\theta(y\mid x)
=\operatorname{softmax}([z_{safe},z_{unsafe}]/T_K),
\quad y\in\{safe,unsafe\}.
\]

Use the objective:

\[
\mathcal L_{KL-SFT}
=\mathcal L_{SFT}
+\beta\;E_{x\sim\mathcal D_{train}}
\left[KL(q_0(\cdot\mid x)\|q_\theta(\cdot\mid x))\right],
\]

where \(q_0\) is the frozen untuned base's decision-head distribution, precomputed before training.

Primary comparison:

- KL-SFT is ID-noninferior to SFT within 0.020 macro-AP;
- KL-SFT has higher OOD macro-AP than SFT.

This is a guard-decision preservation penalty, not a guarantee that the complete language model distribution remains close.

### RQ4 — Do ranking conclusions survive deployment constraints?

Primary operating point: TPR at calibration FPR ≤ 5%. Sensitivity: 1% and 10%.

Also report prevalence-adjusted positive predictive value:

\[
PPV(\pi)=
\frac{\pi\,TPR}
{\pi\,TPR+(1-\pi)FPR},
\quad \pi\in\{0.001,0.01,0.05\}.
\]

A “regime-dependent ordering” claim requires a preregistered pair whose ordering differs between macro-AP and TPR@5% FPR, with paired uncertainty supporting both relevant differences. Merely comparing native F1 with AP is not a threshold-only result.

---

## 4. Data and split implementation

### 4.1 New canonical configuration

Add `configs/paper_a.yaml` containing:

```yaml
schema_version: 1
data_seed: 42
train_sources:
  - beavertails
  - toxicchat
  - prompt_injections
  - jailbreak_classification
exclude_train_families:
  - or_bench
final_seeds: [42, 43, 44, 45, 46]
models:
  qwen25_15b:
    id: Qwen/Qwen2.5-1.5B-Instruct
    revision: REQUIRED
  smollm2_17b:
    id: HuggingFaceTB/SmolLM2-1.7B-Instruct
    revision: REQUIRED
  smollm3_3b:
    id: HuggingFaceTB/SmolLM3-3B
    revision: REQUIRED
  qwen3_4b:
    id: Qwen/Qwen3-4B
    revision: REQUIRED
primary_metrics:
  - macro_average_precision
  - tpr_at_calibration_fpr_0_05
id_noninferiority_margin: 0.02
```

`REQUIRED` must make the program fail; it must never silently resolve a moving revision.

### 4.2 Manifest builder

Add `experiments/prepare_paper_a_manifests.py`.

Each row must have:

```text
sample_id
source
source_config
source_revision
source_row_id
split
label
text
normalized_text_sha256
family_id
label_provenance
license_class
```

Outputs:

```text
artifacts/paper_a/manifests/train.jsonl
artifacts/paper_a/manifests/tune_dev.jsonl
artifacts/paper_a/manifests/calibration.jsonl
artifacts/paper_a/manifests/id_test.jsonl
artifacts/paper_a/manifests/transfer_test.jsonl
artifacts/paper_a/manifests/ood_test.jsonl
artifacts/paper_a/manifests/stress_positive_only.jsonl
artifacts/paper_a/manifests/manifest.json
```

Split policy:

- Rebuild the current 3,279-row clean training pool once and freeze it.
- Split the 1,102 current development rows into `tune_dev` and `calibration`, stratified by source and label, preferably 60/40.
- Keep the 1,658 seen-source rows as `id_test`.
- Keep JailbreakBench and XSTest as `transfer_test`.
- Use WildGuardTest, WildJailbreak, and OR-Bench-Hard as the primary source-family OOD suite only after the audit proves OR-Bench train count is zero.
- Keep HarmBench outside macro-AP because it is all positive.
- Freeze ExpGuardTest only after configs are locked; it is external confirmation, not a hyperparameter source.

### 4.3 Leakage and family audit

Add `experiments/audit_paper_a_splits.py`.

It must perform:

1. normalized exact-text overlap;
2. conflicting-label exact overlap;
3. character n-gram MinHash or locality-sensitive near-duplicate search;
4. source-family membership audit;
5. row-count and class-count audit;
6. model/data revision audit;
7. train-license summary.

Hard assertions:

```text
OR-Bench rows in train == 0
exact train vs any final-eval overlap == 0
conflicting-label overlap == 0
every row has source revision and content hash
every near-duplicate candidate has an adjudication record
```

The program should emit both machine-readable JSON and a Markdown report.

### 4.4 ToxicChat fork

The current recipe trains on 655 ToxicChat rows. Decide before final training:

- If model adapters will be redistributed and the data license is incompatible, make a license-clean primary pool without ToxicChat and use ToxicChat only as transfer evaluation.
- If adapters will not be redistributed, document the restriction and publish only scores/configuration where permitted.

Do not describe ToxicChat as evaluation-only while executing a trainer that includes it.

---

## 5. Training implementation

### 5.1 Refactor common data and prompt code

Add:

```text
guard_research/data.py
guard_research/prompts.py
guard_research/labels.py
guard_research/provenance.py
```

Both [`train_guard.py`](../experiments/train_guard.py) and [`train_guard_pref.py`](../experiments/train_guard_pref.py) should import these functions. Training programs must no longer independently download and resample the dataset.

Required runtime arguments:

```text
--manifest artifacts/paper_a/manifests/train.jsonl
--config artifacts/paper_a/LOCK.json
--model-key smollm3_3b
--objective sft|dpo|kl_sft
--seed 42
--out artifacts/paper_a/runs/...
```

### 5.2 Objective parity

Hold constant where meaningful:

- identical ordered training rows;
- identical prompt template and decision tokens;
- LoRA `r=32`, `alpha=64`, dropout 0.05;
- identical seven projection modules;
- effective batch size 8;
- 300 optimizer updates;
- same maximum length;
- same hardware class;
- same evaluation and selection protocol.

Current SFT uses effective batch 4 and DPO uses effective batch 8. Fix that before saying that only the objective changes. Because DPO and SFT consume different amounts of token/forward compute, record total examples, tokens, forward passes, optimizer steps, wall time, and energy proxy; describe the comparison as fixed-row/fixed-update rather than perfectly compute-identical.

### 5.3 KL-SFT without a second model in memory

Add `experiments/precompute_reference_logits.py`.

For every training row, store the untuned base's two decision logits at the last prompt position:

```text
sample_id, safe_logit, unsafe_logit, model_revision, prompt_hash
```

For 3,279 rows and two FP32 logits, the raw numeric payload is only about 26 KB; metadata dominates. This makes the optimization feasible without keeping a frozen 3–4B reference model resident during training.

Add `experiments/train_guard_kl.py` or a `kl_sft` mode in the refactored trainer. The essential loss is:

```python
student_logits = outputs.logits[batch_index, decision_position]
student_two = student_logits[:, [safe_id, unsafe_id]] / kl_temperature
reference_two = batch["reference_two_logits"] / kl_temperature

kl = torch.nn.functional.kl_div(
    torch.log_softmax(student_two, dim=-1),
    torch.softmax(reference_two, dim=-1),
    reduction="batchmean",
) * (kl_temperature ** 2)

loss = completion_ce + beta * kl
```

Implementation requirements:

- `decision_position` is the final prompt token, before the target completion.
- Assert that `safe` and `unsafe` are distinct single tokens for every base.
- Validate reference cache hashes against model revision, tokenizer revision, prompt template, and row IDs.
- Search `beta` only on `tune_dev`; suggested bounded values are `{0.05, 0.1, 0.25, 0.5, 1.0}`.

### 5.4 What the KL term mathematically guarantees

For Bernoulli decision distributions, Pinsker's inequality gives, for each prompt:

\[
|q_\theta(unsafe\mid x)-q_0(unsafe\mid x)|
\leq
\sqrt{\frac{1}{2}KL(q_0\|q_\theta)}.
\]

Taking expectations and using Jensen's inequality:

\[
E_x|q_\theta(unsafe\mid x)-q_0(unsafe\mid x)|
\leq
\sqrt{\frac{1}{2}E_x KL(q_0\|q_\theta)}.
\]

Proof sketch: total variation between two Bernoulli distributions equals the absolute difference in their unsafe probabilities; apply Pinsker pointwise and Jensen to the concave square root.

This proves a bound on average decision-score drift over the distribution where KL is measured. It does **not** prove AP preservation, OOD generalization, or classification-risk preservation without additional margin and distribution-shift assumptions. Those remain empirical hypotheses.

### 5.5 HPO and lock protocol

Replace confirmatory use of [`hpo_guard.py`](../experiments/hpo_guard.py) with `experiments/tune_paper_a.py`.

The tuning process may read only:

- training manifest;
- `tune_dev` manifest;
- model/config definitions.

It must not receive OOD paths, labels, score files, or environment variables.

Suggested bounded search per applicable base/objective:

| Objective | Search |
|---|---|
| SFT | learning rate `{5e-5, 1e-4, 2e-4}`, warmup `{0.03, 0.10}` |
| DPO | learning rate `{5e-6, 1.5e-5, 3e-5}`, beta `{0.1, 0.5}` |
| KL-SFT | SFT learning rate locked from SFT; KL beta `{0.05, 0.1, 0.25, 0.5, 1.0}` |

Keep LoRA capacity fixed. Six short 120-step trials per search are sufficient for this bounded design. Select on ID tune macro-AP only, retain the selected adapter/configuration, and then retrain all final seeds for 300 steps.

Write:

```text
artifacts/paper_a/selected/<model>-<objective>.json
artifacts/paper_a/LOCK.json
```

`LOCK.json` must include a creation timestamp, git SHA, all manifests and hashes, model revisions, selected hyperparameters, metrics, margins, final contrasts, and final seeds. Final scoring must refuse to run without it.

The existing HPO JSON files remain explicitly labeled `pilot_invalid_for_confirmation` because all 48 trials computed final OOD metrics and deleted trial adapters.

---

## 6. Scoring, metrics, and statistical implementation

### 6.1 One canonical metric library

Add `guard_research/metrics.py` and remove local AP/AUROC implementations from producing paths.

Tie-aware non-interpolated average precision:

\[
AP=\sum_j(R_j-R_{j-1})P_j,
\]

where \(j\) indexes unique score thresholds; every equal-score group enters together. The implementation must call `sklearn.metrics.average_precision_score`.

AUROC must call `sklearn.metrics.roc_auc_score`, which assigns half-credit to score ties.

Primary metrics:

1. macro-AP across benchmark sources;
2. TPR at calibration FPR ≤5%;
3. macro partial AUROC over FPR ∈ `[0, 0.05]`;
4. calibration NLL and Brier score;
5. realized test FPR and its confidence interval;
6. latency p50/p95/p99 on one pinned substrate.

Secondary:

- pooled AP;
- full AUROC;
- optimal F1, explicitly labeled oracle/descriptive;
- ECE and reliability plots;
- prevalence-adjusted PPV.

### 6.2 Calibration and conservative threshold selection

Fit temperature on the calibration split by minimizing binary NLL with LBFGS. Do not use the current coarse grid as the final method.

For target FPR \(\alpha\), choose the lowest threshold that maximizes recall while the one-sided 95% Clopper–Pearson upper bound on calibration FPR is at most \(\alpha\). If no threshold is feasible, emit `NO_FEASIBLE_THRESHOLD`.

Report the realized test FPR; do not call it exactly matched.

### 6.3 Evaluation program

Add `experiments/eval_paper_a.py`.

Requirements:

- read only frozen manifests and `LOCK.json`;
- validate row, model, adapter, tokenizer, prompt, and config hashes;
- score the untuned base once per checkpoint and reuse it;
- score `tune_dev`, calibration, ID, transfer, OOD, and stress sets separately;
- never select a threshold on test rows;
- synchronize CUDA/MPS around timing;
- write one row per sample/system/seed with raw logit difference and probability;
- never validate a cache by length alone.

Recommended score schema:

```text
sample_id
content_sha256
source
split
gold
model_key
model_revision
objective
seed
adapter_sha256
prompt_hash
safe_logit
unsafe_logit
score_raw
score_calibrated
latency_ms
```

Store compact Parquet or compressed NPZ plus an immutable metadata JSON.

### 6.4 Uncertainty

Report two distinct uncertainty sources:

1. Conditional evaluation uncertainty: resample prompt-family clusters within each benchmark, preserving paired scores across compared systems.
2. Training uncertainty: summarize five seeds and pair the same seed across objectives.

For the primary panel effect, use a hierarchical paired bootstrap:

1. sample checkpoint keys with replacement from the fixed panel for a sensitivity interval;
2. within checkpoint, sample paired training seeds;
3. within each benchmark, sample semantic-family clusters;
4. compute each benchmark metric and macro-average;
5. compute the paired contrast.

Because there are only four model checkpoints and three core OOD families, the estimand remains this fixed panel. Do not claim inference over all future guard architectures or arbitrary domains.

Also report:

- per-seed values rather than only mean/range;
- leave-one-OOD-benchmark-out sensitivity;
- leave-one-base-out sensitivity;
- Holm correction for secondary pairwise contrasts;
- Kendall tau-b for rank association across metrics;
- direct paired intervals for every claimed delta.

McNemar is permitted only for paired binary error-rate questions, not as a test of F1 or AP.

---

## 7. Exact proof and test obligations

### 7.1 Mathematical propositions to retain

Only two supporting propositions are needed.

**Proposition 1 — monotone calibration invariance.** If \(g\) is strictly increasing, AP and AUROC computed from \(s\) and \(g(s)\) are identical when ties are handled consistently.

Proof: a strictly increasing map preserves every pairwise ordering and every unique-score threshold partition. AUROC depends only on pairwise ordering; AP depends only on the precision/recall sequence induced by those threshold partitions.

Qualification: an empirical CDF is weakly monotone and may create new ties, so the proposition does not directly apply to PIT scores.

**Proposition 2 — decision-score drift bound.** The KL-SFT Pinsker bound in Section 5.4 bounds average two-token score drift on the KL sampling distribution.

Do not present either proposition as the paper's novelty.

### 7.2 Required automated tests

Add:

```text
tests/test_metrics.py
tests/test_manifests.py
tests/test_thresholds.py
tests/test_cache_alignment.py
tests/test_objective_parity.py
tests/test_kl_loss.py
tests/test_lock_enforcement.py
```

Minimum cases:

- AP/AUROC equal sklearn for random arrays and degenerate cases.
- AP/AUROC do not change when rows inside a tied-score group are permuted.
- changing test labels cannot change a calibration threshold.
- OR-Bench train count is zero.
- normalized train/test exact overlap is zero.
- all final score files join one-to-one by `sample_id` and content hash.
- SFT/DPO/KL use the same manifest and LoRA modules.
- KL loss is zero when student and reference decision logits are equal.
- KL loss is positive for a perturbed student distribution.
- evaluation refuses an unlocked configuration.

### 7.3 Property test for threshold correctness

For small arrays, exhaustively enumerate all unique score boundaries and verify that the selected threshold has maximum recall among candidates satisfying the conservative FPR condition. This is a complete proof-by-exhaustion for the implementation on each test case.

---

## 8. Runner, artifact, and reproduction implementation

### 8.1 Final runner

Replace publication use of `stage2.sh` with `experiments/run_paper_a.py` or harden it substantially.

Current scaffolding problems:

- hard-coded `cd ~/guard`;
- omits the primary SmolLM3 cell;
- includes GRPO, DeepSeek, and 8B cells outside the recommended design;
- uses only three seeds;
- suppresses most logs through `tail`;
- does not enforce a manifest or lock hash.

Required runner states:

```text
prepare -> audit -> tune -> lock -> train -> score -> analyze -> validate
```

Every state should be resumable and idempotent. A failed or missing cell must remain visible; no table may silently omit it.

### 8.2 Analysis generator

Replace publication use of `aggregate_clean_sweep.py` with `experiments/analyze_paper_a.py`.

The new analyzer must generate:

```text
artifacts/paper_a/analysis/results.json
artifacts/paper_a/analysis/claim_checks.json
artifacts/paper_a/analysis/tables/*.tex
artifacts/paper_a/analysis/figures/*.pdf
artifacts/paper_a/analysis/report.md
```

No paper number or figure coordinate may be hand-transcribed.

### 8.3 Fresh-clone contract

The committed or DOI-hosted artifact must include:

- manifests and hashes;
- lockfile;
- selected configurations and HPO logs;
- per-row compact scores;
- statistical outputs;
- generated tables/figures;
- adapter checksums and immutable download identifiers;
- environment lock;
- data/model card;
- one command for cached reproduction.

CI should run from `git archive`, not the dirty working tree. It must not rely on ignored `notebooks/outputs`, obsolete `scripts/` paths, or user-specific absolute paths.

---

## 9. Staged commands

Commands after implementing the files:

```bash
source .venv/bin/activate
pip install -r notebooks/requirements.txt

pytest -q \
  tests/test_metrics.py \
  tests/test_manifests.py \
  tests/test_thresholds.py \
  tests/test_cache_alignment.py \
  tests/test_objective_parity.py \
  tests/test_kl_loss.py \
  tests/test_lock_enforcement.py

python experiments/prepare_paper_a_manifests.py \
  --config configs/paper_a.yaml \
  --out artifacts/paper_a/manifests

python experiments/audit_paper_a_splits.py \
  --manifest artifacts/paper_a/manifests/manifest.json \
  --out artifacts/paper_a/audit

python experiments/tune_paper_a.py \
  --config configs/paper_a.yaml \
  --manifest artifacts/paper_a/manifests/manifest.json

python experiments/lock_paper_a.py \
  --config configs/paper_a.yaml \
  --selected artifacts/paper_a/selected \
  --out artifacts/paper_a/LOCK.json

python experiments/run_paper_a.py train \
  --lock artifacts/paper_a/LOCK.json

python experiments/run_paper_a.py score \
  --lock artifacts/paper_a/LOCK.json

python experiments/analyze_paper_a.py \
  --lock artifacts/paper_a/LOCK.json \
  --results artifacts/paper_a/results \
  --out artifacts/paper_a/analysis

python experiments/validate_paper_a.py \
  --root artifacts/paper_a
```

Interim smoke test using current programs, not acceptable for publication:

```bash
OR_BENCH_CAP=0 \
GUARD_SEED=42 \
MODEL_ID=HuggingFaceTB/SmolLM3-3B \
OUT=outputs/sm3-clean-s42 \
python -u experiments/train_guard.py

FROZEN_ROWS=notebooks/outputs/frozen_eval_rows.json \
MODEL_ID=HuggingFaceTB/SmolLM3-3B \
ADAPTER=outputs/sm3-clean-s42/adapter \
TAG=sm3-clean-s42 \
python -u experiments/guard_eval_pipeline.py
```

Do not aggregate the smoke result until the threshold-selection, manifest, and cache-alignment issues are corrected.

---

## 10. Paper structure and evidence mapping

### 10.1 Section plan

1. Introduction: fixed-policy guard specialization under source shift.
2. Related work: guard evaluation/calibration, fixed-policy source shift, unseen-policy adaptation, objective-dependent robustness.
3. Problem formulation: specialization frontier and evaluation regimes.
4. Data and controlled design: immutable manifests, source-family exclusion, models, objectives, seeds.
5. RQ1 results: base-to-SFT movement.
6. RQ2/RQ3 results: DPO and KL-SFT mitigation.
7. RQ4 results: low-FPR and prevalence sensitivity.
8. Robustness: leave-one-source/base, prompt template, latency, label audit.
9. Limitations and conclusion.

### 10.2 Claim-to-artifact table

| Proposed statement | Required artifact | Pass condition |
|---|---|---|
| SFT improves ID discrimination | `rq1_sft_base.json` | lower paired 95% interval for mean ID delta > 0 |
| SFT reduces OOD ranking | `rq1_sft_base.json`, LOBO table | upper paired 95% interval < 0 and no LOBO sign reversal |
| DPO is a better frontier point | `rq2_dpo_sft.json` | ID noninferiority and OOD superiority both pass |
| KL-SFT preserves the base better | `rq3_kl_sft.json` | ID noninferiority and positive OOD contrast both pass |
| Rankings are regime-dependent | `rq4_regimes.json` | preregistered pair reverses with supported paired contrasts |
| Result extends beyond development suite | locked ExpGuard result | scored only after `LOCK.json`, no overlap, direction consistent |
| Local execution is practical | pinned latency trace | all systems measured on same hardware/protocol |

### 10.3 Novelty boundary

[ACL 2026 APT](https://aclanthology.org/2026.acl-long.748/) already shows policy overfitting and proposes augmented-policy training for unseen-policy transfer. Paper A must distinguish fixed-policy source shift from unseen-policy generalization.

GuardBench already shows that general instruction models can rival guard-specific models. Calibration, matched operating points, constrained objectives, and robust/base-tuned ensembles also have prior art.

The defensible contribution is therefore:

> controlled same-base, same-data, multi-seed attribution of how SFT, DPO, and a decision-head KL constraint move small fixed-policy guards across an ID/source-shift frontier.

Do not claim the first observation that a base can outperform a guard.

---

## 11. Compute, storage, and schedule

### 11.1 Compute estimate

Observed repository anchors:

- existing 120-step A100 HPO trials: approximately 6–9 minutes including scoring;
- current 300-step MPS runs range from tens of minutes to more than an hour depending on the base.

Budget for the recommended plan:

| Work | Approximate A100-equivalent budget |
|---|---:|
| bounded tuning | 8–14 GPU-hours |
| 40 final training runs | 12–20 GPU-hours |
| all scoring and latency | 5–10 GPU-hours |
| retries/validation reserve | 5 GPU-hours |
| **Total** | **30–49 GPU-hours** |

Four parallel GPUs should complete the machine work in roughly 1–2 days, excluding downloads, debugging, and analysis. A minimum three-seed plan should fit in approximately 18–30 GPU-hours. Full DPO/KL multi-seed execution is not a practical laptop-only study.

Forty LoRA adapters at roughly 230 MB each require about 9–10 GB. Reserve 30–60 GB for model caches and 5–10 GB for logs, score artifacts, and checkpoints. Delete intermediate checkpoints only after checksums and final validation pass.

### 11.2 Schedule

1. Week 1: canonical metrics, manifest builder, overlap audit, tests, lock schema.
2. Week 2: bounded tuning and smoke execution on all four bases.
3. Week 3: five-seed final matrix and scoring.
4. Week 4: statistical analysis, label audit, reproduction package.
5. Week 5: manuscript rewrite and external locked confirmation.

---

## 12. Final acceptance checklist

### Engineering/data

- [ ] All model and dataset revisions are immutable.
- [ ] OR-Bench count in training is zero.
- [ ] Exact train/final-eval overlap is zero.
- [ ] Every near-duplicate candidate is logged and adjudicated.
- [ ] One manifest hash is shared across all objectives and seeds.
- [ ] All 40 recommended final cells are complete, or missing cells are explicitly reported.
- [ ] Base scores are reused rather than recomputed inconsistently.
- [ ] Every score is keyed by sample and content hash.
- [ ] Final tables and figures are generated from result artifacts.
- [ ] Fresh-clone cached reproduction succeeds.

### Scientific

- [ ] Primary endpoints and margins were frozen before final scoring.
- [ ] AP/AUROC are tie-safe and permutation-tested.
- [ ] Thresholds use calibration rows only.
- [ ] All important deltas have direct paired intervals.
- [ ] Training-seed uncertainty is separate from row uncertainty.
- [ ] Macro-over-benchmark results are primary; pooled AP is secondary.
- [ ] HarmBench is not used for AP/AUROC.
- [ ] Claims are restricted to the fixed model/benchmark panel.
- [ ] Failed hypotheses are reported as negative results rather than rewritten after test access.

### Go/no-go

Submit the paper only if the clean multi-seed base-to-SFT effect is interpretable and at least one of DPO or KL-SFT yields a stable frontier improvement. If the SFT OOD sign becomes heterogeneous, reshape the paper around predictable heterogeneity rather than forcing the original “fine-tuning hurts” conclusion.

