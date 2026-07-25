# Paper C — Pre-registration v2

## What Does Reference Centering Buy a Safety Guard?

> **Protocol status: design freeze candidate, written before any v2 claim-bearing run.**
> This document supersedes [`paper-c-prereg.md`](paper-c-prereg.md), which is retained as
> the v1 amendment record. The v2 study becomes frozen only when its finalized configuration,
> code, development decisions, Stage-1 adapter inventory, and this file are bound into a new
> `artifacts/paper_c_dpo_v2/LOCK.json`. Until then every execution is development-only.

## 1. Question and contribution

For the fixed four-checkpoint Paper A panel, does label-derived Direct Preference
Optimization (DPO) improve the **represented-source versus dataset-held-out transfer
frontier** of a compact prompt-safety guard, or does it merely change pairwise normalization,
example selection, and update magnitude?

The contribution is not the first use of DPO for a guard and not the first safety-classifier
comparison of SFT, DPO, and RL. GuardReasoner and DT-Guard already use hard-case DPO in
reasoning-rich guard pipelines, and AIMS compares intent-aware SFT, DPO, and GRPO. The
remaining gap is a same-information, one-token factorization that holds the manifest, labels,
initial adapter, model panel, seeds, prompt, scorer, and selected rows fixed while separating:

1. full-vocabulary verdict cross-entropy from temperature-scaled two-verdict pairwise training;
2. pairwise normalization from DPO's frozen-reference centering; and
3. the loss effect from uncertainty-based example selection.

Primary prior-art boundary:

- [GuardReasoner](https://arxiv.org/abs/2501.18492): reasoning SFT followed by hard-sample DPO;
- [AIMS / Paved with True Intents](https://arxiv.org/abs/2606.27210): additional human intent
  supervision across SFT, DPO, distillation, and GRPO;
- [DT-Guard](https://arxiv.org/abs/2607.06326): staged mixed-mode SFT, hard-case SFT, and
  rollout-contrastive DPO.

## 2. Exact objectives

For input `x`, let

```text
s_theta(x) = z_theta(unsafe | x) - z_theta(safe | x)
y(x)       = +1 for unsafe, -1 for safe
m_theta(x) = y(x) * s_theta(x)
```

All Stage-2 arms start from the same immutable Stage-1 SFT adapter and receive the same
one-token verdict target. Only the loss changes:

```text
VerdictCE = -log softmax(z_theta over the full vocabulary)[gold verdict token]
PairCE    = softplus(-beta * m_theta)
DPO       = softplus(-beta * (m_theta - m_ref))
```

`m_ref` is precomputed in evaluation mode from the exact frozen Stage-1 adapter on the exact
rendered Stage-2 prompt. The reference-margin artifact is hashed. In this binary, one-token
setting, DPO is therefore a **reference-centered margin loss**, not new preference information.
The same `beta` is applied to PairCE and DPO, so `DPO - PairCE` changes reference centering without
also introducing a tenfold loss-temperature difference. `beta` is a loss-scale/temperature parameter;
this protocol does not call it a KL leash. Achieved
two-verdict divergence from Stage 1 is measured rather than assumed.

The two load-bearing contrasts are:

```text
C_pair,R  = M_R(PairCE) - M_R(VerdictCE)
C_ref,R   = M_R(DPO)    - M_R(PairCE)
C_total,R = M_R(DPO)    - M_R(VerdictCE)
```

for regime `R` in represented-source or dataset-held-out transfer evaluation.

## 3. Fixed panel, data, and initialization

- **Checkpoints:** Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct, SmolLM3-3B, and
  Qwen3-4B at the immutable Paper A model and tokenizer revisions.
- **Seeds:** 42, 43, 44, 45, and 46. The data-order seed is fixed independently.
- **Stage 1:** regenerate one completion-only LoRA-SFT adapter per checkpoint and seed with
  the locked Paper A recipe. One adapter's exact bytes are cloned into every matched Stage-2
  arm for that checkpoint and seed. Existing Paper A score files are not a substitute for the
  missing adapter bytes.
- **Stage-2 pool:** the same 1,200-row Paper A training manifest, assigned by global `family_id` to a
  locked approximately 80% update pool and 20% objective-development pool. A deterministic
  multi-stratum objective balances source/label counts without splitting the few families that span
  sources. Stage 1
  saw all 1,200 rows; the split is explicitly a Stage-2 selection set, not a fully unseen cohort.
- **Evaluation:** Paper A calibration, represented-source, transfer, OR-Bench, and HarmBench
  manifests are reused by hash for retrospective continuity. They are not used for objective,
  hyperparameter, or checkpoint selection.

## 4. Objective-by-selection factorial

For each checkpoint and seed, Stage-1 scores on the Stage-2 update pool define two equal-size,
source-and-label-matched selections:

- **uncertain:** within each `(source, gold)` stratum, the top 25% by two-verdict entropy;
- **matched_random:** the same number of rows from the remainder, ordered by a locked
  SHA-256 selection key and seed `20260725`.

The primary Stage-2 matrix is:

```text
{VerdictCE, PairCE, DPO} x {uncertain, matched_random}
```

This is `4 checkpoints x 5 seeds x 6 cells = 120` short Stage-2 runs, plus 20 shared Stage-1
adapters. Every objective within a sampler receives identical rows, row order, minibatch
boundaries, prompt bytes, LoRA initialization, optimizer family, learning-rate schedule, zero Stage-2 dropout,
and maximum update count.

The primary causal-by-design comparisons are within a fixed sampler. The interaction

```text
(DPO - VerdictCE)_uncertain - (DPO - VerdictCE)_matched_random
```

tests whether uncertainty selection changes DPO's relative effect. It does not identify a universal
hard-example mechanism outside this fixed design.

## 5. Training and checkpoint selection

The frozen candidate configuration uses:

- Stage-2 learning rate `5e-6` for every arm;
- cosine schedule, warmup ratio `0.03`, effective batch size `4`;
- maximum 200 updates, with checkpoints at 25, 50, 100, and 200;
- shared PairCE/DPO `beta = 0.1` for the primary arms;
- Stage-2 dropout `0.0` in every arm, so cached evaluation-mode reference margins equal the
  step-zero policy margins before optimization;
- the same trainable LoRA modules as Paper A.

The selected checkpoint for each matched cell is the earliest checkpoint whose Stage-2
objective-development macro-AP reaches `Stage-1 macro-AP - 0.02`. If no checkpoint reaches the
target, the arm is marked **target infeasible**; its 200-step checkpoint may be scored only as a
clearly labeled descriptive fallback and is excluded from the target-matched success gate. Its lower
movement may not be interpreted as transfer preservation. The step number uniquely orders candidates,
so there is no post-hoc tie breaker.

Sensitivity-only runs may use shared PairCE/DPO `beta in {0.03, 0.3}` after the primary configuration and analysis
code are locked. They cannot replace or redefine the primary arm.

## 6. Outcomes

### Primary

- tie-aware benchmark-macro average precision on represented-source rows;
- tie-aware benchmark-macro average precision on dataset-held-out transfer rows;
- seed-paired `C_ref`, `C_pair`, and `C_total` contrasts with shared checkpoint/family/seed
  resampling weights.

### Deployment and reliability

- raw and temperature-scaled Brier score and NLL;
- transfer TPR/FPR at the calibration-only 5% FPR operating point;
- OR-Bench benign false-positive rate;
- HarmBench recall;
- low-prevalence precision curves at prevalences `{0.001, 0.005, 0.01, 0.05, 0.10}`. Invalid-output
  rate is not reported for direct logit scoring because no text verdict is generated.

### Mechanism diagnostics

- signed verdict-margin change from Stage 1;
- `KL(policy || Stage-1)` over normalized `{safe, unsafe}` logits, reported separately on the exact
  locked represented and transfer rows;
- full-vocabulary next-token KL only if a future locked scorer schema stores the necessary logits;
- entropy, saturation, margin tails, LoRA norm, updates, examples, tokens, GPU-seconds, and
  reference-logit cache identity.

Movement is a post-treatment diagnostic. Matching or conditioning on it estimates transfer
efficiency at comparable drift; it does not replace the ordinary objective contrast and is not
called a causal mediator.

### Secondary, descriptive

The fixed Paper B base-plus-adapter output average is applied to the selected Stage-2 adapters.
Composition-minus-adapter transfer is reported by objective and achieved movement. No monotonic
composition claim is confirmatory in Paper C v2.

## 7. Decision rules

### Retrospective Paper A suite

Report point estimates and two-sided 95% paired hierarchical-bootstrap intervals. These results are
estimation-only because the benchmarks and legacy pilots have been inspected. No interval crossing
zero is interpreted as proof of a null.

### Sealed confirmatory cohort

Before unsealing, freeze adapters, selected checkpoints, source code, configuration, analysis,
non-inferiority margins, and artifact hashes. The cohort must be family-disjoint, contain at least
two independently sourced benchmark families, and have played no role in development.

DPO is called a **better reference-centering frontier than PairCE** only if both simultaneous
one-sided 97.5% hierarchical-bootstrap bounds satisfy:

```text
LCB(C_ref, transfer)    >  0.00
LCB(C_ref, represented) > -0.02
```

and the predeclared OR-Bench-style false-positive and HarmBench-style recall harm margins are not
crossed. Otherwise the result is described as a trade-off, no demonstrated advantage, or infeasible,
as appropriate.

## 8. Exclusions and interpretation limits

- GRPO, KTO, ORPO, IPO, and learned reward models are outside the primary matrix.
- Direct base-to-DPO is an optional mechanistic ablation, not a primary condition.
- ExpGuard, mortgage, and all current Paper A rows are retrospective or exploratory; none can become
  prospective by relocking.
- This fixed four-model panel does not identify a universal model-size or architecture effect.
- A hard-label DPO result does not establish the value of human preferences, policy preferences, or
  reasoning traces.
- A future policy-conditional `{allow, block, escalate}` study requires separately adjudicated
  policy-dependent preferences and is not part of this protocol.

## 9. Failure states

A cell is invalid if any of the following occurs:

- Stage-2 arms do not start from the same adapter hash;
- prompt bytes, tokenizer revisions, verdict-token IDs, selected rows, order, or minibatches differ;
- the frozen reference-margin hash differs from the run metadata;
- an adapter is not loaded during scoring;
- evaluation rows influence hyperparameters or checkpoint selection;
- a final run occurs from a dirty or source-hash-mismatched checkout;
- software, lock, manifest, or analysis hashes fail verification.

Invalid and failed runs are retained and reported; they are never silently replaced.

## 10. Provenance and amendments

The fresh namespace is `artifacts/paper_c_dpo_v2/`. Its final lock binds:

- the Paper A parent lock and all consumed manifest hashes;
- `configs/paper_c_dpo_v2.json`;
- this preregistration, the development plan, and code design;
- every execution source and test used for lock/train/eval/analyze;
- Stage-1 adapter and run-metadata hashes;
- selection, reference-margin, score, and analysis artifact schemas;
- the exact environment and clean git commit.

Any post-lock change creates a dated amendment and a fresh artifact namespace. It never overwrites
the v1 record, the v2 lock, or Paper A artifacts.
