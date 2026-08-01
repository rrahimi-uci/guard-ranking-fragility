# Review of `improvement-proposal.md`, and what was implemented from it

**Reviewed:** 2026-08-01 · **Proposal snapshot:** 2026-08-01 · **Reviewer:** engineering pass over
the committed repository, with every checkable assertion recomputed rather than read back.

This document does three things: it records which of the proposal's factual claims survive contact
with the repository, it says which recommendations were accepted and which were rejected and why,
and it states plainly what was *not* done. The proposal's own evidence discipline is applied to the
proposal itself.

---

## 1. Verdict in one paragraph

The proposal is **substantially sound and unusually well-sourced**. All seven load-bearing external
citations resolve to real works with the titles and numbers it attributes to them — including the
two July-2026 preprints, which is the class of citation most likely to be confabulated. Every data,
split, and result-ledger claim reproduces exactly from committed artifacts. Its central
methodological criticism — that the repository reports average-ranking metrics while every
deployment claim it makes is about a false-alarm budget — is **correct, material, and had no
implementation to answer it**. That criticism is what this pass acted on. Its *primary* recommended
arm (T1) is a four-to-six-month data campaign and was **not** implementable here; saying otherwise,
or synthesising its results, would violate the evidence rules this repository exists to enforce.

Two corrections matter. One numerical claim is **wrong** (`15 of 16` should be `14 of 16`), and the
mechanism behind its most interesting code finding is **misdiagnosed** — though a real and slightly
worse defect sits next to the one it names.

---

## 2. Claim-by-claim verification

### 2.1 Data, splits, and the result ledger — all reproduce

Every claim in proposal §2.2–2.4 was recomputed from `artifacts/` rather than read from summary
JSON. All confirmed: the 1,200-row balanced manifest; evaluation counts 677 / 1,580 / 400 / 200 and
451 calibration rows; zero exact and zero `family_id` train↔eval overlap; six cross-split pairs at
the 0.80 MinHash sensitivity; exactly one conflicting-label prompt *inside* evaluation
(`jailbreak_classification` id_test `unsafe` vs `wildjailbreak` transfer `safe`); calibration
family-disjoint from every reported test; 300 steps × effective batch 4 = one pass. All thirteen
ledger rows reproduce.

Three ledger rows cite the wrong file (the HarmBench and matched-budget rows point at
`claim_checks.json`, which contains no stress entry; the adaptation preservation row points at
`claim_checks.json`, which carries only the bound, not the point estimate). Cosmetic, but worth
fixing in the proposal.

### 2.2 One numerical claim is REFUTED

> §3.3: "`beta=.5` improves transfer AP on **15 of 16** model/source cells; the exception is Qwen2.5
> on XSTest (`−0.010`)."

The true count is **14 of 16**, with **two** exceptions: Qwen2.5/XSTest (`−0.0100`, the quoted value
is exactly right) and **SmolLM2/WildJailbreak (`−0.0710`)**. The count is 14 under both seed-mean and
seed-pooled aggregation, and the second negative is not a seed fluke — it is negative in 5/5 seeds.
The sentence is also self-contradicting: the very next sentence in the proposal names the −0.071
SmolLM2 loss it just excluded.

A related claim is **partly** right: "the largest costs concentrate in prompt injection and the
smaller Smol models" holds for marginal means but not for individual cells — the single largest
represented cost is SmolLM2 on **ToxicChat** (`−0.0768`), and SmolLM3-3B (`−0.0568`) is a worse
marginal than Qwen2.5-1.5B (`−0.0080`), so "the smaller Smol models" is loose.

### 2.3 The KL code finding is real but misdiagnosed — and the true defect is worse

> §3.2(7): "The adapter-disabled reference is obtained from the same training-mode model … Dropout
> may therefore contribute noise to the measured KL."

The code path is exactly as described ([`run_paper_a_sft.py:301-304`](../experiments/run_paper_a_sft.py#L301-L304)).
The *mechanism* is not:

- **Reference side — LATENT, not active, on the Act I panel.** All four pinned checkpoints set
  `attention_dropout: 0.0` and carry no other dropout field, and PEFT short-circuits `lora_dropout`
  on the `disable_adapters` branch. Probed directly: the reference forward is bit-identical across
  repeated calls in training mode for all four architectures.
- **Reference side — ACTIVE on the adaptation panel.** `ibm-granite/granite-guardian-3.1-2b` ships
  **`attention_dropout: 0.1`**. The analysis-preregistered starting-type study
  ([`starting_type_common.py`](../experiments/starting_type_common.py)) shares the same code shape,
  so one of its five purpose-built families was anchored to a *stochastic* reference. The proposal
  did not find this, because it only looked at the Act I trainer.
- **Student side — ACTIVE everywhere, and unmentioned.** The KL is taken against `outputs.logits`
  from the CE forward, which runs in training mode with `lora_dropout=0.05`. So the penalty is
  measured between a dropout-perturbed student and the base. That is defensible *as a regularizer*
  (you constrain the sampled sub-network), but it means the logged KL is a noisy, upward-biased
  estimate — a fact that was nowhere recorded.

> §4.3: the released KL artifacts retain no achieved KL, loss curve, or run metadata.

**Confirmed, and stronger than stated.** `_kl_running` was overwritten every micro-batch and never
handed to the trainer log stream, so `final_kl` was the KL of whichever micro-batch ran last and
**no achieved-KL curve existed anywhere by construction** — not merely by non-release.

§3.2 items 2 and 3 are confirmed exactly: the completion is exactly two positions (verdict + EOS,
enforced by a single-token decision contract that hard-fails otherwise), so half the KL mass anchors
post-verdict formatting; and the KL is full-vocabulary while evaluation keeps only two logits.

### 2.4 External citations — all real

| Cited as | Verified |
|---|---|
| HaloGuard 1.0, arXiv:2607.02079 | Real. Qwen3.5-based 0.8B/4B; **1,259,451** records; the 3.5% vs 4.7% FPR inconsistency is real and self-documented on the model card |
| DT-Guard, arXiv:2607.06326 | Real. "Reasoning-Active Training, Reasoning-Free Inference", 4B backbone |
| `Qwen/Qwen3.5-4B` | Real, with a `-Base` sibling; HaloGuard's declared parent |
| APT, ACL 2026 long 748 | Real. "48,192 training instances" from 24,096 — exact |
| Reasoning's Razor, EACL 2026 long 190 | Real, and its abstract states the low-FPR failure the proposal relies on |
| LS-Guard / FlexGuard 2026 | Real |
| HarmAug / PIGuard / SafeRoute | Real |

**A risk the proposal missed.** `Qwen/Qwen3.5-4B` is **not a text-only decoder**: it is
`Qwen3_5ForConditionalGeneration`, multimodal (24-layer vision encoder, image/video token IDs), with
a **hybrid Gated DeltaNet / full-attention** layer pattern. The proposal's M0 mentions a
"Qwen3.5/DeltaNet LoRA target mapping" in one clause but never flags that the whole one-token-margin
scoring contract, the LoRA target-module set, and the latency envelope all assume a homogeneous
text-only causal LM. That is a substantially larger M0 than budgeted.

Two smaller ones: the "roughly 0.8–1.26M structured examples" range is verified only at its upper
bound (Qwen3Guard's corpus size is not disclosed in its abstract), and LS-Guard's "orthogonality"
mechanism is not stated in the retrievable abstract.

---

## 3. What was accepted, and why

Ranked by expected impact per unit of engineering, restricted to what could be **executed and
verified** in this pass.

### A1 — Low-FPR metrics as first-class, canonical primitives · **IMPLEMENTED**

*Why.* This is the proposal's strongest argument (§4.1) and it was unanswerable: `guard_research/`
had `average_precision`, `auroc`, `brier`, `log_loss_` and **no** partial AUC, no TPR@FPR, no
low-FPR helper anywhere in the repository. Every deployment sentence in the paper is about a 5%
budget; every headline number was an average over the whole ranking. Nothing established that a
change in one implied the same change in the other.

*What.* `partial_auc` (one-way, McClish-truncated, normalized to mean-TPR-in-region) and
`tpr_at_fpr` (conservative under ties), with `LOW_FPR_MAX = 0.05` as the single source of truth for
the budget. Eight new tests, including a cross-check that the unnormalized area equals the value
`sklearn.roc_auc_score(max_fpr=…)` standardizes, and a tie test asserting the frontier's
coarse-integer failure mode is not silently interpolated away.

*Where — and a correction to the proposal's plan.* §11.2 says to put these in
`guard_research.metrics`. **That would break the Paper A release contract, and it did.**
`guard_research/metrics.py` is one of six files in `RELEASE_CACHE_SOURCE_FILES`, hashed and
committed inside `artifacts/paper_a_sft_v2/` as the definition of "the metrics this release was
computed with". Adding a purely additive function to it changes its bytes, and the release analyzer
correctly fails closed with `release analyzer/verifier source tree mismatch` — turning four
environment-gated artifacts into hard failures. The metrics therefore live in a new sibling module,
[`guard_research/operating_point.py`](../guard_research/operating_point.py). Anyone acting on §11.2
as written should make the same substitution.

*Risk.* Low. Nothing in the sealed module changed; the release contract verifies unchanged.

### A2 — Re-read Act I and the KL control in the operating region · **IMPLEMENTED**

*Why.* A1 is only worth having if it changes a conclusion. It does.

*What.* [`papers/unified-report/low_fpr.py`](../papers/unified-report/low_fpr.py), pure arithmetic on
the same committed `score_raw`/`gold`/`family_id` columns, with the report's own family-aware paired
bootstrap (2,000 replicates resampling `family_id` clusters *and* training seeds). Emits two tables
and two macro files, byte-checked by `make verify`.

*Result — Act I.* **No sign flips** on any of eight cells: the qualitative claim survives being read
at a deployable operating point. But macro-AP **understates both halves of the trade**:

| | Δ macro-AP | Δ pAUC[0,.05] | Δ TPR@5% | amplification |
|---|---:|---:|---:|---:|
| Represented (panel) | +0.323 | +0.686 | +0.678 | 2.1× |
| Transfer (panel) | −0.059 | −0.174 | −0.199 | 2.9× |
| Transfer, worst cell (Qwen3-4B) | −0.150 | **−0.409** | **−0.437** | 2.7× |

*Result — KL.* The dial is far steeper than reported. At β=0.5 the transfer gain is +0.061 AP but
**+0.149 pAUC / +0.163 TPR@5%** (2.4×), while the represented cost is −0.035 AP but **−0.214 pAUC**
(6.2×). Consequently the adaptation study's registered `−0.02` non-inferiority margin is missed by
**~1.7× on AP but ~10.7× in the operating region** — the preregistered failure is emphatic, not
marginal.

*Risk.* The intervals are conditional on the same fixed panel as everything else; this is a
re-reading of existing evidence, not new evidence. Stated as such in the paper.

### A3 — Deterministic KL reference + achieved-KL curve · **IMPLEMENTED**

*Why.* §2.3 above. A regularizer whose realised strength was never recorded cannot be diagnosed, and
an anchor that is deterministic only because someone else's config happened to default to 0.0 is a
silent dependency.

*What.* Both trainers now force eval mode for the reference forward only (restoring the prior mode),
accumulate a running mean of achieved KL, emit it into the trainer log stream so a curve exists, and
persist `kl_curve` / `kl_reference_mode` / `kl_student_mode` in run metadata. The student side is
deliberately left stochastic, and that choice is now documented rather than accidental.

*Risk / honesty note.* **No published number changes.** Re-running the KL sweep needs a GPU and 60
training cells; this fix changes future runs only. On the Act I panel the reference was already
deterministic, so nothing there was ever affected. On the adaptation panel the Granite family's
`attention_dropout: 0.1` means its released KL cells *were* anchored to a stochastic reference — a
disclosed limitation, not a corrected result.

---

## 4. What was rejected, and why

| Item | Decision | Reason |
|---|---|---|
| **T1 — policy-conditioned counterfactual frontier distillation (25k→100k→400k)** | **Not implementable here** | Requires frontier-teacher collection under *provider terms the proposal itself says must be resolved first*, two-reviewer human adjudication, ledger registration of every new source before collection, and GPU-weeks. None can be produced in an engineering pass, and generating any of its numbers would be fabrication. The design is sound; it needs a funded campaign, not a commit. |
| **T2 — selective boundary reasoning supervision** | **Not implementable here** | Explicitly gated on a 100k T1 student existing. |
| **M0 — score Qwen3.5-4B / HaloGuard-4B** | **Deferred, and re-scoped** | Needs 4B-model downloads and GPU. More importantly the review found M0 is *underspecified*: Qwen3.5 is multimodal with hybrid Gated DeltaNet attention, so the scoring contract, LoRA target set and latency envelope are all open questions, not a one-week task. |
| Corrected-KL factorial, DoRA/MoRA, SAM/R-Drop, DPO/GRPO, model soups, routers, committees | **Agree — reject** | The proposal's own reasoning is right, and the repository has already measured that ensembling and scale do not close the frontier gap. |
| Embedding-space overlap audit | **Deferred** | Genuinely open (it is the paper's own roadmap item 2), but it requires pinning an encoder *into the lock* — an identity decision, which is exactly why the paper files it as a roadmap item rather than a script. Not something to decide unilaterally. |

---

## 5. What this changes in the paper

The low-FPR re-analysis is added as evidence at the **same tier as the rest of Act I** —
retrospective, estimation-only, conditional on the fixed panel. It does not license a new claim
about deployment; it re-prices claims the report already makes, using the metric the report's own
deployment language implies. Three existing statements are sharpened rather than reversed:

1. Act I's transfer cost, quoted as `−0.059` macro-AP, is `−0.199` in budget recall.
2. The KL "tradeoff dial" guidance now carries its operating-region exchange rate.
3. §sec:adaptation's registered RQ2 failure is restated with the multiple of the margin.

And one is newly qualified: SmolLM2-1.7B, the single checkpoint the report says *improves* on
transfer (`+0.040` AP), is `+0.011` pAUC and `−0.012` TPR@5% — at a deployable budget its gain is
indistinguishable from zero.

---

## 6. Recommended next steps, in order

1. **Fix the two proposal errors** before it is used to plan work: `15 of 16` → `14 of 16`, and
   replace the training-mode-reference mechanism with the student-dropout / Granite findings above.
2. **Re-scope M0** around the Qwen3.5 architecture change; budget it as an integration project.
3. **Re-run the KL sweep** on the corrected trainer when a GPU is available, and publish the
   achieved-KL curves. This is the cheapest way to convert the KL arm from an outcome table into a
   diagnosis, and it is now possible because the curve is recorded.
4. Only then consider T1, and only after the provider-terms and ledger questions are closed.
