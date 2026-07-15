# Paper C — Pre-registration (objective axis: SFT vs DPO vs GRPO)

> **Status: written BEFORE any DPO/GRPO run.** This document fixes the hypotheses, design, metrics,
> analysis, and decision rules in advance so the objective-axis result cannot be reverse-fit to the
> data (no HARKing). Its SHA-256 is bound into the Paper C lock (`artifacts/paper_c_objective_v2/`)
> before the first final training run; any change after that is a new, dated revision.
> The evidence tier is **retrospective / estimation-only** (it reuses Paper A's inspected v2 manifest),
> **not confirmatory** — see §7.

## 1. Question

For the fixed four-checkpoint panel (Qwen2.5-1.5B, SmolLM2-1.7B, SmolLM3-3B, Qwen3-4B) and one frozen
1,200-row decontaminated training manifest, **does the choice of fine-tuning objective change how a
compact prompt-safety guard trades represented-source discrimination against dataset-held-out
transfer** — and does an output-space base+adapter composition recover transfer in proportion to how
far the objective moves the model from its base?

## 2. Design (fixed in advance)

- **Panel & data:** identical to Paper A v2 — same 4 checkpoints, same frozen `train.jsonl` (1,200 rows),
  same calibration/represented-ID/transfer/OR-Bench/HarmBench manifests, consumed by SHA-256. Reused by
  hash so SFT/DPO/GRPO are directly comparable; **no new data**.
- **Objectives (minimal-viable):** `base` (reused), `sft` (reused Paper A arm), **`dpo`**, **`grpo`**.
  Seeds 42–46 (5), data-order seed 42 — matching Paper A. Deferred (Step 3): KTO, ORPO, β sweep, IPO/RLOO.
- **Guard head & score (identical across every arm):** single next-token verdict; canonical score
  `s(x) = z_unsafe − z_safe` from one forward pass; derived probability via the two-way softmax. Every
  objective is evaluated on this same score — no arm is scored by a sampled string.
- **LoRA recipe:** identical to Paper A (r=32, α=64, dropout 0.05, target q,k,v,o,gate,up,down; max len
  1024; completion-only where applicable).

## 3. Reward / preference construction (verifiable label — no learned reward model)

The label is the reward. Introducing a neural reward model would add the reward-hacking surface that
RLVR recipes (DeepSeek-R1, Tulu-3) deliberately avoid. Built deterministically from the frozen manifest
(`preference_recipe_sha256` recorded in the lock):

- **DPO:** pair `chosen` = correct verdict token, `rejected` = wrong verdict token; reference = frozen base; knob β (default first).
- **GRPO:** reward = **graded margin** on `z_unsafe − z_safe` toward the correct side (not raw 0/1), to
  avoid single-token gradient starvation (once the guard is confident, all sampled 1-token verdicts agree
  → group std → 0 → no gradient). Dr.GRPO/RLOO-style **unnormalized** advantage (two reward values make
  std-normalization pure difficulty bias). Class/severity-weighted correctness if the manifest is skewed.

## 4. Pre-registered hypotheses

- **H1 (objective orders the trade-off by movement).** All objectives can fit the verdict, so
  represented-source AP will rise to a high ceiling for every arm (as SFT already does, ≈0.98). The
  objectives will differ mainly on **transfer**, and the difference will track **how far each objective
  moves the tuned model from its base** (measured as the base↔tuned shift in calibrated P(unsafe) on
  represented rows). Predicted transfer-preservation order (weak, pre-registered): **SFT ≳ DPO ≳ GRPO**
  (objectives that move less from base preserve more transfer).
- **The GRPO single-token null (headline pre-registration).** Because our action space is **one token,
  two outcomes**, the exploration/multi-token-reasoning degrees of freedom that let RL generalize in the
  literature are **absent**. We therefore predict **GRPO's transfer edge over SFT will be small or null**
  — a *bounded negative result is the expected, reportable outcome*, not a failure.
- **H2 (objective × composition).** The base+adapter calibrated-average composition will recover transfer
  **monotonically in the amount the objective moved from base** — i.e., objectives that specialize more
  (move further) leave more transfer for composition to recover.

## 5. Metrics & analysis (fixed in advance)

- **Primary:** tie-aware macro average precision (the Paper A definition) on represented-source and on
  dataset-held-out transfer, per (checkpoint, objective, seed); the paired base→objective delta, and the
  fixed-panel aggregate. **Reported as observed points with two-sided 95% paired hierarchical-bootstrap
  intervals** (10,000 replicates; family + seed resampling), conditional on the fixed panel — descriptive,
  no significance test.
- **Mediator:** the base↔tuned movement (calibrated P(unsafe) shift / forward-KL) per arm — the H1/H2 axis.
- **Composition (H2):** the existing base+adapter operator scored at analysis time for each objective
  (≈ free); recovery = composition-minus-objective transfer AP vs the movement mediator.
- **Heterogeneity:** per-checkpoint reported; no post-hoc subgroup promoted to a headline.
- **Reused unchanged:** `guard_research.prompts` (byte-identical rendering), the tie-aware macro-AP, the
  hierarchical paired bootstrap, and the composition machinery.

## 6. Decision rules / what would falsify each claim

- **H1 falsified** if transfer ordering does not track base-movement (e.g., the least-moving objective loses
  the most transfer), or if objectives are statistically indistinguishable on transfer.
- **GRPO-null falsified (a genuine positive)** if GRPO's transfer AP exceeds SFT's by an interval strictly
  above 0 on the fixed-panel aggregate — which we would report as a surprising positive, not quietly fold in.
- **H2 falsified** if composition recovery is unrelated to (or inversely related to) base-movement across
  objectives.

## 7. What this does NOT establish

- **Not confirmatory / not prospective.** The manifest and transfer benchmarks were inspected during Paper A
  development; a locked rerun cannot make inspected data prospective. This is an estimation-only description
  of this fixed panel — confirmatory use needs a separately locked, genuinely uninspected cohort.
- **Not a general "RL vs SFT" verdict.** Single-token binary guard, 4 compact checkpoints, one manifest;
  the GRPO null is bounded to this setting and is *consistent with*, not a refutation of, multi-token RL
  generalization results.
- **No causal or universal claim**; intervals are conditional on the panel and datasets.

## 8. Provenance

Fresh artifact root `artifacts/paper_c_objective_v2/` with its own lock reusing Paper A's manifest/audit
bindings **by hash**; the lock records the objectives recipe, `preference_recipe_sha256`, the exact
software versions (fail-closed on mismatch), and **this file's SHA-256**. Paper A's artifacts are never
modified. New trainer/analysis code is committed before the final run (clean-git execution gate).
