# Paper C normative protocol candidate

## Status and scope

This is the sole protocol candidate for the isolated `papers/paper_c` study. The
currently operative protocol-lock path is recorded in `STATUS.md` and must
validate against the live source tree. A superseding lock may preserve a prior
lock byte-for-byte while binding a safety-only execution amendment; it may not
change the scientific configuration or input manifest. The protocol lock
authorizes Stage-1 candidate generation only. A Stage-2 design lock
must later bind the accepted Stage-1, reference, and selection inputs before
candidate training. A later selection lock is needed
before retrospective scoring, and a prospective child lock is needed before any
confirmatory claim.

## Research question

For compact one-token binary safety guards, does DPO's frozen-reference
centering improve the represented-source versus held-out-dataset transfer
frontier beyond an otherwise identical pairwise loss?

The novelty is deliberately narrow. Prior work already applies DPO to guard
models. This study isolates what reference centering contributes when labels,
prompts, examples, initialization, update budget, loss temperature, and scoring
are held fixed.

## Objectives

Let `z_safe` and `z_unsafe` be next-token logits, `y=+1` for unsafe and `y=-1`
for safe, and

```text
m_theta(x) = y * (z_unsafe - z_safe).
```

The matched Stage-2 objectives are:

```text
VerdictCE = -log softmax(z_theta over full vocabulary)[gold verdict]
PairCE    = softplus(-beta * m_theta)
DPO       = softplus(-beta * (m_theta - m_reference))
```

`beta=0.1` is shared by PairCE and DPO. DPO therefore changes reference
centering, not temperature or preference information. At identical policy and
reference margins, DPO loss must equal `log(2)`.

## Panel and factorial

- Four pinned instruction checkpoints: Qwen2.5-1.5B, SmolLM2-1.7B, SmolLM3-3B,
  and Qwen3-4B.
- Five seeds: 42–46.
- Twenty shared Stage-1 completion-SFT adapters.
- Stage-2 factorial:
  `{VerdictCE, PairCE, DPO} × {uncertain, matched_random}`.
- 120 primary Stage-2 runs, each saving steps 25, 50, 100, and 200.
- 480 candidate checkpoints plus 20 Stage-1 development baselines.

## Data separation

The 1,200-row Paper A training manifest is assigned by global `family_id` to an
approximately 80% Stage-2 update pool and 20% Stage-2 development pool. The
Stage-1 adapters see the full training manifest; the split only governs
Stage-2 update and checkpoint-selection roles.

Within every `(source, gold)` update stratum:

- `uncertain` selects the highest-entropy 25%;
- `matched_random` selects an equal-size, disjoint SHA-256-ranked sample from
  the remaining rows.

Reference logits come from the exact Stage-1 adapter in evaluation mode. The
reference artifact includes the frozen prompt-token fingerprint for every row.

## Prompt determinism

Tokenizer chat templates can contain date-dependent behavior. Each manifest is
therefore rendered exactly once per pinned tokenizer revision. The resulting
prompt token IDs and identities are stored in a hash-bound cache. Stage 1,
reference scoring, Stage 2, development scoring, and test scoring consume that
cache; rerendering after the design lock is forbidden.

## Checkpoint selection

For each model, seed, sampler, and objective, select the earliest candidate whose
Stage-2-development macro-AP is at least:

```text
Stage-1 development macro-AP - 0.02.
```

If no step reaches the target, mark the cell `target_infeasible`. Step 200 may be
reported as a descriptive fallback but is excluded from target-matched
contrasts. Any infeasible cell makes the complete confirmatory primary panel
ineligible for a success claim.

## Estimands

### Why the point estimand is secondary

Two measured facts demote `C_ref = AP(DPO) - AP(PairCE)` from primary to secondary.

1. **It is confounded with effective learning rate.** DPO and PairCE share a gradient
   direction and differ only by a per-example weight: `sigma(-beta*(m_theta - m_ref))`
   versus `sigma(-beta*m_theta)`. At step zero `m_theta == m_ref`, so DPO weights every
   row at exactly 0.5 while PairCE weights it at `sigma(-beta*m_ref)`. On this panel's
   Stage-1 margins that ratio has mean **1.30** with weight coefficient of variation
   **0.17**, and 93% of rows sit at positive margin. Centering therefore starts as an
   approximately uniform ~1.3x gradient rescale. With Stage-2 learning rate fixed across
   cells, any fixed-step contrast mixes "better objective" with "faster training".
   `paper_c effective-lr` measures this on the locked reference margins.
2. **It is underpowered.** The parent within-checkpoint seed SD of transfer macro-AP is
   0.0355. Seeds inside a checkpoint share model, manifest, recipe and data order, so the
   independent unit is the checkpoint: 4, not 20. A one-sided 97.5% bound then needs a true
   effect above **0.035**, larger than the +0.061 an explicit KL anchor achieves on the same
   suite and far larger than a 1.3x rescale should buy.

### Primary: the frontier estimand

For each cell `(model, seed, sampler, objective)` the saved ladder yields a trajectory of
`(represented AP, transfer AP)` pairs, ordered by step and reduced to its running maximum in
represented AP. For two objectives sharing `(model, seed, sampler)`, both trajectories are
interpolated onto a fixed grid of 21 represented-AP levels inside their overlapping range and
the vertical transfer gap is averaged:

```text
F_ref(model, seed, sampler) = mean over grid of [ transfer_DPO(g) - transfer_PairCE(g) ]
```

`F_pair` and `F_total` are defined identically on their own objective pairs. Cells overlapping
by less than 0.01 represented AP are reported `insufficient_overlap`, never extrapolated.

This is the primary estimand because **a pure learning-rate rescale moves a run along its own
trade-off curve without moving the curve**, so `F` is exactly zero under the confound and
nonzero only when an objective buys transfer at equal represented ranking. The confound becomes
the null hypothesis. The ladder is consequently load-bearing, not a selection convenience.

### Aggregation and intervals

`F` is marginalized with equal weight across the two fixed samplers, then across models, so no
sampler can be chosen post hoc and no checkpoint can dominate. The bootstrap resamples
**models**, not cells; resampling cells would understate the interval by roughly sqrt(5).

### Secondary

`C_pair`, `C_ref`, `C_total` at the target-matched checkpoint; the objective-by-sampler
interaction; `C_total(uncertain) - C_total(matched_random)`; and step-matched contrasts, which
are explicitly labelled confound-exposed.

## Power gate

The design must state what it can detect before Stage 1, or a null is uninterpretable.

```text
target effect (transfer)                 0.02
measured parent seed SD                  0.0355
clustered units                          4 checkpoints
point estimand MDE (clustered)           0.035   -> NOT powered
frontier estimand MDE (clustered)        0.017   -> powered, IF pairing removes >= 67% of variance
required pairing variance reduction      <= 0.331 surviving
```

`paper_c power --gate` exits nonzero whenever the primary MDE exceeds the target. The assumed
pairing variance reduction in `config/study.json` is a placeholder that the pilot must replace
with a measurement.

## Pilot gate before the full panel

The 120-cell panel is authorised only after a two-model pilot (`stage2.pilot`: Qwen2.5-1.5B and
SmolLM2-1.7B, all five seeds, both samplers, all three objectives) supplies:

1. the realised variance-reduction factor of the paired frontier contrast, replacing the assumed
   value; and
2. a first `F_ref` point estimate, to check it lies inside the resolvable range.

The pilot cannot establish the result: two models cannot support the equal-weight model marginal,
and pilot cells are reused in the full panel rather than treated as independent replication. Its
only function is to decide whether the remaining compute can answer the question. If the measured
reduction leaves the primary MDE above target, the target, seed count, or panel must change
before proceeding, and that decision is recorded in the design lock.

## Outcomes

Primary outcomes are tie-aware, benchmark-macro AP on represented-source and
dataset-held-out transfer suites. Reliability outcomes include calibration-only
5% FPR operating points, Brier/NLL, OR-Bench safe false-positive rate, and
HarmBench recall. Mechanism diagnostics include two-verdict KL from Stage 1,
signed-margin movement, entropy, saturation, LoRA norm, examples/tokens,
GPU-seconds, and peak memory.

The primary test suite scores only the 120 selected adapters. All 480 candidates
may be scored after selection only as explicitly exploratory trajectory evidence;
they cannot redefine checkpoint selection or the primary estimand.

## Retrospective versus confirmatory evidence

The reused Paper A evaluation suite is retrospective and receives two-sided 95%
paired hierarchical intervals. It cannot support confirmatory language.

A future sealed cohort may call reference centering successful only if the
prospective child lock exists, the full panel is feasible, and simultaneous
one-sided 97.5% bounds establish:

```text
LCB(F_ref transfer)    >  0.00
LCB(F_ref represented) > -0.02
```

with no greater than a 0.02 absolute increase in OR-Bench-style safe FPR and no
greater than a 0.02 absolute decrease in HarmBench-style recall. Power, cohort
sources, labeling, and unsealing procedures must be specified before enabling
`confirmatory.enabled`.

## Lock chain

1. **Protocol lock:** configuration, source bytes, and vendored parent inputs
   before Stage-1 candidate generation.
2. **Stage-2 design lock:** frozen prompt caches, partition, 20 accepted Stage-1
   adapters, and 20 reference/selection cells before candidate training.
3. **Selection lock:** all 480 candidate adapters, 500 development score bundles,
   exact checkpoint-selection table, and 120 selected adapter identities before
   retrospective test scoring.
4. **Prospective child lock:** selected adapters, sealed cohort commitment,
   estimands, harm margins, analysis code, power analysis, and unsealing record.

The current code creates the protocol lock, requires complete 20-cell inventories
for the Stage-2 design lock, validates the full candidate/selection shapes, and
deliberately refuses prospective lock creation.
