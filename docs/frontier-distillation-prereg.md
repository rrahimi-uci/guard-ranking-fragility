# Phase B preflight notes — frontier-distilled SLM

**This is not a preregistration.** `papers/unified-report/proposal.md` §20 is normative for the
frontier-distillation study: it fixes the primary student, the sole confirmatory hypothesis, the
release gates, and the phase gates. This file records only Phase B preflight material (§20.17)
produced without tripping the Phase A gate, plus one licensing gate Phase A must resolve.

Machine-readable form: `artifacts/frontier_distill_v1/protocol/claim_registry.json`.

## Why there is no independent protocol here

An earlier draft of this document proposed its own confirmatory predicates — AP-on-transfer over
the four-checkpoint Act I panel. Those are **withdrawn**, because they conflict with the
normative source rather than complement it:

| This draft proposed | §20 fixes |
| --- | --- |
| 4-checkpoint Act I panel | SmolLM3-3B base as the primary student (§20.5) |
| AP on Act I `transfer_test` | `H_frontier`: TPR@5%FPR vs gpt-5.4 low (§20.3) |
| analysis-preregistered, not data-blind | sealed `D_fresh`, split before teacher calls (§20.6) |
| teacher-scores-only as the headline arm | `D2_kd`, one rung of an 8-arm ladder (§20.10) |

The last row matters most. §20.4 argues that a student trained only to reproduce teacher
verdicts has *no principled basis for systematically correcting the teacher* — so a
teacher-only arm is a ladder rung, not a candidate for the primary claim. Crossing the frontier
line needs information absent from a single GPT verdict: independent gold, counterfactual pairs,
disagreement mining, a low-FPR objective, and a replay anchor.

**No training has been run.** §20.1 declares the study unauthorized for claim-bearing training
or data collection, and §20.17 Phase A gates all external teacher collection and all GPU
training behind a reviewed and hashed preregistration.

## What this preflight contributes

### 1. A candidate `D_teacher` pool, committed and text-free

`artifacts/frontier_distill_v1/teacher/` — 4,188 rows of gpt-5.4 (low) risk score over
`beavertails` + `openai_moderation`, written by `gpt-baseline/export_distill_teacher.py`.

Two properties make this Phase-A-safe: the teacher calls were **already made** for the
evaluation baseline, so no new external collection occurred; and `gpt-baseline/raw/` is
gitignored working state that exists only on the machine that ran the baseline, so nothing
bound by a LOCK could have read it. The export lifts the signal into committed provenance, the
same fix `export_expguard_scores.py` applied to the ExpGuard evaluation rows.

### 2. Measured teacher quality — heterogeneous, and it changes the pool design

| Source | *n* | prevalence | teacher AUC (gpt-5.4 low) |
| --- | ---: | ---: | ---: |
| `openai_moderation` | 1,663 | .308 | **.9512** |
| `beavertails` | 2,525 | .567 | **.7343** |

Roughly 60% of the candidate pool by row count carries a teacher signal at AUC .734 — label
convention disagreement rather than teacher capability. §20.6's quota design and §20.7's
annotation contract should treat these as two tiers, not one pool.

**Proposed weak-teacher gate**, for adoption into the §20 lock at Phase A: a source whose
measured teacher AUC is below 0.80 may not carry a primary claim alone, and its contribution
must be reported separately. Recorded now, before any distilled score exists, so it can never
become a post hoc exclusion. It is tripped in advance by `beavertails`.

### 3. Proposed hold-clean rule

No row from `expguard`, `xstest`, `jailbreakbench`, `wildguardtest` or `wildjailbreak` may enter
any arm's training **or selection** data, at any weight. ExpGuard is the report's only external
expert-annotated evaluation surface; the other four are Act I's `transfer_test`. This
operationalizes §20.6's "no ExpGuard evaluation text in any training or selection role" and
extends it across the Act I transfer suite.

### 4. Licensing gate — OPEN, and blocking for any released artifact

1. **`beavertails` is CC-BY-NC-4.0**, `commercial_use: false`,
   `derived_output_license: CC-BY-NC-4.0_inherited`. Weights trained on it inherit a
   noncommercial encumbrance — which bears directly on a study whose motivation is a *deployable*
   low-cost guard. Options: drop it from the pool, accept a research-only artifact, or obtain
   terms permitting the derived use.
2. **Provider terms on teacher outputs.** Distilling a provider's model outputs into a competing
   safety classifier is governed by that provider's terms of use.

Publishing the *measurement* is a different posture from shipping the *weights*; this gate
governs the latter and does not block the former. Both are human decisions, consistent with the
ledger's `default_decision: local_only`.

## Motivating evidence committed alongside this

`artifacts/frontier_general_h2h/h2h.json` (via `experiments/eval_frontier_general_h2h.py`)
establishes that **the frontier gap is a property of the regime, not of model size**. On
represented sources the panel's small tuned guards beat gpt-5.4 (low) — Qwen2.5-**1.5B** SFT on
`prompt_injections` reaches TPR@5%FPR .948 (AUROC .9928) against the reference's .741 (AUROC
.8731), paired Δ +0.185 [+0.065, +0.423], with 5 of 12 panel-SFT cells clearing zero at 95%. On
Act I transfer sources the same guards lose, and tuning is what costs them.

This bears on §20 in two ways: it adds a regime dimension to the §20.2 target-to-beat table,
which currently reads the gap off ExpGuard alone — an external source the panel never trained
on, hence the transfer regime only; and it supplies direct evidence for the §20.4 premise that
the remaining gap is a *transfer* gap rather than a capacity gap.

Flavor: retrospective, estimation-only. These rows and this panel were inspected during
development.
