# Paper C v2 — Development and Execution Plan

> **Current state (2026-07-25): protocol rewrite plus partial P1–P7 development scaffold; no
> claim-bearing lock.** Offline loss/split/selection primitives, a fail-closed lock skeleton, shared
> trainer, selected-checkpoint scorer, analyzer, and focused tests now exist; the focused and full
> repository test suites pass locally. The old TRL runner fails
> closed. Paper A scores exist, but the required Stage-1 adapters, reference-logit producer,
> Stage-2-development scorer, authoritative inventories, GPU smoke evidence, and Paper C result do not.

## Outcome

Deliver an auditable study in `artifacts/paper_c_dpo_v2/` that can answer whether DPO's frozen-reference
centering improves guard transfer beyond PairCE and continued verdict SFT, separately on uncertain and
matched-random examples.

## Non-negotiable gates

1. No evaluation manifest is exposed to hyperparameter or checkpoint selection.
2. No final run occurs until the lock binds code, configuration, parent inputs, Stage-1 adapters,
   selection/reference artifacts, environment, and clean git commit.
3. Every Stage-2 arm for a `(model, seed)` starts from the same adapter hash.
4. Every within-sampler objective comparison consumes the same rows, order, batches, and maximum updates.
5. The scorer must load every non-base adapter and bind its bytes to score metadata.
6. Retrospective and prospective outputs use different directories and claim language.

## Work breakdown

### P0 — Preserve and narrow the protocol

Deliverables:

- superseded v1 marker in `docs/paper-c-prereg.md`;
- normative `docs/paper-c-prereg-v2.md`;
- refreshed design rationale, this development plan, and code design;
- `configs/paper_c_dpo_v2.json`.

Gate: all live Paper C indexes point to v2, and stale "code ready," "first guard DPO," objective-zoo,
GRPO-null, and ExpGuard-prospective claims are removed or marked historical.

### P1 — Pure, offline-verifiable study primitives

Files:

- `experiments/paper_c_dpo_common.py`;
- `tests/test_paper_c_dpo.py`.

Implement exact losses, stable two-verdict probabilities/entropy, deterministic family split,
uncertain/matched-random selection, condition-grid construction, hashes, and metadata validation.

Gate: analytic loss identities and deterministic-selection tests pass without model downloads.

### P2 — Lock builder and preflight validator

Files:

- `experiments/lock_paper_c_dpo.py`;
- `experiments/validate_paper_c_dpo.py` or an equivalent lock subcommand.

The development lock may be built before Stage 1. Finalization must require:

- verified Paper A parent lock and manifest hashes;
- clean git commit and bound execution sources;
- exact training/scoring environment;
- 20 valid Stage-1 adapter/run-metadata pairs;
- frozen Stage-2 development/update family split;
- frozen selection and reference-margin schemas.

Gate: changing any bound byte makes validation fail.

### P3 — Regenerate the shared Stage-1 adapters

Use the parent Paper A recipe and immutable model/tokenizer revisions. Write to the Paper C namespace,
not Paper A. Each `(model, seed)` produces one adapter that is never modified in place.

Gate per cell:

- adapter reloads;
- decision-token IDs and prompt template match the lock;
- wrapper-preserving truncation passes;
- run metadata binds manifest, recipe, environment, source, and adapter hashes.

Gate for P3: 20/20 valid adapters. Paper A score files alone do not satisfy this gate.

### P4 — Freeze Stage-2 split, selections, and reference margins

For each Stage-1 cell:

1. split the 1,200 training rows by family into the locked Stage-2 update/development pools;
2. score Stage 1 on those prompts once;
3. write text-free per-row safe/unsafe logits and correct signed reference margins;
4. select uncertain and matched-random update rows by the v2 algorithm;
5. hash selected sample IDs, ordered rows, rendered-prompt fingerprints, and margin values.

Gate: every sampler has the planned source/label counts; no family crosses update/development; repeated
generation is byte-identical.

Implementation status: split/selection conversion exists; the lock-bound Stage-1 reference scorer and
inventory builder are still missing. Therefore this gate is open and the `finalize` subcommand is
intentionally disabled rather than emitting a partial final lock.

### P5 — One shared Stage-2 trainer

File: `experiments/train_paper_c_dpo.py`.

One data path and model path serve `verdict_ce`, `pair_ce`, and `dpo`. Objective-specific branches may
compute only the scalar loss. Reference margins are read from the frozen cache, not recomputed with an
unbound model. Save checkpoints at 25/50/100/200 updates plus a compute ledger.

Development sequence:

1. pure dry run for all six conditions;
2. tiny local model integration test if cached weights exist;
3. one real GPU smoke cell for all three objectives on identical rows;
4. compare initialization hashes, first-batch IDs, prompt hashes, losses, gradients, and adapter reload;
5. only then launch the 120 Stage-2 cells.

Gate: no objective-specific hidden trainer defaults; DPO initialization loss is `log(2)` when policy
and reference margins are identical, within numerical tolerance.

### P6 — Paper C-specific scoring

File: `experiments/eval_paper_c_dpo.py`.

Reuse Paper A's low-level prompt budgeting, token resolution, calibration, and threshold helpers, but not
its hard-coded `{base,sft}` inventory. Score:

- each base once;
- each Stage-1 adapter once;
- every selected Stage-2 checkpoint.

Emit per-row logits plus immutable bundle metadata. Synthetic mode must visibly distinguish conditions
and verify that every non-base condition loads an adapter.

Gate: complete condition grid, immutable adapter inventory, cache invalidation on any identity drift,
and no train rows in the evaluation artifact.

Implementation status: a development-only selected-checkpoint retrospective path exists. Claim-bearing
final scoring is intentionally disabled. Separate `score-reference` and `score-stage2-dev` paths,
including multi-checkpoint inventory keys, remain required before P6 closes.

### P7 — Checkpoint selection and retrospective analysis

File: `experiments/analyze_paper_c_dpo.py`.

First select checkpoints using only the Stage-2 development artifact. Freeze a selection table. Then
analyze the already-scored Paper A evaluation rows without reopening selection.

Required outputs:

- represented and transfer macro-AP by objective/sampler/model/seed;
- `C_pair`, `C_ref`, `C_total`, and objective-by-selection interaction;
- paired hierarchical intervals and leave-one-source/model sensitivity;
- calibration, operating-point, stress, movement, saturation, and compute tables;
- descriptive base-plus-adapter composition;
- machine-readable decisions and a provenance manifest.

Gate: analysis regenerates from committed text-free scores without a GPU or network.

Implementation status: the two-phase analyzer exists and refuses test rows during checkpoint selection.
Both commands are development-only and cannot be exercised end to end until the dedicated
development-score artifact and final selection bindings exist. Compute/LoRA diagnostics and
full-vocabulary KL remain unavailable under the current scorer schema and must be marked unavailable
rather than inferred.

### P8 — Prospective confirmation

Acquire and audit a genuinely uninspected, family-disjoint cohort only after P7 code is frozen. Before
unsealing, bind adapter/checkpoint hashes, margins, harm gates, analysis code, and output schema in a
prospective child lock.

Gate: an unsealing record proves that no cohort content or aggregate result was available earlier.

### P9 — Manuscript integration

Working title:

> **What Does Reference Centering Buy a Safety Guard? A Matched SFT, PairCE, and DPO Study**

The manuscript must lead with the loss decomposition and matched design. It may describe Paper A results
as retrospective continuity, but only P8 can carry confirmatory language.

## Artifact layout

```text
artifacts/paper_c_dpo_v2/
  LOCK.json
  locks/                         # amendments / prospective child lock
  stage1/<model>/seed_<seed>/    # immutable shared SFT adapters
  splits/                        # Stage-2 update/dev family assignments
  reference/<model>/seed_<seed>/ # text-free logits + margins + metadata
  selections/<model>/seed_<seed>/
  runs/<model>/seed_<seed>/<sampler>/<objective>/
  scores/retrospective/
  scores/prospective/
  analysis/retrospective/
  analysis/prospective/
  provenance/
```

## Estimated execution size

- Stage 1: 20 adapters.
- Primary Stage 2: 120 short runs.
- Scoring: bases + Stage 1 + all saved Stage-2 checkpoints.
- Sensitivity: deferred until the primary artifact is immutable.

Cost is not estimated from the stale v1 TRL timing. A three-objective smoke must measure tokens,
GPU-seconds, and peak memory before procurement or launch.

## Explicitly deferred

- GRPO/KTO/ORPO/IPO objective zoo;
- policy-conditional `{allow, block, escalate}` DPO;
- SME adjudication of the mortgage policy preferences;
- native multi-class guards whose decision contract is not exactly the binary Paper A head;
- any claim that movement causally mediates transfer or composition.
