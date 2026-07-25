# Paper C v2 — Code and Artifact Design

## 1. Design principle

Paper C v2 uses one model-loading path, one prompt-building path, one dataset/collator, one optimizer
configuration, and one training loop. The objective branch is allowed to change only the scalar loss.
This prevents an objective comparison from silently becoming a comparison of TRL trainers and defaults.

## 2. Loss API

`experiments/paper_c_dpo_common.py` defines framework-independent scalar versions for testing and the
canonical formulas used by the torch trainer:

```python
signed_margin = y * (unsafe_logit - safe_logit)

verdict_ce = cross_entropy(full_vocab_logits, gold_verdict_token_id)
pair_ce    = softplus(-beta * signed_margin)
dpo        = softplus(-beta * (signed_margin - reference_signed_margin))
```

PairCE and DPO use the same locked `beta`; otherwise their comparison would mix reference centering
with loss temperature. The DPO reference is the frozen Stage-1 adapter, not the untuned base. Its signed margins are computed
once in evaluation mode, written text-free by `sample_id`, and bound by SHA-256. At initialization,
policy and reference margins must match and mean DPO loss must equal `log(2)`.

## 3. Data flow

```text
Paper A parent lock + train manifest
              |
              v
      Stage-1 SFT adapters
              |
              +--> family update/dev split
              |
              +--> frozen prompt logits/reference margins
                              |
                              +--> uncertain IDs
                              +--> matched-random IDs
                                      |
                                      v
                     shared Stage-2 trainer (3 losses)
                                      |
                                      v
                      Paper C scorer and analyzer
```

Raw prompt text remains only in licensed local manifests. Selection, score, and release artifacts use
sample IDs, content hashes, labels where redistributable, and logits/derived statistics.

## 4. Module responsibilities

### `paper_c_dpo_common.py`

- parse and validate the v2 JSON configuration;
- normalize gold labels without hard-coded completion strings;
- compute signed margins, stable probabilities, entropy, PairCE, and DPO losses;
- create deterministic family splits and source/label-matched selections;
- enumerate the exact six-condition grid;
- hash ordered identities and canonical objects;
- validate text-free reference and selection artifacts.

This module imports no transformers/torch code and is the main offline test surface.

### `lock_paper_c_dpo.py`

- verify and bind the Paper A parent lock;
- bind configuration, protocol, execution sources, tests, environment, and artifact schemas;
- inventory Stage-1 adapters and run metadata;
- issue development locks and, only after all gates, a final lock;
- refuse in-place amendments.

### `train_paper_c_dpo.py`

- resolve tokenizer/model revisions and verdict IDs exclusively from the lock;
- use `paper_a_common.budgeted_prompt` with wrapper preservation;
- clone the same Stage-1 adapter into each Stage-2 cell;
- use a fixed sampler generator independent of training seed;
- predeclare objective, sampler, shared pairwise beta, zero Stage-2 dropout, learning rate,
  checkpoints, and compute ledger;
- retain failures with traceback and partial metadata.

No generic TRL DPO trainer is needed for the primary binary objective. Implementing the exact loss
directly removes sequence/EOS/reference/dropout defaults that would otherwise differ across arms.

### `eval_paper_c_dpo.py`

- load adapters for every condition except `base`;
- score the same two decision-token logits in one forward pass;
- calibrate and select thresholds on calibration rows only;
- recompute by default; any future cache may be used only when every
  model/tokenizer/adapter/prompt/row/code/environment hash matches;
- emit a complete bundle inventory and combined-score digest.

### `analyze_paper_c_dpo.py`

- validate the full grid and immutable score contract before statistics;
- select checkpoints from Stage-2 development scores only;
- compute benchmark-macro AP and paired contrasts;
- share family/model/seed bootstrap draws across objectives;
- report total, target-matched, and movement-matched views separately;
- regenerate manuscript tables from committed score artifacts.

## 5. Run metadata

Every Stage-2 run records at least:

```text
study_id, run_id, model_key, model/tokenizer revisions
objective, sampler, seed, training_seed, data_order_seed
lock/config/source/git/environment hashes
stage1_adapter_sha256, stage1_run_meta_sha256
selection_sha256, ordered_sample_ids_sha256
reference_margin_sha256, prompt_fingerprint_sha256
safe/unsafe token strings and IDs
loss formula/version, beta, optimizer/scheduler settings
max steps, completed steps, checkpoint steps
examples, prompt tokens, policy tokens, estimated FLOPs, GPU-seconds, peak memory
adapter/checkpoint hashes, status, failure, traceback
```

Storing the Paper A recipe while silently overriding DPO fields is prohibited. The effective Stage-2
configuration is a first-class hashed object.

## 6. Selection artifact

For each `(model, seed)` the text-free JSONL record contains:

```text
sample_id, content_sha256, family_id, source, gold
stage2_partition, two_verdict_probability_correct, two_verdict_entropy
selection_role, selection_rank
```

Metadata binds the Stage-1 adapter, tokenizer decision IDs, rendered prompt fingerprint, train manifest,
selection algorithm/version, fraction, and seed. Rebuilding the selection must produce identical bytes.

## 7. Reference-margin artifact

For every Stage-2 update/development row:

```text
sample_id, safe_logit, unsafe_logit, signed_reference_margin
```

Metadata binds model/tokenizer/adapter/prompt/input identities and dtype/device policy. DPO training
looks up margins by sample ID and fails on missing, duplicate, extra, or reordered identities.

## 8. Tests

Required offline tests:

- PairCE at `beta=1` equals binary cross-entropy over the two verdict logits; other locked beta
  values are the corresponding temperature-scaled margin loss.
- DPO equals the reference-relative chosen/rejected log-ratio formula.
- DPO initialization loss is `log(2)` and gradients have the correct sign.
- the same beta changes PairCE and DPO scale but not their preference direction.
- family splits never cross partitions and are deterministic.
- uncertain/matched-random selections preserve source/label counts and are deterministic.
- changing a score, sample ID, order, config field, or source byte changes the bound hash.
- the exact condition grid is six cells with no missing or extra names.

Required integration tests before finalization:

- tokenizer-specific verdict strings resolve to distinct single tokens;
- long prompts preserve the classifier wrapper;
- all objectives see identical first batches and initial adapter hashes;
- a tiny model completes, saves, reloads, and changes logits for all three objectives;
- evaluation loads non-base adapters and invalidates stale caches;
- analysis rejects incomplete grids and recovers known synthetic contrasts.

## 9. CLI target

The intended workflow is explicit. Commands marked `PENDING` are deliberate open development tasks,
not implied functionality:

```bash
.venv/bin/python experiments/lock_paper_c_dpo.py init ...
.venv/bin/python experiments/train_paper_c_dpo.py stage1 ...
.venv/bin/python experiments/prepare_paper_c_dpo.py partition ...
# PENDING: score-reference for each Stage-1 cell
.venv/bin/python experiments/prepare_paper_c_dpo.py select ...
# DISABLED until the reference/dev-score producers and candidate-inventory validator exist:
# .venv/bin/python experiments/lock_paper_c_dpo.py finalize \
#   --stage1-inventory ... --stage2-input-inventory ...
.venv/bin/python experiments/train_paper_c_dpo.py stage2 --objective pair_ce --sampler uncertain ...
# PENDING: score all candidate checkpoints on stage2_dev
.venv/bin/python experiments/analyze_paper_c_dpo.py select-checkpoints ...
# Development mode only; final scoring/analysis stays disabled until its lock bindings exist:
.venv/bin/python experiments/eval_paper_c_dpo.py --development --checkpoint-selection ...
.venv/bin/python experiments/analyze_paper_c_dpo.py retrospective --development ...
```

Bulk launch wrappers may be added only after each single-cell command passes and preserves the same
contract. Cloud provisioning and spending remain outside the scientific code path.
