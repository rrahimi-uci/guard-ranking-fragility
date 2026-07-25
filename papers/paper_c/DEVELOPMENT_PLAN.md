# Paper C development plan

All work products and outputs for this plan stay under `papers/paper_c/`.

## P0 — Isolated contract and paper draft

Deliver the self-contained package, exact configuration, protocol, manuscript,
CPU tests, cloud security model, and immutable protocol lock. No imports from
mutable repository training or scoring modules are allowed.

Exit gate: config/loss/path tests pass, the draft PDF compiles with a visible
unrun-study banner, and the protocol lock binds this isolated source tree.

## P1 — Input and prompt freeze

Run `bootstrap-inputs` to copy and verify the parent lock plus all six manifests.
Generate one frozen prompt-token cache per model and manifest. Confirm the locked
verdict token IDs and exact ordered identities.

Exit gate: changing any copied input, prompt token, order, or tokenizer revision
invalidates validation.

## P2 — Stage-1 regeneration

Regenerate 20 completion-SFT adapters inside `artifacts/stage1/`, with the exact
Paper A recipe. Build an inventory binding adapter directories, run metadata,
prompt caches, environment, and hashes.

Exit gate: 20/20 adapters reload; no run is overwritten; all model/seed cells are
present and start from the correct pinned base revision.

## P3 — Reference and selection freeze

Score every Stage-1 adapter on its train prompt cache. Produce text-free logits,
signed margins, the global family split, and per-cell uncertain/matched-random
selections.

Exit gate: every reference identity matches the ordered prompt cache; every
selection is source/label matched, disjoint, and byte-reproducible.

## P4 — Three-objective GPU smoke

Run one model/seed/sampler through VerdictCE, PairCE, and DPO from the same Stage-1
adapter and rows. Record runtime, GPU memory, loss, checkpoint reload, prompt and
sample hashes, and step-zero reference equality.

Exit gate: `SMOKE_AUDIT.json` passes, DPO starts at `log(2)`, all dropout modules
are zeroed, and the three cells share all non-objective identities.

## P5 — Design lock and candidate panel

Freeze the claim-bearing design lock only after P1–P4 inventories exist. Launch
the 120 Stage-2 runs and retain all 480 candidate checkpoints.

Exit gate: exact 120-run/480-checkpoint grid; immutable outputs; failures retained
and blocking rather than silently retried into replacement evidence.

## P6 — Development scoring and selection lock

Score 20 Stage-1 baselines and 480 candidates on Stage-2 development rows. Apply
the earliest-feasible rule without test access. Freeze the 120-cell selection
table and issue the selection child lock.

Exit gate: 500 development bundles, 480 candidates, and 120 selected cells are
complete and hash-bound. Infeasible cells are explicit.

## P7 — Retrospective evaluation and analysis

After selection lock validation, score base/Stage-1/selected Stage-2 models on
the frozen Paper A calibration, represented, transfer, and stress suites. Compute
paired factorial contrasts, reliability tables, sensitivity, and exploratory
candidate trajectories.

Exit gate: analysis regenerates without GPU/network from immutable text-free
scores; claim checks distinguish retrospective estimates from confirmation.

## P8 — Prospective confirmation

Design, power, source, label, and seal a genuinely uninspected cohort. Implement
and freeze the prospective child lock before unsealing.

Exit gate: auditable unsealing record and simultaneous decision-rule output.

## Immediate next executable action

CPU contracts and input bootstrapping are complete. A checksum-verified portable
`gcloud` is installed only in the ignored Paper C build tree, and the dedicated
bucket, runner service account, and empty HF secret exist. The next action is a
single billed smoke VM after the secret receives its explicitly authorized HF
token version and `cloud/preflight.sh` passes. The launcher enforces a two-hour
delete action and a parent-side verified-artifact cleanup; the full panel remains
unauthorized.
