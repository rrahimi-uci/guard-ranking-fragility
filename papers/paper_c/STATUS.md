# Paper C status — ARCHIVED PREDECESSOR

> **SUPERSEDED 2026-07-25.** Paper C is now the study in
> [`specialize_then_align/`](specialize_then_align/README.md) — *Specialize, Then Align?* — per
> `specialize_then_align/provenance/MIGRATION_DECISION.md`. The reference-centering study
> recorded below is historical: its final smoke failed the policy/reference identity check
> before Stage 2, all seven cloud attempts exited unsuccessfully, and it supplies no reusable
> result, pilot estimate, readiness gate, or lock parent. Nothing here may be cited as evidence
> for the successor study.
>
> The design work below — the learning-rate-invariant frontier estimand and the executable power
> gate — remains a correct treatment of *that* study's confound and is retained for the record.
> It does not transfer: the successor's two primary arms share one objective and one reference,
> so the initialization-time gradient-weight asymmetry it was built to neutralize does not arise
> there.

**Date:** 2026-07-25  
**State:** archived; no claim-bearing result, no successor authority

## Design revision: frontier estimand + power gate (2026-07-25)

A pre-execution review found the original primary estimand both confounded and underpowered,
and both defects were fixed before any GPU time was spent.

- **Confound.** At step zero DPO weights every row at 0.5 while PairCE weights it at
  `sigma(-beta*m_ref)`; on this panel that is a mean **1.30x** near-uniform gradient rescale
  (weight CV 0.17, 93% of rows at positive margin). Fixed-step contrasts therefore mix
  objective with effective learning rate. `paper_c effective-lr` measures it.
- **Power.** Measured parent seed SD is 0.0355 and the independent unit is the checkpoint, so
  the point estimand's MDE is **0.035** against a 0.02 target: not powered. The frontier
  estimand's MDE is **0.017**: powered, conditional on pairing removing >= 67% of variance.
- **Fix.** The primary estimand is now the transfer gap at matched represented AP over the
  checkpoint ladder, which is exactly zero under a pure rescale. `paper_c power --gate` blocks
  Stage 2, and a two-model pilot must replace the assumed variance reduction with a measured one.


## Completed in this isolated workspace

- narrowed research question and exact three-objective factorization;
- candidate configuration with immutable model/tokenizer revisions;
- deterministic, hash-bound snapshot of the six parent Paper A manifests;
- family-disjoint Stage-2 development partition (960 update / 240 development rows);
- validated safety-amended protocol lock authorizing only Stage-1 candidates:
  `artifacts/locks/PROTOCOL_LOCK_SUPERSEDING_009.json`
  (`07576b8f0ec81f1168b9a087777d8dade8b58658ecccc8aa02c5e45d8a5cf5f8`);
- pure CPU objective, split, selection, hashing, and path-safety contracts;
- self-contained Stage-1/Stage-2 GPU runner foundation;
- GCP credential-safe preflight and launch plan;
- private GCP bucket, least-privilege runner service account, and empty HF secret;
- two-hour auto-delete GPU launcher with verified result retrieval;
- manuscript draft with explicit placeholders instead of invented results;
- offline contract tests and a compiled six-page LaTeX draft.

## Lock chain state after the design revision

The design revision changed `config/study.json` (new `analysis.primary_estimand`,
`analysis.bootstrap_cluster_unit`, `analysis.power`, and `stage2.pilot`). The protocol lock hashes
the configuration **file bytes**, so every existing `PROTOCOL_LOCK*` now reports
`configuration bytes drifted`. That is the contract working as intended, not a defect.

Per `PROTOCOL.md`, a *superseding* lock may not change scientific configuration -- only bind a
safety-only execution amendment. This change is scientific, so the chain requires a **fresh**
protocol lock rather than `SUPERSEDING_010`. Nothing is lost: no GPU run, no Stage-1 adapter and no
claim-bearing result was ever produced under the old locks, and they remain on disk as the
historical record of the superseded design.

Re-locking is deliberately left to the researcher, since the protocol lock is the artifact that
authorizes Stage-1 generation:

```bash
make protocol-lock    # bootstrap-inputs -> validate-config -> power --gate -> create-protocol-lock
```

`make validate` (config + objective identities) and `make test` pass unchanged; only
`validate-lock` against a pre-revision lock fails, and it should.

## Open gates

- run `make power` and record the gate result in the design lock (no GPU needed);
- regenerate 20 Stage-1 adapters in this namespace;
- implement and run reference-margin scoring for all Stage-1 cells;
- freeze the per-cell uncertain and matched-random selections;
- pass a three-objective single-GPU smoke with identical initial state and rows;
- finish development-score and selected-checkpoint scorers;
- bind the full candidate inventory into a final lock;
- run the two-model Stage-2 pilot and re-run the power gate on its measured pairing variance;
- launch the remaining Stage-2 cells only if the gate passes;
- run retrospective analysis, then acquire a genuinely sealed confirmation cohort.

Until those gates close, the manuscript must use future tense for experiments and
must not report performance conclusions.

The authorized `HF_TOKEN` transfer created Secret Manager version 1 in
`paper-c-hf-token` without exposing the token. No GPU VM result exists yet.
