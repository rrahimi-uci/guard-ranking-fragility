# paper-c-specialize-align-mortgage-v1

Phase 4 study package for study `paper_c_specialize_align_mortgage_v1`, migrated from
`papers/paper_c/specialize_then_align/` per Phase 5 of the repository layout plan.

**This is a copy, not a move.** The predecessor tree remains intact as a compatibility
surface; the plan prohibits moving an active tree before the new version verifies
independently. Both verify identically at 66 passed / 1 failed.

**The study is stopped and this package authorizes nothing.** Copying does not upgrade
a development result. All five readiness gates remain false, no sealed cohort exists,
and no claim is made about whether specialist-sourced preferences improve a guard —
see `STATUS.md` and `provenance/MIGRATION_MANIFEST.json`.

```
config/       study and smoke configuration
environment/  pinned GPU runtime record; the Cloud SDK is NOT vendored here
schemas/      sample, policy, preference, cohort, readiness contracts
src/          paper_c_sta package
tests/        CPU-only scientific contract tests
tools/        entrypoints for cells, proposal, scoring, pilot freeze
cloud/        VM runner scripts; expect gcloud/gsutil on PATH
manuscript/   the stopped study's write-up
provenance/   migration manifest, external-object manifest, predecessor pointer
locks/        tracked candidate lock (non-authorizing)
```

## Verify

```bash
make test          # 66 passed, 1 failed — the declared expected_fail
make validate      # configuration contracts
make readiness     # exits nonzero by design until all five gates carry evidence
```

The declared failure is `test_tracked_candidate_lock_binds_live_sources_and_authorizes_nothing`:
the candidate lock binds live source bytes and the source has evolved. It is recorded as
`expected_fail` in [`studies/registry.yaml`](../registry.yaml), and `make check-locks`
errors if it ever starts passing.

## Repository discovery

Fixed-parent discovery was replaced with marker-based discovery during migration.
`parents[2]` is correct at the predecessor's depth of 3 but resolves *outside* the
repository at this package's depth of 2, which would have silently read corpora from the
wrong place. `contracts.repository_root()` now searches for a marker instead, and the fix
was applied to both trees so they stay behaviourally identical.

## Outputs

New local execution belongs in ignored `runs/paper-c-specialize-align-mortgage-v1/`.
Existing development artefacts live in GCS and are inventoried in
`provenance/EXTERNAL_OBJECT_MANIFEST.json` (114 objects, 19.15 GiB, development-only).
