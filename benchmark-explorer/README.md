# Benchmark Explorer — withdrawn; superseded by `apps/benchmark-explorer/`

This directory's published artifact has been **withdrawn from tracking**, and its
generator removed, because both violated the distribution ledger.

## What was wrong

`index.public.html` was a tracked, pushed, 55.4 MB single-file blob inlining 16,146 rows
across 10 sources. Among them were all 2,000 rows of `mortgage_guard_bench_2k_v0_1_0`,
whose own [`LICENSE_NOT_SELECTED.md`](../data/mortgage_guard_bench_2k_v0_1_0/LICENSE_NOT_SELECTED.md)
states that no publication license has been selected and that legal review is required.
[`benchmarks/registry/distribution.yaml`](../benchmarks/registry/distribution.yaml) records
that source as `local_only` with `permits_redistribution: false`. The file was verified to
carry substantive prompt text, not identifiers alone.

`generate.py` inlined every row of every source regardless of license, with no ledger
consulted and no allowlist. Keeping it would let the same artifact be recreated and
re-committed by anyone who ran the documented command. It is removed rather than fixed;
its replacement already exists and is gated. Recover it from Git history if needed.

## Use instead

[`apps/benchmark-explorer/`](../apps/benchmark-explorer/) builds from a positive allowlist
and fails closed:

```bash
make -C .. explorer-public     # ledger-gated; emits text only for approved sources
python apps/benchmark-explorer/src/build.py --target local   # full text, ignored dist/, never published
```

A source absent from the ledger is treated as forbidden, not permitted-by-default, and the
build refuses to run on an unknown source id. Eight negative tests in
[`apps/benchmark-explorer/tests/`](../apps/benchmark-explorer/tests/) cover the gate, and
[`tests/test_no_unlicensed_publication.py`](../tests/test_no_unlicensed_publication.py)
fails the root suite if a bulk artifact like this one is ever tracked again.

**As of the current ledger, exactly one source is approved for verbatim redistribution:**
`mortgage_benchmark_v1_hmda2022`, CC BY 4.0. A public build may therefore emit its text with
the required attribution; every other source still contributes counts and labels only, with
text replaced by content hashes.

## Still open — needs a human decision

Withdrawing the file from tracking stops further distribution *from this repository*. It
does **not** remove the blob from Git history, and it does not retract the copies already
pushed to the public remote. Purging it from history is a separate, irreversible migration
that rewrites published commits and breaks every existing clone, so it requires explicit
approval and coordination — it has not been done.

The local copy is retained, untracked, as `index.public.html.withdrawn-local`.
