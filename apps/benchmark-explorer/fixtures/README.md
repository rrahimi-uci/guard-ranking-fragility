# CI fixtures

Three hand-written synthetic rows, one per source id, carrying no upstream text. They
exist so `check-fast` can exercise the ledger logic in a clean clone.

They are **not** a reproduction of any benchmark and do not authorize a public build.
A passing fixture build says the allowlist works; it says nothing about whether the
full-data build is redistributable. That question is answered only by
`benchmarks/registry/distribution.yaml`.
