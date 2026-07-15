# Paper B — Compose, Don't Tune?

This folder contains the first executable Paper B workspace:

- [`compose_dont_tune.tex`](compose_dont_tune.tex) — formal 11pt `article` draft;
- [`PLAN.md`](PLAN.md) — local development checklist and link to the authoritative plan;
- [`code/build_pilot_artifacts.py`](code/build_pilot_artifacts.py) — fail-closed renderer for the
  exact reviewed clean-v2 retrospective result;
- [`code/validate_protocol.py`](code/validate_protocol.py) — prospective protocol validator;
- [`config/prospective_protocol.template.json`](config/prospective_protocol.template.json) —
  explicit incomplete contract for the future claim-bearing run;
- [`generated/`](generated/) — deterministic manuscript inputs and their hash manifest; and
- [`tests/`](tests/) — evidence and protocol regression tests.

## Current status

The article is a **retrospective working draft**. Paper A's clean-v2 scores are lock-verified, but
the Paper B operator was developed after that lock and the transfer benchmarks had been inspected.
No prospective Paper B claim has run. The draft therefore reports conditional fixed-panel
estimates and a future protocol; it does not claim dominance, Pareto improvement, noninferiority,
a universal remedy, a mechanism, or deployment readiness.

## Build and verify

From this folder:

```sh
make evidence
make verify
make pdf
```

Or run `make` for verification plus PDF compilation. The build uses `tectonic` and writes temporary
compiler files to `build/`, then copies the final PDF to `compose_dont_tune.pdf`.

`make protocol` validates the current draft contract and prints its known gaps.
`make protocol-locked` is the future claim-bearing gate and intentionally fails against the
template until the prospective cohort, margin, controls, statistics, systems measurements, and
software lock are complete.

## Evidence boundary

The generated pilot inputs are anchored to:

- Paper A lock SHA-256 `cabc8dee9b158773ce0be86f799ec3833c33c18787a2aa74d05ed1a261682c25`;
- score SHA-256 `b941ddbaea7057ab1f224c510687ec5748916f5eca6a78e1d1f429e0ede5a1c3`;
- composition-result SHA-256 `92c2cbc3ea71d5e6c72bf0e6f7eb0d3ef15f0e61f9fffaada885dade460e3ccc`;
- exact status `clean_v2_retrospective_estimation`.

Changing any of those inputs fails generation until the new evidence is explicitly reviewed and
re-anchored. Future prospective artifacts should use a separate Paper B namespace and must not
overwrite this retrospective preview.
