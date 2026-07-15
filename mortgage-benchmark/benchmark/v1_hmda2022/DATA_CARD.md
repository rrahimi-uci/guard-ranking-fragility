# Mortgage Guardrail Benchmark — Data Card

**Version:** 0.1.0-agentic
**Built:** deterministic (seed 20260714) by the agentic HMDA-grounded generator in this folder.

## What this is
A request-screening benchmark for a mortgage-specific safety guardrail. Each row is one
incoming request to a mortgage workflow, carrying TWO independent policy labels
(`general_safety_gold` G, `mortgage_policy_gold` D), a derived `final_intervention_gold`, an
`action_gold` lattice, quadrant, and severity. Scenarios are grounded in the public HMDA 2022
National Loan-Level Snapshot for realism; **no real individual record or PII is reproduced** —
grounding uses aggregate/de-identified fields only.

## Honesty / construct validity
- Prompts are **synthetic**; harmful requests are *represented for detection*, never operational
  recipes. `contains_real_pii=false` is a hard schema constant.
- Labels are **policy-card-consistent, not legally authoritative**. `legal_review_status`
  records this. Confirmatory fair-lending claims require the SME-adjudicated subset (not yet done).
- The `private_test` split is **sealed**: it is not in this bundle; only its text-free index is.

## Splits
- `train`: 604 rows
- `dev`: 149 rows
- `public_test`: 146 rows
- `private_test` (sealed, not distributed): 95 rows

## License
LICENSE NOT YET SELECTED. HMDA public data are a U.S. Government work (17 U.S.C. §105, no U.S. copyright); confirm no separate FFIEC/CFPB terms-of-use restriction before redistributing generated prompts.

## Reproduce
See the folder README. `make all` rebuilds the whole benchmark from the frozen design + seed.
