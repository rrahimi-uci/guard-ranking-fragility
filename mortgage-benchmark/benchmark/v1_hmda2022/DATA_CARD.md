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
- The `private_test` split was **intended** to be sealed, but the committed release directory includes `private_test.jsonl` with full text, and the GPT baseline has already scored it (241 = 146 + 95 rows). Treat it as dataset-held-out by convention, **not** as a sealed cohort.

## Splits
- `train`: 604 rows
- `dev`: 149 rows
- `public_test`: 146 rows
- `private_test` (committed here with text; intended-sealed, already spent): 95 rows

## License
**CC BY 4.0**, selected 2026-07-27 by Reza Rahimi, PhD (JazzX AI). Redistribution — including
verbatim rows — is permitted with attribution:

> MortgageGuardBench v1_hmda2022, Reza Rahimi, PhD (JazzX AI), licensed CC BY 4.0.

The factual grounding is the public HMDA 2022 loan-level snapshot, a U.S. Government work
carrying no U.S. copyright (17 U.S.C. §105). The FFIEC/CFPB terms-of-use question this card
previously named as a precondition was checked on 2026-07-27 and no separate restriction was
found. The normative record is `benchmarks/registry/distribution.yaml`.

Three limits survive the licence and travel with any redistribution. The prompts are
**synthetic** and the labels are **LLM-judge and policy-card-consistent, not SME-adjudicated**,
so the data supports measurement and not a fair-lending or compliance finding about any lender,
model, or population. `CHECKSUMS.txt` covers **release bytes only** — not the generator, judge,
configuration, or code. And the prompts **solicit violations by design**, so a redistributor
should still handle them as harmful-content samples (`sensitive_text_class: synthetic_harmful`).

*Licence-change record.* Before 2026-07-27 this section stated that no licence had yet been
selected. Changing it altered this file's digest, so `CHECKSUMS.txt` was re-frozen for
`DATA_CARD.md` alone:
`c235e00e59e05619966baed1035c39375e7dfe4abee0a37fe02c85579f18bd18` → the value now recorded
there. **No data file changed**: `dev.jsonl`, `public_test.jsonl`, `private_test.jsonl`,
`train.jsonl`, `MANIFEST.json`, `SOURCES.json`, and `public_index.json` all keep their original
digests, so the frozen benchmark itself is untouched and every published number still stands.

## Reproduce
See the folder README. `make all` rebuilds the whole benchmark from the frozen design + seed.
