# Mortgage Guardrail Benchmark — frozen release

`v1_hmda2022/` is the **fixed, versioned** mortgage-guardrail benchmark built by the agentic
generator in this folder (grounded in the public HMDA 2022 snapshot). It is committed as a
static artifact: the LLM *generation* is intentionally **not** reproducible (temperature > 0), so
the released dataset itself is the citable, checksummed object — like the HMDA snapshot it is
grounded in. Results are reproduced by **evaluating guards against this frozen data**, not by
regenerating it.

## Contents (`v1_hmda2022/`)
- `train.jsonl` (604) · `dev.jsonl` (149) · `public_test.jsonl` (146) — distributed splits.
- `private_test.jsonl` (95) — the intended sealed/confirmatory split; committed here for
  in-repo reproducibility, but treat it as held-out (don't tune on it).
- `public_index.json` — text-free index (ids, hashes, labels, family links, counts).
- `DATA_CARD.md`, `SOURCES.json`, `MANIFEST.json`, `CHECKSUMS.txt`.

**994 rows total.** Every row is `synthetic=true`, `contains_real_pii=false` (verified at freeze:
0 violations). Dual labels: `general_safety_gold` (G) × `mortgage_policy_gold` (D). Quadrants:
G0/D0 450, G0/D1 502, G1/D1 42, G1/D0 0. Domains: fair_lending 204, fraud 112, udaap 90,
disclosure 66, atr_qm 54, privacy 18, benign 450.

## Honest status (must read before citing)
- Labels are **LLM-judge, policy-card-consistent — NOT SME-adjudicated** (all 24 cards
  `sme_signoff:false`; the judge report says "self-consistency, NOT a Fleiss-κ human study").
  Confirmatory fair-lending claims require SME adjudication first.
- The **G1/D0 quadrant is empty** (the safety model would not author generic jailbreaks), so the
  orthogonal-2×2 is only partly populated.
- Decontamination was run against the **legacy** Paper A general sources (exact-hash); re-running
  against the v2 sources + near-dup is required before any joint Paper A/B claim.
- No guard has been scored yet here — baseline guard evaluations are the next step (see the
  generator's `magen/evaluate.py` for the canonical-metric evaluator).

## License
See `v1_hmda2022/DATA_CARD.md` — a license must be selected before external redistribution; HMDA
grounding is a U.S. Government work (17 U.S.C. §105).
