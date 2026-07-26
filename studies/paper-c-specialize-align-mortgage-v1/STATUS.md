# Paper C v2 status — STOPPED before the primary panel

**Date:** 2026-07-26
**State:** stopped by decision after the disjoint pilot; the primary panel was never
authorised and no confirmatory claim exists

## Why it stopped

The pilot did what a pilot is for: it found that the instrument could not express its
own treatment. Under a three-action head with gold-based adjudication and
deterministically derived structured fields, the two candidate-source inventories were
**98.9% and 97.7% byte-identical**. The two CM-DPO arms therefore trained on the same
pairs for ~98% of matched events, and the primary contrast
`specialist_cm_dpo - generalist_cm_dpo` was not estimable.

A fix was implemented and verified — letting each teacher author its own verdict drops
byte identity to **14.4%** — but the study was stopped before spending the remaining
~5 hours and ~$50 to run it, because the pilot arms sat within 0.006 of each other
against a projected primary resolution of ±0.007. The decision was to bank the
methodological findings rather than pursue a probable bounded null.

## What was completed

- 13,266-row corpus, 84.9% reused from existing repository benchmarks, 15.1% generated
  to fill two genuine gaps; 10,990 split units with zero family or content-family
  leakage.
- 60 infrastructure cells on A100: 10 joint multitask references, 50 category
  specialists. 400 steps on 6,608 rows in 92-105 s per cell.
- 4 preference inventories, category-wise temperature calibrated.
- The complete 44-cell pilot panel: 4 references + 20 specialists + 20 students across
  all five arms, trained and scored over the checkpoint ladder.
- Pilot freeze executed.

## What was never done, and is therefore not claimable

- the 66-cell primary panel;
- any sealed evaluation — no separately authored sealed cohort exists;
- the three ensemble baselines (implemented, never scored);
- SME sign-off, signed annotation rubric, licence ledger, two-reviewer adjudication.
  All five readiness gates remain false. Adjudication was automatic against gold, not
  human, and no row is counsel-reviewed.
- **No claim about whether specialise-then-align works.** The single measured contrast
  is invalid by construction and must not be cited in either direction.

## Findings that stand

1. **Source invariance.** With three actions, gold-based adjudication, and structured
   fields derived from gold, cross-model preference pairs are near-source-invariant:
   100% identical chosen actions, 98% byte-identical pairs, between-arm policy movement
   of 0.13 logits against 2.4 within-arm. Generative teacher authoring restores the
   treatment (identity 98.9% -> 14.4%) and, as a side effect, populates the
   `teacher_agreement` stratum that was structurally empty before (0 -> 303 of 355).
2. **Calibration infeasibility at 1.5-1.7B.** No operating point satisfies a 5%
   worst-category false-alarm target, a 10% review budget, and a bounded
   intervene-miss rate simultaneously, in any of the 20 pilot cells.
3. **Derived cohort size.** The specificity-cohort minimum is 1,825 ALLOW rows per core
   category from a Wald half-width of 0.01 on a 5% false-alarm rate, replacing the
   2,000 that was asserted in six places and derived in none.

## Implementation defects found and fixed

Four would have silently corrupted the full panel:

- `composite_alignment_loss` was imported by the trainer and never called; the
  soft worst-category term, gold anchor, and replay KL never reached the optimizer.
- `response_logprob` built log-softmax over the whole sequence x 151k vocabulary when
  only response positions contribute, exhausting a 40 GB A100 once the composite added
  its two extra forwards.
- Teacher probabilities were uncalibrated; fitted temperatures span 0.11 to 5.02, so
  raw softmax confounded teacher sharpness with teacher quality.
- `torch_pair_loss` silently ignored reference tensors on the uncentered arm.

Also added but never exercised: abstain enforcement out-of-expertise, and three
inference-time ensemble baselines (independent, OR-vote, routed).

## Restarting

The corpus, all modules, 60 infrastructure cells and 4 pair inventories are intact in
`gs://jazzx-gcp-poc-1-paper-c/v2/sta/`. A restart needs only the generative pair
inventories completed (~1.5 h), 20 pilot students re-run (~30 min), and the go/no-go
contrast read. Nothing already computed is wasted.
