# Design & planning notes

Working notes behind the study. They are historical design documents, not the paper;
the authoritative results live in [`../paper-a/`](../paper-a) and
[`../artifacts/paper_a_sft/`](../artifacts/paper_a_sft).

## Paper A — *The Benchmark Chooses the Winner*

- [paper-a-minimal-refactor-plan.md](paper-a-minimal-refactor-plan.md) — the plan
  that contracted the broad study into the focused specialization result, and
  specified the auditable pipeline and repository structure this repo implements.
- [paper-a-specialization-frontier-plan.md](paper-a-specialization-frontier-plan.md) —
  the design of the represented-vs-transfer specialization measurement.

## Paper B / broad study (future work)

Planning notes for the earlier broad study and the planned follow-up. The
corresponding code is quarantined under [`../legacy/`](../legacy).

- [paper-b-joint-compliance-stack-plan.md](paper-b-joint-compliance-stack-plan.md) —
  a joint general + domain-specific compliance guard stack.
- [paper-b-development-checkpoint-plan.md](paper-b-development-checkpoint-plan.md) —
  development checkpoints for the follow-up.
- [smollm3-guard-plan.md](smollm3-guard-plan.md) — plan behind the SmolLM3 guard
  reproduction notebook (the broad-study notebooks were removed from the repo;
  their builders remain under [`../legacy/experiments/`](../legacy/experiments)).
- [mortgage-benchmark-hard-results.md](mortgage-benchmark-hard-results.md) — results
  from the hardened mortgage-compliance benchmark.

## Reviews & background

- [review-2026-07-12.md](review-2026-07-12.md) — a technical re-review of the earlier
  broad-study manuscript; the recommendation to narrow to the focused specialization
  result is what this repo now implements.
- [metrics-survey.md](metrics-survey.md) — survey of related-work evaluation metrics
  (background for the evaluation protocol).
