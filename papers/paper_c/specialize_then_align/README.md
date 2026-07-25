# Paper C v2: Specialize, Then Align

This directory contains the active redesign of Paper C. It is a scientifically
new study, not an amendment to the stopped reference-centering experiment in the
parent directory.

Working title:

> **Specialize, Then Align? A Controlled Study of Cross-Model Preference
> Learning for Safety and US Mortgage-Risk Guardrails**

## Research question

Can category-specialized small guard models provide useful preference signal to
a single deployable student without sacrificing benign specificity or
worst-category generality?

The study crosses the same two pinned SLM backbones with five core categories:
three general-safety categories, `mortgage_fair_lending`, and the deliberately
narrow `mortgage_closed_end_advertising` category.
Every primary event has one adjudicated **focal category**. That category is
annotation and evaluation metadata, not an oracle category supplied to the
deployed student. Cross-category compositions are reported separately and make
no confirmatory claim.

The deployed action space is `ALLOW`, `REVIEW`, or `INTERVENE`. `ABSTAIN` is a
specialist eligibility state, and `NO_CONSENSUS` is an aggregation status;
neither is a fourth action or a gold-label substitute. `REVIEW` is reserved for
events whose decisive facts, coverage, or governing policy are genuinely
unresolved.

Mortgage events include a request, proposed response or action, structured
context, US-federal jurisdiction, policy date, and an immutable
`policy_context`. That object binds the policy snapshot ID and canonical object
hash, the policy-vintage lock, the exact authority IDs and text applied, and the
text hash. They screen for policy risk; they do not approve or deny credit,
provide legal advice, or certify compliance.

## Five matched alignment arms

All student arms start from the same backbone/seed-specific joint multitask SFT
reference:

1. `gold_sft` — continued training on adjudicated structured gold;
2. `soft_distill` — distillation of the three calibrated specialist action
   probabilities only, not tags, policy IDs, or free-text rationales;
3. `specialist_pairce` — uncentered PairCE on specialist-origin pairs;
4. `generalist_cm_dpo` — CM-DPO on pairs proposed by the joint generalist;
5. `specialist_cm_dpo` — the same CM-DPO objective on specialist-origin pairs.

The primary contrast is
`specialist_cm_dpo - generalist_cm_dpo`. Because the objective, source events,
reference, optimizer, and accounting are matched, this contrast targets the
incremental value of category-specialist candidate generation rather than DPO
itself. Reference centering, pairwise learning, and the total method effect are
secondary contrasts. A per-backbone/seed comparison manifest fails closed if
the objective hash, reference, opportunity events, retained-pair event IDs,
category/action/stratum quotas, gold anchor, replay, optimizer, serialization,
review protocol, checkpoint ladder, pair count, or token budget differs between
the two CM-DPO arms.

Pair log probabilities are the sum of response-token log probabilities with all
prompt tokens masked. A pair is rejected if either candidate is truncated.

## Frozen run accounting

| Panel | References | Specialists | Aligned students | Total cells |
|---|---:|---:|---:|---:|
| Disjoint pilot: seeds 7 and 8 | 4 | 20 | 20 | 44 |
| Primary: seeds 42, 43, and 44 | 6 | 30 | 30 | 66 |
| Entire planned program | 10 | 50 | 50 | 110 |

Pilot families, seeds, preferences, checkpoints, and outputs are disjoint from
the primary and sealed namespaces. None of the 44 pilot cells is reused in the
66-cell primary panel. The pilot is completed first; only nuisance estimates,
feasibility results, and runtime accounting may inform the final prospective
freeze.

## Data and evaluation boundaries

- Balanced `ALLOW / REVIEW / INTERVENE` triplets are construct and stress
  material, not an operational-prevalence sample.
- The primary design requires an independent calibration stream and an
  independently authored sealed stream with at least 2,000 `ALLOW` examples per
  core category in each.
- The prespecified capacity-evaluation mixture is 94% `ALLOW`, 5% `REVIEW`, and
  1% `INTERVENE`; it is not an estimate of deployment prevalence. The 10%
  review budget applies only to this mixture, never to the balanced triplets.
- Temperature calibration and the two action thresholds are fit only on the
  calibration split. Checkpoints are selected on a separate
  `checkpoint_selection` split. The selected checkpoint is scored once on the
  sealed cohort.
- Policy-date routing is opt-in: only rows with
  `temporal_evaluation_eligible=true` are compared with the 2026-07-20 cutoff.
  Such rows declare the derived `temporal_policy_side` as `pre_cutoff` or
  `post_cutoff`. Ordinary current-vintage mortgage rows remain eligible for the
  four family splits.
- The candidate vintage inventory currently contains only an unsigned
  post-cutoff snapshot. A pre-cutoff row therefore fails validation, and no
  policy-time result is claimable until both sides and their authority archives
  are SME-signed. Within each archive, every excerpt hash is bound to the exact
  authority IDs and archived source-byte hashes it supports; cross-authority
  hash matching fails closed.
- Inference resamples scenario families only and is conditional on the two named
  backbones and three named primary seeds. Across-seed dispersion is descriptive;
  no architecture- or seed-population claim is permitted.

## Folder contract

Everything for this redesign stays here. It does not import or write through the
legacy `paper_c` package.

- `config/`: study, smoke, taxonomy, and time-stamped policy candidates.
- `schemas/`: sample, policy, authority-archive, preference, specialist,
  consensus, cohort, matched-comparison, and readiness-evidence contracts.
- `src/paper_c_sta/`: executable design contracts and objectives.
- `tests/`: CPU-only scientific contract tests.
- `provenance/`: explicit relationship to the stopped v1 study.
- `manuscript/`: the unrun paper draft and bibliography.
- `locks/`: tracked immutable protocol and run-lock manifests.
- `artifacts/`, `inputs/`, `build/`: ignored local outputs only.

## Current evidence boundary

The prospective design is specified, but the study is unrun. Existing mortgage
scenarios are candidate-development material, never legal gold. The mortgage
policy pack and the general-safety policy rubric remain unsigned candidates.
No predecessor smoke result is evidence for this study.

Readiness uses exactly five gates: mortgage-policy SME sign-off, annotation-rubric
sign-off, licence ledger, completed power pilot, and sealed cohorts. A gate can
be true only when its matching `readiness_evidence` record contains a
workspace-relative `artifact_path`, exact `artifact_sha256`, `lock_id`, issuance
time, and approver identities, and the file resolves and hashes correctly. A
boolean without evidence fails closed.

The candidate lock also freezes an ordered pilot-to-primary chain: pilot
data/policy, pilot calibration, pilot preferences, the post-pilot prospective
primary-protocol freeze, distinct primary data/policy and preference locks,
aligned-candidate and checkpoint-selection locks, then sealed confirmation.

## Local checks

```bash
make test
make validate
make readiness
make paper
```

`make readiness` intentionally exits nonzero until all five evidence-bound gates
validate. No primary GPU panel is authorized before the prospective freeze.
