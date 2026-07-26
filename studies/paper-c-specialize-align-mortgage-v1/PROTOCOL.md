# Normative protocol candidate

## 1. Claim, event, and focal category

Paper C tests whether category specialists add transferable guard behavior beyond
what a joint generalist can provide under the same CM-DPO objective. The unit is
a structured guard event:

```text
(request, proposed_response_or_action, context, jurisdiction,
 policy_as_of, policy_context)
    -> (ALLOW | REVIEW | INTERVENE, focal_category, tags, policy_ids, confidence)
```

Each primary event has exactly one independently adjudicated focal category.
The stored `category` field denotes that focal category. It is supervision and
stratification metadata, not an oracle input to the deployed student. Events with
materially overlapping categories are excluded from the confirmatory panel and
reported only as a separate composition stress test.

General-safety request rows carry null `policy_context`, jurisdiction, and policy
date; their signed policy rubric is bound at the manifest level. Every mortgage
row instead carries immutable `policy_context` containing `snapshot_id`,
`snapshot_object_sha256`, `policy_vintage_lock_id`, `policy_as_of`,
`authority_ids`, the exact `policy_text` used for adjudication, and its
`content_sha256`. The snapshot-object digest binds the complete policy snapshot;
the text digest binds the row-visible excerpt. Each cited authority must cover
the focal category and be effective on `policy_as_of` (and not expired by that
date). Mortgage rows additionally require a proposed response or action,
US-federal jurisdiction, and structured `actor_role`, `product`,
`transaction_stage`, `applicable_regime`, and `coverage_facts`. The guard neither
approves credit nor certifies legality.

## 2. Actions, abstention, and consensus

The only deployable and gold actions are:

- `ALLOW`: the proposed response or action may proceed without guard
  intervention under the focal policy;
- `REVIEW`: decisive facts, authority coverage, policy interaction, or intent is
  genuinely unresolved and requires qualified human review;
- `INTERVENE`: the proposed response or action should be refused, constrained,
  or replaced with a safe alternative.

`ABSTAIN` is a specialist eligibility state. An out-of-expertise or otherwise
ineligible specialist emits no semantic action candidate and cannot vote
`ALLOW`, `REVIEW`, or `INTERVENE`. `NO_CONSENSUS` is an aggregation status. It
preserves every full eligible structured teacher candidate together with its
immutable vote, hash, calibration, backbone, and seed lineage for adjudication.
It must never be converted into a synthetic `REVIEW` target or probability
vector. Neither state can become gold without independent adjudication.

## 3. Category crossover and policy scope

Every core category is trained on both pinned backbones and every seed in its
panel. No category is assigned to a unique architecture.

General-safety core:

1. toxicity and abusive content;
2. jailbreak and policy evasion;
3. prompt injection and data exfiltration.

US-federal mortgage core:

4. `mortgage_fair_lending`: fair lending and equal access;
5. `mortgage_closed_end_advertising`: closed-end dwelling-secured credit
   advertising.

Origination disclosures, underwriting/adverse action, steering/referrals,
servicing/loss mitigation, valuation/appraisal, privacy/reporting, and
servicemember protections are excluded from specialist training. They are
prespecified transfer probes only after their authority and coverage records are
SME-signed. Mortgage outcomes are always disaggregated; no pooled result can hide
a failed mortgage category.

## 4. Data, labels, and disjoint cohorts

Each construct family begins with a minimally changed triplet:

- an `INTERVENE` request or proposed action;
- a compliant near-neighbor that should be `ALLOW`ed;
- a fact-dependent boundary case whose correct semantic action is `REVIEW`.

Triplets are construct-balanced stress material. They do not estimate operational
prevalence, and the capacity-mixture review budget does not apply to them. The initial
mortgage construction target is 400 families and 1,200 rows; the final
confirmatory family count is frozen only after the disjoint pilot.

Primary family assignment is 50% `specialist_train`, 20% `alignment_pool`, 15%
`calibration`, and 15% `checkpoint_selection`. A separately authored sealed
cohort is never part of that allocation. The calibration and sealed cohorts each
contain at least 2,000 independent `ALLOW` examples per core category. Their
families, templates, sources, and policy vintages must be disjoint from training,
alignment, checkpoint selection, and each other.

Every row stores both `family_id` and `content_family_id`. Isolation units are
the **transitive closure of `family_id` ↔ `content_family_id`**: a family and a
content family that share any row are one unit, and a unit is assigned to exactly
one split. Split validation fails if any ordinary family or semantic content
family crosses a split boundary.

**Source-level disjointness is not claimed on the reuse corpus, because it is not
achievable there.** An earlier version of this section additionally bound grouping
by `provenance.source_id` and failed validation on any source crossing a split.
Unioning on `source_id` over the reuse corpus yields 9 isolation units across 9
datasets — fewer units than the four splits need populated at the prescribed 50 /
20 / 15 / 15 proportions, so no assignment satisfies it and the constraint is
infeasible rather than merely strict. Two consequences, stated rather than
buried: this study **cannot** support a held-out-*source* transfer claim, and
`splits.validate_split_isolation()` still checks all three levels but is reachable
only from the test suite — it is not wired into the pipeline, so on the reuse
corpus it is a specification of intent, not an operating control. A study that
needs held-out-source transfer must author enough independent sources to make the
constraint satisfiable first.

Temporal routing is explicitly opt-in. Only mortgage rows carrying
`temporal_evaluation_eligible=true` are compared against the configured
2026-07-20 cutoff and may enter `temporal_test`. A current-vintage policy date by
itself does not make a row temporal; ordinary rows with the flag set to `false`
remain eligible for the four family splits. Temporal and ordinary namespaces are
family-, content-family-, and source-disjoint. Each temporal row declares the
cutoff-derived `temporal_policy_side` as `pre_cutoff` or `post_cutoff`; a mismatch
fails closed.

The prespecified capacity-evaluation mixture is 94% `ALLOW`, 5% `REVIEW`, and
1% `INTERVENE`; it is not a measured deployment-prevalence estimate. This
mixture alone is subject to the 10% maximum review budget. Protected-class
near-neighbors and benign low-prevalence streams are mandatory specificity
gates.

All claim-bearing mortgage rows require two qualified, independent reviewers and
a separate adjudicator. Each stores decisive coverage facts, policy snapshot and
authority IDs, effective and retrieval dates, reviewer identities, rationale,
and source lineage. Legacy benchmark flags, policy-card targets, and model votes
are candidate metadata only.

## 5. Specialists, calibration, and candidate sources

The primary specialist grid is:

```text
5 categories x 2 backbones x 3 primary seeds = 30 specialists
```

Each specialist and each joint-generalist candidate generator receives its own
category-wise temperature map fitted only on family-disjoint calibration data.
Both source inventories bind the calibration ID and lock. Confidence is derived
from that calibrated action distribution; it is not a freely generated number.
Out-of-expertise specialist behavior is `ABSTAIN`.

For a target backbone, specialist candidates come only from the other backbone.
At least two distinct teacher seeds must pass the frozen expertise and confidence
gates. Agreement produces `CANDIDATE_CONSENSUS`; insufficient teachers or action
disagreement produces `NO_CONSENSUS`, with all eligible individual candidates
retained. Both agreement and disagreement strata are sent to blinded human
adjudication.

The matched source control uses complete candidates from the other-backbone joint
generalist on the same source events. Reviewers are blinded to source identity,
backbone identity, candidate order, and study arm.

## 6. Preference construction and objective

Reviewers choose the more policy-faithful complete output and record the decisive
difference. A retained pair must be consistent with adjudicated gold and differ
substantively in action, tags, policy grounding, or review behavior. A difference
only in confidence, formatting, model identity, or candidate order is invalid.
Each preference binds its calibration lock and source aggregation, the chosen
and rejected vote IDs, canonical candidate hashes, per-teacher backbone/seed/
calibration/source lineage, the blinded randomized review packet, and the
independently adjudicated reference label. Mortgage preferences also bind the
same immutable seven-field policy context as their source event plus the
canonical hash of that context object.

For chosen output `y+`, rejected output `y-`, and frozen joint-SFT reference
`pi_0`, CM-DPO uses:

```text
l_i = softplus(-beta * [
        log pi(y+|x) - log pi(y-|x)
      - log pi_0(y+|x) + log pi_0(y-|x)
])
```

At exact policy/reference equality the unweighted pair loss is `log(2)`. Pair
weighting can change contribution magnitude but never reverse or replace human
adjudication. The frozen `pair_logprob_reduction` is
`sum_response_token_logprob_with_prompt_masked`: prompt tokens contribute no
log probability, and response-token log probabilities are summed, not averaged.
The frozen `candidate_length_rule` is
`reject_pair_if_either_candidate_is_truncated`. Sequence serialization,
masking, length handling, and log-probability reduction are identical between
the two CM-DPO source arms.

## 7. Five matched student arms and estimands

Every backbone/seed starts from the same joint multitask SFT reference. The five
arms are:

```text
gold_sft             continued adjudicated-gold SFT
soft_distill         three calibrated specialist action probabilities only
specialist_pairce    uncentered PairCE on specialist-origin pairs
generalist_cm_dpo    reference-centered DPO on joint-generalist-origin pairs
specialist_cm_dpo    reference-centered DPO on specialist-origin pairs
```

All arms receive the same source-event identities, adjudicated gold anchor,
retention replay, optimizer family, checkpoint ladder, and token accounting
where their objective permits. The two CM-DPO arms additionally match objective,
reference, pair-construction budget, serialization, and reviewer protocol.
Only PairCE and specialist CM-DPO share identical specialist-origin pairs.
Soft distillation transfers only the calibrated probabilities for `ALLOW`,
`REVIEW`, and `INTERVENE`; it does not distill tags, policy IDs, rationale text,
or hidden teacher token logits.

For every backbone/seed, the comparison manifest binds the exact CM-DPO
objective hash (including beta, category-DRO temperature, anchor/replay weights,
log-probability reduction, and truncation rule). It also requires identical
opportunity-event IDs, retained-pair event IDs, category/action/agreement-stratum
quota manifest, pair count, token budget, reference, optimizer, checkpoint
ladder, serialization, gold anchor, replay, and review protocol. Candidate-source
identity and candidate-content inventory are the only intended differences.

The primary contrast is:

```text
C_specialist_source = specialist_cm_dpo - generalist_cm_dpo
```

It estimates the incremental value of specialist-origin rather than
joint-generalist-origin candidates under the same reference-centered objective.
Prespecified secondary contrasts are:

```text
C_centering = specialist_cm_dpo - specialist_pairce
C_pair      = specialist_pairce - soft_distill
C_total     = specialist_cm_dpo - gold_sft
```

All non-SFT objectives use the same category-DRO, adjudicated-gold, and retention
components:

```text
L = tau * logmeanexp(L_category / tau)
  + lambda_gold * L_adjudicated_gold
  + lambda_retain * KL(pi_reference || pi_student)
```

## 8. Pilot, freeze, and run accounting

The pilot is a separate experiment using seeds 7 and 8 and a family namespace
disjoint from primary and sealed data. Its order is:

1. build four joint references and 20 specialists;
2. calibrate specialists and create both candidate-source inventories;
3. complete blinded preference adjudication;
4. train 20 pilot students across the five arms;
5. estimate paired family covariance, threshold feasibility, calibration, and
   runtime; the two pilot seeds provide only a descriptive stability check;
6. freeze final family counts, margins, thresholds, accounting, and the primary
   protocol.

No primary cell is run before that freeze, and no pilot family, seed,
preference, checkpoint, or model is reused in the primary panel.

```text
pilot:   4 references + 20 specialists + 20 students = 44 cells
primary: 6 references + 30 specialists + 30 students = 66 cells
total:                                                  110 cells
```

Learned routers are reported as separately accounted inference baselines and are
not silently included in these alignment-cell totals. The oracle focal-category
router is an upper bound, never a deployable result.

## 9. Calibration, checkpoint selection, and sealed scoring

The three-way action rule uses two thresholds in fixed priority order:

```text
if p(INTERVENE) >= t_i:
    INTERVENE
else if 1 - p(ALLOW) >= t_r:
    REVIEW
else:
    ALLOW
```

Temperature parameters, `t_i`, and `t_r` are fitted only on the calibration
split over the benign-FPR grid `{1%, 2%, 3%, 4%, 5%}`. A checkpoint is then
selected on the separate `checkpoint_selection` split by maximizing the frozen
worst-category frontier subject to capacity-mixture FPR and review constraints; ties
select the earliest checkpoint. Neither calibration nor checkpoint selection
may inspect sealed labels. The selected checkpoint is scored once on the sealed
cohort.

## 10. Outcomes, decision rule, and inference

The primary outcome is the worst-core-category specificity-controlled frontier
for `C_specialist_source` on the prespecified capacity-evaluation mixture.
`REVIEW` does not silently count as `INTERVENE`: report false allow, false
intervention, review rate, action confusion, and risk-coverage separately.
Report per-category AP,
calibration, protected-pair invariance, held-out-family transfer, router
regret, and mortgage policy-time transfer. Held-out-*source* transfer is
explicitly out of scope: the reuse corpus cannot be split source-disjointly (see
the isolation section), so the quantity is not estimable here.

The two pinned backbones and three primary seeds are a fixed named panel.
Inference resamples scenario families only while pairing arm comparisons within
backbone, seed, source event, and checkpoint rule. Across-seed dispersion is
descriptive. Claims are conditional on these backbones and seeds; the study
cannot claim architecture- or seed-population generality. Primary intervals are
simultaneous and one-sided; secondary contrasts use Holm correction.

The pilot freezes final family counts. Planning targets are a 0.02 primary
worst-category-frontier effect, one-sided alpha 0.025, and 80% power, with a
separately oriented 0.02 worst-category noninferiority margin. Metric-specific
signs and tolerances for FPR and review rate must be frozen before primary
training.

## 11. Mortgage boundary and time

The candidate policy pack is US federal and was retrieved 2026-07-25. It contains
authority records for both core and all seven held-out mortgage categories, but
none becomes gold authority until source bytes, coverage, and dates are archived
and SME-signed. Live-file decisions, state law, licensing/usury,
investor/GSE/FHA/VA/USDA overlays, disputed coverage, and incomplete deadline
calculations default to semantic `REVIEW` because facts or authority are
unresolved—not because a specialist abstained.

The current candidate vintage inventory intentionally registers only the
unsigned post-2026-07-21 snapshot; `pre_cutoff` coverage is explicitly
`missing`. Runtime validation rejects any row whose snapshot hash, vintage-lock
ID, or date interval is absent from that inventory. Mortgage validation is
claim-bearing by default: it additionally requires a complete SME-signed
two-sided inventory and verifies the row excerpt hash and authority IDs against
an archived-authority manifest. Each authorized-excerpt record binds one unique
excerpt hash to the exact authority IDs whose archived bytes support it; matching
an excerpt hash and an unrelated authority elsewhere in the same manifest is
invalid. Candidate development must explicitly opt out of claim-bearing
validation and cannot populate evaluation evidence.

The Regulation B amendment effective July 21, 2026 rejects ECOA effects-test
liability while the current Fair Housing Act rule retains discriminatory-effects
liability. Statistical-disparity cases are a separately locked temporal and
coverage-interaction stratum and default to `REVIEW` absent a fully adjudicated
coverage record.

## 12. Readiness and lock chain

Readiness has exactly five gates:

1. `mortgage_policy_sme_signed`;
2. `annotation_rubric_signed`;
3. `licence_ledger_complete`;
4. `power_pilot_complete`;
5. `sealed_cohorts_created`.

Every true gate requires a matching non-null `readiness_evidence` object with a
workspace-relative `artifact_path`, 64-character `artifact_sha256`, nonempty
`lock_id`, `issued_utc`, and qualified `approver_ids`. The artifact must exist and
its bytes must match. Unknown, missing, null, unhashed, mismatched, or extra gate
names fail closed.

The lock chain is:

1. candidate protocol/taxonomy lock, authorizing no training;
2. signed data-and-policy pilot lock;
3. pilot specialist/calibration and adjudicated-preference locks;
4. post-pilot prospective primary-protocol lock;
5. primary data/policy and preference locks;
6. aligned-candidate and checkpoint-selection locks;
7. one-time sealed-confirmation lock.

The candidate lock machine-encodes the child sequence as
`pilot_data_policy_lock`, `pilot_specialist_calibration_lock`,
`pilot_adjudicated_preference_lock`,
`post_pilot_prospective_primary_protocol_lock`, `primary_data_policy_lock`,
`primary_adjudicated_preference_lock`, `primary_aligned_candidate_lock`,
`primary_checkpoint_selection_lock`, and `sealed_confirmation_lock`. Thus no
pilot data or preference artifact can masquerade as its primary counterpart, and
the prospective primary freeze must precede every primary artifact.

No predecessor lock is a parent or superseded lock for this study.
