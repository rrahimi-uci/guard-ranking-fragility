"""Candidate generation and preference-pair construction for the two CM-DPO arms.

This is the module the primary contrast rests on.  ``specialist_cm_dpo`` and
``generalist_cm_dpo`` share an objective, a reference, a source-event set, and a
token budget; the *only* thing that differs is which model proposed the candidates.
So everything here is written to keep the two inventories matched on every axis
except provenance, and to make an unmatched pair impossible to emit rather than
merely unlikely.

Three stages:

``propose``
    Run a teacher (a category specialist, or the joint generalist) over alignment
    events and record a complete structured candidate plus calibrated action
    probabilities.  Teachers are always *other-backbone*: a candidate never comes
    from the backbone that will train on it, so the arm cannot self-distill.

``adjudicate``
    Rank the two candidates for an event blind to model identity and presentation
    order.  Confidence-only and formatting-only differences are rejected, because a
    pair that differs in neither action nor substantive field teaches nothing and
    would inflate the inventory.

``build_pairs``
    Emit schema-conforming preference records with full lineage, then verify the two
    source inventories are matched.

Evidence tier: at this tier the adjudicators are LLMs, recorded as such in
``reviewer_ids``.  The protocol's two-qualified-human-reviewer requirement is *not*
met by anything in this module, and the readiness gates stay closed accordingly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import random

from .contracts import ContractError, canonical_json_bytes, canonical_sha256
from .modeling import ACTIONS

CANDIDATE_SOURCES = ("category_specialist", "joint_generalist")
STRATA = ("teacher_agreement", "teacher_disagreement")
# Fields whose difference makes a pair worth training on.  A pair differing only in
# confidence or whitespace is rejected: it carries no preference signal.
SUBSTANTIVE_FIELDS = ("action", "category", "violation_tags", "policy_ids")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def candidate_text(candidate: Mapping) -> str:
    """The exact serialization a pair's log-probability is computed over."""
    body = {
        "category": candidate.get("category"),
        "violation_tags": list(candidate.get("violation_tags") or [])[:6],
        "policy_ids": list(candidate.get("policy_ids") or [])[:6],
    }
    tail = json.dumps(body, separators=(", ", ": "))[1:]
    return f'{candidate["action"]}", ' + tail.lstrip()


def substantive_difference(left: Mapping, right: Mapping) -> list[str]:
    """Which substantive fields actually differ. Empty means the pair is rejectable."""
    differing = []
    for field in SUBSTANTIVE_FIELDS:
        a, b = left.get(field), right.get(field)
        if isinstance(a, list) or isinstance(b, list):
            a, b = sorted(a or []), sorted(b or [])
        if a != b:
            differing.append(field)
    return differing


def make_candidate(*, action: str, category: str, violation_tags: Sequence[str],
                   policy_ids: Sequence[str], probabilities: Mapping[str, float],
                   vote_id: str) -> dict:
    if action not in ACTIONS:
        raise ContractError(f"candidate action must be one of {ACTIONS}")
    if set(probabilities) != set(ACTIONS):
        raise ContractError("candidate probabilities must cover exactly the three actions")
    total = sum(float(probabilities[a]) for a in ACTIONS)
    if not 0.99 <= total <= 1.01:
        raise ContractError("candidate probabilities must sum to one")
    candidate = {
        "action": action,
        "category": category,
        "violation_tags": list(violation_tags),
        "policy_ids": list(policy_ids),
        "probabilities": {a: float(probabilities[a]) for a in ACTIONS},
        "vote_id": vote_id,
    }
    candidate["text"] = candidate_text(candidate)
    candidate["candidate_sha256"] = canonical_sha256(
        {k: candidate[k] for k in SUBSTANTIVE_FIELDS}
    )
    return candidate


def adjudicate(sample: Mapping, left: Mapping, right: Mapping, *, seed: int,
               reviewer_ids: Sequence[str]) -> dict | None:
    """Rank two candidates against adjudicated gold, blind to identity and order.

    Returns None when the pair carries no preference signal, so the caller counts a
    rejection instead of training on noise.
    """
    fields = substantive_difference(left, right)
    if not fields:
        return None
    gold = str((sample.get("gold") or {}).get("action"))
    if gold not in ACTIONS:
        raise ContractError("adjudication requires a gold action on the sample")

    # Presentation order is randomised per event and the ranking is computed from
    # gold alone, so neither model identity nor slot position can influence it.
    rng = random.Random(f"{sample['sample_id']}::{seed}")
    presented = [left, right]
    if rng.random() < 0.5:
        presented.reverse()
    correct = [c for c in presented if c["action"] == gold]
    wrong = [c for c in presented if c["action"] != gold]
    if len(correct) == 1 and len(wrong) == 1:
        chosen, rejected, why = correct[0], wrong[0], "chosen matches adjudicated gold action"
    elif len(correct) == 2:
        # Both right on the action: prefer the better-evidenced structured answer.
        gold_ids = set((sample.get("gold") or {}).get("policy_ids") or [])
        scored = sorted(
            presented,
            key=lambda c: (len(gold_ids & set(c.get("policy_ids") or [])),
                           -len(c.get("violation_tags") or [])),
            reverse=True,
        )
        if substantive_difference(scored[0], scored[1]) == []:
            return None
        chosen, rejected = scored[0], scored[1]
        why = "both actions correct; chosen cites more of the gold policy authorities"
    else:
        return None  # both wrong on the action: no defensible preference
    return {
        "chosen": chosen,
        "rejected": rejected,
        "substantive_difference_fields": fields,
        "adjudication_rationale": why,
        "reviewer_ids": list(reviewer_ids),
        "presentation_order_sha256": _sha(
            json.dumps([c["vote_id"] for c in presented]).encode("utf-8")
        ),
    }


def build_pairs(samples: Sequence[Mapping], candidates: Mapping[str, Mapping[str, Mapping]],
                *, candidate_source: str, target_backbone_key: str,
                teacher_backbone_keys: Sequence[str], teacher_seeds: Sequence[int],
                teacher_cells: Sequence[str], calibration_lock_id: str,
                adjudicator_id: str, reviewer_ids: Sequence[str],
                seed: int = 20260725) -> dict:
    """Build one source's preference inventory.

    ``candidates[sample_id]`` maps a slot name to a candidate; exactly two slots are
    required so both sources contribute the same number of proposals per event.
    """
    if candidate_source not in CANDIDATE_SOURCES:
        raise ContractError(f"candidate_source must be one of {CANDIDATE_SOURCES}")
    records, rejected_counts = [], {"no_substantive_difference": 0, "both_actions_wrong": 0,
                                    "missing_candidates": 0}
    for sample in samples:
        slots = candidates.get(sample["sample_id"])
        if not slots or len(slots) != 2:
            rejected_counts["missing_candidates"] += 1
            continue
        left, right = [slots[k] for k in sorted(slots)]
        verdict = adjudicate(sample, left, right, seed=seed, reviewer_ids=reviewer_ids)
        if verdict is None:
            gold = str((sample.get("gold") or {}).get("action"))
            key = ("both_actions_wrong"
                   if left["action"] != gold and right["action"] != gold
                   else "no_substantive_difference")
            rejected_counts[key] += 1
            continue
        chosen, rejected = verdict["chosen"], verdict["rejected"]
        policy = sample.get("policy_context")
        stratum = ("teacher_agreement" if left["action"] == right["action"]
                   else "teacher_disagreement")
        records.append({
            "preference_id": f"{candidate_source}::{sample['sample_id']}",
            "sample_id": sample["sample_id"],
            "family_id": sample["family_id"],
            "category": sample["category"],
            "domain": sample["domain"],
            "policy_context": policy,
            "policy_context_sha256": canonical_sha256(policy) if policy else canonical_sha256({}),
            "candidate_source": candidate_source,
            "source_stratum": stratum,
            "target_backbone_key": target_backbone_key,
            "teacher_backbone_keys": list(teacher_backbone_keys),
            "teacher_seeds": list(teacher_seeds),
            "teacher_cells": list(teacher_cells),
            "chosen": chosen,
            "rejected": rejected,
            "chosen_vote_id": chosen["vote_id"],
            "rejected_vote_id": rejected["vote_id"],
            "chosen_candidate_sha256": chosen["candidate_sha256"],
            "rejected_candidate_sha256": rejected["candidate_sha256"],
            "calibration_lock_id": calibration_lock_id,
            "aggregation_id": verdict["presentation_order_sha256"],
            "gold_action": str((sample.get("gold") or {}).get("action")),
            "adjudicated_gold": {
                "action": str((sample.get("gold") or {}).get("action")),
                "category": sample["category"],
                "policy_ids": list((sample.get("gold") or {}).get("policy_ids") or []),
            },
            "substantive_difference_fields": verdict["substantive_difference_fields"],
            "reviewer_ids": verdict["reviewer_ids"],
            "adjudicator_id": adjudicator_id,
            "adjudication_rationale": verdict["adjudication_rationale"],
            "adjudication_status": "resolved",
            "model_identities_hidden": True,
            "candidate_order_randomized": True,
            "policy_ids": list((sample.get("gold") or {}).get("policy_ids") or []),
        })
    return {"records": records, "rejected": rejected_counts}


def assert_sources_matched(left: Sequence[Mapping], right: Sequence[Mapping]) -> dict:
    """Refuse two inventories that differ in anything but candidate provenance.

    The primary contrast is only interpretable if the two CM-DPO arms saw the same
    events in the same quotas.  This is the data-side twin of the run-side check in
    :mod:`comparisons`.
    """
    if not left or not right:
        raise ContractError("both candidate-source inventories must be non-empty")
    sources = {r["candidate_source"] for r in left} | {r["candidate_source"] for r in right}
    if sources != set(CANDIDATE_SOURCES):
        raise ContractError(f"expected exactly the two candidate sources, got {sorted(sources)}")
    left_events = {r["sample_id"] for r in left}
    right_events = {r["sample_id"] for r in right}
    if left_events != right_events:
        only_left = sorted(left_events - right_events)[:3]
        only_right = sorted(right_events - left_events)[:3]
        raise ContractError(
            f"CM-DPO source inventories cover different events; "
            f"only-left={only_left} only-right={only_right}"
        )
    if len(left) != len(right):
        raise ContractError(f"pair counts differ: {len(left)} vs {len(right)}")

    def quota(records):
        out: dict[str, int] = {}
        for record in records:
            key = f"{record['category']}::{record['gold_action']}"
            out[key] = out.get(key, 0) + 1
        return out

    if quota(left) != quota(right):
        raise ContractError("category/action quotas differ between candidate sources")
    return {
        "matched_events": len(left_events),
        "pairs_per_source": len(left),
        "quota_sha256": canonical_sha256(quota(left)),
        "strata": {
            source: {
                stratum: sum(1 for r in records if r["source_stratum"] == stratum)
                for stratum in STRATA
            }
            for source, records in (("left", left), ("right", right))
        },
    }
