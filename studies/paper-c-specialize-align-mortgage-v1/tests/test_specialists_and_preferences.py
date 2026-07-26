from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from paper_c_sta.contracts import ContractError, canonical_sha256
from paper_c_sta.preferences import locked_pair_weight, validate_preference
from paper_c_sta.specialists import SpecialistVote, aggregate_specialists


def _vote(*, backbone: str, seed: int, category: str = "toxicity_abuse", action: str = "intervene"):
    probabilities = {
        "allow": 0.05,
        "review": 0.05,
        "intervene": 0.90,
    }
    if action == "allow":
        probabilities = {"allow": 0.90, "review": 0.05, "intervene": 0.05}
    candidate = {
        "action": action,
        "category": category,
        "violation_tags": ["abuse"],
        "policy_ids": [],
        "confidence": 0.9,
        "confidence_source": "calibrated_action_distribution",
    }
    return {
        "vote_id": f"vote-{backbone}-{seed}",
        "sample_id": "s-1",
        "family_id": "family-s-1",
        "specialist_category": category,
        "backbone_key": backbone,
        "seed": seed,
        "target_backbone_key": "qwen25_15b",
        "qualified": True,
        "abstain": False,
        "probabilities": probabilities,
        "calibration_id": f"cal-{backbone}-{seed}",
        "candidate": candidate,
        "candidate_sha256": canonical_sha256(candidate),
    }


def _calibration_inventory(*votes, qualified=True):
    return {
        "lock_id": "calibration-lock-1",
        "entries": [
            {
                "calibration_id": vote["calibration_id"],
                "specialist_category": vote["specialist_category"],
                "backbone_key": vote["backbone_key"],
                "seed": vote["seed"],
                "qualified": qualified,
            }
            for vote in votes
        ],
    }


def test_leave_one_backbone_out_excludes_self_teacher():
    votes = [
        _vote(backbone="qwen25_15b", seed=42),
        _vote(backbone="smollm2_17b", seed=42),
        _vote(backbone="smollm2_17b", seed=43),
    ]
    result = aggregate_specialists(
        votes,
        target_category="toxicity_abuse",
        target_backbone="qwen25_15b",
        minimum_distinct_seeds=2,
        minimum_confidence=0.7,
        qualified_calibration_inventory=_calibration_inventory(*votes),
    )
    assert result["status"] == "candidate_consensus"
    assert result["candidate_action"] == "intervene"
    assert result["excluded"]["self_backbone"] == 1
    assert result["teacher_backbones"] == ["smollm2_17b"]
    assert result["is_gold"] is False
    assert result["requires_adjudication"] is True
    assert result["calibration_lock_id"] == "calibration-lock-1"
    assert len(result["eligible_candidates"]) == 2
    assert len(result["aggregation_id"]) == 64


def test_disagreement_abstains_and_never_becomes_gold():
    votes = [
        _vote(backbone="smollm2_17b", seed=42, action="allow"),
        _vote(backbone="smollm2_17b", seed=43, action="intervene"),
    ]
    result = aggregate_specialists(
        votes,
        target_category="toxicity_abuse",
        target_backbone="qwen25_15b",
        minimum_distinct_seeds=2,
        minimum_confidence=0.7,
        qualified_calibration_inventory=_calibration_inventory(*votes),
    )
    assert result["status"] == "no_consensus"
    assert result["reason"] == "teacher_disagreement"
    assert result["routing_action"] == "abstain"
    assert result["candidate_action"] is None
    assert result["probabilities"] is None
    assert result["is_gold"] is False


def test_aggregation_rejects_cross_sample_and_duplicate_teacher_cells():
    first = _vote(backbone="smollm2_17b", seed=42)
    second = _vote(backbone="smollm2_17b", seed=43)
    second["sample_id"] = "different-sample"
    with pytest.raises(ContractError, match="mix multiple samples"):
        aggregate_specialists(
            [first, second],
            target_category="toxicity_abuse",
            target_backbone="qwen25_15b",
            minimum_distinct_seeds=2,
            minimum_confidence=0.7,
            qualified_calibration_inventory=_calibration_inventory(first, second),
        )

    duplicate = deepcopy(first)
    with pytest.raises(ContractError, match="duplicate specialist teacher cell"):
        aggregate_specialists(
            [first, duplicate],
            target_category="toxicity_abuse",
            target_backbone="qwen25_15b",
            minimum_distinct_seeds=1,
            minimum_confidence=0.7,
            qualified_calibration_inventory=_calibration_inventory(first),
        )


def test_specialist_vote_requires_strict_identity_and_candidate_consistency():
    bad_action = _vote(backbone="smollm2_17b", seed=42)
    bad_action["candidate"]["action"] = "allow"
    with pytest.raises(ContractError, match="top action"):
        SpecialistVote.from_mapping(bad_action)

    bad_confidence = _vote(backbone="smollm2_17b", seed=42)
    bad_confidence["candidate"]["confidence"] = 0.8
    with pytest.raises(ContractError, match="calibrated probability"):
        SpecialistVote.from_mapping(bad_confidence)

    missing_calibration = _vote(backbone="smollm2_17b", seed=42)
    missing_calibration["calibration_id"] = ""
    with pytest.raises(ContractError, match="calibration_id"):
        SpecialistVote.from_mapping(missing_calibration)

    unqualified_nonabstaining = _vote(backbone="smollm2_17b", seed=42)
    unqualified_nonabstaining["qualified"] = False
    with pytest.raises(ContractError, match="must abstain"):
        SpecialistVote.from_mapping(unqualified_nonabstaining)

    valid_abstention = _vote(backbone="smollm2_17b", seed=42)
    valid_abstention.update({
        "qualified": False,
        "abstain": True,
        "candidate": None,
        "candidate_sha256": None,
    })
    assert SpecialistVote.from_mapping(valid_abstention).candidate is None

    invalid_abstention = _vote(backbone="smollm2_17b", seed=42)
    invalid_abstention["abstain"] = True
    with pytest.raises(ContractError, match="cannot carry a candidate"):
        SpecialistVote.from_mapping(invalid_abstention)


def test_tied_specialist_probabilities_fail_closed():
    vote = _vote(backbone="smollm2_17b", seed=42)
    vote["probabilities"] = {"allow": 0.5, "review": 0.5, "intervene": 0.0}
    vote["candidate"]["action"] = "review"
    vote["candidate"]["confidence"] = 0.5
    with pytest.raises(ContractError, match="unique top action"):
        SpecialistVote.from_mapping(vote)


def test_locked_calibration_inventory_is_authoritative_and_lineage_is_bound():
    votes = [
        _vote(backbone="smollm2_17b", seed=42),
        _vote(backbone="smollm2_17b", seed=43),
    ]
    result = aggregate_specialists(
        votes,
        target_category="toxicity_abuse",
        target_backbone="qwen25_15b",
        minimum_distinct_seeds=2,
        minimum_confidence=0.7,
        qualified_calibration_inventory=_calibration_inventory(
            *votes, qualified=False
        ),
    )
    assert result["status"] == "no_consensus"
    assert result["reason"] == "insufficient_teachers"
    assert result["excluded"]["locked_unqualified"] == 2

    tampered = deepcopy(votes[0])
    tampered["candidate"]["violation_tags"] = ["different"]
    with pytest.raises(ContractError, match="candidate hash"):
        SpecialistVote.from_mapping(tampered)


def _preference():
    chosen = {
        "action": "intervene",
        "category": "mortgage_closed_end_advertising",
        "violation_tags": ["misleading_claim"],
        "policy_ids": ["REG_Z_1026_24"],
        "confidence": 0.9,
        "confidence_source": "calibrated_action_distribution",
    }
    rejected = {
        "action": "allow",
        "category": "mortgage_closed_end_advertising",
        "violation_tags": [],
        "policy_ids": [],
        "confidence": 0.6,
        "confidence_source": "calibrated_action_distribution",
    }
    chosen_hash = canonical_sha256(chosen)
    rejected_hash = canonical_sha256(rejected)
    policy_text = "12 CFR 1026.24 governs closed-end credit advertising."
    policy_context = {
        "snapshot_id": "us_federal_mortgage_risk_triage_2026_07_25_candidate",
        "snapshot_object_sha256": "b" * 64,
        "policy_vintage_lock_id": "candidate_post_2026_07_21_vintage_unapproved",
        "policy_as_of": "2026-07-25",
        "authority_ids": ["REG_Z_1026_24"],
        "policy_text": policy_text,
        "content_sha256": hashlib.sha256(policy_text.encode("utf-8")).hexdigest(),
    }
    return {
        "preference_id": "p-1",
        "sample_id": "m-1",
        "family_id": "mf-1",
        "category": "mortgage_closed_end_advertising",
        "domain": "mortgage",
        "policy_context": policy_context,
        "policy_context_sha256": canonical_sha256(policy_context),
        "candidate_source": "category_specialist",
        "source_stratum": "teacher_disagreement",
        "target_backbone_key": "qwen25_15b",
        "teacher_backbone_keys": ["smollm2_17b"],
        "teacher_seeds": [42, 43],
        "chosen": chosen,
        "rejected": rejected,
        "chosen_vote_id": "vote-smol-42",
        "rejected_vote_id": "vote-smol-43",
        "chosen_candidate_sha256": chosen_hash,
        "rejected_candidate_sha256": rejected_hash,
        "calibration_lock_id": "calibration-lock-1",
        "aggregation_id": "a" * 64,
        "teacher_cells": [
            {
                "backbone_key": "smollm2_17b",
                "seed": 42,
                "vote_id": "vote-smol-42",
                "calibration_id": "cal-smol-42",
                "candidate_sha256": chosen_hash,
                "candidate_source": "category_specialist",
            },
            {
                "backbone_key": "smollm2_17b",
                "seed": 43,
                "vote_id": "vote-smol-43",
                "calibration_id": "cal-smol-43",
                "candidate_sha256": rejected_hash,
                "candidate_source": "category_specialist",
            },
        ],
        "model_identities_hidden": True,
        "candidate_order_randomized": True,
        "adjudication_status": "resolved",
        "substantive_difference_fields": [
            "action", "violation_tags", "policy_ids"
        ],
        "adjudicated_gold": {
            "action": "intervene",
            "category": "mortgage_closed_end_advertising",
            "violation_tags": ["misleading_claim"],
            "policy_ids": ["REG_Z_1026_24"],
            "reference_label_id": "gold-label-1",
        },
        "gold_action": "intervene",
        "reviewer_ids": ["r-a", "r-b"],
        "adjudicator_id": "r-c",
        "adjudication_rationale": "The chosen output detects a misleading mortgage claim.",
        "policy_ids": ["REG_Z_1026_24"],
    }


def test_preference_is_cross_backbone_and_human_grounded():
    validate_preference(_preference(), minimum_teacher_seeds=2)
    bad = deepcopy(_preference())
    bad["teacher_backbone_keys"] = ["qwen25_15b"]
    with pytest.raises(ContractError, match="teach itself"):
        validate_preference(bad, minimum_teacher_seeds=2)


def test_chosen_action_must_match_adjudicated_gold():
    bad = deepcopy(_preference())
    bad["gold_action"] = "allow"
    with pytest.raises(ContractError, match="adjudicated gold"):
        validate_preference(bad, minimum_teacher_seeds=2)


def test_preference_rejects_confidence_only_pair_and_bad_grounding():
    confidence_only = deepcopy(_preference())
    confidence_only["rejected"] = deepcopy(confidence_only["chosen"])
    confidence_only["rejected"]["confidence"] = 0.8
    confidence_only["rejected_candidate_sha256"] = canonical_sha256(
        confidence_only["rejected"]
    )
    confidence_only["teacher_cells"][1]["candidate_sha256"] = confidence_only[
        "rejected_candidate_sha256"
    ]
    with pytest.raises(ContractError, match="non-substantive"):
        validate_preference(confidence_only, minimum_teacher_seeds=2)

    bad_grounding = deepcopy(_preference())
    bad_grounding["policy_ids"] = ["WRONG_POLICY"]
    with pytest.raises(ContractError, match="adjudicated grounding"):
        validate_preference(bad_grounding, minimum_teacher_seeds=2)


def test_preference_strict_domain_identity_and_registered_values():
    bad_domain = deepcopy(_preference())
    bad_domain["domain"] = "other"
    with pytest.raises(ContractError, match="invalid domain"):
        validate_preference(bad_domain, minimum_teacher_seeds=2)

    with pytest.raises(ContractError, match="unknown backbone"):
        validate_preference(
            _preference(),
            minimum_teacher_seeds=2,
            known_backbone_keys={"qwen25_15b"},
        )

    with pytest.raises(ContractError, match="unknown policy"):
        validate_preference(
            _preference(),
            minimum_teacher_seeds=2,
            known_policy_ids={"SOME_OTHER_POLICY"},
        )


def test_preference_requires_blinding_gold_and_paired_teacher_lineage():
    not_blinded = deepcopy(_preference())
    not_blinded["model_identities_hidden"] = False
    with pytest.raises(ContractError, match="must be true"):
        validate_preference(not_blinded, minimum_teacher_seeds=2)

    incomplete_gold = deepcopy(_preference())
    del incomplete_gold["adjudicated_gold"]["reference_label_id"]
    with pytest.raises(ContractError, match="adjudicated_gold missing"):
        validate_preference(incomplete_gold, minimum_teacher_seeds=2)

    mismatched_cells = deepcopy(_preference())
    mismatched_cells["teacher_cells"][1]["seed"] = 44
    with pytest.raises(ContractError, match="teacher seed list"):
        validate_preference(mismatched_cells, minimum_teacher_seeds=2)

    wrong_vote_hash = deepcopy(_preference())
    wrong_vote_hash["teacher_cells"][0]["candidate_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="wrong candidate hash"):
        validate_preference(wrong_vote_hash, minimum_teacher_seeds=2)

    wrong_source = deepcopy(_preference())
    wrong_source["candidate_source"] = "joint_generalist"
    with pytest.raises(ContractError, match="disagree with candidate_source"):
        validate_preference(wrong_source, minimum_teacher_seeds=2)

    tampered_policy = deepcopy(_preference())
    tampered_policy["policy_context"]["policy_text"] += " Changed."
    with pytest.raises(ContractError, match="text hash mismatch"):
        validate_preference(tampered_policy, minimum_teacher_seeds=2)


def test_locked_pair_weight_uses_independent_reliability_only():
    assert locked_pair_weight(
        0.7,
        reliability_lock_id="reliability-lock-1",
        reliability_record_id="record-1",
    ) == pytest.approx(0.7)
    assert locked_pair_weight(
        0.001,
        reliability_lock_id="reliability-lock-1",
        reliability_record_id="record-1",
        floor=0.05,
    ) == pytest.approx(0.05)
    with pytest.raises(ContractError, match="reliability_lock_id"):
        locked_pair_weight(
            0.7,
            reliability_lock_id="",
            reliability_record_id="record-1",
        )
