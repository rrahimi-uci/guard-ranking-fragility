from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from paper_c_sta import contracts
from paper_c_sta.contracts import (
    ContractError,
    canonical_sha256,
    expected_training_grid,
    load_config,
    load_policy_snapshot,
    load_taxonomy,
    output_path,
    readiness_blockers,
    validate_config,
    validate_task_event,
)


def test_primary_config_and_exact_crossed_grid():
    config = load_config()
    grid = expected_training_grid(config)
    assert len(grid["references"]) == 6
    assert len(grid["specialists"]) == 30
    assert len(grid["aligned"]) == 30
    assert config["pilot"]["expected_training_cells"] == 44
    assert len({
        (row["category"], row["backbone"], row["seed"])
        for row in grid["specialists"]
    }) == 30


def test_config_rejects_category_backbone_confound():
    config = deepcopy(load_config())
    config["specialists"]["grid"] = "one_category_per_backbone"
    with pytest.raises(ContractError, match="crossover"):
        validate_config(config)


def test_readiness_fails_closed():
    blockers = readiness_blockers(load_config())
    assert "mortgage_policy_sme_signed" in blockers
    assert "sealed_cohorts_created" in blockers


def test_output_cannot_escape_new_study():
    with pytest.raises(ContractError, match="escapes"):
        output_path("../legacy-overwrite.json")


def test_mortgage_event_requires_context_two_reviewers_and_adjudicator():
    policy_text = "Regulation B prohibits discrimination and discouragement."
    policy_snapshot = load_policy_snapshot()
    event = {
        "sample_id": "m-1",
        "family_id": "family-1",
        "content_family_id": "content-family-1",
        "category": "mortgage_fair_lending",
        "domain": "mortgage",
        "request": "Draft a response for this scenario.",
        "proposed_response": "Use the protected trait to change the quoted terms.",
        "context": {
            "actor_role": "creditor",
            "product": "closed_end_mortgage",
            "transaction_stage": "marketing",
            "applicable_regime": ["ECOA", "Regulation B"],
            "coverage_facts": {"consumer_credit": True, "dwelling_secured": True},
        },
        "jurisdiction": "US_federal",
        "policy_as_of": "2026-07-25",
        "temporal_evaluation_eligible": False,
        "policy_context": {
            "snapshot_id": "us_federal_mortgage_risk_triage_2026_07_25_candidate",
            "snapshot_object_sha256": canonical_sha256(policy_snapshot),
            "policy_vintage_lock_id": "candidate_post_2026_07_21_vintage_unapproved",
            "policy_as_of": "2026-07-25",
            "authority_ids": ["REG_B_1002_4"],
            "policy_text": policy_text,
            "content_sha256": hashlib.sha256(policy_text.encode("utf-8")).hexdigest(),
        },
        "gold": {
            "action": "intervene",
            "violation_tags": ["disparate_treatment"],
            "policy_ids": ["REG_B_1002_4"],
            "rationale": "The proposed action explicitly changes terms on a protected basis.",
            "reviewer_ids": ["reviewer-a", "reviewer-b"],
            "adjudicator_id": "adjudicator-c",
        },
        "provenance": {
            "source_id": "synthetic-1",
            "content_sha256": "0" * 64,
            "licence_id": "internal-research",
            "synthetic": True,
            "contains_real_pii": False,
        },
    }
    validate_task_event(event, claim_bearing=False)
    event["gold"]["reviewer_ids"] = ["reviewer-a"]
    with pytest.raises(ContractError, match="two distinct reviewers"):
        validate_task_event(event, claim_bearing=False)


def test_policy_snapshot_carries_2026_temporal_conflict():
    taxonomy = load_taxonomy()
    policy = load_policy_snapshot(taxonomy=taxonomy)
    conflict = {row["id"]: row for row in policy["temporal_conflicts"]}
    assert conflict["ECOA_FHA_EFFECTS_2026"]["required_action"] == "review"


def test_output_path_admits_the_storage_contract_runs_directory(tmp_path):
    """<repo>/runs/<slug>/ must be writable, or the storage contract is unreachable."""
    target = contracts.runs_root() / "run_2026_07_26" / "cell.json"
    assert contracts.output_path(target) == target.resolve()
    assert contracts.runs_root().name == "paper-c-specialize-align-mortgage-v1"
    assert contracts.runs_root().parent.name == "runs"
    # the slug is the study, not the executing directory: both trees agree
    assert contracts.runs_root().parent.parent == contracts.repository_root()


def test_output_path_still_rejects_everything_outside_both_roots(tmp_path):
    """Two permitted roots, not open containment."""
    for bad in (contracts.repository_root() / "artifacts/paper_a_sft_v2/LOCK.json",
                contracts.repository_root() / "runs/some-other-study/x.json",
                tmp_path / "elsewhere.json"):
        with pytest.raises(contracts.ContractError) as excinfo:
            contracts.output_path(bad)
        assert "permitted roots" in str(excinfo.value)


def test_explicit_root_is_not_widened_by_the_runs_allowance(tmp_path):
    """An explicit root= is a containment request; runs/ must not escape it."""
    with pytest.raises(contracts.ContractError) as excinfo:
        contracts.output_path(contracts.runs_root() / "x.json", root=tmp_path)
    assert "requested root" in str(excinfo.value)
    inside = tmp_path / "ok.json"
    assert contracts.output_path("ok.json", root=tmp_path) == inside.resolve()


def test_relative_paths_still_resolve_inside_the_workspace(tmp_path):
    """Regression: 'runs/x' used to land inside the workspace and must keep doing so.

    The workspace stays the default root, so committed development output and every
    existing config path resolve exactly as before this allowance was added.
    """
    assert contracts.output_path("artifacts/generated/x.json") == (
        contracts.project_root() / "artifacts/generated/x.json").resolve()
