from __future__ import annotations

from copy import deepcopy

import pytest

from paper_c_sta.comparisons import cm_dpo_objective_sha256, validate_matched_source_comparison
from paper_c_sta.contracts import ContractError, load_config


def _manifest() -> dict:
    config = load_config()
    shared = {
        "reference_checkpoint_sha256": "1" * 64,
        "source_event_manifest_sha256": "2" * 64,
        "source_event_ids_sha256": "3" * 64,
        "retained_pair_event_ids_sha256": "b" * 64,
        "pair_quota_manifest_sha256": "c" * 64,
        "adjudicated_gold_manifest_sha256": "4" * 64,
        "retention_replay_manifest_sha256": "5" * 64,
        "optimizer_config_sha256": "6" * 64,
        "objective_config_sha256": cm_dpo_objective_sha256(config),
        "serialization_config_sha256": "7" * 64,
        "review_protocol_sha256": "8" * 64,
        "checkpoint_steps": [25, 50, 100, 200],
        "pair_quota": 1000,
        "token_budget": 500000,
    }
    return {
        "comparison_id": "qwen-seed-42-source-comparison",
        "backbone_key": "qwen25_15b",
        "seed": 42,
        "arms": {
            "generalist_cm_dpo": {
                **shared,
                "run_id": "qwen-42-generalist",
                "arm": "generalist_cm_dpo",
                "candidate_source": "joint_generalist",
                "pair_inventory_sha256": "9" * 64,
            },
            "specialist_cm_dpo": {
                **shared,
                "run_id": "qwen-42-specialist",
                "arm": "specialist_cm_dpo",
                "candidate_source": "category_specialist",
                "pair_inventory_sha256": "a" * 64,
            },
        },
    }


def test_matched_source_comparison_accepts_only_candidate_source_change():
    validate_matched_source_comparison(_manifest(), config=load_config())


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_event_ids_sha256", "b" * 64),
        ("retained_pair_event_ids_sha256", "e" * 64),
        ("pair_quota_manifest_sha256", "f" * 64),
        ("reference_checkpoint_sha256", "c" * 64),
        ("objective_config_sha256", "d" * 64),
        ("pair_quota", 999),
        ("token_budget", 499999),
        ("checkpoint_steps", [25, 50, 200]),
    ],
)
def test_source_comparison_rejects_unmatched_nuisance(field, value):
    manifest = _manifest()
    manifest["arms"]["specialist_cm_dpo"][field] = value
    with pytest.raises(ContractError, match="unmatched"):
        validate_matched_source_comparison(manifest)


def test_source_comparison_rejects_wrong_source_and_unknown_cell():
    wrong_source = _manifest()
    wrong_source["arms"]["specialist_cm_dpo"]["candidate_source"] = "joint_generalist"
    with pytest.raises(ContractError, match="wrong candidate source"):
        validate_matched_source_comparison(wrong_source)

    unknown_seed = _manifest()
    unknown_seed["seed"] = 7
    with pytest.raises(ContractError, match="non-primary seed"):
        validate_matched_source_comparison(unknown_seed, config=load_config())
