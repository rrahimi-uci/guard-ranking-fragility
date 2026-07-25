from __future__ import annotations

from copy import deepcopy
import math

import pytest

from paper_c_sta.contracts import ContractError, load_config, sha256_ordered
from paper_c_sta.locks import source_inventory
from paper_c_sta.splits import assign_event_split, assign_family_split, validate_split_isolation


def _fractions():
    return {
        "specialist_train": 0.5,
        "alignment_pool": 0.2,
        "calibration": 0.15,
        "checkpoint_selection": 0.15,
    }


def _split_config():
    return {
        "data": {
            "family_split_seed": 20260725,
            "family_split": _fractions(),
            "temporal_policy_cutoff": "2026-07-20",
        }
    }


def _row(family_id, split, *, source_id=None, content_family_id=None, **extra):
    return {
        "family_id": family_id,
        "content_family_id": content_family_id or f"content-{family_id}",
        "source_id": source_id or f"source-{family_id}",
        "split": split,
        "temporal_evaluation_eligible": False,
        **extra,
    }


def test_family_assignment_is_deterministic():
    fractions = _fractions()
    first = assign_family_split("family-123", seed=20260725, fractions=fractions)
    second = assign_family_split("family-123", seed=20260725, fractions=fractions)
    assert first == second


def test_only_explicit_temporal_cohort_is_held_out_on_both_cutoff_sides():
    for policy_as_of, side in (
        ("2026-07-19", "pre_cutoff"),
        ("2026-07-25", "post_cutoff"),
    ):
        event = {
            "family_id": f"f-{side}",
            "policy_as_of": policy_as_of,
            "temporal_evaluation_eligible": True,
            "temporal_policy_side": side,
        }
        assert assign_event_split(event, config=_split_config()) == "temporal_test"

    ordinary = {
        "family_id": "ordinary-current-vintage",
        "policy_as_of": "2026-07-25",
        "temporal_evaluation_eligible": False,
    }
    assert assign_event_split(ordinary, config=_split_config()) in set(_fractions())


def test_family_leakage_fails_closed():
    rows = [
        _row("same", "specialist_train"),
        _row("same", "checkpoint_selection"),
    ]
    with pytest.raises(ContractError, match="leak"):
        validate_split_isolation(rows)


def test_split_contract_rejects_nonfinite_fraction_and_unknown_role():
    fractions = {
        "specialist_train": math.nan,
        "alignment_pool": 0.2,
        "calibration": 0.1,
        "checkpoint_selection": 0.7,
    }
    with pytest.raises(ContractError, match="finite"):
        assign_family_split("family", seed=1, fractions=fractions)

    with pytest.raises(ContractError, match="unknown split"):
        validate_split_isolation([_row("family", "invented")])


def test_temporal_rows_require_canonical_dates_and_correct_partition():
    with pytest.raises(ContractError, match="lacks policy_as_of"):
        validate_split_isolation(
            [
                _row(
                    "family",
                    "temporal_test",
                    temporal_evaluation_eligible=True,
                    temporal_policy_side="post_cutoff",
                )
            ],
            config=_split_config(),
        )

    with pytest.raises(ContractError, match="ISO date"):
        assign_event_split(
            {
                "family_id": "family",
                "policy_as_of": "not-a-date",
                "temporal_evaluation_eligible": True,
            },
            config=_split_config(),
        )

    with pytest.raises(ContractError, match="frozen cutoff"):
        validate_split_isolation(
            [
                {
                    **_row("family", "checkpoint_selection"),
                    "policy_as_of": "2026-07-25",
                    "temporal_evaluation_eligible": True,
                    "temporal_policy_side": "post_cutoff",
                }
            ],
            config=_split_config(),
        )


def test_split_isolation_covers_content_families_and_sources():
    with pytest.raises(ContractError, match="content families leak"):
        validate_split_isolation(
            [
                _row("f-1", "specialist_train", content_family_id="shared-content"),
                _row("f-2", "calibration", content_family_id="shared-content"),
            ]
        )

    with pytest.raises(ContractError, match="sources leak"):
        validate_split_isolation(
            [
                _row("f-1", "specialist_train", source_id="shared-source"),
                _row("f-2", "calibration", source_id="shared-source"),
            ]
        )


def test_source_inventory_is_internally_hash_bound():
    inventory = source_inventory()
    assert sha256_ordered(inventory["files"]) == inventory["aggregate_sha256"]
    assert any(row["path"] == "PROTOCOL.md" for row in inventory["files"])


def test_smoke_is_not_primary_evidence():
    smoke = deepcopy(load_config("config/smoke.json"))
    assert smoke["profile"] == "smoke"
    assert smoke["task"]["mortgage_claim"] == "infrastructure_only"
