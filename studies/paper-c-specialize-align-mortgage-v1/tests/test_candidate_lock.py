from __future__ import annotations

from copy import deepcopy

import pytest

from paper_c_sta.contracts import (
    ContractError,
    canonical_sha256,
    output_path,
    read_json,
    write_json,
)
from paper_c_sta.locks import (
    DEFAULT_LOCK,
    REQUIRED_CHILDREN,
    create_protocol_candidate,
    validate_protocol_candidate,
)


def test_tracked_candidate_lock_binds_live_sources_and_authorizes_nothing():
    lock = validate_protocol_candidate(DEFAULT_LOCK)
    assert output_path(DEFAULT_LOCK).is_file()
    assert lock["status"] == "candidate_only_no_data_or_training_authorized"
    assert lock["data_build_authorized"] is False
    assert lock["gpu_training_authorized"] is False
    assert lock["claim_authorized"] is False
    assert tuple(lock["required_children"]) == REQUIRED_CHILDREN
    assert "post_pilot_prospective_primary_protocol_lock" in REQUIRED_CHILDREN
    assert "pilot_data_policy_lock" in REQUIRED_CHILDREN
    assert "primary_data_policy_lock" in REQUIRED_CHILDREN
    assert "pilot_adjudicated_preference_lock" in REQUIRED_CHILDREN
    assert "primary_adjudicated_preference_lock" in REQUIRED_CHILDREN


def test_candidate_lock_is_immutable_and_cannot_bind_smoke():
    with pytest.raises(ContractError, match="refusing to overwrite"):
        create_protocol_candidate()
    with pytest.raises(ContractError, match="only the primary profile"):
        create_protocol_candidate(
            "config/smoke.json", out_path="artifacts/forbidden-smoke-lock.json"
        )


def test_rehashed_metadata_tamper_still_fails_semantic_validation():
    tampered = deepcopy(read_json(DEFAULT_LOCK))
    tampered["status"] = "training_authorized"
    tampered.pop("lock_sha256")
    tampered["lock_sha256"] = canonical_sha256(tampered)
    path = output_path("artifacts/test-tampered-candidate-lock.json")
    try:
        write_json(path, tampered)
        with pytest.raises(ContractError, match="status drifted"):
            validate_protocol_candidate(path, bind_live_sources=False)
    finally:
        path.unlink(missing_ok=True)
