"""Machine-check the matched CM-DPO source comparison."""

from __future__ import annotations

from collections.abc import Mapping
import re

from .contracts import ContractError, canonical_sha256, validate_config


HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARMS = {
    "generalist_cm_dpo": "joint_generalist",
    "specialist_cm_dpo": "category_specialist",
}
MATCHED_FIELDS = (
    "reference_checkpoint_sha256",
    "source_event_manifest_sha256",
    "source_event_ids_sha256",
    "retained_pair_event_ids_sha256",
    "pair_quota_manifest_sha256",
    "adjudicated_gold_manifest_sha256",
    "retention_replay_manifest_sha256",
    "optimizer_config_sha256",
    "objective_config_sha256",
    "serialization_config_sha256",
    "review_protocol_sha256",
    "checkpoint_steps",
    "pair_quota",
    "token_budget",
)
HASH_FIELDS = tuple(field for field in MATCHED_FIELDS if field.endswith("sha256"))


def cm_dpo_objective_sha256(config: Mapping) -> str:
    """Hash every frozen field that defines the two confirmatory DPO losses."""
    validate_config(config)
    alignment = config["alignment"]
    preferences = config["preferences"]
    return canonical_sha256({
        "objective": "reference_centered_dpo",
        "beta": preferences["beta"],
        "reference": alignment["reference"],
        "category_dro_temperature": alignment["category_dro_temperature"],
        "lambda_gold": alignment["lambda_gold"],
        "lambda_retain": alignment["lambda_retain"],
        "pair_logprob_reduction": alignment["pair_logprob_reduction"],
        "candidate_length_rule": alignment["candidate_length_rule"],
    })


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a nonempty string")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field} must be a positive integer")
    return value


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _arm_record(name: str, value: object) -> dict:
    if not isinstance(value, Mapping):
        raise ContractError(f"comparison arm {name} must be an object")
    required = {
        "run_id", "arm", "candidate_source", "pair_inventory_sha256",
        *MATCHED_FIELDS,
    }
    missing = required - set(value)
    if missing:
        raise ContractError(f"comparison arm {name} missing fields: {sorted(missing)}")
    if value.get("arm") != name:
        raise ContractError(f"comparison arm identity drifted: {name}")
    if value.get("candidate_source") != ARMS[name]:
        raise ContractError(f"comparison arm has the wrong candidate source: {name}")
    _nonempty(value.get("run_id"), f"{name}.run_id")
    _hash(value.get("pair_inventory_sha256"), f"{name}.pair_inventory_sha256")
    for field in HASH_FIELDS:
        _hash(value.get(field), f"{name}.{field}")
    checkpoints = value.get("checkpoint_steps")
    if (
        not isinstance(checkpoints, list)
        or not checkpoints
        or any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0
            for step in checkpoints
        )
        or checkpoints != sorted(set(checkpoints))
    ):
        raise ContractError(f"{name}.checkpoint_steps must be unique positive integers")
    _positive_int(value.get("pair_quota"), f"{name}.pair_quota")
    _positive_int(value.get("token_budget"), f"{name}.token_budget")
    return dict(value)


def validate_matched_source_comparison(
    manifest: Mapping,
    *,
    config: Mapping | None = None,
) -> None:
    """Reject any source comparison that changes more than candidate provenance."""
    if not isinstance(manifest, Mapping):
        raise ContractError("source-comparison manifest must be an object")
    _nonempty(manifest.get("comparison_id"), "comparison_id")
    backbone = _nonempty(manifest.get("backbone_key"), "backbone_key")
    seed = _positive_int(manifest.get("seed"), "seed")
    arms = manifest.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise ContractError("source comparison requires exactly the two CM-DPO arms")
    records = {name: _arm_record(name, arms[name]) for name in ARMS}
    left = records["generalist_cm_dpo"]
    right = records["specialist_cm_dpo"]
    for field in MATCHED_FIELDS:
        if left[field] != right[field]:
            raise ContractError(f"CM-DPO source comparison is unmatched: {field}")
    if left["run_id"] == right["run_id"]:
        raise ContractError("CM-DPO arms require distinct run IDs")
    if left["pair_inventory_sha256"] == right["pair_inventory_sha256"]:
        raise ContractError("candidate-source pair inventories must be distinct")
    if config is not None:
        validate_config(config)
        if backbone not in config["backbones"]:
            raise ContractError("comparison names an unknown backbone")
        if seed not in config["seeds"]:
            raise ContractError("comparison names a non-primary seed")
        if left["checkpoint_steps"] != config["alignment"]["checkpoint_steps"]:
            raise ContractError("comparison checkpoint ladder disagrees with config")
        if left["objective_config_sha256"] != cm_dpo_objective_sha256(config):
            raise ContractError("comparison objective hash disagrees with config")
