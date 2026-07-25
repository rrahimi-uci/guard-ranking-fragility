"""Human-grounded cross-model preference contracts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
import hashlib
import math
from numbers import Real
import re

from .contracts import ACTIONS, ContractError, canonical_sha256


REQUIRED_CANDIDATE_FIELDS = {
    "action", "category", "violation_tags", "policy_ids", "confidence",
    "confidence_source",
}
DOMAINS = {"general_safety", "mortgage"}
CANDIDATE_SOURCES = {"category_specialist", "joint_generalist"}
SOURCE_STRATA = {"teacher_agreement", "teacher_disagreement"}
SUBSTANTIVE_FIELDS = {"action", "violation_tags", "policy_ids", "review_behavior"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractError(f"{field} must be a nonempty string")
    return value


def _unique_strings(
    value: object,
    field: str,
    *,
    minimum: int = 0,
) -> list[str]:
    if not isinstance(value, list):
        raise ContractError(f"{field} must be a list")
    cleaned = [_nonempty_string(item, f"{field} item") for item in value]
    if len(cleaned) < minimum:
        raise ContractError(f"{field} requires at least {minimum} item(s)")
    if len(set(cleaned)) != len(cleaned):
        raise ContractError(f"{field} must not contain duplicates")
    return cleaned


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContractError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ContractError(f"{field} must lie in [0,1]")
    return number


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field} must be a positive integer")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _strict_true(value: object, field: str) -> None:
    if value is not True:
        raise ContractError(f"{field} must be true")


def _iso_date(value: object, field: str) -> str:
    text = _nonempty_string(value, field)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ContractError(f"{field} must be an ISO date") from exc
    if parsed.isoformat() != text:
        raise ContractError(f"{field} must use canonical YYYY-MM-DD form")
    return text


def _validate_candidate(candidate: object, *, category: str, name: str) -> dict:
    if not isinstance(candidate, Mapping):
        raise ContractError(f"{name} candidate must be an object")
    missing = REQUIRED_CANDIDATE_FIELDS - set(candidate)
    if missing:
        raise ContractError(f"{name} candidate missing fields: {sorted(missing)}")
    if candidate.get("action") not in ACTIONS:
        raise ContractError(f"{name} candidate has invalid action")
    if candidate.get("category") != category:
        raise ContractError(f"{name} candidate has wrong category")
    if candidate.get("confidence_source") != "calibrated_action_distribution":
        raise ContractError(f"{name} confidence lacks calibrated provenance")
    _unique_strings(candidate.get("violation_tags"), f"{name}.violation_tags")
    _unique_strings(candidate.get("policy_ids"), f"{name}.policy_ids")
    _probability(candidate.get("confidence"), f"{name}.confidence")
    return dict(candidate)


def _substantive_candidate(candidate: Mapping) -> tuple:
    """Fields that can justify a preference; self-confidence cannot."""
    return (
        candidate["action"],
        tuple(sorted(candidate["violation_tags"])),
        tuple(sorted(candidate["policy_ids"])),
    )


def _validate_adjudicated_gold(value: object, *, category: str) -> dict:
    if not isinstance(value, Mapping):
        raise ContractError("adjudicated_gold must be an object")
    required = {
        "action", "category", "violation_tags", "policy_ids", "reference_label_id"
    }
    missing = required - set(value)
    if missing:
        raise ContractError(f"adjudicated_gold missing fields: {sorted(missing)}")
    if value.get("action") not in ACTIONS:
        raise ContractError("adjudicated_gold has an invalid action")
    if value.get("category") != category:
        raise ContractError("adjudicated_gold has the wrong category")
    tags = _unique_strings(value.get("violation_tags"), "adjudicated_gold.violation_tags")
    policy_ids = _unique_strings(value.get("policy_ids"), "adjudicated_gold.policy_ids")
    reference_label_id = _nonempty_string(
        value.get("reference_label_id"), "adjudicated_gold.reference_label_id"
    )
    return {
        "action": value["action"],
        "category": category,
        "violation_tags": tags,
        "policy_ids": policy_ids,
        "reference_label_id": reference_label_id,
    }


def _validate_teacher_cells(value: object) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ContractError("teacher_cells must be a nonempty list")
    cells: list[dict] = []
    identities: set[tuple[str, str, int]] = set()
    vote_ids: set[str] = set()
    for index, cell in enumerate(value):
        if not isinstance(cell, Mapping):
            raise ContractError(f"teacher_cells[{index}] must be an object")
        backbone = _nonempty_string(
            cell.get("backbone_key"), f"teacher_cells[{index}].backbone_key"
        )
        seed = _positive_integer(cell.get("seed"), f"teacher_cells[{index}].seed")
        vote_id = _nonempty_string(
            cell.get("vote_id"), f"teacher_cells[{index}].vote_id"
        )
        calibration_id = _nonempty_string(
            cell.get("calibration_id"), f"teacher_cells[{index}].calibration_id"
        )
        candidate_sha256 = _sha256(
            cell.get("candidate_sha256"),
            f"teacher_cells[{index}].candidate_sha256",
        )
        candidate_source = cell.get("candidate_source")
        if candidate_source not in CANDIDATE_SOURCES:
            raise ContractError(f"teacher_cells[{index}] has invalid candidate_source")
        identity = (candidate_source, backbone, seed)
        if identity in identities or vote_id in vote_ids:
            raise ContractError("teacher_cells contain duplicate identities")
        identities.add(identity)
        vote_ids.add(vote_id)
        cells.append({
            "backbone_key": backbone,
            "seed": seed,
            "vote_id": vote_id,
            "calibration_id": calibration_id,
            "candidate_sha256": candidate_sha256,
            "candidate_source": candidate_source,
        })
    return cells


def _validate_policy_context(
    value: object,
    context_sha256: object,
    *,
    domain: str,
) -> dict | None:
    if domain == "general_safety":
        if value is not None or context_sha256 is not None:
            raise ContractError("general-safety preferences cannot carry mortgage policy context")
        return None
    if not isinstance(value, Mapping):
        raise ContractError("mortgage preference requires immutable policy_context")
    required = {
        "snapshot_id", "snapshot_object_sha256", "policy_vintage_lock_id",
        "policy_as_of", "authority_ids", "policy_text", "content_sha256",
    }
    missing = required - set(value)
    if missing:
        raise ContractError(f"policy_context missing fields: {sorted(missing)}")
    extra = set(value) - required
    if extra:
        raise ContractError(f"policy_context has unknown fields: {sorted(extra)}")
    _nonempty_string(value.get("snapshot_id"), "policy_context.snapshot_id")
    _sha256(
        value.get("snapshot_object_sha256"),
        "policy_context.snapshot_object_sha256",
    )
    _nonempty_string(
        value.get("policy_vintage_lock_id"),
        "policy_context.policy_vintage_lock_id",
    )
    _iso_date(value.get("policy_as_of"), "policy_context.policy_as_of")
    _unique_strings(value.get("authority_ids"), "policy_context.authority_ids", minimum=1)
    policy_text = _nonempty_string(value.get("policy_text"), "policy_context.policy_text")
    expected_text_hash = hashlib.sha256(policy_text.encode("utf-8")).hexdigest()
    if value.get("content_sha256") != expected_text_hash:
        raise ContractError("policy_context text hash mismatch")
    if _sha256(context_sha256, "policy_context_sha256") != canonical_sha256(value):
        raise ContractError("policy_context object hash mismatch")
    return dict(value)


def validate_preference(
    preference: Mapping,
    *,
    minimum_teacher_seeds: int,
    minimum_mortgage_reviewers: int = 2,
    category_domains: Mapping[str, str] | None = None,
    known_backbone_keys: set[str] | None = None,
    known_policy_ids: set[str] | None = None,
) -> None:
    if not isinstance(preference, Mapping):
        raise ContractError("preference must be an object")
    minimum_teacher_seeds = _positive_integer(
        minimum_teacher_seeds, "minimum_teacher_seeds"
    )
    minimum_mortgage_reviewers = _positive_integer(
        minimum_mortgage_reviewers, "minimum_mortgage_reviewers"
    )
    required = {
        "preference_id", "sample_id", "family_id", "category", "domain",
        "target_backbone_key", "teacher_backbone_keys", "teacher_seeds",
        "chosen", "rejected", "gold_action", "reviewer_ids", "adjudicator_id",
        "adjudication_rationale", "policy_ids", "adjudicated_gold",
        "calibration_lock_id", "aggregation_id", "teacher_cells",
        "chosen_vote_id", "rejected_vote_id", "chosen_candidate_sha256",
        "rejected_candidate_sha256", "candidate_source", "source_stratum",
        "policy_context", "policy_context_sha256", "substantive_difference_fields",
        "adjudication_status", "model_identities_hidden",
        "candidate_order_randomized",
    }
    missing = required - set(preference)
    if missing:
        raise ContractError(f"preference missing fields: {sorted(missing)}")
    for field in ("preference_id", "sample_id", "family_id"):
        _nonempty_string(preference.get(field), field)
    category = _nonempty_string(preference.get("category"), "category")
    domain = preference.get("domain")
    if domain not in DOMAINS:
        raise ContractError("preference has invalid domain")
    policy_context = _validate_policy_context(
        preference.get("policy_context"),
        preference.get("policy_context_sha256"),
        domain=domain,
    )
    candidate_source = preference.get("candidate_source")
    if candidate_source not in CANDIDATE_SOURCES:
        raise ContractError("preference has invalid candidate_source")
    source_stratum = preference.get("source_stratum")
    if source_stratum not in SOURCE_STRATA:
        raise ContractError("preference has invalid source_stratum")
    if category_domains is not None:
        if category not in category_domains:
            raise ContractError("preference category is not registered")
        if category_domains[category] != domain:
            raise ContractError("preference domain disagrees with category")
    chosen = _validate_candidate(preference["chosen"], category=category, name="chosen")
    rejected = _validate_candidate(preference["rejected"], category=category, name="rejected")
    if canonical_sha256(chosen) == canonical_sha256(rejected):
        raise ContractError("chosen and rejected candidates are identical")
    if _substantive_candidate(chosen) == _substantive_candidate(rejected):
        raise ContractError("chosen and rejected differ only in non-substantive fields")
    declared_differences = _unique_strings(
        preference.get("substantive_difference_fields"),
        "substantive_difference_fields",
        minimum=1,
    )
    if not set(declared_differences) <= SUBSTANTIVE_FIELDS:
        raise ContractError("preference declares an unknown substantive difference")
    actual_differences: set[str] = set()
    if chosen["action"] != rejected["action"]:
        actual_differences.add("action")
    if set(chosen["violation_tags"]) != set(rejected["violation_tags"]):
        actual_differences.add("violation_tags")
    if set(chosen["policy_ids"]) != set(rejected["policy_ids"]):
        actual_differences.add("policy_ids")
    if (chosen["action"] == "review") != (rejected["action"] == "review"):
        actual_differences.add("review_behavior")
    if set(declared_differences) != actual_differences:
        raise ContractError("declared substantive differences do not match candidates")
    chosen_hash = _sha256(
        preference.get("chosen_candidate_sha256"), "chosen_candidate_sha256"
    )
    rejected_hash = _sha256(
        preference.get("rejected_candidate_sha256"), "rejected_candidate_sha256"
    )
    if chosen_hash != canonical_sha256(chosen):
        raise ContractError("chosen candidate hash mismatch")
    if rejected_hash != canonical_sha256(rejected):
        raise ContractError("rejected candidate hash mismatch")
    chosen_vote_id = _nonempty_string(preference.get("chosen_vote_id"), "chosen_vote_id")
    rejected_vote_id = _nonempty_string(
        preference.get("rejected_vote_id"), "rejected_vote_id"
    )
    if chosen_vote_id == rejected_vote_id:
        raise ContractError("chosen and rejected candidates require distinct vote IDs")
    _nonempty_string(preference.get("calibration_lock_id"), "calibration_lock_id")
    _sha256(preference.get("aggregation_id"), "aggregation_id")
    if preference.get("adjudication_status") != "resolved":
        raise ContractError("preference adjudication must be resolved")
    _strict_true(
        preference.get("model_identities_hidden"), "model_identities_hidden"
    )
    _strict_true(
        preference.get("candidate_order_randomized"),
        "candidate_order_randomized",
    )
    adjudicated_gold = _validate_adjudicated_gold(
        preference.get("adjudicated_gold"), category=category
    )
    gold_action = preference.get("gold_action")
    if gold_action not in ACTIONS:
        raise ContractError("preference has invalid gold action")
    if chosen["action"] != gold_action or gold_action != adjudicated_gold["action"]:
        raise ContractError("chosen action is inconsistent with adjudicated gold")
    gold_policy_ids = _unique_strings(preference.get("policy_ids"), "policy_ids")
    if (
        set(chosen["policy_ids"]) != set(gold_policy_ids)
        or set(gold_policy_ids) != set(adjudicated_gold["policy_ids"])
    ):
        raise ContractError("chosen policy IDs disagree with adjudicated grounding")
    if set(chosen["violation_tags"]) != set(adjudicated_gold["violation_tags"]):
        raise ContractError("chosen violation tags disagree with adjudicated gold")
    target = _nonempty_string(
        preference.get("target_backbone_key"), "target_backbone_key"
    )
    teachers = _unique_strings(
        preference.get("teacher_backbone_keys"), "teacher_backbone_keys", minimum=1
    )
    if target in teachers:
        raise ContractError("target backbone cannot teach itself")
    teacher_cells = _validate_teacher_cells(preference.get("teacher_cells"))
    if {cell["candidate_source"] for cell in teacher_cells} != {candidate_source}:
        raise ContractError("teacher_cells disagree with candidate_source")
    cell_backbones = {cell["backbone_key"] for cell in teacher_cells}
    cell_seeds = {cell["seed"] for cell in teacher_cells}
    if set(teachers) != cell_backbones:
        raise ContractError("teacher backbone list disagrees with teacher_cells")
    if known_backbone_keys is not None:
        unknown = ({target} | set(teachers)) - set(known_backbone_keys)
        if unknown:
            raise ContractError(f"preference has unknown backbone keys: {sorted(unknown)}")
    seeds = preference.get("teacher_seeds")
    if (
        not isinstance(seeds, list)
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0
            for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
        or len(seeds) < minimum_teacher_seeds
    ):
        raise ContractError("preference lacks enough distinct teacher seeds")
    if set(seeds) != cell_seeds:
        raise ContractError("teacher seed list disagrees with teacher_cells")
    cells_by_vote = {cell["vote_id"]: cell for cell in teacher_cells}
    if chosen_vote_id not in cells_by_vote or rejected_vote_id not in cells_by_vote:
        raise ContractError("chosen and rejected vote IDs are absent from teacher_cells")
    if cells_by_vote[chosen_vote_id]["candidate_sha256"] != chosen_hash:
        raise ContractError("chosen vote lineage has the wrong candidate hash")
    if cells_by_vote[rejected_vote_id]["candidate_sha256"] != rejected_hash:
        raise ContractError("rejected vote lineage has the wrong candidate hash")
    reviewers = _unique_strings(preference.get("reviewer_ids"), "reviewer_ids", minimum=1)
    if domain == "mortgage":
        if len(reviewers) < minimum_mortgage_reviewers:
            raise ContractError("mortgage preference lacks two distinct reviewers")
        if not gold_policy_ids:
            raise ContractError("mortgage preference lacks policy grounding")
        if not set(gold_policy_ids) <= set(policy_context["authority_ids"]):
            raise ContractError("adjudicated policy IDs are absent from policy_context")
    if known_policy_ids is not None:
        unknown_policy_ids = (
            set(chosen["policy_ids"])
            | set(rejected["policy_ids"])
            | set(gold_policy_ids)
        ) - set(known_policy_ids)
        if unknown_policy_ids:
            raise ContractError(
                f"preference has unknown policy IDs: {sorted(unknown_policy_ids)}"
            )
    adjudicator = _nonempty_string(preference.get("adjudicator_id"), "adjudicator_id")
    if adjudicator in set(reviewers):
        raise ContractError("preference requires a separate adjudicator")
    if not str(preference.get("adjudication_rationale", "")).strip():
        raise ContractError("preference lacks adjudication rationale")


def locked_pair_weight(
    reliability_score: float,
    *,
    reliability_lock_id: str,
    reliability_record_id: str,
    floor: float = 0.05,
    ceiling: float = 1.0,
) -> float:
    """Clip an independently locked reliability weight.

    Candidate confidence and pair ordering are deliberately absent from this API.
    """
    score = _probability(reliability_score, "reliability_score")
    _nonempty_string(reliability_lock_id, "reliability_lock_id")
    _nonempty_string(reliability_record_id, "reliability_record_id")
    floor = _probability(floor, "floor")
    ceiling = _probability(ceiling, "ceiling")
    if floor <= 0 or floor > ceiling:
        raise ContractError("invalid pair-weight clipping bounds")
    return min(ceiling, max(floor, score))
