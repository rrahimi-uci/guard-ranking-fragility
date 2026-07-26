"""Leave-one-backbone-out specialist gating and calibrated aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real

from .contracts import ACTIONS, ContractError, canonical_sha256


REQUIRED_CANDIDATE_FIELDS = {
    "action", "category", "violation_tags", "policy_ids", "confidence",
    "confidence_source",
}


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractError(f"{field} must be a nonempty string")
    return value


def _strict_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{field} must be boolean")
    return value


def _strict_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field} must be a positive integer")
    return value


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContractError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ContractError(f"{field} must lie in [0,1]")
    return number


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ContractError(f"{field} must be a list")
    cleaned = [_nonempty_string(item, f"{field} item") for item in value]
    if len(set(cleaned)) != len(cleaned):
        raise ContractError(f"{field} must not contain duplicates")
    return cleaned


@dataclass(frozen=True)
class SpecialistVote:
    vote_id: str
    sample_id: str
    family_id: str
    category: str
    backbone: str
    seed: int
    target_backbone: str
    qualified: bool
    abstain: bool
    probabilities: dict[str, float]
    calibration_id: str
    candidate: dict | None
    candidate_sha256: str | None

    @classmethod
    def from_mapping(cls, value: Mapping) -> "SpecialistVote":
        if not isinstance(value, Mapping):
            raise ContractError("specialist vote must be an object")
        probabilities = value.get("probabilities")
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(ACTIONS):
            raise ContractError("specialist probabilities must contain the three actions")
        normalized = {
            action: _probability(probabilities[action], f"probabilities.{action}")
            for action in ACTIONS
        }
        if abs(sum(normalized.values()) - 1.0) > 1e-6:
            raise ContractError("specialist probabilities must sum to one")
        category = _nonempty_string(
            value.get("specialist_category"), "specialist_category"
        )
        qualified = _strict_bool(value.get("qualified"), "qualified")
        abstain = _strict_bool(value.get("abstain"), "abstain")
        if not qualified and not abstain:
            raise ContractError("an unqualified specialist must abstain")
        candidate = value.get("candidate")
        candidate_sha256: str | None
        if abstain:
            if candidate is not None or value.get("candidate_sha256") is not None:
                raise ContractError("an abstaining specialist cannot carry a candidate")
            candidate_sha256 = None
        else:
            if not isinstance(candidate, Mapping):
                raise ContractError("non-abstaining specialist candidate must be an object")
            missing = REQUIRED_CANDIDATE_FIELDS - set(candidate)
            if missing:
                raise ContractError(
                    f"specialist candidate missing fields: {sorted(missing)}"
                )
            if candidate.get("action") not in ACTIONS:
                raise ContractError("specialist candidate has no valid action")
            if candidate.get("category") != category:
                raise ContractError(
                    "candidate category disagrees with specialist category"
                )
            if candidate.get("confidence_source") != "calibrated_action_distribution":
                raise ContractError("candidate confidence lacks calibrated provenance")
            _string_list(candidate.get("violation_tags"), "candidate.violation_tags")
            _string_list(candidate.get("policy_ids"), "candidate.policy_ids")
            candidate_confidence = _probability(
                candidate.get("confidence"), "candidate.confidence"
            )
            maximum = max(normalized.values())
            top_actions = [
                action for action, probability in normalized.items()
                if math.isclose(probability, maximum, rel_tol=0.0, abs_tol=1e-12)
            ]
            if len(top_actions) != 1:
                raise ContractError(
                    "specialist probabilities must have a unique top action"
                )
            if candidate.get("action") != top_actions[0]:
                raise ContractError(
                    "candidate action disagrees with calibrated top action"
                )
            if not math.isclose(
                candidate_confidence,
                normalized[top_actions[0]],
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ContractError(
                    "candidate confidence disagrees with calibrated probability"
                )
            candidate_sha256 = canonical_sha256(candidate)
            if value.get("candidate_sha256") != candidate_sha256:
                raise ContractError(
                    "candidate hash does not bind the structured candidate"
                )
        return cls(
            vote_id=_nonempty_string(value.get("vote_id"), "vote_id"),
            sample_id=_nonempty_string(value.get("sample_id"), "sample_id"),
            family_id=_nonempty_string(value.get("family_id"), "family_id"),
            category=category,
            backbone=_nonempty_string(value.get("backbone_key"), "backbone_key"),
            seed=_strict_int(value.get("seed"), "seed"),
            target_backbone=_nonempty_string(
                value.get("target_backbone_key"), "target_backbone_key"
            ),
            qualified=qualified,
            abstain=abstain,
            probabilities=normalized,
            calibration_id=_nonempty_string(
                value.get("calibration_id"), "calibration_id"
            ),
            candidate=dict(candidate) if candidate is not None else None,
            candidate_sha256=candidate_sha256,
        )

    @property
    def top_action(self) -> str:
        return max(ACTIONS, key=lambda action: self.probabilities[action])

    @property
    def confidence(self) -> float:
        return self.probabilities[self.top_action]


def _validated_vote(value: Mapping | SpecialistVote) -> SpecialistVote:
    if isinstance(value, SpecialistVote):
        value = {
            "vote_id": value.vote_id,
            "sample_id": value.sample_id,
            "family_id": value.family_id,
            "specialist_category": value.category,
            "backbone_key": value.backbone,
            "seed": value.seed,
            "target_backbone_key": value.target_backbone,
            "qualified": value.qualified,
            "abstain": value.abstain,
            "probabilities": value.probabilities,
            "calibration_id": value.calibration_id,
            "candidate": value.candidate,
            "candidate_sha256": value.candidate_sha256,
        }
    return SpecialistVote.from_mapping(value)


def _locked_calibrations(value: Mapping) -> tuple[str, dict[str, dict]]:
    if not isinstance(value, Mapping):
        raise ContractError("qualified calibration inventory must be an object")
    lock_id = _nonempty_string(value.get("lock_id"), "calibration inventory lock_id")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ContractError("qualified calibration inventory must contain entries")
    by_id: dict[str, dict] = {}
    cells: set[tuple[str, str, int]] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ContractError(f"calibration inventory entry {index} must be an object")
        calibration_id = _nonempty_string(
            entry.get("calibration_id"), f"calibration entry {index} calibration_id"
        )
        category = _nonempty_string(
            entry.get("specialist_category"),
            f"calibration entry {index} specialist_category",
        )
        backbone = _nonempty_string(
            entry.get("backbone_key"), f"calibration entry {index} backbone_key"
        )
        seed = _strict_int(entry.get("seed"), f"calibration entry {index} seed")
        qualified = _strict_bool(
            entry.get("qualified"), f"calibration entry {index} qualified"
        )
        cell = (category, backbone, seed)
        if calibration_id in by_id or cell in cells:
            raise ContractError("qualified calibration inventory contains duplicates")
        cells.add(cell)
        by_id[calibration_id] = {
            "specialist_category": category,
            "backbone_key": backbone,
            "seed": seed,
            "qualified": qualified,
        }
    return lock_id, by_id


def aggregate_specialists(
    values: Sequence[Mapping | SpecialistVote],
    *,
    target_category: str,
    target_backbone: str,
    minimum_distinct_seeds: int,
    minimum_confidence: float,
    qualified_calibration_inventory: Mapping,
    expected_sample_id: str | None = None,
) -> dict:
    """Return a candidate consensus, never a gold label.

    Self-backbone, out-of-category, unqualified, abstaining, and low-confidence
    votes are excluded. Any lack of consensus produces a routing-layer abstention,
    never a synthetic task-level ``review`` label.
    """
    target_category = _nonempty_string(target_category, "target_category")
    target_backbone = _nonempty_string(target_backbone, "target_backbone")
    if (
        isinstance(minimum_distinct_seeds, bool)
        or not isinstance(minimum_distinct_seeds, int)
        or minimum_distinct_seeds <= 0
    ):
        raise ContractError("minimum_distinct_seeds must be positive")
    minimum_confidence = _probability(minimum_confidence, "minimum_confidence")
    if expected_sample_id is not None:
        expected_sample_id = _nonempty_string(expected_sample_id, "expected_sample_id")
    calibration_lock_id, calibration_entries = _locked_calibrations(
        qualified_calibration_inventory
    )
    votes = [_validated_vote(value) for value in values]
    if not votes:
        raise ContractError("cannot aggregate an empty specialist set")
    sample_ids = {vote.sample_id for vote in votes}
    if len(sample_ids) != 1:
        raise ContractError("specialist votes mix multiple samples")
    sample_id = next(iter(sample_ids))
    family_ids = {vote.family_id for vote in votes}
    if len(family_ids) != 1:
        raise ContractError("specialist votes mix multiple families")
    family_id = next(iter(family_ids))
    if expected_sample_id is not None and sample_id != expected_sample_id:
        raise ContractError("specialist votes belong to a different sample")
    teacher_cells: set[tuple[str, str, str, str, int, str]] = set()
    vote_ids: set[str] = set()
    for vote in votes:
        identity = (
            vote.sample_id,
            vote.family_id,
            vote.category,
            vote.backbone,
            vote.seed,
            vote.target_backbone,
        )
        if identity in teacher_cells or vote.vote_id in vote_ids:
            raise ContractError("duplicate specialist teacher cell")
        teacher_cells.add(identity)
        vote_ids.add(vote.vote_id)
    eligible: list[SpecialistVote] = []
    excluded = {
        "self_backbone": 0,
        "wrong_category": 0,
        "not_in_locked_calibration_inventory": 0,
        "locked_unqualified": 0,
        "submitted_abstain": 0,
        "low_confidence": 0,
    }
    for vote in votes:
        if vote.target_backbone != target_backbone:
            raise ContractError("vote was generated for a different target backbone")
        if vote.backbone == target_backbone:
            excluded["self_backbone"] += 1
            continue
        if vote.category != target_category:
            excluded["wrong_category"] += 1
            continue
        calibration = calibration_entries.get(vote.calibration_id)
        if calibration is None:
            excluded["not_in_locked_calibration_inventory"] += 1
            continue
        expected_identity = (
            calibration["specialist_category"],
            calibration["backbone_key"],
            calibration["seed"],
        )
        if expected_identity != (vote.category, vote.backbone, vote.seed):
            raise ContractError("vote identity disagrees with locked calibration")
        if not calibration["qualified"]:
            excluded["locked_unqualified"] += 1
            continue
        if vote.abstain:
            excluded["submitted_abstain"] += 1
            continue
        if vote.confidence < minimum_confidence:
            excluded["low_confidence"] += 1
            continue
        eligible.append(vote)

    seeds = {vote.seed for vote in eligible}
    top_actions = {vote.top_action for vote in eligible}
    enough = len(seeds) >= minimum_distinct_seeds
    unanimous = len(top_actions) == 1
    eligible_candidates = [
        {
            "vote_id": vote.vote_id,
            "candidate_sha256": vote.candidate_sha256,
            "candidate": vote.candidate,
            "calibration_id": vote.calibration_id,
            "backbone_key": vote.backbone,
            "seed": vote.seed,
        }
        for vote in sorted(eligible, key=lambda item: (item.backbone, item.seed, item.vote_id))
    ]
    eligible_vote_ids = [candidate["vote_id"] for candidate in eligible_candidates]
    eligible_candidate_sha256 = [
        candidate["candidate_sha256"] for candidate in eligible_candidates
    ]
    if not enough or not unanimous:
        reason = "insufficient_teachers" if not enough else "teacher_disagreement"
        result = {
            "status": "no_consensus",
            "reason": reason,
            "routing_action": "abstain",
            "sample_id": sample_id,
            "family_id": family_id,
            "calibration_lock_id": calibration_lock_id,
            "candidate_action": None,
            "probabilities": None,
            "eligible_votes": len(eligible),
            "eligible_seeds": sorted(seeds),
            "teacher_backbones": sorted({vote.backbone for vote in eligible}),
            "eligible_candidates": eligible_candidates,
            "eligible_vote_ids": eligible_vote_ids,
            "eligible_candidate_sha256": eligible_candidate_sha256,
            "excluded": excluded,
            "requires_adjudication": True,
            "is_gold": False,
        }
        result["aggregation_id"] = canonical_sha256(result)
        return result

    weights = [vote.confidence for vote in eligible]
    denominator = sum(weights)
    probabilities = {
        action: sum(
            weight * vote.probabilities[action]
            for vote, weight in zip(eligible, weights, strict=True)
        ) / denominator
        for action in ACTIONS
    }
    result = {
        "status": "candidate_consensus",
        "reason": None,
        "routing_action": "candidate",
        "sample_id": sample_id,
        "family_id": family_id,
        "calibration_lock_id": calibration_lock_id,
        "candidate_action": next(iter(top_actions)),
        "probabilities": probabilities,
        "eligible_votes": len(eligible),
        "eligible_seeds": sorted(seeds),
        "teacher_backbones": sorted({vote.backbone for vote in eligible}),
        "eligible_candidates": eligible_candidates,
        "eligible_vote_ids": eligible_vote_ids,
        "eligible_candidate_sha256": eligible_candidate_sha256,
        "excluded": excluded,
        "requires_adjudication": True,
        "is_gold": False,
    }
    result["aggregation_id"] = canonical_sha256(result)
    return result
