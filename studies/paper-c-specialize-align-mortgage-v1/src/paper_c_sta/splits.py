"""Deterministic family and policy-time separation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
import hashlib
import math
from numbers import Real

from .contracts import ContractError, FAMILY_SPLITS


DEVELOPMENT_SPLITS = FAMILY_SPLITS
EVALUATION_SPLITS = ("public_test", "sealed_test", "temporal_test")
KNOWN_SPLITS = frozenset(DEVELOPMENT_SPLITS + EVALUATION_SPLITS)


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractError(f"{field} must be a nonempty string")
    return value


def _strict_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError("split seed must be a positive integer")
    return value


def _strict_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{field} must be boolean")
    return value


def _fraction(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContractError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0 < number < 1:
        raise ContractError(f"{field} must be finite and lie in (0,1)")
    return number


def _validated_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(fractions, Mapping):
        raise ContractError("split fractions must be an object")
    if tuple(fractions) != DEVELOPMENT_SPLITS:
        raise ContractError(
            f"split roles and order must be exactly {DEVELOPMENT_SPLITS}"
        )
    clean = {
        name: _fraction(fractions[name], f"split fraction {name}")
        for name in DEVELOPMENT_SPLITS
    }
    if not math.isclose(sum(clean.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError("split fractions must sum to one")
    return clean


def _split_contract(config: Mapping) -> tuple[int, dict[str, float], date]:
    if not isinstance(config, Mapping):
        raise ContractError("split config must be an object")
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ContractError("split config requires a data object")
    seed = _strict_seed(data.get("family_split_seed"))
    fractions = _validated_fractions(data.get("family_split"))
    cutoff = _iso_date(data.get("temporal_policy_cutoff"), "temporal_policy_cutoff")
    return seed, fractions, cutoff


def _iso_date(value: object, field: str) -> date:
    text = _nonempty_string(value, field)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ContractError(f"{field} must be an ISO date") from exc
    if parsed.isoformat() != text:
        raise ContractError(f"{field} must use canonical YYYY-MM-DD form")
    return parsed


def _policy_side(policy_as_of: date, cutoff: date) -> str:
    return "pre_cutoff" if policy_as_of <= cutoff else "post_cutoff"


def _unit_interval(identity: str, seed: int) -> float:
    payload = f"{seed}\0{identity}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / 2**64


def assign_family_split(
    family_id: str,
    *,
    seed: int,
    fractions: Mapping[str, float],
) -> str:
    family_id = _nonempty_string(family_id, "family_id")
    seed = _strict_seed(seed)
    clean_fractions = _validated_fractions(fractions)
    draw = _unit_interval(family_id, seed)
    cumulative = 0.0
    for name, fraction in clean_fractions.items():
        cumulative += fraction
        if draw < cumulative:
            return name
    return DEVELOPMENT_SPLITS[-1]


def assign_event_split(
    event: Mapping,
    *,
    config: Mapping,
) -> str:
    if not isinstance(event, Mapping):
        raise ContractError("split event must be an object")
    seed, fractions, cutoff = _split_contract(config)
    temporal_eligible = _strict_bool(
        event.get("temporal_evaluation_eligible"), "temporal_evaluation_eligible"
    )
    if temporal_eligible:
        policy_as_of = event.get("policy_as_of")
        if policy_as_of is None:
            raise ContractError("temporal-eligible event lacks policy_as_of")
        side = _policy_side(_iso_date(policy_as_of, "policy_as_of"), cutoff)
        declared_side = event.get("temporal_policy_side")
        if declared_side is not None and declared_side != side:
            raise ContractError("temporal policy side disagrees with the locked cutoff")
        return "temporal_test"
    return assign_family_split(event.get("family_id"), seed=seed, fractions=fractions)


def validate_split_isolation(
    rows: Sequence[Mapping],
    *,
    config: Mapping | None = None,
) -> None:
    """Fail if any family, content family, or source crosses a split boundary.

    NOT WIRED INTO THE PIPELINE. Every caller is in the test suite, so on the reuse
    corpus this states intent rather than enforcing it -- and it would in fact fail
    there, at the source level: unioning on source_id yields 9 isolation units across
    9 datasets, fewer than the four splits need populated at 50/20/15/15, so no
    assignment satisfies it. The constraint is infeasible on that corpus, not merely
    strict.

    Left checking all three levels rather than weakened to the two that are
    achievable. Relaxing it would make the function pass on a corpus that cannot
    support a held-out-source claim, which is the failure it exists to prevent; the
    honest statement is that the corpus does not meet the contract. PROTOCOL.md now
    says so, and puts held-out-source transfer out of scope.

    A study that needs source isolation must author enough independent sources to
    make it satisfiable, then call this from the ingest path.
    """
    if not rows:
        raise ContractError("split inventory is empty")
    cutoff = _split_contract(config)[2] if config is not None else None
    family_splits: dict[str, set[str]] = defaultdict(set)
    content_family_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if not isinstance(row, Mapping):
            raise ContractError("split row must be an object")
        family_id = _nonempty_string(row.get("family_id"), "family_id")
        content_family_id = _nonempty_string(
            row.get("content_family_id"), "content_family_id"
        )
        source_id = row.get("source_id")
        if source_id is None and isinstance(row.get("provenance"), Mapping):
            source_id = row["provenance"].get("source_id")
        source_id = _nonempty_string(source_id, "source_id")
        split = _nonempty_string(row.get("split"), "split")
        temporal_eligible = _strict_bool(
            row.get("temporal_evaluation_eligible"),
            "temporal_evaluation_eligible",
        )
        if split not in KNOWN_SPLITS:
            raise ContractError(f"unknown split: {split}")
        family_splits[family_id].add(split)
        content_family_splits[content_family_id].add(split)
        source_splits[source_id].add(split)
        if cutoff is None:
            if temporal_eligible or split == "temporal_test":
                raise ContractError("temporal rows require a locked split config")
        else:
            policy_as_of = row.get("policy_as_of")
            if temporal_eligible and policy_as_of is None:
                raise ContractError("temporal-eligible row lacks policy_as_of")
            if temporal_eligible:
                side = _policy_side(_iso_date(policy_as_of, "policy_as_of"), cutoff)
                if row.get("temporal_policy_side") != side:
                    raise ContractError(
                        "temporal policy side disagrees with the locked cutoff"
                    )
            if (split == "temporal_test") != temporal_eligible:
                raise ContractError("policy-time split violates the frozen cutoff")
    leaked = sorted(family for family, splits in family_splits.items() if len(splits) > 1)
    if leaked:
        raise ContractError(f"families leak across splits: {leaked[:5]}")
    leaked_content = sorted(
        family for family, splits in content_family_splits.items() if len(splits) > 1
    )
    if leaked_content:
        raise ContractError(
            f"content families leak across splits: {leaked_content[:5]}"
        )
    leaked_sources = sorted(
        source for source, splits in source_splits.items() if len(splits) > 1
    )
    if leaked_sources:
        raise ContractError(f"sources leak across splits: {leaked_sources[:5]}")
