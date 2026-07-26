"""Inference-time ensemble baselines, and the abstain rule they depend on.

The aligned student is only worth training if it beats what you can get by *composing
the specialists you already have* at inference time.  Without these baselines the
study can report "specialist candidates help the student" while a routed ensemble
quietly does better than the student for free, which would undercut the premise of
distilling into one model.  Three baselines, all requiring no new training:

``independent``
    Every specialist scores every event; probabilities are averaged over the
    specialists that did not abstain.

``or_vote``
    The classic guard ensemble: intervene if *any* qualified specialist says
    intervene, else review if any says review, else allow.  Maximises recall and is
    the usual straw man for false alarms.

``routed``
    Each event goes only to the specialist for its focal category.  This is the
    strongest baseline and the one that matters: if routing matches the student, the
    student buys nothing but a single deployable artifact.

Abstain is enforced here rather than assumed.  A specialist asked about a category it
was not trained on must abstain, *not* vote allow -- a silent "allow" from an
unqualified specialist is a false negative that the OR rule cannot see and the
average silently dilutes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .contracts import ContractError
from .modeling import ACTIONS

STRATEGIES = ("independent", "or_vote", "routed")
ABSTAIN = "abstain"


def qualified(specialist_category: str, event_category: str) -> bool:
    """A specialist is qualified only on its own focal category."""
    return specialist_category == event_category


def specialist_vote(
    specialist_category: str,
    event_category: str,
    probabilities: Mapping[str, float],
    *,
    min_confidence: float = 0.0,
) -> dict:
    """One specialist's vote, abstaining when unqualified or under-confident."""
    if set(probabilities) != set(ACTIONS):
        raise ContractError("a vote needs probabilities over exactly the three actions")
    if not qualified(specialist_category, event_category):
        return {"specialist_category": specialist_category, "abstain": True,
                "reason": "out_of_expertise", "action": None,
                "probabilities": dict(probabilities)}
    top = max(ACTIONS, key=lambda a: probabilities[a])
    if probabilities[top] < min_confidence:
        return {"specialist_category": specialist_category, "abstain": True,
                "reason": "below_min_confidence", "action": None,
                "probabilities": dict(probabilities)}
    return {"specialist_category": specialist_category, "abstain": False,
            "reason": None, "action": top, "probabilities": dict(probabilities)}


def combine(votes: Sequence[Mapping], *, strategy: str,
            event_category: str | None = None) -> dict:
    """Combine specialist votes under one strategy.

    Returns ``no_consensus`` when every specialist abstained.  That is an aggregation
    status, not a fourth action and not a silent allow: a deployment must route it to
    a human, and scoring must count it as such rather than as a benign prediction.
    """
    if strategy not in STRATEGIES:
        raise ContractError(f"strategy must be one of {STRATEGIES}")
    active = [v for v in votes if not v["abstain"]]
    if not active:
        return {"strategy": strategy, "action": None, "no_consensus": True,
                "n_qualified": 0, "probabilities": None}

    if strategy == "or_vote":
        # escalation-ordered: any intervene wins, then any review
        for action in ("intervene", "review"):
            if any(v["action"] == action for v in active):
                return {"strategy": strategy, "action": action, "no_consensus": False,
                        "n_qualified": len(active), "probabilities": None}
        return {"strategy": strategy, "action": "allow", "no_consensus": False,
                "n_qualified": len(active), "probabilities": None}

    if strategy == "routed":
        if event_category is None:
            raise ContractError("routed combination needs the event's focal category")
        routed = [v for v in active if v["specialist_category"] == event_category]
        if not routed:
            return {"strategy": strategy, "action": None, "no_consensus": True,
                    "n_qualified": 0, "probabilities": None}
        probs = routed[0]["probabilities"]
    else:  # independent: average over qualified specialists
        probs = {
            a: sum(v["probabilities"][a] for v in active) / len(active)
            for a in ACTIONS
        }
    total = sum(probs.values()) or 1.0
    probs = {a: probs[a] / total for a in ACTIONS}
    return {
        "strategy": strategy,
        "action": max(ACTIONS, key=lambda a: probs[a]),
        "no_consensus": False,
        "n_qualified": len(active),
        "probabilities": probs,
    }


def ensemble_predictions(events: Sequence[Mapping],
                         specialist_probabilities: Mapping[str, Mapping[str, Mapping[str, float]]],
                         *, strategy: str, min_confidence: float = 0.0) -> list[dict]:
    """Predict every event under one ensemble strategy.

    ``specialist_probabilities[sample_id][specialist_category]`` gives that
    specialist's three action probabilities for that event.
    """
    out = []
    for event in events:
        by_specialist = specialist_probabilities.get(event["sample_id"], {})
        votes = [
            specialist_vote(spec_cat, event["category"], probs,
                            min_confidence=min_confidence)
            for spec_cat, probs in sorted(by_specialist.items())
        ]
        combined = combine(votes, strategy=strategy, event_category=event["category"])
        out.append({
            "sample_id": event["sample_id"],
            "family_id": event["family_id"],
            "category": event["category"],
            "gold_action": event["gold"]["action"] if "gold" in event else event["gold_action"],
            # A no-consensus event is scored as a miss rather than dropped: dropping it
            # would let a baseline improve its accuracy by abstaining more often.
            "predicted": combined["action"] or "no_consensus",
            "no_consensus": combined["no_consensus"],
            "n_qualified": combined["n_qualified"],
            "strategy": strategy,
        })
    return out


def abstain_audit(events: Sequence[Mapping],
                  specialist_probabilities: Mapping[str, Mapping[str, Mapping[str, float]]]
                  ) -> dict:
    """Check that out-of-expertise specialists abstain instead of voting allow."""
    total = leaked = 0
    for event in events:
        for spec_cat, probs in (specialist_probabilities.get(event["sample_id"]) or {}).items():
            if qualified(spec_cat, event["category"]):
                continue
            total += 1
            vote = specialist_vote(spec_cat, event["category"], probs)
            if not vote["abstain"]:
                leaked += 1
    return {
        "out_of_expertise_votes": total,
        "leaked_non_abstain": leaked,
        "abstain_enforced": leaked == 0,
    }
