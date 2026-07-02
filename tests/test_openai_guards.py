from agent_bouncer.eval.openai_guards import moderation_to_hazard
from agent_bouncer.taxonomy import Hazard


def test_moderation_maps_flagged_category():
    assert moderation_to_hazard({"hate": True, "violence": False}) == Hazard.HATE
    assert moderation_to_hazard({"sexual/minors": True}) == Hazard.CHILD_EXPLOITATION


def test_moderation_unmapped_flag_falls_back():
    assert moderation_to_hazard({"brand_new_category": True}) == Hazard.NON_VIOLENT_CRIMES


def test_moderation_skips_unflagged_and_unknown():
    # unknown flagged category is skipped; the mapped flagged one wins
    assert moderation_to_hazard({"unknown": True, "violence": True}) == Hazard.VIOLENT_CRIMES
    # nothing flagged -> still returns a generic hazard (only called when flagged)
    assert moderation_to_hazard({"hate": False}) == Hazard.NON_VIOLENT_CRIMES
