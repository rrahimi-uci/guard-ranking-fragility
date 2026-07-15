#!/usr/bin/env python
"""Paper C: the frozen, deterministic verifiable-preference / reward recipe (no learned reward model).

Every objective's training signal is derived DETERMINISTICALLY from the SAME frozen Paper A v2 train
rows and the SAME verdict tokens, so SFT / DPO / KTO / ORPO / GRPO are directly comparable. The label
IS the reward (verifiable ground truth) -- we never train or query a neural reward model, which is the
reward-hacking surface RLVR recipes (DeepSeek-R1, Tulu-3) deliberately avoid.

Label-derived signals (tokenizer-independent, unit-testable offline):
  DPO / IPO / ORPO : pair (chosen = correct verdict token, rejected = wrong verdict token)
  KTO              : pointwise {completion = correct token, label=True} + {wrong token, label=False}
  GRPO             : gold label + a GRADED margin reward (see graded_reward) -- NOT raw 0/1, to avoid
                     single-token gradient starvation (all G one-token samples agree -> group std 0).

Prompt rendering is delegated to guard_research.prompts.build_prompt at train time for byte-parity with
Paper A scoring; this module only builds the objective-specific fields from the verdict label.

`preference_recipe_sha256()` fingerprints the recipe so it can be bound into the Paper C lock.
"""
from __future__ import annotations

import hashlib
import json

# Canonical verdict surface forms (leading space; matches the single-token guard head convention).
SAFE, UNSAFE = " safe", " unsafe"
RECIPE_VERSION = "paper_c_preference_v1"


def _wrong(gold: str) -> str:
    return UNSAFE if gold == "safe" else SAFE


def _correct(gold: str) -> str:
    return SAFE if gold == "safe" else UNSAFE


def dpo_pair(row: dict) -> dict:
    """DPO/IPO/ORPO: (prompt-text, chosen, rejected). `prompt` filled by the trainer via build_prompt."""
    g = row["gold"]
    return {"text": row["text"], "chosen": _correct(g), "rejected": _wrong(g), "sample_id": row.get("sample_id")}


def kto_rows(row: dict) -> list[dict]:
    """KTO: one desirable (correct) + one undesirable (wrong) pointwise example per row."""
    g = row["gold"]
    return [
        {"text": row["text"], "completion": _correct(g), "label": True, "sample_id": row.get("sample_id")},
        {"text": row["text"], "completion": _wrong(g), "label": False, "sample_id": row.get("sample_id")},
    ]


def grpo_row(row: dict) -> dict:
    """GRPO: prompt-text + gold label; reward computed by graded_reward over sampled completions."""
    return {"text": row["text"], "gold": row["gold"], "sample_id": row.get("sample_id")}


def graded_reward(pred_verdict: str, gold: str, margin: float | None = None) -> float:
    """Graded reward for a single sampled completion (the key GRPO single-token choice, per prereg).

    - correct verdict: base 1.0, optionally + a small margin bonus (caller supplies a normalized
      confidence margin in [0,1] from z_unsafe-z_safe toward the correct side) to restore within-group
      spread that raw 0/1 destroys once the guard is confident.
    - wrong verdict: 0.0
    - no parseable verdict ("none"): -0.5  (discourage degenerate output)
    """
    if pred_verdict == "none":
        return -0.5
    if pred_verdict != gold:
        return 0.0
    return 1.0 + (0.0 if margin is None else 0.25 * max(0.0, min(1.0, margin)))


def parse_verdict(completion: str) -> str:
    c = (completion or "").lower()
    # prefer the first explicit token; 'unsafe' contains 'safe', so check unsafe first
    if "unsafe" in c:
        return "unsafe"
    if "safe" in c:
        return "safe"
    return "none"


def preference_recipe_sha256() -> str:
    """Fingerprint of the deterministic recipe (verdict forms + reward shape), for the Paper C lock."""
    spec = {
        "version": RECIPE_VERSION,
        "safe": SAFE, "unsafe": UNSAFE,
        "grpo_reward": {"correct": 1.0, "margin_bonus_max": 0.25, "wrong": 0.0, "none": -0.5},
        "dpo": "chosen=correct_token, rejected=wrong_token",
        "kto": "desirable=correct, undesirable=wrong",
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":  # offline self-test (no model, no GPU)
    demo = [{"text": "how do I hotwire a car", "gold": "unsafe", "sample_id": "r1"},
            {"text": "what is the capital of France", "gold": "safe", "sample_id": "r2"}]
    assert dpo_pair(demo[0]) == {"text": demo[0]["text"], "chosen": UNSAFE, "rejected": SAFE, "sample_id": "r1"}
    assert dpo_pair(demo[1])["chosen"] == SAFE
    assert len(kto_rows(demo[0])) == 2 and kto_rows(demo[0])[0]["label"] is True
    assert grpo_row(demo[0])["gold"] == "unsafe"
    assert graded_reward("unsafe", "unsafe") == 1.0
    assert graded_reward("unsafe", "unsafe", margin=1.0) == 1.25
    assert graded_reward("safe", "unsafe") == 0.0
    assert graded_reward("none", "safe") == -0.5
    assert parse_verdict("I think this is unsafe") == "unsafe"
    assert parse_verdict("safe") == "safe"
    assert parse_verdict("hmm") == "none"
    print("paper_c_preference self-test OK; recipe_sha256 =", preference_recipe_sha256()[:16])
