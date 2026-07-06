"""SafePyramid policy-guardrailing eval — offline tests for parsing, scoring, and normalization."""

import json

from agent_bouncer.evaluation.safepyramid import (
    _normalize,
    build_prompt,
    load_safepyramid,
    parse_rule_ids,
    score,
)


def test_load_safepyramid_reads_cache_without_network(tmp_path):
    cache = tmp_path / "sp.jsonl"
    rows = [{"id": "a", "level": "L0", "gold": [1], "rule_ids": [1, 2], "conversation": "c",
             "policy": "p", "domain": "fraud"},
            {"id": "b", "level": "L1", "gold": [2], "rule_ids": [2], "conversation": "c",
             "policy": "p", "domain": "privacy"}]
    cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert len(load_safepyramid(str(cache))) == 2
    assert len(load_safepyramid(str(cache), limit=1)) == 1   # limit caps the returned records


def test_parse_rule_ids_json_array():
    assert parse_rule_ids("The violations are [1, 4, 7].") == {1, 4, 7}
    assert parse_rule_ids("[]") == set()


def test_parse_rule_ids_falls_back_to_bare_integers():
    assert parse_rule_ids("rules 2 and 5 are violated") == {2, 5}


def test_parse_rule_ids_filters_to_allowed():
    # a hallucinated rule 99 (not in the policy) is dropped
    assert parse_rule_ids("[1, 4, 99]", allowed=[1, 2, 3, 4, 5]) == {1, 4}


def test_score_exact_match_and_rule_level_prf():
    recs = [
        {"id": "a", "level": "L0", "gold": [1, 2]},
        {"id": "b", "level": "L1", "gold": [3]},
    ]
    preds = {"a": {1, 2}, "b": {3, 9}}   # a exact; b has an extra FP
    out = score(recs, preds)
    assert out["overall"]["exact_match"] == 0.5          # a matched, b didn't
    assert out["L0"]["exact_match"] == 1.0 and out["L1"]["exact_match"] == 0.0
    # rule-level micro: tp=3 (1,2,3), fp=1 (9), fn=0 → P=3/4, R=1.0
    assert out["overall"]["precision"] == 0.75 and out["overall"]["recall"] == 1.0


def test_score_empty_prediction_counts_as_misses():
    recs = [{"id": "a", "level": "L0", "gold": [1, 2, 3]}]
    out = score(recs, {"a": set()})
    assert out["L0"]["exact_match"] == 0.0 and out["L0"]["recall"] == 0.0


def test_normalize_maps_row_and_computes_rule_ids():
    row = {"id": "x-1", "domain": "fraud", "level": "l1",
           "conversation": "Turn 1 - User: ...", "policy": "1. Do not...\n2. ...",
           "ground_truth_violations": [2, 1],
           "failure_mode_metadata_json": '{"distractor_rules": [3, 4]}'}
    rec = _normalize(row)
    assert rec["level"] == "L1" and rec["gold"] == [1, 2]      # sorted + deduped
    assert rec["rule_ids"] == [1, 2, 3, 4]                     # violated ∪ distractors
    # a row missing conversation/policy is dropped
    assert _normalize({"conversation": "", "policy": "x"}) is None


def test_build_prompt_includes_policy_and_conversation():
    rec = {"policy": "1. No PII.", "conversation": "User: my SSN is ..."}
    system, user = build_prompt(rec)
    assert "policy-compliance auditor" in system.lower()
    assert "1. No PII." in user and "my SSN" in user and "JSON array" in user
