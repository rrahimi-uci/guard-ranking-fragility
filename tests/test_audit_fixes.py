"""Regression tests for bugs found in the correctness audit."""

import json

import pytest


# --- #10: reasoning judge token budget scales with effort (empty verdict was mis-scored SAFE) ---
def test_reasoning_budget_scales_with_effort():
    from agent_bouncer.evaluation.openai_guards import build_chat_kwargs
    budgets = {e: build_chat_kwargs("gpt-5.2", "hi", reasoning_effort=e)["max_completion_tokens"]
               for e in ("low", "medium", "high")}
    assert budgets["low"] < budgets["medium"] < budgets["high"]
    assert budgets["high"] >= 8192
    # standard chat models keep the tight max_tokens path (no reasoning budget)
    std = build_chat_kwargs("gpt-4o-mini", "hi")
    assert std["max_tokens"] == 40 and "max_completion_tokens" not in std


# --- #4: DPO hyperparameters actually reach the trainer config ---
def test_build_config_writes_dpo_section():
    from agent_bouncer.training.runner import build_config
    cfg = build_config("qwen3-0.6b", "dpo", "data/x.jsonl", "out",
                       {"epochs": 5, "lr": 1e-5, "beta": 0.2, "max_steps": 100}, seed=1)
    # user hyperparameters take effect...
    assert cfg["dpo"]["epochs"] == 5 and cfg["dpo"]["lr"] == 1e-5
    assert cfg["dpo"]["beta"] == 0.2 and cfg["dpo"]["max_steps"] == 100
    # ...alongside MPS-safe defaults that keep the concatenated logits under INT_MAX
    assert cfg["dpo"]["batch_size"] == 2 and cfg["dpo"]["max_length"] == 1024


def test_build_config_grpo_still_correct():
    from agent_bouncer.training.runner import build_config
    cfg = build_config("qwen3-0.6b", "grpo", "data/x.jsonl", "out", {"max_steps": 30}, seed=1)
    assert cfg["mode"] == "reasoning" and cfg["grpo"]["steps"] == 30
    assert "dpo" not in cfg


# --- #1 + #12: GRPO scored in reasoning mode; 1.7B decoder labelled 1.7B ---
def test_build_commands_grpo_uses_reasoning_mode_and_correct_params(monkeypatch):
    pytest.importorskip("fastapi")  # serving.api needs the serve extra (absent in light CI)
    from agent_bouncer.serving import api
    monkeypatch.setattr(api.Path, "is_dir", lambda self: True)  # pretend checkpoints exist
    cfg = api.RunConfig(benchmarks=["xstest"],
                        guards=["decoder-grpo-0.6B", "decoder-sft-1.7B"], per_class=10)
    cmds = [" ".join(c) for c in api._build_commands(cfg)]
    grpo = next(c for c in cmds if "decoder-grpo-0.6B" in c and "eval_added_guard" in c)
    assert "--mode reasoning" in grpo
    sft17 = next(c for c in cmds if "decoder-sft-1.7B" in c and "eval_added_guard" in c)
    assert "--params 1.7B" in sft17 and "--mode sft" in sft17


# --- #2: experiment index writes are atomic and survive a corrupt index ---
def test_record_survives_corrupt_index(monkeypatch, tmp_path):
    from agent_bouncer.tracking import experiments as X
    monkeypatch.setattr(X, "EXP_DIR", str(tmp_path))
    monkeypatch.setattr(X, "INDEX", str(tmp_path / "index.json"))
    (tmp_path / "index.json").write_text("{ this is not valid json")  # simulate a truncated write
    eid = X.record(X.Experiment(id="m-sft-1", kind="train", model_key="m", technique="sft"))
    assert eid == "m-sft-1"
    idx = json.loads((tmp_path / "index.json").read_text())
    assert [e["id"] for e in idx] == ["m-sft-1"]  # recovered, not left empty
    # a second record appends rather than clobbering
    X.record(X.Experiment(id="m-eval-1", kind="eval", model_key="m", technique="eval"))
    idx = json.loads((tmp_path / "index.json").read_text())
    assert {e["id"] for e in idx} == {"m-sft-1", "m-eval-1"}


# --- #7: benchmark re-render keeps every scored guard (incl. med/high tiers + ensembles) ---
def test_render_report_keeps_all_guards_via_append():
    from agent_bouncer.evaluation.report import render_benchmark_report
    results = {"b1": {"openai-gpt-5.2-high": {"precision": .9, "recall": .9, "f1": .9,
                                              "roc_auc": .9, "fpr_on_benign": .1,
                                              "latency_p50_ms": 1, "latency_p90_ms": 2,
                                              "throughput_per_s": 1},
                      "ensemble-custom": {"precision": .8, "recall": .8, "f1": .8,
                                          "roc_auc": .8, "fpr_on_benign": .2,
                                          "latency_p50_ms": 1, "latency_p90_ms": 2,
                                          "throughput_per_s": 1}}}
    present = {g for b in results for g in results[b]}
    canon = ["keyword-baseline", "openai-gpt-5.2-low"]  # neither guard is canonical
    order = [g for g in canon if g in present] + sorted(present - set(canon))
    md = render_benchmark_report(results, {"b1": {"axis": "guardrail"}}, guard_order=order)
    assert "openai-gpt-5.2-high" in md and "ensemble-custom" in md


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


# --- deep-review round 2 fixes ---
def test_grpo_batch_is_divisible_by_num_generations():
    # TRL requires generation_batch_size (batch × grad_accum) to be a multiple of
    # num_generations; tying batch to num_generations makes every offered value valid.
    from agent_bouncer.training.runner import build_config
    for ng in (2, 4, 6, 8):
        g = build_config("qwen3-0.6b", "grpo", "t", "/o", {"num_generations": ng}, 0)["grpo"]
        assert g["num_generations"] == ng and g["batch_size"] == ng and "grad_accum" in g
        assert (g["batch_size"] * g["grad_accum"]) % ng == 0


def test_auto_test_workers_is_single_stream_by_default():
    from agent_bouncer.training.runner import _auto_test_workers
    assert _auto_test_workers("encoder", "cpu", 0) == 1   # honest single-stream latency
    assert _auto_test_workers("decoder", "mps", 0) == 1
    assert _auto_test_workers("encoder", "cpu", 4) == 4   # explicit request still honored


def test_build_config_encoder_no_validation_by_default():
    from agent_bouncer.training.runner import build_config
    assert "validation" not in build_config("distilbert", "sft", "tr.jsonl", "/o", {}, 42)["data"]
    cfg = build_config("distilbert", "sft", "tr.jsonl", "/o", {"validation": "val.jsonl"}, 42)
    assert cfg["data"]["validation"] == "val.jsonl"


# --- results-computation audit (round 3) ---------------------------------------------------

def test_beavertails_dedup_unsafe_wins():
    """A prompt that appears both safe and unsafe collapses to one UNSAFE record (unsafe wins),
    so benchmark gold labels are internally consistent (no contradictory pairs)."""
    from agent_bouncer.data.loaders import dedup_unsafe_wins
    recs = [
        {"text": "steal corn?", "label": "safe", "hazard": "none"},
        {"text": "steal corn?", "label": "unsafe", "hazard": "non_violent_crimes"},
        {"text": "hello", "label": "safe", "hazard": "none"},
        {"text": "  Steal   corn? ", "label": "safe", "hazard": "none"},  # whitespace/case dup
    ]
    out = dedup_unsafe_wins(recs)
    by = {" ".join(r["text"].lower().split()): r for r in out}
    assert len(out) == 2
    assert by["steal corn?"]["label"] == "unsafe"
    assert by["steal corn?"]["hazard"] == "non_violent_crimes"  # carries the unsafe hazard
    assert by["hello"]["label"] == "safe"


def test_training_loader_uses_disjoint_split_for_beavertails(monkeypatch):
    """The default training loader must draw BeaverTails from 30k_train, never the 30k_test pool
    the benchmark is scored on (train/eval leakage)."""
    from agent_bouncer.data import training_sets as T
    captured = {}
    import agent_bouncer.data.loaders as L
    monkeypatch.setattr(L, "load_beavertails",
                        lambda split="30k_train", **k: captured.setdefault("split", split) or [])
    T.default_training_loader("beavertails")
    assert captured["split"] == "30k_train"


def test_auc_binary_scores_equal_operating_point():
    """rank-AUC over binary (0/1) scores provably equals the operating-point estimate
    (recall+1-fpr)/2 — so unifying every row on rank-AUC changes nothing for hard-decision
    guards while giving continuous-score guards a true swept AUC."""
    from agent_bouncer.core.schema import Decision as D
    from agent_bouncer.evaluation.curves import auc_with_fallback
    from agent_bouncer.evaluation.metrics import compute_metrics
    gold = [D.UNSAFE, D.UNSAFE, D.SAFE, D.SAFE, D.UNSAFE, D.SAFE]
    pred = [D.UNSAFE, D.SAFE, D.UNSAFE, D.SAFE, D.UNSAFE, D.SAFE]
    scores = [1.0 if p == D.UNSAFE else 0.0 for p in pred]
    m = compute_metrics(gold, pred)
    a = auc_with_fallback([1 if g == D.UNSAFE else 0 for g in gold], scores,
                          recall=m.recall, fpr=m.fpr_on_benign)
    assert abs(a - (m.recall + 1 - m.fpr_on_benign) / 2) < 1e-9


def test_auc_fallback_when_single_class():
    """rank-AUC is undefined with one class present -> fall back to the operating-point value."""
    from agent_bouncer.evaluation.curves import auc_with_fallback
    a = auc_with_fallback([1, 1, 1], [1.0, 0.0, 1.0], recall=0.67, fpr=0.0)
    assert abs(a - (0.67 + 1 - 0.0) / 2) < 1e-9


def test_auc_continuous_scores_is_true_swept():
    """With continuous scores the rank-AUC is a genuine ROC-AUC (perfect ranking -> 1.0)."""
    from agent_bouncer.evaluation.curves import auc_with_fallback
    a = auc_with_fallback([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], recall=1.0, fpr=0.0)
    assert abs(a - 1.0) < 1e-9


def test_compute_curves_pr_endpoint_uses_prevalence():
    """The single-operating-point PR endpoint at recall=1 must be the class prevalence
    (n_pos/n), not a hardcoded 0.5, for imbalanced cells."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "compute_curves", "scripts/report/compute_curves.py")
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    m = {"recall": 0.5, "fpr_on_benign": 0.1, "precision": 0.8, "n": 100}
    entry = cc._derive_point(m, {"n_safe": 80, "n_unsafe": 20})   # 20% prevalence
    assert abs(entry["pr"][-1][1] - 0.2) < 1e-9
    # no meta -> recover prevalence algebraically (still not 0.5 for this imbalanced cell)
    entry2 = cc._derive_point(m, None)
    assert entry2["pr"][-1][1] != 0.5
