from agent_bouncer.hardware import hardware_info, hardware_label
from agent_bouncer.models_registry import BASE_MODELS, catalog, get_base_model
from agent_bouncer.training_runner import build_config


def test_new_models_registered_with_rl():
    for key in ("deepseek-r1-1.5b", "smollm2-1.7b", "gemma-1b"):
        m = get_base_model(key)
        assert m.arch == "decoder"
        assert "sft" in m.techniques and "grpo" in m.techniques  # same FT + RL as other SLMs
    assert BASE_MODELS["gemma-1b"].gated is True


def test_catalog_shape():
    cat = catalog()
    keys = {c["key"] for c in cat}
    assert {"distilbert", "qwen3-0.6b", "deepseek-r1-1.5b", "smollm2-1.7b", "gemma-1b"} <= keys
    assert all({"hf_id", "arch", "params", "techniques", "gated"} <= set(c) for c in cat)


def test_build_config_encoder_vs_decoder():
    enc = build_config("distilbert", "sft", "train.jsonl", "out", {"epochs": 3}, 42)
    assert enc["arch"] == "encoder" and enc["train"]["epochs"] == 3
    dec = build_config("smollm2-1.7b", "sft", "train.jsonl", "out", {"max_steps": 20}, 42)
    assert dec["arch"] == "decoder" and dec["train"]["max_steps"] == 20
    grpo = build_config("deepseek-r1-1.5b", "grpo", "train.jsonl", "out", {"max_steps": 40}, 42)
    assert grpo["mode"] == "reasoning" and grpo["grpo"]["steps"] == 40


def test_hardware_info_has_expected_keys():
    info = hardware_info()
    for k in ("platform", "cpu", "cpu_count", "memory_gb", "python", "gpu"):
        assert k in info
    assert isinstance(hardware_label(info), str) and "cores" in hardware_label(info)
