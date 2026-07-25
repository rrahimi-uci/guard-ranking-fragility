import pytest

from paper_c.contracts import ContractError
from paper_c import locks
from paper_c.locks import create_prospective_lock
from paper_c.prompting import SYSTEM_PROMPT, budgeted_prompt, select_decision_tokens


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text == " safe":
            return [11]
        if text == " unsafe":
            return [12]
        return [100 + index for index, _ in enumerate(text.split())]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"token-{value}" for value in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return f"SYSTEM: {messages[0]['content']}\nUSER: {messages[1]['content']}\nASSISTANT:"


def test_prompt_budget_preserves_classifier_wrapper():
    tokenizer = FakeTokenizer()
    prompt, metadata = budgeted_prompt(
        tokenizer,
        " ".join(f"word-{index}" for index in range(200)),
        max_length=80,
        reserved_tokens=2,
    )
    assert SYSTEM_PROMPT in prompt
    assert prompt.endswith("ASSISTANT:")
    assert metadata["truncated"] is True
    assert select_decision_tokens(tokenizer) == {
        "safe_str": " safe", "unsafe_str": " unsafe", "safe_id": 11, "unsafe_id": 12,
    }


def test_prospective_lock_is_deliberately_disabled():
    with pytest.raises(ContractError, match="sealed"):
        create_prospective_lock()


def test_source_inventory_rejects_an_unrecorded_live_file():
    inventory = locks._source_inventory()
    locks.validate_source_inventory(inventory)
    incomplete = {
        "files": inventory["files"][:-1],
        "aggregate_sha256": locks.sha256_ordered(inventory["files"][:-1]),
    }
    with pytest.raises(ContractError, match="gained, lost, or changed"):
        locks.validate_source_inventory(incomplete)


def test_historical_source_inventory_validates_without_live_rebinding():
    inventory = locks._source_inventory()
    locks.validate_recorded_source_inventory(inventory)
    inventory["aggregate_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="historical"):
        locks.validate_recorded_source_inventory(inventory)
