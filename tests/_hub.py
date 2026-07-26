"""Skip helpers for tests that run offline against a locally cached Hugging Face asset.

Several suites set `HF_HUB_OFFLINE=1` deliberately -- they are meant to exercise contract
logic without touching the network. What they did not do is check that the asset they need
is actually in the cache. On a developer machine with a warm cache they pass; on a fresh
CI runner they fail with `LocalEntryNotFoundError`, which is how `make check-fast` came to
describe itself as the hermetic tier while nine of its tests depended on local cache state.

Skipping is the honest outcome: the asset is a real model checkpoint, so downloading it
would make the tier non-hermetic in the other direction. The skip reason names the asset
and how to warm the cache, so a skipped run is actionable rather than silent.
"""

from __future__ import annotations

import pytest


def require_cached_tokenizer(repo_id: str) -> None:
    """Skip the whole module unless `repo_id`'s tokenizer is already cached locally."""
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - any load failure means "not usable offline"
        pytest.skip(
            f"{repo_id} is not in the local Hugging Face cache ({type(exc).__name__}). "
            f"These tests run with HF_HUB_OFFLINE=1 by design. To run them, warm the "
            f"cache once: python -c \"from transformers import AutoTokenizer; "
            f"AutoTokenizer.from_pretrained('{repo_id}')\"",
            allow_module_level=True,
        )


def require_cached_model(repo_id: str) -> None:
    """Skip the whole module unless `repo_id`'s weights are usable offline."""
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(repo_id, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"{repo_id} weights are not in the local Hugging Face cache "
            f"({type(exc).__name__}); these tests load a real checkpoint offline.",
            allow_module_level=True,
        )
