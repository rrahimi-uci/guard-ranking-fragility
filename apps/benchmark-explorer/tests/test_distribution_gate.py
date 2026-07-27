"""Negative tests: unlicensed text must not be able to reach a public build.

Every test here is written to fail if the gate is loosened. A passing suite means the
allowlist is a positive allowlist and the failure mode is closed, not that any source
is publishable.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

APP = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "src"))

from build import (  # noqa: E402
    DistributionError, build, gather, load_ledger, publishable, strip_text,
)

FIXTURES = json.loads((APP / "fixtures/sources.json").read_text())


def test_exactly_the_reviewed_sources_are_publishable():
    """Pin the approved set. This asserted `== set()` until 2026-07-27 and failed when the
    first licence landed, which is how the change was forced to be deliberate. Pinning keeps
    the property that a second approval cannot slip in as a side effect of another edit."""
    assert publishable(load_ledger()) == {"mortgage_benchmark_v1_hmda2022"}, (
        "the publishable set changed. A new approval needs its own licence decision with a "
        "named reviewer and a data card that states the licence -- update this expectation in "
        "the same commit, and check what the public explorer build would now emit."
    )


def test_public_build_emits_no_text_for_sources_that_are_not_approved():
    """The fixtures name no approved source, so a public build must still be entirely text-free.

    This is narrower than it looks and deliberately so: it proves the allowlist withholds text
    for closed sources. It does NOT prove the allowlist would emit text for an approved one --
    that path is unexercised while no fixture names an approved source, and the assertion below
    would legitimately fail if one ever did.
    """
    ledger = load_ledger()
    approved = publishable(ledger)
    assert not (set(FIXTURES) & approved), (
        "a fixture now names an approved source, so a public build may legitimately emit its "
        "text and the text-free assertion below no longer expresses the intent -- split the "
        "fixtures into approved and closed sets instead of loosening the check"
    )
    built = gather(FIXTURES, ledger, target="public")
    blob = json.dumps(built)
    assert "SYNTHETIC FIXTURE" not in blob, "public build leaked row text"
    assert all(s["text_free"] for s in built["sections"])


def test_local_build_does_keep_text_but_is_marked_nonredistributable():
    mf = build(FIXTURES, target="local")
    assert mf["redistributable"] is False
    assert "NOT redistributable" in mf["notice"]
    body = (APP / "dist/local/index.html").read_text()
    assert "NOT REDISTRIBUTABLE" in body


def test_unknown_source_fails_closed():
    """A corpus added without a licensing decision is a build failure."""
    with pytest.raises(DistributionError, match="absent from the ledger"):
        gather({**FIXTURES, "brand_new_corpus": [{"prompt": "x"}]},
               load_ledger(), target="public")


def test_forbidden_source_is_dropped_entirely():
    ledger = load_ledger()
    for source in ledger["sources"]:
        if source["source_id"] == "toxicchat":
            source["redistribution_decision"] = "forbidden"
    built = gather(FIXTURES, ledger, target="public")
    assert "toxicchat" not in {s["source_id"] for s in built["sections"]}
    assert any(w["source_id"] == "toxicchat" and w["reason"] == "forbidden"
               for w in built["withheld"])


def test_publish_text_requires_an_affirmative_license():
    """Flipping only the decision, without the license, must not publish text."""
    ledger = load_ledger()
    for source in ledger["sources"]:
        if source["source_id"] == "toxicchat":
            source["redistribution_decision"] = "publish_text"
            # license still says permits_redistribution is not True
    assert "toxicchat" not in publishable(ledger)
    built = gather(FIXTURES, ledger, target="public")
    section = next(s for s in built["sections"] if s["source_id"] == "toxicchat")
    assert section["text_free"], "decision alone published text without a license"


def test_permissive_default_decision_is_rejected():
    ledger = load_ledger()
    ledger["default_decision"] = "publish_text"
    with pytest.raises(DistributionError, match="non-publishing"):
        gather(FIXTURES, ledger, target="public")


def test_strip_text_is_recursive_and_stable():
    row = {"prompt": "secret", "meta": {"response": "also secret"},
           "items": [{"text": "nested"}], "n": 3}
    once, twice = strip_text(row), strip_text(strip_text(row))
    assert "secret" not in json.dumps(once)
    assert "nested" not in json.dumps(once)
    assert once["n"] == 3
    assert once["items"][0]["text"].startswith("<text-free sha256:")
    assert once == twice, "stripping must be idempotent"
