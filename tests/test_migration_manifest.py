"""Enforce the Phase 5 migration manifest instead of trusting it.

`studies/paper-c-specialize-align-mortgage-v1/provenance/MIGRATION_MANIFEST.json` asserts
that the study package is a faithful copy of `papers/paper_c/specialize_then_align` and
that the old tree is still intact. An assertion written once, by hand, decays: the first
version of that manifest claimed the trees differed by one file when the copy had in fact
dropped three layout placeholders, because the comparison it was based on could not see
files under ignored prefixes. These tests recompute the comparison from the trees on disk
so the claim either holds or the suite fails.
"""

import json
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import tree_digest  # noqa: E402

OLD = _ROOT / "papers/paper_c/specialize_then_align"
NEW = _ROOT / "studies/paper-c-specialize-align-mortgage-v1"
MANIFEST = NEW / "provenance/MIGRATION_MANIFEST.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not MANIFEST.is_file():
        pytest.skip("migration manifest absent; Phase 5 not applied in this checkout")
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def digests(manifest):
    """Tracked-only, matching the manifest: a working-tree digest is clone-unstable.

    The first version compared whatever was on disk. CI has no ignored files, so it
    computed a different hash for an unchanged tree and this test failed on a clean
    checkout -- the check was measuring local disk state, not the migration.
    """
    assert manifest["digest_mode"] == "tracked_only"
    exclude = tuple(manifest["digest_excludes"])
    return (tree_digest.digest(OLD, exclude, tracked_only=True),
            tree_digest.digest(NEW, exclude, tracked_only=True))


def test_predecessor_tree_still_exists(manifest):
    """A copy, not a move: the compatibility surface must not have been relocated."""
    assert manifest["operation"] == "copy_not_move"
    assert OLD.is_dir(), (
        "papers/paper_c/specialize_then_align is gone. Phase 5 copies; it does not move. "
        "Locks, registry entries and the unified report still reference that path."
    )
    assert (OLD / "src/paper_c_sta/contracts.py").is_file()


def test_recorded_tree_hashes_match_disk(manifest, digests):
    """The recorded aggregate hashes must be recomputable, or they are decoration."""
    old, new = digests
    assert old["aggregate_sha256"] == manifest["old_tree"]["aggregate_sha256"], (
        "predecessor tree drifted from the hash recorded at migration time; "
        f"re-run: {manifest['reverify_command']}"
    )
    assert new["aggregate_sha256"] == manifest["new_tree"]["aggregate_sha256"], (
        "study package drifted from the hash recorded at migration time; "
        f"re-run: {manifest['reverify_command']}"
    )
    assert old["file_count"] == manifest["old_tree"]["file_count"]
    assert new["file_count"] == manifest["new_tree"]["file_count"]


def test_divergence_is_exactly_what_the_manifest_declares(manifest, digests):
    """Any undeclared difference between the trees is a migration defect."""
    old, new = digests
    recorded = manifest["divergence"]

    only_old = sorted(set(old["files"]) - set(new["files"]))
    only_new = sorted(set(new["files"]) - set(old["files"]))
    differs = sorted(k for k in set(old["files"]) & set(new["files"])
                     if old["files"][k] != new["files"][k])

    assert only_old == recorded["only_in_old_tree"], (
        f"files present only in the predecessor: {only_old}. Anything here means the copy "
        "is incomplete -- which is exactly the failure the first manifest missed."
    )
    assert only_new == recorded["only_in_new_tree"], f"undeclared new files: {only_new}"
    assert differs == recorded["content_differs"], f"undeclared content drift: {differs}"


def test_no_placeholder_was_dropped_by_the_copy(digests):
    """The specific defect that slipped through: layout placeholders under ignored dirs."""
    old, new = digests
    old_keeps = {k for k in old["files"] if k.endswith(".gitkeep")}
    new_keeps = {k for k in new["files"] if k.endswith(".gitkeep")}
    assert old_keeps, "expected .gitkeep placeholders in the predecessor tree"
    assert old_keeps <= new_keeps, f"placeholders missing from the copy: {sorted(old_keeps - new_keeps)}"


def test_migration_authorizes_nothing(manifest):
    """Copying a stopped study must not upgrade its evidence or authorize a claim."""
    auth = manifest["authorization"]
    assert auth["claim_authorization"] is False
    assert auth["evidence_state"] == "development_only"


def test_both_trees_declare_the_same_expected_failure(manifest):
    """Behavioural equivalence is evidenced by identical outcomes, not by both running."""
    ver = manifest["independent_verification"]
    assert ver["identical_outcomes"] is True
    assert ver["old_tree"]["result"] == ver["new_tree"]["result"]
    assert "test_candidate_lock" in ver["declared_failure"]


def test_digest_rule_counts_placeholders_but_not_output():
    """Guard the coverage rule itself: the bug was a rule that hid the evidence."""
    assert tree_digest.included(NEW, NEW / "artifacts/.gitkeep")
    assert tree_digest.included(NEW, NEW / "src/paper_c_sta/contracts.py")
    assert not tree_digest.included(NEW, NEW / "artifacts/cells/x/result.json")
    assert not tree_digest.included(NEW, NEW / "inputs/corpus.jsonl")
    assert not tree_digest.included(NEW, NEW / "src/__pycache__/contracts.cpython-312.pyc")
