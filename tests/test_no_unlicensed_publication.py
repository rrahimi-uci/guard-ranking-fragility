"""Fail the build if an unlicensed bulk corpus artifact is tracked again.

A tracked, pushed 55 MB `benchmark-explorer/index.public.html` inlined 16,146 rows across
10 sources, including all 2,000 rows of `mortgage_guard_bench_2k_v0_1_0`, whose own
LICENSE file states no publication license has been selected. The rule that permitted it
lived in a `.gitignore` comment -- "Commit index.public.html (public + synthetic only)" --
which asserted a licensing conclusion per *file* while the actual decision is per
*source*. It was therefore wrong the moment a source was added, and nothing checked it.

These tests replace that comment with an executable rule. They are intentionally about
tracked bytes rather than about the build: the gated build in `apps/benchmark-explorer/`
already fails closed, but nothing stopped a person from committing an artifact produced
some other way, which is exactly how this happened.
"""

import json
import pathlib
import re
import subprocess

import pytest
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = _ROOT / "benchmarks/registry/distribution.yaml"

# Roughly a large page of text. Anything tracked above this that is HTML/JSON under a
# publication path is a bulk export, not source.
BULK_BYTES = 2_000_000

PUBLICATION_PREFIXES = ("benchmark-explorer/", "apps/benchmark-explorer/dist/", "docs/")
BULK_SUFFIXES = (".html", ".htm", ".json", ".jsonl", ".csv", ".parquet")

# Content probes: substantive words from the restricted corpora, not identifiers. A
# manifest may legitimately name a source; it may not carry the source's prose.
TEXT_PROBES = ("borrower", "loan-to-value", "underwriting", "adverse action")


_DIGESTISH = re.compile(r"^[0-9a-f]{16,}$")


def _looks_like_digest(value: str) -> bool:
    """Hashes and id strings are not prose, however long they are."""
    return bool(_DIGESTISH.match(value.strip())) or "::" in value


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=_ROOT,
                         capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split("\0") if p]


@pytest.fixture(scope="module")
def restricted_sources() -> set[str]:
    ledger = yaml.safe_load(LEDGER.read_text())
    return {
        s["source_id"] for s in ledger["sources"]
        if s["redistribution_decision"] != "publish_text"
        or s["license"]["permits_redistribution"] is not True
    }


def test_the_withdrawn_explorer_blob_is_not_tracked():
    """The specific artifact, by name, so a re-add is unambiguous."""
    tracked = tracked_files()
    assert "benchmark-explorer/index.public.html" not in tracked, (
        "benchmark-explorer/index.public.html is tracked again. It embeds 2,000 rows of "
        "mortgage_guard_bench_2k_v0_1_0, whose LICENSE_NOT_SELECTED.md states no "
        "publication license has been selected. Build via apps/benchmark-explorer/ instead."
    )
    assert "benchmark-explorer/generate.py" not in tracked, (
        "the ungated generator is back; it inlines every row of every source regardless "
        "of license. Its replacement is apps/benchmark-explorer/src/build.py."
    )


def test_no_tracked_generator_writes_an_ungated_bulk_page():
    """Catch generators by behaviour, not by filename.

    Deleting `benchmark-explorer/generate.py` retired nothing at first: the same code was
    tracked a second time as `apps/benchmark-explorer/src/generate_legacy.py`, copied there
    during the very refactor meant to gate it. A name-based check passed while the capability
    was fully intact, so this looks for what such a generator *does* -- write a
    bulk `index.public.html` -- in any tracked Python except this test.
    """
    self_rel = str(pathlib.Path(__file__).relative_to(_ROOT))
    offenders = []
    for rel in tracked_files():
        if not rel.endswith(".py") or rel == self_rel:
            continue
        path = _ROOT / rel
        if not path.is_file():
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        if "index.public.html" in body and "distribution.yaml" not in body:
            offenders.append(rel)
    assert not offenders, (
        "tracked generators emit index.public.html without consulting the distribution "
        f"ledger: {offenders}. Use apps/benchmark-explorer/src/build.py, which applies "
        "benchmarks/registry/distribution.yaml and fails closed on unknown sources."
    )


def test_no_bulk_export_is_tracked_under_a_publication_path():
    """The general rule, so the next such file is caught before it is pushed."""
    offenders = []
    for rel in tracked_files():
        if not rel.startswith(PUBLICATION_PREFIXES) or not rel.endswith(BULK_SUFFIXES):
            continue
        path = _ROOT / rel
        if path.is_file() and path.stat().st_size > BULK_BYTES:
            offenders.append(f"{rel} ({path.stat().st_size:,} bytes)")
    assert not offenders, (
        "bulk artifacts tracked under a publication path: " + "; ".join(offenders) +
        ". Publication output belongs in ignored dist/, gated by "
        "benchmarks/registry/distribution.yaml."
    )


def test_no_tracked_publication_file_carries_restricted_corpus_text(restricted_sources):
    """Size is a proxy; this checks the thing that actually matters -- the prose."""
    assert restricted_sources, "expected the ledger to restrict at least one source"
    offenders = []
    for rel in tracked_files():
        if not rel.startswith(PUBLICATION_PREFIXES) or not rel.endswith(BULK_SUFFIXES):
            continue
        path = _ROOT / rel
        if not path.is_file() or path.stat().st_size > 50_000_000:
            continue  # a file that large is already caught above; do not read it in
        body = path.read_text(encoding="utf-8", errors="replace").lower()
        hits = [p for p in TEXT_PROBES if body.count(p) >= 5]
        if hits:
            offenders.append(f"{rel} (repeated: {', '.join(hits)})")
    assert not offenders, (
        "tracked publication files carry restricted corpus prose: " + "; ".join(offenders)
    )


def test_every_tracked_corpus_appears_in_the_ledger():
    """The ledger must know about corpora tracked *outside* the explorer, too.

    `data/guard_benchmark_hard.jsonl` -- 334 rows of verbatim prompt text, force-added past
    the `/data/` ignore rule -- sat on the public remote while being absent from the ledger
    entirely. The build-time guard could not catch it: `gather()` refuses an unknown source
    when *building*, but nothing looked at corpora already committed by another route. A
    ledger with a hole in it is not a decision record.
    """
    ledger = yaml.safe_load(LEDGER.read_text())
    # A payload may name a file or a directory of splits, so match on prefix.
    known = [s["payload"]["path"].rstrip("/")
             for s in ledger["sources"] if s.get("payload", {}).get("path")]

    def in_ledger(rel: str) -> bool:
        return any(rel == k or rel.startswith(k + "/") for k in known)

    unlisted = []
    for rel in tracked_files():
        if not rel.endswith((".jsonl", ".ndjson")):
            continue
        if rel.startswith(("apps/", "tests/")) or "fixture" in rel:
            continue  # fixtures are authored for CI, not redistributed corpora
        path = _ROOT / rel
        if not path.is_file() or path.stat().st_size < 10_000 or in_ledger(rel):
            continue
        # Decide by content, not by directory name: a text-free manifest (row hash ->
        # label/provenance) redistributes nothing and needs no licensing decision. This
        # is measured rather than assumed, so a manifest that silently starts carrying
        # prompt text is caught instead of being waved through by its path.
        first = path.open(encoding="utf-8", errors="replace").readline()
        try:
            row = json.loads(first)
        except json.JSONDecodeError:
            row = {}
        prose = [k for k, v in row.items()
                 if isinstance(v, str) and len(v) > 60 and " " in v.strip()
                 and not _looks_like_digest(v)]
        if prose:
            unlisted.append(f"{rel} (text fields: {', '.join(sorted(prose))})")
    assert not unlisted, (
        "tracked corpora absent from benchmarks/registry/distribution.yaml: "
        + "; ".join(unlisted)
        + ". Every redistributed corpus needs a recorded licensing decision, including "
        "ones committed outside the explorer."
    )


def test_ledger_still_authorizes_no_verbatim_redistribution(restricted_sources):
    """If this ever fails, a licensing decision changed and this guard needs review."""
    ledger = yaml.safe_load(LEDGER.read_text())
    approved = [s["source_id"] for s in ledger["sources"]
                if s["redistribution_decision"] == "publish_text"
                and s["license"]["permits_redistribution"] is True]
    assert not approved, (
        f"sources are now approved for verbatim redistribution: {approved}. That may be "
        "correct, but this test and the explorer README both state the opposite -- update "
        "them deliberately rather than letting the claim rot."
    )
