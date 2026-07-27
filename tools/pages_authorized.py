#!/usr/bin/env python
"""Decide whether the HTML edition may be published, from the distribution ledger.

Serving a page on GitHub Pages is verbatim public redistribution of everything it contains.
The repository's rule is per *source*, not per file -- that distinction is the whole reason a
55 MB explorer artifact was purged from every commit -- so this asks the ledger about the
specific sources the page depends on, listed in its own PUBLICATION_REQUIREMENTS.json.

Fail-closed by construction: a missing requirements file, an unknown source id, an
unresolved decision, or a licence that does not affirmatively permit redistribution all
refuse. Silence is never consent.

Exit codes:  0 = authorized    1 = refused    2 = the check itself could not run

Usage:  python tools/pages_authorized.py [--artifact papers/unified-report-html] [--quiet]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "benchmarks/registry/distribution.yaml"
DEFAULT_ARTIFACT = "papers/unified-report-html"


def decide(artifact_dir: str) -> tuple[bool, list[str], list[str]]:
    """Return (authorized, reasons_refused, notes)."""
    refused: list[str] = []
    notes: list[str] = []

    req_path = ROOT / artifact_dir / "PUBLICATION_REQUIREMENTS.json"
    if not req_path.is_file():
        return False, [f"{req_path.relative_to(ROOT)} is missing; a publishable artifact must "
                       "declare which sources it needs approved"], notes
    req = json.loads(req_path.read_text())
    required = req.get("requires_publication_approval_for")
    if required is None:
        return False, [f"{req_path.relative_to(ROOT)} does not declare "
                       "requires_publication_approval_for"], notes

    ledger = yaml.safe_load(LEDGER.read_text())
    by_id = {s["source_id"]: s for s in ledger["sources"]}

    for source_id in required:
        src = by_id.get(source_id)
        if src is None:
            refused.append(f"{source_id} is not in the ledger at all (default: local_only)")
            continue
        decision = src.get("redistribution_decision")
        permits = src.get("license", {}).get("permits_redistribution")
        if decision != "publish_text":
            refused.append(f"{source_id}: redistribution_decision is {decision!r}, "
                           "not 'publish_text'")
        if permits is not True:
            refused.append(f"{source_id}: license.permits_redistribution is {permits!r}, "
                           "not an affirmative True")
        if decision == "publish_text" and permits is True:
            notes.append(f"{source_id}: approved for publish_text")

    if not required:
        notes.append("the artifact declares no source dependencies")
    return not refused, refused, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT,
                    help=f"artifact directory to check (default: {DEFAULT_ARTIFACT})")
    ap.add_argument("--quiet", action="store_true", help="print only the verdict line")
    args = ap.parse_args(argv)

    try:
        authorized, refused, notes = decide(args.artifact)
    except Exception as exc:                                  # noqa: BLE001
        print(f"PAGES GATE ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Refusing, because a gate that cannot evaluate itself must not permit.",
              file=sys.stderr)
        return 2

    if authorized:
        print(f"PAGES AUTHORIZED for {args.artifact}")
        if not args.quiet:
            for n in notes:
                print(f"  - {n}")
        return 0

    print(f"PAGES REFUSED for {args.artifact}")
    for r in refused:
        print(f"  - {r}")
    if not args.quiet:
        print("\nThis is the repository's standing rule, not a build error: no benchmark "
              "source is\napproved for verbatim public redistribution, and serving this page "
              "would redistribute\none. To open the gate, record the licensing decision in "
              f"{LEDGER.relative_to(ROOT)}\n(license.permits_redistribution: true and "
              "redistribution_decision: publish_text) after the\nreview that decision "
              "requires, then re-run. Nothing here decides the licence for you.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
