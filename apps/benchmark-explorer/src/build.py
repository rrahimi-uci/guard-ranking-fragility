#!/usr/bin/env python
"""Ledger-driven explorer build. Fails closed on any unapproved source.

The predecessor at `benchmark-explorer/generate.py` emitted one 53 MB HTML blob with
every row of every source inlined, regardless of license. That blob is now a tracked,
pushed artifact containing 2,000 rows of a dataset whose own LICENSE file says no
publication license has been selected. This build exists so that cannot recur.

Two targets, and the difference is enforced rather than remembered:

``public``
    Emits text only for sources whose ledger entry says ``publish_text``. Every other
    source contributes counts and labels but no prompt or response text. A source
    absent from the ledger is treated as forbidden, not as permitted-by-default.

``local``
    Emits full text for local inspection. Never published, written under an ignored
    path, and stamped with a machine-readable marker asserting it is not
    redistributable.

The build refuses to run if it sees a source id that the ledger does not mention, so
adding a corpus without a licensing decision is a build failure rather than a silent
publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

APP = pathlib.Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
LEDGER = REPO / "benchmarks/registry/distribution.yaml"

TEXT_FIELDS = ("text", "prompt", "user_prompt", "request", "response",
               "candidate_response", "scenario", "title", "policy")
STRIPPED_MARKER = "<text-free sha256:"


class DistributionError(RuntimeError):
    """Raised when a build would emit text a ledger decision does not permit."""


def load_ledger() -> dict:
    import yaml

    ledger = yaml.safe_load(LEDGER.read_text())
    if ledger.get("default_decision") not in ("local_only", "text_free_only"):
        raise DistributionError(
            f"ledger default_decision must be non-publishing, got "
            f"{ledger.get('default_decision')!r}"
        )
    return ledger


def decisions(ledger: dict) -> dict[str, str]:
    return {s["source_id"]: s["redistribution_decision"] for s in ledger["sources"]}


def publishable(ledger: dict) -> set[str]:
    """Positive allowlist: only affirmatively redistributable sources."""
    return {
        s["source_id"] for s in ledger["sources"]
        if s["redistribution_decision"] == "publish_text"
        and s["license"]["permits_redistribution"] is True
    }


def strip_text(row: dict) -> dict:
    """Replace every text-bearing field with a stable hash of its content."""
    out = {}
    for key, value in row.items():
        if key in TEXT_FIELDS and isinstance(value, str) and value:
            out[key] = value if value.startswith(STRIPPED_MARKER) else (
                f"{STRIPPED_MARKER}{hashlib.sha256(value.encode()).hexdigest()[:16]}>")
        elif isinstance(value, dict):
            out[key] = strip_text(value)
        elif isinstance(value, list):
            out[key] = [strip_text(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


def gather(rows_by_source: dict[str, list[dict]], ledger: dict, *, target: str) -> dict:
    """Apply the ledger to each source and report exactly what was withheld."""
    if ledger.get("default_decision") not in ("local_only", "text_free_only"):
        raise DistributionError(
            "ledger default_decision must be non-publishing, got "
            f"{ledger.get('default_decision')!r}; validating this only in load_ledger() "
            "left the invariant bypassable by any caller holding a ledger dict"
        )
    known = decisions(ledger)
    allowed = publishable(ledger)

    unknown = sorted(set(rows_by_source) - set(known))
    if unknown:
        raise DistributionError(
            f"sources absent from the ledger cannot be built: {unknown}. "
            f"Add a decision to {LEDGER.relative_to(REPO)} first."
        )

    sections, withheld = [], []
    for source_id, rows in sorted(rows_by_source.items()):
        decision = known[source_id]
        if decision == "forbidden":
            withheld.append({"source_id": source_id, "reason": "forbidden", "rows": len(rows)})
            continue
        if target == "public" and source_id not in allowed:
            sections.append({"source_id": source_id, "text_free": True,
                             "row_count": len(rows),
                             "rows": [strip_text(r) for r in rows]})
            withheld.append({"source_id": source_id, "reason": decision, "rows": len(rows)})
            continue
        sections.append({"source_id": source_id, "text_free": False,
                         "row_count": len(rows), "rows": rows})
    return {"target": target, "sections": sections, "withheld": withheld,
            "publishable_sources": sorted(allowed)}


def manifest(built: dict, ledger: dict) -> dict:
    """Fail-closed expected section/count/hash manifest."""
    sections = [
        {"source_id": s["source_id"], "text_free": s["text_free"],
         "row_count": s["row_count"],
         "rows_sha256": hashlib.sha256(
             json.dumps(s["rows"], sort_keys=True, separators=(",", ":")).encode()
         ).hexdigest()}
        for s in built["sections"]
    ]
    return {
        "schema_version": 1,
        "target": built["target"],
        "ledger_id": ledger["ledger_id"],
        "ledger_reviewed_on": ledger["reviewed_on"],
        "sections": sections,
        "total_rows": sum(s["row_count"] for s in sections),
        "text_bearing_sources": [s["source_id"] for s in built["sections"]
                                 if not s["text_free"]],
        "withheld": built["withheld"],
        "redistributable": built["target"] == "public" and bool(built["publishable_sources"]),
        "notice": (
            "No source is approved for verbatim redistribution; this build carries "
            "text only for sources the ledger affirmatively permits."
            if built["target"] == "public"
            else "LOCAL ONLY -- contains full source text and is NOT redistributable."
        ),
    }


def source_notices(ledger: dict, built: dict) -> list[str]:
    present = {s["source_id"] for s in built["sections"]}
    return [
        f"{s['source_id']}: {s['license']['spdx_or_name']} -- "
        f"{s.get('attribution_notice') or 'no attribution notice recorded'}"
        for s in ledger["sources"] if s["source_id"] in present
    ]


def render(built: dict, mf: dict, notices: list[str]) -> str:
    banner = ("PUBLIC (allowlist-only)" if built["target"] == "public"
              else "LOCAL ONLY -- NOT REDISTRIBUTABLE")
    return f"""<!doctype html>
<meta charset="utf-8"><title>Benchmark explorer -- {built['target']}</title>
<!-- {mf['notice']} -->
<style>body{{font:14px/1.5 system-ui;margin:2rem;max-width:60rem}}
.b{{background:#fef3c7;border-left:4px solid #b7791f;padding:.75rem 1rem;margin:1rem 0}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #e5e7eb;padding:.35rem .5rem;text-align:left}}
code{{background:#f3f4f6;padding:0 .25rem}}</style>
<h1>Benchmark explorer</h1>
<div class="b"><b>{banner}</b><br>{mf['notice']}</div>
<h2>Sections</h2>
<table><tr><th>source</th><th>rows</th><th>text</th></tr>
{chr(10).join(
    f"<tr><td><code>{s['source_id']}</code></td><td>{s['row_count']}</td>"
    f"<td>{'text-free' if s['text_free'] else 'full text'}</td></tr>"
    for s in mf['sections'])}
</table>
<h2>Withheld</h2>
<ul>{''.join(f"<li><code>{w['source_id']}</code>: {w['reason']} ({w['rows']} rows)</li>"
             for w in mf['withheld']) or '<li>none</li>'}</ul>
<h2>Source notices</h2>
<ul>{''.join(f'<li>{n}</li>' for n in notices) or '<li>none</li>'}</ul>
<p>Assets are emitted per source beside this shell; this page is deliberately small.</p>
"""


def build(rows_by_source: dict[str, list[dict]], *, target: str) -> dict:
    if target not in ("public", "local"):
        raise DistributionError("target must be public or local")
    ledger = load_ledger()
    built = gather(rows_by_source, ledger, target=target)
    mf = manifest(built, ledger)
    notices = source_notices(ledger, built)

    out = APP / "dist" / target
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render(built, mf, notices))
    (out / "manifest.json").write_text(json.dumps(mf, indent=2) + "\n")
    # per-source assets keep the shell small instead of one monolithic blob
    assets = out / "assets"
    assets.mkdir(exist_ok=True)
    for section in built["sections"]:
        (assets / f"{section['source_id']}.json").write_text(
            json.dumps(section, sort_keys=True, separators=(",", ":"))
        )
    return mf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("public", "local"), required=True)
    parser.add_argument("--fixtures", action="store_true",
                        help="build from tracked CI fixtures instead of local corpora")
    args = parser.parse_args()

    if args.fixtures:
        rows_by_source = json.loads((APP / "fixtures/sources.json").read_text())
    else:
        print("full-data build requires local corpora; use --fixtures for CI",
              file=sys.stderr)
        return 2
    try:
        mf = build(rows_by_source, target=args.target)
    except DistributionError as exc:
        print(f"DISTRIBUTION ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: mf[k] for k in
                      ("target", "total_rows", "text_bearing_sources", "redistributable")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
