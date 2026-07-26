#!/usr/bin/env python
"""Aggregate digest of a source tree, for verifying that a migration copied it faithfully.

Written because the first pass at this comparison hashed only non-ignored files and so
reported "one divergent file" while the new tree was in fact missing two placeholder
files. A digest whose coverage rule is implicit will eventually be trusted for something
it never looked at, so the rule is stated here and printed with every result.

Coverage rule
-------------
Include every regular file under the tree, EXCEPT:

* build residue -- ``__pycache__/``, ``*.pyc``/``*.pyo``, ``.pytest_cache/``,
  ``*.egg-info/``, ``.DS_Store``
* the *contents* of generated/local output directories -- ``artifacts/``, ``inputs/``,
  ``build/`` -- because those hold development output and licensed corpora that a
  migration deliberately does not copy

but DO include ``.gitkeep`` placeholders inside those directories, since a placeholder is
tracked source that defines the layout, not output. That single exception is what the
first pass got wrong.

Deliberately independent of ``git check-ignore``: the two trees do not have the same
ignore semantics. ``papers/paper_c/.gitignore`` excludes ``artifacts/`` as a directory,
which makes the package's own ``!artifacts/.gitkeep`` negation dead there but live in
``studies/``. A digest that inherited that asymmetry could not compare the trees at all.

Usage
-----
    python tools/tree_digest.py <tree> [<tree> ...] [--json] [--diff]

``--diff`` requires exactly two trees and lists relative paths that are missing from one
side or differ in content. ``--exclude <relpath>`` omits a path from the digest -- needed
for a manifest that records the hash of the tree it lives in, which is otherwise stale as
soon as it is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".mypy_cache", ".ruff_cache"}
SKIP_SUFFIXES = (".pyc", ".pyo")
SKIP_NAMES = {".DS_Store"}
OUTPUT_DIRS = ("artifacts", "inputs", "build")
PLACEHOLDER = ".gitkeep"

COVERAGE_RULE = (
    "all regular files except build residue (__pycache__, *.pyc, .pytest_cache, "
    "*.egg-info, .DS_Store) and except the contents of artifacts/, inputs/, build/ -- "
    "but .gitkeep placeholders inside those directories ARE included"
)


def included(root: pathlib.Path, path: pathlib.Path) -> bool:
    rel = path.relative_to(root)
    parts = rel.parts
    if any(p in SKIP_DIRS or p.endswith(".egg-info") for p in parts):
        return False
    if path.name in SKIP_NAMES or path.name.endswith(SKIP_SUFFIXES):
        return False
    if parts and parts[0] in OUTPUT_DIRS and len(parts) > 1:
        return path.name == PLACEHOLDER
    return True


def digest(root: pathlib.Path, exclude: tuple[str, ...] = ()) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    entries = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or not included(root, path):
            continue
        rel = str(path.relative_to(root))
        if rel in exclude:
            continue
        entries[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    joined = "\n".join(f"{k} {v}" for k, v in sorted(entries.items())).encode()
    out = {
        "path": str(root),
        "coverage_rule": COVERAGE_RULE,
        "file_count": len(entries),
        "aggregate_sha256": hashlib.sha256(joined).hexdigest(),
        "files": entries,
    }
    if exclude:
        out["excluded"] = list(exclude)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trees", nargs="+", type=pathlib.Path)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--diff", action="store_true", help="compare exactly two trees")
    ap.add_argument("--exclude", action="append", default=[], metavar="RELPATH",
                    help="relative path to omit from the digest; repeatable. Use this for "
                         "a manifest that records its own tree's hash, which would "
                         "otherwise be stale the moment it is written.")
    args = ap.parse_args()

    results = [digest(t, tuple(args.exclude)) for t in args.trees]

    if args.diff:
        if len(results) != 2:
            raise SystemExit("--diff requires exactly two trees")
        a, b = results
        only_a = sorted(set(a["files"]) - set(b["files"]))
        only_b = sorted(set(b["files"]) - set(a["files"]))
        differ = sorted(k for k in set(a["files"]) & set(b["files"])
                        if a["files"][k] != b["files"][k])
        report = {"coverage_rule": COVERAGE_RULE,
                  "a": {k: a[k] for k in ("path", "file_count", "aggregate_sha256")},
                  "b": {k: b[k] for k in ("path", "file_count", "aggregate_sha256")},
                  "only_in_a": only_a, "only_in_b": only_b, "content_differs": differ,
                  "identical": not (only_a or only_b or differ)}
        print(json.dumps(report, indent=2))
        return 0

    if args.json:
        print(json.dumps([{k: r[k] for k in
                           ("path", "coverage_rule", "file_count", "aggregate_sha256")}
                          for r in results], indent=2))
    else:
        print(f"coverage: {COVERAGE_RULE}\n")
        for r in results:
            print(f"{r['aggregate_sha256']}  {r['file_count']:>3} files  {r['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
