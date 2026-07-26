#!/usr/bin/env python
"""Check that relative Markdown links in indexes and publication READMEs resolve.

Only local links are checked; external URLs are out of scope. Anchors are stripped
before resolution, so `docs/x.md#section` checks `docs/x.md`.

Usage:
    python tools/check_markdown_links.py [path ...]
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
DEFAULT_TARGETS = [
    "README.md",
    "studies/README.md",
    "papers/README.md",
    "docs/README.md",
    "docs/architecture/repository-layout-v2.md",
    "experiments/README.md",
]
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#", "data:")


def check(path: pathlib.Path) -> list[str]:
    problems: list[str] = []
    text = path.read_text(errors="replace")
    for target in LINK.findall(text):
        target = target.strip()
        if target.startswith(SKIP_PREFIXES) or not target:
            continue
        # strip anchor and any title suffix
        clean = target.split("#", 1)[0].split(" ", 1)[0]
        if not clean:
            continue
        resolved = (path.parent / clean).resolve()
        if not resolved.exists():
            problems.append(f"{path.relative_to(ROOT)}: broken link -> {target}")
    return problems


def main(argv: list[str]) -> int:
    targets = argv[1:] or DEFAULT_TARGETS
    problems: list[str] = []
    checked = 0
    for name in targets:
        path = ROOT / name
        if not path.is_file():
            print(f"SKIP  {name} (absent)")
            continue
        checked += 1
        problems += check(path)
    for problem in problems:
        print(f"ERROR {problem}")
    print(f"\nchecked {checked} file(s), {len(problems)} broken link(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
