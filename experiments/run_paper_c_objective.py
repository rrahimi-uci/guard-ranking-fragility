#!/usr/bin/env python
"""Fail-closed marker for the superseded Paper C v1 TRL runner.

The v1 script changed objective, trainer defaults, target support, learning
rate, dropout, reference handling, and data order together.  It is retained at
this path only so old notes fail clearly instead of silently launching an
invalid evidence run.

Use ``train_paper_c_dpo.py`` under the candidate v2 protocol instead.  That
implementation remains development-only until every gate in
``docs/paper-c-development-plan.md`` passes.
"""
from __future__ import annotations

import sys


MESSAGE = (
    "Paper C v1 is superseded and intentionally non-runnable. "
    "Read docs/paper-c-prereg-v2.md and use experiments/train_paper_c_dpo.py "
    "only for explicitly marked development work."
)


def main(argv: list[str] | None = None) -> int:
    del argv
    print(f"[paper-c v1] ERROR: {MESSAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
