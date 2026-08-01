#!/usr/bin/env python
"""Assert that an offline reconstruction of h2h.json matches the committed artifact.

Backs `make verify-heavy`. The committed artifact is written by the LIVE path (gpt-baseline/raw
plus the corpus, neither of which is in the repository); the reconstruction is written by the
OFFLINE path (the committed text-free per-row artifact plus the committed score parquet). If the
two agree on every cell and on the aggregate, then a clean checkout really can re-derive the
head-to-head numbers, which is the claim the reproducibility section makes.

`meta` is excluded from the comparison on purpose: it records which path produced the file, so
the two are *expected* to differ there and nowhere else.
"""
from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: compare_h2h.py COMMITTED.json RECONSTRUCTED.json", file=sys.stderr)
        return 2
    a, b = (json.load(open(p)) for p in argv[1:])
    bad = [k for k in ("aggregate", "sources")
           if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k), sort_keys=True)]
    if bad:
        print(f"DRIFT in {', '.join(bad)}: the offline reconstruction differs from the "
              f"committed artifact.", file=sys.stderr)
        return 1
    print("h2h.json: offline reconstruction is identical to the committed artifact "
          "(aggregate + every source cell)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
