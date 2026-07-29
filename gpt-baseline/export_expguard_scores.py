#!/usr/bin/env python
"""Export GPT ExpGuard predictions as text-free score files the report can reproduce from.

    .venv/bin/python gpt-baseline/export_expguard_scores.py

`gpt-baseline/raw/` is gitignored working state, so the unified report cannot regenerate a
table from it -- `reproduce.py --check` must run with no network and no GPU off committed
inputs only. This writes the same artifact the four local checkpoints already commit:

    artifacts/expguard_external/scores_<guard>.json     {row_hash: score}

Identical format, identical key space. The row hash is `sha256(prompt)[:16]`, matching
`_row_id` in `experiments/eval_expguard_external.py`, so every file in that directory --
local checkpoints and GPT configs alike -- is keyed the same way and joins to
`labels_index.json` without the prompt text ever being written.

The score is the model's self-reported 0-100 risk. Unlike the local guards' raw logit
margin it is a coarse integer (roughly 50-65 distinct values over 2,275 rows), which the
report's caption states, because it caps how finely AP can resolve the ranking.

Rows the provider refused (400 `Invalid prompt`) have no score and are simply absent, so
each guard's row count is reported alongside its metrics rather than silently imputed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import runner as rn  # noqa: E402

OUT_DIR = HERE.parent / "artifacts" / "expguard_external"

# guard key -> (model, effort). Keys are filesystem- and LaTeX-safe and sort after the
# local checkpoints, so the combined table lists local guards first.
GUARDS = {
    "gpt54_low": ("gpt-5.4", "low"),
    "gpt54_medium": ("gpt-5.4", "medium"),
    "gpt54_high": ("gpt-5.4", "high"),
    "gpt54mini_low": ("gpt-5.4-mini", "low"),
    "gpt54mini_medium": ("gpt-5.4-mini", "medium"),
    "gpt54mini_high": ("gpt-5.4-mini", "high"),
}


def export() -> dict[str, int]:
    written = {}
    for guard, (model, effort) in GUARDS.items():
        preds = rn.read_done(rn.pred_path(model, effort, "expguard"))
        scores = {
            rid: float(rec["raw"]["risk"])
            for rid, rec in preds.items()
            if rec.get("ok") and isinstance(rec.get("raw"), dict)
            and isinstance(rec["raw"].get("risk"), (int, float))
        }
        if not scores:
            continue
        path = OUT_DIR / f"scores_{guard}.json"
        # sort_keys so the committed bytes are stable across runs
        path.write_text(json.dumps(scores, sort_keys=True, separators=(",", ":")) + "\n")
        written[guard] = len(scores)
    return written


if __name__ == "__main__":
    if not OUT_DIR.is_dir():
        print(f"missing {OUT_DIR}", file=sys.stderr)
        raise SystemExit(1)
    counts = export()
    if not counts:
        print("no ExpGuard predictions found in gpt-baseline/raw/", file=sys.stderr)
        raise SystemExit(1)
    labels = json.loads((OUT_DIR / "labels_index.json").read_text())
    for guard, n in sorted(counts.items()):
        scored = json.loads((OUT_DIR / f"scores_{guard}.json").read_text())
        missing = len(labels) - len(set(labels) & set(scored))
        print(f"scores_{guard}.json  n={n:5d}  unscored_rows={missing}")
