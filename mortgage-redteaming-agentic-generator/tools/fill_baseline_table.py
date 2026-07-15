#!/usr/bin/env python
"""Splice a committed baseline_table.json into the mortgage papers' baseline tables.

Replaces the content between <!-- BASELINE_TABLE_START --> and <!-- BASELINE_TABLE_END --> in
each target markdown file with a rendered results table. Idempotent; safe to re-run.

Usage: python fill_baseline_table.py baseline_table.json paper1.md [paper2.md ...]
"""
import json
import re
import sys

START, END = "<!-- BASELINE_TABLE_START -->", "<!-- BASELINE_TABLE_END -->"


def _fmt(x, nd=3):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def render_table(bt: dict) -> str:
    rows = bt.get("table", [])
    split = bt.get("eval_split", "public_test")
    lines = [f"*Baseline zero-shot instruction guards on the frozen benchmark ({split} split; "
             "146 rows: 75 G0/D1, 6 G1, 3 protected pairs), via `score_guards.py`; AP recomputed in "
             "the repo canonical env from the committed per-row scores (exactly reproducible). "
             "Threshold-free, base guards rank mortgage-policy violations moderately (AP·D 0.67–0.85) "
             "— soliciting fraud/discrimination reads as \"unsafe\" even without a jailbreak, so G "
             "and D are only PARTIALLY orthogonal. Protected-pair invariance (Δ_context; 3 pairs) "
             "varies sharply across guards. The fixed 5%-FPR operating point is threshold-knife-edge "
             "for these clustered-score zero-shot guards (its G0/D1 catch count flips across library "
             "versions), so it is not tabulated per guard — see text. Small-sample, LLM-judge labels "
             "— illustrative, not confirmatory.*", "",
             "| Guard | AP · G | AP · D | AP · final | Δ_context (fairness) |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['guard']} | {_fmt(r.get('AP_G'))} | {_fmt(r.get('AP_D'))} | "
                     f"{_fmt(r.get('AP_final'))} | {_fmt(r.get('delta_context'))} |")
    if bt.get("skipped"):
        names = ", ".join(s["guard"] for s in bt["skipped"])
        lines.append("")
        lines.append(f"*Not scored (unavailable in this run): {names}.*")
    return "\n".join(lines)


def main() -> int:
    bt = json.load(open(sys.argv[1]))
    block = f"{START}\n{render_table(bt)}\n{END}"
    for path in sys.argv[2:]:
        txt = open(path).read()
        new = re.sub(re.escape(START) + r".*?" + re.escape(END), block, txt, flags=re.DOTALL)
        open(path, "w").write(new)
        print(f"filled baseline table in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
