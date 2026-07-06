#!/usr/bin/env python
"""Evaluate a policy-configurable LLM judge on SafePyramid (in-context policy guardrailing).

For each item the judge is shown the POLICY (numbered rules) + the CONVERSATION and must return the
set of violated rule numbers. We score exact-set-match + rule-level micro P/R/F1, per level L0/L1/L2,
and write outputs/safepyramid_results.json.

Usage:
    python scripts/eval/run_safepyramid.py --model gpt-4o-mini --limit 60
    python scripts/eval/run_safepyramid.py --model gpt-5.2 --reasoning-effort low
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import agent_bouncer  # noqa: F401  (auto-loads .env → OPENAI_API_KEY / HF_TOKEN)
from agent_bouncer.evaluation.openai_guards import build_chat_kwargs, is_reasoning_model
from agent_bouncer.evaluation.safepyramid import build_prompt, load_safepyramid, parse_rule_ids, score

RESULTS = "outputs/safepyramid_results.json"


def _judge(client, model: str, record: dict, reasoning_effort: str | None) -> set[int]:
    system, user = build_prompt(record)
    # reuse the reasoning/chat routing, but swap in the policy-audit messages
    kwargs = build_chat_kwargs(model, user, reasoning_effort=reasoning_effort)
    kwargs["messages"] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if not is_reasoning_model(model) and reasoning_effort is None:
        kwargs["max_tokens"] = 200  # room for the JSON array of rule numbers
    try:
        resp = client.chat.completions.create(**kwargs)
        return parse_rule_ids(resp.choices[0].message.content or "", allowed=record["rule_ids"])
    except Exception as exc:  # noqa: BLE001 - one bad item shouldn't kill the run
        print(f"  !! {record['id']} failed: {type(exc).__name__}: {exc}")
        return set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--reasoning-effort", default=None, help="low|medium|high for reasoning models")
    ap.add_argument("--limit", type=int, default=None, help="cap items PER LEVEL (cost control)")
    ap.add_argument("--levels", nargs="*", default=["L0", "L1", "L2"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refresh", action="store_true", help="re-download the dataset cache")
    args = ap.parse_args()

    recs = load_safepyramid(refresh=args.refresh)
    recs = [r for r in recs if r["level"] in set(args.levels)]
    if args.limit:  # keep a balanced cap per level
        seen: dict[str, int] = {}
        capped = []
        for r in recs:
            if seen.get(r["level"], 0) < args.limit:
                capped.append(r)
                seen[r["level"]] = seen.get(r["level"], 0) + 1
        recs = capped
    if not recs:
        raise SystemExit("no SafePyramid records selected")

    from openai import OpenAI
    client = OpenAI()
    print(f"scoring {len(recs)} items with {args.model}"
          f"{' (' + args.reasoning_effort + ')' if args.reasoning_effort else ''} ...")

    preds: dict[str, set[int]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_judge, client, args.model, r, args.reasoning_effort): r for r in recs}
        done = 0
        for fut in futs:
            r = futs[fut]
            preds[r["id"]] = fut.result()
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(recs)}")

    result = score(recs, preds)
    name = f"openai-{args.model}" + (f"-{args.reasoning_effort}" if args.reasoning_effort else "")
    os.makedirs("outputs", exist_ok=True)
    blob = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {"judges": {}}
    blob.setdefault("judges", {})[name] = result
    with open(RESULTS, "w") as fh:
        json.dump(blob, fh, indent=2)

    print(f"\n== SafePyramid — {name} ==")
    print(f"{'level':8s} {'n':>5s} {'exact':>7s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    for lvl in ("overall", "L0", "L1", "L2"):
        m = result.get(lvl)
        if m:
            print(f"{lvl:8s} {m['n']:>5d} {m['exact_match']:>7.3f} "
                  f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
