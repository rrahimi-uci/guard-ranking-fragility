#!/usr/bin/env python
"""Score gpt-5.4 and gpt-5.4-mini at low/medium/high reasoning on every benchmark.

    .venv/bin/python gpt-baseline/run_all.py --concurrency 200          # the full run
    .venv/bin/python gpt-baseline/run_all.py --mock --limit 20          # offline smoke
    .venv/bin/python gpt-baseline/run_all.py --limit 8 --estimate       # priced pilot
    .venv/bin/python gpt-baseline/run_all.py --report-only              # re-render only

Resumable by construction: the runner appends each row's prediction to
``raw/{model}__{effort}__{benchmark}.jsonl`` the moment it lands, and a rerun builds its
work queue from what is *missing* from those files. Killing the run costs the in-flight
rows and nothing else.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import datasets as ds  # noqa: E402
import metrics as mx  # noqa: E402
import report as rp  # noqa: E402
import runner as rn  # noqa: E402
from tasks import TASKS, instruction_digest  # noqa: E402

DEFAULT_MODELS = ("gpt-5.4", "gpt-5.4-mini")

# Public list prices per 1M tokens, used only to print an order-of-magnitude estimate.
# Override with --price MODEL=IN,OUT. Never treat these as billing truth.
ASSUMED_PRICES = {
    "gpt-5.4": (1.25, 10.0),
    "gpt-5.4-mini": (0.25, 2.0),
}

BENCH_NOTES = {
    "mortgage_hmda2022": (
        "MortgageGuardBench v1_hmda2022, `public_test` + `private_test`. Target is "
        "`mortgage_policy_gold`: `intervene` = positive, `allow` = negative. Prompt-only "
        "— the rows' `policy_context` is a list of bare card ids with no text, so no "
        "policy is supplied to the model."
    ),
    "mortgage_guard_bench_2k": (
        "MortgageGuardBench-2K `test` split, restricted to the 90 `structured_label` "
        "rows whose gold carries a decision. `noncompliant`, `needs_human_review`, "
        "`unsafe` and `unsupported` are positive; `compliant`, `safe` and `supported` "
        "are negative. The other 210 test rows are multiple-choice action selection and "
        "numeric calculation, which have no positive class, so they are excluded rather "
        "than coerced into a label."
    ),
    "safepyramid": (
        "SafePyramid, all 3,000 test rows (1,000 conversations × levels L0/L1/L2 — rows "
        "are therefore **not independent**, and no confidence interval here accounts for "
        "that clustering). Native task: predict the net violated rule set. Scored "
        "micro-averaged over every (row, candidate rule) pair, where the candidate "
        "universe is the rule numbers enumerated in the row's own policy text."
    ),
    "expguard": (
        "ExpGuard (`6rightjade/expguardmix`, config `expguardtest`) — 2,275 "
        "expert-annotated prompts in **finance (964), healthcare (771) and law (540)**. "
        "Native `prompt_label` safe/unsafe on the prompt only, matching Paper A's "
        "formulation. Gated upstream, so its text is fetched from the Hub and never "
        "written to disk here; row ids are `sha256(prompt)[:16]`, identical to "
        "`experiments/eval_expguard_external.py`, and all 2,275 join exactly against the "
        "committed `artifacts/expguard_external/labels_index.json` with zero label or "
        "domain mismatches. These GPT numbers therefore sit on precisely the rows the "
        "four local checkpoints were scored on — see the comparison below."
    ),
    "guard_benchmark_hard": (
        "In-house mortgage / fair-lending hard negatives. `label_binary`: `flag` = "
        "positive, `allow` = negative."
    ),
}
GENERAL_NOTE = (
    "Public general-safety corpus, full split from `data/benchmarks/full/`. Native "
    "`safe` / `unsafe` prompt labels; exact-duplicate prompts de-duplicated by text hash."
)


def build_jobs(data, models, efforts, force=False):
    """Work queue = every (model, effort, row) whose prediction is not already on disk."""
    jobs, cached = [], 0
    for model in models:
        for effort in efforts:
            for bench, records in data.items():
                if not records:
                    continue
                done = {} if force else rn.read_done(rn.pred_path(model, effort, bench))
                for rec in records:
                    prev = done.get(rec.rid)
                    if prev is not None and prev.get("ok"):
                        cached += 1
                        continue
                    jobs.append(rn.Job(model=model, effort=effort, record=rec))
    return jobs, cached


def collect(data, models, efforts):
    """results[model][effort][benchmark] -> metrics, from whatever is on disk."""
    out: dict[str, dict[str, dict[str, dict]]] = {}
    for model in models:
        for effort in efforts:
            for bench, records in data.items():
                if not records:
                    continue
                preds = rn.read_done(rn.pred_path(model, effort, bench))
                scored = mx.score(records, preds)
                slice_key = ds.SLICE_BY.get(bench)
                if slice_key:
                    scored["slice_key"] = slice_key
                    scored["slices"] = mx.score_slices(records, preds, slice_key)
                out.setdefault(model, {}).setdefault(effort, {})[bench] = scored
    return out


def disk_usage(data, models, efforts) -> dict:
    """Cumulative tokens and calls across *every* invocation, read back from raw/.

    A resumed run's in-memory `Usage` only knows about the rows it personally issued, so
    reporting it as the run total understates a multi-invocation run by orders of
    magnitude. This sums every line in the prediction files, superseded attempts
    included, since each attempt was separately billed.
    """
    per_model: dict[str, dict[str, int]] = {}
    calls = rows = 0
    for model in models:
        for effort in efforts:
            for bench in data:
                path = rn.pred_path(model, effort, bench)
                if not path.is_file():
                    continue
                with path.open() as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        slot = per_model.setdefault(
                            model, {"calls": 0, "in": 0, "out": 0, "reasoning": 0})
                        slot["calls"] += 1
                        slot["in"] += int(rec.get("in_tok") or 0)
                        slot["out"] += int(rec.get("out_tok") or 0)
                        slot["reasoning"] += int(rec.get("reasoning_tok") or 0)
                        calls += 1
                rows += len(rn.read_done(path))
    return {
        "per_model": per_model,
        "attempts_billed": calls,
        "rows_with_a_prediction": rows,
        "input_tokens": sum(s["in"] for s in per_model.values()),
        "output_tokens": sum(s["out"] for s in per_model.values()),
        "reasoning_tokens": sum(s["reasoning"] for s in per_model.values()),
    }


def block_consistency(data, models, efforts) -> dict:
    """How reproducible the provider's prompt filter is across identical requests.

    A block is a property of the prompt, so a deterministic filter would block a given
    row in all six configs or none. Measuring it instead of assuming it: the histogram
    below is what licenses (or forbids) reading per-config `Blocked` counts as a signal.
    """
    n_configs = len(models) * len(efforts)
    per_row: dict[tuple[str, str], set] = {}
    for model in models:
        for effort in efforts:
            for bench in data:
                for rid, rec in rn.read_done(rn.pred_path(model, effort, bench)).items():
                    if not rec.get("ok") and mx.is_provider_block(rec):
                        per_row.setdefault((bench, rid), set()).add((model, effort))
    hist: dict[int, int] = {}
    for configs_hit in per_row.values():
        hist[len(configs_hit)] = hist.get(len(configs_hit), 0) + 1
    return {
        "n_configs": n_configs,
        "distinct_rows_ever_blocked": len(per_row),
        "rows_by_config_count": {str(k): v for k, v in sorted(hist.items())},
        "rows_blocked_in_every_config": hist.get(n_configs, 0),
        "by_benchmark": {
            bench: sum(1 for (b, _) in per_row if b == bench) for bench in sorted(
                {b for (b, _) in per_row})
        },
    }


def expguard_baseline() -> dict | None:
    """The committed local-checkpoint ExpGuard table, for side-by-side comparison.

    Written by ``experiments/eval_expguard_external.py`` for Paper A's four checkpoints
    at pinned revisions. It reports AP and AUROC only -- those guards are scored by a
    logit margin with no decision threshold, so they have no precision/recall/F1 to
    compare against, and only the ranking metrics are commensurable.
    """
    path = HERE.parent / "artifacts/expguard_external/baseline_expguard.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def per_config_stats(data, models, efforts):
    """Reasoning-token and latency aggregates per config, over every benchmark file.

    Percentiles need the raw sample, so they are computed here from the prediction
    files rather than pooled from the per-benchmark summaries (percentiles do not add).
    """
    stats = {}
    for model in models:
        for effort in efforts:
            rows = tok = 0
            latencies, queued = [], []
            for bench in data:
                for rec in rn.read_done(rn.pred_path(model, effort, bench)).values():
                    if not rec.get("ok"):
                        continue
                    rows += 1
                    tok += int(rec.get("reasoning_tok") or 0)
                    if rec.get("latency_s") is not None:
                        latencies.append(float(rec["latency_s"]))
                    if rec.get("queued_s") is not None:
                        queued.append(float(rec["queued_s"]))
            if rows:
                stats[f"{model} / {effort}"] = {
                    "rows": rows,
                    "mean_reasoning": tok / rows,
                    "latency": rn.latency_stats(latencies),
                    "queue_wait": rn.latency_stats(queued),
                }
    return stats


def price(usage: dict, prices: dict) -> tuple[float, list[str]]:
    total, lines = 0.0, []
    for model, slot in sorted(usage["per_model"].items()):
        p_in, p_out = prices.get(model, (0.0, 0.0))
        cost = slot["in"] / 1e6 * p_in + slot["out"] / 1e6 * p_out
        total += cost
        lines.append(f"    {model}: {slot['in']/1e6:.2f}M in + {slot['out']/1e6:.2f}M out "
                     f"→ ${cost:,.2f} (at ${p_in}/${p_out} per 1M)")
    return total, lines


def project_cost(benchmarks, models, efforts, prices) -> list[str]:
    """Project the full-run cost from whatever pilot predictions are on disk.

    Weighted per (benchmark, model, effort), because a flat calls-ratio extrapolation
    is badly wrong here: SafePyramid rows cost ~30x a general-safety row in input and
    ~50x in output, and a `--limit N` pilot samples every benchmark equally while the
    full run is 22% SafePyramid.
    """
    full = {b: len(rows) for b, rows in ds.load_all(benchmarks)[0].items()}
    lines, grand = [], 0.0
    for model in models:
        m_in = m_out = 0.0
        missing = []
        for effort in efforts:
            for bench, n_full in full.items():
                seen = [r for r in rn.read_done(rn.pred_path(model, effort, bench)).values()
                        if r.get("ok")]
                if not seen:
                    missing.append(f"{bench}/{effort}")
                    continue
                m_in += sum(r["in_tok"] for r in seen) / len(seen) * n_full
                m_out += sum(r["out_tok"] for r in seen) / len(seen) * n_full
        p_in, p_out = prices.get(model, (0.0, 0.0))
        cost = m_in / 1e6 * p_in + m_out / 1e6 * p_out
        grand += cost
        lines.append(f"    {model}: {m_in/1e6:.1f}M in + {m_out/1e6:.1f}M out → ${cost:,.0f}")
        if missing:
            lines.append(f"      (no pilot data for {len(missing)} cells, omitted)")
    lines.append(f"    projected full-run total ≈ ${grand:,.0f} "
                 f"({sum(full.values()) * len(models) * len(efforts):,} calls)")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--efforts", nargs="+", default=list(rn.EFFORTS),
                    choices=["low", "medium", "high"])
    ap.add_argument("--benchmarks", nargs="+", default=list(ds.ALL_BENCHMARKS))
    ap.add_argument("--limit", type=int, default=None,
                    help="first N rows per benchmark (pilot runs)")
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=600.0, help="per-request seconds")
    ap.add_argument("--mock", action="store_true",
                    help="deterministic offline predictions; no API calls")
    ap.add_argument("--force", action="store_true",
                    help="re-run rows that already have a prediction on disk")
    ap.add_argument("--report-only", action="store_true",
                    help="score what is on disk and re-render result.md")
    ap.add_argument("--estimate", action="store_true",
                    help="after running, extrapolate the full-run cost from this run")
    ap.add_argument("--price", action="append", default=[],
                    metavar="MODEL=IN,OUT", help="override assumed $/1M token prices")
    ap.add_argument("--out", type=Path, default=HERE / "result.md")
    args = ap.parse_args()

    # Repoint the prediction cache before anything reads it, so a mock run can neither
    # read real predictions nor write fake ones where a real run would trust them.
    if args.mock:
        rn.RAW = rn.RAW_MOCK
        if args.out == HERE / "result.md":
            args.out = HERE / "result.mock.md"

    prices = dict(ASSUMED_PRICES)
    for spec in args.price:
        model, _, pair = spec.partition("=")
        p_in, _, p_out = pair.partition(",")
        prices[model] = (float(p_in), float(p_out))

    print("Loading benchmarks…", flush=True)
    data, skipped = ds.load_all(args.benchmarks, limit=args.limit)
    for bench, records in data.items():
        pos = sum(1 for r in records if r.label == 1)
        extra = ""
        if records and records[0].task == "rule_attribution":
            pairs = sum(len(r.candidate_rules) for r in records)
            viol = sum(len(r.gold_rules) for r in records)
            extra = f"  ({pairs:,} (row,rule) pairs, {viol:,} violated)"
        print(f"  {bench:26s} {len(records):6,d} rows  {pos:6,d} positive{extra}")
    n_rows = sum(len(v) for v in data.values())
    n_configs = len(args.models) * len(args.efforts)
    print(f"  → {n_rows:,} rows × {n_configs} configs = {n_rows * n_configs:,} calls\n")

    usage_summary = {"jobs_total": 0, "jobs_cached_before_run": 0, "jobs_executed": 0,
                     "jobs_failed": 0, "retries": 0, "wall_seconds": 0.0,
                     "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                     "per_model": {}}

    if not args.report_only:
        jobs, cached = build_jobs(data, args.models, args.efforts, force=args.force)
        print(f"{len(jobs):,} calls to make ({cached:,} already cached on disk)")
        if jobs:
            if not args.mock:
                rn.load_env()
                import os
                if not os.environ.get("OPENAI_API_KEY"):
                    print("OPENAI_API_KEY not found in environment or .env", file=sys.stderr)
                    return 2
            started = time.time()
            usage = asyncio.run(rn.run_all(jobs, concurrency=args.concurrency,
                                           mock=args.mock, cached=cached,
                                           timeout=args.timeout))
            usage_summary = usage.summary()
            usage_summary["wall_seconds"] = round(time.time() - started, 1)
            if not args.mock:
                total, lines = price(usage_summary, prices)
                print("\n  billed tokens (assumed list prices, not billing truth):")
                print("\n".join(lines))
                print(f"    total ≈ ${total:,.2f}")
                if args.estimate:
                    print("\n  benchmark-weighted projection to the full run:")
                    print("\n".join(project_cost(args.benchmarks, args.models,
                                                 args.efforts, prices)))
        else:
            print("nothing to do; everything is cached. Scoring what is on disk.")

    print("\nScoring…", flush=True)
    results = collect(data, args.models, args.efforts)

    benchmarks = []
    for bench, records in data.items():
        if not records:
            continue
        kind = "rule_attribution" if records[0].task == "rule_attribution" else "binary"
        benchmarks.append({
            "name": bench, "kind": kind, "task": records[0].task,
            "n_rows": len(records),
            "note": BENCH_NOTES.get(bench, GENERAL_NOTE),
        })
    binary_names = [b["name"] for b in benchmarks if b["kind"] == "binary"]

    pooled = {}
    for model in args.models:
        for effort in args.efforts:
            per = {b: results[model][effort][b] for b in binary_names
                   if b in results.get(model, {}).get(effort, {})}
            per = {k: v for k, v in per.items() if v.get("kind") == "binary"}
            pooled[f"{model} / {effort}"] = mx.pooled_binary(per) if per else {"n": 0}

    tasks_used: dict[str, dict] = {}
    for b in benchmarks:
        slot = tasks_used.setdefault(b["task"], {
            "digest": instruction_digest(b["task"]),
            "instruction": TASKS[b["task"]]["instruction"],
            "benchmarks": [],
        })
        slot["benchmarks"].append(b["name"])

    caveats = [
        "Precision, recall and F1 use the model's hard verdict at its own decision "
        "point; no threshold was tuned on any of these sets. AUC is threshold-free, so "
        "a config can have a strong AUC and a weak F1 purely from where it chooses to "
        "sit on its own ROC curve — read the two together.",
        "AUC is AUROC over the model's self-reported 0–100 risk score. Reasoning models "
        "on the Responses API expose no token logprobs, so this is the only graded "
        "signal available. Integer scores tie heavily; the canonical tie-aware "
        "implementations in `guard_research.metrics` are used for both AUROC and AP.",
        "MortgageGuardBench-2K contributes only its 90 binary-decidable test rows; the "
        "other 210 are multiple-choice or numeric tasks with no positive class.",
        "Failed rows (transport errors after retries, or a response truncated by the "
        "output-token ceiling even at the task's cap) are excluded from every metric and "
        "counted in the `Failed` column. They are never imputed as negatives.",
        "`Blocked` counts prompts OpenAI's platform refused with a 400 before the model "
        "saw them. These are not model verdicts and are not scored; see the note under "
        "§1 for the effect on recall and for evidence that the filter is not "
        "deterministic across identical requests.",
        "ExpGuard is **not in `benchmarks/registry/distribution.yaml`**. The ledger's "
        "`default_decision: local_only` therefore governs it, which the text-free "
        "prediction files here comply with — but the source has no explicit reviewed "
        "entry, and adding one is a licensing decision for a human, not something this "
        "script should assume. Its prompts are fetched from the Hub at run time under "
        "`HF_TOKEN` and are never written to disk.",
        "Latency was measured at concurrency 200 against one account, so it characterises "
        "the throughput regime, not single-request latency. Medians are comparable across "
        "configs; tails are confounded by run order (see §3).",
        "Labels are taken as given. Several of these corpora have known label noise — "
        "`toxicchat` and `openai_moderation` in particular — and the mortgage sources "
        "are synthetic with LLM-judge or policy-card-consistent labels, not "
        "SME-adjudicated. These are baseline numbers, not ground-truth accuracy.",
        "The `prompt_safety` instruction is `guard_research.prompts.SYSTEM_PROMPT` "
        "verbatim plus a JSON output contract, so this baseline answers the same "
        "question as the repository's trained guards. The other three instructions are "
        "new and specific to this run; all four are printed above with digests.",
    ]
    if "safepyramid" in data and data["safepyramid"]:
        caveats += [
            "SafePyramid's AUC is derived from hard labels plus per-rule confidences: "
            "every rule the model did not list scores 0, so the negative pool is one "
            "large tie. It is comparable across configs but is not a calibrated ranking.",
            "SafePyramid's rows are conversations repeated at three dependency levels, "
            "so rows are clustered and not independent; treat per-level differences as "
            "descriptive.",
        ]
    else:
        caveats.append(
            "**SafePyramid was not run.** Its 3,000 rows average 3.8k input and 5.8k "
            "output tokens (high effort spends 8-13k reasoning tokens reconciling ~20 "
            "interacting rules), which would have been roughly 85% of the total cost of "
            "this run. Add it later with `--benchmarks safepyramid`; the cache is "
            "per-benchmark, so nothing here is re-requested.")
    if args.mock:
        caveats.insert(0, "**This is a mock run.** Predictions are deterministic "
                          "pseudo-random values seeded by row id; no model was called.")
    if args.limit:
        caveats.insert(0, f"**Partial run:** `--limit {args.limit}` capped each "
                          "benchmark, so the row counts below are not the full sets.")

    summary = {
        "run_id": hashlib.sha256(
            f"{args.models}|{args.efforts}|{args.benchmarks}|{args.limit}".encode()
        ).hexdigest()[:12],
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "mock": args.mock,
        "limit": args.limit,
        "concurrency": args.concurrency,
        "max_attempts": rn.MAX_ATTEMPTS,
        "configs": [{"model": m, "effort": e} for m in args.models for e in args.efforts],
        "benchmarks": benchmarks,
        "results": results,
        "pooled": pooled,
        "usage": usage_summary,
        "cumulative": disk_usage(data, args.models, args.efforts),
        "per_config": per_config_stats(data, args.models, args.efforts),
        "tasks": tasks_used,
        "expguard_local_baseline": expguard_baseline(),
        "block_consistency": block_consistency(data, args.models, args.efforts),
        "skipped": skipped,
        "caveats": caveats,
        "assumed_prices_per_1m": prices,
    }
    summary_path = HERE / ("summary.mock.json" if args.mock else "summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    rp.write(summary, args.out)
    print(f"wrote {args.out} and {summary_path}")

    for model in args.models:
        for effort in args.efforts:
            pool = pooled[f"{model} / {effort}"]
            if pool.get("n"):
                print(f"  {model:14s} {effort:6s} pooled binary "
                      f"P={pool['precision']:.3f} R={pool['recall']:.3f} "
                      f"F1={pool['f1']:.3f} macroAUC={pool['macro_auc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
