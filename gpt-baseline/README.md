# gpt-baseline

Frontier-model reference numbers for the guard task: **gpt-5.4** and **gpt-5.4-mini** at
**low / medium / high** reasoning effort, scored on the benchmark test sets available to
this repository that admit a binary or per-rule ground truth. Output is
[result.md](result.md) (tables) and `summary.json` (the same numbers, machine-readable).

Benchmarks, and where their rows come from:

| Benchmark | Rows | Source |
| :--- | ---: | :--- |
| 7 general-safety corpora | 10,191 | `data/benchmarks/full/` |
| `guard_benchmark_hard` | 334 | `data/guard_benchmark_hard.jsonl` |
| `mortgage_hmda2022` | 241 | `mortgage-benchmark/benchmark/v1_hmda2022/{public,private}_test` |
| `mortgage_guard_bench_2k` | 90 | `data/mortgage_guard_bench_2k_v0_1_0/splits/` (binary-decidable slice) |
| `expguard` | 2,275 | **fetched from the Hub** — finance 964 / healthcare 771 / law 540 |
| `safepyramid` | 3,000 | `data/benchmarks/safepyramid.jsonl` — implemented, not run by default |

ExpGuard is the one benchmark whose text is not on disk: it is gated, so this repo commits
only text-free artifacts for it under `artifacts/expguard_external/`. The loader fetches
the prompts with `HF_TOKEN` and uses `sha256(prompt)[:16]` row ids identical to
`experiments/eval_expguard_external.py`, so all 2,275 rows join exactly against the
committed `labels_index.json` and the four local checkpoints' scores — which is what makes
the head-to-head table in `result.md` §2 valid. It is **not** in
`benchmarks/registry/distribution.yaml`; the ledger's `local_only` default governs it, and
adding an explicit entry is a human licensing decision.

This is a *baseline*, not a study: it has no registry entry, no LOCK file, and it is not
wired into `Makefile`. It answers one question — where does an off-the-shelf frontier
model sit on these sets, and what does each reasoning tier buy?

## Run it

```bash
# full run (resumable; rerunning skips rows already predicted)
.venv/bin/python gpt-baseline/run_all.py --concurrency 200

# offline plumbing check — deterministic fake predictions, no API calls, no spend.
# Writes to raw_mock/ + result.mock.md, so it cannot contaminate real predictions.
.venv/bin/python gpt-baseline/run_all.py --mock --limit 20

# priced pilot: run 6 rows per benchmark, then project the full-run cost
.venv/bin/python gpt-baseline/run_all.py --limit 6 --estimate

# re-score and re-render from predictions already on disk (no API calls)
.venv/bin/python gpt-baseline/run_all.py --report-only

# one model, one effort, one benchmark
.venv/bin/python gpt-baseline/run_all.py --models gpt-5.4-mini --efforts low \
    --benchmarks xstest
```

`OPENAI_API_KEY` is read from the repo `.env` (never overriding a real export).

## Layout

| File | Role |
| :--- | :--- |
| `datasets.py` | Loaders → `Record(rid, benchmark, task, text, label, …)`. Also the label mappings and the skip log. `python datasets.py` prints a census. |
| `tasks.py` | The four frozen instructions and their JSON schemas, with digests. `python tasks.py` prints them. |
| `runner.py` | Async Responses-API runner: one global queue, one semaphore, per-row append, retry/backoff, latency capture. |
| `metrics.py` | Precision/recall/F1 from the hard verdict; AUROC and AP from `guard_research.metrics`. |
| `report.py` | Renders `result.md` from `summary.json`. |
| `run_all.py` | CLI orchestrator. |
| `raw/` | Per-row predictions, `{model}__{effort}__{benchmark}.jsonl`. Gitignored. |
| `raw_mock/` | Where `--mock` writes instead, so fake predictions can never be mistaken for paid ones. Gitignored. |

## Design decisions worth knowing

**Resumable, per row.** Each prediction is appended to its per-config file the instant it
lands, and the work queue is built from what is *missing* from those files. A crash, a
Ctrl-C, or a rate-limit meltdown costs only the in-flight rows. This is also how a
partial run is extended: add `--benchmarks safepyramid` later and nothing already paid
for is re-requested.

**No benchmark text is ever written.** Prediction records carry the row id, verdict, risk
score, token counts and latency — never the prompt. Most sources in
`benchmarks/registry/distribution.yaml` are `local_only` or `text_free_only`, so this
follows the same text-free convention as `experiments/eval_expguard_external.py`. `raw/`
is gitignored regardless.

**AUC comes from a self-reported score.** The Responses API exposes no token logprobs for
reasoning models, so there is no logit margin to rank on — the way
`experiments/eval_expguard_external.py` ranks local checkpoints. Instead each response
carries an integer 0–100 `risk` alongside the verdict, and AUROC is computed over that.
Integer scores tie heavily, which is exactly why AUROC and AP are imported from
`guard_research.metrics` (tie-aware, non-interpolated) rather than hand-rolled.

**Truncation is a failure, not a negative.** A reasoning model can spend its entire output
budget thinking and return `status: incomplete` with no JSON. That row is retried at
double the budget up to the task's `max_output_cap`, and only then recorded `ok: false`.
Failed rows are dropped from every metric and counted in the report's `Failed` column —
never imputed as safe, which would reward a config for timing out on hard rows.

**Reasoning tokens are the entire cost.** A general-safety prompt costs ~200 input and
~110 output tokens. A SafePyramid row costs ~3.8k input and ~5.8k output, because high
effort spends 8–13k reasoning tokens reconciling ~20 interacting rules. Use
`--limit N --estimate` before any large run: it projects per (benchmark, model, effort)
rather than scaling by call count, which would be off by ~2× on a mixed set.

## Task contracts

`prompt_safety` is `guard_research.prompts.SYSTEM_PROMPT` **verbatim** plus a JSON output
contract, so the baseline answers the same question as the repository's trained guards.
`mortgage_intervention`, `compliance_audit` and `rule_attribution` are new and specific to
this run. All four are printed in `result.md` with digests, so a later prompt edit is
detectable against a previous `summary.json`.

Positive class is uniformly **"the guard should act"**: unsafe, intervene, problematic, or
rule-violated. Label mappings per benchmark, and every row dropped at load time with its
reason, are in `result.md` §7.
