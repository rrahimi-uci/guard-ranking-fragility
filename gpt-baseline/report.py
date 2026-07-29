"""Render result.md from summary.json."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _f(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    try:
        if value != value:  # nan
            return "n/a"
    except TypeError:
        return str(value)
    return f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    # Pipes are padded on both sides in every row, separator included: markdownlint's
    # table-pipe-style rule flags the unpadded `|:---|---:|` form.
    if align is None:
        align = "l" + "r" * (len(header) - 1)
    sep = "| " + " | ".join(
        (":---" if a == "l" else "---:") for a in align
    ) + " |"
    out = ["| " + " | ".join(header) + " |", sep]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


def _config_label(model: str, effort: str) -> str:
    return f"{model} / {effort}"


def _expguard_comparison(summary, configs, labels, metrics_for) -> str:
    """GPT configs against this repo's four local checkpoints on identical rows."""
    base = summary["expguard_local_baseline"]
    domains = [d for d in base.get("domains", []) ]
    out = ["#### Versus this repository's local guards\n"]
    out.append(
        "The same 2,275 rows, joined by row hash. AP and AUROC only: the local "
        "checkpoints are scored by a raw logit margin with no decision threshold, so "
        "they have no precision/recall/F1 to compare against and only the ranking "
        "metrics are commensurable. Local numbers are read from "
        "`artifacts/expguard_external/baseline_expguard.json` — not recomputed here.\n")
    header = ["Guard", "AP", "AUROC"] + [f"{d} AP" for d in domains]
    rows = []
    for (model, effort), label in zip(configs, labels):
        m = metrics_for(model, effort, "expguard")
        sl = m.get("slices", {})
        rows.append([f"**{label}**", _f(m.get("ap"), 4), _f(m.get("auc"), 4)]
                    + [_f(sl.get(d, {}).get("ap"), 4) for d in domains])
    for entry in base.get("table", []):
        per = entry.get("per_domain", {})
        rows.append([entry["guard"], _f(entry.get("overall_ap"), 4),
                     _f(entry.get("overall_auroc"), 4)]
                    + [_f(per.get(d, {}).get("ap"), 4) for d in domains])
    out.append(_table(header, rows) + "\n")
    out.append(
        "Caveat on reading this as a ranking: the local checkpoints are 1.5B-4B "
        "instruction models scored by logit margin, a continuous signal with no ties. "
        "The GPT configs emit an integer 0-100 risk score, which ties heavily and caps "
        "how well AP can resolve the ranking. A GPT AP below a local checkpoint's is "
        "therefore not by itself evidence of a worse guard — part of the gap is the "
        "coarser score. The direction that *is* safe to read is a large margin, not a "
        "small one.\n")
    return "\n".join(out)


def render(summary: dict) -> str:
    configs = [(c["model"], c["effort"]) for c in summary["configs"]]
    labels = [_config_label(m, e) for m, e in configs]
    results = summary["results"]          # results[model][effort][benchmark] = metrics
    benchmarks = summary["benchmarks"]    # ordered, with group + kind
    binary = [b for b in benchmarks if b["kind"] == "binary"]
    rules = [b for b in benchmarks if b["kind"] != "binary"]

    def metrics_for(model, effort, bench):
        return results.get(model, {}).get(effort, {}).get(bench, {})

    parts: list[str] = []
    parts.append("# GPT baseline — gpt-5.4 and gpt-5.4-mini as safety guards\n")
    cum = summary.get("cumulative", {})
    last = summary["usage"]["jobs_total"]
    parts.append(
        f"Run id `{summary['run_id']}` · finished {summary['finished_at']} · "
        f"{cum.get('rows_with_a_prediction', last):,} rows scored from "
        f"{cum.get('attempts_billed', last):,} billed API calls"
        + (" (re-scored from cached predictions; no new calls)" if last == 0 else
           f" (this invocation issued {last:,} of them, at concurrency "
           f"{summary['concurrency']})" if last != cum.get("attempts_billed") else
           f" at concurrency {summary['concurrency']}")
        + (" · **MOCK RUN, no API calls**" if summary.get("mock") else "")
        + "\n"
    )
    covered = ", ".join(f"`{b['name']}`" for b in benchmarks)
    parts.append(
        "Two models × three reasoning efforts (`low`, `medium`, `high`). Positive class "
        "is uniformly *the guard should act* — unsafe, intervene, problematic, or "
        "rule-violated. Precision, recall and F1 come from the model's hard verdict; AUC "
        "is AUROC over the model's own 0–100 risk score, since the Responses API exposes "
        "no token logprobs for reasoning models.\n"
    )
    parts.append(f"**Benchmarks covered ({len(benchmarks)}):** {covered}. Anything this "
                 "run did not cover is named in §7 with the reason — read that list "
                 "before treating these as complete.\n")

    # ─────────────────────────────────────────────────────────────────── headline
    parts.append("## 1. Headline\n")
    parts.append(
        "Row-weighted micro precision/recall/F1 pooled over the "
        f"{len(binary)} binary benchmarks ({summary['pooled'][labels[0]]['n']:,} rows), "
        "plus the macro mean of their per-benchmark AUCs. Pooled AUC is deliberately "
        "not reported: pooling scores across benchmarks with different prevalences "
        "measures the mix as much as the model.\n"
    )
    rows = []
    for (model, effort), label in zip(configs, labels):
        pool = summary["pooled"][label]
        lat = summary.get("per_config", {}).get(label, {}).get("latency", {})
        rows.append([
            label, f"{pool['n']:,}", _f(pool["precision"]), _f(pool["recall"]),
            _f(pool["f1"]), _f(pool["macro_auc"]), _f(pool["accuracy"]),
            str(pool["n_failed"]), _f(lat.get("mean"), 2), _f(lat.get("p50"), 2),
            _f(lat.get("p90"), 2), _f(lat.get("p99"), 2),
        ])
    parts.append(_table(
        ["Config", "n", "Precision", "Recall", "F1", "Macro AUC", "Accuracy", "Failed",
         "mean s", "p50 s", "p90 s", "p99 s"],
        rows) + "\n")

    parts.append(
        "Latency columns are per-request seconds over **all** benchmarks for that "
        "config. See §3 for the per-benchmark breakdown and the load caveat.\n")

    blocked = {label: summary["pooled"][label].get("n_provider_blocked", 0)
               for label in labels}
    if any(blocked.values()):
        parts.append(
            "### Prompts the platform refused\n\n"
            "Some prompts never reached the model: the API returned "
            "`400 Invalid prompt: we've limited access to this content for safety "
            "reasons`. That is a **provider-level block**, not a model verdict, so those "
            "rows have no prediction and are excluded from the metrics above — they are "
            "not counted as catches. Because nearly all of them are genuine positives, "
            "the recall above is a slight *underestimate* of what the deployed system "
            "(platform filter + model) would achieve; the last column is that optimistic "
            "bound, crediting every blocked positive as caught.\n")
        rows = []
        for label in labels:
            pool = summary["pooled"][label]
            n_b = pool.get("n_provider_blocked", 0)
            n_bp = pool.get("n_provider_blocked_positive", 0)
            rows.append([
                label, str(n_b), str(n_bp),
                f"{n_bp / n_b:.2f}" if n_b else "—",
                _f(pool.get("recall")),
                _f(pool.get("recall_with_blocks_as_caught")),
            ])
        parts.append(_table(
            ["Config", "Blocked", "of which labelled unsafe", "fraction",
             "Recall (as measured)", "Recall (blocks credited)"], rows) + "\n")
        bc = summary.get("block_consistency") or {}
        if bc.get("distinct_rows_ever_blocked"):
            hist = bc.get("rows_by_config_count", {})
            spread = ", ".join(f"{v} row{'s' if v != 1 else ''} in {k}"
                               for k, v in sorted(hist.items(), key=lambda kv: int(kv[0])))
            parts.append(
                f"**The filter is not deterministic.** A block is a property of the "
                f"prompt, so a reproducible filter would refuse a given row in all "
                f"{bc['n_configs']} configs or none. In fact "
                f"{bc['distinct_rows_ever_blocked']} distinct rows were blocked at least "
                f"once, and only {bc['rows_blocked_in_every_config']} were blocked in "
                f"every config — the spread is {spread} (of {bc['n_configs']}). So the "
                "per-config `Blocked` counts above differ for reasons unrelated to the "
                "model or its reasoning effort, and differences between configs in that "
                "column should be read as noise, not behaviour.\n")
            if bc.get("by_benchmark"):
                rows = [[k, str(v)] for k, v in sorted(bc["by_benchmark"].items(),
                                                       key=lambda kv: -kv[1])]
                parts.append("Distinct rows ever blocked, by benchmark:\n")
                parts.append(_table(["Benchmark", "Rows"], rows) + "\n")

    if rules:
        parts.append(
            "SafePyramid is scored separately because its unit is a (conversation, rule) "
            "pair rather than a prompt — micro-averaged over every candidate rule.\n")
        rows = []
        for (model, effort), label in zip(configs, labels):
            m = metrics_for(model, effort, rules[0]["name"])
            if not m:
                continue
            rows.append([
                label, f"{m.get('n', 0):,}", _f(m.get("precision")), _f(m.get("recall")),
                _f(m.get("f1")), _f(m.get("auc")), _f(m.get("accuracy")),
                str(m.get("n_failed", 0)),
            ])
        parts.append(_table(
            ["Config", "(row,rule) pairs", "Precision", "Recall", "F1", "AUC",
             "Accuracy", "Failed rows"], rows) + "\n")

    # ──────────────────────────────────────────────────────── per-benchmark detail
    parts.append("## 2. Every benchmark × every config\n")
    parts.append(
        "`n` is scored rows"
        + (" (scored (row,rule) pairs for SafePyramid)" if rules else "")
        + "; `prev.` is the positive-class prevalence; `AP` is tie-aware average "
        "precision, included because it is the more informative ranking metric on the "
        "skewed sets. `Blocked` is prompts the provider refused before the model saw "
        "them.\n")
    for bench in benchmarks:
        name = bench["name"]
        rows = []
        for (model, effort), label in zip(configs, labels):
            m = metrics_for(model, effort, name)
            if not m:
                continue
            rows.append([
                label, f"{m.get('n', 0):,}", _f(m.get("prevalence"), 3),
                _f(m.get("precision")), _f(m.get("recall")), _f(m.get("f1")),
                _f(m.get("auc")), _f(m.get("ap")), _f(m.get("accuracy")),
                str(m.get("n_failed", 0)), str(m.get("n_provider_blocked", 0)),
            ])
        note = bench.get("note", "")
        parts.append(f"### {name}\n")
        parts.append(f"{note}\n" if note else "")
        parts.append(_table(
            ["Config", "n", "prev.", "Precision", "Recall", "F1", "AUC", "AP",
             "Accuracy", "Failed", "Blocked"], rows) + "\n")

        # per-slice breakdown (ExpGuard domains, SafePyramid levels)
        first = next((metrics_for(m, e, name) for m, e in configs
                      if metrics_for(m, e, name).get("slices")), {})
        if first.get("slices"):
            key = first.get("slice_key", "slice")
            slice_names = list(first["slices"])
            parts.append(f"**By `{key}`**\n")
            for metric, title in (("f1", "F1"), ("auc", "AUC"), ("ap", "AP"),
                                  ("recall", "Recall"), ("precision", "Precision")):
                rows = []
                for (model, effort), label in zip(configs, labels):
                    sl = metrics_for(model, effort, name).get("slices", {})
                    rows.append([label] + [
                        _f(sl.get(s, {}).get(metric)) for s in slice_names])
                counts = " · ".join(
                    f"{s} n={first['slices'][s].get('n', 0):,}" for s in slice_names)
                parts.append(f"*{title}* ({counts})\n")
                parts.append(_table(["Config"] + slice_names, rows) + "\n")

        if name == "expguard" and summary.get("expguard_local_baseline"):
            parts.append(_expguard_comparison(summary, configs, labels, metrics_for))

    # ──────────────────────────────────────────────────────────────────── latency
    parts.append("## 3. Latency\n")
    parts.append(
        "Per-request seconds, measured around the API call that succeeded — so the "
        "figures exclude the wait for a concurrency slot and any retry backoff, both of "
        f"which are recorded separately. **These are latencies under load**: observed "
        f"with up to {summary['concurrency']} requests in flight, not for an isolated "
        "request, so they are a throughput-regime characterisation and an upper bound on "
        "single-request latency.\n")
    rows = []
    for (model, effort), label in zip(configs, labels):
        slot = summary.get("per_config", {}).get(label, {})
        lat, queue = slot.get("latency", {}), slot.get("queue_wait", {})
        rows.append([
            label, f"{lat.get('n', 0):,}", _f(lat.get("mean"), 2), _f(lat.get("p50"), 2),
            _f(lat.get("p90"), 2), _f(lat.get("p99"), 2), _f(lat.get("max"), 1),
            _f(slot.get("mean_reasoning"), 0), _f(queue.get("p50"), 1),
        ])
    parts.append(_table(
        ["Config", "Calls", "mean s", "p50 s", "p90 s", "p99 s", "max s",
         "mean reasoning tok", "p50 queue wait s"], rows) + "\n")
    parts.append(
        "Two things to read carefully here.\n\n"
        "**`p50` is the trustworthy column.** Every config's median sits within a few "
        "hundred milliseconds of the others, and the medians order the way reasoning "
        "effort predicts. The tails do not: the whole run is one shared queue against one "
        "account, so a transient burst of connection errors or provider-side congestion "
        "lands on whichever config happens to be in flight at the time and inflates that "
        "config's p90/p99/max. Differences in the tail between configs are largely run "
        "order, not model behaviour — do not read them as a latency ranking.\n\n"
        "**`queue wait` is not a system property.** All jobs are enqueued at once and "
        "admitted 200 at a time, so a job's wait is essentially its position in a "
        "65k-deep queue; it grows monotonically through the run and says nothing about "
        "the model. It is shown only to confirm it is excluded from the latency "
        "columns.\n")

    parts.append("### Per benchmark\n")
    for stat, title in (("mean", "Mean seconds"), ("p50", "p50 seconds"),
                        ("p90", "p90 seconds"), ("p99", "p99 seconds")):
        rows = []
        for bench in benchmarks:
            row = [bench["name"]]
            for model, effort in configs:
                row.append(_f(metrics_for(model, effort, bench["name"])
                              .get("latency", {}).get(stat), 2))
            rows.append(row)
        parts.append(f"**{title}**\n")
        parts.append(_table(["Benchmark"] + labels, rows) + "\n")

    # ───────────────────────────────────────────────────────────────────── pivots
    parts.append("## 4. Side-by-side pivots\n")
    for metric, title in (("f1", "F1"), ("auc", "AUC"), ("recall", "Recall"),
                          ("precision", "Precision")):
        rows = []
        for bench in benchmarks:
            row = [bench["name"]]
            for model, effort in configs:
                row.append(_f(metrics_for(model, effort, bench["name"]).get(metric)))
            rows.append(row)
        parts.append(f"### {title}\n")
        parts.append(_table(["Benchmark"] + labels, rows) + "\n")

    # ──────────────────────────────────────────────────────────────── cost / usage
    usage = summary["usage"]
    parts.append("## 5. Cost and throughput\n")
    parts.append(
        "Cumulative across every invocation, read back from the prediction files — "
        "superseded retry attempts included, since each was separately billed. Dollar "
        "figures apply the assumed list prices in `summary.json` and are an estimate, "
        "not billing truth.\n")
    prices = summary.get("assumed_prices_per_1m", {})
    rows, grand = [], 0.0
    for model, slot in sorted(cum.get("per_model", usage["per_model"]).items()):
        p_in, p_out = prices.get(model, (0.0, 0.0))
        cost = slot["in"] / 1e6 * p_in + slot["out"] / 1e6 * p_out
        grand += cost
        rows.append([model, f"{slot['calls']:,}", f"{slot['in'] / 1e6:.2f}M",
                     f"{slot['out'] / 1e6:.2f}M", f"{slot['reasoning'] / 1e6:.2f}M",
                     f"${cost:,.2f}"])
    rows.append(["**total**", "", "", "", "", f"**${grand:,.2f}**"])
    parts.append(_table(["Model", "Calls", "Input tokens", "Output tokens",
                         "of which reasoning", "Est. cost"], rows) + "\n")
    if usage["jobs_total"]:
        parts.append(
            f"Last invocation: {usage['jobs_total']:,} calls in "
            f"{usage['wall_seconds'] / 60:.1f} min at concurrency "
            f"{summary['concurrency']} · {usage['retries']:,} retries · "
            f"{usage['jobs_failed']:,} rows still unscored after "
            f"{summary.get('max_attempts', 6)} attempts.\n")
    unscored = sum(m.get("n_failed", 0) + m.get("n_provider_blocked", 0)
                   for cfg in summary["results"].values()
                   for eff in cfg.values() for m in eff.values())
    parts.append(
        f"Across all configs, {unscored:,} of "
        f"{cum.get('rows_with_a_prediction', 0):,} row-config cells ended without a "
        "usable prediction; the breakdown by cause is in the `Failed` and `Blocked` "
        "columns of §2.\n")

    # ──────────────────────────────────────────────────────────────── method notes
    parts.append("## 6. What was asked of the model\n")
    for task, spec in summary["tasks"].items():
        parts.append(f"**`{task}`** (digest `{spec['digest']}`), used by: "
                     f"{', '.join(spec['benchmarks'])}\n")
        parts.append("```\n" + spec["instruction"] + "\n```\n")

    parts.append("## 7. Label mappings, exclusions and caveats\n")
    parts.append("\n".join(f"- {c}" for c in summary["caveats"]) + "\n")
    if summary.get("skipped"):
        parts.append("Rows dropped at load time, by reason:\n")
        rows = [[k, str(v)] for k, v in sorted(summary["skipped"].items())]
        parts.append(_table(["Reason", "Rows"], rows) + "\n")

    parts.append("## 8. Reproducing\n")
    parts.append(
        "```bash\n"
        "# full run (resumable: rerunning skips rows already in gpt-baseline/raw/)\n"
        ".venv/bin/python gpt-baseline/run_all.py --concurrency 200\n\n"
        "# offline plumbing check, no API calls\n"
        ".venv/bin/python gpt-baseline/run_all.py --mock --limit 20\n\n"
        "# rebuild this file from existing predictions, no API calls\n"
        ".venv/bin/python gpt-baseline/run_all.py --report-only\n"
        "```\n")
    parts.append(
        "Per-row predictions live in `gpt-baseline/raw/` as "
        "`{model}__{effort}__{benchmark}.jsonl`. They carry the row id, verdict, risk "
        "score and token counts and **no benchmark text**, because most sources in "
        "`benchmarks/registry/distribution.yaml` are `local_only` or "
        "`text_free_only`. The directory is gitignored anyway.\n")
    return "\n".join(parts)


def write(summary: dict, path: Path | None = None) -> Path:
    path = path or HERE / "result.md"
    path.write_text(render(summary))
    return path


if __name__ == "__main__":
    write(json.loads((HERE / "summary.json").read_text()))
    print(f"wrote {HERE / 'result.md'}")
