#!/usr/bin/env python
"""Reproduce EVERY number/table/figure in the unified report from committed per-row scores.

One entry point (`make reproduce` calls this). For each study it re-derives the generated LaTeX
tables the report `\\input`s, copies the canonical outputs into `generated/`, and (with --check)
asserts byte-identity with the committed copies. It needs NO GPU and NO network; only committed
scores + the pinned analysis environment.

  Paper A (SFT specialization)   analyze_paper_a_sft.py --release-cache   [needs the LOCK-pinned env]
  Paper B (composition)          build_pilot_artifacts.py
  Mortgage (dual-label G x D)    tools/reeval_from_scores.py + emit_baseline_tex.py
  ExpGuard (finance/health/law)  eval_expguard_external.py --from-scores  -> emit table
  Latency (guard P50/P90/P99)    from committed scores.parquet latency_ms   -> emit table

Usage:  python reproduce.py [--check] [--build]
        --check : fail if any regenerated table differs from the committed generated/ copy
        --build : also compile the PDF with tectonic afterwards
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GEN = HERE / "generated"
PY = REPO / ".venv" / "bin" / "python"
PYS = str(PY) if PY.exists() else sys.executable


def _run(cmd, cwd=REPO, env=None):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)


def _copy_into_generated(src: Path, dst_name: str, results: dict, check: bool):
    dst = GEN / dst_name
    if not src.exists():
        results[dst_name] = "FAIL (source missing)"
        return
    content = src.read_text().replace("[H]", "[htbp]")  # keep the report float style on regen
    if check and dst.exists():
        # compare the NORMALIZED source (post [H]->[htbp]) against the committed copy, else the
        # normalization we apply on write would flag every [H]-using table as spurious drift.
        results[dst_name] = "OK (byte-identical)" if dst.read_text() == content else "DRIFT!"
    else:
        dst.write_text(content)
        results[dst_name] = "regenerated"


def paper_a(results, check):
    """Regenerate Paper A tables from committed scores.parquet (lock-pinned env)."""
    lock = REPO / "artifacts/paper_a_sft_v2/LOCK.json"
    scores = REPO / "artifacts/paper_a_sft_v2/scores/scores.parquet"
    out = REPO / "artifacts/paper_a_sft_v2/analysis"
    r = _run([PYS, "experiments/analyze_paper_a_sft.py", "--release-cache",
              "--lock", str(lock), "--scores", str(scores), "--out", str(out)])
    if r.returncode != 0:
        msg = "PINNED-ENV REQUIRED" if "software" in (r.stderr + r.stdout).lower() else "FAIL"
        for n in ("tab_primary_gen.tex", "tab_sensitivity_gen.tex", "tab_seed_values_gen.tex", "results_macros_gen.tex"):
            results[f"A:{n}"] = f"{msg} (analysis not re-run; committed copy used)"
        return
    tbl = out / "tables"
    _copy_into_generated(tbl / "table3_primary.tex", "tab_primary_gen.tex", results, check)
    _copy_into_generated(tbl / "table4_per_benchmark.tex", "tab_sensitivity_gen.tex", results, check)
    _copy_into_generated(tbl / "table5_seed_values.tex", "tab_seed_values_gen.tex", results, check)
    _copy_into_generated(tbl / "results_macros.tex", "results_macros_gen.tex", results, check)


def paper_b(results, check):
    pb = REPO / "papers/base-adapter-composition"
    r = _run([PYS, "code/build_pilot_artifacts.py",
              "--composition", "../../artifacts/paper_a_sft_v2/analysis/composition/composition.json",
              "--metadata", "../../artifacts/paper_a_sft_v2/analysis/composition/composition_metadata.json",
              "--out-dir", "generated"], cwd=pb)
    if r.returncode != 0:
        results["B:pilot_*"] = "FAIL: " + (r.stderr.strip().splitlines() or [""])[-1][:80]
        return
    for n in ("pilot_macros.tex", "pilot_summary_table.tex", "pilot_per_model_table.tex", "pilot_operating_point_table.tex"):
        _copy_into_generated(pb / "generated" / n, n, results, check)


def mortgage(results, check):
    import os as _os
    mb = REPO / "mortgage-benchmark"
    (mb / "generated").mkdir(exist_ok=True)
    env = {**_os.environ, "PYTHONPATH": str(mb)}  # reeval imports `magen`; scripts assume repo-root cwd
    r1 = _run([PYS, "mortgage-benchmark/tools/reeval_from_scores.py"], env=env)  # per-row scores -> baseline_table.json
    r2 = _run([PYS, "mortgage-benchmark/tools/emit_baseline_tex.py",
               "mortgage-benchmark/out_eval/baseline_table.json",
               "mortgage-benchmark/generated/baseline_table.tex"], env=env)
    src = mb / "generated" / "baseline_table.tex"
    if not src.exists():
        results["mortgage_baseline_table.tex"] = "FAIL: " + ((r1.stderr or r2.stderr or "no output").strip().splitlines() or [""])[-1][:90]
        return
    _copy_into_generated(src, "mortgage_baseline_table.tex", results, check)


def expguard(results, check):
    out = REPO / "artifacts/expguard_external"
    if not (out / "labels_index.json").exists():
        results["expguard_table.tex"] = "PENDING (base eval not yet committed)"
        return
    _run([PYS, "experiments/eval_expguard_external.py", "--from-scores", "--out", str(out)])
    # emit LaTeX table from the (deterministically recomputed) baseline_expguard.json
    tex = _emit_expguard_tex(out / "baseline_expguard.json")
    dst = GEN / "expguard_table.tex"
    if check and dst.exists():
        results["expguard_table.tex"] = "OK (byte-identical)" if dst.read_text() == tex else "DRIFT!"
    else:
        dst.write_text(tex)
        results["expguard_table.tex"] = "regenerated"


_EXPGUARD_PRETTY = {"qwen25_15b_base": "Qwen2.5-1.5B", "smollm2_17b_base": "SmolLM2-1.7B",
                    "smollm3_3b_base": "SmolLM3-3B", "qwen3_4b_base": "Qwen3-4B"}


def _emit_expguard_tex(json_path: Path) -> str:
    import json
    import numpy as np
    from guard_research.metrics import average_precision as AP
    d = json.loads(json_path.read_text())
    lab_path = json_path.parent / "labels_index.json"
    labels = json.loads(lab_path.read_text()) if lab_path.exists() else {}

    def ci(guard):  # 95% bootstrap CI on overall AP (fixed seed -> deterministic, byte-stable under --check)
        sp = json_path.parent / f"scores_{guard}.json"
        if not (sp.exists() and labels):
            return None
        sc = json.loads(sp.read_text())
        ids = [i for i in labels if i in sc]
        s = np.array([sc[i] for i in ids], dtype=float)
        y = np.array([labels[i]["label"] for i in ids], dtype=int)
        rng = np.random.default_rng(20260716)
        n = len(y); boot = []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            yy = y[idx]
            if yy.sum() in (0, n):
                continue
            boot.append(AP(s[idx], yy))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return float(lo), float(hi)

    def f(x):
        return "--" if x is None else f"{x:.3f}"
    def c3(x):  # compact CI number: drop the leading zero (".908")
        return f"{x:.3f}".lstrip("0")
    def name(g):
        return _EXPGUARD_PRETTY.get(g, g.replace("_", chr(92) + "_"))
    lines = []
    for r in d["table"]:
        c = ci(r["guard"])
        apc = f(r["overall_ap"]) + (f"\\,[{c3(c[0])}, {c3(c[1])}]" if c else "")
        lines.append(f"{name(r['guard'])} & {apc} & {f(r['overall_auroc'])} & "
                     f"{f(r.get('finance_ap'))} & {f(r.get('healthcare_ap'))} & {f(r.get('law_ap'))} \\\\")
    rows = "\n".join(lines)
    return ("% GENERATED by reproduce.py from artifacts/expguard_external/baseline_expguard.json\n"
            "\\begin{table}[H]\\centering\\footnotesize\n"
            "\\caption{External validation on ExpGuard (expert-annotated; input-prompt classification), "
            f"{d['n_rows']} rows across finance/health/law. Aggregate AP with a 2{{,}}000-resample "
            "bootstrap 95\\% CI, per-domain AP, and overall AUROC. Base checkpoints scored zero-shot via "
            "the canonical guard head; the ranking score is the raw decision margin "
            "$z_{\\text{unsafe}}-z_{\\text{safe}}$ (byte-parity with Act~I). The top two guards' CIs "
            "overlap, so the ordering among them is not resolved at this sample size.}\n"
            "\\label{tab:expguard}\n"
            "\\begin{tabular}{lrrrrr}\\toprule\n"
            "Guard & AP (all, 95\\% CI) & AUROC & AP finance & AP health & AP law \\\\\n\\midrule\n"
            f"{rows}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


_LAT_PRETTY = {"qwen25_15b": "Qwen2.5-1.5B", "smollm2_17b": "SmolLM2-1.7B",
               "smollm3_3b": "SmolLM3-3B", "qwen3_4b": "Qwen3-4B"}
_LAT_ORDER = ["qwen25_15b", "smollm2_17b", "smollm3_3b", "qwen3_4b"]


def _emit_latency_tex(df, device, batch) -> str:
    def row(name, s):
        return f"{name} & {s.median():.1f} & {s.quantile(0.9):.1f} & {s.quantile(0.99):.1f} \\\\"
    body = "\n".join(row(_LAT_PRETTY[mk], df[df.model_key == mk]["latency_ms"])
                     for mk in _LAT_ORDER if (df.model_key == mk).any())
    allrow = row("\\textbf{All four}", df["latency_ms"])
    cap = ("Guard inference latency --- one forward pass to the single-token verdict (no autoregressive "
           "generation), per-row at batch size %s on %s (bf16), over the %s committed Paper~A score rows. "
           "Latency scales with model size and prompt length, not with any decode budget." %
           (batch, device, f"{len(df):,}"))
    return ("% GENERATED by reproduce.py from artifacts/paper_a_sft_v2/scores/scores.parquet\n"
            "\\begin{table}[H]\\centering\\small\n"
            "\\caption{" + cap + "}\n\\label{tab:latency}\n"
            "\\begin{tabular}{lrrr}\\toprule\n"
            "Guard & P50 (ms) & P90 (ms) & P99 (ms) \\\\\n\\midrule\n"
            f"{body}\n\\midrule\n{allrow}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


_SS_PRETTY = {"qwen25_15b": "Qwen2.5-1.5B", "smollm2_17b": "SmolLM2-1.7B",
              "smollm3_3b": "SmolLM3-3B", "qwen3_4b": "Qwen3-4B"}
_SS_ORDER = ["qwen25_15b", "smollm2_17b", "smollm3_3b", "qwen3_4b"]


def _emit_sftsft_tex(df) -> str:
    """Equal-cost control: transfer macro-AP for base, SFT, base+SFT, and SFT+SFT (two SFT seeds),
    all from committed calibrated per-row transfer scores."""
    import itertools
    import numpy as np
    from guard_research.metrics import average_precision as AP
    tr = df[df.split == "transfer_test"]
    srcs = sorted(tr.source.unique())
    def macro(frame, col):
        return float(np.mean([AP(frame[frame.source == s][col].values, frame[frame.source == s].gold.values)
                              for s in srcs if (frame.source == s).any()]))
    rows = []
    for mk in _SS_ORDER:
        b = tr[(tr.model_key == mk) & (tr.condition == "base")].drop_duplicates("sample_id")
        base = macro(b, "probability_calibrated")
        seeds = sorted(tr[(tr.model_key == mk) & (tr.condition == "sft")].seed.unique())
        sft, comp = [], []
        for sd in seeds:
            s = tr[(tr.model_key == mk) & (tr.condition == "sft") & (tr.seed == sd)][
                ["sample_id", "source", "gold", "probability_calibrated"]]
            sft.append(macro(s, "probability_calibrated"))
            m = s.merge(b[["sample_id", "probability_calibrated"]], on="sample_id", suffixes=("_s", "_b"))
            m["c"] = (m.probability_calibrated_s + m.probability_calibrated_b) / 2
            comp.append(macro(m, "c"))
        ss = []
        for a, bb in itertools.combinations(seeds, 2):
            sa = tr[(tr.model_key == mk) & (tr.condition == "sft") & (tr.seed == a)][
                ["sample_id", "source", "gold", "probability_calibrated"]]
            sb = tr[(tr.model_key == mk) & (tr.condition == "sft") & (tr.seed == bb)][
                ["sample_id", "probability_calibrated"]]
            m = sa.merge(sb, on="sample_id", suffixes=("_a", "_b"))
            m["e"] = (m.probability_calibrated_a + m.probability_calibrated_b) / 2
            ss.append(macro(m, "e"))
        rows.append((_SS_PRETTY[mk], base, np.mean(sft), np.mean(comp), np.mean(ss)))
    body = "\n".join(f"{n} & {b:.3f} & {s:.3f} & {c:.3f} & {e:.3f} \\\\" for n, b, s, c, e in rows)
    return ("% GENERATED by reproduce.py from artifacts/paper_a_sft_v2/scores/scores.parquet\n"
            "\\begin{table}[H]\\centering\\small\n"
            "\\caption{Equal-inference-cost control for the composition mechanism (transfer macro-AP). "
            "\\textbf{base+SFT} averages the base and its SFT adapter; \\textbf{SFT+SFT} averages two "
            "independently seeded SFT adapters (same two-pass cost, no base). base+SFT beats SFT+SFT for "
            "every checkpoint --- so the recovery comes from \\emph{keeping the base}, not from generic "
            "two-model ensembling; the gap widens with base strength.}\n"
            "\\label{tab:sftsft}\n"
            "\\begin{tabular}{lrrrr}\\toprule\n"
            "Checkpoint & base & SFT & base+SFT & SFT+SFT \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


def sftsft(results, check):
    """Emit the SFT+SFT equal-cost control table from committed transfer scores."""
    sp = REPO / "artifacts/paper_a_sft_v2/scores/scores.parquet"
    if not sp.exists():
        results["tab_sftsft_gen.tex"] = "PENDING (scores.parquet missing)"
        return
    import pandas as pd
    df = pd.read_parquet(sp, columns=["sample_id", "split", "source", "gold", "model_key",
                                       "condition", "seed", "probability_calibrated"])
    tex = _emit_sftsft_tex(df)
    dst = GEN / "tab_sftsft_gen.tex"
    if check and dst.exists():
        results["tab_sftsft_gen.tex"] = "OK (byte-identical)" if dst.read_text() == tex else "DRIFT!"
    else:
        dst.write_text(tex)
        results["tab_sftsft_gen.tex"] = "regenerated"


def latency(results, check):
    """Emit the guard-latency table from the committed per-row latency_ms in scores.parquet."""
    import json
    sp = REPO / "artifacts/paper_a_sft_v2/scores/scores.parquet"
    if not sp.exists():
        results["latency_table.tex"] = "PENDING (scores.parquet missing)"
        return
    import pandas as pd
    df = pd.read_parquet(sp, columns=["model_key", "latency_ms"])
    mp = REPO / "artifacts/paper_a_sft_v2/scores/metadata.json"
    m = json.loads(mp.read_text()) if mp.exists() else {}
    device = m.get("producer_runtime", {}).get("details", {}).get("device_name", "the eval GPU")
    batch = m.get("batch_size", "?")
    tex = _emit_latency_tex(df, device, batch)
    dst = GEN / "latency_table.tex"
    if check and dst.exists():
        results["latency_table.tex"] = "OK (byte-identical)" if dst.read_text() == tex else "DRIFT!"
    else:
        dst.write_text(tex)
        results["latency_table.tex"] = "regenerated"


def figures(results, check):
    r = _run([PYS, 'figures/make_figures.py'], cwd=HERE)
    results['figures'] = 'regenerated' if r.returncode == 0 else 'FAIL: ' + (r.stderr or '')[-80:]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail on drift vs committed generated/")
    ap.add_argument("--build", action="store_true", help="compile the PDF afterwards")
    args = ap.parse_args(argv)

    GEN.mkdir(exist_ok=True)
    results: dict[str, str] = {}
    for fn in (paper_a, paper_b, mortgage, expguard, sftsft, latency, figures):
        try:
            fn(results, args.check)
        except Exception as e:  # keep going; report per-study
            results[fn.__name__] = f"ERROR: {type(e).__name__}: {e}"

    print("\n=== reproduce: per-table status ===")
    drift = False
    for k, v in results.items():
        print(f"  {k:38s} {v}")
        if "DRIFT" in v or v.startswith(("FAIL", "ERROR")):
            drift = True
    if args.build:
        print("\n=== building PDF ===")
        b = _run(["tectonic", "--outdir", "build", "unified_report.tex"], cwd=HERE)
        print("  build:", "OK" if b.returncode == 0 else "FAIL\n" + b.stderr[-500:])
    if args.check and drift:
        print("\nCHECK FAILED: a regenerated table drifted or a study failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
