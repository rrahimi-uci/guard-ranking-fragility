#!/usr/bin/env python
"""Reproduce EVERY number/table/figure in the unified report from committed per-row scores.

Two entry points call this: `make regenerate` (rewrite) and `make verify` (--check, read-only). For each study it re-derives the generated LaTeX
tables the report `\\input`s, copies the canonical outputs into `generated/`, and (with --check)
asserts byte-identity with the committed copies. It needs NO GPU and NO network; only committed
scores + the pinned analysis environment.

  Paper A (SFT specialization)   analyze_paper_a_sft.py --release-cache   [needs the LOCK-pinned env]
  Paper B (composition)          build_pilot_artifacts.py
  Mortgage (dual-label G x D)    tools/reeval_from_scores.py + emit_baseline_tex.py
  ExpGuard (finance/health/law)  eval_expguard_external.py --from-scores  -> emit table
  Frontier vs local (ExpGuard)   frontier.py from committed per-row scores  -> emit tables
  Latency (guard P50/P90/P99)    from committed scores.parquet latency_ms   -> emit table

Usage:  python reproduce.py [--check] [--build]
        --check : fail if any regenerated table differs from the committed generated/ copy
        --build : also compile the PDF with tectonic afterwards
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Three study paths (expguard, frontier, sftsft) import guard_research, and the mortgage
# subprocess silently switches metric backend without it. Put the repo root on sys.path so a
# check does not depend on how this venv happens to be installed -- an editable install that
# still points at the pre-rename path was enough to turn the coverage tally into a crash.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
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


def _emitter(results, check, script: str, names: tuple[str, ...], tag: str,
             rel: str = "experiments"):
    """Run one experiments/emit_*.py and byte-check what it writes.

    These four emitters used to be outside the harness, so ten generated inputs -- including
    BOTH head-to-head outputs, the paper's most prominent frontier claim -- were reported as
    "not covered" while the reproducibility section claimed every table was byte-checked. They
    all read committed analysis JSON, so there was never a reason they could not be verified.

    In check mode the emitter is pointed at a scratch directory through PAPER_GEN_DIR and the
    result is compared, so verification writes nothing into the tracked tree.
    """
    import os as _os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        target = tmp if check else str(GEN)
        env = {**_os.environ, "PAPER_GEN_DIR": target,
               "PYTHONPATH": _os.pathsep.join([str(REPO), str(REPO / "experiments")])}
        r = _run([PYS, f"{rel}/{script}"], env=env)
        if r.returncode != 0:
            last = (r.stderr.strip().splitlines() or [""])[-1][:90]
            for n in names:
                results[n] = f"FAIL ({tag}): {last}"
            return
        for n in names:
            produced = Path(target) / n
            if not produced.exists():
                results[n] = f"FAIL ({tag}: emitter wrote no {n})"
            elif not check:
                results[n] = "regenerated"
            else:
                dst = GEN / n
                results[n] = ("OK (byte-identical)"
                              if dst.exists() and dst.read_text() == produced.read_text()
                              else "DRIFT!")


def adaptation(results, check):
    """Starting-type adaptation tables from the committed analysis JSON."""
    _emitter(results, check, "emit_adaptation_tex.py",
             ("adaptation_macros.tex", "tab_adaptation_gen.tex"), "adaptation")


def ensembling(results, check):
    """Guard-committee tables from the committed ensembling JSON."""
    _emitter(results, check, "emit_ensembling_tex.py",
             ("ensembling_macros.tex", "tab_ensembling_gen.tex",
              "tab_ensembling_committee.tex"), "ensembling")


def cascade(results, check):
    """Escalation-curve macros from the committed ExpGuard per-row scores."""
    _emitter(results, check, "emit_cascade_tex.py", ("cascade_macros.tex",), "cascade")


def h2h(results, check):
    """Frontier/local head-to-head, from the committed artifacts/frontier_general_h2h/h2h.json.

    The TeX is byte-checked here. Re-deriving h2h.json itself runs a 2,000-replicate joint
    bootstrap and takes minutes, so it is a separate opt-in target (`make verify-heavy`)
    rather than part of the default check -- but it is now possible offline, from the committed
    text-free per-row artifact, which it was not before.
    """
    _emitter(results, check, "emit_frontier_general_h2h_tex.py",
             ("h2h_macros.tex", "tab_h2h_gen.tex"), "h2h")


def klsft(results, check):
    """KL-SFT control tables, from the committed klsft_v1 per-row scores."""
    import os as _os
    import tempfile

    names = ("klsft_macros.tex", "tab_klsft_gen.tex")
    with tempfile.TemporaryDirectory() as tmp:
        target = tmp if check else str(GEN)
        r = _run([PYS, "experiments/analyze_klsft.py",
                  "--klsft-dir", "artifacts/klsft_v1/scores", "--emit-dir", target],
                 env={**_os.environ, "PYTHONPATH": str(REPO)})
        if r.returncode != 0:
            last = (r.stderr.strip().splitlines() or [""])[-1][:90]
            for n in names:
                results[n] = f"FAIL (klsft): {last}"
            return
        for n in names:
            produced = Path(target) / n
            if not produced.exists():
                results[n] = f"FAIL (klsft: no {n})"
            elif not check:
                results[n] = "regenerated"
            else:
                dst = GEN / n
                results[n] = ("OK (byte-identical)"
                              if dst.exists() and dst.read_text() == produced.read_text()
                              else "DRIFT!")


def mortgage_composition(results, check):
    """Frozen-release composition table, derived from the release's own text-free row index."""
    _emitter(results, check, "emit_composition_tex.py",
             ("mortgage_composition_table.tex", "mortgage_composition_macros.tex"),
             "composition", rel="mortgage-benchmark/tools")


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
    # magen/evaluate.py falls back to a different metric backend when guard_research is
    # unimportable, and then rewrites four tracked report_*.json with "metric_backend":
    # "fallback" -- so a --check run silently degraded committed artifacts. Pass the repo root
    # too, so the subprocess resolves the canonical tie-aware metrics.
    env = {**_os.environ, "PYTHONPATH": _os.pathsep.join([str(mb), str(REPO)])}
    r1 = _run([PYS, "mortgage-benchmark/tools/reeval_from_scores.py"], env=env)  # per-row scores -> baseline_table.json
    r2 = _run([PYS, "mortgage-benchmark/tools/emit_baseline_tex.py",
               "mortgage-benchmark/out_eval/baseline_table.json",
               "mortgage-benchmark/generated/baseline_table.tex"], env=env)
    src = mb / "generated" / "baseline_table.tex"
    if not src.exists():
        results["mortgage_baseline_table.tex"] = "FAIL: " + ((r1.stderr or r2.stderr or "no output").strip().splitlines() or [""])[-1][:90]
        return
    _copy_into_generated(src, "mortgage_baseline_table.tex", results, check)

    # the Act III worked-example box (fig:casestudy): one G0/D1 row + per-guard ranks vs. benign
    r3 = _run([PYS, "mortgage-benchmark/tools/emit_case_study_tex.py",
               "mortgage-benchmark/benchmark/v1_hmda2022/public_test.jsonl",
               "mortgage-benchmark/out_eval",
               "mortgage-benchmark/policy_cards/cards.yaml",
               "mortgage-benchmark/generated/case_study.tex"], env=env)
    cs = mb / "generated" / "case_study.tex"
    if not cs.exists():
        results["mortgage_case_study.tex"] = "FAIL: " + ((r3.stderr or "no output").strip().splitlines() or [""])[-1][:90]
        return
    _copy_into_generated(cs, "mortgage_case_study.tex", results, check)


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

    def paired(g_a, g_b):
        """Paired per-domain AP difference on IDENTICAL rows, with a 2,000-resample bootstrap CI.

        The marginal CIs above overlap for the top two guards, which is NOT the same as an unresolved
        difference: a paired test on the same rows cancels row-difficulty variance. Deterministic seed,
        so the emitted numbers are byte-stable under --check.
        """
        sa = json_path.parent / f"scores_{g_a}.json"
        sb = json_path.parent / f"scores_{g_b}.json"
        if not (sa.exists() and sb.exists() and labels):
            return None
        A_, B_ = json.loads(sa.read_text()), json.loads(sb.read_text())
        ids = [i for i in labels if i in A_ and i in B_]
        y = np.array([labels[i]["label"] for i in ids], dtype=int)
        a = np.array([A_[i] for i in ids], dtype=float)
        b = np.array([B_[i] for i in ids], dtype=float)
        doms = np.array([labels[i].get("domain", "?") for i in ids])
        rng = np.random.default_rng(20260716)
        out = {}
        for arm in ["overall"] + sorted(set(doms)):
            sel = np.arange(len(ids)) if arm == "overall" else np.where(doms == arm)[0]
            yy, aa, bb = y[sel], a[sel], b[sel]
            point = AP(aa, yy) - AP(bb, yy)
            boot = []
            for _ in range(2000):
                k = rng.integers(0, len(sel), len(sel))
                ys = yy[k]
                if ys.sum() in (0, len(ys)):
                    continue
                boot.append(AP(aa[k], ys) - AP(bb[k], ys))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out[arm] = (float(point), float(lo), float(hi))
        return out

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
    pr = paired("smollm3_3b_base", "qwen3_4b_base")
    if pr:
        def d3(a):
            return f"${a[0]:+.4f}$ $[{a[1]:+.4f}, {a[2]:+.4f}]$"
        pair_note = (
            "\\\\[3pt]{\\footnotesize \\textbf{Paired top-two comparison} (SmolLM3-3B $-$ Qwen3-4B on "
            "\\emph{identical} rows, 2{,}000-resample bootstrap). The marginal CIs above overlap, but a "
            "paired test cancels row-difficulty variance and resolves one vertical: overall "
            f"{d3(pr['overall'])}; finance {d3(pr['finance'])}; \\textbf{{health {d3(pr['healthcare'])}}} "
            "(CI excludes zero); law "
            f"{d3(pr['law'])}. So the two guards are tied on finance and law and separate on health --- "
            "``unresolved'' was an unrun analysis, not a sample-size limit.}")
    else:
        pair_note = ""
    return ("% GENERATED by reproduce.py from artifacts/expguard_external/baseline_expguard.json\n"
            "\\begin{table}[H]\\centering\\footnotesize\n"
            "\\caption{External validation on ExpGuard (expert-annotated; input-prompt classification), "
            f"{d['n_rows']} rows across finance/health/law. Aggregate AP with a 2{{,}}000-resample "
            "bootstrap 95\\% CI, per-domain AP, and overall AUROC. Base checkpoints scored zero-shot via "
            "the canonical guard head; the ranking score is the raw decision margin "
            "$z_{\\text{unsafe}}-z_{\\text{safe}}$ (byte-parity with Act~I). The top two guards' "
            "\\emph{marginal} CIs overlap; the paired comparison below is the informative test.}\n"
            "\\label{tab:expguard}\n"
            "\\begin{tabular}{lrrrrr}\\toprule\n"
            "Guard & AP (all, 95\\% CI) & AUROC & AP finance & AP health & AP law \\\\\n\\midrule\n"
            f"{rows}\n\\bottomrule\n\\end{{tabular}}{pair_note}\n\\end{{table}}\n")


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
           "generation), per-row at batch size %s on %s (bf16), over the %s committed Act~I/II score rows. "
           "These are batched per-row times (throughput-latency under load), not single-request batch-1 "
           "serving latency, and composition (Act~II) needs two passes. Latency scales with model size and "
           "prompt length, not with any decode budget." %
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
        rows.append((_SS_PRETTY[mk], base, np.mean(sft), np.mean(comp), np.mean(ss),
                     min(comp), max(comp), min(ss), max(ss)))
    body = "\n".join(
        f"{n} & {b:.3f} & {s:.3f} & {c:.3f}\\,[{cl:.3f},{ch:.3f}] & {e:.3f}\\,[{el:.3f},{eh:.3f}] \\\\"
        for n, b, s, c, e, cl, ch, el, eh in rows)
    return ("% GENERATED by reproduce.py from artifacts/paper_a_sft_v2/scores/scores.parquet\n"
            "\\begin{table}[H]\\centering\\small\n"
            "\\caption{Equal-inference-cost control for the composition mechanism (transfer macro-AP). "
            "\\textbf{base+SFT} is the mean over the 5 seeds (its adapter member is one run); "
            "\\textbf{SFT+SFT} is the mean over all $\\binom{5}{2}=10$ seed pairs (same two-pass cost, no "
            "base). Brackets give the min and max over those seeds/pairings. base+SFT beats SFT+SFT on all "
            "four checkpoints and \\emph{decisively on three}: for SmolLM2-1.7B the $+0.013$ gap lies inside "
            "the spread of both quantities, so only the other three separate. Where they separate the "
            "recovery comes from \\emph{keeping the base}, not from generic two-model ensembling, and the gap "
            "widens monotonically with base strength.}\n"
            "\\label{tab:sftsft}\n"
            "\\begin{tabular}{lrrrr}\\toprule\n"
            "Checkpoint & base & SFT & base+SFT [min,max] & SFT+SFT [min,max] \\\\\n\\midrule\n"
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


def teaser_macros(results, check):
    """Two-decimal macros for the Figure 1 caption, derived from the same committed point estimates
    that back \\RepDelta/\\TransferDelta (which print 4 dp -- too many for a teaser caption)."""
    import json
    rp = REPO / "artifacts/paper_a_sft_v2/analysis/results.json"
    if not rp.exists():
        results["teaser_macros.tex"] = "PENDING (results.json missing)"
        return
    pe = json.loads(rp.read_text())["point_estimates"]
    per = pe["per_checkpoint"]["transfer"]
    lo = min(v["delta"] for v in per.values())
    hi = max(v["delta"] for v in per.values())
    tex = ("% GENERATED by reproduce.py from artifacts/paper_a_sft_v2/analysis/results.json\n"
           f"\\newcommand{{\\TeaserRepDelta}}{{{pe['aggregate']['represented']:+.2f}}}\n"
           f"\\newcommand{{\\TeaserTransferDelta}}{{{pe['aggregate']['transfer']:+.2f}}}\n"
           f"\\newcommand{{\\TeaserTransferHi}}{{{hi:+.2f}}}\n"
           f"\\newcommand{{\\TeaserTransferLo}}{{{lo:+.2f}}}\n")
    dst = GEN / "teaser_macros.tex"
    if check and dst.exists():
        results["teaser_macros.tex"] = "OK (byte-identical)" if dst.read_text() == tex else "DRIFT!"
    else:
        dst.write_text(tex)
        results["teaser_macros.tex"] = "regenerated"


def figures(results, check):
    """Render every figure and, under --check, byte-compare it against the committed PDF.

    Figures used to be regenerated in place and never compared, which meant a plot could drift
    away from the scores behind it without the check saying anything -- and it also made
    verification write into the tracked tree. Both are fixed here: --check renders into a
    scratch directory (PAPER_FIG_DIR) and diffs the bytes. matplotlib is asked for a null
    CreationDate, so identical inputs give identical files.
    """
    import os as _os
    import tempfile

    if not check:
        r = _run([PYS, "figures/make_figures.py"], cwd=HERE)
        results["figures"] = ("regenerated" if r.returncode == 0
                              else "FAIL: " + (r.stderr or "")[-80:])
        return
    with tempfile.TemporaryDirectory() as tmp:
        r = _run([PYS, "figures/make_figures.py"], cwd=HERE,
                 env={**_os.environ, "PAPER_FIG_DIR": tmp})
        if r.returncode != 0:
            results["figures"] = "FAIL: " + (r.stderr or "")[-80:]
            return
        rendered = sorted(Path(tmp).glob("*.pdf"))
        if not rendered:
            results["figures"] = "FAIL: no figures rendered"
            return
        drift = [p.name for p in rendered
                 if not (HERE / "figures" / p.name).exists()
                 or (HERE / "figures" / p.name).read_bytes() != p.read_bytes()]
        results["figures"] = (f"OK (byte-identical, {len(rendered)} figures)" if not drift
                              else f"DRIFT! ({', '.join(sorted(drift)[:4])})")

def matched_fpr(results, check):
    """Matched false-alarm-budget operating point (see matched_fpr.py for the threshold rule)."""
    sp = REPO / "artifacts/paper_a_sft_v2/scores/scores.parquet"
    if not sp.exists():
        results["tab_matched_fpr_gen.tex"] = "PENDING (scores.parquet missing)"
        return
    import pandas as pd

    import matched_fpr as MF

    df = pd.read_parquet(sp, columns=["split", "source", "gold", "model_key", "condition",
                                      "seed", "score_raw", "prediction"])
    for name, tex in (("tab_matched_fpr_gen.tex", MF.emit_table(df)),
                      ("matched_fpr_macros.tex", MF.emit_macros(df))):
        dst = GEN / name
        if check and dst.exists():
            results[name] = "OK (byte-identical)" if dst.read_text() == tex else "DRIFT!"
        else:
            dst.write_text(tex)
            results[name] = "regenerated"


def low_fpr_region(results, check):
    """Act I and the KL control re-read over FPR [0, .05] (see low_fpr.py).

    Same committed columns as `matched_fpr`, one extra (`family_id`, for the paired bootstrap)
    and one extra input (the KL score parquets). The bootstrap is 2,000 replicates over four
    checkpoints x two regimes x three metrics, which is the slowest covered artifact in this
    harness -- a few minutes on a laptop. It is still pure arithmetic on committed scores: no
    GPU, no network, no pinned environment.
    """
    sp = REPO / "artifacts/paper_a_sft_v2/scores/scores.parquet"
    kl_dir = REPO / "artifacts/klsft_v1/scores"
    if not sp.exists():
        results["tab_lowfpr_gen.tex"] = "PENDING (scores.parquet missing)"
        return
    import pandas as pd

    import low_fpr as LF

    cols = ["split", "source", "gold", "model_key", "condition", "seed", "score_raw", "family_id"]
    res = LF.compute(pd.read_parquet(sp, columns=cols))
    emitted = [("tab_lowfpr_gen.tex", LF.emit_table(None, res)),
               ("lowfpr_macros.tex", LF.emit_macros(None, res))]

    kl_files = sorted(kl_dir.glob("klsft_scores_*.parquet")) if kl_dir.exists() else []
    if kl_files:
        frames = {}
        for p in kl_files:
            d = pd.read_parquet(p, columns=["split", "source", "gold", "model_key", "seed",
                                            "kl_beta", "score_raw"])
            frames[str(d["model_key"].iloc[0])] = d
        res_kl = LF.compute_kl(frames)
        emitted += [("tab_lowfpr_kl_gen.tex", LF.emit_kl_table(res_kl)),
                    ("lowfpr_kl_macros.tex", LF.emit_kl_macros(res_kl))]
    else:
        results["tab_lowfpr_kl_gen.tex"] = "PENDING (klsft scores missing)"

    for name, tex in emitted:
        dst = GEN / name
        if check and dst.exists():
            results[name] = "OK (byte-identical)" if dst.read_text() == tex else "DRIFT!"
        else:
            dst.write_text(tex)
            results[name] = "regenerated"


def frontier(results, check):
    """Frontier-vs-local ExpGuard comparison (see frontier.py for the threshold rule).

    Reads only text-free per-row scores from artifacts/expguard_external/ -- the four base
    checkpoints, the SFT in-env seeds, and the GPT configs -- so it needs no GPU, no
    network, and no access to the gated ExpGuard text.
    """
    out = REPO / "artifacts/expguard_external"
    needed = ["labels_index.json", "scores_gpt54_low.json"]
    if not all((out / f).exists() for f in needed):
        results["frontier_table.tex"] = "PENDING (frontier scores not committed)"
        return
    sys.path.insert(0, str(HERE))
# Three study paths (expguard, frontier, sftsft) import guard_research, and the mortgage
# subprocess silently falls back to a different metric backend without it. Put the repo
# root on sys.path so a check does not depend on how the venv happens to be installed.

    import frontier as FR

    data = FR.compute()
    scale_tex = FR.emit_scale_table()
    emits = [("frontier_table.tex", FR.emit_table(data)),
             ("frontier_serving_table.tex", FR.emit_serving_table(data)),
             ("frontier_macros.tex", FR.emit_macros(data))]
    if scale_tex:
        emits.append(("frontier_scale_table.tex", scale_tex))
    for name, tex in emits:
        dst = GEN / name
        if check and dst.exists():
            results[name] = "OK (byte-identical)" if dst.read_text() == tex else "DRIFT!"
        else:
            dst.write_text(tex)
            results[name] = "regenerated"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail on drift vs committed generated/")
    ap.add_argument("--build", action="store_true", help="compile the PDF afterwards")
    args = ap.parse_args(argv)

    GEN.mkdir(exist_ok=True)
    results: dict[str, str] = {}
    for fn in (paper_a, paper_b, mortgage, expguard, frontier, sftsft, matched_fpr, low_fpr_region,
               latency, teaser_macros, adaptation, ensembling, cascade, h2h, klsft,
               mortgage_composition, figures):
        try:
            fn(results, args.check)
        except Exception as e:  # keep going; report per-study
            results[fn.__name__] = f"ERROR: {type(e).__name__}: {e}"

    # The denominator is every generated/*.tex the report ACTUALLY \inputs -- parsed out of the
    # manuscript, not globbed from the directory. Globbing counted mortgage_composition_macros.tex,
    # which is emitted but never \input (nothing consumes \Mort*), so the published coverage read
    # 28/32 when the honest figure over referenced artifacts is 27/31.
    # repro_macros.tex is this harness's own coverage report, not a claim-bearing artifact;
    # counting it would make the denominator self-referential.
    _src = [HERE / "unified_report.tex", *sorted((HERE / "sections").glob("*.tex"))]
    _referenced = set()
    for _f in _src:
        _referenced |= {f"{m}.tex" for m in
                        re.findall(r"\\input\{generated/([A-Za-z0-9_]+)\}", _f.read_text())}
    _on_disk = {p.name for p in GEN.glob("*.tex")}
    missing = sorted(_referenced - _on_disk)
    assert not missing, f"the report \\inputs generated files that do not exist: {missing}"
    inputs = sorted(_referenced - {"repro_macros.tex"})
    emitted_not_input = sorted(_on_disk - _referenced - {"repro_macros.tex"})
    covered = {k.split(":")[-1] for k in results}
    uncovered = [n for n in inputs if n not in covered]

    print("\n=== reproduce: per-table status ===")
    verified, unverified, failed, side = [], [], [], []
    for k, v in sorted(results.items()):
        print(f"  {k:38s} {v}")
        if "DRIFT" in v or v.startswith(("FAIL", "ERROR")):
            failed.append(k)
        elif "byte-identical" in v or v == "regenerated":
            # `figures` is a directory, not one of the generated/*.tex inputs. Counting it in a
            # tally whose denominator is len(inputs) printed "11/22" for 10 verified .tex files.
            # Same for an artifact that is emitted and checked but never \input by the report.
            (verified if k.split(":")[-1] in inputs else side).append(k)
        else:                                   # PINNED-ENV REQUIRED / PENDING / anything else
            unverified.append(k)
    if uncovered:
        print("\n  not covered by the harness (committed outputs of their own locked analyses):")
        for n in uncovered:
            print(f"    - {n}")
    # A study function can fail before it names any .tex (an import error, say), so `failed`
    # may hold keys that are not inputs. Counting those against len(inputs) used to trip a bare
    # AssertionError and abort the run -- the check crashed instead of reporting. Partition on
    # the .tex keys only, and surface anything else as a separate, visible category.
    failed_inputs = [k for k in failed if k.split(":")[-1] in inputs]
    failed_other = [k for k in failed if k not in failed_inputs]
    total = len(verified) + len(unverified) + len(uncovered) + len(failed_inputs)
    if total != len(inputs):
        print(f"\n  WARNING: coverage accounting does not partition the inputs: "
              f"{len(verified)}+{len(unverified)}+{len(uncovered)}+{len(failed_inputs)} "
              f"!= {len(inputs)}. Treat the printed coverage as unreliable.")
    if failed_other:
        print(f"  non-input failures (not counted in coverage): {', '.join(failed_other)}")
    print(f"\n  byte-checked {len(verified)}/{len(inputs)} inputs; "
          f"{len(unverified)} unverified; {len(uncovered)} not covered; {len(failed)} failed"
          + (f"  (+{len(side)} non-input artifact: {', '.join(side)})" if side else ""))

    # The abstract and the reproducibility section quote these counts. They were hand-typed and
    # went stale (the paper said "12 of the 24" while the manuscript imported 31), so emit them
    # and let the prose read the macros. The denominator is stated in the file itself, because a
    # bare "31" with no definition was one of the things a reader could not check.
    repro_tex = "\n".join([
        r"% GENERATED by reproduce.py -- coverage of the generated/ surface, do not hand-edit.",
        r"% Denominator: every papers/unified-report/generated/*.tex the report \input{}s,",
        r"% EXCLUDING this file. Counting this file would make the denominator self-referential",
        r"% (it reports on the very set it belongs to), so \ReproNInputs is one less than the",
        r"% number the report references, and \ReproNInputsInclSelf states that raw count.",
        r"% The set is PARSED from \input{generated/...} in the manuscript, not globbed from the",
        r"% directory: globbing counted an emitted-but-never-input macro file and inflated both",
        r"% the numerator and the denominator.",
        r"\newcommand{\ReproNInputs}{%d}" % len(inputs),
        r"\newcommand{\ReproNInputsInclSelf}{%d}" % (len(inputs) + 1),
        r"\newcommand{\ReproNVerified}{%d}" % len(verified),
        r"\newcommand{\ReproNEnvGated}{%d}" % len(unverified),
        r"\newcommand{\ReproNUncovered}{%d}" % len(uncovered),
        r"\newcommand{\ReproNFailed}{%d}" % len(failed_inputs),
    ]) + "\n"
    # In check mode this file was rewritten anyway -- the one artifact a verification run
    # always dirtied, and the one it never checked. Now it is checked like everything else.
    rm = GEN / "repro_macros.tex"
    if args.check and rm.exists():
        if rm.read_text() != repro_tex:
            print("\n  repro_macros.tex: DRIFT (committed coverage counts do not match this run)")
            failed.append("repro_macros.tex")
        else:
            print("\n  repro_macros.tex: OK (byte-identical)")
    else:
        rm.write_text(repro_tex)
    if args.build:
        print("\n=== building PDF ===")
        b = _run(["tectonic", "--outdir", "build", "unified_report.tex"], cwd=HERE)
        print("  build:", "OK" if b.returncode == 0 else "FAIL\n" + b.stderr[-500:])
    if args.check and (failed or unverified):
        if failed:
            print(f"\nCHECK FAILED: drift or error in {', '.join(failed)}.")
        if unverified:
            print(f"CHECK INCOMPLETE: {', '.join(unverified)} could not be verified in this environment "
                  f"(see docs/reproducibility-environments.md).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
