"""Frontier-model reference point on ExpGuard: what a hosted guardrail buys, and costs.

Act III reports four *base* checkpoints on ExpGuard (\\Cref{tab:expguard}), and the
limitations section names two gaps: the tuned comparison, and the absence of any
frontier-model reference. This module closes both on the same 2,275 expert-annotated rows,
and pairs the accuracy result with the serving cost that accuracy is bought at.

Three families of guard, all scored on identical rows and joined by row hash:

  base          the four pinned checkpoints, zero-shot         (scores_<key>_base.json)
  SFT (in-env)  the same four after ordinary SFT, 5 seeds      (scores_sft_<key>_beta0_seed<n>.json)
  frontier      gpt-5.4 / gpt-5.4-mini at three efforts        (scores_gpt54*.json)

WHAT THE SFT ROWS ARE. Not the Act~I release adapters -- those were produced on an
ephemeral GCP runner whose bucket was deleted at cleanup. These come from the KL-SFT
sweep's beta=0 arm: same LOCK sha256, same train manifest sha256, same LoRA recipe, same
pinned base revisions, but a distinct execution with different `adapter_sha256` values.
That is precisely the `sft_inenv` vs `sft_committed` distinction `klsft_summary.json`
already draws, and the tables label it **SFT (in-env)** everywhere. Act~I's headline SFT
numbers are unaffected and are not restated here.

WHY TWO SCORE SCALES CANNOT BE COMPARED NAIVELY. The local guards rank by a raw logit
margin -- continuous, effectively tie-free. The frontier configs rank by a self-reported
integer 0--100 risk, because the Responses API exposes no logprobs for reasoning models;
that yields roughly 50--65 distinct values over 2,275 rows. Heavy ties cap how finely AP
can resolve a ranking, so the coarse score *handicaps* the frontier rows. A frontier AP
below a local one would therefore be ambiguous; a frontier AP above one is conservative.
AP and AUROC come from `guard_research.metrics` (tie-aware, non-interpolated) for exactly
this reason.

THE PRIMARY COLUMN IS TPR AT A MATCHED FALSE-ALARM BUDGET, not each model's own verdict.
Act~I already established that comparing recall at unequal alarm rates is not a comparison
of discriminative power (see matched_fpr.py). The frontier configs sit at a self-chosen
2.3--3.4% FPR, the local guards at whatever their margin implies, so every row here is
re-thresholded to a common 5% FPR on the same rows before recall is read off.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCORES = REPO / "artifacts" / "expguard_external"
# Scale-ladder extension checkpoints live in their own root, because they are not part of
# Act I's locked four-checkpoint panel (see artifacts/qwen3_scale_ext/LOCK.json).
SCALE_SCORES = REPO / "artifacts" / "qwen3_scale_ext" / "expguard"

DOMAINS = ("finance", "healthcare", "law")
FPR_BUDGET = 0.05
BOOT_N = 2000
BOOT_SEED = 20260716          # matches the existing ExpGuard CI convention
SEEDS = (42, 43, 44, 45, 46)

BASE_ORDER = ["qwen25_15b", "smollm2_17b", "smollm3_3b", "qwen3_4b"]
PRETTY = {"qwen25_15b": "Qwen2.5-1.5B", "smollm2_17b": "SmolLM2-1.7B",
          "smollm3_3b": "SmolLM3-3B", "qwen3_4b": "Qwen3-4B"}
# The scale ladder. Same family as qwen3_4b and -- verified before sealing the extension
# lock -- the same prompt_template_sha256 and the same single-token decision pair, so within
# {4B, 8B, 32B} base parameter count is the only quantity that varies.
SCALE_ORDER = ["qwen3_8b", "qwen3_32b"]
SCALE_PRETTY = {"qwen3_8b": "Qwen3-8B", "qwen3_32b": "Qwen3-32B"}
PARAMS_B = {"qwen25_15b": 1.5, "smollm2_17b": 1.7, "smollm3_3b": 3.0,
            "qwen3_4b": 4.0, "qwen3_8b": 8.0, "qwen3_32b": 32.0}

FRONTIER_ORDER = ["gpt54_low", "gpt54_medium", "gpt54_high",
                  "gpt54mini_low", "gpt54mini_medium", "gpt54mini_high"]
FRONTIER_PRETTY = {
    "gpt54_low": "gpt-5.4 (low)", "gpt54_medium": "gpt-5.4 (medium)",
    "gpt54_high": "gpt-5.4 (high)", "gpt54mini_low": "gpt-5.4-mini (low)",
    "gpt54mini_medium": "gpt-5.4-mini (medium)", "gpt54mini_high": "gpt-5.4-mini (high)",
}

# Serving figures. Local: committed batched A100 latency (tab:latency), same source.
# Frontier: measured on the ExpGuard rows themselves at concurrency 200, and $/1k prompts
# from the billed token counts at the assumed list prices recorded in gpt-baseline.
LOCAL_LATENCY_MS = {"qwen25_15b": 10.4, "smollm2_17b": 11.9,
                    "smollm3_3b": 20.1, "qwen3_4b": 25.2}
FRONTIER_SERVING = {   # p50 ms, p99 ms, $ per 1k prompts
    "gpt54_low": (1553, 4523, 0.80), "gpt54_medium": (1837, 6424, 1.18),
    "gpt54_high": (2009, 6594, 1.60), "gpt54mini_low": (1665, 16933, 0.18),
    "gpt54mini_medium": (1638, 7161, 0.29), "gpt54mini_high": (1804, 5909, 0.41),
}


# ─────────────────────────────────────────────────────────────────────── loading
def load_labels() -> dict:
    return json.loads((SCORES / "labels_index.json").read_text())


def _read(name: str) -> dict | None:
    """Read a score file from the ExpGuard root, falling back to the extension root."""
    for root in (SCORES, SCALE_SCORES):
        path = root / name
        if path.is_file():
            return json.loads(path.read_text())
    return None


def sft_seed_files(model_key: str, beta: str = "beta0") -> list[dict]:
    out = []
    for seed in SEEDS:
        d = _read(f"scores_sft_{model_key}_{beta}_seed{seed}.json")
        if d:
            out.append(d)
    return out


# ─────────────────────────────────────────────────────────────────────── metrics
def _vectors(scores: dict, labels: dict, domain: str | None = None):
    import numpy as np
    ids = [i for i in labels if i in scores
           and (domain is None or labels[i]["domain"] == domain)]
    s = np.asarray([scores[i] for i in ids], dtype=float)
    y = np.asarray([labels[i]["label"] for i in ids], dtype=int)
    return s, y, ids


def tpr_at_fpr(s, y, budget: float = FPR_BUDGET) -> float:
    """Recall at the threshold whose FPR on these rows is at most `budget`.

    `method="higher"` takes the conservative side of the quantile, the same convention
    matched_fpr.py fixes for Act~I, so a tie in the negative scores cannot buy recall.
    """
    import numpy as np
    neg = s[y == 0]
    if neg.size == 0 or (y == 1).sum() == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - budget, method="higher")
    return float((s[y == 1] > thr).mean())


def metrics(scores: dict, labels: dict) -> dict:
    from guard_research.metrics import auroc, average_precision
    s, y, ids = _vectors(scores, labels)
    row = {"n": len(ids), "tpr": tpr_at_fpr(s, y),
           "ap": average_precision(s, y), "auroc": auroc(s, y)}
    for dom in DOMAINS:
        sd, yd, _ = _vectors(scores, labels, dom)
        row[f"ap_{dom}"] = average_precision(sd, yd)
    return row


def mean_over_seeds(seed_scores: list[dict], labels: dict) -> dict:
    """Metric computed per seed, then averaged -- Paper A's convention.

    Averaging the *metrics* rather than the scores is deliberate: the five adapters are
    five draws of the same recipe, and pooling their margins would blend five different
    score scales into one ranking that belongs to no guard.
    """
    per = [metrics(s, labels) for s in seed_scores]
    keys = ["tpr", "ap", "auroc"] + [f"ap_{d}" for d in DOMAINS]
    out = {k: sum(p[k] for p in per) / len(per) for k in keys}
    out["n"] = per[0]["n"]
    out["n_seeds"] = len(per)
    out["tpr_min"] = min(p["tpr"] for p in per)
    out["tpr_max"] = max(p["tpr"] for p in per)
    return out


def paired_delta(a: dict, b: dict, labels: dict, n_boot: int = BOOT_N,
                 seed: int = BOOT_SEED) -> dict:
    """Paired row bootstrap of (a - b) on the rows both guards scored.

    Paired because the two guards see identical rows: resampling rows jointly cancels
    row-difficulty variance, which is the same argument tab:expguard's paired note makes
    for the top-two local comparison.
    """
    import numpy as np
    from guard_research.metrics import average_precision
    ids = [i for i in labels if i in a and i in b]
    sa = np.asarray([a[i] for i in ids], float)
    sb = np.asarray([b[i] for i in ids], float)
    y = np.asarray([labels[i]["label"] for i in ids], int)

    def stat(idx):
        return (tpr_at_fpr(sa[idx], y[idx]) - tpr_at_fpr(sb[idx], y[idx]),
                average_precision(sa[idx], y[idx]) - average_precision(sb[idx], y[idx]))

    obs = stat(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    bt, ba = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        t, p = stat(idx)
        bt.append(t)
        ba.append(p)
    pct = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))  # noqa: E731
    return {"n": len(ids), "d_tpr": obs[0], "ci_tpr": pct(bt),
            "d_ap": obs[1], "ci_ap": pct(ba)}


def compute() -> dict:
    """Every row of the comparison, from committed text-free scores only."""
    labels = load_labels()
    rows = {"base": {}, "sft": {}, "scale": {}, "scale_sft": {}, "frontier": {}}
    for key in BASE_ORDER:
        base = _read(f"scores_{key}_base.json")
        if base:
            rows["base"][key] = metrics(base, labels)
        seeds = sft_seed_files(key)
        if seeds:
            rows["sft"][key] = mean_over_seeds(seeds, labels)
    for key in SCALE_ORDER:
        base = _read(f"scores_{key}_base.json")
        if base:
            rows["scale"][key] = metrics(base, labels)
        seeds = sft_seed_files(key)
        if seeds:
            rows["scale_sft"][key] = mean_over_seeds(seeds, labels)
    for key in FRONTIER_ORDER:
        d = _read(f"scores_{key}.json")
        if d:
            rows["frontier"][key] = metrics(d, labels)
    return {"labels_n": len(labels), "rows": rows}


# ──────────────────────────────────── Act I regimes for the scale ladder (parquet)
#
# Separate from everything above, and never pooled with it: these are the *inspected
# panel* sources (Acts I--II evidence flavour), whereas the ExpGuard rows are external
# expert-annotated. \Cref{tab:ledger} forbids averaging the two, so they are computed and
# reported apart. The regime definitions and the macro-AP convention below reproduce Act I's
# committed `base_represented` / `base_transfer` to four decimals for all four panel
# checkpoints, which is the check that licenses putting an 8B or 32B row beside them.
PANEL_SCORES = REPO / "artifacts/paper_a_sft_v2/scores/scores.parquet"
SCALE_PARQUET = REPO / "artifacts/qwen3_scale_ext/scores"
KLSFT_SUMMARY = REPO / "artifacts/klsft_v1/klsft_summary.json"
REGIME_REPRESENTED = ("toxicchat", "prompt_injections", "jailbreak_classification")
REGIME_TRANSFER = ("jailbreakbench", "xstest", "wildguardtest", "wildjailbreak")


def _macro_ap(frame, sources) -> float:
    import numpy as np
    from guard_research.metrics import average_precision
    vals = []
    for src in sources:
        d = frame[frame.source == src]
        if len(d) and d.gold.nunique() > 1:
            vals.append(average_precision(d.score_raw.values, d.gold.values))
    return float(np.mean(vals)) if vals else float("nan")


def regimes(frame) -> tuple[float, float]:
    """(represented, transfer) macro-AP under Act I's split + source convention."""
    return (_macro_ap(frame[frame.split == "id_test"], REGIME_REPRESENTED),
            _macro_ap(frame[frame.split == "transfer_test"], REGIME_TRANSFER))


def compute_scale_regimes() -> dict | None:
    """Panel base rows, the tuned panel mean, and the scale-ladder rows, all on Act I regimes."""
    if not PANEL_SCORES.is_file():
        return None
    import pandas as pd
    cols = ["source", "split", "gold", "score_raw", "model_key", "condition"]
    ref = pd.read_parquet(PANEL_SCORES, columns=cols)
    out = {"panel_base": {}, "ladder": {}, "tuned_panel_mean": None}
    for key in BASE_ORDER:
        d = ref[(ref.model_key == key) & (ref.condition == "base")]
        if len(d):
            r, t = regimes(d)
            out["panel_base"][key] = {"represented": r, "transfer": t}
    out["ladder_sft"] = {}
    for key in SCALE_ORDER:
        p = SCALE_PARQUET / f"scores_{key}_base.parquet"
        if p.is_file():
            r, t = regimes(pd.read_parquet(p))
            out["ladder"][key] = {"represented": r, "transfer": t}
        # SFT arm: metric per seed, then averaged (Act I's convention).
        seeds = sorted(SCALE_PARQUET.glob(f"scores_{key}_sft_seed*.parquet"))
        if seeds:
            import numpy as np
            per = [regimes(pd.read_parquet(f)) for f in seeds]
            out["ladder_sft"][key] = {
                "represented": float(np.mean([x[0] for x in per])),
                "transfer": float(np.mean([x[1] for x in per])),
                "n_seeds": len(per),
            }
    # The tuned reference is Act I's *committed release* adapters, read from the KL-SFT
    # summary rather than recomputed, so this row is the published quantity.
    if KLSFT_SUMMARY.is_file():
        import numpy as np
        s = json.loads(KLSFT_SUMMARY.read_text())
        out["tuned_panel_mean"] = {
            "represented": float(np.mean([d["sft_committed_represented"] for d in s])),
            "transfer": float(np.mean([d["sft_committed_transfer"] for d in s])),
            "base_represented": float(np.mean([d["base_represented"] for d in s])),
            "base_transfer": float(np.mean([d["base_transfer"] for d in s])),
            "n": len(s),
        }
    return out


def emit_scale_table(data: dict | None = None) -> str:
    data = data or compute_scale_regimes()
    if not data or not data["ladder"]:
        return ""
    both = {**PRETTY, **SCALE_PRETTY}
    lines = ["\\multicolumn{4}{l}{\\emph{Act~I panel, base (zero-shot)}} \\\\"]
    for k in BASE_ORDER:
        if k in data["panel_base"]:
            m = data["panel_base"][k]
            lines.append(f"{both[k]} & {PARAMS_B[k]:g} & {_f(m['represented'], 4)} & "
                         f"{_f(m['transfer'], 4)} \\\\")
    tm = data.get("tuned_panel_mean")
    if tm:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{Act~I panel after SFT, mean of "
                     f"{tm['n']} checkpoints (committed release adapters)}}}} \\\\")
        lines.append(f"Panel mean, SFT & -- & {_f(tm['represented'], 4)} & "
                     f"{_f(tm['transfer'], 4)} \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{4}{l}{\\emph{Scale ladder, base (zero-shot; outside the "
                 "locked panel)}} \\\\")
    for k in SCALE_ORDER:
        if k in data["ladder"]:
            m = data["ladder"][k]
            lines.append(f"\\textbf{{{both[k]}}} & {PARAMS_B[k]:g} & "
                         f"{_f(m['represented'], 4)} & {_f(m['transfer'], 4)} \\\\")
    if data.get("ladder_sft"):
        ns = next(iter(data["ladder_sft"].values()))["n_seeds"]
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{Scale ladder after SFT (in-env), "
                     f"mean of {ns} seeds}}}} \\\\")
        for k in SCALE_ORDER:
            if k in data["ladder_sft"]:
                m = data["ladder_sft"][k]
                b = data["ladder"].get(k)
                delta = ""
                if b:
                    delta = (f" \\;\\footnotesize$({m['represented']-b['represented']:+.3f},\\,"
                             f"{m['transfer']-b['transfer']:+.3f})$")
                lines.append(f"{both[k]}, SFT & {PARAMS_B[k]:g} & "
                             f"{_f(m['represented'], 4)} & {_f(m['transfer'], 4)}{delta} \\\\")
    body = "\n".join(lines)
    cap = ("Scaling the base versus tuning a small one, on Act~I's own two regimes "
           "(macro-AP; represented $=$ in-distribution sources on \\code{id\\_test}, "
           "transfer $=$ held-out sources on \\code{transfer\\_test}). The convention here "
           "reproduces Act~I's committed \\code{base\\_represented} and "
           "\\code{base\\_transfer} to four decimals for all four panel checkpoints, which is "
           "what licenses placing extension rows beside them. SFT buys represented AP and "
           "pays for it in transfer; a larger \\emph{untuned} base buys much of the same "
           "represented AP and keeps its transfer. These are inspected-panel numbers and are "
           "reported apart from --- never pooled with --- the external ExpGuard numbers in "
           "\\Cref{tab:frontier}. The scale rows are a deployment-choice contrast, not a "
           "controlled one: Qwen3-32B is $8$--$21\\times$ the parameters of the panel "
           "checkpoints and costs accordingly.")
    return ("% GENERATED by reproduce.py from artifacts/ (frontier.py)\n"
            "\\begin{table}[H]\\centering\\small\n"
            f"\\caption{{{cap}}}\n\\label{{tab:frontier-scale}}\n"
            "\\begin{tabular}{lrrr}\\toprule\n"
            "Guard & Params (B) & Represented AP & Transfer AP \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


# ────────────────────────────────────────────────────────────────────────── LaTeX
def _f(v, nd=3):
    if v is None or v != v:
        return "--"
    return f"{v:.{nd}f}".lstrip("0") if 0 < v < 1 else f"{v:.{nd}f}"


def _best(section: dict, field: str) -> tuple[str, float] | None:
    if not section:
        return None
    k = max(section, key=lambda x: section[x][field])
    return k, section[k][field]


def emit_table(data: dict | None = None) -> str:
    data = data or compute()
    r = data["rows"]
    labels = load_labels()

    def line(name, m, bold=False):
        nm = f"\\textbf{{{name}}}" if bold else name
        return (f"{nm} & {m['n']} & {_f(m['tpr'])} & {_f(m['ap'], 4)} & {_f(m['auroc'], 4)} & "
                f"{_f(m['ap_finance'], 4)} & {_f(m['ap_healthcare'], 4)} & {_f(m['ap_law'], 4)} \\\\")

    def pname(k):
        both = {**PRETTY, **SCALE_PRETTY}
        return f"{both[k]} ({PARAMS_B[k]:g}B)" if k in PARAMS_B else both[k]

    parts = []
    parts.append("\\multicolumn{8}{l}{\\emph{Act~I panel, base (zero-shot)}} \\\\")
    parts += [line(pname(k), r["base"][k]) for k in BASE_ORDER if k in r["base"]]
    if r["sft"]:
        n_seeds = next(iter(r["sft"].values()))["n_seeds"]
        parts.append("\\midrule")
        parts.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{Act~I panel, SFT (in-env), mean of {n_seeds} seeds}}}} \\\\")
        parts += [line(pname(k), r["sft"][k]) for k in BASE_ORDER if k in r["sft"]]
    if r["scale"]:
        parts.append("\\midrule")
        parts.append("\\multicolumn{8}{l}{\\emph{Scale-ladder extension, base (zero-shot; "
                     "outside the locked panel)}} \\\\")
        parts += [line(pname(k), r["scale"][k]) for k in SCALE_ORDER if k in r["scale"]]
    if r["scale_sft"]:
        n_seeds = next(iter(r["scale_sft"].values()))["n_seeds"]
        parts.append("\\midrule")
        parts.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{Scale-ladder extension, SFT (in-env), "
                     f"mean of {n_seeds} seeds}}}} \\\\")
        parts += [line(pname(k), r["scale_sft"][k]) for k in SCALE_ORDER if k in r["scale_sft"]]
    parts.append("\\midrule")
    parts.append("\\multicolumn{8}{l}{\\emph{Frontier, hosted API (zero-shot)}} \\\\")
    parts += [line(FRONTIER_PRETTY[k], r["frontier"][k], bold=True)
              for k in FRONTIER_ORDER if k in r["frontier"]]
    body = "\n".join(parts)

    # Paired footnote: the best hosted config against the best *open* guard available,
    # searching base and scale-ladder rows together so the comparison is against the
    # strongest open-weight option in the study, not merely the strongest panel member.
    note = ""
    bf = _best(r["frontier"], "tpr")
    open_pool = {**r["base"], **r["scale"]}
    bl = _best(open_pool, "tpr")
    best_local_sft = _best({**r["sft"], **r["scale_sft"]}, "tpr") if (
        r["sft"] or r["scale_sft"]) else None
    if bf and bl:
        both = {**PRETTY, **SCALE_PRETTY}
        fs = _read(f"scores_{bf[0]}.json")
        bkey = bl[0]
        bs = _read(f"scores_{bkey}_base.json")
        d = paired_delta(fs, bs, labels)
        note = (
            "\\\\[3pt]{\\footnotesize \\textbf{Paired comparison} "
            f"({FRONTIER_PRETTY[bf[0]]} $-$ {both[bkey]} base, the strongest open guard here, "
            f"on the {d['n']:,} rows both scored; {BOOT_N:,}-resample paired row bootstrap). "
            f"$\\Delta$TPR@5\\%FPR $= {d['d_tpr']:+.4f}$ "
            f"$[{d['ci_tpr'][0]:+.4f}, {d['ci_tpr'][1]:+.4f}]$; "
            f"$\\Delta$AP $= {d['d_ap']:+.4f}$ $[{d['ci_ap'][0]:+.4f}, {d['ci_ap'][1]:+.4f}]$. ")
        # Within-family scale contrast: same prompt hash, same rows, only size differs.
        if "qwen3_32b" in r["scale"] and "qwen3_4b" in r["base"]:
            ds = paired_delta(_read("scores_qwen3_32b_base.json"),
                              _read("scores_qwen3_4b_base.json"), labels)
            note += ("Within the Qwen3 family, $8\\times$ the parameters (4B~$\\to$~32B) buys "
                     f"$\\Delta$TPR $= {ds['d_tpr']:+.4f}$ "
                     f"$[{ds['ci_tpr'][0]:+.4f}, {ds['ci_tpr'][1]:+.4f}]$ --- "
                     "about as much as the gap that remains. ")
        if best_local_sft:
            sk = best_local_sft[0]
            seeds = sft_seed_files(sk)
            ds2 = [paired_delta(fs, s, labels, n_boot=200) for s in seeds]
            mt = sum(x["d_tpr"] for x in ds2) / len(ds2)
            ma = sum(x["d_ap"] for x in ds2) / len(ds2)
            note += (f"Against the strongest tuned guard ({both[sk]} SFT in-env), averaged "
                     f"over its {len(ds2)} seeds: $\\Delta$TPR@5\\%FPR $= {mt:+.4f}$, "
                     f"$\\Delta$AP $= {ma:+.4f}$. ")
        note += ("The frontier score is a coarse integer 0--100 risk with heavy ties, which "
                 "caps AP resolution, so these deltas are conservative.}")

    cap = ("Frontier hosted guardrails against this report's local guards on ExpGuard "
           f"({data['labels_n']:,} expert-annotated finance/health/law prompts), joined by row "
           "hash so every guard is scored on identical rows. \\textbf{TPR@5\\%FPR} is recall at a "
           "matched false-alarm budget --- each guard is re-thresholded on these rows to a "
           "common $5\\%$ FPR, because the models sit at very different self-chosen operating "
           "points (the frontier configs alarm on only $2.3$--$3.4\\%$ of negatives at their own "
           "verdict). Local guards rank by the raw margin $z_{\\text{unsafe}}-z_{\\text{safe}}$; "
           "frontier guards by a self-reported integer risk, the only graded signal the "
           "Responses API exposes for reasoning models. SFT rows are the \\emph{in-env} beta$=0$ "
           "re-execution of the Act~I recipe, not the Act~I release adapters (different "
           "\\code{adapter\\_sha256}); Act~I's numbers are unchanged and not restated here.")
    return ("% GENERATED by reproduce.py from artifacts/expguard_external/ (frontier.py)\n"
            "\\begin{table}[H]\\centering\\footnotesize\n"
            f"\\caption{{{cap}}}\n\\label{{tab:frontier}}\n"
            "\\begin{tabular}{lrrrrrrr}\\toprule\n"
            "Guard & $n$ & TPR@5\\%FPR & AP & AUROC & AP fin & AP health & AP law \\\\\n"
            "\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}{note}\n\\end{{table}}\n")


def emit_serving_table(data: dict | None = None) -> str:
    data = data or compute()
    r = data["rows"]
    lines = ["\\multicolumn{5}{l}{\\emph{Local guards --- one forward pass, batched on A100}} \\\\"]
    for k in BASE_ORDER:
        if k not in r["base"]:
            continue
        tpr = r["sft"][k]["tpr"] if k in r["sft"] else r["base"][k]["tpr"]
        tag = "SFT" if k in r["sft"] else "base"
        lines.append(f"{PRETTY[k]} ({tag}) & {LOCAL_LATENCY_MS[k]:.1f} & -- & "
                     f"self-hosted & {_f(tpr)} \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{l}{\\emph{Frontier --- hosted API, measured on the ExpGuard rows}} \\\\")
    for k in FRONTIER_ORDER:
        if k not in r["frontier"]:
            continue
        p50, p99, usd = FRONTIER_SERVING[k]
        lines.append(f"\\textbf{{{FRONTIER_PRETTY[k]}}} & {p50:,} & {p99:,} & "
                     f"\\${usd:.2f} & {_f(r['frontier'][k]['tpr'])} \\\\")
    body = "\n".join(lines)
    cap = ("What the frontier accuracy in \\Cref{tab:frontier} costs to serve. Local P50 is the "
           "committed batched A100 figure from \\Cref{tab:latency}; frontier latency is measured "
           "on the ExpGuard rows themselves at concurrency 200 (throughput-regime, an upper "
           "bound on an isolated request). Dollar figures are billed tokens at assumed public "
           "list prices --- an estimate, not billing truth --- and exclude the fixed cost of "
           "owning a GPU, which is what the self-hosted column elides. The gap is roughly two "
           "orders of magnitude in median latency for one order of magnitude fewer missed "
           "unsafe prompts, and three properties do not appear in any column: prompts leave the "
           "operator's infrastructure, the operating point is only coarsely selectable, and the "
           "provider may refuse a prompt before the guard ever sees it.")
    return ("% GENERATED by reproduce.py from artifacts/expguard_external/ (frontier.py)\n"
            "\\begin{table}[H]\\centering\\small\n"
            f"\\caption{{{cap}}}\n\\label{{tab:frontier-serving}}\n"
            "\\begin{tabular}{lrrrr}\\toprule\n"
            "Guard & P50 (ms) & P99 (ms) & \\$/1k prompts & TPR@5\\%FPR \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


def emit_macros(data: dict | None = None) -> str:
    """\\Frontier* macros so the prose never hard-codes a number."""
    data = data or compute()
    r = data["rows"]
    out = ["% GENERATED by reproduce.py from artifacts/expguard_external/ (frontier.py)"]

    def mac(name, val):
        # Macro names must be letters only -- a digit ends the control sequence, so
        # \FrontierBestP50 parses as \FrontierBestP followed by stray "50", which
        # typesets in the preamble and fails with "Missing \begin{document}".
        assert name.isalpha(), f"macro name must be letters only: {name!r}"
        out.append(f"\\newcommand{{\\Frontier{name}}}{{{val}}}")

    both = {**PRETTY, **SCALE_PRETTY}
    labels = load_labels()
    bf = _best(r["frontier"], "tpr")
    bb = _best(r["base"], "tpr")                      # best Act I panel base
    bopen = _best({**r["base"], **r["scale"]}, "tpr")  # best open guard overall
    bs = _best({**r["sft"], **r["scale_sft"]}, "tpr") if (r["sft"] or r["scale_sft"]) else None
    if bf:
        mac("BestName", FRONTIER_PRETTY[bf[0]].replace("_", "\\_"))
        mac("BestTpr", _f(bf[1]))
        mac("BestAp", _f(r["frontier"][bf[0]]["ap"], 4))
    if bb:
        mac("BestBaseName", both[bb[0]])
        mac("BestBaseTpr", _f(bb[1]))
    if bopen:
        mac("BestOpenName", both[bopen[0]])
        mac("BestOpenTpr", _f(bopen[1]))
        mac("BestOpenParams", f"{PARAMS_B[bopen[0]]:g}")
    if bs:
        mac("BestSftName", both[bs[0]])
        mac("BestSftTpr", _f(bs[1]))
        src = r["sft"] if bs[0] in r["sft"] else r["scale_sft"]
        mac("NSeeds", src[bs[0]]["n_seeds"])
    if bf and bopen:
        d = paired_delta(_read(f"scores_{bf[0]}.json"),
                         _read(f"scores_{bopen[0]}_base.json"), labels)
        mac("GainOverOpen", f"{d['d_tpr']:+.3f}")
        mac("GainOverOpenCI", f"[{d['ci_tpr'][0]:+.3f}, {d['ci_tpr'][1]:+.3f}]")
    if bf and bb:
        d = paired_delta(_read(f"scores_{bf[0]}.json"),
                         _read(f"scores_{bb[0]}_base.json"), labels)
        mac("GainOverBase", f"{d['d_tpr']:+.3f}")
        mac("GainOverBaseCI", f"[{d['ci_tpr'][0]:+.3f}, {d['ci_tpr'][1]:+.3f}]")
    # Within-family scale contrast, and how it compares to the residual frontier gap.
    if "qwen3_32b" in r["scale"] and "qwen3_4b" in r["base"]:
        ds = paired_delta(_read("scores_qwen3_32b_base.json"),
                          _read("scores_qwen3_4b_base.json"), labels)
        mac("ScaleGain", f"{ds['d_tpr']:+.3f}")
        mac("ScaleGainCI", f"[{ds['ci_tpr'][0]:+.3f}, {ds['ci_tpr'][1]:+.3f}]")
        mac("ScaleFactor", "8")
    if "qwen3_8b" in r["scale"] and "qwen3_4b" in r["base"]:
        d8 = paired_delta(_read("scores_qwen3_8b_base.json"),
                          _read("scores_qwen3_4b_base.json"), labels)
        mac("EightBvsFourB", f"{d8['d_tpr']:+.3f}")
        mac("EightBvsFourBCI", f"[{d8['ci_tpr'][0]:+.3f}, {d8['ci_tpr'][1]:+.3f}]")
    # SFT effect on external transfer, per checkpoint: the sign split is the finding.
    # Span the ladder too: the sign split is the finding, so it must count every
    # checkpoint for which both a base and a tuned arm exist, not just the panel.
    all_base = {**r["base"], **r["scale"]}
    all_sft = {**r["sft"], **r["scale_sft"]}
    deltas = {k: all_sft[k]["tpr"] - all_base[k]["tpr"]
              for k in (BASE_ORDER + SCALE_ORDER) if k in all_sft and k in all_base}
    if deltas:
        worst = min(deltas, key=deltas.get)
        best = max(deltas, key=deltas.get)
        mac("SftMeanDelta", f"{sum(deltas.values())/len(deltas):+.3f}")
        mac("SftWorstName", both[worst])
        mac("SftWorstDelta", f"{deltas[worst]:+.3f}")
        mac("SftBestName", both[best])
        mac("SftBestDelta", f"{deltas[best]:+.3f}")
        mac("SftNumHurt", sum(1 for v in deltas.values() if v < 0))
        mac("SftNumTotal", len(deltas))
    if bf:
        p50, _p99, usd = FRONTIER_SERVING[bf[0]]
        slow = p50 / LOCAL_LATENCY_MS[bb[0]] if bb else float("nan")
        mac("BestMedianMs", f"{p50:,}")
        mac("BestCost", f"{usd:.2f}")
        mac("Slowdown", f"{slow:.0f}")
    return "\n".join(out) + "\n"
