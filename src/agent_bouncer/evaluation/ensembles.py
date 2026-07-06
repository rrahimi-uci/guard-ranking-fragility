"""Offline ensemble evaluation over dumped per-sample predictions.

Guards are expensive to run but cheap to combine: once each guard's per-sample
predictions are dumped (``outputs/predictions/<guard>.json`` from the benchmark
run), any ensemble of them can be scored instantly with no model inference. This
module is the single source of truth for that combine-and-score logic — used by
both the ``eval_ensembles.py`` CLI and the Workbench ``/api/ensemble`` endpoint.

Prediction file format (from ``dump_predictions`` / the benchmark run)::

    {benchmark: [[y, u, s, ms], ...]}   # y=gold-unsafe, u=pred-unsafe, s=score, ms=latency

Sample order matches the cached benchmark subset, so members align by index.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence

from agent_bouncer.core.schema import Decision
from agent_bouncer.evaluation.curves import auc_with_fallback
from agent_bouncer.evaluation.metrics import compute_metrics
from agent_bouncer.models.ensemble import STRATEGIES, combine

PRED_DIR = "outputs/predictions"


def sample_key(text: str) -> str:
    """Stable identity for a benchmark sample (normalized-text hash). Dumped as the 5th element of
    each prediction row so ensemble members are aligned by the ACTUAL prompt, not by list position —
    positional alignment silently corrupts metrics when two guards were scored on different
    leakage-filtered subsets of the same benchmark (same length, different rows)."""
    norm = " ".join((text or "").lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def _keyed_index(member: list) -> dict:
    """Map each row to a composite ``(sample_key, occurrence_index)`` so DUPLICATE prompts (same
    text) are preserved as distinct entries instead of collapsing to one — the k-th occurrence of a
    text in one member aligns with the k-th occurrence in another (benchmark subset order is
    deterministic, so those are the same original sample)."""
    seen: dict[str, int] = {}
    out: dict[tuple, list] = {}
    for r in member:
        k = r[4]
        i = seen.get(k, 0)
        seen[k] = i + 1
        out[(k, i)] = r
    return out


def _align_rows(rows: list[list]) -> list[list] | None:
    """Align member prediction-row lists to a common sample order, or return ``None`` if they can't
    be safely aligned (caller skips that benchmark).

    - When EVERY member dumped a per-row key (5th element), align by the INTERSECTION of composite
      ``(sample_key, occurrence)`` identities — correct even if members were scored on different
      subsets/orders of the same benchmark, and preserving duplicate prompts.
    - For legacy dumps without keys, fall back to positional alignment (equal length required).

    Either way, a final guard requires the gold column to agree across members at every aligned
    position — any disagreement means the rows are misaligned, so the benchmark is skipped."""
    if not rows or any(len(member) == 0 for member in rows):
        return None
    if all(len(r) > 4 for member in rows for r in member):          # keyed dumps → align by identity
        maps = [_keyed_index(member) for member in rows]
        common = set(maps[0]).intersection(*maps[1:])
        if not common:
            return None
        order = sorted(common)
        aligned = [[mp[k] for k in order] for mp in maps]
    else:                                                          # legacy positional fallback
        n = len(rows[0])
        if any(len(r) != n for r in rows):
            return None
        aligned = rows
    n = len(aligned[0])
    for i in range(n):                                             # gold columns must agree
        if any(member[i][0] != aligned[0][i][0] for member in aligned[1:]):
            return None
    return aligned


def load_predictions(pred_dir: str = PRED_DIR) -> dict[str, dict]:
    """Load every ``<guard>.json`` prediction file in ``pred_dir`` → {guard: {bench: rows}}."""
    preds: dict[str, dict] = {}
    if os.path.isdir(pred_dir):
        for fname in sorted(os.listdir(pred_dir)):
            if fname.endswith(".json"):
                try:
                    preds[fname[:-5]] = json.load(open(os.path.join(pred_dir, fname)))
                except (ValueError, OSError):
                    continue
    return preds


def available_members(pred_dir: str = PRED_DIR) -> list[str]:
    """Guard names that have dumped predictions (candidate ensemble members)."""
    return sorted(load_predictions(pred_dir))


def _auc(gold: Sequence[Decision], scores: Sequence[float], m: dict) -> float:
    """True swept AUC when scores vary, else the single-operating-point estimate."""
    return auc_with_fallback(
        [1 if g == Decision.UNSAFE else 0 for g in gold], list(scores),
        recall=m["recall"], fpr=m["fpr_on_benign"],
    )


def evaluate_ensemble(
    preds: dict[str, dict],
    members: Sequence[str],
    strategy: str = "majority",
    *,
    weights: Sequence[float] | None = None,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Score an ensemble of ``members`` over their shared benchmarks.

    Returns ``{benchmark: metrics_dict}`` (metrics include ``roc_auc``). Raises
    ``ValueError`` with an actionable message for the API to surface on bad input.
    """
    members = list(members)
    if not members:
        raise ValueError("select at least one member guard")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {', '.join(STRATEGIES)}")
    missing = [m for m in members if m not in preds]
    if missing:
        raise ValueError(f"no dumped predictions for: {', '.join(missing)} — run the benchmark suite first")
    if weights is not None and len(list(weights)) != len(members):
        raise ValueError("weights length must match the number of members")

    benches = set.intersection(*[set(preds[m]) for m in members])
    if not benches:
        raise ValueError("the selected members share no common benchmark")

    out: dict[str, dict] = {}
    for bench in sorted(benches):
        rows = _align_rows([preds[m][bench] for m in members])
        if rows is None:  # members can't be safely aligned on this benchmark — skip it
            continue
        n = len(rows[0])
        gold, pred, lat, scores = [], [], [], []
        for i in range(n):
            gold.append(Decision.UNSAFE if rows[0][i][0] == 1 else Decision.SAFE)
            unsafe, sc = combine([(bool(r[i][1]), r[i][2]) for r in rows],
                                 strategy, weights=weights, threshold=threshold)
            pred.append(Decision.UNSAFE if unsafe else Decision.SAFE)
            scores.append(sc)
            lat.append(sum(r[i][3] for r in rows))  # members run sequentially
        metrics = compute_metrics(gold, pred, lat).to_dict()
        metrics["roc_auc"] = _auc(gold, scores, metrics)
        out[bench] = metrics
    if not out:
        raise ValueError("members have mismatched sample counts on every shared benchmark")
    return out


_MACRO_KEYS = ("precision", "recall", "f1", "roc_auc", "fpr_on_benign",
               "latency_p50_ms", "latency_p90_ms", "throughput_per_s")


def macro_average(per_bench: dict[str, dict]) -> dict[str, float]:
    """Mean of each metric across the benchmarks scored."""
    if not per_bench:
        return {}
    return {k: round(sum(per_bench[b][k] for b in per_bench) / len(per_bench), 4) for k in _MACRO_KEYS}


def _member_macro(preds: dict, name: str) -> dict:
    """Macro-averaged metrics for a single guard from its own dumped predictions."""
    per: dict[str, dict] = {}
    for bench, rows in preds.get(name, {}).items():
        if not rows:
            continue
        gold = [Decision.UNSAFE if r[0] == 1 else Decision.SAFE for r in rows]
        pred = [Decision.UNSAFE if r[1] == 1 else Decision.SAFE for r in rows]
        m = compute_metrics(gold, pred, [r[3] for r in rows]).to_dict()
        m["roc_auc"] = _auc(gold, [r[2] for r in rows], m)
        per[bench] = m
    return macro_average(per)


def evaluate_cascade(preds: dict[str, dict], stage1: str, stage2: str) -> dict[str, dict]:
    """Score a two-stage cascade: a high-recall GATE (``stage1``) flags candidates, then a
    high-precision FILTER (``stage2``) runs ONLY on the flagged ones. Final unsafe = stage1 AND
    stage2 (so precision rises), and per-sample latency = stage1 + (stage2 only when stage1 flags) —
    so average latency is far below running both models on every input, which is what lets a pair of
    small models rival a frontier model's quality at a fraction of the cost.

    Returns ``{benchmark: metrics_dict}`` (incl. ``roc_auc``). Raises ``ValueError`` on bad input."""
    for m in (stage1, stage2):
        if m not in preds:
            raise ValueError(f"no dumped predictions for {m!r} — test it on the benchmarks first")
    if stage1 == stage2:
        raise ValueError("a cascade needs two different models (a recall gate + a precision filter)")

    benches = set(preds[stage1]).intersection(preds[stage2])
    if not benches:
        raise ValueError("the gate and filter share no common benchmark")

    out: dict[str, dict] = {}
    for bench in sorted(benches):
        rows = _align_rows([preds[stage1][bench], preds[stage2][bench]])
        if rows is None:
            continue
        r1, r2 = rows
        gold, pred, lat, scores = [], [], [], []
        for a, b in zip(r1, r2, strict=True):
            gold.append(Decision.UNSAFE if a[0] == 1 else Decision.SAFE)
            gate_flags = bool(a[1])
            # final decision + score are the intersection (both must flag); non-gated inputs never
            # reach the filter, so their score short-circuits to the gate's (min == intersection).
            unsafe, sc = combine([(gate_flags, a[2]), (bool(b[1]), b[2])], "intersection")
            pred.append(Decision.UNSAFE if unsafe else Decision.SAFE)
            scores.append(sc)
            lat.append(a[3] + (b[3] if gate_flags else 0.0))  # filter runs only on flagged inputs
        metrics = compute_metrics(gold, pred, lat).to_dict()
        metrics["roc_auc"] = _auc(gold, scores, metrics)
        out[bench] = metrics
    if not out:
        raise ValueError("gate and filter have mismatched samples on every shared benchmark")
    return out


def optimize_cascade(preds: dict[str, dict], *, pool: Sequence[str] | None = None) -> dict:
    """Build the natural recall→precision cascade: the highest-recall model as the gate and the
    highest-precision (different) model as the filter, from ``pool`` (default every guard).

    Robust to unscorable pairs: candidate gates (recall-desc) × filters (precision-desc) are tried
    until one yields a scorable cascade, so a stale/tiny prediction dump that shares no alignable
    benchmark with the rest can't wedge the whole search."""
    allowed = set(pool) if pool is not None else None
    names = sorted(n for n in preds if allowed is None or n in allowed)
    if len(names) < 2:
        raise ValueError(
            f"need at least 2 models with dumped predictions to build a cascade; have {len(names)}"
            f"{' in the selected pool' if allowed is not None else ''} — test more models first"
        )
    macro = {n: _member_macro(preds, n) for n in names}
    by_recall = sorted(names, key=lambda n: macro[n].get("recall") or 0.0, reverse=True)
    by_prec = sorted(names, key=lambda n: macro[n].get("precision") or 0.0, reverse=True)
    for stage1 in by_recall:                       # gate: prefer highest recall
        for stage2 in by_prec:                     # filter: prefer highest precision
            if stage2 == stage1:
                continue
            try:
                per_bench = evaluate_cascade(preds, stage1, stage2)
            except ValueError:
                continue                           # this pair shares no alignable benchmark; try next
            return {"stage1": stage1, "stage2": stage2, "per_bench": per_bench,
                    "macro": macro_average(per_bench),
                    "stage1_recall": macro[stage1].get("recall"),
                    "stage2_precision": macro[stage2].get("precision")}
    raise ValueError(
        "no gate/filter pair shares alignable benchmark samples — re-test the models on the SAME "
        "benchmarks (and remove stale single-benchmark prediction dumps) so a cascade can be scored"
    )


def evaluate_deferral(preds: dict[str, dict], stage1: str, stage2: str,
                      *, low: float = 0.4, high: float = 0.6) -> dict[str, dict]:
    """Score a CONFIDENCE-DEFERRAL cascade: a cheap first model (``stage1``) decides the inputs it's
    confident about (score ≤ ``low`` → safe, score ≥ ``high`` → unsafe) and DEFERS only the uncertain
    middle band (``low`` < score < ``high``) to an expert (``stage2``). Unlike the AND cascade this is
    a router — recall isn't capped by either stage — so it can improve the operating point when the
    cheap model has a calibrated confidence signal. Per-sample latency = stage1 + (stage2 only on the
    deferred fraction); each cell records ``defer_rate``.

    NOTE: with binary 0/1 scores (the current decoder/OpenAI judges) nothing falls in the open band,
    so it degenerates to stage1 (``defer_rate`` = 0) — a continuous-score model (the encoder, or a
    calibrated head) is what makes deferral pay off."""
    for m in (stage1, stage2):
        if m not in preds:
            raise ValueError(f"no dumped predictions for {m!r} — test it on the benchmarks first")
    if stage1 == stage2:
        raise ValueError("deferral needs two different models (a cheap decider + an expert)")
    benches = set(preds[stage1]).intersection(preds[stage2])
    if not benches:
        raise ValueError("the decider and expert share no common benchmark")

    out: dict[str, dict] = {}
    for bench in sorted(benches):
        rows = _align_rows([preds[stage1][bench], preds[stage2][bench]])
        if rows is None:
            continue
        r1, r2 = rows
        gold, pred, lat, scores, deferred = [], [], [], [], 0
        for a, b in zip(r1, r2, strict=True):
            gold.append(Decision.UNSAFE if a[0] == 1 else Decision.SAFE)
            s1 = a[2]
            if low < s1 < high:                       # uncertain → defer to the expert
                unsafe, sc = bool(b[1]), b[2]
                deferred += 1
                ms = a[3] + b[3]
            else:                                     # confident → the cheap model decides
                unsafe, sc, ms = s1 >= high, s1, a[3]
            pred.append(Decision.UNSAFE if unsafe else Decision.SAFE)
            scores.append(sc)
            lat.append(ms)
        metrics = compute_metrics(gold, pred, lat).to_dict()
        metrics["roc_auc"] = _auc(gold, scores, metrics)
        metrics["defer_rate"] = round(deferred / len(r1), 4) if r1 else 0.0
        out[bench] = metrics
    if not out:
        raise ValueError("decider and expert have mismatched samples on every shared benchmark")
    return out


_DEFERRAL_BANDS = ((0.4, 0.6), (0.3, 0.7), (0.25, 0.75), (0.2, 0.8), (0.5, 0.5))


def optimize_deferral(preds: dict[str, dict], *, pool: Sequence[str] | None = None) -> dict:
    """Search (cheap decider × expert × confidence band) for the deferral cascade with the best
    macro-F1 (latency as tiebreak), from ``pool`` (default every guard)."""
    allowed = set(pool) if pool is not None else None
    names = sorted(n for n in preds if allowed is None or n in allowed)
    if len(names) < 2:
        raise ValueError(
            f"need at least 2 models with dumped predictions to build a deferral cascade; have "
            f"{len(names)}{' in the selected pool' if allowed is not None else ''} — test more first"
        )
    best = None
    for s1 in names:
        for s2 in names:
            if s1 == s2:
                continue
            for lo, hi in _DEFERRAL_BANDS:
                try:
                    pb = evaluate_deferral(preds, s1, s2, low=lo, high=hi)
                except ValueError:
                    continue
                mac = macro_average(pb)
                dr = round(sum(pb[b]["defer_rate"] for b in pb) / len(pb), 4)
                key = (mac.get("f1") or 0.0, -(mac.get("latency_p50_ms") or 0.0))
                if best is None or key > best[0]:
                    best = (key, s1, s2, lo, hi, pb, mac, dr)
    if best is None:
        raise ValueError("no decider/expert pair shares alignable benchmark samples")
    _, s1, s2, lo, hi, pb, mac, dr = best
    return {"stage1": s1, "stage2": s2, "low": lo, "high": hi,
            "per_bench": pb, "macro": mac, "defer_rate": dr}


def _q_statistic(correct_a: list[int], correct_b: list[int]) -> float:
    """Yule's Q on two models' per-sample correctness. +1 = they make the SAME errors (redundant),
    0 = independent, negative = complementary (a good ensembling signal)."""
    n11 = n00 = n10 = n01 = 0
    for a, b in zip(correct_a, correct_b, strict=True):
        if a and b:
            n11 += 1
        elif not a and not b:
            n00 += 1
        elif a and not b:
            n10 += 1
        else:
            n01 += 1
    denom = n11 * n00 + n01 * n10
    return (n11 * n00 - n01 * n10) / denom if denom else 0.0


def diversity_report(preds: dict[str, dict], members: Sequence[str]) -> dict:
    """Quantify whether ``members`` are diverse enough for ensembling to help. Aligns them by sample
    identity across shared benchmarks, then reports per-model accuracy, mean pairwise disagreement +
    error-correlation (Yule's Q), the ORACLE ceiling (accuracy if you could pick the right member
    per sample), and how often every member over-blocks the SAME benign prompt. Returns a verdict."""
    members = list(members)
    if len(members) < 2:
        raise ValueError("need at least 2 members to assess diversity")
    missing = [m for m in members if m not in preds]
    if missing:
        raise ValueError(f"no dumped predictions for: {', '.join(missing)}")

    # Find the largest jointly-alignable member set: an outlier (e.g. a stale single-benchmark dump)
    # would otherwise collapse the shared intersection, so greedily drop the smallest-coverage member
    # until the rest align on a common benchmark set.
    dropped: list[str] = []
    kept = list(members)
    ok_benches: list[str] = []
    while len(kept) >= 2:
        benches = set.intersection(*[set(preds[m]) for m in kept])
        ok_benches = [b for b in sorted(benches)
                      if _align_rows([preds[m][b] for m in kept]) is not None]
        if ok_benches:
            break
        worst = min(kept, key=lambda m: sum(len(v) for v in preds[m].values()))
        kept.remove(worst)
        dropped.append(worst)
    if len(kept) < 2 or not ok_benches:
        raise ValueError("the selected members share no alignable benchmark samples")
    members = kept

    correct = {m: [] for m in members}   # per-sample 1/0 correctness, aligned across members
    flags = {m: [] for m in members}     # per-sample predicted-unsafe, aligned
    gold: list[int] = []
    for bench in ok_benches:
        rows = _align_rows([preds[m][bench] for m in members])
        if rows is None:
            continue
        for i in range(len(rows[0])):
            y = rows[0][i][0]
            gold.append(y)
            for mi, m in enumerate(members):
                u = rows[mi][i][1]
                flags[m].append(u)
                correct[m].append(int(u == y))
    n = len(gold)
    if not n:
        raise ValueError("members have mismatched samples on every shared benchmark")

    per_member = [{"name": m, "accuracy": round(sum(correct[m]) / n, 4)} for m in members]
    best_single = max(pm["accuracy"] for pm in per_member)
    oracle = sum(1 for i in range(n) if any(correct[m][i] for m in members)) / n

    disagr, qs = [], []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = members[i], members[j]
            disagr.append(sum(1 for k in range(n) if flags[a][k] != flags[b][k]) / n)
            qs.append(_q_statistic(correct[a], correct[b]))

    benign = [i for i in range(n) if gold[i] == 0]
    all_flag = (sum(1 for i in benign if all(flags[m][i] for m in members)) / len(benign)
                if benign else 0.0)

    headroom = round(oracle - best_single, 4)
    if headroom < 0.03:
        verdict, why = "redundant", (
            "Members make almost the same decisions (near-zero oracle headroom), so no combiner can "
            "beat the best single model — you need more DIVERSE or stronger base models, not more "
            "ensembling.")
    elif headroom < 0.10:
        verdict, why = "some-complementarity", (
            "There is modest headroom: a well-chosen ensemble may edge out the best single model, but "
            "the gains will be small.")
    else:
        verdict, why = "diverse", (
            "Members are complementary (large oracle headroom) — ensembling should meaningfully beat "
            "any single member.")
    if all_flag > 0.5:
        why += (f" Also, every member over-blocks the SAME benign prompt {round(all_flag * 100)}% of "
                "the time, so intersection/cascade can't cut that over-blocking.")

    return {
        "n_samples": n, "members": sorted(per_member, key=lambda p: -p["accuracy"]),
        "dropped": dropped,
        "mean_disagreement": round(sum(disagr) / len(disagr), 4) if disagr else 0.0,
        "mean_error_correlation": round(sum(qs) / len(qs), 4) if qs else 0.0,
        "best_single_accuracy": round(best_single, 4), "oracle_accuracy": round(oracle, 4),
        "headroom": headroom, "benign_all_flag_rate": round(all_flag, 4),
        "verdict": verdict, "explanation": why,
    }


# Strategies that ignore the threshold (no sweep needed).
_HARD_STRATEGIES = ("union", "intersection", "majority")
_MAX_CANDIDATES = 2500  # keep the search responsive (a few seconds, offline)


def _objective_key(macro: dict, objective: str, fpr_cap: float):
    """Sort key (higher is better) for ranking candidate ensembles."""
    f1 = macro.get("f1") or 0.0
    fpr = macro.get("fpr_on_benign")
    fpr = 1.0 if fpr is None else fpr
    p50 = macro.get("latency_p50_ms") or 0.0
    if objective == "f1":            # best quality, latency as tiebreak
        return (f1, -fpr, -p50)
    if objective == "fpr":           # least over-blocking, then quality
        return (-fpr, f1, -p50)
    # "balanced" (default): best F1 among those within the over-blocking budget
    within = 1 if fpr <= fpr_cap else 0
    return (within, f1, -fpr, -p50)


def optimize_ensemble(
    preds: dict[str, dict],
    *,
    objective: str = "balanced",
    fpr_cap: float = 0.2,
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
    min_members: int = 2,
    max_members: int | None = None,
    top_k: int = 5,
    pool: Sequence[str] | None = None,
) -> dict:
    """Search member subsets × strategies for the best ensemble, scored offline.

    ``objective`` — ``"balanced"`` (best macro-F1 with FPR@benign ≤ ``fpr_cap``),
    ``"f1"`` (best macro-F1), or ``"fpr"`` (least over-blocking). ``pool`` restricts the
    candidate members (e.g. the small models only); defaults to every guard with predictions.
    Returns the winning config + macro/per-benchmark metrics and the top-``top_k`` candidates.
    The search is bounded (``_MAX_CANDIDATES``) so it stays responsive.
    """
    import itertools

    if objective not in ("balanced", "f1", "fpr"):
        raise ValueError(f"unknown objective {objective!r}; choose balanced, f1, or fpr")
    allowed = set(pool) if pool is not None else None
    names = sorted(n for n in preds if allowed is None or n in allowed)
    if len(names) < min_members:
        raise ValueError(
            f"need at least {min_members} guards with dumped predictions to build an ensemble; "
            f"have {len(names)}{' in the selected pool' if allowed is not None else ''} — "
            "test more models on the benchmarks first"
        )
    hi = min(max_members or len(names), len(names))
    subsets = [list(c) for r in range(min_members, hi + 1) for c in itertools.combinations(names, r)]
    thr = list(thresholds)
    # If the space is too big, shrink it (drop the threshold sweep, cap subset size) so the
    # search stays fast — exhaustive for the usual handful of guards, bounded beyond that.
    if len(subsets) * (len(_HARD_STRATEGIES) + len(thr)) > _MAX_CANDIDATES:
        thr = [0.5]
        subsets = [s for s in subsets if len(s) <= 5]

    scored: list[tuple] = []
    for members in subsets:
        for strat in _HARD_STRATEGIES:
            try:
                pb = evaluate_ensemble(preds, members, strat)
            except ValueError:
                continue
            scored.append((members, strat, None, macro_average(pb), pb))
        for t in thr:
            try:
                pb = evaluate_ensemble(preds, members, "mean", threshold=t)
            except ValueError:
                continue
            scored.append((members, "mean", t, macro_average(pb), pb))
    if not scored:
        raise ValueError("no scorable ensembles from the available predictions")

    scored.sort(key=lambda r: _objective_key(r[3], objective, fpr_cap), reverse=True)
    best = scored[0]
    candidates = [
        {"members": m, "strategy": s, "threshold": t,
         **{k: macro.get(k) for k in
            ("precision", "recall", "f1", "roc_auc", "fpr_on_benign", "latency_p50_ms")}}
        for (m, s, t, macro, _pb) in scored[:top_k]
    ]
    return {
        "best": {"members": best[0], "strategy": best[1], "threshold": best[2],
                 "macro": best[3], "per_bench": best[4]},
        "candidates": candidates,
        "objective": objective,
        "n_evaluated": len(scored),
    }
