#!/usr/bin/env python
"""Export the frontier teacher signal for the distillation study as committed, text-free files.

    .venv/bin/python gpt-baseline/export_distill_teacher.py

`gpt-baseline/raw/` is gitignored working state that exists only on the machine that ran the
baseline, so nothing bound by a LOCK may read from it. This lifts the teacher signal into a
committed artifact, exactly as `export_expguard_scores.py` did for the ExpGuard evaluation
rows -- same reasoning, different purpose: these rows are *training* input, so their
provenance has to be pinned before a single distilled score exists.

Pool: the two sources the GPT baseline scored that the Act~I manifest does NOT represent and
that the report does NOT evaluate on --

    beavertails         3,021 corpus rows  (teacher AUC .734 -- weak, see below)
    openai_moderation   1,680 corpus rows  (teacher AUC .951)

ExpGuard, xstest, jailbreakbench, wildguardtest and wildjailbreak are deliberately excluded:
distilling on them would convert the report's only external and held-out evaluation surfaces
into represented sources and destroy the estimand. That exclusion is a locked predicate of
the study, not a preference -- see artifacts/frontier_distill_v1/protocol/claim_registry.json.

Keyed by `content_sha256` (NFC-normalized), the local pipeline's convention, so the teacher
joins the panel's own manifests and score rows directly. The GPT baseline keys its raw
predictions by `sha256(text)[:16]` instead; the mapping is re-derived from the local corpus
here rather than stored, so no prompt text is written.

Teacher quality is heterogeneous and must not be averaged away: gpt-5.4's own AUC is .951 on
openai_moderation but only .734 on beavertails, which is label-convention disagreement rather
than capability. Per-source teacher AUC is therefore recorded in the provenance file so the
analysis can weight or stratify by it instead of assuming one teacher quality.

Writes (all text-free):
  artifacts/frontier_distill_v1/teacher/scores_<config>.json  {content_sha256: risk 0-100}
  artifacts/frontier_distill_v1/teacher/labels_index.json     {content_sha256: {label, source}}
  artifacts/frontier_distill_v1/teacher/provenance.json       pool, counts, teacher AUC, terms
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import runner as rn  # noqa: E402

from guard_research.metrics import auroc  # noqa: E402
from guard_research.provenance import content_sha256  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "frontier_distill_v1" / "teacher"
CORPUS = ROOT / "data" / "benchmarks" / "full"

POOL = ["beavertails", "openai_moderation"]

# Excluded from the pool by locked predicate, with the reason each exclusion protects.
HELD_CLEAN = {
    "expguard": "the report's only external expert-annotated evaluation surface",
    "xstest": "Act~I transfer_test",
    "jailbreakbench": "Act~I transfer_test",
    "wildguardtest": "Act~I transfer_test",
    "wildjailbreak": "Act~I transfer_test",
    "toxicchat": "already represented in Act~I train.jsonl",
    "prompt_injections": "already represented in Act~I train.jsonl",
    "jailbreak_classification": "already represented in Act~I train.jsonl",
}

GUARDS = {
    "gpt54_low": ("gpt-5.4", "low"),
    "gpt54_medium": ("gpt-5.4", "medium"),
    "gpt54_high": ("gpt-5.4", "high"),
    "gpt54mini_low": ("gpt-5.4-mini", "low"),
    "gpt54mini_medium": ("gpt-5.4-mini", "medium"),
    "gpt54mini_high": ("gpt-5.4-mini", "high"),
}

SAFE_LABELS = {"safe", "0", "false", "benign", "no"}
UNSAFE_LABELS = {"unsafe", "1", "true", "harmful", "jailbreak", "yes"}


def corpus_index(source: str) -> tuple[dict[str, str], dict[str, int]]:
    """(gpt rid -> content_sha256, content_sha256 -> label), derived, never stored as text."""
    rid2norm, labels, seen = {}, {}, set()
    for line in (CORPUS / f"{source}.jsonl").open():
        row = json.loads(line)
        text = str(row.get("text") or row.get("prompt") or row.get("input") or "")
        if not text.strip():
            continue
        token = str(row.get("label", row.get("gold"))).strip().lower()
        if token in SAFE_LABELS:
            label = 0
        elif token in UNSAFE_LABELS:
            label = 1
        else:
            continue
        rid = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if rid in seen:            # matches gpt-baseline/datasets.py de-duplication
            continue
        seen.add(rid)
        norm = content_sha256(text)
        rid2norm[rid] = norm
        labels[norm] = label
    return rid2norm, labels


def export() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_index: dict[str, dict] = {}
    per_config: dict[str, dict[str, float]] = {g: {} for g in GUARDS}

    for source in POOL:
        rid2norm, labels = corpus_index(source)
        for norm, label in labels.items():
            labels_index[norm] = {"label": int(label), "source": source}
        for guard, (model, effort) in GUARDS.items():
            preds = rn.read_done(rn.pred_path(model, effort, source))
            for rid, rec in preds.items():
                if not (rec.get("ok") and isinstance(rec.get("raw"), dict)):
                    continue
                risk = rec["raw"].get("risk")
                norm = rid2norm.get(rid)
                if norm is not None and isinstance(risk, (int, float)):
                    per_config[guard][norm] = float(risk)

    for guard, scores in per_config.items():
        if scores:
            (OUT_DIR / f"scores_{guard}.json").write_text(
                json.dumps(scores, sort_keys=True, separators=(",", ":")) + "\n")
    (OUT_DIR / "labels_index.json").write_text(
        json.dumps(labels_index, sort_keys=True, separators=(",", ":")) + "\n")

    # per-source teacher AUC: the analysis must be able to stratify on teacher quality
    quality = {}
    for source in POOL:
        ids = [k for k, v in labels_index.items() if v["source"] == source]
        y = [labels_index[k]["label"] for k in ids]
        quality[source] = {"n": len(ids), "prevalence": round(sum(y) / len(y), 4)}
        for guard, scores in per_config.items():
            have = [(scores[k], labels_index[k]["label"]) for k in ids if k in scores]
            if len(have) > 1 and len({h[1] for h in have}) == 2:
                quality[source][f"teacher_auc_{guard}"] = round(
                    auroc([h[0] for h in have], [h[1] for h in have]), 4)
                quality[source][f"teacher_n_{guard}"] = len(have)

    prov = {
        "study_id": "frontier_distill_v1",
        "role": "teacher signal for the distillation arms -- TRAINING input, not evaluation",
        "pool": POOL,
        "held_clean_and_why": HELD_CLEAN,
        "key": "content_sha256 = sha256(normalize_text(prompt)), the local pipeline's convention",
        "source_of_predictions": "gpt-baseline/raw/ (gitignored working state), lifted here so a "
                                 "LOCK can bind the teacher signal",
        "teacher_quality": quality,
        "text_free": "both pool sources are text_free_only in the distribution ledger; this "
                     "directory stores digests, integer risk scores and labels only",
        "license_encumbrance": {
            "beavertails": "CC-BY-NC-4.0, commercial_use false, derived_output_license "
                           "CC-BY-NC-4.0_inherited -- weights trained on it inherit a "
                           "noncommercial encumbrance",
            "openai_moderation": "MIT_upstream_unverified, derived_output_license "
                                 "none_until_decision",
            "teacher_outputs": "distilling provider model outputs into a competing classifier "
                               "is governed by the provider's terms; this is a human licensing "
                               "decision and is a blocking gate in the claim registry",
        },
        "counts": {g: len(s) for g, s in per_config.items() if s},
        "n_rows_labelled": len(labels_index),
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
    return prov


if __name__ == "__main__":
    if not (HERE / "raw").is_dir():
        print("missing gpt-baseline/raw/ -- run the baseline first", file=sys.stderr)
        raise SystemExit(1)
    p = export()
    print(f"pool={p['pool']}  rows={p['n_rows_labelled']}")
    for guard, n in sorted(p["counts"].items()):
        print(f"  scores_{guard}.json  n={n}")
    for source, q in p["teacher_quality"].items():
        print(f"  {source:20s} n={q['n']:5d} prev={q['prevalence']:.3f} "
              f"teacher_auc(gpt54_low)={q.get('teacher_auc_gpt54_low')}")
