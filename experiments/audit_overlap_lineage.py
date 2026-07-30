#!/usr/bin/env python
"""Overlap and provenance-lineage audit between training roles and evaluation surfaces.

Closes two open items with one instrument:

  1. Report roadmap item 2 -- "complete the v2 decontamination audit". The Act~I manifest is
     reported as clean-v2 but the formal n-gram / near-duplicate check against the v2 transfer
     suite was never run (limitations-validation.tex, "asserted, not yet audited"). Residual
     train/transfer overlap would INFLATE measured transfer, i.e. make specialization look
     milder than it is, so an unrun audit is a caveat pointing the wrong way.

  2. proposal.md Section 20.17 Phase A -- "build the overlap and lineage audit on
     metadata/fixtures", and Section 20.6's requirement of "exact, normalized, n-gram,
     embedding, and provenance-lineage overlap audits" before any teacher labeling or training.
     The candidate teacher pool must be shown disjoint from every evaluation surface first.

Five checks, cheapest to strictest:

  exact              identical raw text (sha256 of the bytes)
  normalized         identical after NFC, casefold, punctuation strip, whitespace collapse
  ngram_containment  fraction of a probe row's word 5-grams present anywhere in the reference
                     corpus -- catches templated and lightly-edited reuse
  shingle_jaccard    max character-5-gram Jaccard against any single reference row -- catches
                     paraphrase-level near duplicates
  lineage            family_id / upstream_family_id collisions, which catch reuse that shares
                     an upstream record even when the rendered text differs

The embedding check in Section 20.6 is deliberately NOT implemented here: it needs a pinned
encoder whose identity would have to enter the lock, so it belongs in the Phase A schema review
rather than in this script. That omission is reported in the output rather than left implicit.

Text-free output. Offending rows are reported by `content_sha256`, never by text, so the
report is publishable under the ledger's `text_free_only` decisions.

Run from the repo root:
    .venv/bin/python experiments/audit_overlap_lineage.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from guard_research.provenance import content_sha256  # noqa: E402

MANIFESTS = os.path.join(ROOT, "artifacts", "paper_a_sft_v2", "manifests")
CORPUS = os.path.join(ROOT, "data", "benchmarks", "full")
OUT = os.path.join(ROOT, "artifacts", "overlap_audit")

NGRAM_N = 5          # word n-gram size for containment
SHINGLE_N = 5        # character shingle size for Jaccard
# Report thresholds. 0.5 containment is "half this row's 5-grams already appear in training",
# which is a leakage signal well below verbatim copying.
CONTAINMENT_LEVELS = [0.30, 0.50, 0.80, 1.00]
JACCARD_LEVELS = [0.50, 0.70, 0.90]

SAFE_LABELS = {"safe", "0", "false", "benign", "no"}
UNSAFE_LABELS = {"unsafe", "1", "true", "harmful", "jailbreak", "yes"}

_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")


def norm_key(text: str) -> str:
    """Aggressive normalization: NFC, casefold, punctuation stripped, whitespace collapsed."""
    t = unicodedata.normalize("NFC", text).casefold()
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def raw_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def words(text: str) -> list[str]:
    return norm_key(text).split()


def ngrams(text: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    w = words(text)
    if len(w) < n:
        return {tuple(w)} if w else set()      # short rows: the whole row is its own gram
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def shingles(text: str, n: int = SHINGLE_N) -> set[str]:
    t = norm_key(text).replace(" ", "")
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


# ─────────────────────────────────────────────────────────────── loading


class Row:
    __slots__ = ("digest", "text", "source", "family_id", "upstream")

    def __init__(self, digest, text, source, family_id, upstream):
        self.digest, self.text, self.source = digest, text, source
        self.family_id, self.upstream = family_id, upstream


def load_manifest(split: str) -> list[Row]:
    out = []
    for line in open(os.path.join(MANIFESTS, f"{split}.jsonl")):
        r = json.loads(line)
        text = str(r.get("text_or_download_reference") or "")
        if not text.strip():
            continue
        out.append(Row(r.get("content_sha256") or content_sha256(text), text,
                       r.get("source"), r.get("family_id"), r.get("upstream_family_id")))
    return out


def load_corpus(source: str) -> list[Row]:
    out, seen = [], set()
    path = os.path.join(CORPUS, f"{source}.jsonl")
    if not os.path.isfile(path):
        return out
    for line in open(path):
        r = json.loads(line)
        text = str(r.get("text") or r.get("prompt") or r.get("input") or "")
        if not text.strip():
            continue
        token = str(r.get("label", r.get("gold"))).strip().lower()
        if token not in SAFE_LABELS and token not in UNSAFE_LABELS:
            continue
        d = content_sha256(text)
        if d in seen:
            continue
        seen.add(d)
        out.append(Row(d, text, source, None, None))
    return out


# ─────────────────────────────────────────────────────────────── checks


def containment_and_jaccard(probe: list[Row], ref: list[Row]) -> dict:
    """For each probe row: fraction of its n-grams seen anywhere in ref, and best per-row Jaccard.

    Two different questions. Containment is against the reference corpus as a WHOLE, which is
    what catches templated reuse spread over many rows. Jaccard is against the single closest
    reference row, which is what catches a near-duplicate pair.
    """
    index: dict[tuple[str, ...], set[int]] = defaultdict(set)
    ref_grams, ref_shingles = [], []
    for i, r in enumerate(ref):
        g = ngrams(r.text)
        ref_grams.append(g)
        ref_shingles.append(shingles(r.text))
        for gram in g:
            index[gram].add(i)

    per_row, unmeasurable = [], 0
    for p in probe:
        pg = ngrams(p.text)
        if not pg:
            unmeasurable += 1
            continue
        hit = Counter()
        seen_any = 0
        for gram in pg:
            posting = index.get(gram)
            if posting:
                seen_any += 1
                for i in posting:
                    hit[i] += 1
        containment = seen_any / len(pg)

        # Jaccard only against the rows that already share an n-gram; a row sharing none
        # cannot be a near-duplicate, so this is exact, not an approximation.
        ps = shingles(p.text)
        best_j, best_i = 0.0, None
        for i in hit:
            inter = len(ps & ref_shingles[i])
            if inter:
                j = inter / len(ps | ref_shingles[i])
                if j > best_j:
                    best_j, best_i = j, i
        per_row.append({
            "digest": p.digest, "source": p.source, "containment": round(containment, 4),
            "best_jaccard": round(best_j, 4),
            "nearest_ref_digest": ref[best_i].digest if best_i is not None else None,
            "nearest_ref_source": ref[best_i].source if best_i is not None else None,
            "n_words": len(words(p.text)),
        })

    def at_least(rows, key, lvl):
        return sum(1 for r in rows if r[key] >= lvl)

    # Per-source breakdown, because a pooled count hides which evaluation surface is affected.
    # A row shorter than the n-gram window has all of its grams trivially present, so short
    # rows are counted separately rather than allowed to inflate the containment tallies.
    by_source: dict[str, dict] = {}
    for src in sorted({r["source"] for r in per_row if r["source"]}):
        rows = [r for r in per_row if r["source"] == src]
        long_rows = [r for r in rows if r["n_words"] >= NGRAM_N * 2]
        by_source[src] = {
            "n": len(rows),
            "containment_at_least": {f"{lvl:.2f}": at_least(rows, "containment", lvl)
                                     for lvl in CONTAINMENT_LEVELS},
            "jaccard_at_least": {f"{lvl:.2f}": at_least(rows, "best_jaccard", lvl)
                                 for lvl in JACCARD_LEVELS},
            "max_jaccard": max((r["best_jaccard"] for r in rows), default=0.0),
            "n_short_rows_excluded_from_long": len(rows) - len(long_rows),
            "long_rows_only": {
                "n": len(long_rows),
                "containment_at_least_0.80": at_least(long_rows, "containment", 0.80),
                "jaccard_at_least_0.70": at_least(long_rows, "best_jaccard", 0.70),
            },
        }

    return {
        "n_probe": len(probe), "n_ref": len(ref),
        "by_source": by_source,
        "n_unmeasurable_by_ngram": unmeasurable,
        "containment_at_least": {f"{lvl:.2f}": at_least(per_row, "containment", lvl)
                                 for lvl in CONTAINMENT_LEVELS},
        "jaccard_at_least": {f"{lvl:.2f}": at_least(per_row, "best_jaccard", lvl)
                             for lvl in JACCARD_LEVELS},
        "max_containment": max((r["containment"] for r in per_row), default=0.0),
        "max_jaccard": max((r["best_jaccard"] for r in per_row), default=0.0),
        # the 25 worst rows, by digest only, so a human can adjudicate without redistributing text
        "worst_rows": sorted(per_row, key=lambda r: (-r["best_jaccard"], -r["containment"]))[:25],
    }


def digest_overlap(probe: list[Row], ref: list[Row]) -> dict:
    p_raw = {raw_key(r.text): r.digest for r in probe}
    r_raw = {raw_key(r.text) for r in ref}
    p_norm = defaultdict(set)
    for r in probe:
        p_norm[hashlib.sha256(norm_key(r.text).encode()).hexdigest()].add(r.digest)
    r_norm = {hashlib.sha256(norm_key(r.text).encode()).hexdigest() for r in ref}
    exact = sorted(d for k, d in p_raw.items() if k in r_raw)
    normalized = sorted(d for k, ds in p_norm.items() if k in r_norm for d in ds)
    return {"exact_n": len(exact), "exact_digests": exact[:50],
            "normalized_n": len(normalized), "normalized_digests": normalized[:50],
            "normalized_beyond_exact_n": len(set(normalized) - set(exact))}


def lineage_overlap(probe: list[Row], ref: list[Row]) -> dict:
    def ids(rows, attr):
        return {getattr(r, attr) for r in rows if getattr(r, attr)}
    fam = ids(probe, "family_id") & ids(ref, "family_id")
    ups = ids(probe, "upstream") & ids(ref, "upstream")
    return {"family_id_collisions": len(fam), "upstream_family_id_collisions": len(ups),
            "family_id_examples": sorted(fam)[:20], "upstream_examples": sorted(ups)[:20],
            "measurable": bool(ids(probe, "family_id") and ids(ref, "family_id"))}


# ─────────────────────────────────────────────────────────────── main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    train = load_manifest("train")
    calib = load_manifest("calibration")
    id_test = load_manifest("id_test")
    transfer = load_manifest("transfer_test")
    pool = load_corpus("beavertails") + load_corpus("openai_moderation")

    # (probe, reference) pairs. Probe is the surface whose integrity is in question; reference
    # is the corpus that must not have contaminated it.
    pairs = [
        ("act1_transfer_test__vs__act1_train", transfer, train,
         "roadmap item 2: the pending v2 decontamination audit. Leakage here would inflate "
         "measured transfer and make specialization look milder than it is."),
        ("act1_id_test__vs__act1_train", id_test, train,
         "id_test is a held-out-ROWS split of represented sources, so some lineage overlap is "
         "expected by construction; verbatim row reuse is not."),
        ("act1_transfer_test__vs__teacher_pool", transfer, pool,
         "Phase A: the candidate teacher pool must not contaminate Act I's transfer suite."),
        ("act1_id_test__vs__teacher_pool", id_test, pool,
         "Phase A: pool vs the represented evaluation split."),
        ("teacher_pool__vs__act1_train", pool, train,
         "how much of the candidate teacher pool is already represented in Act I training."),
        ("act1_train__vs__act1_calibration", train, calib,
         "control: train and calibration are both represented-source splits and should be "
         "row-disjoint, which makes this pair a check on the audit itself."),
    ]

    results = {}
    for name, probe, ref, why in pairs:
        if not probe or not ref:
            results[name] = {"skipped": "empty role"}
            continue
        results[name] = {
            "why": why,
            "digest": digest_overlap(probe, ref),
            "fuzzy": containment_and_jaccard(probe, ref),
            "lineage": lineage_overlap(probe, ref),
        }

    payload = {
        "meta": {
            "closes": ["unified-report roadmap item 2 (v2 decontamination audit)",
                       "proposal.md Section 20.17 Phase A overlap and lineage audit"],
            "checks": {
                "exact": "sha256 of raw bytes",
                "normalized": "NFC + casefold + punctuation strip + whitespace collapse",
                "ngram_containment": f"word {NGRAM_N}-gram containment against the reference "
                                     f"corpus as a whole",
                "shingle_jaccard": f"max character {SHINGLE_N}-gram Jaccard against any single "
                                   f"reference row",
                "lineage": "family_id / upstream_family_id collisions",
            },
            "not_implemented": {
                "embedding_overlap": "Section 20.6 also requires an embedding-space check. It "
                                     "needs a pinned encoder whose identity must enter the lock, "
                                     "so it belongs to the Phase A schema review, not to this "
                                     "script. This audit is therefore necessary but not "
                                     "sufficient for the Phase A gate."
            },
            "roles": {"act1_train": len(train), "act1_calibration": len(calib),
                      "act1_id_test": len(id_test), "act1_transfer_test": len(transfer),
                      "teacher_pool": len(pool)},
            "text_free": "offending rows are identified by content_sha256 only",
            "expguard_not_covered": "ExpGuard text is gated and never written to disk, so it is "
                                    "absent from this audit. Its disjointness from the teacher "
                                    "pool must be established separately under HF_TOKEN before "
                                    "any teacher training.",
            "thresholds": {"containment": CONTAINMENT_LEVELS, "jaccard": JACCARD_LEVELS},
        },
        "pairs": results,
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "overlap_audit.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print(f"roles: {payload['meta']['roles']}\n")
    for name, r in results.items():
        if "skipped" in r:
            print(f"{name}: skipped")
            continue
        d, f = r["digest"], r["fuzzy"]
        print(f"{name}")
        print(f"   exact={d['exact_n']}  normalized={d['normalized_n']} "
              f"(+{d['normalized_beyond_exact_n']} beyond exact)")
        print(f"   containment>=0.50: {f['containment_at_least']['0.50']}  "
              f">=0.80: {f['containment_at_least']['0.80']}  max={f['max_containment']:.3f}")
        print(f"   jaccard>=0.70: {f['jaccard_at_least']['0.70']}  "
              f">=0.90: {f['jaccard_at_least']['0.90']}  max={f['max_jaccard']:.3f}")
        print(f"   lineage: family={r['lineage']['family_id_collisions']} "
              f"upstream={r['lineage']['upstream_family_id_collisions']} "
              f"(measurable={r['lineage']['measurable']})")
        if f["n_unmeasurable_by_ngram"]:
            print(f"   note: {f['n_unmeasurable_by_ngram']} probe rows too short to n-gram")
    print(f"\nwrote {os.path.join(args.out, 'overlap_audit.json')}")


if __name__ == "__main__":
    main()
