"""Manifest-construction helpers for Paper A (builder + audit + tests share this).

Hashing / normalization / MinHash come from ``guard_research.provenance`` (the
single source of truth, plan sec 13.1). Family clustering (LSH banding + union
find), deterministic hash-ranking, and the calibration/ID family split are built
here on top of those primitives so the manifest builder, the split audit, and
the tests are byte-identical by construction.

If ``guard_research.provenance`` is not importable (package not landed yet), a
minimal local fallback matching its frozen rules is used.
TODO(paper-a): drop the fallback once guard_research.provenance is guaranteed.

(This module is intentionally separate from ``experiments/paper_a_common.py``,
which serves the lock/train/eval/analyze pipeline.)
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

# --- make repo-root + experiments/ importable regardless of entrypoint --------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- provenance primitives: prefer guard_research, else local fallback --------
try:
    from guard_research.provenance import (  # type: ignore
        MINHASH_JACCARD_THRESHOLD,
        content_sha256,
        estimated_jaccard,
        minhash_signature,
        normalize_text,
        sha256_of_file,
        sha256_of_obj,
    )

    PROVENANCE_SOURCE = "guard_research.provenance"
except Exception:  # pragma: no cover - fallback only when package absent
    # TODO(paper-a): remove this fallback once guard_research.provenance ships.
    import json as _json
    import unicodedata as _ud

    PROVENANCE_SOURCE = "paper_a_manifest_lib:local_fallback"
    MINHASH_JACCARD_THRESHOLD = 0.85
    _MERSENNE_31 = (1 << 31) - 1
    _MINHASH_SEED = 20260712

    def normalize_text(t) -> str:
        t = "" if t is None else str(t)
        t = _ud.normalize("NFKC", t).lower()
        return " ".join(t.split())

    def content_sha256(t) -> str:
        return hashlib.sha256(normalize_text(t).encode("utf-8")).hexdigest()

    def sha256_of_obj(obj) -> str:
        payload = _json.dumps(obj, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def sha256_of_file(path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    def _char_shingles(norm_text, ngram):
        if not norm_text:
            return []
        if len(norm_text) < ngram:
            return [norm_text]
        return list({norm_text[i : i + ngram] for i in range(len(norm_text) - ngram + 1)})

    def _base_hash(sh):
        d = hashlib.blake2b(sh.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(d, "little") % _MERSENNE_31

    def _perm_coeffs(num_perm):
        rs = np.random.RandomState(_MINHASH_SEED)
        a = rs.randint(1, _MERSENNE_31, size=num_perm).astype(np.uint64)
        b = rs.randint(0, _MERSENNE_31, size=num_perm).astype(np.uint64)
        return a, b

    def minhash_signature(text, num_perm=256, ngram=5) -> np.ndarray:
        norm = normalize_text(text)
        sh = _char_shingles(norm, ngram)
        a, b = _perm_coeffs(num_perm)
        if not sh:
            return np.full(num_perm, _MERSENNE_31, dtype=np.uint64)
        bases = np.array([_base_hash(s) for s in sh], dtype=np.uint64)
        hashed = (a[:, None] * bases[None, :] + b[:, None]) % np.uint64(_MERSENNE_31)
        return hashed.min(axis=1).astype(np.uint64)

    def estimated_jaccard(sig_a, sig_b) -> float:
        a = np.asarray(sig_a)
        b = np.asarray(sig_b)
        if a.shape != b.shape:
            raise ValueError("signature length mismatch")
        if a.size == 0:
            return 1.0
        return float(np.mean(a == b))


# --- frozen family-construction parameters (plan sec 6.7) ---------------------
NGRAM = 5
NUM_PERM = 256
LSH_BANDS = 32
LSH_ROWS = 8  # LSH_BANDS * LSH_ROWS must equal NUM_PERM
assert LSH_BANDS * LSH_ROWS == NUM_PERM

# Canonical split names, aligned with the plan sec 6.6 manifest file stems.
SPLIT_TRAIN = "train"
SPLIT_CALIBRATION = "calibration"
SPLIT_ID = "id_test"
SPLIT_TRANSFER = "transfer_test"
SPLIT_ORBENCH = "orbench_safe_stress"
SPLIT_HARMBENCH = "harmbench_positive_stress"

MANIFEST_FILES = [
    "train",
    "calibration",
    "id_test",
    "transfer_test",
    "orbench_safe_stress",
    "harmbench_positive_stress",
]
# The five splits a training/validation path must never read (fail-closed set).
EVAL_SPLITS = {SPLIT_CALIBRATION, SPLIT_ID, SPLIT_TRANSFER, SPLIT_ORBENCH, SPLIT_HARMBENCH}

# Canonical row-schema fields (plan sec 6.5).
ROW_SCHEMA_FIELDS = [
    "sample_id",
    "source",
    "source_config",
    "source_revision",
    "source_row_id",
    "split",
    "label",
    "label_provenance",
    "text_or_download_reference",
    "content_sha256",
    "family_id",
    "license_id",
    "redistribution_class",
    "known_overlap_disposition",
]


def to_gold(label) -> int:
    """Map a label to gold 1(unsafe)/0(safe)."""
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)):
        return 1 if int(label) == 1 else 0
    s = str(label).strip().lower()
    if s in ("unsafe", "harmful", "1", "true"):
        return 1
    if s in ("safe", "unharmful", "benign", "0", "false"):
        return 0
    raise ValueError(f"unrecognized label: {label!r}")


# --- deterministic hash ranking (plan sec 6.4.1 step 6) -----------------------
def rank_key(data_seed, source_row_id, content_hash) -> str:
    """SHA-256 hex of (data_seed, source_row_id, content_sha256); sort ascending."""
    return hashlib.sha256(
        f"{data_seed}|{source_row_id}|{content_hash}".encode("utf-8")
    ).hexdigest()


def family_sort_key(data_seed, source, family_id) -> str:
    """SHA-256 hex of (data_seed, source, family_id) for the cal/ID family order."""
    return hashlib.sha256(
        f"{data_seed}|{source}|{family_id}".encode("utf-8")
    ).hexdigest()


# --- union-find ---------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# --- LSH banding over MinHash signatures --------------------------------------
def lsh_candidate_pairs(sigs, bands=LSH_BANDS, rows=LSH_ROWS):
    """Set of candidate index pairs (i<j) sharing a band bucket.

    Banding (32 bands x 8 rows) over 256 permutations puts the S-curve midpoint
    near Jaccard 0.65, so all pairs with true Jaccard >= 0.80 are recalled with
    probability ~0.997+ and then exactly re-scored by the caller.
    """
    pairs = set()
    n = len(sigs)
    for band in range(bands):
        s = band * rows
        e = s + rows
        buckets = {}
        for i in range(n):
            key = sigs[i][s:e].tobytes()
            buckets.setdefault(key, []).append(i)
        for idxs in buckets.values():
            if len(idxs) > 1:
                for a in range(len(idxs)):
                    ia = idxs[a]
                    for b in range(a + 1, len(idxs)):
                        ib = idxs[b]
                        pairs.add((ia, ib) if ia < ib else (ib, ia))
    return pairs


def build_minhash_signatures(texts):
    """MinHash signatures for a list of texts (list of np.uint64 arrays)."""
    return [minhash_signature(t, num_perm=NUM_PERM, ngram=NGRAM) for t in texts]


def edges_at_threshold(sigs, cand_pairs, threshold):
    """List of (i, j, est_jaccard) with estimated Jaccard >= threshold."""
    out = []
    for (i, j) in cand_pairs:
        est = estimated_jaccard(sigs[i], sigs[j])
        if est >= threshold:
            out.append((i, j, est))
    return out


def connected_components(n, edges, extra_unions=None):
    """Union-find components. edges: iterable of (i, j, ...); extra_unions: (i, j)."""
    uf = UnionFind(n)
    for e in edges:
        uf.union(e[0], e[1])
    if extra_unions:
        for (i, j) in extra_unions:
            uf.union(i, j)
    comp = {}
    for i in range(n):
        comp.setdefault(uf.find(i), []).append(i)
    return list(comp.values())


def family_id_for_component(member_content_hashes) -> str:
    """family_id = sha256 of the lexicographically smallest content hash (plan 6.7 step 7)."""
    smallest = min(member_content_hashes)
    return hashlib.sha256(smallest.encode("utf-8")).hexdigest()


def build_families(texts, content_hashes, upstream_edges=None, sigs=None):
    """Assign a family_id to every row (plan sec 6.7 steps 3-7).

    texts/content_hashes are aligned lists. upstream_edges is an iterable of
    (i, j) authoritative-family edges. Returns
    (family_of, sigs, cand_pairs, edges_085, comps, stats).
    """
    n = len(content_hashes)
    if sigs is None:
        sigs = build_minhash_signatures(texts)
    cand = lsh_candidate_pairs(sigs)
    edges = edges_at_threshold(sigs, cand, MINHASH_JACCARD_THRESHOLD)
    up = list(upstream_edges) if upstream_edges else []
    comps = connected_components(n, edges, extra_unions=up)
    family_of = [None] * n
    multi = 0
    for members in comps:
        fid = family_id_for_component([content_hashes[m] for m in members])
        for m in members:
            family_of[m] = fid
        if len(members) > 1:
            multi += 1
    stats = {
        "n_rows": n,
        "n_candidate_pairs": len(cand),
        "n_minhash_edges_085": len(edges),
        "n_upstream_edges": len(up),
        "n_components": len(comps),
        "n_multi_member_components": multi,
        "num_perm": NUM_PERM,
        "ngram": NGRAM,
        "lsh_bands": LSH_BANDS,
        "lsh_rows": LSH_ROWS,
        "threshold": MINHASH_JACCARD_THRESHOLD,
        "provenance_source": PROVENANCE_SOURCE,
    }
    return family_of, sigs, cand, edges, comps, stats


# --- calibration / ID family-level split (plan sec 6.4.2) ---------------------
def split_calibration_id(source_rows, source, data_seed, cal_frac=0.40):
    """Greedy whole-family 40/60 calibration/ID split for one represented source.

    source_rows: list of dicts each with 'family_id'. Returns
    (calibration_family_ids: set, assignment: dict family_id -> split_label).
    Adds families (in the frozen sha256 order) to calibration while doing so does
    not move the calibration row count strictly farther from the 40% target; the
    first family that would move it farther, and all remaining families, go to ID.
    """
    fam_rows = {}
    for r in source_rows:
        fam_rows.setdefault(r["family_id"], []).append(r)
    total = len(source_rows)
    target = cal_frac * total
    fams = sorted(fam_rows, key=lambda f: family_sort_key(data_seed, source, f))
    cal_ids = set()
    cal_count = 0
    stopped = False
    for f in fams:
        if stopped:
            continue
        sz = len(fam_rows[f])
        if abs((cal_count + sz) - target) <= abs(cal_count - target):
            cal_ids.add(f)
            cal_count += sz
        else:
            stopped = True
    assignment = {f: (SPLIT_CALIBRATION if f in cal_ids else SPLIT_ID) for f in fams}
    return cal_ids, assignment
