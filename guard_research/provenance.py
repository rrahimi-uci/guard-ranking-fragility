"""Provenance helpers: text normalization, content/object/file hashing, and a
dependency-light character-n-gram MinHash for near-duplicate family detection.

Normalization and hashing here define the frozen rules referenced by the data
audit (plan sec 6.4.1 step 3 and sec 6.7): Unicode NFKC, lowercase, collapsed
whitespace, stripped, punctuation preserved. Content hashes are always taken
over the *normalized* text (never the raw text), so exact-overlap and family
detection are invariant to casing / whitespace / compatibility-form noise.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata

import numpy as np

__all__ = [
    "normalize_text",
    "content_sha256",
    "sha256_of_obj",
    "sha256_of_file",
    "minhash_signature",
    "estimated_jaccard",
    "MINHASH_JACCARD_THRESHOLD",
]

# Prespecified candidate-generation threshold for near-duplicate families
# (plan sec 6.7): add an undirected edge when estimated Jaccard >= 0.85. This is
# a *candidate* threshold for adjudication, NOT proof of semantic equivalence.
MINHASH_JACCARD_THRESHOLD = 0.85

# 31-bit Mersenne prime for the compact MinHash universal-hash permutations.
# Keeps a*base < 2**62 so uint64 arithmetic never overflows.
_MERSENNE_31 = (1 << 31) - 1
# Fixed seed so signatures are reproducible across processes/runs.
_MINHASH_SEED = 20260712


def normalize_text(t) -> str:
    """Frozen normalization rule: NFKC, lowercase, collapse whitespace, strip.

    Punctuation is preserved. Idempotent, so it is safe to call on text that may
    already be normalized. ``None`` becomes the empty string.
    """
    t = "" if t is None else str(t)
    t = unicodedata.normalize("NFKC", t)
    t = t.lower()
    # split() on no args splits on any run of whitespace and drops leading/
    # trailing whitespace; join with a single space -> collapse + strip.
    t = " ".join(t.split())
    return t


def content_sha256(t) -> str:
    """SHA-256 hex digest of the *normalized* text."""
    return hashlib.sha256(normalize_text(t).encode("utf-8")).hexdigest()


def sha256_of_obj(obj) -> str:
    """SHA-256 hex digest of an object serialized as canonical JSON.

    Canonical form uses ``sort_keys=True`` and compact separators so equivalent
    configs/manifests hash identically regardless of key order or whitespace.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_of_file(path) -> str:
    """SHA-256 hex digest of a file's bytes (streamed in chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _char_shingles(norm_text: str, ngram: int):
    """Set of character n-grams over normalized text.

    Short strings (shorter than one n-gram) contribute the whole string as a
    single shingle so they still get a stable signature.
    """
    if not norm_text:
        return []
    if len(norm_text) < ngram:
        return [norm_text]
    return list({norm_text[i : i + ngram] for i in range(len(norm_text) - ngram + 1)})


def _base_hash(shingle: str) -> int:
    """Stable 31-bit base hash of a shingle (blake2b -> int -> mod prime)."""
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % _MERSENNE_31


def _perm_coeffs(num_perm: int):
    """Deterministic (a, b) universal-hash coefficients for ``num_perm`` slots.

    Reseeded from a fixed seed on every call, so two signatures computed with
    the same ``num_perm`` use the same permutations and are directly comparable.
    """
    rs = np.random.RandomState(_MINHASH_SEED)
    a = rs.randint(1, _MERSENNE_31, size=num_perm).astype(np.uint64)
    b = rs.randint(0, _MERSENNE_31, size=num_perm).astype(np.uint64)
    return a, b


def minhash_signature(text, num_perm: int = 256, ngram: int = 5) -> np.ndarray:
    """MinHash signature over character ``ngram``-grams of the normalized text.

    Uses ``datasketch`` if it is importable, otherwise a compact numpy
    implementation (universal hashing of blake2b shingle hashes). Both backends
    return a length-``num_perm`` numpy array of hash values; compare two
    signatures from the *same backend and num_perm* with :func:`estimated_jaccard`.
    Do not mix signatures produced by different backends.

    Text is normalized internally via :func:`normalize_text`, matching the
    frozen family-construction rule (plan sec 6.7): NFKC + lowercase +
    collapsed whitespace before shingling.
    """
    norm = normalize_text(text)
    shingles = _char_shingles(norm, ngram)

    # Preferred backend: datasketch (if the reproducibility environment has it).
    try:
        from datasketch import MinHash  # type: ignore
    except Exception:
        MinHash = None

    if MinHash is not None:
        m = MinHash(num_perm=num_perm)
        for sh in shingles:
            m.update(sh.encode("utf-8"))
        return np.asarray(m.hashvalues, dtype=np.uint64)

    # Compact fallback: signature[i] = min_s ((a_i * base_s + b_i) mod prime).
    a, b = _perm_coeffs(num_perm)
    if not shingles:
        # No content -> fill with the prime sentinel so two empty texts match.
        return np.full(num_perm, _MERSENNE_31, dtype=np.uint64)
    bases = np.array([_base_hash(sh) for sh in shingles], dtype=np.uint64)
    # (num_perm, S) matrix of hashed shingle values, then min over shingles.
    hashed = (a[:, None] * bases[None, :] + b[:, None]) % np.uint64(_MERSENNE_31)
    return hashed.min(axis=1).astype(np.uint64)


def estimated_jaccard(sig_a, sig_b) -> float:
    """Estimated Jaccard similarity = fraction of matching MinHash slots.

    Signatures must be the same length (same ``num_perm``) and from the same
    backend. The candidate near-duplicate threshold is
    :data:`MINHASH_JACCARD_THRESHOLD` (0.85).
    """
    a = np.asarray(sig_a)
    b = np.asarray(sig_b)
    if a.shape != b.shape:
        raise ValueError(f"signature length mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 1.0
    return float(np.mean(a == b))
