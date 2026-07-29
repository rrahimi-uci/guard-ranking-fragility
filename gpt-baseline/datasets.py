"""Benchmark loaders for the GPT baseline.

Every loader returns a list of text-free-serialisable ``Record``s. A record carries
the prompt text in memory only; nothing written under ``gpt-baseline/raw/`` ever
contains benchmark prose, because most sources in
``benchmarks/registry/distribution.yaml`` are ``local_only`` or ``text_free_only``.
Rows are keyed by their upstream id when they have one and by a sha256 prefix of the
text when they do not -- the same convention as
``experiments/eval_expguard_external.py``.

Four task families, because "precision/recall/F1/AUC" needs a binary (or per-rule
binary) target and the benchmarks do not all natively have one:

``prompt_safety``          native safe/unsafe prompt labels (7 general-safety corpora
                           plus the in-house hard-negative set).
``mortgage_intervention``  v1_hmda2022 ``mortgage_policy_gold`` allow/intervene.
``compliance_audit``       MortgageGuardBench-2K structured-label rows: is the
                           candidate response acceptable as-is, or not?
``rule_attribution``       SafePyramid: which of the supplied numbered rules does the
                           conversation violate? Scored per (row, rule) pair.

Label polarity is uniform: ``label == 1`` is the positive class and always means
"the guard should act" (unsafe / intervene / problematic / rule violated).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ─────────────────────────────────────────────────────────────────── label vocabularies
SAFE_LABELS = {"safe", "benign", "0", "false", "allow"}
UNSAFE_LABELS = {"unsafe", "harmful", "1", "true", "intervene", "flag"}

# MortgageGuardBench-2K decision -> positive(1) / negative(0). The three
# structured-label task types use three different decision vocabularies; all of them
# reduce to "is the candidate response acceptable as-is". `needs_human_review` is
# positive: a guard that waves it through has failed the row.
MGB2K_DECISIONS = {
    "compliant": 0,
    "noncompliant": 1,
    "needs_human_review": 1,
    "safe": 0,
    "unsafe": 1,
    "supported": 0,
    "unsupported": 1,
}

GENERAL_SAFETY_SOURCES = (
    "beavertails",
    "jailbreak_classification",
    "jailbreakbench",
    "openai_moderation",
    "prompt_injections",
    "toxicchat",
    "xstest",
)

_RULE_NUM = re.compile(r"(?m)^(\d+)\.\s")


@dataclass
class Record:
    rid: str
    benchmark: str
    task: str
    text: str
    label: int | None = None
    # rule_attribution only
    gold_rules: tuple[int, ...] = ()
    candidate_rules: tuple[int, ...] = ()
    meta: dict = field(default_factory=dict)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class SkipLog:
    """Counts dropped rows by reason so the report can state what was excluded."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def skip(self, reason: str) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1


# ───────────────────────────────────────────────────────────── general safety corpora
def load_general_safety(name: str, skips: SkipLog) -> list[Record]:
    path = REPO / "data/benchmarks/full" / f"{name}.jsonl"
    if not path.is_file():  # the sampled sets are the fallback, as in paper_c ingest
        path = REPO / "data/benchmarks" / f"{name}.jsonl"
    if not path.is_file():
        skips.skip(f"missing_source:{name}")
        return []
    out, seen = [], set()
    for row in _read_jsonl(path):
        token = str(row.get("label", row.get("gold"))).strip().lower()
        if token in SAFE_LABELS:
            label = 0
        elif token in UNSAFE_LABELS:
            label = 1
        else:
            skips.skip(f"unmapped_label:{name}:{token}")
            continue
        text = str(row.get("text") or row.get("prompt") or row.get("input") or "")
        if not text.strip():
            skips.skip(f"empty_prompt:{name}")
            continue
        rid = _text_id(text)
        if rid in seen:  # identical prompts appear more than once upstream
            skips.skip(f"duplicate_text:{name}")
            continue
        seen.add(rid)
        out.append(Record(rid=rid, benchmark=name, task="prompt_safety", text=text,
                          label=label, meta={"hazard": row.get("hazard")}))
    return out


# ──────────────────────────────────────────────────────── in-house hard negatives
def load_guard_hard(skips: SkipLog) -> list[Record]:
    path = REPO / "data/guard_benchmark_hard.jsonl"
    if not path.is_file():
        skips.skip("missing_source:guard_benchmark_hard")
        return []
    out = []
    for row in _read_jsonl(path):
        token = str(row.get("label_binary", "")).strip().lower()
        if token in SAFE_LABELS:
            label = 0
        elif token in UNSAFE_LABELS:
            label = 1
        else:
            skips.skip(f"unmapped_label:guard_benchmark_hard:{token}")
            continue
        text = str(row.get("text") or "")
        if not text.strip():
            skips.skip("empty_prompt:guard_benchmark_hard")
            continue
        out.append(Record(rid=str(row["id"]), benchmark="guard_benchmark_hard",
                          task="prompt_safety", text=text, label=label,
                          meta={"difficulty": row.get("difficulty"),
                                "trap_type": row.get("trap_type")}))
    return out


# ─────────────────────────────────────────────────────── mortgage: v1_hmda2022 tests
def load_hmda(skips: SkipLog) -> list[Record]:
    """Prompt-only mortgage intervention on the two held-out splits.

    ``policy_context`` on these rows is a list of bare card ids (``["D01", ...]``) with
    no text in the row, so it is not shown to the model: this is the prompt-only guard
    formulation Paper A uses, not a supplied-policy task.
    """
    root = REPO / "mortgage-benchmark/benchmark/v1_hmda2022"
    out = []
    for split in ("public_test", "private_test"):
        path = root / f"{split}.jsonl"
        if not path.is_file():
            skips.skip(f"missing_source:hmda:{split}")
            continue
        for row in _read_jsonl(path):
            token = str(row.get("mortgage_policy_gold") or "").strip().lower()
            if token == "allow":
                label = 0
            elif token == "intervene":
                label = 1
            else:
                skips.skip(f"unmapped_policy_gold:hmda:{token or 'missing'}")
                continue
            text = str(row.get("user_prompt") or "")
            if not text.strip():
                skips.skip("empty_prompt:hmda")
                continue
            out.append(Record(rid=str(row["id"]), benchmark="mortgage_hmda2022",
                              task="mortgage_intervention", text=text, label=label,
                              meta={"split": split, "domain": row.get("domain"),
                                    "quadrant": row.get("quadrant")}))
    return out


# ────────────────────────────────────────────────── mortgage: MortgageGuardBench-2K
def _mgb2k_input(row: dict) -> str:
    cards = row.get("policy_context") or []
    lines = []
    for card in cards:
        if isinstance(card, dict):
            cid = card.get("card_id", "CARD")
            prec = card.get("precedence")
            head = f"[{cid}]" + (f" (precedence {prec})" if prec is not None else "")
            lines.append(f"{head} {card.get('text', '')}".strip())
        else:
            lines.append(f"[{card}]")
    policy = "\n".join(lines) or "no policy card supplied by the source"
    return (
        f"POLICY CARDS\n{policy}\n\n"
        f"TASK\n{row.get('user_prompt', '')}\n\n"
        f"SCENARIO\n{row.get('scenario', '')}\n\n"
        f"CANDIDATE RESPONSE\n{row.get('candidate_response', '')}"
    )


def load_mgb2k(skips: SkipLog) -> list[Record]:
    """The binary-mappable slice of the 300-row test split.

    Only the 90 ``structured_label`` rows carry a decision that reduces to a binary
    guard verdict. The other 210 rows are multiple-choice action selection and numeric
    calculation, which have no positive/negative class at all, so precision, recall,
    F1 and AUC are undefined for them; they are counted in the skip log rather than
    coerced into a label.
    """
    path = REPO / "data/mortgage_guard_bench_2k_v0_1_0/splits/test_answer_key.jsonl"
    if not path.is_file():
        skips.skip("missing_source:mgb2k")
        return []
    out = []
    for row in _read_jsonl(path):
        gold = row.get("gold_answer")
        if isinstance(gold, str):
            try:
                gold = json.loads(gold)
            except json.JSONDecodeError:
                gold = None
        if not isinstance(gold, dict) or "decision" not in gold:
            skips.skip(f"no_binary_decision:mgb2k:{row.get('task_type')}")
            continue
        label = MGB2K_DECISIONS.get(str(gold["decision"]).strip().lower())
        if label is None:
            skips.skip(f"unmapped_decision:mgb2k:{gold['decision']}")
            continue
        out.append(Record(rid=str(row["id"]), benchmark="mortgage_guard_bench_2k",
                          task="compliance_audit", text=_mgb2k_input(row), label=label,
                          meta={"task_type": row.get("task_type"),
                                "gold_decision": gold["decision"],
                                "difficulty": row.get("difficulty")}))
    return out


# ────────────────────────────────────────────── expert domains: finance/health/law
EXPGUARD_DATASET = "6rightjade/expguardmix"
EXPGUARD_TEST_FILE = "expguardtest.parquet"


def load_expguard(skips: SkipLog) -> list[Record]:
    """ExpGuard test split: expert-annotated prompts in finance, healthcare and law.

    The only benchmark here whose text is *not* on disk. ExpGuard is gated, so this repo
    commits only text-free artifacts for it (``artifacts/expguard_external/``:
    per-checkpoint scores plus a ``{row_hash -> label, domain}`` index). The prompts are
    fetched from the Hub with ``HF_TOKEN``; nothing this module reads is written back out.

    Row ids are ``sha256(prompt)[:16]``, byte-identical to ``_row_id`` in
    ``experiments/eval_expguard_external.py``, and rows are de-duplicated by id keeping
    the first -- both deliberate, so these predictions join directly against the
    committed ``labels_index.json`` and the four local checkpoints' score files, and the
    GPT numbers land on exactly the rows the local baseline was computed on.

    Task is prompt-only ``prompt_label`` classification, matching Paper A's formulation,
    so ``baseline_expguard.json`` is a like-for-like comparison. ``response`` and
    ``response_label`` exist upstream and are ignored.
    """
    import os

    token = os.environ.get("HF_TOKEN")
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(EXPGUARD_DATASET, EXPGUARD_TEST_FILE,
                               repo_type="dataset", token=token)
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - gated dataset, offline, or no token
        skips.skip(f"expguard_unavailable:{type(exc).__name__}")
        return []

    out, seen = [], set()
    for _, row in frame.iterrows():
        prompt = str(row["prompt"])
        token_label = str(row["prompt_label"]).strip().lower()
        if token_label in SAFE_LABELS:
            label = 0
        elif token_label in UNSAFE_LABELS:
            label = 1
        else:
            skips.skip(f"unmapped_label:expguard:{token_label}")
            continue
        rid = _text_id(prompt)
        if rid in seen:
            skips.skip("duplicate_text:expguard")
            continue
        seen.add(rid)
        out.append(Record(rid=rid, benchmark="expguard", task="prompt_safety",
                          text=prompt, label=label,
                          meta={"domain": str(row["domain"]).strip().lower(),
                                "prompt_category": row.get("prompt_category")}))
    return out


# ───────────────────────────────────────────────────────────────────── SafePyramid
def load_safepyramid(skips: SkipLog) -> list[Record]:
    """Native violated-rule-set task, scored per (row, rule) pair.

    The candidate universe is the rule numbers actually enumerated in the row's
    ``policy`` text -- that is the set the model can choose from. It is a superset of
    the row's ``rule_ids`` field, which is in turn a superset of ``gold``; the
    difference only widens the negative pool, and is reported.
    """
    path = REPO / "data/benchmarks/safepyramid.jsonl"
    if not path.is_file():
        skips.skip("missing_source:safepyramid")
        return []
    out = []
    for row in _read_jsonl(path):
        policy = str(row.get("policy") or "")
        universe = tuple(sorted({int(n) for n in _RULE_NUM.findall(policy)}))
        gold = tuple(sorted(int(g) for g in (row.get("gold") or [])))
        if not universe:
            skips.skip("no_numbered_rules:safepyramid")
            continue
        if not set(gold) <= set(universe):
            skips.skip("gold_outside_rule_universe:safepyramid")
            continue
        text = (
            f"CONVERSATION\n{row.get('conversation', '')}\n\n"
            f"POLICY\n{policy}"
        )
        out.append(Record(rid=str(row["id"]), benchmark="safepyramid",
                          task="rule_attribution", text=text, gold_rules=gold,
                          candidate_rules=universe,
                          meta={"level": row.get("level"), "domain": row.get("domain"),
                                "n_scored_rule_ids": len(row.get("rule_ids") or [])}))
    return out


# ────────────────────────────────────────────────────────────────────────── registry
BENCHMARK_GROUPS = {
    "general_safety": GENERAL_SAFETY_SOURCES,
    "guard_hard": ("guard_benchmark_hard",),
    "mortgage": ("mortgage_hmda2022", "mortgage_guard_bench_2k"),
    "expert_domains": ("expguard",),
    "safepyramid": ("safepyramid",),
}
# Benchmarks whose rows carry a natural sub-population worth reporting separately.
SLICE_BY = {"expguard": "domain", "safepyramid": "level"}
ALL_BENCHMARKS = tuple(b for group in BENCHMARK_GROUPS.values() for b in group)


def load_benchmark(name: str, skips: SkipLog) -> list[Record]:
    if name in GENERAL_SAFETY_SOURCES:
        return load_general_safety(name, skips)
    if name == "expguard":
        return load_expguard(skips)
    if name == "guard_benchmark_hard":
        return load_guard_hard(skips)
    if name == "mortgage_hmda2022":
        return load_hmda(skips)
    if name == "mortgage_guard_bench_2k":
        return load_mgb2k(skips)
    if name == "safepyramid":
        return load_safepyramid(skips)
    raise KeyError(f"unknown benchmark {name!r}; known: {', '.join(ALL_BENCHMARKS)}")


def load_all(names=None, limit: int | None = None
             ) -> tuple[dict[str, list[Record]], dict[str, int]]:
    skips = SkipLog()
    out = {}
    for name in (names or ALL_BENCHMARKS):
        rows = load_benchmark(name, skips)
        if limit:
            rows = rows[:limit]
        out[name] = rows
    return out, skips.counts


if __name__ == "__main__":  # quick census
    data, skipped = load_all()
    total = 0
    for name, rows in data.items():
        pos = sum(1 for r in rows if r.label == 1)
        rules = sum(len(r.candidate_rules) for r in rows)
        violated = sum(len(r.gold_rules) for r in rows)
        extra = f", {rules} (row,rule) pairs, {violated} violated" if rules else ""
        print(f"{name:28s} n={len(rows):6d} positives={pos:6d}{extra}")
        total += len(rows)
    print(f"{'TOTAL':28s} n={total:6d}")
    print("skipped:", json.dumps(skipped, indent=2, sort_keys=True))
