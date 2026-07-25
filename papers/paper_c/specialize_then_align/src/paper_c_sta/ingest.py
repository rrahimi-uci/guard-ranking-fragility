"""Map existing repository corpora into the Paper C v2 sample contract.

The study is reuse-first.  Roughly twenty thousand rows already exist in this
repository -- MortgageGuardBench-2K, the frozen HMDA mortgage benchmark, and the
general-safety corpora -- and every one of them is already synthetic-or-public,
hashed, and family-organised.  Generating replacements would be both wasteful and
scientifically worse, because the existing sets carry provenance this module can
propagate rather than invent.

Two rules govern everything here.

*Fail closed on mapping.*  A source row is ingested only when its gold label
determines exactly one of ``allow`` / ``review`` / ``intervene``.  Rows whose gold
is a numeric answer, a multiple-choice id, or an action code with no three-action
reading are **skipped and counted**, never coerced.  ``ingest_report`` returns the
skip reasons so the manifest states what was dropped instead of implying full
coverage.

*Propagate provenance, never upgrade it.*  ``legal_review_status`` and
``licence_id`` are carried through from the source.  Nothing in this module can
turn a machine-validated row into a reviewed one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import re

import hashlib

from .contracts import (
    ContractError,
    canonical_json_bytes,
    canonical_sha256,
    output_path,
    read_json,
    sha256_file,
    validate_config,
    write_json,
)

CORE_CATEGORIES = (
    "toxicity_abuse",
    "jailbreak_evasion",
    "prompt_injection_exfiltration",
    "mortgage_fair_lending",
    "mortgage_closed_end_advertising",
)
ACTIONS = ("allow", "review", "intervene")

# --------------------------------------------------------------- action mapping
# MortgageGuardBench-2K states gold as one of several task-specific shapes.  Only
# the decision-bearing shapes have a three-action reading; the rest are skipped.
MGB2K_DECISION_TO_ACTION = {
    "compliant": "allow",
    "safe": "allow",
    "supported": "allow",
    "needs_human_review": "review",
    "noncompliant": "intervene",
    "unsafe": "intervene",
    "unsupported": "intervene",
}

# MGB2K domain/subdomain -> Paper C v2 core category.  Domains with no core-category
# reading (mortgage_math, income_assets, ...) are deliberately absent: they are
# valid mortgage material but not one of the five focal categories.
MGB2K_DOMAIN_TO_CATEGORY = {
    "fair_lending": "mortgage_fair_lending",
}
MGB2K_SUBDOMAIN_TO_CATEGORY = {
    "document_prompt_injection": "prompt_injection_exfiltration",
    "tool_data_exfiltration": "prompt_injection_exfiltration",
    "tool_prompt_injection": "prompt_injection_exfiltration",
    "prompt_injection": "prompt_injection_exfiltration",
    "training_on_pii": "prompt_injection_exfiltration",
    "pii_personal_email": "prompt_injection_exfiltration",
}

# General-safety corpora -> core category and upstream licence.  "unverified" is a
# real value: the licence ledger readiness gate must fail until a human resolves it.
GENERAL_SAFETY_SOURCES = {
    "toxicchat": ("toxicity_abuse", "cc-by-nc-4.0-upstream-unverified"),
    "beavertails": ("toxicity_abuse", "cc-by-nc-4.0-upstream-unverified"),
    "openai_moderation": ("toxicity_abuse", "mit-upstream-unverified"),
    "jailbreak_classification": ("jailbreak_evasion", "unverified"),
    "jailbreakbench": ("jailbreak_evasion", "mit-upstream-unverified"),
    "prompt_injections": ("prompt_injection_exfiltration", "unverified"),
}
# safepyramid is deliberately excluded.  It is a multi-label rule-attribution task
# over ten domains (defamation, IP, privacy, ...) with a mean of 6.2 violated rules
# per row and *no* clean rows at all.  Mapping it to jailbreak_evasion would be
# wrong on both the category and the label, and adding 3,000 all-intervene rows
# would worsen the class balance it appears to fix.
EXCLUDED_SOURCES = {
    "safepyramid": "multi_label_rule_attribution_no_benign_rows_and_non_jailbreak_domains",
}
SAFE_LABELS = {"safe", "benign", "0", "false", "allow"}
UNSAFE_LABELS = {"unsafe", "harmful", "1", "true", "intervene"}

_SLUG = re.compile(r"[^a-z0-9]+")


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _slug(text: str) -> str:
    return _SLUG.sub("_", str(text).strip().lower()).strip("_") or "unknown"


def _repo_root() -> Path:
    """The monorepo root, three levels above this workspace."""
    return output_path(".").resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _policy_context(config: Mapping, authority_ids: list[str], text: str) -> dict:
    """Bind a mortgage row to the configured snapshot object and vintage lock."""
    snapshot = read_json(output_path(config["mortgage_policy_path"]))
    inventory = read_json(output_path(config["mortgage_policy_vintage_inventory_path"]))
    return {
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_object_sha256": canonical_sha256(snapshot),
        "policy_vintage_lock_id": inventory["inventory_id"],
        "policy_as_of": snapshot["retrieved_on"],
        "authority_ids": sorted(set(authority_ids)) or ["UNBOUND"],
        "policy_text": text,
        "content_sha256": _sha_bytes(text.encode("utf-8")),
    }


# ------------------------------------------------------------------ MGB2K
def ingest_mgb2k(config: Mapping) -> tuple[list[dict], dict[str, int]]:
    root = _repo_root() / "data/mortgage_guard_bench_2k_v0_1_0"
    path = root / "data/mortgage_guard_bench_full.jsonl"
    if not path.is_file():
        return [], {"source_missing": 1}
    samples: list[dict] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for row in _read_jsonl(path):
        category = MGB2K_SUBDOMAIN_TO_CATEGORY.get(row.get("subdomain")) \
            or MGB2K_DOMAIN_TO_CATEGORY.get(row.get("domain"))
        if category is None:
            skip(f"domain_not_a_core_category:{row.get('domain')}")
            continue
        gold = row.get("gold_answer")
        if isinstance(gold, str):
            try:
                gold = json.loads(gold)
            except json.JSONDecodeError:
                gold = None
        if not isinstance(gold, Mapping) or "decision" not in gold:
            skip(f"gold_has_no_three_action_decision:{row.get('task_type')}")
            continue
        action = MGB2K_DECISION_TO_ACTION.get(str(gold["decision"]))
        if action is None:
            skip(f"unmapped_decision:{gold['decision']}")
            continue

        cards = row.get("policy_context") or []
        authority_ids = [c.get("card_id") for c in cards if isinstance(c, Mapping) and c.get("card_id")]
        policy_text = "\n".join(
            str(c.get("text", "")) for c in cards if isinstance(c, Mapping)
        ).strip() or "no policy card supplied by source"
        body = canonical_json_bytes(row)
        samples.append({
            "sample_id": f"MGB2K::{row['id']}",
            "family_id": f"mgb2k::{_slug(row.get('template_family', row['id']))}",
            "content_family_id": f"mgb2k::{_slug(row.get('subdomain', row.get('domain', 'x')))}",
            "category": category,
            "domain": "mortgage",
            "request": str(row.get("user_prompt") or row.get("scenario") or ""),
            "proposed_response": row.get("candidate_response"),
            "context": {
                "actor_role": row.get("task_type"),
                "product": row.get("loan_product"),
                "transaction_stage": row.get("loan_stage"),
                "applicable_regime": row.get("domain"),
                "coverage_facts": row.get("scenario"),
            },
            "jurisdiction": "US_federal",
            "policy_as_of": row.get("source_snapshot_date"),
            "temporal_evaluation_eligible": False,
            "temporal_policy_side": None,
            "split": None,
            "policy_context": _policy_context(config, authority_ids, policy_text),
            "gold": {
                "action": action,
                "category": category,
                "violation_tags": list(row.get("risk_tags") or []),
                "policy_ids": sorted(set(authority_ids)),
                "rationale": str(row.get("rationale") or "source rationale absent"),
                "reviewer_ids": [],
                "adjudicator_id": None,
            },
            "provenance": {
                "source_id": "mortgage_guard_bench_2k_v0_1_0",
                "content_sha256": _sha_bytes(body),
                "licence_id": "not_selected_upstream",
                "synthetic": bool(row.get("synthetic", True)),
                "contains_real_pii": bool(row.get("contains_real_pii", False)),
                "legal_review_status": row.get(
                    "legal_review_status",
                    "synthetic_machine_validated_not_counsel_reviewed",
                ),
                "upstream_split": row.get("split"),
                "difficulty": row.get("difficulty"),
            },
        })
    return samples, skipped


# ------------------------------------------------------- HMDA mortgage benchmark
def ingest_hmda(config: Mapping) -> tuple[list[dict], dict[str, int]]:
    root = _repo_root() / "mortgage-benchmark/benchmark/v1_hmda2022"
    samples: list[dict] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    # The benign rows are the fair-lending specificity partner: same generator, same
    # families, gold ALLOW.  Excluding them would leave the category with violations
    # only, which is exactly the pathology the specificity cohorts exist to prevent.
    keep_domains = {"fair_lending", "benign"}
    for split_file in sorted(root.glob("*.jsonl")):
        for row in _read_jsonl(split_file):
            domain = str(row.get("domain") or "")
            if domain not in keep_domains:
                skip(f"hmda_domain_not_core:{domain or 'missing'}")
                continue
            action = str(row.get("mortgage_policy_gold") or "").strip().lower()
            if action not in ACTIONS:
                skip(f"hmda_unmapped_policy_gold:{action or 'missing'}")
                continue
            g_label = row.get("general_safety_gold")
            cards = row.get("policy_context") or []
            if isinstance(cards, str):
                cards = [cards]
            authority_ids = [
                c.get("card_id") if isinstance(c, Mapping) else str(c) for c in cards
            ]
            body = canonical_json_bytes(row)
            samples.append({
                "sample_id": f"HMDA::{row.get('id', _sha_bytes(body)[:16])}",
                "family_id": f"hmda::{_slug(row.get('family_id') or row.get('id', 'x'))}",
                "content_family_id": f"hmda::{_slug(row.get('content_family') or row.get('subdomain') or domain)}",
                "category": "mortgage_fair_lending",
                "domain": "mortgage",
                "request": str(row.get("user_prompt") or row.get("scenario") or ""),
                "proposed_response": None,
                "context": {
                    "actor_role": row.get("role"),
                    "product": row.get("product"),
                    "transaction_stage": row.get("subdomain"),
                    "applicable_regime": domain,
                    "coverage_facts": row.get("scenario"),
                },
                "jurisdiction": "US_federal",
                "policy_as_of": row.get("policy_as_of"),
                "temporal_evaluation_eligible": False,
                "temporal_policy_side": None,
                "split": None,
                "policy_context": _policy_context(
                    config,
                    [a for a in authority_ids if a],
                    "\n".join(str(a) for a in authority_ids) or "hmda policy cards",
                ),
                "gold": {
                    "action": action,
                    "category": "mortgage_fair_lending",
                    "violation_tags": [t for t in (row.get("subdomain"), row.get("trap_type"), row.get("severity")) if t],
                    "policy_ids": sorted({str(a) for a in authority_ids if a}),
                    "rationale": str(row.get("rationale") or "hmda benchmark coding"),
                    "reviewer_ids": [],
                    "adjudicator_id": None,
                },
                "provenance": {
                    "source_id": "mortgage_benchmark_v1_hmda2022",
                    "content_sha256": _sha_bytes(body),
                    "licence_id": "repo_internal_synthetic",
                    "synthetic": True,
                    "contains_real_pii": False,
                    "legal_review_status": row.get(
                        "legal_review_status",
                        "synthetic_machine_validated_not_counsel_reviewed",
                    ),
                    "upstream_split": split_file.stem,
                    "general_safety_label_g": g_label,
                },
            })
    return samples, skipped


# ------------------------------------------------------------- general safety
def ingest_general_safety(config: Mapping) -> tuple[list[dict], dict[str, int]]:
    root = _repo_root() / "data/benchmarks/full"
    if not root.is_dir():
        root = _repo_root() / "data/benchmarks"
    samples: list[dict] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for stem, (category, licence) in sorted(GENERAL_SAFETY_SOURCES.items()):
        path = root / f"{stem}.jsonl"
        if not path.is_file():
            skip(f"missing_source:{stem}")
            continue
        for index, row in enumerate(_read_jsonl(path)):
            raw = row.get("label", row.get("gold"))
            token = str(raw).strip().lower()
            if token in SAFE_LABELS:
                action = "allow"
            elif token in UNSAFE_LABELS:
                action = "intervene"
            else:
                skip(f"unmapped_label:{stem}")
                continue
            text = str(row.get("prompt") or row.get("text") or row.get("input") or "")
            if not text.strip():
                skip(f"empty_prompt:{stem}")
                continue
            body = canonical_json_bytes(row)
            samples.append({
                "sample_id": f"{stem.upper()}::{index:06d}",
                # No upstream family structure: one family per row is the
                # conservative choice, because merging unrelated rows into a shared
                # family would understate leakage risk at split time.
                "family_id": f"{stem}::{index:06d}",
                "content_family_id": f"{stem}::{index:06d}",
                "category": category,
                "domain": "general_safety",
                "request": text,
                "proposed_response": row.get("response"),
                "context": None,
                "jurisdiction": None,
                "policy_as_of": None,
                "temporal_evaluation_eligible": False,
                "temporal_policy_side": None,
                "split": None,
                "policy_context": None,
                "gold": {
                    "action": action,
                    "category": category,
                    "violation_tags": list(row.get("tags") or []),
                    "policy_ids": [],
                    "rationale": "upstream binary safety label mapped to the action space",
                    "reviewer_ids": [],
                    "adjudicator_id": None,
                },
                "provenance": {
                    "source_id": stem,
                    "content_sha256": _sha_bytes(body),
                    "licence_id": licence,
                    "synthetic": False,
                    "contains_real_pii": False,
                    "legal_review_status": "public_corpus_not_counsel_reviewed",
                },
            })
    return samples, skipped


# --------------------------------------------------------------------- driver
INGESTORS = {
    "mgb2k": ingest_mgb2k,
    "hmda": ingest_hmda,
    "general_safety": ingest_general_safety,
}


def ingest_report(config: Mapping) -> dict:
    """Run every ingestor and return samples plus an honest coverage report."""
    validate_config(config)
    samples: list[dict] = []
    skipped: dict[str, dict[str, int]] = {}
    for name, fn in INGESTORS.items():
        rows, drops = fn(config)
        samples.extend(rows)
        skipped[name] = drops

    seen: set[str] = set()
    duplicates = 0
    unique: list[dict] = []
    for sample in samples:
        if sample["sample_id"] in seen:
            duplicates += 1
            continue
        seen.add(sample["sample_id"])
        unique.append(sample)

    by_category: dict[str, dict[str, int]] = {
        category: dict.fromkeys(ACTIONS, 0) for category in CORE_CATEGORIES
    }
    for sample in unique:
        by_category[sample["category"]][sample["gold"]["action"]] += 1
    empty = sorted(c for c in CORE_CATEGORIES if not sum(by_category[c].values()))
    return {
        "samples": unique,
        "report": {
            "total_ingested": len(unique),
            "duplicate_sample_ids_dropped": duplicates,
            "by_category": by_category,
            "families": len({s["family_id"] for s in unique}),
            "content_families": len({s["content_family_id"] for s in unique}),
            "skipped": skipped,
            "core_categories_with_no_rows": empty,
            "gap_generation_required": empty,
            "legal_review_status": "no source is counsel reviewed; every row is "
                                   "machine-validated or public-corpus tier",
        },
    }


def write_ingest(config: Mapping, *, out_dir: str = "artifacts/ingest") -> dict:
    result = ingest_report(config)
    target = output_path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows_path = target / "samples.jsonl"
    with rows_path.open("w") as handle:
        for sample in result["samples"]:
            handle.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")
    report = dict(result["report"])
    report["samples_path"] = str(rows_path.relative_to(output_path(".")))
    report["samples_sha256"] = sha256_file(rows_path)
    write_json(target / "ingest_report.json", report)
    return report
