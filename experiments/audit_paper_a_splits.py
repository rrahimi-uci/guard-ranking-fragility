#!/usr/bin/env python
"""Audit the Paper A manifests (plan sec 6.7) and enforce the hard assertions.

Computes (over the written manifests, independently of the builder's bookkeeping):
  1. normalized exact-text overlap, train vs each eval manifest;
  2. conflicting-label exact overlap (reported separately from same-label);
  3. char-5gram MinHash near-duplicate sensitivity at 0.80 / 0.85 / 0.90;
  4. source-family membership validation (train/eval family disjointness);
  5. class and row counts;
  6. source-revision presence;
  7. license inventory;
  8. family/cluster construction validation;
  9. train/calibration/test role validation;
 10. proof that OR-Bench and BeaverTails counts in training are zero.

Hard assertions (EXIT NONZERO if any fails):
  or_bench_train_count == 0
  beavertails_train_count == 0
  exact_train_vs_eval_overlap == 0
  conflicting_label_overlap == 0
  every_row_has_source_revision == true
  every_row_has_content_hash == true
  every_near_duplicate_candidate_has_disposition == true

Usage:
  .venv/bin/python experiments/audit_paper_a_splits.py \
    --config configs/paper_a_sft.yaml \
    --manifest artifacts/paper_a_sft/manifests/manifest.json \
    --out artifacts/paper_a_sft/audit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_manifest_lib as L  # noqa: E402

TRAIN_SPLIT = L.SPLIT_TRAIN
EVAL_SPLITS = [L.SPLIT_CALIBRATION, L.SPLIT_ID, L.SPLIT_TRANSFER, L.SPLIT_ORBENCH, L.SPLIT_HARMBENCH]
REPRESENTED_SOURCES = {"toxicchat", "prompt_injections", "jailbreak_classification"}
TRANSFER_SOURCES = {"jailbreakbench", "xstest", "wildguardtest", "wildjailbreak"}
FORBIDDEN_TRAIN_SOURCES = {"orbench", "or_bench", "or-bench", "beavertails", "beaver_tails"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_manifests(manifest_dir):
    out = {}
    for stem in L.MANIFEST_FILES:
        path = os.path.join(manifest_dir, f"{stem}.jsonl")
        out[stem] = load_jsonl(path)
    return out


def audit(config_path, manifest_json_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    manifest_dir = os.path.dirname(manifest_json_path)
    manifest_meta = json.load(open(manifest_json_path))
    M = read_manifests(manifest_dir)

    all_rows = [r for rows in M.values() for r in rows]
    train_rows = M["train"]
    eval_stems = ["calibration", "id_test", "transfer_test",
                  "orbench_safe_stress", "harmbench_positive_stress"]
    eval_rows = [r for s in eval_stems for r in M[s]]

    report = {"manifest_dir": os.path.relpath(manifest_dir, _ROOT),
              "config_sha256": manifest_meta.get("config_sha256"),
              "provenance": manifest_meta.get("provenance")}

    # ---- 5. class + row counts ---------------------------------------------
    counts = {}
    for stem, rows in M.items():
        counts[stem] = {
            "n_rows": len(rows),
            "class_counts": dict(Counter(r["label"] for r in rows)),
            "per_source": {s: dict(Counter(r["label"] for r in rows if r["source"] == s))
                           for s in sorted({r["source"] for r in rows})},
        }
    report["counts"] = counts

    # ---- 6/? every row has revision + content hash + all schema fields -----
    missing_rev = [r.get("sample_id") for r in all_rows if not r.get("source_revision")]
    missing_hash = [r.get("sample_id") for r in all_rows if not r.get("content_sha256")]
    schema_missing = defaultdict(list)
    for r in all_rows:
        for fld in L.ROW_SCHEMA_FIELDS:
            if fld not in r or r.get(fld) in (None, ""):
                # known_overlap_disposition == "none" is a valid present value
                schema_missing[fld].append(r.get("sample_id"))
    report["schema_completeness"] = {
        "n_rows_total": len(all_rows),
        "rows_missing_source_revision": len(missing_rev),
        "rows_missing_content_sha256": len(missing_hash),
        "fields_with_missing_values": {k: len(v) for k, v in schema_missing.items()},
    }

    # content hash recomputation check (does stored hash match normalization?)
    hash_mismatch = [r["sample_id"] for r in all_rows
                     if r.get("content_sha256") != L.content_sha256(r["text_or_download_reference"])]
    report["schema_completeness"]["content_hash_recompute_mismatch"] = len(hash_mismatch)

    # ---- one-to-one join check (unique sample_id across all manifests) -----
    sid_counts = Counter(r["sample_id"] for r in all_rows)
    dup_sids = [s for s, c in sid_counts.items() if c > 1]
    report["join_validation"] = {
        "n_rows": len(all_rows), "n_unique_sample_id": len(sid_counts),
        "duplicate_sample_ids": dup_sids[:20], "n_duplicate_sample_ids": len(dup_sids),
        "one_to_one": len(dup_sids) == 0,
    }

    # ---- 1/2. exact normalized-text overlap, train vs each eval ------------
    train_hash = defaultdict(list)   # content_sha256 -> list of train rows
    for r in train_rows:
        train_hash[r["content_sha256"]].append(r)
    overlap_by_eval = {}
    total_exact = 0
    total_conflict = 0
    total_samelabel = 0
    conflict_records = []
    for stem in eval_stems:
        same = conflict = 0
        recs = []
        for r in M[stem]:
            for tr in train_hash.get(r["content_sha256"], []):
                if tr["label"] == r["label"]:
                    same += 1
                else:
                    conflict += 1
                    conflict_records.append({
                        "content_sha256": r["content_sha256"],
                        "train": {"source": tr["source"], "label": tr["label"],
                                  "sample_id": tr["sample_id"]},
                        "eval": {"split": stem, "source": r["source"], "label": r["label"],
                                 "sample_id": r["sample_id"]}})
                recs.append(r["content_sha256"])
        overlap_by_eval[stem] = {"exact_overlap": len(recs),
                                 "same_label": same, "conflicting_label": conflict}
        total_exact += len(recs); total_samelabel += same; total_conflict += conflict
    report["exact_overlap"] = {
        "total_exact_train_vs_eval": total_exact,
        "total_same_label": total_samelabel,
        "total_conflicting_label": total_conflict,
        "by_eval": overlap_by_eval,
        "conflict_records": conflict_records[:50],
    }

    # ---- 3. near-dup MinHash sensitivity 0.80/0.85/0.90 --------------------
    texts = [r["text_or_download_reference"] for r in all_rows]
    sides = ["train" if r["split"] == TRAIN_SPLIT else "eval" for r in all_rows]
    sigs = L.build_minhash_signatures(texts)
    cand = L.lsh_candidate_pairs(sigs)
    sens = {}
    cross_pairs_by_thr = {}
    for thr in (0.80, 0.85, 0.90):
        edges = L.edges_at_threshold(sigs, cand, thr)
        cross = [(i, j, e) for (i, j, e) in edges if sides[i] != sides[j]]
        within_train = sum(1 for (i, j, _) in edges if sides[i] == sides[j] == "train")
        within_eval = sum(1 for (i, j, _) in edges if sides[i] == sides[j] == "eval")
        sens[f"{thr:.2f}"] = {
            "total_near_dup_pairs": len(edges),
            "cross_train_eval_pairs": len(cross),
            "within_train_pairs": within_train,
            "within_eval_pairs": within_eval,
        }
        cross_pairs_by_thr[f"{thr:.2f}"] = cross
    # detail for the cross pairs that remain at the 0.85 candidate threshold
    remaining_085 = [
        {"est_jaccard": round(e, 4),
         "a": {"split": all_rows[i]["split"], "source": all_rows[i]["source"],
               "sample_id": all_rows[i]["sample_id"]},
         "b": {"split": all_rows[j]["split"], "source": all_rows[j]["source"],
               "sample_id": all_rows[j]["sample_id"]}}
        for (i, j, e) in cross_pairs_by_thr["0.85"]]
    report["near_dup_sensitivity"] = {
        "candidate_pairs_generated": len(cand),
        "by_threshold": sens,
        "remaining_cross_split_at_0.85": remaining_085[:50],
        "note": ("0.85 is the prespecified candidate-generation threshold; the "
                 "builder removes the train-side of every cross-split component at "
                 ">=0.85, so cross_train_eval_pairs at 0.85 must be 0. Counts at "
                 "0.80/0.90 are reported for sensitivity only; dispositions are not "
                 "changed after final scores are viewed."),
    }

    # ---- 4/8. family validation --------------------------------------------
    train_fams = {r["family_id"] for r in train_rows}
    eval_fams = {r["family_id"] for r in eval_rows}
    shared_fams = sorted(train_fams & eval_fams)
    fam_all = Counter(r["family_id"] for r in all_rows)
    report["family_validation"] = {
        "n_families_total": len(fam_all),
        "n_train_families": len(train_fams),
        "n_eval_families": len(eval_fams),
        "train_eval_shared_families": len(shared_fams),
        "shared_family_ids_sample": shared_fams[:20],
        "all_rows_have_family_id": all(r.get("family_id") for r in all_rows),
        "family_stats_from_build": manifest_meta.get("family_stats"),
    }

    # ---- 7. license inventory ----------------------------------------------
    lic = defaultdict(lambda: defaultdict(int))
    for r in all_rows:
        lic[r["source"]][r.get("license_id", "unknown")] += 1
    redist = defaultdict(lambda: defaultdict(int))
    for r in all_rows:
        redist[r["source"]][r.get("redistribution_class", "unknown")] += 1
    report["license_inventory"] = {s: dict(v) for s, v in lic.items()}
    report["redistribution_inventory"] = {s: dict(v) for s, v in redist.items()}

    # ---- 9. role validation ------------------------------------------------
    role_problems = []
    for r in train_rows:
        if r["source"] not in REPRESENTED_SOURCES:
            role_problems.append(f"train row from non-train source {r['source']}")
    for r in M["calibration"] + M["id_test"]:
        if r["source"] not in REPRESENTED_SOURCES:
            role_problems.append(f"cal/id row from non-represented source {r['source']}")
    for r in M["transfer_test"]:
        if r["source"] not in TRANSFER_SOURCES:
            role_problems.append(f"transfer row from non-transfer source {r['source']}")
    orbench_labels = set(r["label"] for r in M["orbench_safe_stress"])
    harmbench_labels = set(r["label"] for r in M["harmbench_positive_stress"])
    if orbench_labels != {"safe"}:
        role_problems.append(f"orbench stress not single-class safe: {orbench_labels}")
    if harmbench_labels != {"unsafe"}:
        role_problems.append(f"harmbench stress not single-class unsafe: {harmbench_labels}")
    report["role_validation"] = {"problems": role_problems[:50], "ok": len(role_problems) == 0,
                                 "orbench_labels": sorted(orbench_labels),
                                 "harmbench_labels": sorted(harmbench_labels)}

    # ---- 10. forbidden-source train counts ---------------------------------
    train_src_counts = Counter(r["source"] for r in train_rows)
    or_bench_train_count = sum(v for k, v in train_src_counts.items()
                               if k in ("orbench", "or_bench", "or-bench"))
    beavertails_train_count = sum(v for k, v in train_src_counts.items()
                                  if k in ("beavertails", "beaver_tails"))
    report["forbidden_train_sources"] = {
        "train_source_counts": dict(train_src_counts),
        "or_bench_train_count": or_bench_train_count,
        "beavertails_train_count": beavertails_train_count,
    }

    # ---- disposition coverage of near-dup candidates -----------------------
    removals = manifest_meta.get("removals", {})
    n_exact_removed = removals.get("exact_train_vs_eval", {}).get("count", 0)
    n_cross_removed = removals.get("cross_split_near_dup", {}).get("count", 0)
    cross_pairs_085 = sens["0.85"]["cross_train_eval_pairs"]
    every_candidate_disposed = (cross_pairs_085 == 0)
    report["disposition_coverage"] = {
        "build_exact_overlaps_removed": n_exact_removed,
        "build_cross_split_components_removed": n_cross_removed,
        "undisposed_cross_split_pairs_at_0.85_in_final_manifests": cross_pairs_085,
        "every_near_duplicate_candidate_has_disposition": every_candidate_disposed,
        "known_wildjailbreak_overlaps_in_build": manifest_meta.get(
            "known_wildjailbreak_overlaps", {}).get("count"),
    }

    # ---- HARD ASSERTIONS ---------------------------------------------------
    assertions = {
        "or_bench_train_count == 0": or_bench_train_count == 0,
        "beavertails_train_count == 0": beavertails_train_count == 0,
        "exact_train_vs_eval_overlap == 0": total_exact == 0,
        "conflicting_label_overlap == 0": total_conflict == 0,
        "every_row_has_source_revision == true": len(missing_rev) == 0,
        "every_row_has_content_hash == true": len(missing_hash) == 0,
        "every_near_duplicate_candidate_has_disposition == true": every_candidate_disposed,
    }
    report["hard_assertions"] = assertions
    all_pass = all(assertions.values())
    report["all_hard_assertions_pass"] = all_pass

    # ---- write audit.json + audit.md ---------------------------------------
    json_path = os.path.join(out_dir, "audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)

    md = build_markdown(report)
    with open(os.path.join(out_dir, "audit.md"), "w", encoding="utf-8") as f:
        f.write(md)

    # ---- console summary ---------------------------------------------------
    print("== HARD ASSERTIONS ==")
    for k, v in assertions.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nexact train<->eval overlap: {total_exact} "
          f"(same-label {total_samelabel}, conflicting {total_conflict})")
    print(f"near-dup cross-split pairs @0.80/0.85/0.90: "
          f"{sens['0.80']['cross_train_eval_pairs']}/"
          f"{sens['0.85']['cross_train_eval_pairs']}/"
          f"{sens['0.90']['cross_train_eval_pairs']}")
    print(f"train/eval shared families: {len(shared_fams)}")
    print(f"or_bench_train={or_bench_train_count}  beavertails_train={beavertails_train_count}")
    print(f"one-to-one joins: {report['join_validation']['one_to_one']}")
    print(f"\nwrote {json_path}")
    print(f"wrote {os.path.join(out_dir, 'audit.md')}")

    if not all_pass:
        print("\n!! ONE OR MORE HARD ASSERTIONS FAILED", file=sys.stderr)
        raise SystemExit(1)
    print("\nALL HARD ASSERTIONS PASS")
    return report


def build_markdown(report):
    a = report["hard_assertions"]
    lines = []
    lines.append("# Paper A — data audit\n")
    lines.append(f"Manifest dir: `{report['manifest_dir']}`  ")
    lines.append(f"Provenance: `{report['provenance'].get('provenance_source')}` "
                 f"(MinHash backend `{report['provenance'].get('minhash_backend', 'n/a')}`, "
                 f"{report['provenance'].get('minhash_num_perm')} perms, "
                 f"{report['provenance'].get('minhash_ngram')}-gram, "
                 f"J>={report['provenance'].get('minhash_jaccard_threshold')})\n")

    lines.append("## Hard assertions\n")
    lines.append("| assertion | result |")
    lines.append("|---|---|")
    for k, v in a.items():
        lines.append(f"| `{k}` | {'PASS' if v else '**FAIL**'} |")
    lines.append(f"\n**All hard assertions pass: {report['all_hard_assertions_pass']}**\n")

    lines.append("## Row and class counts\n")
    lines.append("| manifest | rows | safe | unsafe |")
    lines.append("|---|---:|---:|---:|")
    for stem, c in report["counts"].items():
        cc = c["class_counts"]
        lines.append(f"| {stem} | {c['n_rows']} | {cc.get('safe', 0)} | {cc.get('unsafe', 0)} |")

    lines.append("\n### Per-source counts\n")
    for stem, c in report["counts"].items():
        lines.append(f"- **{stem}**: " +
                     "; ".join(f"{s} (safe {v.get('safe', 0)}, unsafe {v.get('unsafe', 0)})"
                               for s, v in c["per_source"].items()))

    lines.append("\n## Exact train↔eval overlap\n")
    eo = report["exact_overlap"]
    lines.append(f"- total exact overlap: **{eo['total_exact_train_vs_eval']}** "
                 f"(same-label {eo['total_same_label']}, conflicting {eo['total_conflicting_label']})")
    lines.append("\n| eval split | exact | same-label | conflicting |")
    lines.append("|---|---:|---:|---:|")
    for stem, v in eo["by_eval"].items():
        lines.append(f"| {stem} | {v['exact_overlap']} | {v['same_label']} | {v['conflicting_label']} |")

    lines.append("\n## Near-duplicate sensitivity (char-5gram MinHash)\n")
    lines.append("| threshold | total pairs | cross train↔eval | within-train | within-eval |")
    lines.append("|---|---:|---:|---:|---:|")
    for thr, v in report["near_dup_sensitivity"]["by_threshold"].items():
        lines.append(f"| {thr} | {v['total_near_dup_pairs']} | {v['cross_train_eval_pairs']} | "
                     f"{v['within_train_pairs']} | {v['within_eval_pairs']} |")
    lines.append(f"\n> {report['near_dup_sensitivity']['note']}\n")

    lines.append("## Family validation\n")
    fv = report["family_validation"]
    lines.append(f"- total families: {fv['n_families_total']}; train families {fv['n_train_families']}; "
                 f"eval families {fv['n_eval_families']}")
    lines.append(f"- **train↔eval shared families: {fv['train_eval_shared_families']}** "
                 f"(0 expected: cross-split near-dup train members are removed)")

    lines.append("\n## Disposition coverage\n")
    dc = report["disposition_coverage"]
    lines.append(f"- exact overlaps removed at build: {dc['build_exact_overlaps_removed']}")
    lines.append(f"- cross-split near-dup components removed at build: "
                 f"{dc['build_cross_split_components_removed']}")
    lines.append(f"- known WildJailbreak overlaps removed: "
                 f"{dc['known_wildjailbreak_overlaps_in_build']}")
    lines.append(f"- undisposed cross-split near-dup pairs at 0.85 in final manifests: "
                 f"{dc['undisposed_cross_split_pairs_at_0.85_in_final_manifests']}")

    lines.append("\n## License inventory\n")
    lines.append("| source | license(s) | redistribution class(es) |")
    lines.append("|---|---|---|")
    for s in sorted(report["license_inventory"]):
        lic = ", ".join(f"{k} ({v})" for k, v in report["license_inventory"][s].items())
        rd = ", ".join(f"{k} ({v})" for k, v in report["redistribution_inventory"][s].items())
        lines.append(f"| {s} | {lic} | {rd} |")

    lines.append("\n## Role and join validation\n")
    lines.append(f"- role validation ok: {report['role_validation']['ok']}")
    lines.append(f"- OR-Bench stress labels: {report['role_validation']['orbench_labels']}; "
                 f"HarmBench stress labels: {report['role_validation']['harmbench_labels']}")
    lines.append(f"- one-to-one joins (unique sample_id): {report['join_validation']['one_to_one']}")
    lines.append(f"- OR-Bench in train: {report['forbidden_train_sources']['or_bench_train_count']}; "
                 f"BeaverTails in train: {report['forbidden_train_sources']['beavertails_train_count']}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    audit(args.config, args.manifest, args.out)


if __name__ == "__main__":
    main()
