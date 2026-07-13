#!/usr/bin/env python
"""Create artifacts/paper_a_sft/LOCK.json (plan sec 14.1).

The lock is created AFTER manifests/tests/smoke validation and BEFORE final
training. It freezes every input to the fixed-panel base-vs-LoRA-SFT study so
that final-evaluation code can refuse an absent or mismatched lock.

Records (plan sec 14.1):
  git sha + dirty-state policy; model/tokenizer revisions; data revisions and
  manifest hashes; source inclusions/exclusions; license branch; prompt
  template + hash (model-independent spec hash plus per-checkpoint rendered
  template hashes and decision token ids); training recipe; seeds; metrics;
  target FPR + confidence method; primary contrasts; analysis_mode
  (powered_confirmatory | precision_focused); power-report hash + seed-count
  decision; statistical resampling rules; table/figure specs; failure handling;
  artifact paths.

Refuses to overwrite an existing lock without --force.

Usage:
  python experiments/lock_paper_a_sft.py \
    --config configs/paper_a_sft.yaml \
    --manifest artifacts/paper_a_sft/manifests/manifest.json \
    --audit artifacts/paper_a_sft/audit/audit.json \
    --power artifacts/paper_a_sft/design/power_report.json \
    --out artifacts/paper_a_sft/LOCK.json
"""
from __future__ import annotations

import argparse
import os
import sys
import pathlib

# path bootstrap so `import guard_research` / sibling module both resolve
_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_common as C  # noqa: E402

MANIFEST_FILES = [
    "train.jsonl", "calibration.jsonl", "id_test.jsonl", "transfer_test.jsonl",
    "orbench_safe_stress.jsonl", "harmbench_positive_stress.jsonl",
]


def _obj_sha256(obj) -> str:
    try:
        from guard_research.provenance import sha256_of_obj  # type: ignore
        return sha256_of_obj(obj)
    except Exception:
        import json
        return C.sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def resolve_models(config: dict) -> dict:
    """Model/tokenizer revisions: config['models'] is authoritative; fall back to panel."""
    models = {}
    cfg_models = config.get("models") or {}
    for key in C.MODEL_KEYS:
        panel = C.MODEL_PANEL[key]
        cm = cfg_models.get(key, {}) if isinstance(cfg_models, dict) else {}
        model_id = cm.get("model_id", panel["model_id"])
        rev = cm.get("model_revision", cm.get("revision", panel["revision"]))
        tok_rev = cm.get("tokenizer_revision", rev)
        models[key] = {
            "model_id": model_id,
            "model_revision": rev,
            "tokenizer_revision": tok_rev,
            "dtype": cm.get("dtype", config.get("dtype", "bfloat16")),
            "attn_implementation": cm.get("attn_implementation", config.get("attn_implementation")),
            "trust_remote_code": cm.get("trust_remote_code", True),
            "revision_source": "config" if key in cfg_models else "panel_default",
        }
    return models


def probe_tokenizers(models: dict, require: bool) -> dict:
    """Load each tokenizer at its pinned revision and freeze decision tokens +
    rendered-template hash (plan sec 7). Degrades gracefully if models are absent."""
    out = {}
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        if require:
            raise
        return {k: {"status": "transformers_unavailable", "error": str(e)} for k in models}
    for key, m in models.items():
        rec = {"status": "unavailable"}
        try:
            tok = AutoTokenizer.from_pretrained(
                m["model_id"], revision=m["tokenizer_revision"],
                trust_remote_code=m.get("trust_remote_code", True))
            dt = C.resolve_decision_tokens(tok)
            rec = {
                "status": "ok",
                "safe_str": dt["safe_str"], "unsafe_str": dt["unsafe_str"],
                "safe_token_id": dt["safe_id"], "unsafe_token_id": dt["unsafe_id"],
                "prompt_template_sha256": C.template_sha256(tok),
            }
        except Exception as e:  # missing model / gated / no single-token convention
            rec = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            if require:
                raise RuntimeError(f"tokenizer probe failed for {key}: {e}") from e
        out[key] = rec
    return out


def build_lock(args) -> dict:
    config = C.load_config(args.config)
    prompt = C.prompt_identity()
    models = resolve_models(config)

    # -- manifests: hash the index + each split file that exists --
    manifests_dir = args.manifests_dir or (
        os.path.dirname(args.manifest) if args.manifest else C.DEFAULT_ARTIFACTS["manifests"])
    manifests = {"dir": manifests_dir, "index": None, "splits": {}}
    if args.manifest and os.path.exists(args.manifest):
        manifests["index"] = {"path": args.manifest, "sha256": C.sha256_file(args.manifest)}
    for fn in MANIFEST_FILES:
        p = os.path.join(manifests_dir, fn)
        if os.path.exists(p):
            manifests["splits"][fn] = {"path": p, "sha256": C.sha256_file(p),
                                       "rows": sum(1 for _ in open(p, "r", encoding="utf-8") if _.strip())}
        else:
            manifests["splits"][fn] = {"path": p, "sha256": None, "rows": None, "missing": True}
    train_manifest_sha256 = manifests["splits"].get("train.jsonl", {}).get("sha256")

    # -- audit / power provenance --
    audit = None
    if args.audit and os.path.exists(args.audit):
        audit = {"path": args.audit, "sha256": C.sha256_file(args.audit)}
    power = None
    seed_count_decision = None
    analysis_mode = args.analysis_mode or config.get("analysis_mode") or "precision_focused"
    if args.power and os.path.exists(args.power):
        preport = C.read_json(args.power)
        power = {"path": args.power, "sha256": C.sha256_file(args.power)}
        seed_count_decision = preport.get("seed_count_decision") or preport.get("decision")
        if args.analysis_mode is None and preport.get("analysis_mode"):
            analysis_mode = preport["analysis_mode"]
    if analysis_mode not in ("precision_focused", "powered_confirmatory"):
        raise SystemExit(f"invalid analysis_mode: {analysis_mode!r}")

    git = C.git_provenance()
    if args.require_clean and git.get("git_tracked_dirty"):
        raise SystemExit(
            "refusing to lock: tracked working tree is dirty and --require-clean was set. "
            "Commit changes or drop --require-clean.")

    recipe = dict(C.DEFAULT_RECIPE)
    # overlay config recipe values when present
    for k_cfg, k_lock in (("max_steps", "max_steps"), ("max_length", "max_length"),
                          ("learning_rate", "learning_rate"), ("warmup_ratio", "warmup_ratio"),
                          ("effective_batch", "effective_batch")):
        if k_cfg in config:
            recipe[k_lock] = config[k_cfg]
    if isinstance(config.get("lora"), dict):
        recipe["lora"].update({k: config["lora"][k] for k in ("r", "alpha", "dropout")
                               if k in config["lora"]})

    seeds = list(config.get("seeds", C.DEFAULT_SEEDS))
    data_seed = config.get("data_seed", C.DEFAULT_DATA_ORDER_SEED)
    data_order_seed = config.get("data_order_seed", C.DEFAULT_DATA_ORDER_SEED)
    target_fpr = float(config.get("target_fpr", C.DEFAULT_TARGET_FPR))
    reps = int(config.get("bootstrap_replicates", C.DEFAULT_BOOTSTRAP_REPLICATES))
    boot_seed = int(config.get("bootstrap_seed", C.DEFAULT_BOOTSTRAP_SEED))

    tok_probe = {}
    if args.probe_tokenizers:
        tok_probe = probe_tokenizers(models, require=args.require_tokenizer_probe)

    lock = {
        "schema_version": config.get("schema_version", 1),
        "study_id": config.get("study_id", "paper_a_sft"),
        "created_utc": C.utcnow(),
        "config": {"path": args.config, "sha256": C.sha256_file(args.config),
                   "obj_sha256": _obj_sha256(config)},
        "git": {**git, "dirty_state_policy": ("require_clean_tracked" if args.require_clean
                                              else "recorded_not_enforced")},
        "license_branch": config.get("data_branch", config.get("license_branch",
                                                               "academic_noncommercial")),
        "data": {
            "data_seed": data_seed,
            "data_order_seed": data_order_seed,
            "train_sources": config.get("train_sources",
                                        C.REGIME_BENCHMARKS["represented"]),
            "excluded_train_sources": config.get("excluded_train_sources",
                                                 ["beavertails", "or_bench"]),
            "rows_per_source": config.get("rows_per_source", 400),
            "rows_per_source_label": config.get("rows_per_source_label", 200),
        },
        "models": models,
        "tokenizer_probe": tok_probe,
        "prompt": {
            "version": C.PROMPT_VERSION,
            "prompt_spec_sha256": prompt["prompt_sha256"],
            "spec_source": prompt["source"],
            "system_prompt": prompt["system"],
            "per_model_template_sha256": {k: v.get("prompt_template_sha256")
                                          for k, v in tok_probe.items()},
        },
        "recipe": recipe,
        "seeds": seeds,
        "n_checkpoints": len(C.MODEL_KEYS),
        "n_seeds": len(seeds),
        "n_final_cells": len(C.MODEL_KEYS) * len(seeds),
        "metrics": {
            "canonical_module": "guard_research.metrics",
            "primary": ["tie_aware_average_precision_per_benchmark",
                        "macro_ap_represented", "macro_ap_transfer", "base_to_sft_delta"],
            "secondary": ["auroc", "tpr_at_calibration_target_fpr", "realized_fpr",
                          "brier", "log_loss", "orbench_benign_fpr", "harmbench_recall"],
            "primary_metric": config.get("primary_metric", "macro_average_precision"),
        },
        "operating_point": {
            "target_fpr": target_fpr,
            "threshold_module": "guard_research.thresholds.select_threshold",
            "confidence_method": "clopper_pearson_one_sided_95_on_pooled_calibration_negatives",
            "no_feasible_sentinel": "NO_FEASIBLE_THRESHOLD",
        },
        "regime_benchmarks": C.REGIME_BENCHMARKS,
        "primary_contrasts": {
            "unit": "per_checkpoint_base_vs_sft",
            "regimes": ["represented", "transfer"],
            "estimand": "theta = (mean_delta_represented, mean_delta_transfer)",
            "delta": "M_R(SFT_{b,r}) - M_R(base_b)",
            "aggregate": ("fixed_panel_mean over 4 checkpoints of "
                          "(mean over seeds M_R(SFT_b) - M_R(base_b))"),
        },
        "analysis_mode": analysis_mode,
        "power_report": power,
        "seed_count_decision": seed_count_decision,
        "confidence_method": "hierarchical_paired_poisson_family_bootstrap",
        "resampling_rules": {
            "method": "hierarchical_paired_poisson_family_bootstrap",
            "replicates": reps,
            "rng_seed": boot_seed,
            "checkpoints": "fixed_4_identities_never_resampled",
            "seed_resample": "5_seed_indices_with_replacement_within_each_checkpoint",
            "family_weight": ("one_Poisson(1)_weight_per_global_family_id, applied to all "
                              "rows of that family across every evaluation dataset"),
            "ap": "weighted_tie_aware_average_precision_per_benchmark",
            "weighting_impl": "integer_weight_row_replication_through_canonical_ap",
            "macro": "mean_over_benchmarks_within_regime",
            "delta": "per_checkpoint_delta_then_mean_over_4_checkpoints_no_ckpt_resample",
            "one_sided_lcb_percentile": 5.0,
            "one_sided_ucb_percentile": 95.0,
            "two_sided_percentiles": [2.5, 97.5],
            "zero_effective_class": "reject_replicate_and_redraw_all_family_weights_record_retries",
        },
        "sensitivity": {
            "leave_one_transfer_benchmark_out": True,
            "leave_one_base_out": True,
            "sign_stable_definition": "every leave-one-out estimate shares the full aggregate sign",
        },
        "claim_gates": {
            "gate_a": "LCB95(mean_delta_represented) > 0",
            "gate_b": ("UCB95(mean_delta_transfer) < 0 AND leave-one-transfer-benchmark-out "
                       "sign-stable AND leave-one-base-out sign-stable"),
            "specialization": "gate_a AND gate_b (intersection-union at alpha 0.05)",
            "multiplicity": "Holm across the two component tests if claimed standalone",
            "rq4": "descriptive_only",
            "precision_focused_language": (analysis_mode == "precision_focused"),
        },
        "tables": {
            "table3_primary": {"path": "analysis/tables/table3_primary.tex",
                               "content": "per-base represented/transfer base, SFT mean, "
                                          "delta+interval, 5 seed values, fixed-panel aggregate"},
            "table4_per_benchmark": {"path": "analysis/tables/table4_per_benchmark.tex",
                                     "content": "per-source paired delta, TPR@target FPR, "
                                                "realized FPR, OR-Bench benign FPR, HarmBench recall"},
        },
        "figures": {
            "specialization_plane": {"path": "analysis/figures/specialization_plane.pdf",
                                     "content": "x=represented delta, y=transfer delta; "
                                                "color per checkpoint; point per seed; "
                                                "zero lines; fixed-panel mean marker"},
        },
        "failure_handling": {
            "keep_failed_runs": True,
            "failed_cell_blocks_fixed_panel_aggregate": True,
            "require_all_cells_before_scoring": True,
            "cache_mismatch_policy": "recompute_never_trust_row_count",
            "no_feasible_threshold_is_reportable": True,
        },
        "audit": audit,
        "manifests": manifests,
        "train_manifest_sha256": train_manifest_sha256,
        "artifact_paths": dict(C.DEFAULT_ARTIFACTS),
        "score_code_version": "paper_a_sft_scorer_v1",
        "analysis_code_version": "paper_a_sft_analysis_v1",
        "software_versions": C.software_versions(),
    }
    lock["lock_sha256"] = _obj_sha256({k: v for k, v in lock.items() if k != "lock_sha256"})
    return lock


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write Paper A LOCK.json (plan sec 14.1).")
    ap.add_argument("--config", required=True, help="configs/paper_a_sft.yaml")
    ap.add_argument("--manifest", default=None, help="manifests/manifest.json index")
    ap.add_argument("--manifests-dir", default=None,
                    help="directory holding the split jsonl files (default: --manifest parent)")
    ap.add_argument("--audit", default=None, help="audit/audit.json")
    ap.add_argument("--power", default=None, help="design/power_report.json")
    ap.add_argument("--out", default=C.DEFAULT_ARTIFACTS["lock"], help="output LOCK.json path")
    ap.add_argument("--analysis-mode", default=None,
                    choices=["precision_focused", "powered_confirmatory"])
    ap.add_argument("--force", action="store_true", help="overwrite an existing lock")
    ap.add_argument("--require-clean", action="store_true",
                    help="refuse to lock when the tracked working tree is dirty")
    probe = ap.add_mutually_exclusive_group()
    probe.add_argument("--probe-tokenizers", dest="probe_tokenizers", action="store_true",
                       default=True, help="load each tokenizer to freeze decision tokens (default)")
    probe.add_argument("--skip-tokenizer-probe", dest="probe_tokenizers", action="store_false",
                       help="do not load tokenizers (spec-level prompt hash only)")
    ap.add_argument("--require-tokenizer-probe", action="store_true",
                    help="fail if any tokenizer cannot be loaded/verified")
    args = ap.parse_args(argv)

    out = C.abspath(args.out) if not os.path.isabs(args.out) else args.out
    if os.path.exists(out) and not args.force:
        print(f"[lock] refusing to overwrite existing lock at {out} (use --force).",
              file=sys.stderr)
        return 2

    if not os.path.exists(args.config):
        print(f"[lock] config not found: {args.config}", file=sys.stderr)
        return 2

    lock = build_lock(args)
    C.write_json(out, lock)
    probed = sum(1 for v in lock["tokenizer_probe"].values() if v.get("status") == "ok")
    print(f"[lock] wrote {out}")
    print(f"[lock]   git_sha={lock['git']['git_sha']} dirty={lock['git']['git_dirty']}")
    print(f"[lock]   analysis_mode={lock['analysis_mode']} "
          f"cells={lock['n_final_cells']} seeds={lock['seeds']}")
    print(f"[lock]   prompt_spec_sha256={lock['prompt']['prompt_spec_sha256'][:16]}... "
          f"({lock['prompt']['spec_source']})")
    print(f"[lock]   train_manifest_sha256={str(lock['train_manifest_sha256'])[:16]}...")
    print(f"[lock]   tokenizers probed ok: {probed}/{len(lock['models'])}")
    print(f"[lock]   lock_sha256={lock['lock_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
