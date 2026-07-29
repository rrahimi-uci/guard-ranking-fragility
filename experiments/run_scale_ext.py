#!/usr/bin/env python
"""Train and score the Qwen3 scale-ladder extension on the Act I represented/transfer panel.

    python experiments/run_scale_ext.py --lock artifacts/qwen3_scale_ext/LOCK.json \
        --model-key qwen3_8b --conditions base sft --seeds 42 43 44 45 46 \
        --out-root artifacts/qwen3_scale_ext --device cuda

This is the scale-ladder sibling of `run_klsft_sweep.py`, and it exists for one reason that
script cannot serve: `run_klsft_sweep` restricts `--model-key` to `C.MODEL_KEYS`, which is
derived from `MODEL_PANEL`, whose sha256 is bound by `RELEASE.json`. Extension checkpoints
are therefore unreachable through it. Everything else is deliberately identical -- the same
`train_one_cell` for training and the same `score_bundle` / `assemble_bundle` for scoring --
so an 8B row is comparable to the four panel rows at the level of the scoring primitive, not
merely "computed the same way".

WHY THIS DELIBERATELY BYPASSES verify_lock. `_verify_strict_lock_structure` asserts
`set(models) == set(MODEL_KEYS)`: the four-checkpoint panel is enforced in code, so no
extension lock can pass verification. The sidecar lock is consumed as a plain dict, exactly
as the canonical primitives already consume it (`base_run_meta`, `train_one_cell` and
`score_bundle` all resolve models via `lock_model_panel(lock)` and none re-verify). The
sidecar's own self-hash IS checked here, and the train-manifest sha256 is enforced, so the
training data cannot drift even though the panel check is skipped.

Every cell is written with `run_kind="nonfinal"`. These runs are an extension study; they
are not Act~I release cells and must never be pooled into its fixed-panel estimand.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_common as C  # noqa: E402
from eval_paper_a_sft import (  # noqa: E402
    _default_device,
    assemble_bundle,
    load_scoring_rows,
    score_bundle,
)
from run_paper_a_sft import train_manifest_path, train_one_cell  # noqa: E402


def load_sidecar(path: str) -> dict:
    """Read the extension lock and check its self-hash, skipping the panel-shape check."""
    lock = C.read_json(path)
    stored = lock.get("lock_sha256")
    observed = C.canonical_obj_sha256({k: v for k, v in lock.items() if k != "lock_sha256"})
    if not stored or stored != observed:
        raise SystemExit(f"sidecar lock self-hash mismatch: {path}")
    ext = lock.get("extension_of") or {}
    if not ext.get("lock_sha256"):
        raise SystemExit("sidecar lock has no extension_of provenance; refusing to run")
    return lock


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--model-key", required=True,
                    help="any key in the sidecar lock's panel (not restricted to MODEL_KEYS)")
    ap.add_argument("--conditions", nargs="+", default=["base", "sft"],
                    choices=["base", "sft"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--out-root", default="artifacts/qwen3_scale_ext")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=None, help="debug/smoke only")
    ap.add_argument("--limit", type=int, default=None, help="rows per manifest (debug)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import pandas as pd

    lock = load_sidecar(args.lock)
    mk = args.model_key
    panel = C.lock_model_panel(lock)
    if mk not in panel:
        print(f"{mk} not in sidecar panel: {sorted(panel)}", file=sys.stderr)
        return 2
    m = panel[mk]
    device = args.device or _default_device()
    dtype = m.get("dtype", "bfloat16")
    model_revision = m["model_revision"]
    target_fpr = float(lock.get("operating_point", {}).get("target_fpr",
                                                           C.DEFAULT_TARGET_FPR))

    train_path = train_manifest_path(lock, None)
    if not os.path.exists(train_path):
        print(f"train manifest missing: {train_path}", file=sys.stderr)
        return 2
    if C.sha256_file(train_path) != lock.get("train_manifest_sha256"):
        print("refusing mismatched train manifest", file=sys.stderr)
        return 2

    manifests_dir = C.abspath(C.artifact_paths(lock)["manifests"])
    rows = load_scoring_rows(manifests_dir, args.limit)
    print(f"[scale-ext] {mk} on {device} ({dtype}); {len(rows)} scoring rows", flush=True)

    out_root = C.abspath(args.out_root)
    runs_root = os.path.join(out_root, "runs")
    scores_dir = os.path.join(out_root, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    all_recs: list[dict] = []

    if "base" in args.conditions:
        dst = os.path.join(scores_dir, f"scores_{mk}_base.parquet")
        if args.force or not os.path.exists(dst):
            t0 = time.time()
            logits, prompt_sha, dtoks = score_bundle(
                lock, rows, mk, "base", -1, None, None, device, dtype,
                args.batch_size, False)
            recs, _ = assemble_bundle(lock, rows, logits, mk, model_revision, "base",
                                      -1, None, prompt_sha, dtoks, target_fpr)
            pd.DataFrame(recs).to_parquet(dst, index=False)
            all_recs += recs
            print(f"[scale-ext] base scored -> {os.path.basename(dst)} "
                  f"({round(time.time()-t0,1)}s)", flush=True)
        else:
            print(f"[scale-ext] reuse {os.path.basename(dst)}", flush=True)

    if "sft" in args.conditions:
        for seed in args.seeds:
            out_dir = os.path.join(runs_root, mk, "beta0", f"seed_{seed}")
            adir = C.adapter_dir(out_dir)
            if args.force or not C.adapter_is_present(adir):
                t0 = time.time()
                meta = train_one_cell(lock, mk, seed, out_dir, train_path,
                                      steps=args.max_steps, device=device,
                                      run_kind="nonfinal", kl_beta=0.0)
                if meta.get("status") != "completed":
                    print(f"[scale-ext] TRAIN FAILED {mk} seed {seed}: "
                          f"{meta.get('failure_reason')}", file=sys.stderr)
                    continue
                print(f"[scale-ext] trained {mk} seed {seed} "
                      f"({round(time.time()-t0,1)}s)", flush=True)
            else:
                print(f"[scale-ext] reuse adapter {mk} seed {seed}", flush=True)

            dst = os.path.join(scores_dir, f"scores_{mk}_sft_seed{seed}.parquet")
            if not args.force and os.path.exists(dst):
                print(f"[scale-ext] reuse {os.path.basename(dst)}", flush=True)
                continue
            meta = json.load(open(os.path.join(out_dir, "run_meta.json")))
            adapter_sha = meta.get("adapter_sha256") or C.sha256_dir(adir)
            t0 = time.time()
            logits, prompt_sha, dtoks = score_bundle(
                lock, rows, mk, "sft", seed, adir, adapter_sha, device, dtype,
                args.batch_size, False)
            recs, _ = assemble_bundle(lock, rows, logits, mk, model_revision, "sft",
                                      seed, adapter_sha, prompt_sha, dtoks, target_fpr)
            pd.DataFrame(recs).to_parquet(dst, index=False)
            all_recs += recs
            print(f"[scale-ext] sft seed {seed} scored -> {os.path.basename(dst)} "
                  f"({round(time.time()-t0,1)}s)", flush=True)

    if all_recs:
        combined = os.path.join(scores_dir, f"scores_{mk}_combined.parquet")
        pd.DataFrame(all_recs).to_parquet(combined, index=False)
        print(f"[scale-ext] wrote {os.path.basename(combined)} ({len(all_recs)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
