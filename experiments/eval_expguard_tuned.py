#!/usr/bin/env python
"""Score the SFT / KL-SFT LoRA adapters on ExpGuard (finance / healthcare / law).

`eval_expguard_external.py` covers the four *base* checkpoints; the unified report's
limitations section names the tuned comparison as future work, and this closes it. Same
dataset, same canonical guard head, same raw-margin score, same text-free output contract
-- the only difference is a PEFT adapter loaded onto the pinned base weights.

    python experiments/eval_expguard_tuned.py --adapter-root /tmp/klsft_adapters --beta beta0

WHICH ADAPTERS THESE ARE, PRECISELY. The Paper A v2 release adapters were produced on an
ephemeral GCP runner whose bucket was deleted at cleanup (`provenance/execution-evidence.json`
records `bucket: absent`), so they no longer exist. What does exist is the KL-SFT sweep's
adapter grid in `gs://jazztest-bucket/klsft/` -- 4 checkpoints x beta {0, 0.5, 1} x 5 seeds.
Its `beta0` arm is ordinary SFT: `run_meta.json` reports `study_id: paper_a_sft`,
`condition: sft`, `kl_beta: 0.0`, the same LOCK sha256 as `artifacts/paper_a_sft_v2/RELEASE.json`,
the same train manifest sha256, and the same LoRA recipe (r=32, alpha=64, dropout=0.05).

But its `adapter_sha256` values do NOT match the ones recorded in Act~I's `scores.parquet`.
These are therefore a *distinct execution of the same recipe under the same contract*, not
the Act~I release weights. The repository already has vocabulary for exactly this: KL-SFT's
`klsft_summary.json` reports `sft_committed_*` (Act~I release) beside `sft_inenv_*` (this
re-execution), which differ in the third decimal. Everything emitted here is therefore
labelled **SFT (in-env)** and must never be presented as the Act~I release adapters.

Output: `scores_sft_<model_key>_<beta>_seed<seed>.json`, `{row_hash: raw_margin}`, keyed by
`sha256(prompt)[:16]` exactly like every other file in `artifacts/expguard_external/`. No
prompt text is written -- ExpGuard is gated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.eval_expguard_external import (  # noqa: E402
    DEFAULT_OUT,
    load_expguard,
)

# model_key -> (hub id, pinned revision, batch size). Revisions are LOCK.json's, so the
# adapter is applied to the same base weights it was trained on.
PANEL = {
    "qwen25_15b": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", 8),
    "smollm2_17b": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674", 8),
    "smollm3_3b": ("HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1", 4),
    "qwen3_4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", 4),
}
SEEDS = (42, 43, 44, 45, 46)


def find_adapter(root: Path, model_key: str, beta: str, seed: int) -> Path | None:
    """Locate one adapter dir, tolerating both GCS layouts.

    The sweep wrote two shapes: `adapters_<key>/<beta>/seed_<n>/adapter` for qwen*, and
    `adapters_<key>/<key>/<beta>/seed_<n>/adapter` for smollm*. Globbing rather than
    hard-coding keeps a layout change from silently skipping a cell.
    """
    for pattern in (
        f"{model_key}_{beta}/seed_{seed}/adapter",
        f"adapters_{model_key}/{beta}/seed_{seed}/adapter",
        f"adapters_{model_key}/{model_key}/{beta}/seed_{seed}/adapter",
    ):
        cand = root / pattern
        if (cand / "adapter_config.json").is_file():
            return cand
    hits = sorted(root.glob(f"**/{beta}/seed_{seed}/adapter/adapter_config.json"))
    hits = [h for h in hits if model_key in str(h)]
    return hits[0].parent if hits else None


def resolve_panel(lock_path: Path | None) -> dict[str, tuple[str, str, int]]:
    """Model specs, optionally widened by an extension lock.

    Without a lock the four Act~I checkpoints are used. With one, every model in its panel
    is available -- that is how the Qwen3 scale ladder (8B, 32B) becomes scoreable without
    editing `MODEL_PANEL`, whose sha256 is bound by `RELEASE.json`. Batch size falls back to
    2 for unfamiliar keys because the extension checkpoints are much larger.
    """
    panel = dict(PANEL)
    if lock_path is None:
        return panel
    import json as _json
    lock = _json.loads(Path(lock_path).read_text())
    for key, spec in (lock.get("models") or {}).items():
        rev = spec.get("model_revision") or spec.get("revision")
        if not spec.get("model_id") or not rev:
            continue
        panel[key] = (spec["model_id"], rev, panel.get(key, (None, None, 2))[2] or 2)
    return panel


def score_base(model_key: str, panel: dict, rows: list[dict], *, dtype: str,
               batch_size: int, device: str | None) -> dict[str, float]:
    """Zero-shot base scoring -- the same path as score_adapter with no adapter attached."""
    return score_adapter(model_key, None, rows, dtype=dtype, batch_size=batch_size,
                         device=device, panel=panel)


def score_adapter(model_key: str, adapter_dir: Path | None, rows: list[dict], *,
                  dtype: str, batch_size: int, device: str | None,
                  panel: dict | None = None) -> dict[str, float]:
    """Raw margin z_unsafe - z_safe at the decision position, with the adapter attached.

    Byte-parity with `eval_expguard_external.score_checkpoint`: same prompt builder, same
    decision-token selection, same last-non-pad position, same raw-margin score. The margin
    is stored rather than a sigmoid because a confident guard saturates the probability and
    collapses its own ranking into a tie.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from guard_research.prompts import build_prompt, select_decision_tokens

    model_id, revision, _ = (panel or PANEL)[model_key]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if dtype == "auto":
        # bf16 is unreliable on MPS; fp32 off-CUDA, matching the base-eval convention.
        dtype = "bfloat16" if device == "cuda" else "float32"
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=td, trust_remote_code=True)
    if adapter_dir is None:
        model = base.to(device).eval()          # zero-shot base
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(adapter_dir), torch_dtype=td)
        model = model.merge_and_unload().to(device).eval()

    dt = select_decision_tokens(tok)
    safe_id, unsafe_id = dt["safe_id"], dt["unsafe_id"]
    out: dict[str, float] = {}
    empty_cache = getattr(getattr(torch, device, None), "empty_cache", None)
    with torch.no_grad():
        for step, i in enumerate(range(0, len(rows), batch_size)):
            chunk = rows[i:i + batch_size]
            prompts = [build_prompt(tok, r["prompt"]) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(device)
            logits = model(**enc).logits
            last = enc["attention_mask"].sum(1) - 1
            for k, r in enumerate(chunk):
                lz = logits[k, last[k]]
                out[r["id"]] = float(lz[unsafe_id]) - float(lz[safe_id])
            del logits, enc
            if empty_cache and step % 16 == 0:
                empty_cache()
    del model, base
    if empty_cache:
        empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_expguard_tuned", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-root", type=Path, default=None,
                    help="local dir holding the adapters pulled from gs://jazztest-bucket/klsft/results")
    ap.add_argument("--base", action="store_true",
                    help="score the zero-shot BASE checkpoints instead of adapters")
    ap.add_argument("--lock", type=Path, default=None,
                    help="extension lock whose model panel widens the scoreable set "
                         "(e.g. artifacts/qwen3_scale_ext/LOCK.json for the 8B/32B ladder)")
    ap.add_argument("--beta", default="beta0", choices=["beta0", "beta0p5", "beta1"],
                    help="beta0 = ordinary SFT; beta0p5 / beta1 = KL-SFT arms")
    ap.add_argument("--models", nargs="+", default=list(PANEL))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None, help="first N rows (throughput probe)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="override the panel default")
    ap.add_argument("--parquet-path", default=None, help="local ExpGuard parquet (tokenless)")
    ap.add_argument("--force", action="store_true", help="rescore cells already on disk")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    panel = resolve_panel(args.lock)
    unknown = [m for m in args.models if m not in panel]
    if unknown:
        print(f"unknown model keys {unknown}; known: {sorted(panel)}", file=sys.stderr)
        return 2
    rows = load_expguard(limit=args.limit, parquet_path=args.parquet_path)
    print(f"ExpGuard rows: {len(rows)}", flush=True)

    import time
    done = 0

    if args.base:
        for model_key in args.models:
            dst = args.out / f"scores_{model_key}_base.json"
            if dst.exists() and not args.force:
                print(f"  skip {dst.name} (exists)", flush=True)
                continue
            bs = args.batch_size or panel[model_key][2]
            t0 = time.time()
            scores = score_base(model_key, panel, rows, dtype=args.dtype,
                                batch_size=bs, device=args.device)
            dt = time.time() - t0
            dst.write_text(json.dumps(scores, sort_keys=True, separators=(",", ":")) + "\n")
            done += 1
            print(f"  wrote {dst.name}  n={len(scores)}  {dt:.1f}s "
                  f"({len(rows)/dt:.1f} rows/s)", flush=True)
        print(f"scored {done} base checkpoints")
        return 0

    if args.adapter_root is None:
        print("--adapter-root is required unless --base is given", file=sys.stderr)
        return 2
    for model_key in args.models:
        for seed in args.seeds:
            dst = args.out / f"scores_sft_{model_key}_{args.beta}_seed{seed}.json"
            if dst.exists() and not args.force:
                print(f"  skip {dst.name} (exists)", flush=True)
                continue
            adapter = find_adapter(args.adapter_root, model_key, args.beta, seed)
            if adapter is None:
                print(f"  MISSING adapter: {model_key} {args.beta} seed {seed}", flush=True)
                continue
            bs = args.batch_size or panel[model_key][2]
            t0 = time.time()
            scores = score_adapter(model_key, adapter, rows, dtype=args.dtype,
                                   batch_size=bs, device=args.device, panel=panel)
            dt = time.time() - t0
            dst.write_text(json.dumps(scores, sort_keys=True, separators=(",", ":")) + "\n")
            done += 1
            print(f"  wrote {dst.name}  n={len(scores)}  {dt:.1f}s "
                  f"({len(rows)/dt:.1f} rows/s)", flush=True)
    print(f"scored {done} adapter cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
