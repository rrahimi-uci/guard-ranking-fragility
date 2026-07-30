#!/usr/bin/env python
"""Score released purpose-built guards on ExpGuard through their native verdict contracts.

    python experiments/eval_expguard_guards.py --out artifacts/expguard_external

The frontier comparison had base instruction checkpoints, our SFT arms, a scale ladder and a
hosted frontier model -- but no *purpose-built* guard, so it could not answer the first
question a practitioner asks: "why not just run Llama Guard?" This scores six released guards
on the same 2,275 ExpGuard rows, keyed by the same row hash, so they drop straight into
`tab:frontier`.

ExpGuard is unusually good ground for this comparison: none of these guards was trained on
it, so unlike the Act~I panel (which contains `wildguardtest`, WildGuard's own benchmark)
there is no home-field advantage to discount.

EACH GUARD IS SCORED ON ITS OWN TERMS. A purpose-built guard does not answer the frozen
`safe`/`unsafe` prompt the general checkpoints answer -- ShieldGemma emits Yes/No against a
supplied policy, Granite emits Yes/No against a risk definition, WildGuard fills a
"Harmful request: yes/no" slot, Qwen3Guard emits a top-level Safety label, Llama Guard emits
`\\n\\nsafe` or `\\n\\nunsafe\\nS<k>`. `experiments/guard_contracts.py` holds those renderings;
this script uses them rather than forcing a foreign prompt on a guard, which would measure
our prompt rather than their guard.

THE DECISION POSITION IS PER-CONTRACT, and getting it wrong is silent. Llama Guard's verdict
follows a two-newline prefix, so the last prompt position carries the distribution over
`\\n\\n`, not over the verdict -- reading there yields a *constant* margin for every input.
The committed starting-type study has exactly that failure recorded in it: 36,388 rows of one
unique score for `llama_guard_3_1b`, chance-level AP, described in the paper as a "degenerate
cell". Two independent bugs caused it (an empty-conversation render plus the newline read
position); both are fixed in `guard_contracts.py`, and this scorer teacher-forces each
contract's completion prefix so the read lands on the verdict token.

Output: `scores_guard_<key>.json`, `{row_hash: margin}` where margin is
logit(unsafe-word) - logit(safe-word) at the verdict position. Text-free, like every other
file in that directory.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.eval_expguard_external import DEFAULT_OUT, load_expguard  # noqa: E402
from experiments.guard_contracts import GUARD_CONTRACTS  # noqa: E402

# key -> (hub id, contract name, batch size). Revisions are pinned in
# configs/starting_type_adaptation_v1.yaml; resolved at run time and recorded in the sidecar.
PANEL = {
    "shieldgemma_2b": ("google/shieldgemma-2b", "shieldgemma_yes_no", 8),
    "qwen3guard_gen_06b": ("Qwen/Qwen3Guard-Gen-0.6B", "qwen3guard_toplevel", 16),
    "qwen3guard_gen_4b": ("Qwen/Qwen3Guard-Gen-4B", "qwen3guard_toplevel", 8),
    "granite_guardian_31_2b": ("ibm-granite/granite-guardian-3.1-2b", "granite_yes_no", 8),
    "llama_guard_3_1b": ("meta-llama/Llama-Guard-3-1B", "llama_guard_toplevel", 16),
    "wildguard_7b": ("allenai/wildguard", "wildguard_prompt_harm", 4),
}


def score_guard(key: str, rows: list[dict], *, device: str, batch_size: int,
                max_len: int = 1024) -> tuple[dict[str, float], dict]:
    """Native-contract margin at the verdict position, for every ExpGuard row."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, contract_name, _ = PANEL[key]
    # Fast->slow fallback, same as starting_type_common.load_study_tokenizer: some released
    # guards ship a SentencePiece tokenizer.model that transformers' fast path mis-routes to
    # the tiktoken converter (allenai/wildguard, a Mistral tokenizer). Keeping the behaviour
    # identical to the study's loader means a guard tokenizes the same way in both places.
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()

    contract = GUARD_CONTRACTS[contract_name]()
    values = contract._decision_values()
    safe_id = tok.encode(values["safe"], add_special_tokens=False)[0]
    unsafe_id = tok.encode(values["unsafe"], add_special_tokens=False)[0]
    if safe_id == unsafe_id:
        raise ValueError(f"{key}: decision words collapse to one token id {safe_id}")

    # Teacher-force whatever precedes the verdict in this contract's completion (Llama Guard's
    # "\n\n"; empty for the Yes/No guards) so the scored position is the verdict itself.
    full = contract._completion_string(1)
    verdict_word = values["unsafe"]
    prefix = full[:full.rindex(verdict_word)] if verdict_word in full else ""

    meta = {"model_id": model_id, "contract": contract_name,
            "safe_word": values["safe"], "unsafe_word": values["unsafe"],
            "safe_token_id": int(safe_id), "unsafe_token_id": int(unsafe_id),
            "verdict_prefix": prefix, "render_modes": {}, "payload_carried": 0}

    out: dict[str, float] = {}
    empty_cache = getattr(getattr(torch, device, None), "empty_cache", None)
    with torch.no_grad():
        for step, i in enumerate(range(0, len(rows), batch_size)):
            chunk = rows[i:i + batch_size]
            prompts = []
            for r in chunk:
                rendered, info = contract.render(tok, r["prompt"], max_len, 8)
                mode = info.get("render_mode", "?")
                meta["render_modes"][mode] = meta["render_modes"].get(mode, 0) + 1
                meta["payload_carried"] += int(contract._carries(rendered, r["prompt"]))
                prompts.append(rendered + prefix)
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len + 8).to(device)
            logits = model(**enc).logits
            last = enc["attention_mask"].sum(1) - 1
            for k, r in enumerate(chunk):
                lz = logits[k, last[k]]
                out[r["id"]] = float(lz[unsafe_id]) - float(lz[safe_id])
            del logits, enc
            if empty_cache and step % 16 == 0:
                empty_cache()
    del model
    if empty_cache:
        empty_cache()
    return out, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--guards", nargs="+", default=list(PANEL), choices=list(PANEL))
    ap.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_expguard(limit=args.limit)
    print(f"ExpGuard rows: {len(rows)} · device {device}", flush=True)

    done = 0
    for key in args.guards:
        dst = args.out / f"scores_guard_{key}.json"
        if dst.exists() and not args.force:
            print(f"  skip {dst.name} (exists)", flush=True)
            continue
        bs = args.batch_size or PANEL[key][2]
        t0 = time.time()
        try:
            scores, meta = score_guard(key, rows, device=device, batch_size=bs)
        except Exception as exc:  # noqa: BLE001 - one bad guard must not kill the sweep
            print(f"  FAILED {key}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            continue
        uniq = len(set(scores.values()))
        dst.write_text(json.dumps(scores, sort_keys=True, separators=(",", ":")) + "\n")
        (args.out / f"scores_guard_{key}.metadata.json").write_text(
            json.dumps({**meta, "n_rows": len(scores), "unique_scores": uniq,
                        "seconds": round(time.time() - t0, 1)}, indent=2, sort_keys=True) + "\n")
        # A degenerate guard is a harness bug, not a result. Say so loudly rather than
        # committing a constant column that reads as a model verdict. The bar is relative to
        # row count: the failure this catches collapses to a handful of values over thousands
        # of rows (Llama Guard managed exactly one), so a fixed threshold would false-positive
        # on any small --limit smoke run.
        flag = ("  <-- DEGENERATE, do not report as a model result"
                if uniq <= max(3, int(0.01 * len(scores))) else "")
        print(f"  wrote {dst.name}  n={len(scores)}  unique={uniq}  "
              f"{round(time.time()-t0,1)}s  modes={meta['render_modes']}{flag}", flush=True)
        done += 1
    print(f"scored {done} guards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
