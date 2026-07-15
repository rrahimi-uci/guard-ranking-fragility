#!/usr/bin/env python
"""Paper C objective-axis trainer: SFT vs DPO vs GRPO (+KTO/ORPO) on the IDENTICAL Paper A v2 panel,
frozen manifest, seeds, LoRA recipe, and single-token verdict head -- so the only thing that varies is
the training objective (H1/H2 in docs/paper-c-prereg.md).

It reuses Paper A's frozen-row loader, LoRA recipe, and run_meta schema (imported from run_paper_a_sft
/ paper_a_common) and only swaps the SFT Trainer for TRL's DPO/GRPO/KTO/ORPO trainer, fed by the
deterministic verifiable-preference recipe in paper_c_preference. The resulting adapter is scored by
the SAME single-token head (eval_paper_a_sft), guaranteeing a like-for-like comparison.

Usage (GPU):
  python experiments/run_paper_c_objective.py --lock <LOCK> --objective dpo --model-key qwen25_15b --seed 42 \
      --out-dir artifacts/paper_c_objective_v2/runs/qwen25_15b/dpo/seed_42
  add --smoke for a 10-step API/wiring check.

STATUS: ready to run; GPU validation of the DPO/GRPO trainers against the pinned TRL is the first step
of the locked run (a single --smoke cell), per the plan's "validate before scaling".
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_a_common as C  # noqa: E402
from run_paper_a_sft import load_train_rows, train_manifest_path, _device  # noqa: E402
import paper_c_preference as P  # noqa: E402

OBJECTIVES = ("sft", "dpo", "kto", "orpo", "grpo")


def paper_c_run_meta(lock, model_key, objective, seed, train_path) -> dict:
    m = C.lock_model_panel(lock).get(model_key, {})
    return {
        "run_id": f"{model_key}_{objective}_seed{seed}_{uuid.uuid4().hex[:8]}",
        "study_id": "paper_c_objective",
        "model_key": model_key, "model_id": m.get("model_id"),
        "model_revision": m.get("model_revision"),
        "condition": objective,          # <-- the objective dimension (vs Paper A's fixed 'sft')
        "objective": objective,
        "seed": seed, "training_seed": seed,
        "data_order_seed": lock.get("data", {}).get("data_order_seed", C.DEFAULT_DATA_ORDER_SEED),
        "train_manifest": train_path,
        "train_manifest_sha256": C.sha256_file(train_path) if os.path.exists(train_path) else None,
        "preference_recipe_sha256": P.preference_recipe_sha256(),
        "prompt_spec_sha256": lock.get("prompt", {}).get("prompt_spec_sha256"),
        "lock_sha256": lock.get("lock_sha256"),
        "recipe": lock.get("recipe"),
        "software_versions": C.software_versions(),
        "device": None, "start_utc": None, "completion_utc": None, "wall_time_s": None,
        "global_steps": None, "dataset_rows": None, "adapter_sha256": None,
        "status": "pending", "failure_reason": None,
    }


def train_objective_cell(lock, objective, model_key, seed, out_dir, train_path,
                         steps=None, smoke=False, device=None) -> dict:
    assert objective in OBJECTIVES, objective
    os.makedirs(out_dir, exist_ok=True)
    meta = paper_c_run_meta(lock, model_key, objective, seed, train_path)
    meta["out_dir"] = out_dir
    meta["device"] = device or _device()
    recipe = lock.get("recipe", C.DEFAULT_RECIPE)
    max_steps = int(steps if steps is not None else (10 if smoke else recipe.get("max_steps", 300)))
    max_len = int(recipe.get("max_length", 1024))
    lora_cfg = recipe.get("lora", C.DEFAULT_RECIPE["lora"])
    per_dev = int(recipe.get("per_device_batch", 1))
    accum = int(recipe.get("gradient_accumulation", 4))
    meta["global_steps"] = max_steps
    meta["start_utc"] = C.utcnow()
    t0 = time.time()

    rows = load_train_rows(train_path)           # <-- IDENTICAL frozen rows as Paper A SFT
    if smoke:
        rows = rows[:40]
    meta["dataset_rows"] = len(rows)

    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer
        from peft import LoraConfig
        from datasets import Dataset as HFDataset
        from guard_research.prompts import build_prompt
        m = C.lock_model_panel(lock)[model_key]
        model_id, revision = m["model_id"], m.get("model_revision")
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        lora = LoraConfig(r=int(lora_cfg["r"]), lora_alpha=int(lora_cfg["alpha"]),
                          lora_dropout=float(lora_cfg.get("dropout", 0.05)), task_type="CAUSAL_LM",
                          target_modules=list(lora_cfg["target_modules"]))
        dev = meta["device"]
        common = dict(output_dir=out_dir, per_device_train_batch_size=per_dev,
                      gradient_accumulation_steps=accum, max_steps=max_steps,
                      lr_scheduler_type="cosine", warmup_ratio=float(recipe.get("warmup_ratio", 0.03)),
                      bf16=(dev == "cuda"), gradient_checkpointing=(dev == "cuda"),
                      logging_steps=10, save_strategy="no", report_to=[], seed=seed)
        adir = os.path.join(out_dir, "adapter")

        if objective == "sft":
            # delegate to Paper A's exact SFT path for a same-recipe reference within this artifact root
            from run_paper_a_sft import train_one_cell
            return train_one_cell(lock, model_key, seed, out_dir, train_path, steps=steps, device=device)

        if objective in ("dpo", "orpo"):
            trainer_cls, cfg_cls = (("DPOTrainer", "DPOConfig") if objective == "dpo"
                                    else ("ORPOTrainer", "ORPOConfig"))
            import trl
            T, Cfg = getattr(trl, trainer_cls), getattr(trl, cfg_cls)
            ds = HFDataset.from_list([{"prompt": build_prompt(tok, r["text"]),
                                       "chosen": P._correct(r["gold"]), "rejected": P._wrong(r["gold"])}
                                      for r in rows])
            kw = dict(beta=0.1) if objective == "dpo" else {}
            cfg = Cfg(learning_rate=5e-6, max_length=max_len, **kw, **common)
            tr = T(model=model_id, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
        elif objective == "kto":
            from trl import KTOTrainer, KTOConfig
            rr = []
            for r in rows:
                for x in P.kto_rows(r):
                    rr.append({"prompt": build_prompt(tok, x["text"]), "completion": x["completion"], "label": x["label"]})
            ds = HFDataset.from_list(rr)
            cfg = KTOConfig(beta=0.1, learning_rate=5e-6, max_length=max_len, **common)
            tr = KTOTrainer(model=model_id, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
        elif objective == "grpo":
            from trl import GRPOTrainer, GRPOConfig
            ds = HFDataset.from_list([{"prompt": build_prompt(tok, r["text"]), "gold": r["gold"]} for r in rows])

            def reward_fn(completions, gold, **kw):
                return [P.graded_reward(P.parse_verdict(c), g) for c, g in zip(completions, gold)]
            g = {k: v for k, v in common.items() if k not in ("per_device_train_batch_size", "gradient_accumulation_steps")}
            ng = 8
            cfg = GRPOConfig(per_device_train_batch_size=ng, gradient_accumulation_steps=2,
                             num_generations=ng, max_completion_length=8, learning_rate=1e-6,
                             beta=0.04, temperature=0.9, **g)
            tr = GRPOTrainer(model=model_id, args=cfg, train_dataset=ds, reward_funcs=[reward_fn],
                             processing_class=tok, peft_config=lora)
        else:
            raise SystemExit(f"unknown objective {objective}")

        tr.train()
        tr.save_model(adir)
        meta["adapter_sha256"] = C.sha256_dir(adir) if hasattr(C, "sha256_dir") else None
        meta["status"] = "smoke" if smoke else "complete"
    except Exception as e:  # keep the run_meta (failed runs are kept, per Paper A discipline)
        meta["status"] = "failed"
        meta["failure_reason"] = f"{type(e).__name__}: {e}"
    meta["wall_time_s"] = round(time.time() - t0, 3)
    meta["completion_utc"] = C.utcnow()
    C.write_json(os.path.join(out_dir, "run_meta.json"), meta)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Paper C objective-axis trainer (SFT/DPO/KTO/ORPO/GRPO).")
    ap.add_argument("--lock", required=True)
    ap.add_argument("--objective", required=True, choices=OBJECTIVES)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-manifest", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow-legacy-lock", action="store_true")
    args = ap.parse_args(argv)
    lock = C.load_lock(args.lock, allow_legacy=args.allow_legacy_lock, verify_files=False)
    train_path = train_manifest_path(lock, args.train_manifest)
    meta = train_objective_cell(lock, args.objective, args.model_key, args.seed, args.out_dir,
                                train_path, steps=args.steps, smoke=args.smoke, device=args.device)
    print(f"[paper-c] {args.objective} {args.model_key} seed{args.seed}: {meta['status']}"
          + (f" ({meta['failure_reason']})" if meta.get("failure_reason") else ""))
    return 0 if meta["status"] in ("complete", "smoke", "dry_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
