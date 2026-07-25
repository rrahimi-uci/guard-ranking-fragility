#!/usr/bin/env python
"""Train one tiny cell end to end to prove the pipeline before any GPU spend.

    python tools/smoke_cell.py [model_id] [revision] [device] [steps]

Uses the smoke profile and a small row sample.  Proves: LoRA attaches, the
completion-only SFT loss backpropagates, the checkpoint ladder saves, the adapter
reloads, and the action head still answers after training.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from paper_c_sta.contracts import output_path, read_json  # noqa: E402
from paper_c_sta.modeling import (  # noqa: E402
    ACTIONS, action_logits, load_backbone, render_prompt,
)
from paper_c_sta.train import run_cell  # noqa: E402

model_id = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
revision = sys.argv[2] if len(sys.argv) > 2 else "main"
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 12

config = read_json(output_path("config/smoke.json"))
config = dict(config)
config["backbones"] = {"smoke": {"model_id": model_id, "revision": revision}}

rows = []
with (ROOT / "artifacts/cohort/samples.jsonl").open() as handle:
    for line in handle:
        row = json.loads(line)
        if row["split"] == "specialist_train":
            rows.append(row)
        if len(rows) >= 48:
            break
print(f"smoke rows: {len(rows)}  actions="
      f"{ {a: sum(1 for r in rows if r['gold']['action'] == a) for a in ACTIONS} }", flush=True)

out = "artifacts/smoke/reference_smoke"
started = time.time()
record = run_cell(
    config, kind="reference", backbone_key="smoke", seed=7, rows=rows,
    out_dir=out, device=device, max_steps=steps, batch_size=2,
    max_length=512, log_every=4,
)
print(f"\ntrained in {record['wall_seconds']}s  ladder={record['checkpoint_ladder']}", flush=True)

saved = sorted(p.name for p in output_path(out).glob("step*"))
print("saved checkpoints:", saved, flush=True)

adapter = str(output_path(out) / saved[-1])
model, tokenizer = load_backbone(model_id, revision, device=device, adapter_path=adapter)
prompts = [render_prompt(r) for r in rows[:3]]
with torch.no_grad():
    probs = torch.softmax(action_logits(model, tokenizer, prompts, device=device).float(), -1)
print("\nreloaded adapter; post-train action head:", flush=True)
for row, dist in zip(rows[:3], probs):
    shown = {a: round(float(x), 3) for a, x in zip(ACTIONS, dist)}
    print(f"  gold={row['gold']['action']:9s} pred={ACTIONS[int(dist.argmax())]:9s} {shown}",
          flush=True)
print(f"\nTOTAL {time.time() - started:.0f}s   SMOKE OK", flush=True)
