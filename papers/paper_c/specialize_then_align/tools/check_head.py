#!/usr/bin/env python
"""Verify the three-way action head and masked response log-probability end to end.

Run with a tiny backbone to prove the plumbing without GPU spend:
    python tools/check_head.py HuggingFaceTB/SmolLM2-135M-Instruct main cpu
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from paper_c_sta.modeling import (  # noqa: E402
    ACTIONS,
    action_logits,
    action_token_ids,
    load_backbone,
    render_prompt,
    render_response,
    response_logprob,
)

model_id = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
revision = sys.argv[2] if len(sys.argv) > 2 else "main"
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"

start = time.time()
model, tokenizer = load_backbone(model_id, revision, device=device)
print(f"loaded {model_id} on {device} in {time.time() - start:.0f}s", flush=True)
print("action first-token ids:", dict(zip(ACTIONS, action_token_ids(tokenizer))), flush=True)

cohort = pathlib.Path(__file__).resolve().parents[1] / "artifacts/cohort/samples.jsonl"
by_action: dict[str, dict] = {}
with cohort.open() as handle:
    for line in handle:
        row = json.loads(line)
        by_action.setdefault(row["gold"]["action"], row)
        if len(by_action) == len(ACTIONS):
            break

picks = [by_action[a] for a in ACTIONS if a in by_action]
prompts = [render_prompt(s) for s in picks]
print("prompt ends at the action position:",
      all(p.endswith('{"action": "') for p in prompts), flush=True)

with torch.no_grad():
    probs = torch.softmax(action_logits(model, tokenizer, prompts, device=device).float(), -1)
for sample, row_probs in zip(picks, probs):
    shown = {a: round(float(x), 3) for a, x in zip(ACTIONS, row_probs)}
    print(f"  gold={sample['gold']['action']:9s} "
          f"pred={ACTIONS[int(row_probs.argmax())]:9s} {shown}  [{sample['category']}]",
          flush=True)

with torch.no_grad():
    value, truncated = response_logprob(
        model, tokenizer, prompts[0], render_response(picks[0]), device=device
    )
print(f"response_logprob={float(value):.2f}  truncated={truncated}", flush=True)
print("target response:", render_response(picks[0])[:120], flush=True)
print("DONE", flush=True)
