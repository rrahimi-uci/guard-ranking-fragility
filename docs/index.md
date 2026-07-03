---
title: "Agent Bouncer — a tiny, fast LLM safety guardrail (SLM)"
description: "Open-source small language model (SLM) guardrail for LLMs and AI agents: detects prompt injection, jailbreaks, and unsafe content on the request path. Fine-tuned + RL-tuned (GRPO), benchmarked vs GPT-4o-mini, GPT-5.2, and OpenAI Moderation."
---

# 🕶️ Agent Bouncer

**A tiny, fast safety bouncer for LLMs and agents** — screens prompts, tool calls, and
outputs *before* they reach your model. SLM guardrails, trained with **fine-tuning + RL**,
**benchmarked** on a standard suite, and shipped with an honest scoreboard and a web studio.

![Benchmark Studio](media/benchmark-studio.png)

## Highlights

- **7-benchmark standard suite** across guardrail, red-teaming, and over-refusal axes.
- **Compared live vs GPT-4o-mini and GPT-5.2 (low reasoning)** + OpenAI Moderation, through one harness.
- **Precision / Recall / F1 / ROC-AUC / latency / throughput / P90** — over-blocking (`fpr_on_benign`) front and center.
- **Full training lifecycle**: model registry (Qwen3, DeepSeek-R1-1.5B, SmolLM2-1.7B, Gemma-1B),
  configurable SFT/GRPO/DPO, **versioning**, **experiment tracking**, **hardware capture**, and
  **train/test leakage guards**.
- **Benchmark Studio** — configure, train, test, and compare from a polished web UI.

## Headline result (7 benchmarks, one harness)

| Guard | Params | macro-F1 | macro-FPR@benign ↓ | p50 ms ↓ |
|-------|-------:|---------:|-------------------:|---------:|
| encoder (distilbert)     | 66M  | 0.579 | 0.288 | **8** |
| decoder-SFT (Qwen3)      | 0.6B | 0.672 | 0.355 | 523 |
| openai-gpt-4o-mini       | api  | **0.788** | 0.266 | 687 |
| **openai-gpt-5.2 (low)** | api  | **0.788** | **0.187** | 1117 |

The **66M encoder ties OpenAI Moderation on macro-F1 at ~22× lower latency**; **GPT-5.2 (low)**
leads on quality *and* over-blocking but is ~140× slower — not a per-call gate.

## Docs

- [Architecture](architecture.md) — how it all fits together (with mermaid diagrams)
- [Benchmarks & results](benchmarks.md) — the full scoreboard + analysis
- [Datasets](datasets.md) · [Taxonomy](taxonomy.md) · [Roadmap](roadmap.md)
- [Auto-generated scoreboard](../outputs/BENCHMARKS.md)

## Run it

```bash
git clone <your-repo-url> agent-bouncer && cd agent-bouncer
make setup && ./start.sh          # → http://127.0.0.1:8000  (Benchmark Studio)
```

Or drive everything from a notebook: [`notebooks/agent_bouncer_studio.ipynb`](../notebooks/agent_bouncer_studio.ipynb).

---
<sub>Apache-2.0 · a **defensive** security tool · no guardrail catches everything — pair with alignment + human review.</sub>
