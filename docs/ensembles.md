# Ensembles — can combining SLMs match GPT-5.2 Low?

**Short answer: partly.** An ensemble of our small guards beats *every* single SLM, matches
GPT-5.2 on over-blocking (FPR), ranking quality (AUC), latency, and cost — but it does **not**
close the gap on headline macro-F1 at a usable operating point. Below is the honest measurement.

## How it's measured (no leakage, no F1-gaming)

1. Each guard is scored **once** per sample on the cached benchmark subsets
   ([`scripts/eval/dump_predictions.py`](../scripts/eval/dump_predictions.py)) — one guard per process, so a
   BERT encoder and a Qwen decoder never co-load (that deadlocks the tokenizer threadpool).
   Output: `outputs/predictions/<guard>.json` = `{benchmark: [[gold, pred, score, latency_ms, sample_key], …]}`.
   The trailing `sample_key` (a hash of the normalized prompt) lets members be aligned by prompt
   identity rather than by list position — so guards scored on differently-filtered subsets combine correctly.
2. Ensembles are combined **offline** from those per-sample rows
   ([`scripts/eval/eval_ensembles.py`](../scripts/eval/eval_ensembles.py)) via the pure
   [`combine()`](../src/agent_bouncer/models/ensemble.py) function — so any combination is free to explore.
   Ensemble latency per sample = **sum** of member latencies (members run sequentially).
3. The `ensemble-tuned` row tunes its decision threshold on a **validation half** and reports on a
   **disjoint test half**, and maximizes F1 **subject to** `val-FPR ≤ 0.20` — i.e. at GPT-5.2's
   over-blocking budget, not by flagging everything to inflate recall.

Reproduce:

```bash
for g in keyword-baseline encoder-distilbert; do python scripts/eval/dump_predictions.py --guard $g; done
python scripts/eval/dump_predictions.py --guard decoder-sft-0.6B  --device mps
python scripts/eval/dump_predictions.py --guard decoder-grpo-0.6B --device mps
python scripts/eval/dump_predictions.py --guard openai-moderation --workers 8
python scripts/eval/eval_ensembles.py --merge ensemble-maj5 ensemble-union2
```

## Results (7 benchmarks, per_class=100, macro-averaged)

> **Illustrative figures from a reference run** — not shipped in the repo. Regenerate your own
> (`make bench` now dumps per-sample predictions automatically, then build ensembles from them).

| Guard / ensemble | macro-F1 | ROC-AUC | FPR (over-block) ↓ | p50 latency |
|---|--:|--:|--:|--:|
| **openai-gpt-5.2-low** *(target)* | **0.804** | **0.823** | 0.184 | 1196 ms |
| openai-gpt-4o-mini | 0.794 | 0.796 | 0.266 | 744 ms |
| — best single SLM (decoder-grpo-0.6B) | 0.673 | 0.667 | 0.377 | 298 ms |
| — encoder-distilbert | 0.579 | 0.703 | 0.288 | **9 ms** |
| `ensemble-union2` (enc ∪ sft) — **max recall** | **0.692** | 0.695 | 0.455 | 291 ms |
| `ensemble-wtd` (weighted) — **best AUC** | 0.664 | **0.773** | 0.335 | 793 ms |
| `ensemble-maj5` (5-way majority) — **balanced** | 0.617 | 0.770 | 0.216 | 793 ms |
| `ensemble-tuned` (F1 s.t. FPR ≤ 0.20) | 0.566 | 0.770 | **0.166** | 799 ms |
| `ensemble-inter2` (enc ∩ sft) — **min over-block** | 0.542 | 0.720 | 0.188 | 291 ms |

Members: `encoder-distilbert`, `decoder-sft-0.6B`, `decoder-grpo-0.6B`, `openai-moderation`,
`keyword-baseline`. Full per-benchmark numbers in `outputs/ensemble_results.json`.

## What this shows

- **Ensembling is a free win over any single SLM.** `union2` lifts macro-F1 to **0.692** vs the best
  single SLM's 0.673, and the weighted ensemble lifts AUC to **0.773** vs the best single 0.703.
- **Competitive with the frontier on three of four axes.** The weighted ensemble's **ranking quality
  (AUC 0.773)** is within 0.05 of GPT-5.2 (0.823) and near gpt-4o-mini (0.796). A consensus ensemble
  **matches GPT-5.2's over-blocking** (`inter2` FPR 0.188, `tuned` 0.166 vs GPT-5.2's 0.184). And it
  does so **1.5–4× faster** (union2 291 ms vs 1196 ms) at a fraction of the cost (local SLMs + one
  cheap Moderation call vs a frontier LLM per request).
- **It does not close the F1 gap.** At GPT-5.2's over-blocking budget (FPR ≈ 0.18) the best ensemble
  reaches macro-F1 ≈ **0.62**, ~0.18 below GPT-5.2's 0.804. You can trade that for recall
  (`union2` F1 0.692) but only by over-blocking 2.5× as often.

## Why — and what would actually close it

The members are **correlated**: the encoder, the 0.6B SFT decoder, and the GRPO decoder are all
fine-tuned on the same BeaverTails-style data, so they tend to make the *same* mistakes. Ensembles
only help when members fail *independently*; Moderation adds some diversity (it drives the AUC/FPR
gains) but is itself weak on these sets. Closing the remaining F1 gap needs **error diversity**, not
more voting — a stronger/larger base model, more or broader training data, or a genuinely different
model family — rather than another copy of the same recipe.

**Bottom line:** ship the ensemble for its better-than-any-single-SLM quality, matched over-blocking,
and lower latency/cost — but don't claim F1 parity with GPT-5.2 Low. The chosen `ensemble-*` columns are
merged into the scoreboard (`outputs/benchmark_results.json`) and appear on the **Leaderboard**.

## Build ensembles interactively in the Workbench

You don't need the CLI to explore ensembles. Every benchmark run dumps each guard's per-sample
predictions to `outputs/predictions/<guard>.json`, and the **Agent Bouncer Workbench → Leaderboard** tab has
an **Ensemble builder**:

1. Pick two or more member guards (any guard that has dumped predictions — trained models, GPT
   judges, OpenAI Moderation, keyword).
2. Choose a strategy — **union · intersection · majority · mean · weighted** — and, for `mean` /
   `weighted`, a decision threshold.
3. Click **Build & score**. The ensemble is combined offline from the dumped predictions (no model
   inference, no cost), scored across the shared benchmarks, and dropped straight onto the
   leaderboard as a new `ensemble-…` row you can compare and export to PDF.

Under the hood this is the same pure [`combine()`](../src/agent_bouncer/models/ensemble.py) and
[`evaluate_ensemble()`](../src/agent_bouncer/evaluation/ensembles.py) used by the CLI, exposed via
`GET /api/ensemble/members` and `POST /api/ensemble`.

### Strategies at a glance

| Strategy | Flags unsafe when… | Tends to |
|---|---|---|
| `union` | **any** member flags | maximize recall (more over-blocking) |
| `intersection` | **all** members flag | maximize precision (less over-blocking) |
| `majority` | a **strict majority** flag | balance the two |
| `mean` | the **mean** of member scores ≥ threshold | soft vote (tunable) |
| `weighted` | the **weighted mean** of scores ≥ threshold | upweight stronger members |

### Recall→precision cascade

A **cascade** chains two models instead of voting: a **high-recall gate** flags every candidate, then
a **high-precision filter** runs *only on the flagged inputs* and makes the final call
(`unsafe = gate AND filter`). Because the (slower, stronger) filter is skipped for everything the gate
clears, the cascade's average latency is `gate + flag-rate × filter` — far below running both on every
input — while precision rises toward the filter's. The Workbench's **⛓ Recall→precision cascade**
button auto-picks the highest-recall small model as the gate and the highest-precision one as the
filter, scores the pair offline, and adds an **`Cascade · recall→precision`** row to the leaderboard so
you can compare its P / R / F1 / AUC / p50 / p90 directly against the GPT baselines.

> **Note — an AND-cascade trades recall for precision; it does not raise both.** Final `unsafe = gate AND filter`,
> so `recall(cascade) ≤ min(recall of the two stages)`. It boosts precision (and, if the gate cheaply clears
> easy negatives, cuts cost) but can't lift recall above the weaker stage. Use it when you want fewer false
> positives; use `union` when you want higher recall.

### Confidence-deferral cascade

A different two-stage design that is *not* recall-capped: a cheap **decider** handles the inputs it's
confident about (score ≤ `low` → safe, score ≥ `high` → unsafe) and **defers only the uncertain middle
band** to an **expert** model. Final latency = decider + (defer-rate × expert), and because it's a router
(not an AND vote) it can improve the whole operating point when the decider has a usable confidence signal.
The **🔀 Confidence-deferral cascade** button auto-searches decider × expert × band for the best macro-F1 and
adds a **`Cascade · confidence-deferral`** row (with its defer-rate). With binary 0/1 scores nothing lands in
the band and it degenerates to the decider alone — a continuous-confidence model (the encoder, or a calibrated
head) is what makes it pay off.

### Diversity report — *can ensembling even help?*

Before trusting any combiner, the **🔬 Diversity report** answers whether the members are complementary at
all. It aligns them by sample identity and reports per-model accuracy, mean pairwise **disagreement**, mean
**error-correlation** (Yule's Q), and the **oracle ceiling** — the accuracy you'd reach if you could pick the
right member for every sample. If the oracle barely exceeds the best single model (small *headroom*), the
members are **redundant** and no combiner will help — you need more diverse or stronger base models. Large
headroom with low error-correlation means ensembling *should* pay off (and points you at soft/router combiners
rather than a hard AND/OR vote).
