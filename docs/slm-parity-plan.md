# Plan — matching GPT-4o-mini / GPT-5-mini with small models

**Goal.** A small-model guard (a single fine-tuned SLM, or an ensemble/cascade of them) that
**matches GPT-4o-mini and GPT-5-mini** on the 7-benchmark suite — same or better macro-F1 *and*
over-blocking (`fpr_on_benign`) — at a **fraction of their latency and cost**. The deliverable is a
reproducible notebook (`notebooks/slm-parity.ipynb`) that demonstrates it end-to-end.

This plan is built from *this repo's own measurements*, not aspiration. The headline reason to believe
it's achievable is in the data (§1).

---

## 0. Where we are (measured, latest run)

Macro-averaged over the 7 benchmarks, one harness. ⚠️ GPT baselines were scored on **n=20** and the
small models/ensembles on **n≈108** this run, so the very first task (§4) is a fair re-eval.

| | macro-F1 | ROC-AUC | FPR@benign | p50 |
|---|--:|--:|--:|--:|
| **openai-gpt-4o-mini** (target) | **0.865** | 0.850 | 0.243 | 790 ms |
| openai-gpt-5.4-mini (target) | 0.773 | 0.793 | 0.214 | 546 ms |
| openai-moderation | 0.641 | 0.721 | **0.129** | 251 ms |
| best single small model (`qwen3-1.7b-sft`) | 0.682 | 0.743 | 0.135 | 311 ms |
| best ensemble so far (`max-F1`, mean of 5) | 0.731 | 0.744 | 0.254 | — |
| best over-block (`cascade` / `min-FPR`) | 0.682 / 0.369 | 0.744 / 0.615 | **0.135 / 0.041** | 515 ms |

**The gap to close:** ~**0.13–0.18 macro-F1** vs GPT-4o-mini (we already beat / match GPT-5-mini's F1
region and Moderation on over-blocking).

---

## 1. Why parity is plausible — the evidence

Five findings from this project define the strategy:

1. **The members are complementary, not redundant.** The diversity report over our 9 small models:
   **oracle ceiling ≈ 0.98 vs best-single ≈ 0.75 (headroom 0.23)**, mean pairwise error-correlation
   (Yule's Q) **0.22** (low), and they over-block the *same* benign prompt only **4.5%** of the time.
   → *If we could route each sample to the right member we'd hit ~0.98.* The information to beat the
   frontier is already in the pool; the bottleneck is the **combiner + calibration**, not capacity.
2. **The RL/DPO decoders are degenerate.** `GRPO`/`DPO` variants over-block at **FPR 0.77–0.99** —
   they collapsed toward "always unsafe," so their high recall is meaningless. Only the **SFT**
   decoders are usable. → Big, cheap recoverable F1 by fixing the reward/preference training.
3. **Red-teaming is the weak axis.** Prompt-injection mean-F1 ≈ **0.51** (lowest of all axes) — the
   SLMs were trained on *content-safety* data, not injection/jailbreak. → A targeted-data gap, not a
   model-size gap.
4. **The scores are binary.** Decoders emit 0/1 (plus a 0.5 fail-closed accident), which kills
   soft-vote, threshold sweeps, and confidence-deferral. The deferral cascade only "worked" by
   exploiting that 0.5 marker. → A **calibrated continuous score** unlocks the whole ensemble toolkit.
5. **Cascades already reach frontier-level over-blocking.** The recall→precision cascade hits
   **FPR 0.135** and `min-FPR` hits **0.041** — at or below Moderation — at local latency. → Usability
   is already solved; the remaining work is *recall/F1* without giving that back.

---

## 2. Strategy — five levers, ordered by expected ROI

| # | Lever | Why (from §1) | Expected effect | Repo tools |
|--:|---|---|---|---|
| L1 | **Fix degenerate RL/DPO training** | #2 | recover FPR on the RL/DPO members → more usable, diverse members | `training/rewards.py` (stronger FP penalty), `training/dpo.py`, `run_training.py` |
| L2 | **Learned router / stacking meta-model** over member scores | #1 | capture the 0.23 oracle headroom that hard union/majority leaves on the table — **highest leverage** | new `ensembles.stack()`; `outputs/predictions/*.json` |
| L3 | **Calibrated continuous score** for the decoders | #4 | unlock soft-vote / weighted / deferral / real AUC | new logit/log-prob head in `models/decoder.py` |
| L4 | **Frontier distillation** (GPT-4o-mini as teacher) | closes the raw quality gap | transfer the frontier decision boundary into a fast SLM student | `build_dataset.py` + teacher labels + `run_training.py` |
| L5 | **Close the red-team gap with targeted data** | #3 | lift the weakest axis (injection F1 0.51) | `build_dataset.py --strategy red_team`; deepset/jailbreak train splits; WildJailbreak/StrongREJECT (gated) |

Then **compose**: cascade/route the improved, decorrelated, calibrated members for frontier F1 at
frontier-beating latency.

**Order of attack:** L2 first (pure offline analysis on cached predictions — cheapest, tells us the
achievable ceiling with today's models), then L3+L1 (retrain a few members), then L4/L5 (new data),
then final composition.

---

## 3. The notebook — `notebooks/slm-parity.ipynb`

A single reproducible notebook that walks the levers and reports parity. Cell plan:

1. **Setup & load** — read `outputs/benchmark_results.json` + `outputs/predictions/*.json`; align
   members by `sample_key` (reuse `ensembles._align_rows`). Print the current gap table (§0).
2. **The ceiling** — reproduce `diversity_report`: per-model accuracy, oracle ceiling, headroom,
   error-correlation. This is the notebook's thesis cell — *"routing could reach 0.98."*
3. **Experiment A — stacking / router (L2).** Fit a small meta-model (logistic regression, then
   gradient-boosted trees) on the members' per-sample `[u, score]` features with grouped CV by
   benchmark; report macro-F1 / FPR vs best-single and vs the frontier. *Runnable today on cached
   preds — the first real parity signal.*
4. **Experiment B — calibration (L3).** Temperature/Platt-scale each member's score; re-run
   `mean`/`weighted` ensembles + the confidence-deferral cascade; show the operating-point curve the
   binary scores couldn't produce.
5. **Experiment C — fix RL/DPO (L1).** Retrain one GRPO + one DPO member with a stronger
   false-positive reward/preference; show FPR drops from ~0.9 toward ~0.15 without losing recall.
6. **Experiment D — red-team data (L5).** Build a `red_team` training set (+ injection/jailbreak
   sources), retrain an SFT decoder, show per-axis lift on `prompt_injections` / `jailbreak_*`.
7. **Experiment E — distillation (L4).** Label a large unlabeled prompt pool with GPT-4o-mini, SFT a
   small decoder (and the 66M encoder) on the teacher labels; measure student vs teacher.
8. **Compose & final parity check** — build the best cascade/router over the improved members; fair
   re-eval at matched n (§4); side-by-side vs GPT-4o-mini / GPT-5-mini on macro-F1, per-axis, FPR,
   p50/p90, and **$ / 1k requests**.
9. **Verdict** — parity table against the success criteria (§5), with honest caveats.

Cells 1–3 run against the artifacts already in the repo; 4–8 call the existing `scripts/` +
Workbench APIs so the notebook stays a thin, reproducible driver, not a fork of the pipeline.

---

## 4. Fair evaluation protocol (do this first)

The current cross-group comparison is invalid (GPT at n=20, SLMs at n≈108).

- Re-score **GPT-4o-mini + GPT-5-mini + Moderation at the same per-class as the SLMs** (n≥108, ideally
  the full benchmarks): `make bench` / the "🛰 Run frontier baselines" button, or `run_benchmarks.py
  --per-class 0 --guards openai-*`.
- Report **per-axis** (guardrail / red-team / over-refusal) *and* macro; **FPR@benign** as the headline;
  **p50/p90 latency** and **cost per 1k requests**.
- Keep the audit invariants: one ROC-AUC definition for every row, leakage-guarded test sets, and the
  mixed-`n` flag off.

---

## 5. Success criteria (definition of done)

A small-model pipeline (single model, ensemble, cascade, or router — all SLM-only, no frontier call at
inference) that, on a **fair** run at matched n:

- **Matches GPT-5-mini:** macro-F1 **≥** GPT-5-mini, with **FPR@benign ≤** GPT-5-mini.
- **Approaches GPT-4o-mini:** macro-F1 **within 0.03** of GPT-4o-mini, with **FPR@benign ≤** GPT-4o-mini.
- **At a fraction of the cost:** **p50 latency ≤ 1/3** of the frontier judge and **no per-request API
  cost** (or one cheap Moderation call at most).
- **Reproducible:** the notebook runs top-to-bottom and regenerates every number.

Stretch: a **single distilled SLM** (not an ensemble) that clears the GPT-5-mini bar — the cleanest,
fastest deployable guard.

---

## 6. Milestones

| Milestone | Levers | Exit signal |
|---|---|---|
| **M1 — fair baseline + ceiling** | §4, L2 | GPT re-scored at matched n; stacking result in the notebook (does routing beat best-single toward oracle?) |
| **M2 — calibrated soft ensembles** | L3 | continuous scores; `mean`/`weighted`/deferral improve on hard votes |
| **M3 — recover the RL/DPO members** | L1 | GRPO/DPO FPR ≤ 0.2 at held recall; more diverse usable members |
| **M4 — red-team + distillation** | L5, L4 | injection/jailbreak axis lifts; a distilled student beats every prior single SLM |
| **M5 — compose + verdict** | all | final cascade/router hits the §5 criteria; write-up + PDF report |

---

## 7. Risks & honest caveats

- **The oracle 0.98 is an upper bound** — it assumes a perfect per-sample router. A real meta-model
  captures only a fraction; if stacking (M1) barely beats best-single, the headroom isn't practically
  reachable and we lean harder on distillation (L4).
- **n=20 GPT numbers are noisy** — the whole comparison must be re-run at matched n before any claim.
- **Distillation inherits the teacher's errors** and costs API calls to label; budget for a bounded
  labeling pool.
- **Gated data** (WildJailbreak, StrongREJECT, WildGuardMix) needs HF license acceptance for L5.
- **Latency of soft ensembles** — running 5 members is slower than one model; the cascade/router must
  keep p50 within the §5 budget (the recall→precision cascade already shows how: run the expensive
  member only when needed).

---

## 8. First concrete step

Scaffold `notebooks/slm-parity.ipynb` with cells 1–3 (load → gap table → oracle ceiling → **stacking
meta-model on the cached predictions**). That single experiment tells us — today, with no retraining —
how much of the 0.23 headroom a learned combiner can actually capture, which decides how much of L3–L5
we need. Then re-score the GPT baselines at matched n (§4) so every subsequent number is honest.
