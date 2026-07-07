# Optimizing Two Small Guards to Match the Mini Judges — Qwen2.5-1.5B + SmolLM3-3B

> Detailed, executable plan. Produced by a 5-workstream reviewer panel (per-model training recipe ·
> single-token verdict head · data curriculum · HPO/calibration · evaluation/ablations/budget), each
> grounded by reading the repo; integrated here. Every config/command references a real module.

## 0. Objective & success criteria

Optimize **exactly two** decoder guards — `qwen2.5-1.5b` (Qwen/Qwen2.5-1.5B-Instruct) and `smollm3-3b`
(HuggingFaceTB/SmolLM3-3B) — to compete with GPT-4o-mini / GPT-5-mini. Because SmolLM3 is 3B, the paper
claim reads **"small (sub-3B) open guards match the mini judges."**

- **Quality metric:** Precision / Recall / F1 only (positive class = `unsafe`), macro + per-axis. No ROC-AUC.
- **Efficiency (separate table):** p50/p90 latency (pinned reference device) + \$/1k requests.
- **Pre-registered parity rule** (via `evaluation/stats.py`, §6): an SLM system S is **non-inferior** to
  mini M iff the paired-bootstrap 95% CI lower bound of `F1(S)−F1(M) ≥ −0.03`; **matches** iff also
  `FPR(S)−FPR(M)` CI upper ≤ 0 and `p50(S) ≤ p50(M)/3`; **beats** iff CI lower > 0; else **inconclusive**.
- **Target numbers to beat** (re-freeze from the matched-n run first — current values are mixed-n):
  GPT-5-mini macro-F1 ≈ 0.762 / FPR ≈ 0.223; GPT-4o-mini ≈ 0.833 / 0.263.

**Why this is achievable:** the earlier SLMs were **under-trained** (1 epoch, LoRA r=16, no
completion-only loss), not capacity-limited; the diversity report shows an oracle ceiling ≈0.98. The gap
is training recipe + a red-team **data** gap + missing calibration — all fixable below.

---

## 1. Prerequisite code changes (blockers — do these first)

These are the fixes that unblock everything; most are small and testable with no GPU.

| # | File | Change | Why |
|--:|---|---|---|
| P1 | `training/sft.py` | Add explicit `target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']` to `LoraConfig`; switch to **completion-only loss** (prompt/completion schema or `DataCollatorForCompletionOnlyLM` + `completion_only_loss=True`); add `gradient_checkpointing=True`, `lr_scheduler_type='cosine'`, `warmup_ratio=0.03` | Today no `target_modules` → PEFT adapts only `q_proj,v_proj` (or KeyErrors on SmolLM3); loss covers the whole 300-token sequence so <5% of gradient hits the verdict. **The core under-training fix.** |
| P2 | `training/hyperparams.py`, `runner.build_config` | Bump decoder-SFT defaults: epochs 1→3, `lora_r` 16→32, `lora_alpha` 32→64; add a `target_modules` knob; thread `trust_remote_code` | Defaults currently encode the under-trained regime |
| P3 | `data/training_sets.py::_TRAIN_SPLIT_LOADERS` | Add train-split loaders for `toxicchat`, `prompt_injections`, `jailbreak_classification` | **Leakage blocker:** they currently fall through to `load_benchmark(src)` = the *eval* split, so training would consume benchmark test rows |
| P4 | `models/decoder.py` | Single-token verdict + logprob head (§3) | Calibrated score + ~40× lower latency |
| P5 | `evaluation/stats.py` (NEW) | `paired_bootstrap_ci`, `mcnemar`, `holm` | No CIs/significance exist today — the biggest eval gap; on the critical path |
| P6 | `evaluation/ensembles.py::_align_rows` | **RAISE** on a missing/mismatched `sample_key` instead of silently skipping; add a per-axis equal-weight macro | Matched-n integrity; stop red-team dilution |
| P7 | `evaluation/openai_guards.py` | Capture `resp.usage` (prompt/completion tokens) → \$/1k via dated price | Enables the cost axis |
| P8 | `evaluation/calibration.py` + `eval_splits.py` (NEW) | temperature/Platt + ECE + iso-FPR threshold; dev/test split by `sample_key` grouped by benchmark | §5 |

---

## 2. Per-model training recipe (SFT is the backbone)

**Shared recipe:** LoRA **r=32 / α=64 / dropout=0.05** on the full linear stack; **3 epochs**; lr **2e-4**
cosine, warmup 0.03; **max_seq_len=1024** (bump to 1536 only if §4 tokenization check shows >2–3% of
prompts exceed 1024); **bf16** (fp16/QLoRA not available on MPS); gradient checkpointing; **completion-only
loss**; effective batch = 8; eval each epoch on a grouped-by-benchmark val split (early-stop if epoch-3 < epoch-2).

**Qwen2.5-1.5B-Instruct** — clean non-reasoning instruct base, no `trust_remote_code`. PEFT maps only
`q_proj,v_proj` by default so the explicit `target_modules` list is essential.
```bash
python scripts/train/run_training.py --model qwen2.5-1.5b --technique sft \
  --train-data data/train_sets/guard-main/train.jsonl \
  --epochs 3 --lr 2e-4 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --max-seq-len 1024 --bf16 --seed 42          # CUDA: --batch 16 --grad-accum 1 | MPS: --batch 4 --grad-accum 4
```

**SmolLM3-3B** — **hybrid think/no-think**; a guard must be **no-think**. Double-lock: (a) append
`/no_think` to the SmolLM3 system prompt (or `apply_chat_template(enable_thinking=False)`) and (b) keep
SFT targets JSON/word-only (no `<think>`). This matters because `DecoderGuard.predict` caps sft-mode
generation at 48 tokens — if the model thinks, the trace overruns, `parse_verdict` returns None, and it
**fails closed to unsafe** (silent over-block). Needs `transformers>=4.53` or `trust_remote_code=True`.
```bash
python scripts/train/run_training.py --model smollm3-3b --technique sft \
  --train-data data/train_sets/guard-main/train.jsonl \
  --epochs 3 --lr 2e-4 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --max-seq-len 1024 --bf16 --seed 42          # CUDA: --batch 8 --grad-accum 2 | MPS: impractical (~7h/run)
```

**RL/DPO are ablation-only** (they collapsed to FPR 0.77–0.99). Root cause: `rewards.py` has
`correctness=1.0 == false_positive_penalty=1.0`, so on balanced data "always-unsafe" ties correct
behavior. Fix `false_positive_penalty≈2.5`, run **from the SFT checkpoint** on GPU, and report FPR
recovery as a negative-result row regardless of outcome.

---

## 3. Single-token verdict + logprob head (`models/decoder.py`)

Turn the guard from a JSON generator into a Llama-Guard-style **one-forward-pass** classifier that also
yields a **calibrated score**.

- **Tokenization (verified):** `' safe'` / `' unsafe'` (leading space) are single tokens for Qwen2.5
  (ids 6092/19860 via the cached 0.5B proxy) and for SmolLM2 (2991/20408); **verify SmolLM3 online**
  before training (gate). Keep `build_prompt` ending in `Verdict:` so the first generated token is the
  leading-space verdict word.
- **New path `_predict_verdict`:** one forward pass → `logits[:, -1, :]` → gather the two label logits →
  `p_unsafe = softmax([l_unsafe, l_safe]/temperature)[0]` → `score = p_unsafe`,
  `decision = UNSAFE if p_unsafe ≥ threshold`. `DecoderGuard` gains `mode='verdict'`, `threshold=0.5`,
  `temperature=1.0`. Fail-closed on any exception / non-finite logits (UNSAFE, score=1.0).
- **`resolve_label_ids(tokenizer)`** picks the single-token variant at load, else raises (fail loud).
- **Training:** add a `verdict` branch to `train_decoder` whose target is the bare word (concentrates
  label mass at position 0 so `P(unsafe)` is meaningful). Keep the JSON/`parse_verdict` paths unchanged
  (GRPO reward + all existing tests still pass).
- **Unlocks:** temperature calibration, iso-FPR thresholds, a non-degenerate deferral cascade
  (`defer_rate>0`), soft-vote ensembles, and the p50/p90 win (1 pass vs 48-token generation).

---

## 4. Data curriculum

Measured train-split inventory (deduped): BeaverTails `30k_train` 7,754 (2,066 safe / 5,688 unsafe);
ToxicChat train 4,964 (~4,590 / 374, **non-commercial license**); deepset prompt-injections 546 (343/203);
jailbreak-classification 1,044 (~510/521); OR-Bench (capped) 4,000 all-safe; **JailbreakBench 200 held
out** (LOBO). Over-refusal negatives come from **OR-Bench**, never XSTest (eval-only) — already wired.

**Primary set `guard-main`** (`over_refusal_aware`, count mode over-weights the small red-team sources):
```bash
python scripts/data/build_dataset.py --strategy over_refusal_aware --name guard-main \
  --sources beavertails toxicchat prompt_injections jailbreak_classification --per-class 4000 --holdout 0.2 --seed 42
```
→ ~4,738 train (guardrail 55% / red-team 24% / over-refusal 21%, ~61% safe — mild safe skew fights
over-blocking). **`guard-main-rel`** = same minus ToxicChat (`--per-class 6000`) for a
license-clean *released* checkpoint. **`rt-lobo`** = `red_team` over the two injection sources for the
isolated red-team→JailbreakBench transfer number.

**Verify:** `meta.json` shows `augmentation_added>0`, `augmentation_source=='or_bench'`, and (after P3)
eval `dropped_leaked` is near-zero for the three added sources.

---

## 5. HPO + calibration + threshold protocol

**Bounded search** (spend budget only on the under-training axes):
- **Fix:** dropout 0.05, α=2·r, effective batch 8, seed 42, max_seq_len 512 for the *search* (pins latency).
- **Sweep (Qwen, 2-stage, ~8 runs):** Stage A `r∈{16,32}×epochs∈{2,3}` (lr 2e-4) → 4 runs; Stage B LR
  refine `{1e-4,3e-4}` on the winner → 2; + 2 seeds on the overall winner. **SmolLM3 reduced (~6 runs):**
  rank sweep at epochs=2, add winning-rank at epochs=3, 1 LR refine, 2 seeds (borrow the epochs decision).
- **Selection (Gate G1):** score each candidate → restrict to **DEV** → calibrate → pick iso-FPR
  threshold → **argmax DEV macro-F1 s.t. DEV macro-FPR ≤ mini budget.** Select on the *same*
  calibrated+thresholded pipeline used for TEST.

**Calibration** (`evaluation/calibration.py`, pure Python): `fit_temperature` (golden-section) &
`fit_platt` (GD) on DEV logits; pick lower-ECE; accept only if post-cal ECE ≤ 0.10 & monotone. Global per
model.

**Threshold:** `choose_threshold_iso_fpr` sweeps τ on DEV, keeps FPR ≤ budget, maximizes macro-F1; select
via grouped 5-fold CV within DEV, refit on full DEV, **freeze**, apply once to disjoint TEST. Report τ=0.5
and unconstrained-max-F1 as reference points. **DEV/TEST split** = 40/60 by `sample_key`, stratified within
benchmark, seed 42, cached to `outputs/eval_splits.json`.

---

## 6. Evaluation & statistics (matched-n)

- **Frozen sample set:** union the leaked keys across *both* SLM training files, drop from **all** guards
  (baselines/GPT have no training file → this is the current mixed-n bug), score everyone on the identical
  surviving rows. Publish **Table 1** = per (guard×benchmark) `n / n_safe / n_unsafe` after filtering.
- **Size policy:** full (`--per-class 0`) for the small sets (XSTest 450, JailbreakBench 200,
  prompt_injections ~116, jailbreak_classification); cap the three big content sets at `per_class=800`.
- **Metrics:** P/R/F1 (macro + **per-axis equal-weight** so red-team isn't diluted); FPR@benign secondary.
- **Statistics:** `paired_bootstrap_ci` (B=10000, strata=benchmark) + `mcnemar` + Holm across (axis×pair);
  attach a 95% CI to every SLM−mini difference; apply the pre-registered rule (§0).
- **Baselines:** keyword; Llama-Guard-3-1B, ShieldGemma-2b, PromptGuard-2-86M (in `baselines.py`);
  WildGuard-7B (add a wrapper or report "not run", gated); OpenAI-Moderation, GPT-4o-mini, GPT-5-mini
  (**pin dated snapshots**).
- **Latency/\$:** measured **separately** on the pinned Mac over a fixed 300-sample micro-bench; the
  **single SLM or the cascade** carries the latency claim — never the sum-latency ensemble. \$/1k from
  captured token usage × dated price (moderation = \$0; local SLM = \$0 marginal + amortized GPU figure).

---

## 7. Ablation table (both models unless noted; every row matched-n, with per-axis F1 + CI vs nearest mini)

| Row | System | Isolates | Gate |
|---|---|---|---|
| R0 | Field baselines (keyword, Llama-Guard-3-1B, ShieldGemma-2b, PromptGuard-2-86M, WildGuard?, moderation, gpt-4o-mini, gpt-5-mini) | the bar | — |
| R1 | base: 1 epoch, r16, content-only | naive-SLM floor (≈0.68) | — |
| R2 | +epochs/rank: 3 ep, r32/α64 | **the under-training fix** | macro-F1 0.68→0.75–0.80 |
| R3 | +red-team (LOBO, hold out JailbreakBench) | red-team lift + transfer | red-team F1 0.51→≥0.70; JBB up; no guardrail regression |
| R4 | +over-refusal (OR-Bench) — **headline candidate, ×3 seeds** | FPR/precision | XSTest FPR −≥10 pts; macro-F1 not down > CI |
| R5 | +verdict head (no retrain) | latency/\$ | p50 −5–40×; F1 within CI of R4 |
| R6 | +calibration + iso-FPR threshold | operating point | dev-selected τ, reported on test |
| R7 | +distillation (independent teacher — never a baseline; gold-only control) | distillation delta | marginal F1 vs R4 |
| R8 | **cascade: Qwen gate → SmolLM3 expert** (deferral) | the deployable system | macro-F1 ≥ best single; p50 ≤ mini/3 |
| R9/R10 | +DPO / +GRPO (from SFT, FP-penalty≈2.5×) | RL negative result | FPR ≤ 0.2 at held recall, else documented degeneracy |

---

## 8. Run schedule, compute budget & critical path

**Phase 0 (infra, no GPU):** build `stats.py` + tests; `_align_rows` raise; per-axis macro; token-usage/\$
capture; (optional) WildGuard wrapper. **Gate G0:** `make test` green + a hand-checked CI toy case.
**Phase 1 (first submittable number):** build `guard-main`; train **R4** for both models on A100;
matched-n full eval of R4 + all R0 baselines + GPT judges with prediction dumps; Mac latency micro-bench;
stats CIs + Table 1. **Gate G1 = the honest, leakage-audited, CI-bearing P/R/F1-vs-cost table** (this alone
is a submittable measurement+analysis paper).
**Phase 2 (efficiency):** verdict head (R5) → calibration (R6) → cascade (R8). **Gate G2:** cascade
non-inferior to GPT-5-mini on F1 at p50 ≤ mini/3.
**Phase 3 (sharpening, parallel):** red-team LOBO isolation (R3), 3-seed spread, distillation (R7), RL
negatives (R9/R10).

**Compute budget** (from `runner._eta_seconds`): MPS ≈ 5 s/step (1.5B) / 10 s/step (3B) → a 3B full SFT is
~7 h on MPS (**don't**). **A100** ≈ 30–50 min/run (Qwen), 60–100 min/run (SmolLM3). Full grid ≈ 18–26
GPU-hours ≈ **\$30–55**; eval passes + API (< \$20; gpt-5-mini reasoning tokens dominate) → **~\$70–100
total**. **Where:** all training + quality/prediction eval on a rented A100 (1–2 days); **latency p50/p90
on the pinned Mac** (the "runs on a laptop" story needs the reference device); offline
cascade/deferral/diversity/stats over cached `outputs/predictions/*.json` on any machine, seconds.

**Critical path to the first defensible table:**
`G0 (stats.py + align-raise + per-axis macro) → train R4 for both on A100 → matched-n full eval of R4 +
open-guard baselines + GPT judges (dump predictions) → Mac latency micro-bench + $/1k → bootstrap CIs +
McNemar + Table 1.` Everything else (verdict head, calibration, cascade, distillation, RL negatives) only
sharpens it.

---

## 9. Top risks & gates

- **Mixed-n comparison (FATAL, current bug):** GPT at n=20–160 vs SLMs at 79–122. → `_align_rows` RAISES;
  Table 1 receipt; matched-n assertion.
- **F1 parity may not hold** (best SLM ~0.68–0.80 vs gpt-4o-mini 0.83). → pre-registered non-inferiority
  CI; if it straddles, report "inconclusive" and fall back to the frontier/measurement framing (dominate
  moderation + match GPT-5-mini on over-blocking/latency/cost).
- **SmolLM3 think-leak** → truncated at 48 tokens → fail-closed over-block. → no-think double-lock; DEV
  probe requires `<think>` in <1% of outputs before any eval.
- **Completion-only loss silently not applied** → F1 stays ~0.68. → log non-`-100` label-token count per
  batch (must ≈ verdict length, not full sequence).
- **P3 loader fix omitted** → training on eval rows. → eval `dropped_leaked` must be near-zero for the
  three added sources.
- **Latency claim rides the wrong config** (ensemble latency = sum, ~3240 ms). → F1 and latency claims must
  be the **same** row (single SLM or cascade), measured on the pinned Mac.
- **Red-team may not lift.** → if red-team F1 < 0.70 or JBB transfer < 0.65, invoke the 2-phase curriculum
  and/or add gated red-team data (WildJailbreak/StrongREJECT).
- **ToxicChat non-commercial** → any released checkpoint trains on `guard-main-rel` (ToxicChat dropped).
- **Re-freeze the mini FPR budget** from the matched-n run before the final iso-FPR TEST report.
