# Agent Bouncer — arXiv Paper Strategy

> **How this was produced.** A 5-lens reviewer panel (experimental design · evaluation/statistics ·
> reproducibility · novelty/positioning · adversarial reviewer), each grounded by reading this repo,
> synthesized by a senior-author pass. Numbers were recomputed from `outputs/benchmark_results.json`.

## Scope decisions (latest guidance) — read first

These two decisions override any conflicting emphasis in the synthesized plan below:

1. **Quality metrics = Precision / Recall / F1 only.** We drop ROC-AUC as a headline (with binary
   decoder scores it degenerates to a single operating point anyway — dropping it also removes the
   "your AUC isn't a real AUC" reviewer attack). Report **macro + per-axis P/R/F1**, positive class =
   `unsafe` (Recall = catch rate; Precision = don't-cry-wolf). `fpr_on_benign` is kept only as an
   optional secondary usability figure (on class-balanced sets Precision already encodes over-blocking).
2. **Efficiency axis (separate from quality) = p50/p90 latency + $/1k requests.** This is the cost
   thesis, not a "quality metric," and stays. The paper's contribution is the **P/R/F1-vs-cost Pareto
   frontier**, not a single-number "we beat mini."

### Efficiency architectures & techniques to leverage (the cost/latency levers)

Ordered by return-on-cost; ✅ already in the repo, 🔧 to build:

- ✅ **Encoder classifier** (ModernBERT/DistilBERT) — one forward pass, `softmax → P(unsafe)`; the
  latency hero and per-call gate.
- 🔧 **Decoder → single-token verdict + logprob** (Llama-Guard style) — read the first-token log-prob
  of `safe`/`unsafe` instead of generating JSON: ~1 forward pass **and** a calibrated score in one
  change (fixes the binary-score problem + cuts decoder latency). *Highest-leverage new technique.*
- 🔧 **Frontier distillation** into the small student — biggest quality-per-parameter lever (with the
  anti-circularity firewall: teacher ≠ any reported baseline).
- ✅ **Cascade / confidence-deferral** — cheap gate clears the confident majority, escalate only the
  uncertain tail: average cost = gate + defer_rate × expert. The deployment-level cost saver.
- ✅ **Quantization + GGUF/MLX/ONNX** (`deploy.py`) — INT4/INT8, CPU/edge-viable, ~minimal quality loss.
- ✅ **LoRA/QLoRA** training; 🔧 **calibration** (temperature/Platt) to unlock thresholds/deferral.
- 🔧 **Keyword/regex stage-0 prefilter**; ⚖️ **ensembles** buy quality but *raise* per-call cost (use
  cascades for cost, ensembles for the quality ceiling).

**Recommended efficient guard:** a calibrated small student (ModernBERT encoder *or* 1.5B decoder with a
single-token verdict head), 4-bit quantized, trained by SFT (+ optional distillation), deployed as a
**calibrated cascade** — reported as P/R/F1 vs latency/$.

---

# Agent Bouncer — arXiv Paper Research Strategy (Senior Author's Integrated Plan)

*Grounded in the repo at commit `28f6860`. Numbers below were recomputed from `outputs/benchmark_results.json` (per_class=80) and cross-checked against `docs/benchmarks.md` / `docs/ensembles.md`. Where I quote a doc I say so; where I recomputed I say "measured".*

## Preface — what the evidence actually says (and where I overrule the brief)

Measured macro over the 7 benchmarks (from `outputs/benchmark_results.json`, **mixed-n, invalid as a comparison** — this is exactly the problem we fix):

| guard | macro-F1 | macro-FPR@benign | p50 ms | n-range |
|---|--:|--:|--:|--:|
| openai-gpt-4o-mini | 0.833 | 0.263 | 743 | 20–160 |
| openai-gpt-5.4-mini | 0.762 | 0.223 | 611 | 20–160 |
| openai-moderation | 0.648 | 0.125 | 192 | 20–160 |
| `qwen3-1.7b-sft` (best single SLM) | 0.682 | 0.135 | 311 | 79–122 |
| `ensemble-best-f1` | 0.731 | 0.254 | 3240 | 79–122 |
| `ensemble-cascade` | 0.682 | 0.135 | 515 | 79–122 |
| RL/DPO decoders (`*-grpo`, `*-dpo`) | 0.66–0.69 | **0.77–0.99** | — | degenerate |
| `deepseek-r1-1.5b-sft` | **0.503** | 0.177 | 271 | under-trained |
| `smollm2-1.7b-sft` | **0.513** | 0.123 | 361 | under-trained |

Three corrections to the brief, verified in-repo:
- **There is no true 1.5B instruct model in the registry.** `src/agent_bouncer/models/registry.py` has `distilbert` (66M), `modernbert-large` (395M), `qwen3-0.6b`, `qwen3-1.7b`, `deepseek-r1-1.5b` (reasoning-distill), `smollm2-1.7b`, `gemma-1b` (gated). The ML expert is right: we must **add `Qwen2.5-1.5B-Instruct`** to be able to make a "~1.5B guard" claim honestly.
- **`deepseek-r1-1.5b-sft` is not "always-unsafe" (FPR 0.99).** Measured it is FPR 0.177 but macro-F1 0.503 — it collapses toward *under-detection*, not over-blocking. The FPR 0.92–0.99 collapse is the **`-grpo`/`-dpo`** variants (all families). Either way the conclusion stands: RL/DPO out, DeepSeek-R1 is a poor short-JSON substrate.
- **The strong claim ("a 1.5B matches the minis on accuracy") is not supported by our own data** — `docs/ensembles.md` states outright "do not claim F1 parity". I therefore commit the paper to the **frontier/reproducibility framing** as primary, F1-parity as a pre-registered stretch goal. This is the only version that survives Reviewer 2.

---

## 1. The claim

**Primary thesis (defensible, what the paper leads with).**
> Under a single leakage-audited harness at matched sample sizes, a ~1.5B open small language model — as a single SFT model and as cheap ensembles/cascades of small guards — *matches* the closed "mini" frontier judges (GPT-4o-mini, GPT-5-mini) and a moderation API on the **deployment-critical axes**: over-blocking (`fpr_on_benign`), ranking quality (ROC-AUC on continuous-score systems), p50 latency, and $/1k requests. On raw macro-F1 the SLM still **trails GPT-4o-mini by a measured, bounded margin**, and a complementarity-ceiling diagnostic (oracle ≈0.98 vs best-single ≈0.75, low Yule's-Q error correlation) **localizes that residual gap to the combiner and training data, not model capacity**. We release the harness, cached subsets, per-sample prediction dumps, and checkpoints so the *open* half of the comparison is bit-reproducible and the *closed* half is pinned to dated snapshots.

**Operational definition of "competes with mini" (pre-registered before the final run).** On an axis, SLM system S is:
- **non-inferior** to mini M iff the paired-bootstrap 95% CI lower bound of `F1(S) − F1(M)` ≥ −δ, δ = 0.03 (matches `slm-parity-plan.md` §5), **and** the CI upper bound of `FPR(S) − FPR(M)` ≤ 0 (no worse over-blocking);
- **matches** iff non-inferior on `fpr_on_benign` **and** ROC-AUC **and** p50 latency ≤ ⅓·M;
- **beats** iff the CI lower bound of the difference > 0;
- otherwise **inconclusive** (reported as such — never dressed up as "competitive").

**Minimal defensible fallback** (if even the frontier framing wobbles under CIs at achievable n): retitle to a *measurement + analysis resource* paper — "we contribute the leakage-audited matched-n harness, the honest over-blocking/latency/cost Pareto frontier across open SLMs vs closed minis, and the complementarity diagnostic; small guards already dominate the moderation API and match GPT-5-mini on over-blocking at ⅓ the latency, and we quantify exactly where they still trail." No unqualified "matches" ever appears.

**Stretch (the headline if it lands):** a single distilled `Qwen2.5-1.5B-Instruct` student clears the GPT-5-mini bar on macro-F1 and comes within δ=0.03 of GPT-4o-mini at ≤ their FPR — the clean, fast, deployable guard.

---

## 2. Method

**Model.** Add `Qwen2.5-1.5B-Instruct` to `registry.py` and designate it **"the 1.5B guard"** (true 1.5B, clean instruct base). Keep `qwen3-0.6b`, `qwen3-1.7b`, `distilbert` (66M), `modernbert-large` (395M) for the size/family ablation. `deepseek-r1-1.5b` appears **only** as a negative-result substrate row.

**Training recipe (SFT is the backbone).**
- Decoder LoRA SFT via `training/sft.py` `train_decoder`, **≥2 epochs, LoRA r≥16** (the 1-epoch/low-rank under-training is why `qwen3-1.7b-sft`/`smollm2-1.7b-sft` look weak — `docs/benchmarks.md` says so explicitly). The trainer is fine; the fix is **data + epochs + rank**, not the trainer.
- Encoder SFT (`distilbert`, `modernbert-large`) — the continuous-score, low-latency member that anchors the cascade.
- **Required code change (highest single-code leverage):** add a continuous probability head to `models/decoder.py`. Today `Verdict.score` is binary 0/1 (line 96) and fail-closed forces 1.0 (line 165), so decoder ROC-AUC degenerates to the operating-point estimate in `curves.auc_with_fallback` and the deferral cascade collapses to stage-1. Read the log-prob of the first "safe"/"unsafe" verdict token and store its normalized value as `score`. This unlocks real swept AUC, calibration (temperature/Platt), soft-vote, and the confidence-deferral router (`evaluate_deferral`).
- **RL/DPO are cut from the critical path.** Present GRPO/DPO as a documented negative-result ablation. Root cause is in `training/rewards.py`: `correctness=1.0 == false_positive_penalty=1.0` (verified), so on balanced data the optimizer wanders to a degenerate corner (FPR 0.77–0.99) under bounded CPU steps. *If* RL stays in the paper at all, run it GPU-side from the SFT checkpoint with FP-penalty ≈2–3× correctness and report whether FPR recovers; otherwise drop it.

**Data mixture** (built with `data/training_sets.py` strategies `mixed` / `red_team` / `over_refusal_aware`):
- **Content safety:** BeaverTails `30k_train` + ToxicChat train.
- **Red-team:** deepset/prompt-injections train + jackhhao/jailbreak-classification train (this is the fix for the F1≈0.51 red-team collapse — a *data* gap, not a size gap).
- **Over-refusal — leakage hazard, must fix:** `training_sets.py` `over_refusal_aware` currently pulls XSTest safe prompts (`loader("xstest")`, line ~179). **XSTest is the over-refusal EVAL set (450 rows) — training on it inflates the headline `fpr_on_benign`.** Source over-refusal negatives from **OR-Bench** (or synthesize benign-but-scary prompts) instead; the eval-time `find_leakage` filter is a backstop, not a license to train on eval data.
- **Calibration split:** carve a dev split (grouped by benchmark) for temperature scaling + threshold selection.

**Distillation (optional, with anti-circularity firewall).** The teacher **must be disjoint from every reported baseline**. Since the 7 benchmarks are already labeled: run **gold-label SFT first** as the control; only add teacher *soft labels* to expand an **unlabeled** prompt pool. If used, teacher = GPT-5.x-**high** (a tier we do not claim parity with) or an ensemble of open guards (Llama-Guard-3 + WildGuard); **GPT-4o-mini / GPT-5-mini stay untouched baselines.** Always ship the gold-only student as the control so the marginal distillation delta is visible.

**Held out.** JailbreakBench (leave-one-benchmark-out target for red-team generalization); a second over-refusal set (OR-Bench/PHTest) as OOD; all gated suites reported "not run"; test split disjoint from every dev/threshold-selection split, grouped by benchmark so no benchmark leaks across.

---

## 3. Evaluation protocol

1. **Matched-n re-run (do this first — it invalidates every current cross-group number).** Re-score **every** guard (SLMs, encoder, open baselines, GPT minis, moderation) on the **identical leakage-filtered `sample_key` set per benchmark**, full benchmarks where feasible (`run_benchmarks.py --per-class 0`, or `download_full_benchmarks.py`: BeaverTails 3021, ToxicChat 5083, OpenAI-Moderation 1680, XSTest 450). Extend `ensembles._align_rows` to **raise, not silently skip**, when a guard is missing a sample. Publish **Table 1 = per (guard×benchmark) n / n_safe / n_unsafe after leakage filtering** as the matched-n receipt. Flag `prompt_injections` (capped ~56/class by `balanced_subset`) as the lowest-power benchmark.
2. **Metrics.** Precision/recall/F1 (positive=unsafe), **`fpr_on_benign` as headline**, ROC-AUC, p50/p90/p95 latency, throughput, **$/1k requests** (new — from logged token counts × pinned list price).
3. **AUC hygiene.** `curves.auc_with_fallback` returns rank-AUC for continuous scores and `(recall+1−fpr)/2` for binary ones. **Do not rank binary-score guards against the continuous encoder by "ROC-AUC."** Relabel the binary value **"Balanced Accuracy (single point)"** and reserve swept ROC-AUC for continuous-score systems (encoder, mean/weighted ensembles, the new calibrated decoder head).
4. **Threshold / operating point.** For continuous guards, select the threshold on the **dev split** to maximize F1 subject to dev-FPR ≤ the mini's FPR (iso-FPR), report on disjoint **test** — the `ensemble-tuned` recipe in `docs/ensembles.md`, but with dev/test carved **grouped by benchmark**. For fixed-point guards (decoders, GPT) also run iso-FPR: tune the SLM to the mini's FPR, compare F1 at equal over-blocking.
5. **Uncertainty (new module `evaluation/stats.py` — nothing in the repo does this today).** `paired_bootstrap_ci(rows_a, rows_b, metric, B=10000, seed=0, strata="benchmark")` over the aligned per-sample rows, and `mcnemar(pred_a, pred_b, gold)` (discordant b,c + exact-binomial p). Attach 95% CIs to **every** headline metric and every SLM−mini difference; **Holm-correct** McNemar across the (axis × pair) family. No claim rests on a point estimate.
6. **Per-axis primary.** Report guardrail / red-team / over-refusal each with CIs, then a **macro that weights the three axes equally** — not the current unweighted mean over 7 benchmarks (which counts guardrail 3×, over-refusal 1× and dilutes the red-team failure).
7. **Multi-seed.** Train each headline SFT member with ≥3 seeds via `run_training.py`; report mean±std; headline uses one pinned seed with the spread disclosed.
8. **Contamination.** Surface `data/split.find_leakage` (exact + Jaccard≥0.9, min_tokens=5) `dropped_leaked` per benchmark as a table; run it over the **union** of every member's training file **plus** any teacher-labeled pool. State plainly that mini training corpora are unknowable → their contamination is an unquantifiable caveat.
9. **SafePyramid — separate table, full split only.** `safepyramid.py` keeps metrics separate already. **Drop the current n=6 result entirely** (verified: `outputs/safepyramid_results.json` has n=2/level); rerun on the full test split with exact-set-match + rule-level micro P/R/F1 per L0/L1/L2 with bootstrap CIs.
10. **Latency honesty.** `ensembles.py` runs members **sequentially → latency = sum**; `ensemble-best-f1` measured at 3240 ms p50 is **not** faster than a mini. Only the **single SLM (~311 ms)** or the **deferral cascade** meets the ⅓-latency bar. The config carrying the F1 claim **must** carry the latency/cost claim. Report network-excluded, same-machine latency (decoders already CPU-pinned in `run_benchmarks.py`) plus API-network broken out separately; stamp device per row.

---

## 4. Ablations — the table that isolates what makes it work

One factor toggled per row; every delta attributable. All rows at matched-n with CIs and per-axis breakout.

| # | System | Isolates |
|--:|---|---|
| 1 | keyword-baseline | floor |
| 2 | distilbert-SFT, BeaverTails-only | in-domain encoder floor |
| 3 | modernbert-large SFT, full mixture | encoder capacity |
| 4 | (3) + temperature calibration | calibration effect |
| 5 | Qwen2.5-1.5B SFT, content-safety only | naive SLM |
| 6 | (5) + red-team data, **LOBO** | red-team axis lift + generalization |
| 7 | (6) + over-refusal data (OR-Bench) | FPR@benign effect |
| 8 | (7) + independent-teacher distillation | distillation delta (vs gold-only control 7) |
| 9 | Qwen2.5-1.5B SFT, **full mixture** | best single 1.5B (headline candidate) |
| 10 | (9) + DPO | preference effect on FPR |
| 11 | (9) + GRPO | **NEGATIVE RESULT / degeneracy** |
| 12 | hard union(encoder, 1.5B) | cheap composition |
| 13 | **stacking router** over member [pred,score], grouped-CV | learned-combiner delta vs oracle ceiling |
| 14 | **encoder-gate → 1.5B-expert deferral cascade** | the deployable system: F1, FPR, p50, defer-rate |
| — | Base-model sub-ablation: Qwen2.5-1.5B vs Qwen3-1.7B vs Qwen3-0.6B vs DeepSeek-R1-1.5B (degenerate) | substrate choice |
| — | Baseline rows: Llama-Guard-3-1B, ShieldGemma-2b, PromptGuard-2-86M, (WildGuard-7B if licensed), OpenAI-Moderation, GPT-4o-mini, GPT-5-mini | the field |

Columns: macro-F1, ROC-AUC (labeled swept vs single-point), FPR@benign, p50/p90, $/1k — all with 95% CIs.

**Row 13 is load-bearing for the thesis:** pair the oracle-ceiling *upper bound* (0.98) with the **achieved** stacker number, or the "gap is combiner/data not capacity" claim is unproven. Recompute the oracle **with and without** the degenerate always-unsafe RL/DPO members — they inflate it by being trivially right on unsafe samples.

---

## 5. Reproducibility package

**Pinned env.** Adopt `uv`, commit `uv.lock` (CPU/MPS + CUDA variants) — pin exact `torch/transformers/trl/peft/accelerate/datasets/tokenizers`. The unbounded ranges in `pyproject.toml [eval]/[train]` are the biggest hole (trl/peft/transformers shift training + generation across minors).

**Pinned revisions.** Add `revision` to `Benchmark` (`evaluation/benchmarks.py`) threaded through `data/loaders.py::_load_hf` (`loaders.py` warns field/label schemas drift). Add `revision` to `registry.py::BaseModel` passed to every `from_pretrained`. Replace floating OpenAI aliases in `openai_guards.py` (`omni-moderation-latest`, `gpt-4o-mini`) with **dated snapshots** (`gpt-4o-mini-2024-07-18`, `omni-moderation-2024-09-26`, dated `gpt-5-mini`) and record the served `.model` per call.

**Seeds + determinism.** seed=42 is already pinned (`balanced_subset`, `train_val_split`, `set_seed`) and decoding is greedy (`do_sample=False`). Add `PYTHONHASHSEED=0`, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG` in train/eval entrypoints; scope bit-repro to a stated reference device.

**Run manifest.** Extend the `_merge_persist` blob in `benchmark_results.json` with git SHA (`tracking/experiments.git_commit`), `hardware_info()`, package versions, per-dataset HF revision, OpenAI snapshot(s), seed, per_class, per-guard n, UTC timestamp — and a **validity assertion that flags mixed-n rows**.

**Released artifacts (currently gitignored via `/outputs/`).** Push SFT LoRA checkpoints to HF (pinned revisions); attach `outputs/predictions/*.json` (sample-key-aligned per-sample dumps — these make every ensemble/cascade/oracle number reproducible offline with zero GPU and zero API), the frozen matched-n `benchmark_results.json` + manifest, `curves.json`, `BENCHMARKS.md`, as a versioned release.

**Single command.** Add `make repro` (none exists): build cached balanced subsets at fixed per_class → `run_benchmarks.py` (all reachable guards, `--no-openai` offline fallback) → `compute_curves.py` → `eval_ensembles.py` → `run_safepyramid.py` → `stats.py` CIs → emit `BENCHMARKS.md` + manifest. Must re-emit byte-identical subset JSONL + identical local-guard metrics; document API-row tolerance.

**Compute disclosure + checklist.** Add `docs/reproducibility.md` + a NeurIPS/JMLR-style filled checklist appendix: datasets + revisions + licenses (ToxicChat non-commercial; gated WildGuard/HarmBench/AdvBench/StrongREJECT/PINT "not run"), wall-clock + `hardware_label()`, seeds, determinism flags, hyperparameters (`configs/model/*.yaml`), leakage protocol, and the exact reviewer command. **State the hard limitation explicitly: closed API rows are reproducible only to the recorded snapshot + released prediction dumps, not bitwise.**

---

## 6. Paper outline

**Title:** *Bouncers, Not Judges: A Leakage-Audited, Matched-n Frontier for Small Open Guard Models vs Closed Mini Judges.*

**Abstract (5 sentences).** (1) A safety guard runs on *every* request, so over-blocking, latency, and cost dominate raw accuracy, yet the field benchmarks against slow, costly, closed, drifting "mini" judges on non-matched splits. (2) We build a single leakage-audited harness — 7 ungated benchmarks over 3 axes plus a policy axis — that scores open SLMs, closed minis, and a moderation API at matched sample sizes with one ROC-AUC definition, paired-bootstrap CIs, and McNemar tests. (3) A ~1.5B open SLM (single SFT, and cheap ensembles/cascades) matches GPT-5-mini and dominates the moderation API on over-blocking, ranking, latency, and cost at ⅓ the latency and no per-request fee, while trailing GPT-4o-mini on macro-F1 by a bounded, reported margin; red-team/injection is the weakest axis (F1≈0.51), a data gap we close with targeted training. (4) A complementarity-ceiling diagnostic (oracle 0.98 vs best-single 0.75, low error correlation) plus an *achieved* stacking router localizes the residual gap to the combiner and data, not model capacity. (5) We release code, pinned env, cached subsets, per-sample prediction dumps, and checkpoints so the open half reproduces bit-for-bit and the closed half is pinned to dated snapshots.

**Sections.** 1 Intro (guard-on-every-request framing) · 2 Related work (open guards: Llama Guard 1/2/3, WildGuard, ShieldGemma, Aegis; rails: NeMo Guardrails; LLM-judge + Moderation: Zheng'23, Markov'23; injection/jailbreak: JailbreakBench, HarmBench, StrongREJECT, GCG, PromptGuard; over-refusal: XSTest, OR-Bench; diversity theory: Kuncheva & Whitaker'03; policy: SafePyramid) · 3 Harness & protocol (leakage, matched-n, alignment, AUC hygiene, stats) · 4 Systems (encoder, 1.5B SFT, ensembles, cascade, router; the continuous-score head; RL negative result) · 5 Results (frontier + per-axis + CIs) · 6 Diagnosis (ceiling + achieved combiner) · 7 SafePyramid · 8 Threats · 9 Reproducibility · 10 Conclusion.

**3 carrying figures/tables.**
- **FIG 1 (money):** Pareto scatter — x = latency (log ms) / $-per-1k, y = macro-F1 with a companion `fpr_on_benign` panel; every guard a point; small composed guards on the deployment frontier.
- **FIG 2 (thesis):** complementarity ceiling — per-model accuracy bars + best-single line + oracle line (with and without degenerate members) + **achieved stacker** marker; inset Yule's-Q heatmap.
- **TABLE 1 (honesty):** per-axis (guardrail/red-team/over-refusal/policy) SLM-vs-mini with 95% CIs, exposing content-safety+over-refusal parity and the red-team collapse.

**Venue.** NeurIPS Datasets & Benchmarks track or a trustworthy/safe-ML workshop (ICLR Building Trust, SoLaR); arXiv primary cs.CL, cross-list cs.CR + cs.LG. This is an eval+analysis+resource paper — pitching a methods main track invites "incremental over Llama Guard."

---

## 7. Threats to validity & mitigations (ranked by kill-probability)

1. **Invalid cross-group comparison (FATAL).** JSON has GPT at n=20–160, SLMs at 79–122, and per-guard leakage filtering removes *different* rows per guard → macro over non-identical subsets. → Matched-n re-run on the frozen intersected `sample_key` set; ship Table 1 receipt; `_align_rows` raises on mismatch; manifest flags mixed-n.
2. **Claim unsupported by our own numbers (FATAL).** Best SLM/ensemble 0.68–0.73 vs 4o-mini 0.833; `ensembles.md` says the F1 gap does not close. → Frontier/reproducibility framing as primary; F1-parity pre-registered as stretch only; never write unqualified "matches."
3. **No natural open-guard baseline (FATAL for positioning).** A ~1.5B guard's peer is Llama-Guard-3-1B / ShieldGemma-2B / WildGuard, not a closed mini + moderation API. → Run the already-wired `baselines.py` guards (Llama-Guard-3-1B, ShieldGemma-2b, PromptGuard-2-86M) through the same harness as the **primary** comparison; minis secondary.
4. **No uncertainty (SEVERE).** n~110, deltas of 0.03–0.05 inside noise; nothing computes CIs. → `stats.py` bootstrap CIs + Holm-corrected McNemar; "inconclusive" when CI straddles the margin.
5. **Distillation circularity (SEVERE).** Teacher = baseline is tautological. → Independent teacher + gold-only control; minis never used as teacher.
6. **Closed moving targets (SEVERE).** Aliases drift/retire. → Dated snapshots, served `.model` logged, raw responses frozen in `outputs/predictions/`, stated as a hard limitation.
7. **Test-set config selection (SEVERE).** `optimize_ensemble` searches ≤2500 configs, `optimize_cascade/deferral` sweep bands, all on the reported set. → Select on dev, report once on test, disclose search count, Holm-correct.
8. **Latency claim evaporates at F1 parity (HIGH).** Ensemble latency = sum (~3240 ms). → Attach latency/cost to the single-SLM/cascade config that carries the F1 claim; break out network vs compute; stamp device.
9. **Over-refusal leakage (HIGH).** `over_refusal_aware` trains on XSTest (the eval set). → Switch to OR-Bench; audit the augmentation never overlaps eval.
10. **Red-team weakest where guardrails matter (HIGH).** injection F1≈0.51. → Per-axis reporting + red-team training + LOBO; never let content-safety mask it in the macro.
11. **Binary-score AUC + degenerate RL rows (MODERATE).** → Continuous head; relabel single-point AUC; RL/DPO to negative-result ablation; recompute oracle without degenerate members.
12. **Label noise (MODERATE).** BeaverTails response-labels used as prompt-labels; ToxicChat/jailbreak noise. → Frame claims as *relative* (same harness, same inputs); report a small human re-annotation agreement estimate; flag in-domain vs OOD benchmarks.

---

## 8. Concrete task list mapped to THIS repo

Ordered; each points at the real module. Effort in eng-days (S≤1, M 2–4, L 5–10).

1. **[M] Add `Qwen2.5-1.5B-Instruct` to `models/registry.py`** as "the 1.5B guard" (mirror the `qwen3-1.7b` entry, decoder, techniques `("sft","dpo","grpo")`). Unblocks every 1.5B claim.
2. **[M] Continuous score head in `models/decoder.py`.** Emit first-token log-prob of safe/unsafe as `Verdict.score`; keep fail-closed decision but derive its score. Add a `test_decoder.py` case asserting non-binary scores. Unlocks AUC/calibration/deferral.
3. **[L] Build `evaluation/stats.py`** — `paired_bootstrap_ci` (B=10000, strata="benchmark") + `mcnemar` over `outputs/predictions/*.json` via `ensembles._align_rows`; wire Holm correction. New `tests/test_stats.py`.
4. **[S] Harden `ensembles._align_rows`** to raise on missing samples; add per-axis equal-weight macro alongside `macro_average`.
5. **[M] Fix over-refusal data path in `data/training_sets.py`** — replace XSTest augmentation with OR-Bench; add an assertion that training sets never intersect eval `sample_key`s.
6. **[M] Pin everything:** `revision` fields in `benchmarks.py`/`loaders.py`/`registry.py`; dated snapshots in `openai_guards.py` + log served `.model`; commit `uv.lock`; add determinism flags to entrypoints.
7. **[L] Matched-n re-run — the gate for all numbers.** `download_full_benchmarks.py` then `run_benchmarks.py --per-class 0` for **all** guards incl. GPT minis, moderation, and the `baselines.py` open guards (`make baselines`/`incumbents`); `dump_predictions.py` per guard; freeze `outputs/predictions/*.json`. Extend the manifest write in the runner.
8. **[M] SFT the 1.5B (+ red-team + over-refusal), ≥2 epochs, r≥16, ×3 seeds** via `scripts/train/run_training.py` + `configs/model/`; add a `red_team` LOBO config (train deepset+jailbreak-classification, hold out JailbreakBench).
9. **[M] Offline analysis on cached preds (zero GPU/API):** reproduce `diversity_report`; implement `ensembles.stack()` (logistic + GBT over member `[pred,score]`, grouped-CV by benchmark); build the encoder→1.5B deferral cascade via `evaluate_deferral` now that scores are continuous. Report achieved-vs-oracle.
10. **[M] Calibration + iso-FPR sweeps** — temperature/Platt on dev, threshold s.t. dev-FPR ≤ mini-FPR, report on disjoint grouped test.
11. **[S] Cost model** — $/1k from logged token counts × pinned list price, added next to p50/p90 in `report.py`/`BENCHMARKS.md`.
12. **[S] SafePyramid full split** via `scripts/eval/run_safepyramid.py`; delete the n=6 artifact; separate table with CIs.
13. **[M] RL negative-result ablation** — GPU GRPO/DPO from SFT ckpt with `RewardWeights(false_positive_penalty≈2–3)`; report FPR recovery or cut.
14. **[M] `make repro` target + `docs/reproducibility.md` + filled checklist**; release checkpoints + prediction dumps + frozen JSON.
15. **[L] Distillation experiment (optional, last)** — independent teacher (GPT-5.x-high or open-guard ensemble) soft-labeling an unlabeled pool via `scripts/data/build_dataset.py`; gold-only control; report separately.

**Critical path to a submittable result:** tasks 1→2→7→3→8→9 (fair matched-n numbers + CIs + the 1.5B + the achieved combiner) — everything else sharpens it.
