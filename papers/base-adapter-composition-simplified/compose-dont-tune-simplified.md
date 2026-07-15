# Compose, Don't Tune? — the plain-language version

*A retrospective study of averaging a base model with its fine-tuned safety guard.*
Reza Rahimi (JazzX AI). Plain-language companion to the formal Paper B draft.

> **Read this first.** This is a **retrospective feasibility** result, not a proven claim. The
> "held-out" transfer benchmarks were kept out of fine-tuning, but we looked at them while
> developing the method — so these numbers *estimate* a trade-off and *motivate* a future,
> properly-locked study. They do **not** prove the method wins in general. Every number below
> is the same one in the formal paper, generated from Paper A's clean-run v2 evidence.

---

## The idea in one paragraph

When you fine-tune a small model into a safety **guard** (a yes/no "is this prompt unsafe?"
classifier), it gets **better on the kinds of data it was trained on** and often **worse on
data it wasn't** — it over-specializes. Paper A measured that. This paper asks a cheap
question: instead of fine-tuning *more carefully*, what if we just **keep the original model
too** and **average its answer with the fine-tuned one**? No extra training — just run both and
combine their scores. We find this **gives back most of the lost transfer performance** at a
**small cost on the trained-on data** — but with real caveats.

---

> **Background: the words you need.**
> - **Guard / base / SFT adapter.** The *base* is the untuned instruction model. *SFT* =
>   supervised fine-tuning turns it into a guard (here with LoRA, a cheap adapter). *Base* means
>   "before guard fine-tuning" — not "untrained."
> - **Represented vs transfer.** *Represented* sources are benchmark families the guard was
>   trained on. *Transfer* sources were held out of training — the test of generalization.
> - **macro-AP.** Average Precision measures ranking quality (can the guard put unsafe prompts
>   above safe ones?), computed per benchmark then averaged. Higher = better; threshold-free.
> - **Calibration.** Turning a raw score into a probability using a held-out *calibration*
>   split, so two different models' scores are on the same scale before you average them.

---

## The tension Paper A found

Fine-tuning a guard is a trade. Across a fixed panel of **4 checkpoints** (Qwen2.5-1.5B,
SmolLM2-1.7B, SmolLM3-3B, Qwen3-4B), **5 fine-tuning seeds each**:

- On **represented** sources, SFT is excellent: macro-AP **0.982** (vs the base's 0.658).
- On **transfer** sources, SFT is *worse than the untuned base*: **0.807** vs the base's **0.866**.

So the fine-tuned guard sharpens where it trained and dulls where it didn't. The base, ironically,
ranks held-out data *better* — it just isn't as sharp on the trained-on data.

## The move: compose, don't tune (more)

For each prompt: run the **base** and one **SFT adapter**, calibrate each one's "unsafe"
probability on the calibration split, and **average the two** with fixed equal weights:

> composed score = ½ · calibrated(base) + ½ · calibrated(SFT)

No new training. Two model passes at inference.

## What we found

| Regime | Base | SFT | **Composed** |
|---|---:|---:|---:|
| Represented macro-AP | 0.658 | **0.982** | 0.962 |
| Transfer macro-AP | 0.866 | 0.807 | **0.883** |

Relative to **SFT**, the composed guard:
- gives up a little on represented data: **−0.019** (95% paired-bootstrap interval [−0.031, −0.010]);
- recovers transfer: **+0.076** ([+0.058, +0.093]).

That's the headline exchange: **a small represented-source loss buys back the transfer the
fine-tune had given away** — landing *above the base* on transfer (0.883 vs 0.866) while staying
close to SFT on represented (0.962 vs 0.982).

**But it is not a free lunch, and not uniform:**
- **vs the base, it's heterogeneous.** The panel-average composition-minus-base transfer gain is
  **+0.017** ([+0.005, +0.030]), but that hides the spread: **positive for 2 checkpoints**
  (SmolLM2, Qwen2.5), **≈ zero** for SmolLM3, and **negative for Qwen3-4B** (the strongest base,
  which the composition actually hurts on transfer). So this is *not* "beats both components
  everywhere."
- **Ranking ≠ calibration.** At a 5%-FPR target set on calibration data, the *realized* transfer
  false-positive rate is **8.1%** (base), **15.5%** (SFT), **11.4%** (composed). The composition
  improves recall over both but **still misses the 5% target** — better ranking does not mean a
  usable threshold transfers.
- **A tempting ablation we won't cherry-pick.** Averaging *logits* instead of calibrated
  probabilities reaches transfer **0.891** — higher than our fixed rule. But we saw the transfer
  numbers while developing this, so picking the best-looking combiner *after* the fact would be
  hidden tuning. We report it as an ablation only; the fixed calibrated-average stays primary.

## Is this real? (the honesty box)

- **Retrospective, not confirmatory.** We inspected the transfer benchmarks during development.
  These are *conditional, fixed-panel estimates* — they justify a future study, they don't settle it.
- **Maybe just "a second model helps."** We have **not** run the crucial control: fine-tune two
  independent SFT adapters and average *those* (an equal-cost SFT+SFT ensemble). If that gets the
  same lift, the benefit is generic variance reduction, not something special about keeping the base.
- **Not compared to the obvious alternatives yet.** No head-to-head against **WiSE-FT**
  (averaging *weights* instead of scores, which is one pass at inference) or against
  **matched-compute preservation tuning** (KL / replay). Those are required before any real claim.
- **Two passes cost real money** (latency, memory, energy) — the paper does not yet report those.

## What it means

In this retrospective panel, a dead-simple average of a base and its fine-tuned guard **trades a
little trained-on accuracy for a real recovery of held-out ranking** — enough to justify a proper,
pre-registered study, not enough to recommend deploying. The honest next step isn't a bolder
headline; it's a **separately-locked prospective study** on a genuinely unseen cohort with the
SFT+SFT control, real WiSE-FT rescoring, matched-compute preservation baselines, an expanded model
panel, and full systems-cost measurements. The formal draft ships that protocol as an executable,
fail-closed checklist.

---

*Numbers: Paper A clean-run v2, lock `cabc8dee…`, composition hash `92c2cbc3…`. See
[`../base-adapter-composition/`](../base-adapter-composition) for the formal draft and the fail-closed generator that produces every
value above.*
