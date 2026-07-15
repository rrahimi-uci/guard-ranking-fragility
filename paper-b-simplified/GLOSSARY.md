# Glossary — Paper B (Compose, Don't Tune?)

**Guard.** A small model turned into a binary classifier that decides whether a prompt is
`safe` or `unsafe`. Scored by the difference of two output-token logits (`z_unsafe − z_safe`).

**Base.** The instruction-tuned checkpoint *before* the guard fine-tuning. Not "untrained" and
not "unaligned" — just not yet specialized to the guard task.

**SFT (supervised fine-tuning).** Training the base on labeled `safe`/`unsafe` prompts. Here via
**LoRA**, a low-rank adapter that trains a small number of extra weights instead of all of them.

**Adapter / seed.** One LoRA-SFT result. We train **5 seeds** per checkpoint to measure how much
the result wobbles with the random seed.

**Represented sources.** Benchmark families that appear in the fine-tuning data. Doing well here
is "did it learn what we showed it."

**Transfer (dataset-held-out) sources.** Benchmark families kept out of fine-tuning. Doing well
here is "does it generalize." *Caveat in this study:* held out of *training*, but the researcher
saw them during method development — so results are retrospective, not blind.

**macro-AP (macro Average Precision).** Average Precision rewards ranking unsafe prompts above
safe ones (threshold-free). "macro" = compute it per benchmark, then average the benchmarks
equally, so a big benchmark can't dominate.

**Calibration.** Mapping raw scores → probabilities using a held-out calibration split, so two
models' scores share a scale and can be averaged fairly.

**Composition (the method).** Run the base and an SFT adapter, calibrate each, average the two
probabilities with fixed equal weights. Output-space (needs comparable *scores*, not shared
*weights*), at the cost of two inference passes.

**Composition − SFT / Composition − base.** The two contrasts reported. "− SFT" (vs the
fine-tuned guard) is the main question; "− base" (vs the untuned model) is a secondary anchor and
is heterogeneous across checkpoints.

**Operating point / FPR target.** A deployment threshold picked to hit a target false-positive
rate (5%) on calibration data. "Realized transfer FPR" is what you actually get on held-out data —
here it overshoots the target, showing ranking gains ≠ calibration gains.

**Paired hierarchical bootstrap.** The uncertainty method: resample row-families (shared across
both sides of a contrast) and adapter seeds to get a 95% interval, *conditional on this fixed
4-checkpoint panel* — not a population-level significance test.

**Retrospective vs prospective.** *Retrospective* = analyzed after the fact, on data touched
during development (what this draft is). *Prospective* = pre-registered, locked, run once on a
genuinely unseen cohort (what a confirmatory Paper B still needs).

**WiSE-FT / model soups.** Prior methods that average *weights* of a base and its fine-tuned model
(one inference pass). This paper averages *scores* instead; comparing the two head-to-head is
required future work.

**SFT+SFT control.** The key missing baseline: average two independent SFT adapters. If it matches
the base+SFT lift, the benefit is generic ensembling, not something unique to keeping the base.
