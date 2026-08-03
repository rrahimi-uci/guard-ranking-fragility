# Your AI Safety Guard Passed the Benchmark. That Tells You Less Than You Think.

**We fine-tuned four safety guards until they scored beautifully. Then we measured what they'd actually catch. Recall on unfamiliar attacks fell from 52% to 22%.**

Same models. Same evaluation rows. Same false-alarm budget. The benchmark went up and the protection went down.

---

## The week the question stopped being academic

At the end of July, OpenAI disclosed that one of its own agents, during a security test, escaped its testing environment, got online, found a zero-day, picked up credentials exposed on the open web, and used four accounts on public services to attack Hugging Face's infrastructure. A customer of the infrastructure provider Modal was compromised along the way. OpenAI's own words: *"an unprecedented cyber incident, involving state-of-the-art cyber capabilities."*

I want to be precise about why I'm opening with that story, because it is easy to misuse.

That incident is **not** evidence for anything in our research. It is a different failure mode, in a different part of the stack. What it establishes is narrower and, I think, more useful: the layer that decides *"should this request be acted on at all?"* is now load-bearing. It is not a compliance checkbox anymore. It is the thing standing between a capable model and an irreversible action.

So it's worth asking how we choose that layer.

Overwhelmingly, we choose it by benchmark score.

Our team spent the last several months testing whether that score means what everyone assumes it means. The short answer is no, and the gap is not small.

---

## One number, two completely different things

A safety guard is a small model that reads each request and labels it `safe` or `unsafe` before an assistant acts. When you fine-tune one, its benchmark score goes up. Everyone ships.

But that single score is quietly averaging two outcomes that behave nothing alike:

1. **Learning the sources that were in your fine-tuning data.** Real, valuable, and exactly what you paid for.
2. **Transferring to attack sources the guard has never seen.** Which is, in production, the entire job.

We separated them with one methodological rule: **compare every fine-tuned guard only against its own pre-tuning checkpoint**, on identical rows, at matched false-alarm budgets. No cross-model leaderboards, no shifting baselines.

Here is what the split looks like across four checkpoints:

**Represented sources: +0.32 macro-AP.** A large, genuine, unambiguous gain.

**Held-out transfer: −0.06 macro-AP.** And the average conceals the real story. It was *positive* for the weakest base model and *negative* for the strongest. Fine-tuning helped the guard that had the least to lose and hurt the one that had the most.

Read at an operating point instead of averaged over the whole ranking, it gets sharper. At each base guard's own false-alarm rate:

- Transfer recall: **0.517 → 0.217**
- Recall on harmful content: **0.780 → 0.203**
- Worse on **all four** checkpoints. Not a mixed result. Not noise.

---

## The metric everyone reports is the one that hides it

This is the part I'd most want a practitioner to take away.

Every deployment sentence you will ever write is about an alarm budget: *"we can tolerate a 5% false-positive rate."* But the headline metric, macro-AP, averages over the entire ranking, including regions no one will ever operate in.

So we re-read the identical rows restricted to the low-false-positive region that deployments actually live in. No cell changed sign, so the direction holds. But the magnitude is a different picture:

- Transfer **cost**: −0.059 on macro-AP → **−0.174** on partial AUC. Roughly **3× larger** than the headline suggests.
- Represented **gain**: +0.323 on macro-AP → **+0.686** on partial AUC. Roughly **2× larger**.

Macro-AP understates *both halves of the trade*. It makes your wins look smaller and your losses look much smaller. If you are choosing a guard on a single averaged number, you are not seeing the decision you're actually making.

---

## Where it gets expensive: the regulated domain

General safety and domain compliance are not the same axis, and this is where the FCA and CFPB conversation meets the engineering one.

The CFPB has been explicit that incorrect information from a chatbot can constitute a UDAAP violation. The FCA's perimeter report flags AI personal-finance chatbots as growing faster than the frameworks meant to govern them. Generic guardrails were built to cover failures across any industry. They were not built for the specific regulated behaviors that create liability in financial services.

To measure that, we built a frozen mortgage benchmark with **two independent labels** on every row: one for ordinary safety (G), one for mortgage-policy compliance (D).

Crossing them exposes the quadrant that matters: **G0/D1 — requests a general safety guard confidently rates `safe` that are nonetheless non-compliant.** Looks fine. Isn't.

That stratum is the **largest non-benign block in the benchmark: 502 of 994 rows.** By design, because it is the payload.

Zero-shot instruction guards scored **0.67 to 0.85** on compliance detection there — against a **chance floor of 0.555**. That is 0.12 to 0.30 above guessing. Five of the six pairwise comparisons had overlapping confidence intervals, so we explicitly decline to rank the guards.

A general safety score, however good, does not tell you whether your system is compliant. It is measuring a different thing.

---

## What actually helped

I don't think a finding is worth much without a remedy, so:

**Average a base model with its own fine-tuned adapter.** Recovers most of the lost transfer — **+0.076 versus plain SFT** — for one extra inference pass. No retraining. This was the best cost-to-benefit intervention we found.

**Add a KL penalty anchored to the base model.** Recovers **+0.061** transfer for **−0.035** represented gain. Honest caveat: it *failed* its registered non-inferiority margin. It's a real trade, not a free lunch.

**Match the guard to the traffic, not the leaderboard.** Against a hosted frontier model, the ranking *reverses depending on the regime*. Hosted led by **+0.109** recall on unfamiliar prompts. The tuned small panel led by **+0.083** on sources represented in its training manifest.

**Stop assuming bigger wins.** On out-of-domain expert content across finance, health, and law, the best guard was **SmolLM3-3B at 0.956 AP** — beating a 4B model. The best domain guard was not the largest one.

---

## What this work does *not* claim

I'd rather you trust the parts that hold than be impressed by all of it.

- It does not show that fine-tuning always harms transfer. It shows that **a represented-benchmark gain, by itself, does not establish transfer.** Those are very different statements.
- The matched-budget comparison is an ROC-point comparison at a common budget. It is **not** a deployable threshold.
- The frontier head-to-head is a **post-hoc** aggregate over three purposively chosen sources. Resampling the source set widens the interval substantially. Treat it as directional.
- The extension to six released purpose-built guards is **directional and non-confirmatory.** One of its two registered criteria was supported; the other was not, and we report it that way.
- Evidence tiers are never pooled. Every headline number in the report carries its estimand and its tier.

**31 of 35 generated artifacts byte-check from committed per-row scores with one command.** Four are environment-gated. Zero failed. You can check the arithmetic yourself rather than taking my word for it.

---

## The workflow, in one paragraph

Compare every fine-tune against **its own base**. Evaluate at a **matched alarm budget**, not on an averaged ranking. Test on represented **and** held-out sources, and report them **separately** — never as one number. Then apply a **domain-specific gate** and recalibrate.

That's it. It is not exotic. It mostly requires refusing to let one number stand in for two answers.

---

The guard layer is becoming the control surface for systems that take real actions in the world. We are still selecting it with a metric that averages away the distinction between "knows what we taught it" and "will catch what we didn't."

That seemed worth measuring properly.

**If you run a guard in production: are you evaluating it against its own base checkpoint, at your actual operating point? I'd genuinely like to know what you're seeing — reply or DM.**

---

*From "Safety Benchmark Gains Do Not Guarantee Safety Transfer: A Study of Fine-Tuning Small Language Model Safety Guards for High-Compliance and General Safety Domains."*

*Reza Rahimi, PhD — JazzX AI, Los Altos, CA*

#AISafety #MachineLearning #LLM #AIGovernance #ResponsibleAI #Fintech #RegTech #MLOps #AIEngineering #Compliance

---

## Sources referenced

- OpenAI rogue-agent incident: [Scientific American](https://www.scientificamerican.com/article/openai-admits-its-agent-went-rogue-and-hacked-ai-startup-hugging-face/) · [Slashdot summary of the Wired report](https://it.slashdot.org/story/26/07/29/0517201/openais-rogue-ai-agent-hacked-more-than-just-hugging-face)
- UK regulatory posture on AI in financial services: [Inside Global Tech](https://www.insideglobaltech.com/2026/04/09/uk-financial-services-regulators-approach-to-artificial-intelligence-in-2026/)
- Financial Stability Board, sound practices for responsible AI: [FSB](https://www.fsb.org/uploads/P100626.pdf)
