# SafePyramid — in-context policy guardrailing

[SafePyramid](https://huggingface.co/datasets/ByteDance/SafePyramid) (ByteDance, arXiv 2606.29887) tests
a *different* guardrail skill from the rest of the suite. Instead of a single prompt labeled
safe/unsafe, each item gives a multi-turn **conversation** plus an application-specific **policy** — a
numbered list of natural-language rules — and the guard must return the **exact set of rules the
assistant violated**. It's the enterprise-guardrail question: *"does this conversation comply with
*my* policy?"*, not *"is this text generically unsafe?"*.

- **3,000 items** · 10 domains (defamation, privacy, fraud, IP, medical/legal advice, …)
- **Three difficulty levels:** `L0` single-rule understanding · `L1` reasoning over rule dependencies
  · `L2` the hardest interactions
- ~6 violated rules and ~21 candidate rules per item

## How it's scored

Because this is a multi-label task, the leaderboard's P/R/F1 don't apply directly. SafePyramid is
scored on its own axes ([`agent_bouncer.evaluation.safepyramid`](../src/agent_bouncer/evaluation/safepyramid.py)):

- **exact-set-match** — the fraction of items where the guard names the *exact* violated-rule set (the
  paper's headline metric),
- **rule-level micro precision / recall / F1** — partial credit over individual rule IDs,

both **overall and per level (L0 / L1 / L2)**.

## Running it

It's a *policy-conditioned* task — the policy goes into the judge's context — so it evaluates a
policy-configurable LLM judge (the fixed local guards aren't policy-aware):

```bash
make safepyramid model=gpt-4o-mini limit=60          # cap items per level for cost control
python scripts/eval/run_safepyramid.py --model gpt-5.2 --reasoning-effort low
```

Results are written to `outputs/safepyramid_results.json` (one block per judge, with the per-level
breakdown). The dataset is downloaded and cached to `data/benchmarks/safepyramid.jsonl` on first run.

## What to expect

Policy guardrailing is **hard** — the paper reports even GPT-5.5 exactly identifies the full violated
set in only ~54% / 35% / 13% of L0 / L1 / L2 cases, and exact-match collapses as rule dependencies
deepen. Treat SafePyramid as the *policy-reasoning* frontier: a model can ace the generic safe/unsafe
benchmarks and still fail here, which is exactly why enterprise deployments need it as a separate axis.
