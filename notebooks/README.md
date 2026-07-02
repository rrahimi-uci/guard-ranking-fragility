# Notebooks

Run and configure the whole framework from a notebook — no CLI needed.

## [`agent_bouncer_studio.ipynb`](agent_bouncer_studio.ipynb)

One self-contained notebook that:

1. **Installs** the package (`.[eval,serve]` + matplotlib).
2. **Configures** the run via a single `CONFIG` cell — benchmarks, guards, test-set size.
3. *(optional)* **Fine-tunes** the encoder guard (SFT) and **RL-tunes** it (GRPO from SFT).
4. **Runs the standard benchmark suite** (7 ungated benchmarks) through the same harness.
5. **Renders** the results table + **plots** Precision / Recall / F1, over-blocking (FPR),
   latency, and **ROC / AUC** inline.
6. **Screens a live prompt** and can **launch the web dashboard**.

```bash
pip install -e '.[eval,serve,notebook]'
jupyter lab notebooks/agent_bouncer_studio.ipynb   # then: Run All
```

Works locally or on Colab (upload the repo, or `pip install` from Git). The default guard set
(`keyword` · `encoder` · GPT-4o-mini · GPT-5.2-low) is the fast, reliable path; the local Qwen
decoders are best scored via the CLI/dashboard, which isolate each in its own process.

> Notebooks stay thin — they call into `agent_bouncer`, they don't reimplement it.
