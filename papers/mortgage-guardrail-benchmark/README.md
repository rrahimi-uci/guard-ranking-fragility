# Paper — Mortgage-Specific Safety-Guardrail Benchmark

Formal draft of the mortgage guardrail benchmark paper (LaTeX, same 11pt-article toolchain as
[`../finetuning-specialization`](../finetuning-specialization) and [`../base-adapter-composition`](../base-adapter-composition)).

- [mortgage_guardrail_benchmark.tex](mortgage_guardrail_benchmark.tex) — the paper source;
  [mortgage_guardrail_benchmark.pdf](mortgage_guardrail_benchmark.pdf) — the compiled draft.
- [refs.bib](refs.bib) — bibliography; [figures/pipeline.dot](figures/pipeline.dot) →
  `figures/pipeline.png` — the construction-pipeline diagram (Figure 1);
  [generated/baseline_table.tex](generated/baseline_table.tex) — the `\input`-ed baseline table.
- Build: `make` (renders the figure via Graphviz, regenerates the baseline table from committed
  scores, then compiles with tectonic → `build/` and copies the PDF up).
- Plain-language edition: [`../mortgage-guardrail-benchmark-simplified/`](../mortgage-guardrail-benchmark-simplified).

## What it is / isn't
A **benchmark-construction + baseline-evaluation** paper for a mortgage-specific guardrail, built
on the *fixed* HMDA-2022-grounded benchmark in
[`../../mortgage-benchmark/benchmark/v1_hmda2022/`](../../mortgage-benchmark/benchmark/v1_hmda2022).
It is **not** a fair-lending findings paper: labels are LLM-judge / policy-card-consistent, not
SME-adjudicated.

## Reproducibility model
- **Generation is intentionally frozen** (LLM at temperature > 0) — the released dataset is the
  citable artifact.
- **Evaluation reproduces**: `mortgage-benchmark/magen/score_guards.py` scores
  any guard over the frozen benchmark → macro-AP (G/D/final), per-quadrant, Δ_context, via the
  canonical `guard_research` metrics. Baseline guard specs:
  `mortgage-benchmark/configs/baseline_guards.json`.

## Baseline results
Reproduce with:
```
python mortgage-benchmark/magen/score_guards.py \
  --guards mortgage-benchmark/configs/baseline_guards.json \
  --benchmark mortgage-benchmark/benchmark/v1_hmda2022 \
  --eval-split public_test --out out_eval
```
The paper's Table (§6) is populated from the committed `out_eval/baseline_table.json`.
