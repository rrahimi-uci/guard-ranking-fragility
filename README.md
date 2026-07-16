# The Benchmark Chooses the Winner

Auditable experiments, papers, and benchmark artifacts for understanding how compact
prompt-safety **guards** specialize, transfer, compose, and behave in high-compliance domains.

> A guard's score is not an intrinsic property of the model: it is *co-produced* by the
> benchmark, the training objective, the decision threshold, and the domain it is read on.

A "guard" is a small model that reads an incoming request and labels it `safe`/`unsafe` before an
assistant acts. The usual recipe — fine-tune a chat model into a guard and report a benchmark score —
hides the quantity a practitioner actually needs: *what did the fine-tune change relative to the same
model before tuning, and does that change survive on data the guard never saw?* This repository answers
that with a **paired, same-checkpoint** design on one fixed panel of four instruction checkpoints
(Qwen2.5-1.5B, SmolLM2-1.7B, SmolLM3-3B, Qwen3-4B), organized as a three-act program:

1. **Specialize** — measure what LoRA-SFT changes relative to each checkpoint (represented vs. transfer).
2. **Compose** — test whether averaging base + adapter recovers transfer without retraining.
3. **Domains** — a dual-labeled mortgage benchmark in depth, plus finance/health/law breadth (ExpGuard).

The **[unified research report](papers/unified-report/unified_report.pdf)** is the synthesis:
*The Benchmark Chooses the Winner: Measuring, Tuning, and Composing Small Safety Guards in
High-Compliance Business Domains.* Everything in it regenerates from committed per-row scores through one
entry point (below); all three acts have committed, reproducible results. See the
[status ledger](papers/unified-report/STATUS.md).

---

## Repository structure

```text
guard-ranking-fragility/
├── papers/                              # all manuscripts (LaTeX + built PDFs)
│   ├── unified-report/                  # ← the three-act synthesis (primary artifact)
│   │   ├── unified_report.tex           #   main document
│   │   ├── unified_report.pdf           #   built PDF (committed; refreshed by `make pdf`)
│   │   ├── sections/                    #   background + related-work + the three acts + limitations
│   │   ├── generated/                   #   auto-generated tables/macros (written by reproduce.py)
│   │   ├── figures/                     #   matplotlib figures + Graphviz .dot flowcharts + make_figures.py
│   │   ├── reproduce.py                 #   one-command "regenerate every result" harness
│   │   ├── refs.bib, STATUS.md, Makefile
│   ├── finetuning-specialization[-simplified]/   # formal Paper A  (+ plain-language edition)
│   ├── base-adapter-composition[-simplified]/    # formal Paper B  (+ plain-language edition)
│   └── mortgage-guardrail-benchmark[-simplified]/# formal mortgage paper (+ plain-language edition)
│
├── guard_research/                      # canonical library: metrics, thresholds, prompts, provenance
├── experiments/                         # Paper A pipeline; composition; ExpGuard scorer
├── mortgage-benchmark/                  # generator (magen/), frozen release, scorer, baselines, tests
├── artifacts/
│   ├── paper_a_sft_v2/                  # primary clean-v2 lock, release, text-free scores, analysis
│   ├── paper_a_sft/                     # immutable archived-v1 evidence
│   └── expguard_external/               # text-free finance/health/law per-row scores (Act III breadth)
├── configs/                            # Paper A config + v2 release anchor
├── docs/                               # reproducibility contracts, pre-registration, plans
├── tests/                              # canonical unit + artifact-contract tests
├── legacy/                             # quarantined earlier broad-study code
├── Makefile  pyproject.toml  requirements.txt  .python-version  .env.example
```

Raw third-party prompt text and large training artifacts stay local/gitignored. Committed release
artifacts keep pinned identifiers, source revisions, content hashes, and **text-free per-row scores**
(row hash → score) rather than redistributing prompts.

---

## Setup

Python **3.12** is supported (see [.python-version](.python-version)).

```bash
python3.12 -m venv .venv
source .venv/bin/activate

make install        # constrained CPU analysis + tests (no training stack)
# or
make install-all    # + the training/scoring (GPU) stack
```

Dependencies are pinned in [requirements.txt](requirements.txt). To **build PDFs** you also need
[Tectonic](https://tectonic-typesetting.github.io/); the two flowchart diagrams additionally use
[Graphviz](https://graphviz.org/) (`dot`) — if `dot` is absent, the committed PNGs are used as-is.
Gated datasets (ExpGuard) need a Hugging Face token; copy [.env.example](.env.example) to `.env`.

---

## Produce the results

**One command regenerates every table and figure in the unified report from committed per-row scores —
no GPU, no network:**

```bash
make -C papers/unified-report reproduce         # regenerate all generated/ tables + figures/
make -C papers/unified-report reproduce-check    # + assert byte-identity with the committed copies
```

`reproduce.py` dispatches to each study and re-derives the exact LaTeX the report `\input`s:

| Study | Source of truth | Notes |
|---|---|---|
| Act I — specialization | `artifacts/paper_a_sft_v2/scores/scores.parquet` | needs the lock-pinned analysis env |
| Act II — composition | `artifacts/paper_a_sft_v2/analysis/composition/` | from committed scores |
| Act III — mortgage | `mortgage-benchmark/out_eval/scores_*.json` | from committed scores |
| Act III — ExpGuard (finance/health/law) | `artifacts/expguard_external/scores_*.json` | `eval_expguard_external.py --from-scores` |
| Latency (P50/P90/P99) | `artifacts/paper_a_sft_v2/scores/scores.parquet` | per-row `latency_ms`, no GPU |

**Reproduce Paper A on its own (no GPU)** from the released v2 cache — the strict
[LOCK.json](artifacts/paper_a_sft_v2/LOCK.json), text-free
[scores.parquet](artifacts/paper_a_sft_v2/scores/scores.parquet), and
[release anchor](configs/paper_a_sft_v2_release_anchor.json):

```bash
make repro       # verify release evidence, then analyze + compare the checked-in inputs
make test        # unit + release-integrity tests
make selftest    # synthetic end-to-end analysis check
```

**Regenerate a Paper A run from scratch (GPU + network)** — never overwrite the released namespace:

```bash
export V2_ROOT=artifacts/paper_a_sft_v2_rerun
make manifests   # 1. pinned, hash-ranked manifests (needs HF access)
make audit       # 2. recompute split-integrity checks (fail-closed)
make lock        # 3. create the strict v2 lock
make train       # 4. GPU: train the 4×5 LoRA-SFT panel
make validate-runs
make eval        # 5. GPU: score bases + adapters
make analyze     # 6. emit tables/figures
```

---

## Build the papers

All papers compile with Tectonic via a per-directory Makefile:

```bash
# The unified report (recommended: refresh results first, then compile)
make -C papers/unified-report all      # = reproduce + pdf
make -C papers/unified-report pdf      # compile only (also copies the PDF to unified_report.pdf)

# The three formal papers
make -C papers/finetuning-specialization pdf
make -C papers/base-adapter-composition pdf
make -C papers/mortgage-guardrail-benchmark pdf
```

| Paper | Formal edition | Plain-language edition |
|---|---|---|
| **Unified three-act report** | [PDF](papers/unified-report/unified_report.pdf) · [LaTeX](papers/unified-report/unified_report.tex) | teaching boxes integrated into the report |
| Fine-tuning specialization (A) | [PDF](papers/finetuning-specialization/benchmark_chooses_the_winner.pdf) | [annotated](papers/finetuning-specialization-simplified/) |
| Base+adapter composition (B) | [PDF](papers/base-adapter-composition/compose_dont_tune.pdf) | [simplified](papers/base-adapter-composition-simplified/) |
| Mortgage guardrail benchmark | [PDF](papers/mortgage-guardrail-benchmark/mortgage_guardrail_benchmark.pdf) | [simplified](papers/mortgage-guardrail-benchmark-simplified/) |

Claim-bearing numbers enter LaTeX only through generated macros/tables (`generated/`), never hand-typed;
`reproduce-check` guards against drift. The report also ships two Graphviz flowcharts of the study's
processes — the [data-split construction](papers/unified-report/figures/data_splits.dot) and the
[paired experimental design](papers/unified-report/figures/experiment_design.dot).

---

## Status

| Track | Main artifact | Honest status |
|---|---|---|
| **Act I — specialization** | [Paper A](papers/finetuning-specialization/benchmark_chooses_the_winner.pdf) | **Complete** clean-v2 retrospective estimate (4 checkpoints × 5 seeds); conditional on this fixed panel, not universal or confirmatory. |
| **Act II — composition** | [Paper B](papers/base-adapter-composition/compose_dont_tune.pdf) | **Retrospective pilot complete.** No separately locked prospective run; controls remain roadmap items. |
| **Act III — mortgage depth** | [frozen benchmark](mortgage-benchmark/benchmark/v1_hmda2022/) | **994-row synthetic benchmark + four-base baselines complete.** LLM-judge / policy-card labels, *not* SME-adjudicated. |
| **Act III — ExpGuard breadth** | [scores](artifacts/expguard_external/) + [evaluator](experiments/eval_expguard_external.py) | **Complete** four-checkpoint base eval on 2,275 finance/health/law prompts; text-free scores committed; tuned comparison is future work. |

Acts I and II are reproducible but **retrospective** (their sources were inspected during development).
The report keeps retrospective, external-expert, and LLM-judge evidence in separate tiers and never pools
them, and makes no causal, universal, deployment, legal, or fair-lending claim.

---

## Headline results (all reproducible from committed scores)

**Act I — LoRA-SFT specializes.** SFT lifts every checkpoint to ≈0.98 represented-source macro-AP
(**+0.323** on average) but changes held-out **transfer** by only **−0.059** on average — hiding opposite
per-checkpoint signs (SmolLM2 **+0.040** … Qwen3-4B **−0.150**). This is an *attractor*: post-SFT scores
collapse to a benchmark-fixed endpoint (transfer 0.807 ± 0.024), so "stronger bases specialize more" is
arithmetic (Δ slopes −1 in the base). At a 5% calibration-FPR target, transfer false alarms rise
(pooled **4.3% → 17.0%**) and HarmBench recall falls (**78% → 60%**).

**Act II — composition recovers transfer.** Averaging the base's and SFT guard's calibrated scores lifts
transfer over SFT for all four checkpoints (**+0.076**) as an ensemble diversity gain — recovery, not
dominance (it can dip below the untuned base), and it restores no transferable threshold.

**Act III — domains.** The frozen [v1_hmda2022](mortgage-benchmark/benchmark/v1_hmda2022/) mortgage
benchmark (994 dual-labeled `G×D` rows; the load-bearing **G0/D1** stratum + a protected-context fairness
gate) shows zero-shot mortgage-policy AP of **0.67–0.85** and a protected-pair gap of **0.000–0.183**. On
external ExpGuard (2,275 expert-annotated prompts), all four base guards rank violations well zero-shot
(AP **0.88–0.96**) — and the best is **SmolLM3-3B (0.956), not the largest model**, a different winner
than the mortgage benchmark picks. The recurring character is Qwen3-4B: strongest base, specializes most,
helped least by composition, yet the best/fairest zero-shot mortgage guard — *the ranking flips with the
benchmark.*

---

## Auditable evidence chain

- **Locks & releases** — Paper A v2 binds config, manifests, audit, prompt rendering, source state, score
  identity, and release anchor; the released score table (79,392 rows, SHA-256 `b941ddba…`) is bound to
  lock `cabc8dee…`.
- **Fail-closed split audit** — 24 hard assertions on overlap, label conflicts, balance, upstream-family
  disjointness, revisions, and near-duplicate dispositions.
- **Text-free scores** — row identities + content hashes + scores, never third-party prompt text.
- **Canonical metrics** — [guard_research/metrics.py](guard_research/metrics.py): sklearn-backed,
  tie-aware, permutation-invariant average precision. (Ranking metrics use the raw decision margin, not a
  saturating probability, so `--from-scores` reproduces exactly.)

## Known boundaries

- Four compact checkpoints from two lineages are a fixed panel, not a model population.
- Acts I/II use dataset-held-out but previously-inspected sources; no confirmatory claim.
- The mortgage benchmark is synthetic and policy-card-consistent, not SME-adjudicated; its G1/D0 quadrant
  is empty. Ranking recovery does not imply threshold/calibration transfer.
- The public Paper A cache cannot independently rehash omitted adapter bytes / full run metadata.

## Citation & license

[CITATION.cff](CITATION.cff) currently cites Paper A. Repository code and original content are
[Apache 2.0](LICENSE); third-party datasets/models retain their own licenses. Review the mortgage
[DATA_CARD.md](mortgage-benchmark/benchmark/v1_hmda2022/DATA_CARD.md) before redistributing generated
prompts.
