# The Benchmark Chooses the Winner

Auditable experiments, papers, and benchmark artifacts for understanding how compact
prompt-safety guards specialize, transfer, compose, and behave in high-compliance domains.

> A guard's score is not an intrinsic property of the model: it is co-produced by the
> benchmark, training objective, decision rule, and domain on which the guard is read.

The repository began as the focused Paper A study of fine-tuning specialization. It now contains
a four-act research program on one fixed panel of four instruction checkpoints:

1. **Specialize:** measure what LoRA-SFT changes relative to each checkpoint before guard SFT.
2. **Objectives:** pre-register an SFT/DPO/GRPO comparison; the GPU run is still pending.
3. **Compose:** test whether keeping the base in a fixed output-space average recovers transfer.
4. **Domains:** evaluate a dual-labeled mortgage benchmark in depth and ExpGuard's
   finance/health/law subsets for external breadth.

The [unified research report](papers/unified-report/unified_report.pdf) is the working synthesis:
*[The Benchmark Chooses the Winner: Honestly Measuring, Tuning, and Composing Small Safety
Guards Across Objectives and Four High-Compliance Domains](papers/unified-report/unified_report.tex)*.
It is **not release-ready**: Paper C has no result and the full ExpGuard panel is incomplete.
See the working [unified-report status ledger](papers/unified-report/STATUS.md), which may lag an
active evaluation run.

## Current research status

| Track | Main artifact | Honest status |
|---|---|---|
| **Act I — Fine-tuning specialization** | [Formal Paper A](papers/finetuning-specialization/benchmark_chooses_the_winner.pdf) | **Complete clean-v2 retrospective estimate.** Lock-bound 4-checkpoint × 5-seed execution; not prospective or universal. |
| **Act II — Objective axis** | [Pre-registration](docs/paper-c-prereg.md), [preference recipe](experiments/paper_c_preference.py), [runner scaffold](experiments/run_paper_c_objective.py) | **Pre-registration and training scaffold only; no result or Paper C PDF.** GPU smoke validation, a Paper C lock, training, scoring, and an analyzer remain pending. |
| **Act III — Base+adapter composition** | [Formal Paper B](papers/base-adapter-composition/compose_dont_tune.pdf) | **Retrospective pilot complete.** No separately locked prospective Paper B run exists; required controls and systems results remain pending. |
| **Act IV — Mortgage depth** | [Mortgage paper](papers/mortgage-guardrail-benchmark/mortgage_guardrail_benchmark.pdf), [frozen benchmark](mortgage-benchmark/benchmark/v1_hmda2022/) | **994-row synthetic benchmark and four-base diagnostic baselines complete.** Labels are LLM-judge / policy-card-consistent, not SME-adjudicated legal findings. |
| **Act IV — ExpGuard breadth** | [Evaluator](experiments/eval_expguard_external.py) | **In progress and unreleased.** Only a partial local base-checkpoint evaluation exists; no complete four-checkpoint ExpGuard claim is made. |
| **Unified report** | [PDF](papers/unified-report/unified_report.pdf), [source](papers/unified-report/unified_report.tex) | **Working draft.** Acts I, III, and mortgage are populated; Paper C and full ExpGuard remain pending, and its reproduction harness is not yet a green release gate. |

Acts I and III are reproducible but retrospective: their benchmark sources were inspected during
development. The report keeps retrospective, external-expert, and LLM-judge evidence separate and
does not make causal, universal, deployment, legal, or fair-lending claims.

## Read the papers

| Track | Formal edition | Plain-language edition |
|---|---|---|
| Unified four-act working report | [PDF](papers/unified-report/unified_report.pdf) · [LaTeX](papers/unified-report/unified_report.tex) | Teaching boxes and practitioner guidance are integrated into the report |
| Fine-tuning specialization (Paper A) | [PDF](papers/finetuning-specialization/benchmark_chooses_the_winner.pdf) · [source](papers/finetuning-specialization/benchmark_chooses_the_winner.tex) | [Annotated PDF](papers/finetuning-specialization-simplified/the-benchmark-chooses-the-winner-annotated.pdf) |
| Base+adapter composition (Paper B) | [PDF](papers/base-adapter-composition/compose_dont_tune.pdf) · [source](papers/base-adapter-composition/compose_dont_tune.tex) | [Simplified PDF](papers/base-adapter-composition-simplified/compose-dont-tune-simplified.pdf) |
| Mortgage guardrail benchmark | [PDF](papers/mortgage-guardrail-benchmark/mortgage_guardrail_benchmark.pdf) · [source](papers/mortgage-guardrail-benchmark/mortgage_guardrail_benchmark.tex) | [Simplified PDF](papers/mortgage-guardrail-benchmark-simplified/mortgage-benchmark-simplified.pdf) |

## Verified results currently in the repository

### Act I — LoRA-SFT specializes on this fixed panel

Across Qwen2.5-1.5B, SmolLM2-1.7B, SmolLM3-3B, and Qwen3-4B (seeds 42–46), SFT on
the locked 1,200-row clean-v2 training manifest produces:

| Effect | Observed Δ macro-AP (base → SFT) | Descriptive 95% paired-bootstrap interval |
|---|---:|---:|
| **Represented sources** | **+0.323** | **[+0.265, +0.369]** |
| **Dataset-held-out transfer** | **−0.059** | **[−0.084, −0.032]** |

SFT raises every checkpoint to about 0.98 represented-source macro-AP, but transfer is
heterogeneous: SmolLM2 **+0.040**, Qwen2.5 **−0.039**, SmolLM3 **−0.087**, and
Qwen3-4B **−0.150**. Every leave-one-checkpoint-out and leave-one-transfer-benchmark-out
aggregate remains negative. At thresholds selected for a 5% calibration FPR target,
benchmark-macro transfer FPR rises from **8.1% to 15.5%** (pooled-negative:
**4.3% to 17.0%**), while HarmBench recall falls from **78.0% to 60.0%**.
Fifteen of twenty seed-level points improve represented-source AP while losing transfer AP.

These are clean-v2, conditional fixed-panel estimates—not a universal “fine-tuning hurts”
conclusion and not a confirmatory finding.

### Act III — Composition recovers transfer relative to SFT

The fixed primary operator averages the base and SFT guard's separately calibrated unsafe
probabilities:

| Guard | Represented macro-AP | Transfer macro-AP |
|---|---:|---:|
| Unadapted instruction checkpoint | 0.658 | 0.866 |
| SFT adapter | **0.982** | 0.807 |
| Base+SFT calibrated average | 0.962 | **0.883** |

Relative to SFT, the observed composition contrast is **−0.019**
([−0.031, −0.010]) on represented sources and **+0.076**
([+0.058, +0.093]) on transfer. The comparison with the base is heterogeneous:
composition is above base for two checkpoints, near zero for one, and below base for Qwen3-4B.
At the same 5% calibration target, composition's realized transfer macro-FPR is **11.4%**.
This is transfer recovery relative to SFT—not Pareto dominance, calibration transfer, or proof that
keeping the base is uniquely better than generic two-pass ensembling.

### Act IV — Mortgage depth; ExpGuard breadth is incomplete

The frozen [v1_hmda2022](mortgage-benchmark/benchmark/v1_hmda2022/) mortgage release contains
**994 synthetic, HMDA-grounded request-screening rows** with two labels:
general safety (G) and mortgage-policy consistency (D). It includes a large G0/D1
stratum—requests that read as generally safe but violate a benchmark policy card—and a
protected-context invariance diagnostic. Four zero-shot base checkpoints obtain mortgage-policy
AP between **0.672 and 0.851** on the committed public-test scores; the observed protected-context
gap ranges from approximately **0.000 to 0.183**.

This benchmark is a measuring instrument, not a legal finding. Its prompts are synthetic, its
policy labels are produced by an LLM judge against written cards, it has no SME adjudication or
reported human agreement, and its G1/D0 quadrant is empty.

The external ExpGuard path targets **2,275 expert-annotated prompts** across finance,
healthcare, and law. The scorer and text-free artifact format are implemented, but the current
four-checkpoint base evaluation is incomplete and local artifacts are not released. The unified
report therefore keeps the full ExpGuard result pending.

## Repository layout

    papers/
      unified-report/                     in-progress four-act synthesis
      finetuning-specialization/          formal Paper A
      finetuning-specialization-simplified/
      base-adapter-composition/           formal Paper B + protocol validator
      base-adapter-composition-simplified/
      mortgage-guardrail-benchmark/       formal benchmark-construction paper
      mortgage-guardrail-benchmark-simplified/

    guard_research/                        canonical metrics, thresholds, prompts, provenance
    experiments/                           Paper A pipeline; composition; Paper C/ExpGuard scaffolds
    configs/                               Paper A config and v2 release anchor
    artifacts/paper_a_sft_v2/              primary clean-v2 lock, release, scores, analysis, provenance
    artifacts/paper_a_sft/                 immutable archived-v1 evidence
    mortgage-benchmark/                    generator, frozen release, scorer, baselines, tests
    docs/                                  reproducibility contracts, preregistration, planning notes
    tests/                                 canonical unit and artifact-contract tests
    legacy/                                quarantined earlier broad-study code

Raw third-party text and large training artifacts remain local or gitignored. Public release
artifacts retain identifiers, hashes, source revisions, and text-free per-row scores where
redistribution permits.

## Install

Python **3.12** is the supported environment (see [.python-version](.python-version)).

    python3.12 -m venv .venv
    source .venv/bin/activate

    make install       # constrained CPU analysis + tests
    # or:
    make install-all   # training/scoring stack + analysis + tests

The direct dependency stack is pinned in [requirements.txt](requirements.txt).
Paper A's executed environments and archived-v1 limitations are documented in
[docs/reproducibility-environments.md](docs/reproducibility-environments.md).
Tectonic is required only to compile LaTeX; Graphviz is additionally required to regenerate
the mortgage pipeline figure.

## Quick verification

    make test
    make paper-verify

These root targets cover the canonical library and Paper A only. They do not verify Paper B,
the mortgage workspace, Paper C, ExpGuard, or the unified report.

## Build the unified working report

    make -C papers/unified-report pdf

The current reproduce script is a **development harness**, not a complete release verifier:

    make -C papers/unified-report reproduce

It dispatches to the Paper A, Paper B, mortgage, ExpGuard, and Paper C analysis paths, but
unfinished studies remain pending, and its check path still needs stricter fail-closed and
non-mutating guarantees before it can certify a release. Do not infer Paper C or complete ExpGuard
results from a successful PDF build or a partial reproduction run.

## Reproduce Paper A from the released v2 cache (no GPU)

The primary no-GPU path consumes the strict final v2
[LOCK.json](artifacts/paper_a_sft_v2/LOCK.json), text-free
[public manifests](artifacts/paper_a_sft_v2/public_manifests/), bound
[scores.parquet](artifacts/paper_a_sft_v2/scores/scores.parquet) with sibling metadata,
the self-hashed [RELEASE.json](artifacts/paper_a_sft_v2/RELEASE.json), and the tracked
[release anchor](configs/paper_a_sft_v2_release_anchor.json).

    make repro       # verify release evidence, analyze, and compare checked-in Paper A inputs
    make paper       # re-verify the inputs, then compile Paper A only
    make test        # root unit and release-integrity tests
    make selftest    # synthetic end-to-end analysis check

The released score table contains **79,392 rows** and is bound by score SHA-256
b941ddbaea7057ab1f224c510687ec5748916f5eca6a78e1d1f429e0ede5a1c3
to Paper A lock
cabc8dee9b158773ce0be86f799ec3833c33c18787a2aa74d05ed1a261682c25.

The public score-only cache binds recorded adapter and run identities but does not include adapter
bytes, raw licensed prompts, complete run directories, or the full cloud archive. Independent
verification of those omitted bytes requires the separately verified execution archive/source
snapshot. See [docs/reproducibility.md](docs/reproducibility.md) for the exact guarantee.

The archived v1 compatibility path is explicit and cannot update publication files:

    make repro-legacy

## Generate a new Paper A clean-v2 run (GPU)

Do not overwrite the released artifacts/paper_a_sft_v2 namespace. Choose a new artifact root:

    export V2_ROOT=artifacts/paper_a_sft_v2_rerun

    make manifests      # 1. source/network access; pinned, hash-ranked manifests
    make audit          # 2. independently recompute split-integrity checks
    make lock           # 3. create the new strict v2 lock
    make train          # 4. GPU: train the 4 × 5 LoRA-SFT panel
    make validate-runs  #    rehash and validate all 20 adapters
    make eval           # 5. GPU: score bases and adapters
    make analyze        # 6. validate the matrix and emit tables/figures

Manifest creation and tokenizer probing need network/Hugging Face access; training and scoring
need a GPU. Use the immutable execution-source snapshot whose bytes match the new lock for
full-archive verification. See [experiments/README.md](experiments/README.md) and
[docs/reproducibility.md](docs/reproducibility.md).

## Build or verify the other workspaces

### Paper B — retrospective composition

    make -C papers/base-adapter-composition verify
    make -C papers/base-adapter-composition pdf

The composition result is anchored to SHA-256
92c2cbc3ea71d5e6c72bf0e6f7eb0d3ef15f0e61f9fffaada885dade460e3ccc.
The checked-in protocol template is intentionally draft_not_executed. The protocol-locked target
is expected to fail until a real prospective contract supplies the cohort, margin, controls,
statistics, software lock, and systems measurements.

### Mortgage benchmark and paper

    make -C mortgage-benchmark smoke PY=../.venv/bin/python
    make -C mortgage-benchmark test  PY=../.venv/bin/python
    make -C papers/mortgage-guardrail-benchmark pdf

Generation is frozen because its LLM-backed construction is stochastic; evaluation reproduces
from the committed release and guard scores. Human/SME sign-off gates are documented in
[mortgage-benchmark/README.md](mortgage-benchmark/README.md).

### Paper C and ExpGuard

- [docs/paper-c-prereg.md](docs/paper-c-prereg.md) fixes the objective-axis hypotheses,
  estimands, GRPO single-token null, and decision rules before execution.
- [experiments/paper_c_preference.py](experiments/paper_c_preference.py) contains the
  deterministic preference/reward recipe and offline self-test.
- [experiments/run_paper_c_objective.py](experiments/run_paper_c_objective.py) is a trainer
  scaffold awaiting GPU smoke validation. No Paper C lock, objective scores, analyzer, result,
  or paper currently exists.
- [experiments/eval_expguard_external.py](experiments/eval_expguard_external.py) supports
  gated Hugging Face loading, a local Parquet input, mock smoke runs, and regeneration from
  text-free scores. The full four-checkpoint artifact is not complete or released.

Code presence is not evidence of a result.

## Auditable evidence chain

- **Locks and releases:** Paper A v2 binds the config, manifests, audit, prompt rendering,
  source state, score identity, and release anchor. Legacy evidence requires an explicit flag.
- **Independent split audit:** audit outputs fail closed on overlap, label conflicts, balance,
  upstream-family disjointness, revisions, and deterministic near-duplicate dispositions.
- **Text-free scores:** scores.parquet stores row identities/content hashes and logits rather
  than redistributing third-party prompt text.
- **Canonical metrics:** [guard_research/metrics.py](guard_research/metrics.py) provides
  sklearn-backed, tie-aware, permutation-invariant average precision; thresholds and provenance
  live beside it.
- **Generated manuscripts:** completed claim-bearing values enter LaTeX through generated macros
  and tables, with byte or hash checks on the mature paths.

## Known boundaries

- Four compact checkpoints from two broad lineages are a fixed panel, not a model population.
- Acts I and III use dataset-held-out but previously inspected sources.
- Paper C's objective-axis result does not exist yet.
- ExpGuard currently has only a partial, unreleased base-checkpoint evaluation.
- The mortgage benchmark is synthetic and policy-card-consistent, not SME-adjudicated or legally
  authoritative; its current G1/D0 quadrant is empty.
- Ranking recovery does not imply threshold or calibration transfer.
- The public Paper A cache cannot independently rehash omitted adapter bytes and full run metadata.

## Citation and licenses

[CITATION.cff](CITATION.cff) currently cites Paper A only.

Repository code and original content are licensed under [Apache 2.0](LICENSE).
Third-party datasets and models retain their own licenses and access conditions. The mortgage
benchmark data card currently records that a separate redistribution license has **not yet been
selected**; review its [DATA_CARD.md](mortgage-benchmark/benchmark/v1_hmda2022/DATA_CARD.md)
before redistributing generated prompts.
