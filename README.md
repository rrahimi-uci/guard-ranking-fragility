# The Benchmark Chooses the Winner

Reproduction code, data manifests, and an auditable evidence chain for the paper

> **The Benchmark Chooses the Winner: Measuring Fine-Tuning Specialization Across Safety-Guard Benchmarks**
> Reza Rahimi (JazzX AI)

Parameter-efficient fine-tuning of a small language model into a prompt-safety
guard **specializes it to the benchmarks it was trained on**. We measure this
directly on a fixed panel of four instruction-tuned checkpoints, each fine-tuned
five times, on a decontaminated corpus — and separate two effects that prior work
conflates: gains on the *sources represented in training* versus transfer to
*held-out datasets*. Scope: every evaluation classifies **input prompts** (unsafe =
harmful content, jailbreak, or prompt injection).

## Headline result

Across the panel (Qwen2.5-1.5B, SmolLM2-1.7B, SmolLM3-3B, Qwen3-4B; seeds 42–46),
LoRA-SFT on a decontaminated 1,200-row corpus:

| Effect | Δ macro-AP (base → SFT) | 95% one-sided bound | Gate |
|---|---|---|---|
| **Represented** sources (in training) | **+0.325** | LCB **+0.281** | A ✓ (gain > 0) |
| **Transfer** to held-out datasets | **−0.050** | UCB **−0.029** | B ✓ (change < 0) |

SFT lifts every checkpoint to ≈0.98 macro-AP on represented sources (largest gain
for the weakest base) but **degrades** aggregate transfer, and the realized
false-positive rate on held-out data rises from **8.3% (base) to 13.7% (SFT)**.
Transfer is heterogeneous by checkpoint — SmolLM2 +0.05, Qwen2.5 −0.03,
Qwen3-4B −0.10, SmolLM3 −0.12 — but both claims are decided by a **family+seed
hierarchical bootstrap** with **intersection-union claim gates** that are
leave-one-family-out sign-stable. The conclusion is a measured **in-source
specialization trade-off**, not a universal "fine-tuning hurts."

All numbers above are regenerated from the committed score table by
[`make repro`](#reproduce-no-gpu) — see [artifacts/paper_a_sft/analysis/](artifacts/paper_a_sft/analysis).

## Repository layout

```text
guard_research/      canonical, auditable library: tie-aware metrics, provenance
                     (hashing + MinHash), the frozen prompt, calibration thresholds
experiments/         the six-step Paper A pipeline (prepare → audit → lock →
                     train → eval → analyze); see experiments/README.md
configs/             paper_a_sft.yaml — the single study config
tests/               unit tests for the canonical metrics, thresholds, manifests
artifacts/paper_a_sft/   the evidence chain: LOCK.json, audit/, analysis/,
                     scores/scores.parquet (row-keyed hashes + logits, no raw text),
                     runmeta/
paper/               the manuscript (tectonic) + generated tables/figures
paper-html/          the HTML edition (self-contained, offline math)
benchmark-explore/   an interactive benchmark explorer webpage
notebooks/           bundled benchmark data + frozen evaluation rows
docs/                design/planning notes
legacy/              the earlier broad study + planned Paper B code (quarantined,
                     still runnable; not part of this reproduction)
```

## Install

Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # library + training + figures + dev
# or, for the exact versions the results were produced with:
pip install -r requirements.txt
```

CPU-only reproduction (below) needs just the core: `pip install -e .`.

## Reproduce (no GPU)

Because `scores/scores.parquet` is committed, every table and figure regenerates
from it on CPU in seconds:

```bash
make repro      # analyze committed scores → tables + figures + claim gates
make test       # 30 unit tests (canonical metrics / thresholds / manifests)
make selftest   # synthetic end-to-end check of the analysis
make paper      # build the PDF (needs tectonic)
```

`make help` lists every target.

## Reproduce from scratch (GPU)

The full pipeline runs in order; steps 1–3 and 6 are CPU, steps 4–5 need a GPU and
Hugging Face access (`HF_TOKEN`; copy [.env.example](.env.example) → `.env`):

```bash
make manifests   # 1. build the decontaminated 1,200-row manifest + held-out sets
make audit       # 2. recompute + hard-assert decontamination (0 train↔eval overlap)
make lock        # 3. freeze config + manifest hashes into LOCK.json
make train       # 4. train the 4 × 5 LoRA-SFT panel
make eval        # 5. score bases + adapters → scores.parquet
make analyze     # 6. macro-AP + bootstrap + claim gates → tables/figures
```

See [experiments/README.md](experiments/README.md) for what each step does.

## Auditable evidence chain

The study is designed to be checkable without rerunning the GPU work:

- **`LOCK.json`** binds the config, data-manifest hashes, and audit result; the
  training and scoring steps refuse to run against a mismatched lock.
- **`audit/`** independently recomputes the decontamination facts and hard-asserts
  them (zero exact/conflicting train↔eval overlap, label balance, family
  disjointness).
- **`scores/scores.parquet`** stores per-row **content hashes and model logits
  only — never raw prompt text** — so the third-party benchmark text is not
  redistributed, yet every metric is recomputable.
- **`analysis/`** holds `results.json`, `claim_checks.json`, per-seed and
  per-benchmark tables, and the figure, all emitted by the canonical metric module.

All metrics come from [guard_research/metrics.py](guard_research/metrics.py)
(sklearn-backed, tie-aware, permutation-invariant) — there is no ad-hoc
average-precision loop anywhere in the pipeline.

## Data & provenance

Training sources are pulled from Hugging Face pinned by revision (ToxicChat,
Prompt-Injections, Jailbreak-Classification) and the held-out evaluation rows from
`notebooks/outputs/frozen_eval_rows.json`. Raw non-commercially-licensed manifest
rows are gitignored; they regenerate deterministically from the builder and config.
Provenance (NFKC-normalized content/family SHA-256, MinHash near-duplicate families)
lives in [guard_research/provenance.py](guard_research/provenance.py).

## Earlier broad study & Paper B

The broad measurement study this paper was distilled from — protocol-dependent
ranking, the SFT/DPO/GRPO objective comparison, a mortgage-compliance case study,
guardrail baselines, an ensemble mitigation, and a fairness probe — is preserved
under [legacy/](legacy) as the basis for a planned follow-up. It is **not** needed to
reproduce this paper. Note the metric caveat in [legacy/README.md](legacy/README.md).

## Citation & license

If you use this work, please cite it via [CITATION.cff](CITATION.cff).
Licensed under [Apache 2.0](LICENSE).
