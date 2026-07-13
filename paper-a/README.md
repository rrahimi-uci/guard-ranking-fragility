# Paper — *The Benchmark Chooses the Winner*

ACM-formatted (`acmart`, `sigconf` + `nonacm`) manuscript for the focused Paper A:
**measuring fine-tuning specialization across safety-guard benchmarks** (4 checkpoints
× 5 seeds, LoRA-SFT on a decontaminated corpus).

## Files

- `benchmark_chooses_the_winner.tex` — the paper (authoritative source).
- `benchmark_chooses_the_winner.pdf` — compiled output.
- `refs.bib` — bibliography.
- `figures/specialization_plane.pdf` — the single figure: represented-source gain vs.
  held-out transfer (the specialization plane).
- `tab_primary_gen.tex`, `tab_sensitivity_gen.tex` — generated tabulars `\input` by the
  manuscript (primary claim table + leave-one-out sensitivity).

Every number, table, and figure is **generated from the committed scores**, not
hand-entered. They are produced by the analysis step and copied into this folder:

```bash
make analyze         # (from repo root) → artifacts/paper_a_sft/analysis/{tables,figures}
```

`artifacts/paper_a_sft/analysis/` holds the canonical outputs
(`tables/table3_primary.tex`, `tables/table4_per_benchmark.tex`,
`figures/specialization_plane.pdf`); the copies here are what the `.tex` consumes.

## Build
```bash
make            # compile benchmark_chooses_the_winner.tex -> .pdf (tectonic; fetches acmart)
make clean      # remove build artifacts
```
Or directly: `tectonic benchmark_chooses_the_winner.tex` (needs network on first run to
fetch `acmart`).

## Provenance
All values trace to `artifacts/paper_a_sft/scores/scores.parquet` via the canonical
metric module (`guard_research/metrics.py`) — see the repository
[`../README.md`](../README.md) "Auditable evidence chain". The earlier broad study's
manuscript, prose draft, and its five figures are preserved in git history (they are
not part of this focused paper); the broad-study code lives under [`../legacy/`](../legacy).
