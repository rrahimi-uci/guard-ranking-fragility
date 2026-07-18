# Benchmark Explorer

Self-contained HTML explorers that show 20 samples (10 safe + 10 unsafe) per benchmark
to build intuition for what each dataset actually contains. Each card shows a real prompt
with its safe/unsafe label, tags, and (for the domain sets) supporting context.

Regenerate everything (deterministic, seed 42):

```bash
python benchmark-explorer/generate.py
```

## Files

| File | Contents | Committed? |
|------|----------|------------|
| `generate.py` | Generator. Reads `data/benchmarks/*.jsonl`, `data/mortgage_guard_bench_2k_v0_1_0`, and (locally) the cached ExpGuard parquet. | ✅ yes |
| `index.public.html` | **Shareable build.** 7 public guard benchmarks + synthetic MortgageGuardBench-2K (8 sections, 160 samples). No gated text. | ✅ yes |
| `index.html` | **Full local build.** Everything above **plus** the gated ExpGuard domains — finance, healthcare, law (11 sections, 220 samples). Embeds gated prompt text. | ❌ **gitignored** |

## Why two files

ExpGuard (`6rightjade/expguardmix`) is a **gated / licensed** dataset. Its prompt text must
not be redistributed, so the repo never commits it (only text-free hashes + labels + scores
live under `artifacts/expguard_external/`).

- `index.html` embeds ExpGuard text and is therefore **gitignored** — for local viewing only.
- `index.public.html` is the gated-free equivalent — safe to commit and share.

`generate.py` builds `index.html` with ExpGuard only if the dataset is present in your local
Hugging Face cache; otherwise it prints a note and `index.html` omits ExpGuard.

> ⚠ Never commit or share `index.html`. Share `index.public.html`.
