# Benchmark Explorer

Self-contained HTML explorers for browsing every available sample in each included benchmark.
Each card shows the source prompt or task, its label, tags, and supporting benchmark context.
Search and label filters run against the complete datasets, with large result sets paginated at
100 samples per page.

Regenerate everything deterministically in source order:

```bash
python3 benchmark-explorer/generate.py
```

## Files

| File | Contents | Committed? |
|------|----------|------------|
| `generate.py` | Generator. Reads the 7 `data/benchmarks/full/*.jsonl` sets, SafePyramid, the hardened mortgage guard set, MortgageGuardBench-2K, and ExpGuard. It can use `HF_TOKEN` from `.env` to cache ExpGuard when needed. | ✅ yes |
| `index.public.html` | **Shareable build.** All 16,146 rows from 10 public/synthetic benchmark sections. No gated text. | ✅ yes |
| `index.html` | **Full local build.** All 18,421 rows in 13 sections: everything above **plus all 2,275 gated ExpGuard rows** across finance, healthcare, and law. Embeds gated prompt text. | ❌ **gitignored** |

## Why two files

ExpGuard (`6rightjade/expguardmix`) is a **gated / licensed** dataset. Its prompt text must
not be redistributed, so the repo never commits it (only text-free hashes + labels + scores
live under `artifacts/expguard_external/`).

- `index.html` embeds ExpGuard text and is therefore **gitignored** — for local viewing only.
- `index.public.html` is the gated-free equivalent — safe to commit and share.

`generate.py` builds `index.html` with ExpGuard when the dataset is already cached or when a valid
`HF_TOKEN` in `.env` can access it; otherwise it prints a note and `index.html` omits ExpGuard.

> ⚠ Never commit or share `index.html`. Share `index.public.html`.
