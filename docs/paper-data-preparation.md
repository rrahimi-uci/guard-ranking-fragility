# Data Preparation (paper methods section — draft)

*This is the reviewer-facing description of how training and evaluation data are prepared. Every
mechanism below is implemented in the repo; file/function references are given so the section is
verifiable, not aspirational.*

## Overview

We evaluate binary safety classification (positive class = `unsafe`) across three axes — **guardrail**,
**red-teaming**, and **over-refusal** — over seven ungated public benchmarks, plus a separate
policy-guardrailing benchmark (SafePyramid) reported on its own axes. Two logically distinct datasets
are prepared, by different code paths, and kept disjoint by four independent leakage safeguards:

1. **Benchmark evaluation sets** — held-out test data used to *score* every guard, built by
   `evaluation.benchmarks.load_benchmark`.
2. **Training sets** — used to *train* the small-model guards, built by
   `data.training_sets.build_training_set`.

A guard is never scored on data it could have trained on; §4 states the guarantees precisely.

## 1. Source normalization to a single schema

Each source dataset uses its own harm scheme. Per-dataset *normalizers*
(`data.loaders`, pure and unit-tested) map every raw row onto one record:

```
{ "text": str, "label": "safe" | "unsafe", "hazard": <taxonomy>, "source": str }
```

Two normalization details matter for correctness:

- **BeaverTails** is a per-(prompt, response) dataset where `is_safe` labels the *response*, so the same
  prompt recurs with conflicting labels. `loaders.dedup_unsafe_wins` aggregates to **one label per
  prompt** (a prompt is `unsafe` iff *any* of its responses is unsafe), so the gold labels are
  internally consistent (no prompt appears as both classes).
- Datasets are loaded from **pinned Hugging Face revisions** (paper release) so the row set is fixed;
  where a source offers multiple splits we state exactly which split each dataset draws from.

## 2. Benchmark evaluation sets

`load_benchmark(name, balanced=True, per_class=N, seed=42)` loads a benchmark (from a local full-set
cache, else Hugging Face) and takes a **class-balanced subset**: `per_class` `safe` + `per_class`
`unsafe`, deterministically shuffled with a **fixed seed (42)** and cached to
`data/benchmarks/<name>.jsonl`. Determinism is the point: the same seed and `per_class` reproduce the
identical subset on every machine, which is what makes matched-`n` comparison across guards possible.

- **Positive class** is `unsafe`; we report macro Precision/Recall/F1 (and per-axis P/R/F1). On
  class-balanced subsets, Precision already reflects over-blocking (a blocked benign prompt is a false
  positive); over-refusal is additionally isolated by the XSTest axis.
- **BeaverTails** is scored on the held-out `30k_test` split — disjoint from the `30k_train` split used
  for training (§3), so a guard trained on BeaverTails content is not evaluated on it.
- For the final paper numbers we score on the **full** benchmarks (not subsampled) where feasible, and
  otherwise report the exact per-`(guard × benchmark)` counts (`n`, `n_safe`, `n_unsafe`).

## 3. Training-set construction

`build_training_set(strategy, sources, per_class=N, holdout_ratio=0.2, seed=42)` composes a training set
from benchmark *sources* and writes `data/train_sets/<name>/{train,test}.jsonl` + a `meta.json`
provenance record.

- **Sources are drawn from splits disjoint from evaluation.** `default_training_loader` pulls each
  source from a training split that does not overlap its benchmark eval split — e.g. BeaverTails from
  `30k_train` (benchmark uses `30k_test`).
- **Split-before-pool.** Each source is split into train/held-out-test *individually* (via
  `split.train_test_split`) **before** pooling, so the returned train and held-out test are disjoint by
  construction; `split.assert_no_leakage` verifies it.
- **Two sampling modes.**
  - *Count mode* (`per_class > 0`): a class-balanced `per_class` subset per source for train, half that
    for the held-out test.
  - *Percentage mode* (`per_class ≤ 0`): pool all sources, stratified `ratio_split` at `holdout_ratio`
    (default 80/20).
- **Composition strategies.** `balanced` (½ safe / ½ unsafe), `mixed` (blend sources for distribution
  diversity), `red_team` (prompt-injection + jailbreak sources — targeting the weakest axis), and
  `over_refusal_aware` (adds benign-but-scary negatives, §3.1).
- **Determinism.** All shuffles/splits use `seed=42`; `split.holdout_count` guarantees a non-empty test
  set whenever the data is splittable.

### 3.1 Over-refusal negatives come from OR-Bench, not the eval set

The `over_refusal_aware` strategy augments **train only** with benign-but-scary prompts so the guard
learns not to over-block. These negatives are sourced from **OR-Bench** (`bench-llm/or-bench`, streamed
and capped; all prompts benign → `safe`) — deliberately **not** XSTest, which is our over-refusal
*evaluation* set. Training on XSTest would inflate the headline over-blocking metric; keeping the two
sources separate (`OVER_REFUSAL_SOURCE = "or_bench"`) removes that circularity. The augmentation
excludes anything already in train/test, records how many prompts were actually added in `meta.json`
(`augmentation_added`/`augmentation_source`/`augmentation_error`), and warns rather than silently
adding zero.

## 4. Leakage safeguards (four independent layers)

A guardrail benchmark is only meaningful if the model never trained on the test items. We enforce this
four ways:

1. **De-duplication within a split** — `train_test_split` de-dups by normalized text (case/whitespace
   insensitive) first, so the same prompt cannot land in both train and its held-out test.
2. **Disjoint-by-construction** — sources are split before pooling; built train and held-out test never
   overlap (`assert_no_leakage`).
3. **Training split ≠ evaluation split** — training pulls disjoint splits (e.g. BeaverTails
   `30k_train`), while benchmarks score the eval split (`30k_test`); over-refusal training uses OR-Bench,
   evaluation uses XSTest.
4. **Test-time contamination filter (defense in depth)** — at scoring, `training.runner.score_guard`
   runs `split.find_leakage(train_recs, benchmark_recs, fuzzy=True)` — **exact** (normalized) match
   plus **near-duplicate** match (word-set Jaccard ≥ 0.9, ≥ 5 tokens) — and **drops** any benchmark
   prompt found in the model's training data, reporting the count as `dropped_leaked` per benchmark.

For every trained guard we additionally run the contamination filter over the **union** of its training
data against each benchmark and publish the `dropped_leaked` counts. The training corpora of the closed
"mini" baselines are unknowable, so their contamination is an unquantifiable caveat we state explicitly
rather than one we can rule out.

## 5. Generalization protocol (recommended for the red-team axis)

Within-dataset train/test splits can flatter a model via near-duplicate memorization. For the
red-teaming claim we therefore use **leave-one-benchmark-out (LOBO)**: train on
`prompt_injections` + `jailbreak_classification` and evaluate on the held-out **JailbreakBench** (and
symmetrically), so the reported red-team lift reflects transfer, not memorization.

## 6. Reproducibility of the data

- **Deterministic:** fixed seed (42) for every subset/split; greedy decoding at eval time; balanced
  subsets cached to JSONL and regenerable byte-for-byte.
- **Pinned:** Hugging Face dataset revisions recorded; balanced subset sizes and per-class counts
  reported per benchmark.
- **Released:** the cached benchmark subsets, the built training sets (`meta.json` provenance), and the
  per-sample prediction dumps (`outputs/predictions/<guard>.json`, each row carrying a stable sample
  key) — so every downstream metric, ensemble, and leakage figure is reproducible offline with no GPU
  and no API access.
- **Gated data** (WildGuardMix, WildJailbreak, HarmBench, StrongREJECT, AdvBench, PINT) requires license
  acceptance; when unavailable it is reported as *not run*, never fabricated.

## 7. Known limitations (stated for honesty)

- **Label noise:** several sources carry noisy labels (e.g. BeaverTails response-safety used as
  prompt-safety; ToxicChat/jailbreak sets). We therefore frame results as **relative** (same harness,
  same inputs, same subsets) rather than as absolute accuracy, and flag in-domain vs OOD benchmarks.
- **Over-refusal coverage:** over-refusal is evaluated primarily on XSTest; broader coverage
  (OR-Bench-Hard / PHTest as OOD eval) is a planned extension.
- **Closed-baseline contamination** cannot be measured (see §4).
