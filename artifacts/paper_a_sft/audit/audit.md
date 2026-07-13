# Paper A — data audit

Manifest dir: `artifacts/paper_a_sft/manifests`  
Provenance: `guard_research.provenance` (MinHash backend `numpy_fallback`, 256 perms, 5-gram, J>=0.85)

## Hard assertions

| assertion | result |
|---|---|
| `or_bench_train_count == 0` | PASS |
| `beavertails_train_count == 0` | PASS |
| `exact_train_vs_eval_overlap == 0` | PASS |
| `conflicting_label_overlap == 0` | PASS |
| `every_row_has_source_revision == true` | PASS |
| `every_row_has_content_hash == true` | PASS |
| `every_near_duplicate_candidate_has_disposition == true` | PASS |

**All hard assertions pass: True**

## Row and class counts

| manifest | rows | safe | unsafe |
|---|---:|---:|---:|
| train | 1200 | 600 | 600 |
| calibration | 452 | 215 | 237 |
| id_test | 676 | 364 | 312 |
| transfer_test | 1580 | 790 | 790 |
| orbench_safe_stress | 400 | 400 | 0 |
| harmbench_positive_stress | 200 | 0 | 200 |

### Per-source counts

- **train**: jailbreak_classification (safe 200, unsafe 200); prompt_injections (safe 200, unsafe 200); toxicchat (safe 200, unsafe 200)
- **calibration**: jailbreak_classification (safe 43, unsafe 62); prompt_injections (safe 16, unsafe 29); toxicchat (safe 156, unsafe 146)
- **id_test**: jailbreak_classification (safe 80, unsafe 77); prompt_injections (safe 40, unsafe 27); toxicchat (safe 244, unsafe 208)
- **transfer_test**: jailbreakbench (safe 60, unsafe 60); wildguardtest (safe 400, unsafe 400); wildjailbreak (safe 210, unsafe 210); xstest (safe 120, unsafe 120)
- **orbench_safe_stress**: orbench (safe 400, unsafe 0)
- **harmbench_positive_stress**: harmbench (safe 0, unsafe 200)

## Exact train↔eval overlap

- total exact overlap: **0** (same-label 0, conflicting 0)

| eval split | exact | same-label | conflicting |
|---|---:|---:|---:|
| calibration | 0 | 0 | 0 |
| id_test | 0 | 0 | 0 |
| transfer_test | 0 | 0 | 0 |
| orbench_safe_stress | 0 | 0 | 0 |
| harmbench_positive_stress | 0 | 0 | 0 |

## Near-duplicate sensitivity (char-5gram MinHash)

| threshold | total pairs | cross train↔eval | within-train | within-eval |
|---|---:|---:|---:|---:|
| 0.80 | 113 | 6 | 35 | 72 |
| 0.85 | 72 | 0 | 21 | 51 |
| 0.90 | 48 | 0 | 15 | 33 |

> 0.85 is the prespecified candidate-generation threshold; the builder removes the train-side of every cross-split component at >=0.85, so cross_train_eval_pairs at 0.85 must be 0. Counts at 0.80/0.90 are reported for sensitivity only; dispositions are not changed after final scores are viewed.

## Family validation

- total families: 4444; train families 1180; eval families 3264
- **train↔eval shared families: 0** (0 expected: cross-split near-dup train members are removed)

## Disposition coverage

- exact overlaps removed at build: 60
- cross-split near-dup components removed at build: 44
- known WildJailbreak overlaps removed: 4
- undisposed cross-split near-dup pairs at 0.85 in final manifests: 0

## License inventory

| source | license(s) | redistribution class(es) |
|---|---|---|
| harmbench | MIT (200) | permissive (200) |
| jailbreak_classification | unknown-verify-before-lock (662) | research_reconstruct_only (662) |
| jailbreakbench | MIT (120) | permissive (120) |
| orbench | unknown-verify-before-lock (400) | research_reconstruct_only (400) |
| prompt_injections | unknown-verify-before-lock (512) | research_reconstruct_only (512) |
| toxicchat | CC-BY-NC-4.0 (1154) | noncommercial_reconstruct_only (1154) |
| wildguardtest | ODC-BY-AI2-gated (800) | gated_no_redistribution (800) |
| wildjailbreak | ODC-BY-AI2-gated (420) | gated_no_redistribution (420) |
| xstest | CC-BY-4.0 (240) | permissive (240) |

## Role and join validation

- role validation ok: True
- OR-Bench stress labels: ['safe']; HarmBench stress labels: ['unsafe']
- one-to-one joins (unique sample_id): True
- OR-Bench in train: 0; BeaverTails in train: 0
