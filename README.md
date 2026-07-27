# The Safety-Guard Benchmark Chooses the Winner

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

Two controls sit beside the acts. A **recipe control** (base-anchored KL-SFT) asks how much of Act I's
transfer loss belongs to the *unregularized* recipe rather than to fine-tuning as such. And one
**analysis-preregistered** study — ten checkpoints across six model families, six of them released vendor
guards — asks whether the same tradeoff hits models that are *already* purpose-built guards. Its claim
registry was committed before any score existed, and it is the one place in the program where a
preregistered criterion is reported as **failed**. RQ1 — ordinary SFT specializes released guards too — is
**supported**: represented-source macro-AP rises **+0.174** (LCB **+0.129**), concentrated relative to
held-out transfer. RQ2 — is KL-SFT *free*? — is **not supported**: it does retain transfer, but at a
represented-source cost of **−0.036** (LCB **−0.060**), which does not clear the preregistered **−0.02**
non-inferiority margin. Treat β as a tradeoff dial, not a default.

The **[unified research report](papers/unified-report/unified_report.pdf)** is the synthesis:
*The Safety-Guard Benchmark Chooses the Winner: Measuring, Tuning, and Composing Small Safety Guards in
High-Compliance Business Domains.* One entry point (below) regenerates and byte-checks the covered
tables and figures from committed per-row scores — and prints the coverage it did *not* achieve, because
that coverage is partial: 12 of the 24 generated artifacts the report `\input`s are byte-checked in any
environment, 4 need the lock-pinned analysis environment, and 8 are committed outputs of their own
locked analyses that are not yet wired into the byte-check.

**Study state lives in one place.** [`studies/registry.yaml`](studies/registry.yaml) is the
normative record of every study's state, evidence tier, contract class, and how to verify it.
[`studies/README.md`](studies/README.md) and [`papers/README.md`](papers/README.md) are generated
views of it; this README summarizes it and does not redefine it. Run `make check-registry`.

---

## Repository structure

Layout follows [`docs/architecture/repository-layout-v2.md`](docs/architecture/repository-layout-v2.md).
Released, path-bound studies stay at their compatibility paths; new structure is prospective.

```text
safety-guard-dynamics/
├── studies/                             # registry.yaml = normative study state (+ schema, generated README)
│   ├── composition/                     # ┐ navigation packages: generated README + `make verify`.
│   ├── expguard/                        # │ Code stays at code_root — LOCK.json binds experiments/,
│   ├── klsft/                           # │ so these point at it rather than copying it.
│   ├── starting-type-adaptation/        # ┘
│   └── paper-c-specialize-align-mortgage-v1/  # self-contained study package (a copy; see below)
├── benchmarks/registry/                 # distribution.yaml = per-source redistribution decisions
├── apps/benchmark-explorer/             # ledger-driven, fail-closed explorer build + negative tests
├── benchmark-explorer/                  # withdrawn predecessor, kept as a documented compatibility surface
├── guard_research/                      # canonical library: metrics, thresholds, prompts, provenance
├── experiments/                         # Paper A pipeline; composition; KL-SFT; ExpGuard; adaptation
├── mortgage-benchmark/                  # generator (magen/), frozen v1 release, scorer, baselines, tests
├── papers/                              # manuscripts (LaTeX + built PDFs); see papers/README.md for state
│   ├── unified-report/                  # ← the three-act synthesis (primary artifact) + slides/
│   ├── unified-report-html/             # the same report as HTML, generated from those same sources
│   ├── finetuning-specialization[-simplified]/    # Paper A  (+ plain-language edition)
│   ├── base-adapter-composition[-simplified]/     # Paper B  (+ plain-language edition)
│   ├── mortgage-guardrail-benchmark[-simplified]/ # mortgage paper (+ plain-language edition)
│   └── paper_c/                         # stopped predecessor + specialize_then_align/ successor
├── artifacts/                           # released & development evidence: locks, manifests, text-free scores
├── configs/                             # study configs + the Paper A v2 release anchor
├── tools/                               # registry validator, link checker, index renderer, study verifier, tree digest
├── docs/architecture/                   # repository-layout-v2.md (the migration plan this tree follows)
├── data/                                # ignored: raw/licensed corpora and download cache
├── runs/                                # ignored: transient execution output, runs/<study_id>/<run_id>/
└── tests/                               # canonical unit + artifact-contract tests
```

Each study has one page under `studies/` answering what it asks, what it may claim, where
its code and evidence live, and how to verify it — `make -C studies/expguard verify`. Those
pages are **generated from the registry**, and their Makefiles look the command up rather
than restating it, so a package cannot disagree with the registry; `make check-registry`
fails if one goes stale. The code itself does not move: `artifacts/paper_a_sft_v2/LOCK.json`
binds `experiments/` paths in eight places, so a copy would either break the lock or
silently diverge from it.

One study now exists in both layouts. `studies/paper-c-specialize-align-mortgage-v1/` is a **copy**
of `papers/paper_c/specialize_then_align/`, not a move: the plan forbids relocating an active tree
before the new location verifies independently, so both are tested and both must stay behaviourally
identical. The evidence that the migration preserved behaviour is that they fail *identically* —
70 passed, 1 failed, that one being a declared `expected_fail` — rather than that the new one merely
runs. Migrating exposed two real defects: fixed-parent repository discovery (`parents[2]`) is correct
at the old depth but resolves outside the repository at the new one; and the storage contract's
`runs/` location was unreachable. Both are fixed in both trees. See `studies/paper-c-specialize-align-mortgage-v1/provenance/MIGRATION_MANIFEST.json`
for the tree hashes.

Four storage classes, enforced rather than described: **source** is tracked, **evidence** enters
`artifacts/<study_id>/` under an allowlist, **raw or licensed data** stays in ignored `data/` with its
decision recorded in `benchmarks/registry/`, and **transient runs** go to ignored `runs/`. Committed
release artifacts keep pinned identifiers, source revisions, content hashes, and **text-free per-row
scores** (row hash → score) rather than redistributing prompts.

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

**One command regenerates the unified report's covered tables and figures from committed per-row scores —
no GPU, no network — and prints what it could *not* cover:**

```bash
make -C papers/unified-report reproduce         # regenerate the covered generated/ tables + figures/
make -C papers/unified-report reproduce-check    # + assert byte-identity with the committed copies
```

`reproduce.py` dispatches to each study and re-derives the exact LaTeX the report `\input`s:

| Study | Source of truth | Notes |
|---|---|---|
| Act I — specialization | `artifacts/paper_a_sft_v2/scores/scores.parquet` | needs the lock-pinned analysis env |
| Act I — matched false-alarm budget | `artifacts/paper_a_sft_v2/scores/scores.parquet` | `matched_fpr.py`; rethresholding is ranking arithmetic, so any env |
| Act II — composition | `artifacts/paper_a_sft_v2/analysis/composition/` | from committed scores |
| Act III — mortgage | `mortgage-benchmark/out_eval/scores_*.json` | from committed scores |
| Act III — ExpGuard (finance/health/law) | `artifacts/expguard_external/scores_*.json` | `eval_expguard_external.py --from-scores` |
| Latency (P50/P90/P99) | `artifacts/paper_a_sft_v2/scores/scores.parquet` | per-row `latency_ms`, no GPU |

**Coverage is partial and the harness says so rather than implying completeness.** Of the 24 generated
artifacts the report `\input`s, **12 are byte-checked** by this command, **4** require the lock-pinned
environment (the Act I tables and the `\Rep*`/`\Transfer*` macros), and **8** — the adaptation, KL-SFT,
ensembling and mortgage-composition tables — are committed outputs of their own locked analyses that are
not yet wired in. The command exits nonzero unless every covered artifact verifies, and it asserts that
those three counts *partition* the inputs exactly, so an artifact cannot fall out of the tally unnoticed.

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
| **Unified three-act report** | [PDF](papers/unified-report/unified_report.pdf) · [HTML](papers/unified-report-html/index.html) · [LaTeX](papers/unified-report/unified_report.tex) | teaching boxes integrated into the report |
| Fine-tuning specialization (A) | [PDF](papers/finetuning-specialization/benchmark_chooses_the_winner.pdf) | [annotated](papers/finetuning-specialization-simplified/) |
| Base+adapter composition (B) | [PDF](papers/base-adapter-composition/compose_dont_tune.pdf) | [simplified](papers/base-adapter-composition-simplified/) |
| Mortgage guardrail benchmark | [PDF](papers/mortgage-guardrail-benchmark/mortgage_guardrail_benchmark.pdf) | [simplified](papers/mortgage-guardrail-benchmark-simplified/) |

Claim-bearing numbers enter LaTeX only through generated macros/tables (`generated/`), never hand-typed;
`reproduce-check` guards against drift. The report also ships two Graphviz flowcharts of the study's
processes — the [data-split construction](papers/unified-report/figures/data_splits.dot) and the
[paired experimental design](papers/unified-report/figures/experiment_design.dot).

**Read it in a browser.** [`papers/unified-report-html/`](papers/unified-report-html/) is an HTML
edition of the unified report — sticky section navigation, tables wider than the prose column, MathJax
formulas, SVG figures, light/dark theme. It is *generated* from the same LaTeX sources and the same
committed `generated/*.tex` artifacts as the PDF, so no number is retyped and none can drift:

```bash
make report-html         # rebuild papers/unified-report-html/index.html
make check-report-html   # assert the committed HTML matches a fresh build
```

The build asserts its own float numbering against the built PDF — it reads every `Table N:` and
`Figure N:` caption out of `unified_report.pdf` and fails if the counts disagree, so a citation of
"Table 4" means the same table in both editions. That check earned its place immediately: it caught
four tables that pandoc had silently degraded into `<br>`-separated text because of `@{}` column
padding, and eleven figures that were invisible because pandoc emits `<embed>`, not `<img>`, for a
PDF graphic. Current state: 22 tables, 16 figures, 10 numbered equations, 224 cross-references,
0 mismatches.

---

## Verification

Tiered, because a single `make test` cannot mean the same thing for a released study and
for a stopped protocol candidate. Each tier names one absolute interpreter, since nested
Makefiles disagree on `PY` vs `PYTHON`.

```bash
make check-registry     # registry + distribution ledger validate; generated indexes current
make check-links        # relative Markdown links in every index resolve
make check-fast         # the above, plus every hermetic test suite
make check-locks        # runs each study's declared verification command
make check-papers       # isolated manuscript builds
make check-data-local   # inventories over ignored local corpora (explicitly not hermetic)
make check-release      # release reproduction under pinned Python 3.12
```

`check-release` exists because Paper A's `LOCK.json` pins a runtime software fingerprint
at Python 3.12 while the development `.venv` is 3.14 — so a local pass would not be
release verification. It skips cleanly, with install instructions, when no suitable 3.12
environment is present.

CI ([`.github/workflows/verify.yml`](.github/workflows/verify.yml)) runs the hermetic tier
plus registry, index-freshness, and link validation on Python 3.12, and fails the build if
any source reaches `publish_text` without an affirmatively redistributable license.

`check-fast` is hermetic in the sense that it needs no network and no ignored corpora —
but suites that load a real checkpoint **skip** rather than run: they set `HF_HUB_OFFLINE=1`
by design and need a warm Hugging Face cache. With a warm cache the root suite reports
**233 passed**; on a fresh clone with a cold one it reports **209 passed / 5 skipped**, those
five being module-level skips that stand for 24 individual tests. CI prints every skip reason
(`-rs`) so the tier cannot quietly shrink. Warm the cache once to run them locally:
`python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')"`.

`check-fast` deliberately tolerates two declared failures — the same one twice. The Paper C
candidate lock binds live source bytes, so it fails once source evolves, and both the
predecessor tree and the migrated study package carry it. Each is recorded as `expected_fail`
with a reason in the registry, and `make check-locks` fails if an `expected_fail` ever starts
passing — a stale blocker is treated as an error, not a relief. That the two trees fail
*identically* (1 failed, 70 passed) is what evidences the migration preserved behaviour;
[`tests/test_migration_manifest.py`](tests/test_migration_manifest.py) fails if they drift.

`make -C papers/unified-report reproduce-check` is **not** in any read-only tier: it
overwrites canonical figures and intermediates. Run it in a disposable worktree.

## Distribution status

**No benchmark source is currently approved for verbatim public redistribution.**
[`benchmarks/registry/distribution.yaml`](benchmarks/registry/distribution.yaml) records a
per-source decision — license, access class, sensitive-text class, reviewer, and payload
hash — and fails closed: a source absent from the ledger defaults to `local_only`, and the
schema refuses `publish_text` unless the license affirmatively permits redistribution.

MortgageGuardBench-2K ships `LICENSE_NOT_SELECTED.md` stating that no publication license
has been chosen and that legal review is required before redistribution, yet all 2,000 of
its rows were embedded in the tracked 55 MB `benchmark-explorer/index.public.html`, along
with several general-safety corpora redistributed as verbatim rows under noncommercial or
unverified upstream terms.

**That artifact and its ungated generator are now withdrawn from tracking** — see
[`benchmark-explorer/README.md`](benchmark-explorer/README.md). The `.gitignore` comment
that authorized it ("Commit index.public.html (public + synthetic only)") asserted a
licensing conclusion per *file* while the real decision is per *source*, so it was wrong
as soon as a source was added, and nothing checked it.
[`tests/test_no_unlicensed_publication.py`](tests/test_no_unlicensed_publication.py)
replaces that comment with eight executable rules: the named blob and generator stay
untracked, no *generator* writes an ungated bulk page (caught by behaviour, not filename),
no bulk export is tracked under a publication path, no tracked publication file carries
restricted-corpus prose, every tracked corpus appears in the ledger, and the "nothing is
approved for redistribution" claim fails loudly if a licensing decision ever changes.

**`papers/` is inside that gate, with a quotation budget.** Adding the HTML edition put a
publishable web page under `papers/`, which the gate had not covered. Simply extending the
path list would have failed for the one reason that is not a licensing problem: a paper
*about* mortgage-policy compliance necessarily uses the vocabulary of mortgage-policy
compliance. So authored documents are separated from bulk exports by what actually
distinguishes them — bounded-and-reviewed versus unbounded. Each authored publication file
declares the restricted-vocabulary count measured when a human reviewed it; **growth fails
the build**, and an undeclared page is held to the strict export rule, so it is guilty until
reviewed. Calibration comes from this repository's own frozen benchmark: a restricted row
carries ≈2.7 probe hits, so the withdrawn 2,000-row export carried on the order of 5,400,
against **11** in the 342 KB manuscript — the paper's own policy vocabulary plus one worked
G0/D1 row. Two limits are stated in the test rather than implied: it is a *change* detector,
not a licence checker, and its probes are mortgage vocabulary, so prose from the
general-safety corpora is not probed at all.

The replacement is [`apps/benchmark-explorer/`](apps/benchmark-explorer/), which reads the ledger
instead of asserting a conclusion. Both build roots are ignored, and a source absent from the ledger is a
build *failure*, not a default-permit:

```bash
make explorer-public    # allowlist-only: text for publish_text sources, content hashes for the rest
make explorer-local     # full text for local inspection; dist/ is ignored and never published
```

A passing fixture build proves the allowlist works; it does not authorize publication. CI runs the public
build and then asserts that no fixture text reached `dist/public/`.

**Purged from history on 2026-07-26.** The explorer blob and
`data/guard_benchmark_hard.jsonl` — 334 rows of prompt text that had been force-added past
the `/data/` ignore rule and were public while absent from the ledger — were removed from
every commit with `git filter-repo`, and `main` was force-pushed. All 317 commits are
preserved; only the two blobs are gone, and the repository shrank from 116 MB to 40 MB.
Purging by current path was not sufficient: one blob had lived at three paths across past
reorganisations, and `git rev-list --objects` prints each object once, so removing one path
merely revealed the next. Stripping by blob id is what actually finished it.

Both files remain on local disk under ignored paths and are fully usable for development.
Two limits worth stating plainly: **a rewrite is not a retraction** — anything already
fetched is out, and GitHub may serve unreferenced objects by SHA until it garbage-collects
(ask GitHub Support to force it; the repository has no forks, which is the main reason this
was worth doing at all) — and every pre-rewrite commit SHA is void, so old links break.
Until a license is selected, no Pages, release, or shard build is authorized.

### GitHub Pages: wired, tested, and refusing

[`.github/workflows/pages.yml`](.github/workflows/pages.yml) can publish the HTML edition,
and **currently will not.** Serving that page is verbatim public redistribution of the one
`v1_hmda2022` row its case study quotes, and that source is still `local_only` with
`permits_redistribution: unknown` — so the sentence above applies to it.

```bash
make pages-authorized     # asks the ledger; exit 1 today, with the reason
```

The artifact declares which sources it needs approved in
[`PUBLICATION_REQUIREMENTS.json`](papers/unified-report-html/PUBLICATION_REQUIREMENTS.json) —
per *source*, because per *file* is the mistake that cost a history rewrite — and
[`tools/pages_authorized.py`](tools/pages_authorized.py) refuses on a missing declaration, an
unknown source id, an unresolved decision, or a licence that is merely not-negative. A gate
that cannot evaluate itself does not permit.

**To publish:** record the licensing decision in
[`benchmarks/registry/distribution.yaml`](benchmarks/registry/distribution.yaml) after the
review it requires. The gate then opens by itself — verified by simulation — and
[`tests/test_pages_gate.py`](tests/test_pages_gate.py) fails on that same change, so the
decision has to be accompanied by a deliberate update to this section rather than absorbed
silently. Nothing in the tooling decides the licence.

Six tests hold the capability shut: removing `needs: authorize`, gutting the authorize job,
or adding a `push` trigger each fail the build (all three verified by trying them). The
workflow is `workflow_dispatch`-only so publication cannot be a side effect of merging. **One
limit it cannot cover:** GitHub's repository *Settings* → "Deploy from a branch" would serve
the tree with no gate at all. No test in a repository can see that switch — leave it off.

## Status

Per-study state is generated from the registry — see [`studies/README.md`](studies/README.md).
The table below is the narrative summary.

| Track | Main artifact | Honest status |
|---|---|---|
| **Act I — specialization** | [Paper A](papers/finetuning-specialization/benchmark_chooses_the_winner.pdf) | **Complete** clean-v2 retrospective estimate (4 checkpoints × 5 seeds); conditional on this fixed panel, not universal or confirmatory. |
| **Act II — composition** | [Paper B](papers/base-adapter-composition/compose_dont_tune.pdf) | **Retrospective pilot complete.** No separately locked prospective run; controls remain roadmap items. |
| **Act III — mortgage depth** | [frozen benchmark](mortgage-benchmark/benchmark/v1_hmda2022/) | **994-row synthetic benchmark + four-base baselines complete.** LLM-judge / policy-card labels, *not* SME-adjudicated. |
| **Act III — ExpGuard breadth** | [scores](artifacts/expguard_external/) + [evaluator](experiments/eval_expguard_external.py) | **Complete** four-checkpoint base eval on 2,275 finance/health/law prompts; text-free scores committed; tuned comparison is future work. |
| **Recipe control — KL-SFT** | [scores](artifacts/klsft_v1/) · [study](studies/klsft/) | **Complete** retrospective four-checkpoint control (β = 0.5, 1.0; 5 seeds), no interval attached. Recovers **+0.061** transfer at a **−0.035** represented cost — mitigation within the SFT family, not restoration. |
| **Preregistered — starting-type adaptation** | [artifacts](artifacts/starting_type_adaptation_v1/) · [study](studies/starting-type-adaptation/) | **Complete but contract-drifted.** The only analysis-preregistered study (10 checkpoints, 6 families, 6 released vendor guards): RQ1 supported, RQ2 **not supported**. No final `LOCK.json` and the authoring-config hash no longer matches; resolve the drift before any new claim. |
| **Paper C — specialize-then-align** | [study](papers/paper_c/specialize_then_align/) · [package](studies/paper-c-specialize-align-mortgage-v1/) | **Stopped after its pilot.** Its result is an identifiability finding: with a three-action head and gold-based adjudication, the two candidate-source inventories were 98% byte-identical and the primary contrast was unidentified. No primary panel, no sealed cohort, no claim. |

Acts I and II are reproducible but **retrospective** (their sources were inspected during development).
The report keeps retrospective, external-expert, and LLM-judge evidence in separate tiers and never pools
them, and makes no causal, universal, deployment, legal, or fair-lending claim.

---

## Headline results (all reproducible from committed scores)

**Act I — LoRA-SFT specializes.** SFT lifts every checkpoint to ≈0.98 represented-source macro-AP
(**+0.323** on average) but changes held-out **transfer** by only **−0.059** on average — hiding opposite
per-checkpoint signs (SmolLM2 **+0.040** … Qwen3-4B **−0.150**). This is an *attractor*: post-SFT scores
collapse to a benchmark-fixed endpoint (transfer 0.807 ± 0.024), so "stronger bases specialize more" is
arithmetic (Δ slopes −1 in the base). At a 5% calibration-FPR target, transfer false alarms nearly
quadruple (pooled **4.3% → 17.0%**) and HarmBench recall falls (**78% → 60%**).

**Act I, read at an equal false-alarm budget.** The apparent consolation — "at least transfer recall rose
+0.06" — was bought with alarms, and does not survive a fair comparison. Give each tuned guard the
threshold at which its transfer false-alarm rate *matches its own base's* and the gain does not shrink, it
**reverses on all four checkpoints**: transfer recall **0.517 → 0.217** (−0.300) and HarmBench recall
**0.780 → 0.203** (−0.577), stable across the three quantile conventions tried (panel mean −0.300 to
−0.290). At an equal budget the tuned guard catches less than half of what its own untuned base catches
off-source. This needs no GPU and no pinned environment — rethresholding is ranking arithmetic on the same
committed `score_raw`/`gold` columns — so it is byte-checked like any other covered artifact, and the
emitter reproduces every published unequal-rate value first, which is how we know the two tables read the
same scores.

**Act II — composition recovers transfer.** Averaging the base's and SFT guard's calibrated scores lifts
transfer over SFT for all four checkpoints (**+0.076**) as an ensemble diversity gain — recovery, not
dominance (it can dip below the untuned base), and it restores no transferable threshold.

**Act III — domains.** The frozen [v1_hmda2022](mortgage-benchmark/benchmark/v1_hmda2022/) mortgage
benchmark (994 dual-labeled `G×D` rows; the load-bearing **G0/D1** stratum + a protected-context fairness
gate) shows zero-shot mortgage-policy AP of **0.67–0.85** — only 0.12–0.30 above this split's **0.555
chance floor**, with five of six pairwise CIs overlapping, so it is read as a direction and not used to
rank guards. The protected-pair gap of **0.000–0.183** rests on three pairs and saturates: Qwen3-4B's
0.000 becomes **0.80** on the raw-margin scale, second largest on the panel, so it is not a fairness
ranking. On external ExpGuard (2,275 expert-annotated prompts), all four base guards rank violations well
zero-shot (AP **0.88–0.96**) — and the best is **SmolLM3-3B (0.956), not the largest model**, a different
winner than the mortgage benchmark picks. The recurring character is Qwen3-4B: strongest base, specializes
most, helped least by composition, yet *numerically* the best-ranking zero-shot mortgage guard —
*the ranking flips with the benchmark.* Its fairness behaviour is a separate question this instrument, at
three pairs and on a saturating scale, cannot answer.

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
- **There is a measured noise floor of ≈0.015 mean / 0.029 worst-case transfer macro-AP.** The KL control's
  β = 0 arm is an accidental repeat of Act I — same recipe, manifest, seeds and scorer, different execution
  environment — and it does not land on the same number. Effects at or below that floor are unresolved:
  composition's **+0.017** aggregate edge over the *base* and KL-SFT's **+0.004** for SmolLM2 are both
  inside the envelope. The bootstrap intervals resample rows and seeds; they do **not** capture this term,
  so they are narrower than a full reproduction would be. Act I's **+0.32** represented gain, the
  matched-budget **−0.300** collapse, and composition's **+0.076** over SFT are comfortably above it.
- ExpGuard reports four paired comparisons with **no multiplicity adjustment**, and the one interval that
  excludes zero does so by +0.0026 — under a Bonferroni split across the three verticals it would not.
  Health *leans* against a tie; it is not a resolved ranking.
- The mortgage benchmark is synthetic and policy-card-consistent, not SME-adjudicated; its G1/D0 quadrant
  is empty. Ranking recovery does not imply threshold/calibration transfer. v1 contains **no protected pair
  on which a violation is scored**, so no controlled protected-trait contrast exists in it — the worked
  case-study rows differ in fact sheet, domain, cited cards and request type, and illustrate a possible
  surface-form effect rather than measuring one.
- The public Paper A cache cannot independently rehash omitted adapter bytes / full run metadata.

## Citation & license

[CITATION.cff](CITATION.cff) cites the unified report — its recorded subtitle still reads
"High-Compliance *Regulated* Domains" where the report's own title says "Business Domains".
Repository code and original content are
[Apache 2.0](LICENSE); third-party datasets/models retain their own licenses. Review the mortgage
[DATA_CARD.md](mortgage-benchmark/benchmark/v1_hmda2022/DATA_CARD.md) before redistributing generated
prompts.
