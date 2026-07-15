# Plan — Unified Research Report (final)

> **What this is.** The complete plan for one accessible, defensible research report that unifies the
> guard-specialization trilogy **and** two new, honestly-scoped experiment tracks: an **objective axis**
> (SFT vs DPO vs GRPO) and a **four-domain** high-compliance case study (law, finance, health, mortgage).
> It has three parts: **A** the experiment program that must run first, **B** the report structure, **C**
> how we keep it high-quality and defensible. Numbers for the *existing* studies are verified against
> source; numbers for the *new* experiments are marked "pending the locked run" — none are invented.
> This is for your review; nothing costs money until you approve Part A.

---

## 0. Executive summary & recommendation

We cannot honestly "merge in" DPO/GRPO or finance/health/law results, because **they do not exist as
auditable artifacts** (verified: `artifacts/` holds SFT runs only; ExpGuard was never run; the broad-study
DPO/GRPO summaries lived in the now-deleted, never-committed `notebooks/outputs/`). So this is a small
**research program → then merge**, run in a **staged, gated** way:

- **Step 1 — free, ~1 hr, no approval:** verify **ExpGuard reachability** (the hard gate on the
  finance/health/law breadth) and finalize this plan. Zero cost; prevents spending into a dead end.
- **Step 2 — the minimal-honest program (~$15–30 spot, ~2 days parallelized):** run the **DPO+GRPO
  objective axis** (× 4 checkpoints × 5 seeds) and the **ExpGuard base-only eval** (finance/health/law),
  reuse the finished **mortgage** benchmark, then write the merged report.
- **Step 3 — deferred (not now):** the full objective sweep (+KTO/ORPO/β) and *building* dual-labeled
  finance/health benchmarks (~$70–350+, ~1 week+, SME/HIPAA-gated). Only if Step-2 results justify it.

**Why:** Step 2 delivers everything the generalized title promises — an objective axis + four
high-compliance domains — at ~1/10th the cost/time of the full build, sidesteps the HIPAA/SME gate
(we use ExpGuard's *existing* health subset, we do not build one), and keeps the report's integrity
intact via an honest "one domain built deep + three external verticals for breadth" framing.

---

# PART A — The experiment program (Phase 0, precedes the report)

## A1. Scope decision

| Dimension | In scope (Step 2, minimal-honest) | Deferred (Step 3) |
|---|---|---|
| Objectives | base, SFT (done) + **DPO, GRPO** (× 4 ckpt × 5 seeds = 40 new adapters) | +KTO, +ORPO, +β sweep, IPO/RLOO |
| Domains | **mortgage** (dual-label, done) + **finance/health/law** via ExpGuard (base-only eval) | *built* dual-label finance/health benchmarks (magen + SME) |
| Evidence tier | retrospective/estimation-only (objective axis on the shared manifest); **prospective** external check (ExpGuard, lock-before-scoring) | prospective locked cohort for the objective axis |

## A2. Track 1 — Objective axis ("Paper C": SFT vs DPO vs GRPO)

**Goal.** Turn the SFT-only campaign into a controlled head-to-head of *training objectives* on the
identical 4-checkpoint panel, same 1,200-row decontaminated manifest, same seeds, same single-token
scorer/analyzer — testing **H1** (does the objective order the represented-vs-transfer trade-off by how
far it moves the model from base?) and **H2** (does base+adapter composition recover transfer in
proportion to that movement?), while **pre-registering GRPO's likely single-token null**.

- **Run matrix (minimal):** {DPO, GRPO} × 4 checkpoints × 5 seeds = **40 new LoRA adapters**; reuse the
  existing SFT arm + base. Scoring: 40 bundles on the ~3,308 locked eval rows; composition evaluated at
  analysis time (≈ free).
- **New code (none of this exists yet — `experiments/` is SFT-only):**
  `experiments/paper_c_preference.py` (deterministic verifiable preference/reward builder from the frozen
  manifest; graded logit-margin reward for GRPO to avoid single-token gradient starvation),
  `experiments/run_paper_c_objective.py` (generalizes `run_paper_a_sft.py` with `--objective`, swapping in
  TRL's DPO/GRPO trainers), a small `paper_a_common.py` patch (parametrize the hard-coded `sft` literal so
  an `objective` dimension exists without disturbing Paper A), `experiments/eval_paper_c.py` +
  `analyze_paper_c.py` (thin reuse of the Paper A scorer/analyzer + a base↔tuned movement report),
  `experiments/lock_paper_c.py` (new `artifacts/paper_c_objective_v2/` lock, reusing Paper A's manifest by
  hash), and **`docs/paper-c-prereg.md`** (written and locked *before* the run).
- **Reused unchanged:** the frozen manifests (consumed by SHA-256, no new download), `guard_research.prompts`
  (byte-identical prompt rendering), the single-token logit-diff head, the tie-aware macro-AP + hierarchical
  paired bootstrap, the composition machinery, and the 20 SFT adapters + 4 base scores already computed.
- **Honesty caveats it will carry:** retrospective/estimation-only on the shared manifest; **GRPO
  single-token null pre-registered** (a bounded negative result is a contribution); conditional on the fixed
  panel; DPO/GRPO may not beat SFT — the value is the controlled comparison. (KTO is the arm to add if you
  want a likely-positive result too.)

## A3. Track 2 — High-compliance domains (law, finance, health, mortgage)

- **Mortgage** — *done*: our dual-labeled (G×D), HMDA-grounded benchmark (994 rows), with the G0/D1 payload
  and the protected-pair fairness gate.
- **Finance / health / law** — via **ExpGuard** (`6rightjade/expguardmix`, config `expguardtest`), an
  **expert-annotated** specialized-domain moderation set. This is the "Optional External Validation: ExpGuard"
  that Paper A pre-planned but did not run. Port `legacy/experiments/expguard_eval.py` onto the audited
  `guard_research` scorer, **author + hash a protocol lock before scoring** (making it a genuine *prospective*
  external check), then run **base-only across the 4 checkpoints** (feasible on the Mac, ~0 GPU) → aggregate
  + per-domain (finance / health / law) AP/AUROC. Extend to paired base-vs-tuned by re-scoring the Track-1
  DPO/GRPO adapters (cheap, inference-only) once they exist.
- **Honest asymmetry (stated in the report):** mortgage is a **dual-label G×D** construction we built;
  finance/health/law are an **external, single-label** moderation set. Different shapes → different evidence.
  A labeling-quality nuance actually *favors* the external set: ExpGuard is **expert-annotated**, whereas the
  mortgage labels are **LLM-judge**. Both facts stated plainly.

## A4. Cost / time / price (parallelized)

| | GPU work | Wall-clock (parallelized) | Approx cloud+API $ |
|---|---|---|---|
| **Minimal-honest (Step 2)** | ~32 L4-hr ≈ 16 A100-hr (40 adapters + scoring) + ExpGuard base ≈ 0 | **~2 working days** (dev ~1 day ∥ ExpGuard base; GPU wave ~4–6 hr on 4–8 GPUs; analysis + report ~½–1 day) | **~$15–30** (spot); GPU is a few hours of it |
| **Full (Step 3)** | ~70 L4-hr ≈ 37 A100-hr | **~1 week+** (long pole = *building* dual-label finance/health + SME/HIPAA review) | **~$70–350+** (GPU $20–135 + LLM-API $40–200+/built domain) |

Rates (approx, us-central1): L4 `g2-standard-8` ~$0.85/hr on-demand · ~$0.30 spot; A100-40GB
`a2-highgpu-1g` ~$3.7/hr · ~$1.2 spot. **Spot cuts GPU ~60–70%** and the training is
checkpoint-restartable (preemption-safe). **A100-spot** is the sweet spot for the 3–4B tier.

## A5. Gates, blockers, prerequisites (in priority order)

1. **⚠️ ExpGuard access — the hard gate.** The dataset is future-dated vs the assistant's knowledge and the
   sandbox network is blocked, so reachability + gating + license + field/split/domain schema are
   **unverified**. Resolve in Step 1. If inaccessible → descope to **mortgage-only domain** + the objective
   axis, and frame finance/health/law as roadmap (title stays honest).
2. **Env not training-ready.** The venv lacks `transformers`/`trl`/`peft`/`accelerate`; install + **freeze**
   the pinned stack, and (because the software-version gate fails a final run on any mismatch) create the
   Paper C lock *after* freezing.
3. **Adapter weights absent.** `artifacts/` has SFT *scores*, not weight dirs → base-only ExpGuard is free;
   paired SFT-on-ExpGuard needs regenerating the 20 SFT adapters (~3–4 A100-hr) unless taken from Track 1.
4. **GCP ops.** `gcloud auth login` (expired before) + a region with A100/L4 capacity (L4 stockouts seen);
   recommend A100-40GB spot.
5. **Clean-git lock gate.** New trainer/analyzer files must be committed before a final run (they change
   `execution_sources`), so Paper C gets a fresh lock + artifact root by design (Paper A untouched).
6. **No SME needed in the minimal path** (we use ExpGuard's existing health subset; SME/HIPAA only bites the
   deferred *build*).

---

# PART B — The final report

## B1. Title (recommended)

> **The Benchmark Chooses the Winner:**
> *Honestly Measuring, Tuning, and Composing Small Safety Guards Across Objectives and Four High-Compliance Domains (Law · Finance · Health · Mortgage)*

Keeps Paper A's memorable anchor and the co-produced-verdict thesis; "Tuning ... Across Objectives" now
covers the SFT/DPO/GRPO axis; "Four High-Compliance Domains" names the breadth honestly. *(Decision: full
subtitle vs a shorter "...Across Objectives and High-Compliance Domains".)*

## B2. Unifying thesis + arc

**Thesis (unchanged, now stronger):** a small guard's benchmark score is **not a fixed property of the
guard** — it is *co-produced by the benchmark you report it on*. Honest evaluation therefore compares a
guard to its **own base**, separates **represented-source** from **held-out transfer** (+ stress), holds
across **fine-tuning objectives**, and — in regulated domains — needs **domain-grounded** evaluation.

**Four-Act arc** carried by the teaching backbone; the same 4 checkpoints recur throughout (Qwen3-4B as the
recurring character):
- **Act I — Specialize:** SFT raises represented-source AP but not transfer (Paper A).
- **Act II — Does the objective matter?** SFT vs DPO vs GRPO: does a different objective specialize less? (new; pre-registered GRPO null).
- **Act III — Compose, don't (re)tune:** recover transfer by averaging base + adapter, without retraining — and does it help across objectives? (Paper B + H2).
- **Act IV — When the domain changes everything:** mortgage (deep, dual-label) + finance/health/law (external breadth) — does the pattern recur, and can a general guard even see domain violations?

## B3. Section outline

- **0. Abstract + Study-Status box** — whole arc in one paragraph; both honesty flavors up front.
- **1. Introduction — why "is this guard good?" is the wrong question** — the leaderboard trap; the four questions; the shared panel; "what is new / not."
- **2. Background you'll need** (teaching) — define once, with mini-examples: guard/logit-diff head, LoRA-SFT, **objectives (SFT/DPO/KTO/ORPO/GRPO), verifiable preference/reward, RLVR**, AP/macro-AP, represented vs transfer, calibration/operating point/FPR, paired bootstrap/estimand, provenance-lock, the mortgage primer + G×D, **ExpGuard / domain-moderation**.
- **3. Shared experimental setup** — panel, LoRA recipe, dataset roles, metric, provenance; the **objective dimension**; the **ExpGuard external protocol** (lock-before-scoring). [panel+recipe table; datasets+roles table]
- **4. Act I — Fine-tuning specializes (SFT)** — rep Δ **+0.3234** [+0.2647,+0.3690] vs transfer **−0.0589** [−0.0837,−0.0321]; per-checkpoint heterogeneity (Qwen3-4B −0.1499); **15/20** seeds specialize. [`tab_primary_gen`, `specialization_plane.pdf`, `tab_sensitivity_gen` top]
- **5. Act I at a deployable threshold** — rep TPR **13.0→76.9%**, transfer FPR **8.1→15.5%**, HarmBench recall **78.0→60.0%**. [`tab_sensitivity_gen` bottom]
- **6. Act II — The objective axis** — SFT vs DPO vs GRPO (+KTO/ORPO if run); H1 movement order; the **pre-registered GRPO single-token null**. *[NEW tables from the locked Paper C run — pending; no numbers invented.]*
- **7. Act III — Compose, don't (re)tune** — operator ½·C_b + ½·C_a; composition vs SFT rep **−0.019** / transfer **+0.076**; per-checkpoint recovery (+0.027…+0.120 vs SFT); ranking≠calibration (11.4% FPR). Plus **does composition recover DPO/GRPO transfer too?** (H2). [`pilot_summary_table`, `pilot_per_model_table`, `pilot_operating_point_table` + *NEW objective×composition table, pending*]
- **8. Act IV — High-compliance domains** — **mortgage** (G0/D1 payload, pipeline, AP·D **0.67–0.85**, Δ_context **0.00–0.18**, fairness gate) + **finance/health/law** (ExpGuard per-domain specialization/transfer). [`pipeline.png`, `tab:composition`, `baseline_table` + *NEW ExpGuard per-domain table, pending*]
- **9. Synthesis — one lesson across objectives, a remedy, and four domains.** [recap matrix]
- **10. The practitioner's decision guide** — build → test → repair → evaluate, each step annotated with its caveat.
- **11. Honesty ledger + what these results do NOT establish** — the evidence tiers side by side.
- **12. Validation roadmap — from estimates to confirmatory use.**
- **13. Conclusion.**
- **Appendix** — unified glossary; per-seed data (`tab_seed_values_gen` + the new objective seeds); provenance/locks.

## B4. Tables & figures inventory

Existing (verified, `\input` from source): `tab_primary_gen`, `tab_sensitivity_gen`, `specialization_plane.pdf`
(Act I); `pilot_summary_table`, `pilot_per_model_table`, `pilot_operating_point_table` (Act III);
`pipeline.png`, `tab:composition`, `baseline_table` (Act IV). **New (produced by Phase 0, then `\input`):**
a Paper C objective table (macro-AP represented/transfer by objective, with seed bars), an
objective×composition table (H2), and an ExpGuard per-domain table (finance/health/law AP/AUROC, base
±SFT/DPO/GRPO). All new numbers come from the locked runs — placeholders until then.

## B5. Glossary additions
DPO / KTO / ORPO / GRPO; preference pair (chosen/rejected); verifiable reward / RLVR (no learned reward
model); reference-KL anchoring; group-relative advantage; ExpGuard / specialized-domain moderation;
expert-annotated vs LLM-judge labels; AUROC (alongside AP).

## B6. Build feasibility (LaTeX — verified)
Single `article` reusing the `finetuning-specialization-simplified` **preamble block** (teaching
`tcolorbox` boxes, TOC, booktabs, amsmath, graphicx, cleveref). **Add** `\usepackage{float}` (imported
`[H]` tables) and copy the `\draftwarning` macro + colors from Paper B. Paper A tables are bare tabulars
(wrap them); Paper B + mortgage + the new tables are full `[H]` floats (`\input` directly). **No macro
collision** between `results_macros_gen` and `pilot_macros`; the new Paper C macros must use a fresh
namespace (e.g. `\Obj*`). Union `refs.bib` dedups 6 shared keys; companion self-citations become internal
cross-references. `\graphicspath` to all figure dirs (or copy in). Every number `\input`; nothing retyped.

---

# PART C — How we make it high-quality and defensible

This is the core of the ask. The report earns trust by construction, not by prose:

1. **Nothing fabricated or unreproducible.** Every figure `\input`s from a generated table bound to a lock;
   no number is retyped. The deleted broad-study DPO/GRPO summaries are **not** used — we regenerate cleanly
   under a lock. If a value can't be produced from a locked run, it doesn't appear.
2. **Pre-registration (the strongest move).** `docs/paper-c-prereg.md` states H1 (movement order), the
   **GRPO single-token null**, and H2 (objective×composition) **before** the run, and is **bound into the
   Paper C lock** — so we cannot HARK (hypothesize after results are known). The GRPO null is reported as a
   pre-registered bounded negative result, which is a contribution, not a disappointment.
3. **Provenance locks + fail-closed.** A fresh `artifacts/paper_c_objective_v2/` lock reuses Paper A's
   manifest **by hash** (guaranteeing like-for-like comparability of SFT/DPO/GRPO), and the ExpGuard
   **protocol is authored and hashed before any scoring** — turning that appendix into a genuine
   *prospective* external check, a strictly stronger evidence tier than the retrospective arms.
4. **Evidence tiers kept visibly separate (never pooled).** Three flavors, each labeled per result:
   (a) SFT/DPO/GRPO on the shared manifest = **retrospective, estimation-only, precision-focused**;
   (b) ExpGuard = **prospective external**, expert-annotated; (c) mortgage = **construction + baseline**,
   LLM-judge (not SME). Two hard rules: never pool retrospective with prospective numbers; never upgrade any
   number to a causal, universal, or fair-lending claim.
5. **Comparability by construction.** Identical panel, manifest, seeds, prompt bytes, single-token scorer,
   tie-aware macro-AP, and hierarchical paired bootstrap across every objective — so an objective difference
   is an objective difference, not a confound.
6. **The mortgage-deep vs external-breadth asymmetry stated plainly** — one domain built in depth
   (dual-label, fairness gate, LLM-judge) + three external verticals for breadth (single-label, expert
   labels). We do not imply four uniform artifacts.
7. **Bounded, interval-reported claims.** Every point estimate carries its 95% interval; results are
   conditional on the fixed panel and datasets; "observed/estimated" on A/B/C deltas, "diagnostic /
   measuring stick" on mortgage, "external replication" on ExpGuard.
8. **Independent adversarial verification before finalize.** A dedicated pass re-checks every headline number
   against its source table and red-teams every claim for over-reach (the same discipline used to harden this
   plan — which caught two real LaTeX build gaps).
9. **Full reproducibility.** Released locks + per-row scores + generated tables + the exact analysis code let
   a third party re-run each number; the report ships the commands.
10. **An explicit "what this does NOT establish"** section + a validation roadmap, so caveats read as a path
    forward (SME adjudication for mortgage; a prospective uninspected cohort for the objective axis; building
    dual-label finance/health; accepting gated open-guard baselines).

### Reproducibility contract (every claimed number regenerates from committed code)

Audited against the current repo; the report and all new work must satisfy this end to end:

- **Numbers → tables:** every figure/number in the report `\input`s a committed *generated* table; nothing is hand-typed (prevents drift).
- **Tables → scores:** each generated table regenerates from **committed per-row scores** by committed analysis code — verified live for **Paper B** (`make -C papers/base-adapter-composition evidence-check`) and **mortgage** (`tools/reeval_from_scores.py`). **Paper A** regenerates from committed `artifacts/paper_a_sft_v2/scores/scores.parquet` via `analyze_paper_a_sft.py`, executed in the **lock-pinned software environment** — `LOCK.json` records exact python/torch/library versions and the analysis **fails closed** on any mismatch (confirmed: it refuses under the py3.14 dev venv). Reproduction recreates the pinned env per `docs/reproducibility*.md`. `paper-verify` confirms the committed tables are byte-identical to the analysis outputs.
- **Scores → weights/data:** all training/eval code is committed (`run_paper_a_sft.py`, `magen/`, and the new Paper C runner + ExpGuard eval); raw third-party rows are referenced by pinned identifier + revision + content hash under their licenses, not redistributed.
- **New work held to the same bar:** Paper C (DPO/GRPO) and the ExpGuard eval each ship committed code + a lock + committed per-row scores + a documented `reproduce` step, so their headline numbers regenerate **without a GPU** (only re-training needs one).
- **One entry point — `make reproduce` / `papers/unified-report/reproduce.py`.** A single function regenerates **every** table/figure/number in the merged report from committed per-row scores, by calling each study's analysis in turn: Paper A (`analyze_paper_a_sft.py`, pinned env), Paper B (`build_pilot_artifacts.py`), mortgage (`reeval_from_scores.py`), Paper C (`analyze_paper_c.py`), ExpGuard (`eval_expguard_external.py --from-scores`). It writes the `\input`-ed generated tables and asserts byte-identity with the committed report tables (a `--check` mode fails closed on drift). It needs **no GPU and no network** — only committed scores + the pinned env.
- **ExpGuard redistribution model (gated dataset):** we do **not** commit ExpGuard prompt text (it is gated/licensed). We commit only our **per-row guard scores** (the `z_unsafe−z_safe` logits) keyed by a text-free row id/hash, plus the metrics code; `reproduce` recomputes per-domain AP/AUROC from those committed scores. Regenerating the scores from scratch requires ExpGuard access (HF token + accepted license) — the same "identifiers+hashes, not raw text" model as Paper A.
- **Known friction to fix:** the root `Makefile` defaults `PY=python3` (system interpreter, no numpy) — reproduction uses the pinned env / `.venv`; new Makefile targets will auto-detect the venv (as paper-b's already does).

**Validation roadmap (from estimates to confirmatory):** (i) SME-adjudicate a stratified mortgage subset
(report Fleiss-κ); (ii) lock a genuinely uninspected prospective cohort for the objective axis; (iii) add the
SFT+SFT and WiSE-FT composition controls; (iv) build dual-label finance/health with expert sign-off; (v)
score the gated open guards (Llama Guard 3, WildGuard) once licenses are accepted.

---

## Open decisions (please confirm)

1. **Go / no-go on Step 1** (free ExpGuard check + finalize this plan). *Recommend: go.*
2. **Objective pair:** **{DPO, GRPO}** (recommended — headline ordering + pre-registered null) or add **KTO** (likely-positive binary-label arm).
3. **Seeds for the first cut:** **5** (full comparability with Paper A) or 3 (faster, extend later).
4. **GPU:** **A100-40GB spot** (recommended) vs L4; confirm GCP region + budget cap (≈ $30 covers minimal).
5. **ExpGuard fallback:** if inaccessible, proceed **mortgage-only domain + objective axis** (title → "…and a High-Compliance Domain") and list finance/health/law as roadmap. *Recommend: yes.*
6. **Title:** confirm the recommended title or the shorter variant.
7. **Format:** LaTeX → PDF (recommended). Markdown fallback available.

## Risks & mitigations
- **ExpGuard gated/absent** → Step-1 check first; descope to mortgage-only if blocked (no fabrication).
- **GRPO null read as failure** → pre-registered and framed as a bounded negative result.
- **Env / TRL version drift** → freeze the stack, smoke-test trainers, lock after freezing.
- **GCP capacity** → A100 spot + checkpoint-restartable training; retry across regions.
- **Scope creep to the full build** → Step 3 is explicitly deferred; minimal path is self-contained.
- **Number drift in the report** → `\input`-only + the adversarial verification pass.

## Where things will live
- `experiments/` — new Paper C trainer/eval/analyze/lock; `artifacts/paper_c_objective_v2/` — its locked outputs.
- `artifacts/expguard_external/` — the locked ExpGuard scores + text-free index.
- `papers/unified-report/` — the report (own `Makefile`, deduped `refs.bib`, `figures/` copies).
- `docs/paper-c-prereg.md` — the pre-registration (bound into the lock).
