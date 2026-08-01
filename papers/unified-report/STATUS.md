# Unified report — build status

> **Paper C, as of 2026-08-01.** Paper C contributes **nothing** to this report and is not
> integrated into the manuscript at any evidence tier. Its lineage, as recorded in
> `studies/registry.yaml` — which is the authority, not this file — runs:
> `paper_c_reference_centering` (stopped) → `paper_c_matched_dpo_scaffold` (never run; the v2
> protocol candidate `docs/paper-c-prereg-v2.md`, now superseded) →
> `paper_c_specialize_align_mortgage_v1` (**stopped after a 44-cell pilot**, with
> `paper_c_sta_study_package_v1` as a migrated copy of the same study, not a second one). The
> successor's outcome is an *identifiability* finding, not a result: with a three-action head,
> gold-based adjudication, and structured fields derived from gold, the two candidate-source
> inventories came out 98.9%/97.7% byte-identical and the primary contrast was unidentified. The
> 66-cell primary panel was never authorized and all five readiness gates remain false;
> `claim_authorization: false` on every entry in the lineage. Both successor suites are declared
> `expected_fail` (the candidate lock binds live source bytes). The GPU runbook formerly recorded
> here must not be used.
>
> An earlier version of this banner described the v2 scaffold as "the current protocol candidate"
> and listed its gates as the open work. That was two studies out of date.

Snapshot of what is done / pending for the merged report. Nothing here fabricates numbers; pending
pieces await their locked runs. Where this file and `studies/registry.yaml` disagree, the registry
wins — it is validated in CI by `tools/validate_registries.py`, and this file is not.

## Present in the checkout

- **Paper C lineage (all superseded or stopped).** `docs/paper-c-prereg-v2.md` and its development
  plan / code design define the never-run matched VerdictCE/PairCE/DPO scaffold; the v1
  preregistration remains as a superseded amendment record. Both are historical: the active
  descendant, `papers/paper_c/specialize_then_align/` (plus its migrated copy under
  `studies/paper-c-specialize-align-mortgage-v1/`), was itself stopped after the disjoint pilot. See
  the banner above and `studies/registry.yaml`.
- **Current report scaffold.** `unified_report.tex` — *Safety Benchmark Gains Do Not Guarantee Safety Transfer:
  A Comprehensive Study of Fine-Tuning Small Language Model Safety Guards for High-Compliance and General Safety Domains* —
  is organized around four research questions (Q1 specialization, Q2 composition, Q3 regulated
  domains, Q4 the hosted-frontier comparison), with the original Act I/II/III labels retained in the
  section titles for continuity. Table 1 is the claim ledger: every headline number with its estimand
  and evidence tier. Paper C v2 is not a question or a placeholder in the manuscript; it can be
  integrated only after locked evidence exists.
- **Reproduce harness.** `reproduce.py` + `make verify` byte-checks the evidence currently integrated
  in the manuscript without rewriting the tree; `make regenerate` is the separate write path. It has
  no Paper C v2 hook because no Paper C score artifact exists.
- **ExpGuard eval code.** `experiments/eval_expguard_external.py` — scores the 4 checkpoints on ExpGuard
  (finance/health/law) via the canonical guard head; commits only text-free per-row scores.
- **Paper C v1 code (superseded).** `experiments/paper_c_preference.py` and
  `experiments/run_paper_c_objective.py` are retained only as compatibility/history markers. They did
  not preserve the Paper A prompt/truncation/data-order/scoring contract; the runner is intentionally
  non-runnable.
- **Paper C v2 scaffold.** Exact offline primitives, lock/train/selected-score/analyze modules, and
  focused unit tests exist. Finalization requires inventories that cannot yet be produced end to end.

## Done and committed (cont.)

- **ExpGuard base eval — COMPLETE (Act III breadth).** All 4 checkpoints scored zero-shot on ExpGuard
  (2,275 rows; finance/health/law) on a spot L4 GPU and independently re-run on Apple MPS (agree to
  3–4 decimals). Overall AP: SmolLM3-3B 0.956, Qwen3-4B 0.951, Qwen2.5-1.5B 0.921, SmolLM2-1.7B 0.883 —
  the best domain guard is *not* the largest model. `tab:expguard` + `fig:expguard-domains` are in Act III;
  text-free per-row scores committed; `reproduce.py --check` passes. NB: the eval now stores the raw
  decision margin (not the saturating sigmoid), which is what makes AP reproducible from committed scores.
  The *tuned* (base-vs-SFT) ExpGuard comparison has since been run and is central to Q4
  (`tab:frontier`), together with the hosted-frontier and scale-ladder arms; what remains open is a
  dual-labeled finance/health construct with expert sign-off.

## Pending gates, if Paper C is ever resumed

These were written against the never-run v2 scaffold. They are **not** the current blockers: the
successor study reached a pilot and stopped on an identifiability problem that no amount of
scaffolding work resolves. Kept as a record of what the v2 protocol still lacked.

- add lock-bound reference-margin and multi-checkpoint Stage-2-development scoring plus inventory builders;
- add tiny-model integration and pass a real three-objective GPU smoke (the focused/full CPU suites pass);
- regenerate and validate 20 shared Stage-1 adapters (only scores are present now);
- freeze Stage-2 family split, uncertainty selections, and reference margins;
- close scorer/analyzer diagnostics, cache, selection-inventory, and reproduction contracts;
- acquire a genuinely sealed cohort if confirmatory claims are desired.

The blocker that actually stopped the successor is upstream of all six: with gold-derived structured
fields and gold-based adjudication, the specialist and generalist candidate inventories were
98.9%/97.7% byte-identical, so the primary contrast was not identified. Resuming means changing the
*design*, not finishing the scaffold.
