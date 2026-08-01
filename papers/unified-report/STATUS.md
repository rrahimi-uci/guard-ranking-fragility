# Unified report — build status

> **Paper C correction, 2026-07-25.** Paper C is unrun. Its original DPO/GRPO protocol and TRL
> runner are superseded; the old runner now fails closed. A partial v2 development scaffold exists,
> but there is no claim-bearing lock, adapter matrix, complete reference/dev scoring path, or result
> artifact. The current protocol candidate is `docs/paper-c-prereg-v2.md`. The GPU runbook formerly
> recorded here must not be used.

Snapshot of what is done / running / pending for the merged report, written during the autonomous
window. Nothing here fabricates numbers; pending pieces await their locked runs.

## Present in the checkout

- **Paper C protocol rewrite.** `docs/paper-c-prereg-v2.md`, the development plan, code design, and
  rationale define the candidate matched VerdictCE/PairCE/DPO study. The v1 preregistration remains
  as a superseded amendment record; the old unified plan remains available in Git history.
- **Current report scaffold.** `unified_report.tex` — *Benchmark Gains Do Not Guarantee Transfer:
  Fine-Tuning Small Language Model Safety Guards* —
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

## Pending Paper C gates

- add lock-bound reference-margin and multi-checkpoint Stage-2-development scoring plus inventory builders;
- add tiny-model integration and pass a real three-objective GPU smoke (the focused/full CPU suites pass);
- regenerate and validate 20 shared Stage-1 adapters (only scores are present now);
- freeze Stage-2 family split, uncertainty selections, and reference margins;
- close scorer/analyzer diagnostics, cache, selection-inventory, and reproduction contracts;
- acquire a genuinely sealed cohort if confirmatory claims are desired.
