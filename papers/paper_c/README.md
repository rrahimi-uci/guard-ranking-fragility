# Paper C: Reference-Centered Preference Training for Safety Guards

This directory is the isolated development workspace for Paper C. New code,
configuration, cloud launch material, generated artifacts, tests, and manuscript
sources live here. Nothing in this project writes to another repository path.

Working title:

> **What Does Reference Centering Buy a Safety Guard? A Matched SFT, PairCE, and DPO Study**

## Scientific question

For a binary, one-token safety guard, DPO with label-derived preferences reduces
to a reference-centered signed-margin loss. The study asks whether that centering
improves represented-source and dataset-held-out transfer beyond:

1. continued full-vocabulary verdict cross-entropy (`VerdictCE`), and
2. the same pairwise logistic loss without reference centering (`PairCE`).

The primary factorial is:

```text
{VerdictCE, PairCE, DPO} x {uncertain, matched_random}
```

over four model checkpoints and five seeds. PairCE and DPO share `beta=0.1`.

The **primary estimand is the transfer gap at matched represented AP** across the
saved checkpoint ladder, not a single-checkpoint AP difference. Reference centering
begins as an approximately uniform 1.30x gradient rescale (see `PROTOCOL.md`), so a
fixed-step contrast cannot separate a better objective from a faster one, and on a
four-checkpoint panel it is underpowered besides. The frontier form is exactly zero
under a pure rescale, which turns that confound into the null hypothesis.

## Folder contract

- `config/`: candidate protocol and immutable model revisions.
- `src/paper_c/`: self-contained scientific implementation.
- `tests/`: CPU-only contract tests.
- `cloud/`: GCP preflight, packaging, and single-cell launch scripts.
- `manuscript/`: LaTeX paper and bibliography.
- `inputs/`: local, ignored copies of parent locks/manifests/adapters.
- `artifacts/`: local, ignored run outputs and locks.
- `build/`: generated PDF and build products.

The code rejects output paths outside this directory. External repository inputs
may be read only by `bootstrap-inputs`; they are copied into ignored `inputs/`
before development or GPU execution.

## Current status

The protocol, pure objective/sampling contracts, isolated configuration, GPU
runner, cloud runbook, and manuscript draft are executable. The current
superseding protocol-lock path is recorded in `STATUS.md`. No Paper C GPU result
or claim-bearing final lock exists yet.

## Local quick start

From this directory:

```bash
make test
make validate
make power      # design MDE; fails closed if the panel cannot detect its target effect
make paper
```

Before a GPU run:

```bash
make doctor
python -m paper_c bootstrap-inputs
```

`HF_TOKEN` is loaded from the repository `.env` only at process runtime and is
never copied into locks, bundles, logs, or metadata. Local loading refuses a
group/world-readable `.env`; its mode must be `0600`, or `HF_TOKEN` must be
injected directly. GCP provisioning uses a dedicated service account and Secret
Manager plus non-secret `PAPER_C_GCP_*` settings.

The lock sequence is `PROTOCOL_LOCK` (Stage 1) → `STAGE2_DESIGN_LOCK`
(candidate training) → `SELECTION_LOCK` (retrospective scoring) → a future
prospective child. No earlier lock authorizes a later evidence phase.

## Claim boundary

Development locks, smoke runs, and retrospective Paper A evaluation are not
confirmatory evidence. Finalization remains fail-closed until all Stage-1
adapters, reference margins, selections, checkpoint candidates, and development
scores are present and hash-bound. A separately sealed cohort is required for a
confirmatory claim.
