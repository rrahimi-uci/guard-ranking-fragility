# Paper B — plain-language edition ("Compose, Don't Tune?")

A teaching-oriented, plain-language version of the formal Paper B draft
([`../paper-b/compose_dont_tune.tex`](../paper-b/compose_dont_tune.tex)), for readers with
basic stats + fine-tuning knowledge — same numbers, gentler exposition, with a glossary.

- [compose-dont-tune-simplified.md](compose-dont-tune-simplified.md) — the plain-language paper.
- [GLOSSARY.md](GLOSSARY.md) — the terms (macro-AP, calibration, transfer, retrospective vs prospective, …).

## Same numbers, same honesty

Every quantitative value here is copied from the formal draft's generated macros
(`paper-b/generated/pilot_macros.tex`), which are produced fail-closed from Paper A's
lock-verified clean-run **v2** composition result
(`artifacts/paper_a_sft_v2/analysis/composition/composition.json`, lock `cabc8dee…`). Nothing
is hand-computed here.

**This is a retrospective feasibility result, not a confirmatory claim.** The transfer
benchmarks were held out of fine-tuning but were *inspected during method development*, and no
separately-locked prospective study has run. So the paper reports a *conditional, fixed-panel
estimate* and a *protocol* for a future confirmatory study — it does **not** claim Pareto
dominance, a universal remedy, a mechanism, or deployment readiness. The plain-language edition
keeps that framing intact; it does not upgrade the certainty of any statement.

## Rendering to PDF

The markdown renders to a print PDF the same way the paper-a-simplified plain edition does
(GitHub-flavored markdown → styled HTML → headless-Chrome print). See the repo's report-to-pdf
workflow; the `.md` is the source of truth.
