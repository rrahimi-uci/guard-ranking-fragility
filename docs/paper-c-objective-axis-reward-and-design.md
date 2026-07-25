# Paper C — Matched SFT, PairCE, and DPO Design Rationale

> **Explanatory companion, not the normative protocol.** The claim-bearing rules live in
> [`paper-c-prereg-v2.md`](paper-c-prereg-v2.md); implementation gates live in
> [`paper-c-development-plan.md`](paper-c-development-plan.md); the executable contract is described
> in [`paper-c-code-design.md`](paper-c-code-design.md). The earlier five-objective SFT/DPO/KTO/ORPO/
> GRPO design was superseded on 2026-07-25 before a claim-bearing lock or run existed.

## 1. Why the study changed

The old design treated objective names as an explanatory axis and predicted an ordering among SFT,
DPO, and GRPO. Three findings make that design too weak for a standalone paper:

1. Recent guard research already applies hard-case DPO and directly compares preference/RL training
   regimes. DPO-for-guards and an SFT/DPO/GRPO scoreboard are no longer credible novelty claims.
2. In this repository's one-token binary head, the label-derived DPO pair contains no preference
   information beyond the class label.
3. The old implementation changed trainer, learning rate, completion support, dropout behavior,
   reference handling, and data order together. Equal nominal steps did not isolate an objective.

The revised paper asks a narrower and stronger question: **what does reference centering buy after
pairwise normalization and selected data are held fixed?**

## 2. The exact reduction

Let the evaluated score be

```text
s_theta(x) = z_theta(unsafe | x) - z_theta(safe | x)
```

and `y` be `+1` for unsafe and `-1` for safe. For `chosen=correct verdict` and
`rejected=wrong verdict`, the DPO log-ratio difference is exactly

```text
y * [(s_theta(x)) - (s_ref(x))].
```

The full-vocabulary normalizer cancels between the two verdict tokens. Consequently:

```text
DPO = softplus(-beta * y * (s_theta - s_ref)).
```

This is a reference-centered binary margin objective. It is useful, but its scientific meaning is
not "learning human preferences." It also does not mechanically enforce a realized KL radius. The
model may move little or far depending on gradients, optimization, data, and stopping.

## 3. Why PairCE is indispensable

Ordinary verdict SFT and DPO differ in at least two ways:

- SFT raises the gold verdict token relative to the full vocabulary.
- DPO raises the gold verdict relative to the other verdict and subtracts a reference margin.

Temperature-matched PairCE removes only the reference distinction:

```text
PairCE = softplus(-beta * y * s_theta).
```

The same locked `beta` is used in both margin losses. Using unscaled PairCE beside `beta=0.1` DPO
would confound reference centering with a tenfold gradient-temperature change. The matched form yields
an interpretable decomposition:

- `PairCE - VerdictCE`: two-verdict target support plus the locked pairwise-temperature effect;
- `DPO - PairCE`: frozen-reference centering effect;
- `DPO - VerdictCE`: total pipeline effect.

Without PairCE, an apparent DPO gain could simply reflect training directly on the same two logits
used by the scorer.

## 4. Why use a common Stage-1 SFT adapter

Modern guard DPO pipelines normally begin from a model that already follows the guard output
contract. A common Stage-1 adapter therefore answers the practical question while improving the
controlled comparison:

- all Stage-2 cells begin from byte-identical trainable adapter weights;
- the DPO reference is the exact shared starting policy;
- continued one-token VerdictCE is the fair "more training" comparator;
- Stage-2 effects are separated from learning the prompt and verdict format.

The current checkout contains Paper A scores but not the 20 adapter directories. Stage 1 must be
regenerated and revalidated; score artifacts cannot recreate weights.

## 5. Why factor example selection

GuardReasoner and DT-Guard report benefits from applying preference optimization to difficult or
unstable cases, but their pipelines change both the selected examples and the loss. Paper C separates
those choices.

Because this guard has exactly two scored actions, uncertainty is available without stochastic
rollouts. The normalized two-verdict entropy provides an exact, reproducible ranking. Within each
source and label stratum, Paper C compares the highest-entropy quarter with an equal-size locked
random sample from the remainder.

The factorial design can distinguish:

- whether uncertain examples are more useful for every objective;
- whether DPO specifically benefits from uncertain examples; and
- whether a reported hard-case DPO gain is actually a data-selection gain.

## 6. What is and is not matched

The primary analysis reports three views rather than pretending one notion of fairness is sufficient:

1. **Total effect under the frozen recipe:** what each objective produces after the same maximum
   updates and exposure.
2. **Target-matched effect:** the earliest development-only checkpoint reaching the represented
   non-inferiority target.
3. **Movement-matched diagnostic:** transfer efficiency at comparable two-verdict divergence.

Movement is downstream of the objective. The third view is a diagnostic controlled-direct-effect-like
comparison, not the total objective effect and not proof of mediation.

## 7. Why GRPO is no longer primary

GRPO answers a different question in a one-token, two-action guard. There is no multi-token reasoning
trajectory to explore, and group-relative updates can vanish when sampled verdicts agree. Recent work
already shows that richer intent/reasoning outputs can make GRPO useful for safety classification.
Including GRPO here would expand trainer and reward confounds while weakening the exact
VerdictCE/PairCE/DPO factorization.

GRPO, KTO, ORPO, and IPO may be future sensitivity studies. They do not belong in the primary v2
matrix.

## 8. Evaluation logic

Ranking alone is insufficient for a deployed guard. The study therefore keeps Paper A's
represented-versus-transfer macro-AP and adds explicit reliability gates:

- calibration NLL and Brier score before and after temperature scaling;
- transfer behavior at a calibration-only 5% FPR target;
- OR-Bench benign false positives;
- HarmBench recall;
- low-prevalence precision and invalid outputs.

The Paper A suite remains retrospective. A new cohort can support confirmation only if no row,
aggregate result, or model-selection signal is inspected before adapters, checkpoints, analysis, and
hashes are frozen.

## 9. Composition's narrower role

The fixed base-plus-adapter output average remains useful because it asks whether errors created by
different losses are equally recoverable. It is cheap and directly connects to Paper B. In v2 it is
secondary and descriptive: the study does not preregister a monotonic objective-to-movement-to-
composition causal chain.

## 10. Prior-art boundary

- [DPO](https://arxiv.org/abs/2305.18290) supplies the reference-relative preference objective.
- [GuardReasoner](https://arxiv.org/abs/2501.18492) uses reasoning SFT plus hard-sample DPO.
- [AIMS](https://arxiv.org/abs/2606.27210) compares intent-aware SFT, DPO, distillation, and GRPO.
- [DT-Guard](https://arxiv.org/abs/2607.06326) uses rollout-guided hard-case SFT and DPO.
- [On Calibration of LLM-based Guard Models](https://arxiv.org/abs/2410.10414) motivates treating
  calibration and distribution-shift reliability as first-class outcomes.

The defensible Paper C contribution is the exact one-token loss decomposition, identical-information
and identical-initialization design, objective-by-selection factorization, multi-model/multi-seed
paired transfer measurement, and a genuinely sealed confirmation—not DPO itself.
