"""Exact scalar and tensor objectives for the matched Paper C comparison."""

from __future__ import annotations

import math

from .contracts import ContractError


def softplus(value: float) -> float:
    value = float(value)
    if value > 0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def signed_margin(safe_logit: float, unsafe_logit: float, gold: int) -> float:
    if gold not in (0, 1):
        raise ContractError(f"gold must be 0 or 1, got {gold!r}")
    sign = 1.0 if gold == 1 else -1.0
    return sign * (float(unsafe_logit) - float(safe_logit))


def pair_ce_loss(policy_signed_margin: float, beta: float) -> float:
    if not math.isfinite(float(beta)) or float(beta) <= 0:
        raise ContractError("beta must be finite and positive")
    return softplus(-float(beta) * float(policy_signed_margin))


def dpo_loss(policy_signed_margin: float, reference_signed_margin: float, beta: float) -> float:
    if not math.isfinite(float(beta)) or float(beta) <= 0:
        raise ContractError("beta must be finite and positive")
    delta = float(policy_signed_margin) - float(reference_signed_margin)
    return softplus(-float(beta) * delta)


def dpo_logratio_loss(
    chosen_policy_logp: float,
    rejected_policy_logp: float,
    chosen_reference_logp: float,
    rejected_reference_logp: float,
    beta: float,
) -> float:
    delta = (
        float(chosen_policy_logp)
        - float(rejected_policy_logp)
        - float(chosen_reference_logp)
        + float(rejected_reference_logp)
    )
    return softplus(-float(beta) * delta)


def two_verdict_probability_unsafe(safe_logit: float, unsafe_logit: float) -> float:
    delta = float(unsafe_logit) - float(safe_logit)
    if delta >= 0:
        z = math.exp(-delta)
        return 1.0 / (1.0 + z)
    z = math.exp(delta)
    return z / (1.0 + z)


def binary_entropy(probability: float) -> float:
    probability = float(probability)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ContractError("probability must lie in [0,1]")
    if probability in (0.0, 1.0):
        return 0.0
    return -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log(1.0 - probability)
    )


def two_verdict_entropy(safe_logit: float, unsafe_logit: float) -> float:
    return binary_entropy(two_verdict_probability_unsafe(safe_logit, unsafe_logit))


def torch_objective_loss(
    *,
    objective: str,
    logits,
    target_ids,
    gold_signs,
    reference_margins,
    safe_token_id: int,
    unsafe_token_id: int,
    beta: float,
):
    """Compute one of the three objectives on next-token logits.

    Torch is imported lazily so contract tests do not require a GPU stack.
    `verdict_ce` uses the full vocabulary. PairCE and DPO use exactly the two
    locked verdict logits and share the same beta.
    """
    import torch.nn.functional as functional

    if objective == "verdict_ce":
        return functional.cross_entropy(logits.float(), target_ids)
    score = logits[:, unsafe_token_id] - logits[:, safe_token_id]
    margins = gold_signs.float() * score.float()
    if objective == "pair_ce":
        return functional.softplus(-float(beta) * margins).mean()
    if objective == "dpo":
        return functional.softplus(
            -float(beta) * (margins - reference_margins.float())
        ).mean()
    raise ContractError(f"unknown objective: {objective}")


def assert_step_zero_identity(
    policy_margins: list[float],
    reference_margins: list[float],
    *,
    beta: float,
    atol: float,
) -> dict:
    if len(policy_margins) != len(reference_margins) or not policy_margins:
        raise ContractError("step-zero policy/reference arrays must be nonempty and aligned")
    errors = [abs(float(policy) - float(reference)) for policy, reference in zip(
        policy_margins, reference_margins, strict=True
    )]
    losses = [dpo_loss(policy, reference, beta) for policy, reference in zip(
        policy_margins, reference_margins, strict=True
    )]
    maximum = max(errors)
    mean_loss = sum(losses) / len(losses)
    if maximum > float(atol):
        raise ContractError(f"step-zero reference margin error {maximum:.6g} exceeds {atol}")
    if abs(mean_loss - math.log(2.0)) > max(float(atol), 1e-6):
        raise ContractError("step-zero DPO loss is not log(2)")
    return {"max_abs_margin_error": maximum, "mean_dpo_loss": mean_loss}

