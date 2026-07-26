"""Exact scalar and tensor objectives for cross-model alignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real

from .contracts import ContractError


def _finite_number(value: object, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContractError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{field} must be finite")
    if nonnegative and number < 0:
        raise ContractError(f"{field} must be nonnegative")
    return number


def softplus(value: float) -> float:
    number = _finite_number(value, "softplus input")
    if number > 0:
        return number + math.log1p(math.exp(-number))
    return math.log1p(math.exp(number))


def _beta(value: float) -> float:
    beta = _finite_number(value, "beta")
    if beta <= 0:
        raise ContractError("beta must be finite and positive")
    return beta


def cross_pairce_loss(
    chosen_policy_logp: float,
    rejected_policy_logp: float,
    *,
    beta: float,
) -> float:
    chosen = _finite_number(chosen_policy_logp, "chosen_policy_logp")
    rejected = _finite_number(rejected_policy_logp, "rejected_policy_logp")
    margin = chosen - rejected
    return softplus(-_beta(beta) * margin)


def cm_dpo_loss(
    chosen_policy_logp: float,
    rejected_policy_logp: float,
    chosen_reference_logp: float,
    rejected_reference_logp: float,
    *,
    beta: float,
) -> float:
    chosen_policy = _finite_number(chosen_policy_logp, "chosen_policy_logp")
    rejected_policy = _finite_number(rejected_policy_logp, "rejected_policy_logp")
    chosen_reference = _finite_number(chosen_reference_logp, "chosen_reference_logp")
    rejected_reference = _finite_number(
        rejected_reference_logp, "rejected_reference_logp"
    )
    margin = (
        chosen_policy
        - rejected_policy
        - chosen_reference
        + rejected_reference
    )
    return softplus(-_beta(beta) * margin)


def categorical_cross_entropy(target_index: int, probabilities: Sequence[float]) -> float:
    if isinstance(target_index, bool) or not isinstance(target_index, int):
        raise ContractError("target index must be an integer")
    if not isinstance(probabilities, Sequence) or isinstance(probabilities, (str, bytes)):
        raise ContractError("probability vector must be a sequence")
    if target_index not in range(len(probabilities)):
        raise ContractError("target index is outside probability vector")
    clean = [
        _finite_number(value, f"probabilities[{index}]", nonnegative=True)
        for index, value in enumerate(probabilities)
    ]
    if not math.isclose(sum(clean), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ContractError("probability vector must sum to one")
    probability = clean[target_index]
    if probability <= 0:
        raise ContractError("target probability must lie in (0,1]")
    return -math.log(probability)


def categorical_kl(target: Sequence[float], policy: Sequence[float]) -> float:
    if len(target) != len(policy) or not target:
        raise ContractError("KL distributions must be nonempty and aligned")
    left = [
        _finite_number(value, f"target[{index}]", nonnegative=True)
        for index, value in enumerate(target)
    ]
    right = [
        _finite_number(value, f"policy[{index}]", nonnegative=True)
        for index, value in enumerate(policy)
    ]
    if abs(sum(left) - 1.0) > 1e-6 or abs(sum(right) - 1.0) > 1e-6:
        raise ContractError("KL distributions must sum to one")
    if any(p > 0 and q <= 0 for p, q in zip(left, right, strict=True)):
        raise ContractError("policy assigns zero mass where target has mass")
    return sum(
        p * math.log(p / q)
        for p, q in zip(left, right, strict=True)
        if p > 0
    )


def weighted_mean(losses: Sequence[float], weights: Sequence[float]) -> float:
    if len(losses) != len(weights) or not losses:
        raise ContractError("losses and weights must be nonempty and aligned")
    clean_losses = [
        _finite_number(value, f"losses[{index}]")
        for index, value in enumerate(losses)
    ]
    clean_weights = [
        _finite_number(value, f"weights[{index}]")
        for index, value in enumerate(weights)
    ]
    if any(value <= 0 for value in clean_weights):
        raise ContractError("weights must be positive")
    return sum(loss * weight for loss, weight in zip(
        clean_losses, clean_weights, strict=True
    )) / sum(clean_weights)


def _validate_expected_categories(
    category_losses: Mapping[str, float],
    expected_categories: Sequence[str] | set[str] | None,
) -> None:
    if expected_categories is None:
        return
    if isinstance(expected_categories, (str, bytes)) or not isinstance(
        expected_categories, (Sequence, set, frozenset)
    ):
        raise ContractError("expected_categories must be a collection")
    expected = list(expected_categories)
    if (
        not expected
        or any(not isinstance(category, str) or not category.strip() for category in expected)
        or len(set(expected)) != len(expected)
    ):
        raise ContractError("expected_categories must be unique nonempty strings")
    if set(category_losses) != set(expected):
        missing = sorted(set(expected) - set(category_losses))
        extra = sorted(set(category_losses) - set(expected))
        raise ContractError(
            f"category loss set mismatch; missing={missing}, extra={extra}"
        )


def soft_worst_category(
    category_losses: Mapping[str, float],
    *,
    temperature: float,
    expected_categories: Sequence[str] | set[str] | None = None,
) -> float:
    if not isinstance(category_losses, Mapping) or not category_losses:
        raise ContractError("category losses are empty")
    _validate_expected_categories(category_losses, expected_categories)
    tau = _finite_number(temperature, "temperature")
    if tau <= 0:
        raise ContractError("temperature must be finite and positive")
    if any(not isinstance(category, str) or not category.strip() for category in category_losses):
        raise ContractError("category loss keys must be nonempty strings")
    values = [
        _finite_number(value, f"category loss {category}", nonnegative=True)
        for category, value in category_losses.items()
    ]
    maximum = max(values)
    # The division by category count makes equal category losses invariant to C.
    return maximum + tau * math.log(
        sum(math.exp((value - maximum) / tau) for value in values) / len(values)
    )


def composite_alignment_loss(
    category_losses: Mapping[str, float],
    *,
    gold_anchor_loss: float,
    retention_kl: float,
    temperature: float,
    lambda_gold: float,
    lambda_retain: float,
    expected_categories: Sequence[str] | set[str] | None = None,
) -> dict[str, float]:
    for name, value in {
        "gold_anchor_loss": gold_anchor_loss,
        "retention_kl": retention_kl,
        "lambda_gold": lambda_gold,
        "lambda_retain": lambda_retain,
    }.items():
        _finite_number(value, name, nonnegative=True)
    robust = soft_worst_category(
        category_losses,
        temperature=temperature,
        expected_categories=expected_categories,
    )
    clean_gold = _finite_number(gold_anchor_loss, "gold_anchor_loss", nonnegative=True)
    clean_retention = _finite_number(retention_kl, "retention_kl", nonnegative=True)
    clean_lambda_gold = _finite_number(lambda_gold, "lambda_gold", nonnegative=True)
    clean_lambda_retain = _finite_number(
        lambda_retain, "lambda_retain", nonnegative=True
    )
    total = robust + clean_lambda_gold * clean_gold + clean_lambda_retain * clean_retention
    return {
        "total": total,
        "soft_worst_category": robust,
        "gold_anchor": clean_gold,
        "retention_kl": clean_retention,
    }


def torch_pair_loss(
    *,
    arm: str,
    chosen_policy_logps,
    rejected_policy_logps,
    chosen_reference_logps=None,
    rejected_reference_logps=None,
    beta: float,
    weights=None,
):
    """Vectorized pair component; imports torch only when training calls it.

    The reference tensors are required for ``cm_dpo`` and must be absent for
    ``cross_pairce``.  Making them optional-but-rejected rather than always
    required stops the uncentered arm from being handed reference tensors that
    it silently ignores, which would let a mismatched pair inventory reach the
    optimizer unnoticed.
    """
    import torch
    import torch.nn.functional as functional

    beta = _beta(beta)
    if arm == "cross_pairce" and (
        chosen_reference_logps is not None or rejected_reference_logps is not None
    ):
        raise ContractError("cross_pairce must not receive reference log-probabilities")
    if arm == "cm_dpo" and (
        chosen_reference_logps is None or rejected_reference_logps is None
    ):
        raise ContractError("cm_dpo requires both reference log-probability tensors")
    tensors = {
        "chosen_policy_logps": chosen_policy_logps,
        "rejected_policy_logps": rejected_policy_logps,
    }
    if arm == "cm_dpo":
        tensors.update({
            "chosen_reference_logps": chosen_reference_logps,
            "rejected_reference_logps": rejected_reference_logps,
        })
    for name, tensor in tensors.items():
        if not torch.is_tensor(tensor):
            raise ContractError(f"{name} must be a tensor")
        if not torch.is_floating_point(tensor):
            raise ContractError(f"{name} must be floating point")
        if tensor.ndim != 1 or tensor.numel() == 0:
            raise ContractError(f"{name} must be a nonempty one-dimensional tensor")
        if not bool(torch.all(torch.isfinite(tensor)).item()):
            raise ContractError(f"{name} must be finite")
    policy_shape = chosen_policy_logps.shape
    if rejected_policy_logps.shape != policy_shape:
        raise ContractError("policy log-probability tensors must have identical shapes")
    if rejected_policy_logps.device != chosen_policy_logps.device:
        raise ContractError("policy log-probability tensors must share a device")
    policy_margin = chosen_policy_logps.float() - rejected_policy_logps.float()
    if arm == "cross_pairce":
        losses = functional.softplus(-beta * policy_margin)
    elif arm == "cm_dpo":
        if (
            chosen_reference_logps.shape != policy_shape
            or rejected_reference_logps.shape != policy_shape
        ):
            raise ContractError("reference and policy tensors must have identical shapes")
        if (
            chosen_reference_logps.device != chosen_policy_logps.device
            or rejected_reference_logps.device != chosen_policy_logps.device
        ):
            raise ContractError("reference and policy tensors must share a device")
        reference_margin = (
            chosen_reference_logps.detach().float()
            - rejected_reference_logps.detach().float()
        )
        losses = functional.softplus(-beta * (policy_margin - reference_margin))
    else:
        raise ContractError("torch_pair_loss accepts cross_pairce or cm_dpo")
    if weights is None:
        return losses.mean()
    if not torch.is_tensor(weights):
        raise ContractError("tensor pair weights must be a tensor")
    clean = weights.float()
    if clean.shape != losses.shape:
        raise ContractError("tensor pair weights must match the loss shape")
    if clean.device != losses.device:
        raise ContractError("tensor pair weights must share the loss device")
    if not bool(torch.all(torch.isfinite(clean)).item()):
        raise ContractError("tensor pair weights must be finite")
    if bool(torch.any(clean <= 0).item()):
        raise ContractError("tensor pair weights must be positive")
    return (losses * clean).sum() / clean.sum()


def torch_composite_alignment_loss(
    category_losses: Mapping[str, object],
    *,
    gold_anchor_loss,
    retention_kl,
    temperature: float,
    lambda_gold: float,
    lambda_retain: float,
    expected_categories: Sequence[str] | set[str] | None = None,
):
    """Differentiable twin of :func:`composite_alignment_loss`.

    The scalar version validates plain floats and is what the contract tests pin.
    Training needs the same arithmetic on autograd tensors, so this mirrors it term
    for term rather than reimplementing it -- ``test_torch_composite_matches_scalar``
    holds the two to the same numbers.
    """
    import torch

    if not isinstance(category_losses, Mapping) or not category_losses:
        raise ContractError("category losses are empty")
    _validate_expected_categories(category_losses, expected_categories)
    tau = _finite_number(temperature, "temperature")
    if tau <= 0:
        raise ContractError("temperature must be finite and positive")
    lam_gold = _finite_number(lambda_gold, "lambda_gold", nonnegative=True)
    lam_retain = _finite_number(lambda_retain, "lambda_retain", nonnegative=True)

    names = sorted(category_losses)
    values = torch.stack([category_losses[name] for name in names])
    maximum = values.max().detach()
    # identical to the scalar form: max + tau*log(mean(exp((L-max)/tau)))
    robust = maximum + tau * torch.log(
        torch.exp((values - maximum) / tau).sum() / values.numel()
    )
    total = robust + lam_gold * gold_anchor_loss + lam_retain * retention_kl
    return {
        "total": total,
        "soft_worst_category": robust,
        "gold_anchor": gold_anchor_loss,
        "retention_kl": retention_kl,
    }
