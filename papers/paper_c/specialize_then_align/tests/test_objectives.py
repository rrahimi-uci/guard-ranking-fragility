from __future__ import annotations

import math

import pytest

from paper_c_sta.objectives import (
    categorical_cross_entropy,
    categorical_kl,
    cm_dpo_loss,
    composite_alignment_loss,
    cross_pairce_loss,
    soft_worst_category,
    torch_pair_loss,
)
from paper_c_sta.contracts import ContractError


def test_cm_dpo_step_zero_is_log_two():
    loss = cm_dpo_loss(0.7, -0.2, 0.7, -0.2, beta=0.1)
    assert loss == pytest.approx(math.log(2.0))


def test_reference_centering_differs_from_uncentered_pairce():
    pair = cross_pairce_loss(0.7, -0.2, beta=0.1)
    centered = cm_dpo_loss(0.7, -0.2, 0.7, -0.2, beta=0.1)
    assert pair < centered


def test_soft_worst_is_equal_loss_invariant_and_near_max():
    assert soft_worst_category({"a": 2.0, "b": 2.0}, temperature=0.1) == pytest.approx(2.0)
    robust = soft_worst_category({"easy": 0.1, "hard": 1.0}, temperature=0.05)
    assert 0.95 < robust <= 1.0


def test_soft_worst_requires_exact_expected_category_set():
    with pytest.raises(ContractError, match="category loss set mismatch"):
        soft_worst_category(
            {"toxicity": 0.2},
            temperature=0.1,
            expected_categories=["toxicity", "mortgage"],
        )
    result = composite_alignment_loss(
        {"toxicity": 0.2, "mortgage": 0.4},
        gold_anchor_loss=0.1,
        retention_kl=0.1,
        temperature=0.1,
        lambda_gold=0.5,
        lambda_retain=0.05,
        expected_categories=["toxicity", "mortgage"],
    )
    assert result["total"] >= result["soft_worst_category"]


def test_composite_alignment_components_are_explicit():
    result = composite_alignment_loss(
        {"general": 0.2, "mortgage": 0.6},
        gold_anchor_loss=0.3,
        retention_kl=0.1,
        temperature=0.1,
        lambda_gold=0.5,
        lambda_retain=0.05,
    )
    assert result["total"] == pytest.approx(
        result["soft_worst_category"] + 0.5 * 0.3 + 0.05 * 0.1
    )


def test_categorical_kl_zero_for_identical_distributions():
    assert categorical_kl([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_scalar_pair_losses_reject_nonfinite_log_probabilities(bad):
    with pytest.raises(ContractError, match="finite"):
        cross_pairce_loss(bad, -0.2, beta=0.1)
    with pytest.raises(ContractError, match="finite"):
        cm_dpo_loss(0.1, -0.2, bad, -0.3, beta=0.1)


def test_categorical_cross_entropy_validates_full_distribution():
    assert categorical_cross_entropy(1, [0.2, 0.8]) == pytest.approx(-math.log(0.8))
    with pytest.raises(ContractError, match="sum to one"):
        categorical_cross_entropy(1, [0.2, 0.7])
    with pytest.raises(ContractError, match="numeric"):
        categorical_cross_entropy(0, [True, 0.0])


def test_torch_pair_loss_rejects_broadcast_and_nonfinite_inputs():
    torch = pytest.importorskip("torch")
    chosen = torch.tensor([0.4, 0.3])
    rejected = torch.tensor([0.1, 0.2])
    reference = torch.tensor([0.0, 0.0])
    with pytest.raises(ContractError, match="identical shapes"):
        torch_pair_loss(
            arm="cm_dpo",
            chosen_policy_logps=chosen,
            rejected_policy_logps=rejected[:1],
            chosen_reference_logps=reference,
            rejected_reference_logps=reference,
            beta=0.1,
        )
    with pytest.raises(ContractError, match="finite"):
        torch_pair_loss(
            arm="cm_dpo",
            chosen_policy_logps=torch.tensor([math.nan]),
            rejected_policy_logps=torch.tensor([0.0]),
            chosen_reference_logps=torch.tensor([0.0]),
            rejected_reference_logps=torch.tensor([0.0]),
            beta=0.1,
        )
    with pytest.raises(ContractError, match="weights must match"):
        torch_pair_loss(
            arm="cross_pairce",
            chosen_policy_logps=chosen,
            rejected_policy_logps=rejected,
            chosen_reference_logps=None,
            rejected_reference_logps=None,
            beta=0.1,
            weights=torch.tensor([1.0]),
        )


def test_torch_dpo_detaches_reference_tensors():
    torch = pytest.importorskip("torch")
    chosen = torch.tensor([0.4, 0.3], requires_grad=True)
    rejected = torch.tensor([0.1, 0.2], requires_grad=True)
    chosen_reference = torch.tensor([0.0, 0.0], requires_grad=True)
    rejected_reference = torch.tensor([0.0, 0.0], requires_grad=True)
    loss = torch_pair_loss(
        arm="cm_dpo",
        chosen_policy_logps=chosen,
        rejected_policy_logps=rejected,
        chosen_reference_logps=chosen_reference,
        rejected_reference_logps=rejected_reference,
        beta=0.1,
    )
    loss.backward()
    assert chosen.grad is not None
    assert rejected.grad is not None
    assert chosen_reference.grad is None
    assert rejected_reference.grad is None


def test_torch_composite_matches_scalar():
    """The differentiable composite must equal the scalar one it mirrors.

    The scalar form is what the contract pins; training uses the tensor form.  If the
    two ever drift, the objective that was reviewed is not the objective that runs.
    """
    torch = pytest.importorskip("torch")
    from paper_c_sta.objectives import (
        composite_alignment_loss,
        torch_composite_alignment_loss,
    )

    cats = ["a", "b", "c", "d", "e"]
    values = [0.4, 1.9, 0.7, 0.2, 1.1]
    kwargs = dict(temperature=0.1, lambda_gold=0.5, lambda_retain=0.05,
                  expected_categories=cats)
    scalar = composite_alignment_loss(
        dict(zip(cats, values)), gold_anchor_loss=1.3, retention_kl=0.04, **kwargs)
    tensor = torch_composite_alignment_loss(
        {c: torch.tensor(v) for c, v in zip(cats, values)},
        gold_anchor_loss=torch.tensor(1.3), retention_kl=torch.tensor(0.04), **kwargs)
    for key in ("total", "soft_worst_category"):
        assert abs(scalar[key] - float(tensor[key])) < 1e-5, key


def test_torch_composite_gradient_reaches_every_category():
    """No category may be silently excluded from the worst-category term."""
    torch = pytest.importorskip("torch")
    from paper_c_sta.objectives import torch_composite_alignment_loss

    cats = ["a", "b", "c", "d", "e"]
    losses = {c: torch.tensor(float(i + 1) / 5, requires_grad=True)
              for i, c in enumerate(cats)}
    out = torch_composite_alignment_loss(
        losses, gold_anchor_loss=torch.tensor(0.0), retention_kl=torch.tensor(0.0),
        temperature=0.5, lambda_gold=0.5, lambda_retain=0.05, expected_categories=cats)
    out["total"].backward()
    assert all(losses[c].grad is not None and float(losses[c].grad) > 0 for c in cats)
