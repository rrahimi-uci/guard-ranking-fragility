"""Contract tests for the frontier estimand and the design-power gate. CPU only, no torch."""

from __future__ import annotations

import math

import pytest

from paper_c.contracts import ContractError
from paper_c.frontier import (
    build_trajectories,
    cluster_bootstrap,
    factorial_marginal,
    interpolate_transfer,
    matched_transfer_gap,
    paired_frontier_gaps,
    trajectory,
)
from paper_c.power import (
    assert_design_powered,
    design_power_report,
    effective_learning_rate_ratio,
    minimum_detectable_effect,
    seed_sd_by_model,
)


def _cell(model, seed, sampler, objective, ladder):
    return [
        {"model_key": model, "seed": seed, "sampler": sampler, "objective": objective,
         "step": step, "represented_ap": rep, "transfer_ap": tra}
        for step, rep, tra in ladder
    ]


class TestTrajectory:
    def test_orders_by_step_and_keeps_running_max_represented(self):
        curve = trajectory([
            {"step": 100, "represented_ap": 0.90, "transfer_ap": 0.80},
            {"step": 25, "represented_ap": 0.85, "transfer_ap": 0.83},
            {"step": 50, "represented_ap": 0.84, "transfer_ap": 0.82},  # regression: dropped
        ])
        assert curve == [(0.85, 0.83), (0.90, 0.80)]

    def test_rejects_duplicate_steps(self):
        with pytest.raises(ContractError):
            trajectory([
                {"step": 25, "represented_ap": 0.8, "transfer_ap": 0.8},
                {"step": 25, "represented_ap": 0.9, "transfer_ap": 0.7},
            ])

    def test_rejects_empty(self):
        with pytest.raises(ContractError):
            trajectory([])


class TestInterpolation:
    def test_refuses_to_extrapolate(self):
        curve = [(0.80, 0.70), (0.90, 0.60)]
        assert interpolate_transfer(curve, 0.75) is None
        assert interpolate_transfer(curve, 0.95) is None

    def test_linear_midpoint(self):
        curve = [(0.80, 0.70), (0.90, 0.60)]
        assert interpolate_transfer(curve, 0.85) == pytest.approx(0.65)


class TestLearningRateInvariance:
    """The property the whole estimand exists for."""

    def test_pure_lr_rescale_gives_zero_gap(self):
        # A uniform gradient rescale moves a run further along the SAME trade-off curve per step.
        # Model that as the treatment reaching, at each step, the represented/transfer pair the
        # control reaches one step later. Same curve, different speed.
        shared = [(0.82, 0.86), (0.86, 0.83), (0.90, 0.79), (0.93, 0.74)]
        control = _cell("m", 42, "uncertain", "pair_ce",
                        [(25, *shared[0]), (50, *shared[1]), (100, *shared[2]), (200, *shared[3])])
        faster = _cell("m", 42, "uncertain", "dpo",
                       [(25, *shared[1]), (50, *shared[2]), (100, *shared[3]),
                        (200, 0.95, 0.71)])
        trajectories = build_trajectories(control + faster)
        gaps = paired_frontier_gaps(trajectories, treatment="dpo", control="pair_ce")
        assert len(gaps) == 1 and gaps[0]["status"] == "ok"
        # Overlapping represented range is traced by identical points, so the gap must vanish.
        assert gaps[0]["gap"] == pytest.approx(0.0, abs=1e-9)

    def test_genuine_frontier_shift_is_detected(self):
        control = _cell("m", 42, "uncertain", "pair_ce",
                        [(25, 0.82, 0.86), (50, 0.86, 0.83), (100, 0.90, 0.79), (200, 0.93, 0.74)])
        better = _cell("m", 42, "uncertain", "dpo",
                       [(25, 0.82, 0.88), (50, 0.86, 0.85), (100, 0.90, 0.81), (200, 0.93, 0.76)])
        trajectories = build_trajectories(control + better)
        gaps = paired_frontier_gaps(trajectories, treatment="dpo", control="pair_ce")
        assert gaps[0]["gap"] == pytest.approx(0.02, abs=1e-9)
        assert gaps[0]["dominates_everywhere"] is True


class TestOverlapGuard:
    def test_insufficient_overlap_is_reported_not_extrapolated(self):
        a = [(0.80, 0.70), (0.805, 0.69)]
        b = [(0.90, 0.60), (0.95, 0.55)]
        result = matched_transfer_gap(a, b)
        assert result["status"] == "insufficient_overlap"
        assert result["gap"] is None


class TestAggregation:
    def test_equal_weight_over_samplers_then_models(self):
        gaps = [
            {"model_key": "a", "seed": 42, "sampler": "uncertain", "status": "ok", "gap": 0.10},
            {"model_key": "a", "seed": 42, "sampler": "matched_random", "status": "ok", "gap": 0.00},
            {"model_key": "b", "seed": 42, "sampler": "uncertain", "status": "ok", "gap": 0.20},
            {"model_key": "b", "seed": 42, "sampler": "matched_random", "status": "ok", "gap": 0.20},
        ]
        out = factorial_marginal(gaps)
        # model a -> mean(0.10, 0.00) = 0.05 ; model b -> 0.20 ; panel -> 0.125
        assert out["value"] == pytest.approx(0.125)
        assert out["n_models"] == 2

    def test_unavailable_rather_than_computed_on_survivors(self):
        out = factorial_marginal([{"model_key": "a", "status": "missing_control", "gap": None}])
        assert out["status"] == "unavailable" and out["value"] is None

    def test_bootstrap_resamples_models_not_cells(self):
        gaps = []
        for model, value in (("a", 0.05), ("b", 0.05), ("c", 0.05), ("d", 0.05)):
            for sampler in ("uncertain", "matched_random"):
                gaps.append({"model_key": model, "seed": 42, "sampler": sampler,
                             "status": "ok", "gap": value})
        out = cluster_bootstrap(gaps, replicates=200, seed=7)
        assert out["status"] == "ok" and out["n_models"] == 4
        # Zero between-model variance -> degenerate interval at the point estimate.
        assert out["ci_lower_95"] == pytest.approx(0.05)
        assert out["point"] == pytest.approx(0.05)

    def test_bootstrap_is_deterministic(self):
        gaps = [{"model_key": m, "seed": 42, "sampler": "uncertain", "status": "ok", "gap": g}
                for m, g in (("a", 0.01), ("b", 0.05), ("c", -0.02), ("d", 0.03))]
        first = cluster_bootstrap(gaps, replicates=500, seed=11)
        second = cluster_bootstrap(gaps, replicates=500, seed=11)
        assert first["ci_lower_95"] == second["ci_lower_95"]
        assert first["ci_upper_95"] == second["ci_upper_95"]


class TestPower:
    def test_clustered_units_widen_the_interval(self):
        clustered = minimum_detectable_effect(sd=0.0334, n_effective=4)
        optimistic = minimum_detectable_effect(sd=0.0334, n_effective=20)
        assert clustered > optimistic
        assert clustered == pytest.approx(0.0327, abs=5e-4)
        assert optimistic == pytest.approx(0.0146, abs=5e-4)

    def test_pairing_reduces_the_detectable_effect(self):
        unpaired = minimum_detectable_effect(sd=0.0334, n_effective=4)
        paired = minimum_detectable_effect(sd=0.0334, n_effective=4,
                                           pairing_variance_reduction=0.25)
        assert paired == pytest.approx(unpaired / 2.0, rel=1e-9)

    def test_seed_sd_from_rows(self):
        rows = [{"model_key": "a", "seed": s, "transfer_ap": v}
                for s, v in zip((42, 43, 44), (0.80, 0.84, 0.82))]
        out = seed_sd_by_model(rows)
        assert out["per_model"]["a"]["n_seeds"] == 3
        assert out["pooled_sd"] == pytest.approx(0.02, abs=1e-9)

    def test_underpowered_design_fails_closed(self):
        variance = {"pooled_sd": 0.0334, "per_model": {"a": {"sd": 0.0334}}, "n_models": 1}
        report = design_power_report(seed_variance=variance, n_models=4, n_seeds=5,
                                    target_effect=0.005,
                                    pairing_variance_reduction=1.0, ladder_points=4)
        assert report["verdict"] == "underpowered"
        with pytest.raises(ContractError, match="underpowered"):
            assert_design_powered(report)

    def test_powered_design_passes(self):
        variance = {"pooled_sd": 0.0334, "per_model": {"a": {"sd": 0.0334}}, "n_models": 1}
        report = design_power_report(seed_variance=variance, n_models=4, n_seeds=5,
                                    target_effect=0.02,
                                    pairing_variance_reduction=0.25, ladder_points=4)
        assert report["verdict"] == "powered"
        assert_design_powered(report)

    def test_required_reduction_is_reported(self):
        variance = {"pooled_sd": 0.0334, "per_model": {"a": {"sd": 0.0334}}, "n_models": 1}
        report = design_power_report(seed_variance=variance, n_models=4, n_seeds=5,
                                    target_effect=0.02,
                                    pairing_variance_reduction=1.0, ladder_points=4)
        required = report["required_pairing_variance_reduction"]
        assert 0 < required < 1
        # Achieving exactly the required reduction should make the design powered.
        tuned = design_power_report(seed_variance=variance, n_models=4, n_seeds=5,
                                    target_effect=0.02,
                                    pairing_variance_reduction=required, ladder_points=4)
        assert tuned["primary_minimum_detectable_effect"] == pytest.approx(0.02, rel=1e-6)


class TestEffectiveLearningRate:
    def test_uniform_positive_margins_are_a_near_uniform_rescale(self):
        # Narrow, positive margin distribution -> high ratio, low dispersion: the confound.
        margins = [5.0, 5.2, 5.1, 4.9, 5.3]
        out = effective_learning_rate_ratio(margins, beta=0.1)
        assert out["effective_lr_ratio"] > 1.2
        assert out["pair_ce_weight_cv"] < 0.05
        assert out["fraction_positive_margin"] == 1.0

    def test_zero_margins_make_the_objectives_identical(self):
        out = effective_learning_rate_ratio([0.0, 0.0, 0.0], beta=0.1)
        assert out["effective_lr_ratio"] == pytest.approx(1.0)

    def test_rejects_bad_beta(self):
        with pytest.raises(ContractError):
            effective_learning_rate_ratio([1.0], beta=0.0)
