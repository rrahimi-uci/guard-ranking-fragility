#!/usr/bin/env python
"""CPU-only tests for the non-HARKing starting-type analyzer (analyze_starting_type_adaptation).

Builds a SYNTHETIC starting-type score table (3 model families / 4 checkpoints, U/SFT/KL x 5 seeds
x 2 splits, 2 benchmark sources per split, KNOWN deltas) and asserts:

  * the four confirmatory predicates + RQ1/RQ2 gates compute correctly on a designed-supported panel;
  * headroom-normalized gains are reported alongside the raw deltas and exceed them;
  * the interpretation wording is selected ONLY by locked bound predicates (a claim registry),
    never by inspecting a point-estimate sign;
  * the beta==0 identity (KL == SFT) yields H_preserve == 0 EXACTLY (and H_cost == 0).

Run directly: python tests/test_analyze_starting_type.py   (or via pytest)
"""
import math
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import analyze_starting_type_adaptation as A  # noqa: E402

REPS = 400  # small but enough for stable LCB signs on the designed synthetic panel


def test_predicates_and_gates_supported():
    df, fmap = A.make_synthetic_scores(kl_equals_sft_heldout=False)
    res = A.analyze(df, family_map=fmap, reps=REPS, rng_seed=A.DEFAULT_RNG_SEED)
    H = res["point_estimates"]["H"]
    st = res["bootstrap"]["stats"]
    checks = res["claim_checks"]
    # designed: SFT gains on represented, loses on held-out; KL preserves held-out at ~zero cost
    assert H["H_gain"] > 0 and H["H_conc"] > 0 and H["H_preserve"] > 0
    assert st["H_gain"]["lcb975_one_sided"] > 0
    assert st["H_conc"]["lcb975_one_sided"] > 0
    assert st["H_preserve"]["lcb975_one_sided"] > 0
    assert st["H_cost"]["lcb975_one_sided"] > -A.NONINF_MARGIN
    assert checks["RQ1"]["supported"] is True
    assert checks["RQ2"]["supported"] is True
    # equal-family means average within family first: famQ has two checkpoints
    fams = res["config"]["model_families"]
    assert set(fams) == {"famQ", "famR", "famS"}, fams
    assert res["config"]["n_checkpoints"] == 4
    print(f"  [ok] RQ1={checks['RQ1']['supported']} RQ2={checks['RQ2']['supported']} "
          f"(3 families, 4 checkpoints; equal-family means)")


def test_headroom_normalized_reported():
    df, fmap = A.make_synthetic_scores(kl_equals_sft_heldout=False)
    res = A.analyze(df, family_map=fmap, reps=REPS, rng_seed=A.DEFAULT_RNG_SEED)
    H = res["point_estimates"]["H"]
    assert "H_gain_norm" in H and "H_conc_norm" in H
    assert math.isfinite(H["H_gain_norm"])
    # 0 < (1 - AP_U) < 1  ->  normalized gain >= raw gain
    assert H["H_gain_norm"] >= H["H_gain"] - 1e-9
    # per-checkpoint records also carry both raw and normalized deltas
    pc = res["point_estimates"]["per_checkpoint"]
    any_ck = next(iter(pc.values()))
    assert "delta_sft" in any_ck["represented"] and "delta_sft_norm" in any_ck["represented"]
    print(f"  [ok] headroom-normalized gains reported alongside raw "
          f"(H_gain={H['H_gain']:+.4f}, H_gain_norm={H['H_gain_norm']:+.4f})")


def test_interpretation_is_predicate_driven_only():
    # Forcing the locked bound predicates deterministically selects the wording, with no data.
    def boot(gain, conc, preserve, cost, held_ucb):
        stats = {
            "H_gain": {"lcb975_one_sided": gain, "ucb975_one_sided": 1.0},
            "H_conc": {"lcb975_one_sided": conc, "ucb975_one_sided": 1.0},
            "H_preserve": {"lcb975_one_sided": preserve, "ucb975_one_sided": 1.0},
            "H_cost": {"lcb975_one_sided": cost, "ucb975_one_sided": 1.0},
            "H_held_sft": {"lcb975_one_sided": -1.0, "ucb975_one_sided": held_ucb},
        }
        # The gates read one panel's bounds, selected by `stats_key`. All three carry the same
        # forced predicates here so this test isolates predicate -> wording; which panel is the
        # DEFAULT is asserted separately below.
        return {"stats": stats, "stats_purpose_built": stats, "stats_general": stats}

    not_sup = A.claim_checks(boot(-1.0, -1.0, -1.0, -1.0, 1.0))
    assert not_sup["RQ1"]["supported"] is False
    assert not_sup["RQ1"]["interpretation"] == A.RQ1_NOT_SUPPORTED
    assert not_sup["RQ2"]["supported"] is False

    # RQ1 supported with a bound-confirmed held-out loss (UCB < 0) vs not
    with_loss = A.claim_checks(boot(0.05, 0.05, 0.05, 0.0, -0.01))
    no_loss = A.claim_checks(boot(0.05, 0.05, 0.05, 0.0, 1.0))
    assert with_loss["RQ1"]["supported"] and no_loss["RQ1"]["supported"]
    assert with_loss["RQ1"]["interpretation"] == A.RQ1_CLAIMS[(True, True, True)]
    assert no_loss["RQ1"]["interpretation"] == A.RQ1_CLAIMS[(True, True, False)]
    assert with_loss["RQ1"]["interpretation"] != no_loss["RQ1"]["interpretation"]

    # RQ2 "preserve but cost fails" vs "preserve at acceptable cost"
    cost_fail = A.claim_checks(boot(0.0, 0.0, 0.05, -1.0, 1.0))
    cost_ok = A.claim_checks(boot(0.0, 0.0, 0.05, 0.0, 1.0))
    assert cost_fail["RQ2"]["supported"] is False
    assert cost_ok["RQ2"]["supported"] is True
    assert cost_fail["RQ2"]["interpretation"] == A.RQ2_CLAIMS[(True, False)]
    assert cost_ok["RQ2"]["interpretation"] == A.RQ2_CLAIMS[(True, True)]
    print("  [ok] interpretation strings selected ONLY by locked bound predicates (non-HARKing)")


def test_gates_read_the_registered_purpose_built_panel():
    """RQ1/RQ2 must be gated on the purpose-built panel, never the pooled mixed panel.

    Regression lock for the defect that made every published H a six-family MIXED-panel
    statistic: the analyzer pooled general and purpose-built checkpoints and grouped by model
    family, so one `qwen` family spanned both starting types. Here the two panels are given
    OPPOSITE forced predicates, so a gate reading the wrong one flips the verdict.
    """
    passing = {k: {"lcb975_one_sided": 0.05, "ucb975_one_sided": 1.0}
               for k in ("H_gain", "H_conc", "H_preserve", "H_cost", "H_held_sft")}
    failing = {k: {"lcb975_one_sided": -1.0, "ucb975_one_sided": 1.0}
               for k in ("H_gain", "H_conc", "H_preserve", "H_cost", "H_held_sft")}

    # purpose-built passes, mixed/general fail -> the default verdict must be "supported"
    c = A.claim_checks({"stats": failing, "stats_purpose_built": passing,
                        "stats_general": failing})
    assert c["estimand_panel"] == "purpose_built"
    assert c["RQ1"]["supported"] is True and c["RQ2"]["supported"] is True

    # and the reverse: a passing mixed panel must NOT be able to carry the gates
    c2 = A.claim_checks({"stats": passing, "stats_purpose_built": failing,
                         "stats_general": passing})
    assert c2["RQ1"]["supported"] is False and c2["RQ2"]["supported"] is False

    # the outcome is never labelled confirmatory, whatever the predicates say
    assert "NOT CONFIRMATORY" in c["confirmatory_status"]
    print("  [ok] RQ1/RQ2 gates read the registered purpose-built panel, not the mixed panel")


def test_panel_split_and_gamma_are_exact():
    """The panel split is real and Gamma is the paired purpose-built-minus-general difference."""
    df, family_map = A.make_synthetic_scores()
    type_map = {k: ("purpose_built" if k != "ckptS" else "general") for k in family_map}
    res = A.analyze(df, family_map=family_map, type_map=type_map, reps=120)
    cfg, pe = res["config"], res["point_estimates"]
    assert cfg["registered_primary_panel"] == "purpose_built"
    assert cfg["n_checkpoints_purpose_built"] + cfg["n_checkpoints_general"] == cfg["n_checkpoints"]
    assert cfg["model_families_general"] == ["famS"]
    # Gamma is a difference of the two panels' H, taken on the SAME draw (not two independent runs)
    for k, tag in (("Gamma_gain", "H_gain"), ("Gamma_conc", "H_conc"),
                   ("Gamma_preserve", "H_preserve"), ("Gamma_cost", "H_cost")):
        assert abs(pe["Gamma"][k] - (pe["H_purpose_built"][tag] - pe["H_general"][tag])) < 1e-12
        assert k in res["bootstrap"]["stats_gamma"]
    # the superseded mixed panel is retained, and labelled as superseded
    assert "superseded" in res["claim_checks_mixed_panel_superseded"]
    print("  [ok] registered panels separated; Gamma exact and bootstrapped on shared draws")


def test_beta0_identity_preserve_zero_exact():
    df0, fmap0 = A.make_synthetic_scores(kl_equals_sft_heldout=True, kl_beta=0.0)
    res0 = A.analyze(df0, family_map=fmap0, primary_beta=0.0, reps=REPS,
                     rng_seed=A.DEFAULT_RNG_SEED)
    H0 = res0["point_estimates"]["H"]
    assert H0["H_preserve"] == 0.0, f"H_preserve must be exactly 0 when KL==SFT, got {H0['H_preserve']!r}"
    assert H0["H_cost"] == 0.0, f"H_cost must be exactly 0 when KL==SFT, got {H0['H_cost']!r}"
    st0 = res0["bootstrap"]["stats"]
    assert st0["H_preserve"]["lcb975_one_sided"] == 0.0
    assert res0["claim_checks"]["RQ2"]["supported"] is False
    # every per-checkpoint preservation is exactly zero too
    for ck in res0["point_estimates"]["per_checkpoint"].values():
        assert ck["represented"]["P"] == 0.0 and ck["heldout"]["P"] == 0.0
    print("  [ok] beta=0 identity: H_preserve == 0.0 EXACTLY (and every P == 0.0)")


if __name__ == "__main__":
    print("=== analyze_starting_type_adaptation tests (CPU, synthetic) ===")
    test_predicates_and_gates_supported()
    test_headroom_normalized_reported()
    test_interpretation_is_predicate_driven_only()
    test_gates_read_the_registered_purpose_built_panel()
    test_panel_split_and_gamma_are_exact()
    test_beta0_identity_preserve_zero_exact()
    print("ALL PASSED")
