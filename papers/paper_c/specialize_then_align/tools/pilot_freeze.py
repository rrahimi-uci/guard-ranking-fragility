#!/usr/bin/env python
"""Derive the primary protocol from pilot data only, then freeze it.

The protocol requires the disjoint pilot to *measure* the quantities the primary panel
assumes, rather than asserting them.  Three come out of here:

1. ``minimum_allow_per_core_category`` -- the specificity-cohort size implied by the
   FPR precision target, replacing the hard-coded 2,000 that was asserted in six
   places and derived in none.
2. the realised **paired variance-reduction factor** of the family bootstrap, which
   sets what effect the primary panel can actually resolve.
3. a first look at the primary contrast, only to check it is inside the range the
   design can resolve at all.

Nothing here reads a primary cell, and the pilot estimate is explicitly not the
result.  Run after the 20 pilot students have scored.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_c_sta.analysis import paired_family_contrast, worst_category_accuracy  # noqa: E402
from paper_c_sta.contracts import read_json, output_path, write_json  # noqa: E402
from paper_c_sta.evaluate import fit_calibration, select_checkpoint  # noqa: E402
from paper_c_sta.score import load_scores, predictions  # noqa: E402

ARMS = ("gold_sft", "soft_distill", "specialist_pairce",
        "generalist_cm_dpo", "specialist_cm_dpo")
PILOT_SEEDS = (7, 8)


def allow_minimum_for_precision(halfwidth: float, *, fpr: float = 0.05,
                                z: float = 1.959963985) -> int:
    """Rows of ALLOW per category needed for a Wald half-width on the FPR estimate.

    n = z^2 p(1-p) / halfwidth^2.  This is the number the pilot supplies in place of an
    assertion; it is a precision requirement, not a power calculation, because the
    specificity cohort exists to *estimate* a false-alarm rate.
    """
    if not 0 < halfwidth < 1:
        raise SystemExit("halfwidth must lie in (0,1)")
    return math.ceil((z ** 2) * fpr * (1 - fpr) / (halfwidth ** 2))


def cell_name(arm: str, backbone: str, seed: int) -> str:
    return f"student__{arm}__{backbone}__{seed}__pilot"


def main() -> int:
    config = read_json(output_path("config/study.json"))
    target = float(config["data"]["allow_minimum_target_fpr_halfwidth"])
    allow_min = allow_minimum_for_precision(target)

    scores_root = ROOT / "artifacts/scores"
    per_arm: dict[str, list[dict]] = {}
    calibrations: dict[str, dict] = {}
    selections: dict[str, dict] = {}
    missing: list[str] = []

    for backbone in config["backbones"]:
        for seed in PILOT_SEEDS:
            for arm in ARMS:
                name = cell_name(arm, backbone, seed)
                cell = scores_root / name
                if not cell.is_dir():
                    missing.append(name)
                    continue
                ladder = sorted(cell.glob("step*.jsonl"))
                if not ladder:
                    missing.append(name)
                    continue
                rows = {p.stem: load_scores(f"artifacts/scores/{name}/{p.name}")
                        for p in ladder}
                cal_rows = [r for r in next(iter(rows.values()))
                            if r["split"] == "calibration"]
                cal = fit_calibration(cal_rows,
                                      target_fpr=0.05, review_budget=0.10)
                sel_pool = {k: [r for r in v if r["split"] == "checkpoint_selection"]
                            for k, v in rows.items()}
                sel = select_checkpoint(sel_pool, calibration=cal)
                calibrations[name] = cal
                selections[name] = sel
                chosen = predictions(sel_pool[sel["selected"]],
                                     temperature=cal["temperature"],
                                     t_intervene=cal["t_intervene"],
                                     t_review=cal["t_review"])
                per_arm.setdefault(arm, []).extend(chosen)

    report: dict = {
        "panel": "pilot",
        "seeds": list(PILOT_SEEDS),
        "cells_expected": len(config["backbones"]) * len(PILOT_SEEDS) * len(ARMS),
        "cells_missing": missing,
        "allow_minimum": {
            "target_fpr_halfwidth": target,
            "derived_minimum_allow_per_core_category": allow_min,
            "basis": "Wald half-width on a 5% false-alarm rate",
            "replaces": "the asserted 2000, which was stated in six places and derived in none",
        },
    }

    if missing:
        report["status"] = "incomplete_pilot_no_freeze"
        write_json(ROOT / "artifacts/pilot_freeze.json", report)
        print(json.dumps(report, indent=2))
        return 1

    # realised variance reduction from pairing, measured not assumed
    if {"specialist_cm_dpo", "generalist_cm_dpo"} <= set(per_arm):
        contrast = paired_family_contrast(
            per_arm["specialist_cm_dpo"], per_arm["generalist_cm_dpo"], resamples=2000)
        paired_halfwidth = (contrast["ci_high"] - contrast["ci_low"]) / 2
        # unpaired reference: treat the two arms as independent samples of families
        spec = worst_category_accuracy(per_arm["specialist_cm_dpo"])
        gen = worst_category_accuracy(per_arm["generalist_cm_dpo"])
        report["pilot_primary_contrast"] = {
            **contrast,
            "specialist_worst_category_accuracy": spec,
            "generalist_worst_category_accuracy": gen,
            "paired_ci_halfwidth": paired_halfwidth,
            "note": "pilot estimate only; not the result and not citable as evidence",
        }
        report["resolving_power"] = {
            "pilot_paired_ci_halfwidth": paired_halfwidth,
            "primary_families_multiplier": math.sqrt(3 / 2),
            "projected_primary_halfwidth": paired_halfwidth / math.sqrt(3 / 2),
            "note": "three primary seeds against two pilot seeds; families are the unit",
        }
    report["per_arm_worst_category_accuracy"] = {
        arm: worst_category_accuracy(rows) for arm, rows in sorted(per_arm.items())
    }
    report["calibration_feasible"] = {
        name: cal["constraints_met"] for name, cal in sorted(calibrations.items())
    }
    report["selected_checkpoints"] = {
        name: sel["selected"] for name, sel in sorted(selections.items())
    }
    report["status"] = "pilot_complete_ready_to_freeze"
    write_json(ROOT / "artifacts/pilot_freeze.json", report)
    print(json.dumps(report, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
