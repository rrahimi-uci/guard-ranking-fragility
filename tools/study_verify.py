#!/usr/bin/env python
"""Run one study's declared verification command, looked up from the registry.

Study packages call this instead of restating their own command. A package that
restated it would be a second source of truth that drifts silently -- the whole point
of `studies/registry.yaml` is that verification is declared once.

The declared expectation is honoured: a study registered `expected_fail` exits 0 when it
fails and *non-zero when it passes*, because a stale blocker that quietly starts passing
is a finding, not a relief.

Usage:
    python tools/study_verify.py <study_id> [--print]
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_study(study_id: str) -> dict:
    import yaml

    reg = yaml.safe_load((ROOT / "studies/registry.yaml").read_text())
    for study in reg["studies"]:
        if study["study_id"] == study_id:
            return study
    known = ", ".join(s["study_id"] for s in reg["studies"])
    raise SystemExit(f"unknown study {study_id!r}. Registered: {known}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("study_id")
    ap.add_argument("--print", action="store_true", dest="show",
                    help="print the declared command without running it")
    args = ap.parse_args()

    study = load_study(args.study_id)
    command = study["verification_command"]
    expected = study["expected_verification_status"]

    if args.show:
        print(command)
        return 0

    print(f"[{args.study_id}] expecting {expected}")
    print(f"$ {command}")
    completed = subprocess.run(command, shell=True, cwd=ROOT)
    passed = completed.returncode == 0

    if expected == "expected_pass":
        if passed:
            print(f"[{args.study_id}] ok")
            return 0
        print(f"[{args.study_id}] FAIL: declared expected_pass but the command failed")
        return 1

    # expected_fail
    reason = (study.get("verification_failure_reason") or "").strip()
    if not passed:
        print(f"[{args.study_id}] ok — failed as declared")
        if reason:
            print(f"  reason on record: {reason.splitlines()[0]}")
        return 0
    print(f"[{args.study_id}] FAIL: declared expected_fail but the command PASSED. "
          "A blocker that has resolved must be re-declared in studies/registry.yaml, "
          "not left as a stale expectation.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
