#!/usr/bin/env python
"""Validate the study registry and the distribution ledger.

Schema validation is necessary but not sufficient, so this also enforces the
cross-cutting rules the layout plan states in prose:

* every declared path exists (a registry that points at nothing is worse than none);
* `supersedes` and predecessor/successor edges resolve, and `supersedes` is acyclic;
* an `expected_pass` verification command that actually fails is an error, and an
  `expected_fail` must carry a reason -- otherwise a real regression hides behind a
  known one;
* `claim_authorization: true` requires evidence beyond development;
* a source may reach `publish_text` only with an affirmatively redistributable
  license, so an unresolved source cannot be published by omission.

Usage:
    python tools/validate_registries.py            # schema + structure + paths
    python tools/validate_registries.py --run-verification   # also execute commands
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUDIES = ROOT / "studies"
BENCHMARKS = ROOT / "benchmarks/registry"

PATH_FIELDS = ("code_root", "environment_path", "status_path")
PATH_LIST_FIELDS = ("config_paths", "protocol_paths", "artifact_roots", "manuscript_paths")


def load_yaml(path: pathlib.Path):
    try:
        import yaml
    except ImportError:
        print("PyYAML is required: pip install pyyaml", file=sys.stderr)
        raise SystemExit(2)
    return yaml.safe_load(path.read_text())


def schema_check(document, schema_path: pathlib.Path, label: str) -> list[str]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return [f"{label}: jsonschema not installed; schema NOT checked"]
    validator = Draft202012Validator(json.loads(schema_path.read_text()))
    return [
        f"{label}: {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
    ]


def check_studies(registry) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    studies = {s["study_id"]: s for s in registry["studies"]}
    if len(studies) != len(registry["studies"]):
        errors.append("studies: duplicate study_id")

    for sid, study in studies.items():
        for field in PATH_FIELDS:
            value = study.get(field)
            if value and not (ROOT / value).exists():
                errors.append(f"{sid}.{field}: missing path {value}")
        for field in PATH_LIST_FIELDS:
            for value in study.get(field) or []:
                if not (ROOT / value).exists():
                    errors.append(f"{sid}.{field}: missing path {value}")

        for field in ("predecessor", "successor"):
            target = study.get(field)
            if target and target not in studies:
                errors.append(f"{sid}.{field}: unknown study_id {target}")
        for target in study.get("supersedes") or []:
            if target not in studies:
                errors.append(f"{sid}.supersedes: unknown study_id {target}")
        for field in ("consumes", "produces"):
            for target in study.get(field) or []:
                if target not in studies:
                    errors.append(f"{sid}.{field}: unknown study_id {target}")

        if study.get("claim_authorization") and study.get("evidence_state") in (
            "none", "development_only"
        ):
            errors.append(
                f"{sid}: claim_authorization=true with evidence_state="
                f"{study['evidence_state']} -- development output is not evidence"
            )
        if study.get("expected_verification_status") == "expected_fail" and not (
            study.get("verification_failure_reason") or ""
        ).strip():
            errors.append(f"{sid}: expected_fail without verification_failure_reason")

        manifest = study.get("external_object_manifest")
        if manifest and not (ROOT / manifest).exists():
            warnings.append(f"{sid}: external_object_manifest not yet created ({manifest})")

    # supersedes must be acyclic
    graph = {sid: list(s.get("supersedes") or []) for sid, s in studies.items()}
    state: dict[str, int] = {}

    def walk(node: str, trail: list[str]) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            errors.append(f"supersedes cycle: {' -> '.join(trail + [node])}")
            return
        state[node] = 1
        for nxt in graph.get(node, []):
            walk(nxt, trail + [node])
        state[node] = 2

    for sid in graph:
        walk(sid, [])
    return errors, warnings


def check_distribution(ledger) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for source in ledger["sources"]:
        sid = source["source_id"]
        if sid in seen:
            errors.append(f"distribution: duplicate source_id {sid}")
        seen.add(sid)
        decision = source["redistribution_decision"]
        permits = source["license"]["permits_redistribution"]
        if decision == "publish_text" and permits is not True:
            errors.append(
                f"distribution/{sid}: publish_text with permits_redistribution={permits!r}"
            )
        if decision in ("local_only", "text_free_only") and not (
            source.get("blocking_reason") or ""
        ).strip():
            warnings.append(f"distribution/{sid}: non-publishing decision without blocking_reason")
    publishable = [s["source_id"] for s in ledger["sources"]
                   if s["redistribution_decision"] == "publish_text"]
    if not publishable:
        warnings.append(
            "distribution: no source is approved for publish_text; "
            "no public text build is authorized"
        )
    return errors, warnings


def run_verifications(registry) -> list[str]:
    """Execute declared commands and compare against the declared expectation."""
    problems: list[str] = []
    for study in registry["studies"]:
        command = study.get("verification_command")
        expected = study.get("expected_verification_status")
        if not command or expected == "not_applicable":
            continue
        proc = subprocess.run(command, shell=True, cwd=ROOT,
                              capture_output=True, text=True, timeout=1800)
        passed = proc.returncode == 0
        if expected == "expected_pass" and not passed:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
            problems.append(
                f"{study['study_id']}: declared expected_pass but FAILED -- {tail[0][:140]}"
            )
        elif expected == "expected_fail" and passed:
            problems.append(
                f"{study['study_id']}: declared expected_fail but PASSED -- "
                "update the registry rather than leaving a stale blocker"
            )
        print(f"  {'ok ' if passed else 'FAIL'} [{expected:14s}] {study['study_id']}", flush=True)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-verification", action="store_true",
                        help="execute each declared verification_command")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    registry = load_yaml(STUDIES / "registry.yaml")
    errors += schema_check(registry, STUDIES / "registry.schema.json", "studies")
    e, w = check_studies(registry)
    errors += e
    warnings += w

    ledger = load_yaml(BENCHMARKS / "distribution.yaml")
    errors += schema_check(ledger, BENCHMARKS / "distribution.schema.json", "distribution")
    e, w = check_distribution(ledger)
    errors += e
    warnings += w

    print(f"studies: {len(registry['studies'])}   sources: {len(ledger['sources'])}")

    if args.run_verification:
        print("\nrunning declared verification commands:")
        errors += run_verifications(registry)

    for warning in warnings:
        print(f"WARN  {warning}")
    for error in errors:
        print(f"ERROR {error}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
