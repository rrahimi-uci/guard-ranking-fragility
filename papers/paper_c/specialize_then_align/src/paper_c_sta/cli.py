"""Small command line surface for design validation and locking."""

from __future__ import annotations

import argparse
import json

from .contracts import expected_training_grid, load_config, readiness_blockers
from .locks import create_protocol_candidate, validate_protocol_candidate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-c-sta")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--config", default="config/study.json")
    grid = subparsers.add_parser("grid")
    grid.add_argument("--config", default="config/study.json")
    readiness = subparsers.add_parser("readiness")
    readiness.add_argument("--config", default="config/study.json")
    lock = subparsers.add_parser("lock-protocol")
    lock.add_argument("--config", default="config/study.json")
    lock.add_argument("--out", default="locks/PROTOCOL_TAXONOMY_CANDIDATE.json")
    check_lock = subparsers.add_parser("validate-lock")
    check_lock.add_argument("--path", default="locks/PROTOCOL_TAXONOMY_CANDIDATE.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        config = load_config(args.config)
        print(json.dumps({"valid": True, "study_id": config["study_id"]}, sort_keys=True))
        return 0
    if args.command == "grid":
        grid = expected_training_grid(load_config(args.config))
        print(json.dumps({key: len(value) for key, value in grid.items()}, sort_keys=True))
        return 0
    if args.command == "readiness":
        blockers = readiness_blockers(load_config(args.config))
        print(json.dumps({"ready": not blockers, "blockers": blockers}, sort_keys=True))
        return 0 if not blockers else 2
    if args.command == "lock-protocol":
        lock = create_protocol_candidate(args.config, out_path=args.out)
        print(json.dumps({"path": args.out, "lock_sha256": lock["lock_sha256"]}, sort_keys=True))
        return 0
    if args.command == "validate-lock":
        lock = validate_protocol_candidate(args.path)
        print(json.dumps({"valid": True, "lock_sha256": lock["lock_sha256"]}, sort_keys=True))
        return 0
    raise AssertionError(args.command)
