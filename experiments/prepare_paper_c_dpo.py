#!/usr/bin/env python
"""Prepare the frozen Stage-2 split and selections for Paper C v2.

This script is intentionally model-free.  Reference logits must be produced by
the lock-bound Stage-1 scorer, then this command deterministically converts them
into the uncertain and matched-random selections.

Commands:
  partition --config C --train-manifest M --out split.jsonl
  select    --config C --partition split.jsonl --reference ref.jsonl
            --stage1-adapter-sha256 SHA --out selection.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import paper_c_dpo_common as P  # noqa: E402


def _config(path: str) -> dict:
    config = P.read_json(path)
    P.validate_config(config)
    return config


def cmd_partition(args: argparse.Namespace) -> int:
    config = _config(args.config)
    train_rows = P.read_jsonl(args.train_manifest)
    stage2 = config["stage2"]
    partition_rows = P.family_partition(
        train_rows,
        development_fraction=float(stage2["development_fraction"]),
        seed=int(stage2["development_split_seed"]),
    )
    P.write_jsonl(args.out, partition_rows)
    metadata = {
        "schema_version": P.SCHEMA_VERSION,
        "algorithm_version": P.PARTITION_ALGORITHM_VERSION,
        "config_sha256": P.canonical_sha256(config),
        "train_manifest_sha256": P.sha256_file(args.train_manifest),
        "partition_sha256": P.sha256_ordered(partition_rows),
        "n_rows": len(partition_rows),
        "counts": {
            partition: sum(row["stage2_partition"] == partition
                           for row in partition_rows)
            for partition in P.PARTITIONS
        },
    }
    P.write_json(str(args.out) + ".metadata.json", metadata)
    print(
        f"[paper-c partition] rows={len(partition_rows)} "
        f"update={metadata['counts']['stage2_update']} "
        f"dev={metadata['counts']['stage2_dev']} -> {args.out}")
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    config = _config(args.config)
    partition_rows = P.read_jsonl(args.partition)
    reference_rows = P.read_jsonl(args.reference)
    stage2 = config["stage2"]
    selection_rows = P.build_selections(
        partition_rows,
        reference_rows,
        uncertain_fraction=float(stage2["uncertain_fraction"]),
        seed=int(stage2["selection_seed"]),
    )
    P.write_jsonl(args.out, selection_rows)
    metadata = P.selection_metadata(
        config=config,
        train_manifest_sha256=args.train_manifest_sha256,
        stage1_adapter_sha256=args.stage1_adapter_sha256,
        reference_sha256=P.sha256_file(args.reference),
        partition_rows=partition_rows,
        selection_rows=selection_rows,
    )
    metadata.update({
        "model_key": args.model_key,
        "seed": int(args.seed),
        "decision_tokens": P.read_json(args.reference_metadata).get("decision_tokens"),
        "prompt_fingerprint_sha256": P.read_json(
            args.reference_metadata).get("prompt_fingerprint_sha256"),
        "reference_metadata_sha256": P.sha256_file(args.reference_metadata),
        "partition_file_sha256": P.sha256_file(args.partition),
        "reference_file_sha256": P.sha256_file(args.reference),
        "selection_file_sha256": P.sha256_file(args.out),
    })
    P.write_json(str(args.out) + ".metadata.json", metadata)
    print(
        f"[paper-c select] uncertain={metadata['counts']['uncertain']} "
        f"matched_random={metadata['counts']['matched_random']} -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic Paper C v2 split/selection artifacts")
    sub = parser.add_subparsers(dest="command", required=True)

    partition = sub.add_parser("partition", help="freeze the family-disjoint Stage-2 split")
    partition.add_argument("--config", required=True)
    partition.add_argument("--train-manifest", required=True)
    partition.add_argument("--out", required=True)
    partition.set_defaults(func=cmd_partition)

    select = sub.add_parser("select", help="build uncertain and matched-random selections")
    select.add_argument("--config", required=True)
    select.add_argument("--partition", required=True)
    select.add_argument("--reference", required=True)
    select.add_argument("--reference-metadata", required=True)
    select.add_argument("--model-key", required=True)
    select.add_argument("--seed", required=True, type=int)
    select.add_argument("--train-manifest-sha256", required=True)
    select.add_argument("--stage1-adapter-sha256", required=True)
    select.add_argument("--out", required=True)
    select.set_defaults(func=cmd_select)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (P.PaperCContractError, OSError, ValueError) as exc:
        print(f"[paper-c prepare] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
