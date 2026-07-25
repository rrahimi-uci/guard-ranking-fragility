"""Authoritative producers for Stage-1, Stage-2 input, candidate, and score inventories."""

from __future__ import annotations

from pathlib import Path

from .contracts import (
    ContractError,
    OBJECTIVES,
    SAMPLERS,
    output_path,
    read_json,
    sha256_directory,
    sha256_file,
    write_jsonl,
)


def _completed_metadata(path: Path, expected_kind: str) -> dict:
    if not path.is_file():
        raise ContractError(f"run metadata is missing: {path}")
    metadata = read_json(path)
    if metadata.get("kind") != expected_kind or metadata.get("status") != "completed":
        raise ContractError(f"run is not a completed {expected_kind}: {path}")
    return metadata


def build_stage1_inventory(
    *, config: dict, runs_root: str | Path, prompt_root: str | Path, out_path: str | Path
) -> list[dict]:
    root = output_path(runs_root)
    prompts = output_path(prompt_root)
    records = []
    for model_key in config["models"]:
        cache = prompts / f"{model_key}.train.jsonl"
        if not cache.is_file():
            raise ContractError(f"Stage-1 prompt cache is missing: {cache}")
        for seed in config["seeds"]:
            run = root / model_key / f"seed_{seed}"
            metadata_path = run / "run_metadata.json"
            metadata = _completed_metadata(metadata_path, "paper_c_stage1")
            if metadata.get("model_key") != model_key or int(metadata.get("seed", -1)) != seed:
                raise ContractError(f"Stage-1 metadata identity mismatch: {metadata_path}")
            adapter = run / "adapter"
            adapter_hash = sha256_directory(adapter)
            if metadata.get("adapter_sha256") != adapter_hash:
                raise ContractError(f"Stage-1 metadata adapter hash mismatch: {adapter}")
            records.append({
                "model_key": model_key,
                "seed": seed,
                "adapter_path": adapter.relative_to(output_path(".")).as_posix(),
                "adapter_sha256": adapter_hash,
                "run_metadata_path": metadata_path.relative_to(output_path(".")).as_posix(),
                "run_metadata_sha256": sha256_file(metadata_path),
                "prompt_cache_path": cache.relative_to(output_path(".")).as_posix(),
                "prompt_cache_sha256": sha256_file(cache),
            })
    if len(records) != 20:
        raise ContractError("Stage-1 inventory producer did not create 20 cells")
    write_jsonl(out_path, records)
    return records


def build_stage2_input_inventory(
    *, config: dict, reference_root: str | Path, selection_root: str | Path,
    out_path: str | Path,
) -> list[dict]:
    references = output_path(reference_root)
    selections = output_path(selection_root)
    records = []
    for model_key in config["models"]:
        for seed in config["seeds"]:
            reference = references / model_key / f"seed_{seed}.jsonl"
            selection = selections / model_key / f"seed_{seed}.jsonl"
            for artifact in (reference, selection):
                if not artifact.is_file():
                    raise ContractError(f"Stage-2 input artifact is missing: {artifact}")
            reference_meta = read_json(output_path(f"{reference}.metadata.json"))
            selection_meta = read_json(output_path(f"{selection}.metadata.json"))
            if reference_meta.get("mode") != "reference" or reference_meta.get("status") != "completed":
                raise ContractError(f"invalid reference metadata: {reference}")
            if selection_meta.get("model_key") != model_key or int(selection_meta.get("seed", -1)) != seed:
                raise ContractError(f"selection metadata identity mismatch: {selection}")
            records.append({
                "model_key": model_key,
                "seed": seed,
                "reference_path": reference.relative_to(output_path(".")).as_posix(),
                "reference_sha256": sha256_file(reference),
                "selection_path": selection.relative_to(output_path(".")).as_posix(),
                "selection_sha256": sha256_file(selection),
            })
    if len(records) != 20:
        raise ContractError("Stage-2 input inventory producer did not create 20 cells")
    write_jsonl(out_path, records)
    return records


def build_candidate_inventory(
    *, config: dict, runs_root: str | Path, out_path: str | Path
) -> list[dict]:
    root = output_path(runs_root)
    records = []
    for model_key in config["models"]:
        for seed in config["seeds"]:
            for sampler in SAMPLERS:
                for objective in OBJECTIVES:
                    run = root / model_key / f"seed_{seed}" / sampler / objective
                    metadata_path = run / "run_metadata.json"
                    metadata = _completed_metadata(metadata_path, "paper_c_stage2")
                    expected_identity = (model_key, seed, sampler, objective)
                    observed_identity = (
                        metadata.get("model_key"), int(metadata.get("seed", -1)),
                        metadata.get("sampler"), metadata.get("objective"),
                    )
                    if observed_identity != expected_identity:
                        raise ContractError(f"Stage-2 run identity mismatch: {metadata_path}")
                    for step in config["stage2"]["checkpoint_steps"]:
                        checkpoint = (metadata.get("checkpoints") or {}).get(str(step))
                        if not isinstance(checkpoint, dict):
                            raise ContractError(f"missing Stage-2 checkpoint {step}: {metadata_path}")
                        adapter = output_path(str(checkpoint.get("path", "")))
                        adapter_hash = sha256_directory(adapter)
                        if adapter_hash != checkpoint.get("sha256"):
                            raise ContractError(f"Stage-2 checkpoint hash mismatch: {adapter}")
                        records.append({
                            "model_key": model_key,
                            "seed": seed,
                            "sampler": sampler,
                            "objective": objective,
                            "step": int(step),
                            "adapter_path": adapter.relative_to(output_path(".")).as_posix(),
                            "adapter_sha256": adapter_hash,
                            "run_metadata_path": metadata_path.relative_to(output_path(".")).as_posix(),
                            "run_metadata_sha256": sha256_file(metadata_path),
                        })
    if len(records) != 480:
        raise ContractError("candidate inventory producer did not create 480 checkpoints")
    write_jsonl(out_path, records)
    return records


def build_development_score_inventory(
    *, config: dict, stage1_root: str | Path, candidate_root: str | Path,
    out_path: str | Path,
) -> list[dict]:
    stage1 = output_path(stage1_root)
    candidates = output_path(candidate_root)
    records = []

    def verify_score(path: Path) -> None:
        if not path.is_file():
            raise ContractError(f"development score file is missing: {path}")
        metadata = read_json(output_path(f"{path}.metadata.json"))
        if metadata.get("mode") != "stage2_dev" or metadata.get("status") != "completed":
            raise ContractError(f"invalid development score metadata: {path}")
        if metadata.get("score_sha256") != sha256_file(path):
            raise ContractError(f"development score hash mismatch: {path}")

    for model_key in config["models"]:
        for seed in config["seeds"]:
            score = stage1 / model_key / f"seed_{seed}.jsonl"
            verify_score(score)
            records.append({
                "score_kind": "stage1", "model_key": model_key, "seed": seed,
                "score_path": score.relative_to(output_path(".")).as_posix(),
                "score_sha256": sha256_file(score),
            })
            for sampler in SAMPLERS:
                for objective in OBJECTIVES:
                    for step in config["stage2"]["checkpoint_steps"]:
                        score = (
                            candidates / model_key / f"seed_{seed}" / sampler / objective
                            / f"step_{step}.jsonl"
                        )
                        verify_score(score)
                        records.append({
                            "score_kind": "candidate", "model_key": model_key,
                            "seed": seed, "sampler": sampler, "objective": objective,
                            "step": int(step),
                            "score_path": score.relative_to(output_path(".")).as_posix(),
                            "score_sha256": sha256_file(score),
                        })
    if len(records) != 500:
        raise ContractError("development score inventory producer did not create 500 bundles")
    write_jsonl(out_path, records)
    return records

