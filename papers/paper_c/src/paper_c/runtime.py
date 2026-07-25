"""Self-contained tokenizer, prompt-cache, model, seed, and environment runtime."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess

from .contracts import (
    ContractError,
    canonical_sha256,
    output_path,
    read_jsonl,
    row_identity,
    row_text,
    sha256_file,
    sha256_ordered,
    write_json,
    write_jsonl,
)
from .prompting import SYSTEM_PROMPT, budgeted_prompt, prompt_template_sha256, select_decision_tokens


def repository_env_path() -> Path:
    return output_path(".").parents[1] / ".env"


def _parse_env_value(path: Path, name: str) -> str | None:
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            return value or None
    return None


def load_hf_token() -> bool:
    """Load only HF_TOKEN into this process; never source or copy the whole .env."""
    if os.environ.get("HF_TOKEN"):
        return True
    env_path = repository_env_path()
    if env_path.is_file() and env_path.stat().st_mode & 0o077:
        raise ContractError(
            "refusing to load HF_TOKEN from a group/world-readable .env; set mode 0600 or inject HF_TOKEN directly"
        )
    token = _parse_env_value(env_path, "HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        return True
    return False


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def software_versions() -> dict:
    names = ("torch", "transformers", "peft", "accelerate", "numpy", "safetensors")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions.update({
        "python": platform.python_version(),
        "platform": platform.platform(),
    })
    return versions


def doctor(config: dict) -> dict:
    env_path = repository_env_path()
    permissions = env_path.stat().st_mode & 0o777 if env_path.exists() else None
    gcloud = shutil.which("gcloud")
    gcloud_project = None
    gcloud_account_present = False
    if gcloud:
        project_result = subprocess.run(
            [gcloud, "config", "get-value", "project"],
            check=False, capture_output=True, text=True, timeout=20,
        )
        if project_result.returncode == 0:
            candidate = project_result.stdout.strip()
            gcloud_project = candidate if candidate and candidate != "(unset)" else None
        account_result = subprocess.run(
            [gcloud, "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
            check=False, capture_output=True, text=True, timeout=20,
        )
        gcloud_account_present = bool(account_result.stdout.strip())
    cuda = {"torch_present": False, "available": False, "device": None}
    try:
        import torch
        cuda["torch_present"] = True
        cuda["available"] = bool(torch.cuda.is_available())
        if cuda["available"]:
            cuda["device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {
        "study_id": config["study_id"],
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or _parse_env_value(env_path, "HF_TOKEN")),
        "openai_token_used_by_paper_c": False,
        "env_file_present": env_path.is_file(),
        "env_file_mode": oct(permissions) if permissions is not None else None,
        "env_file_mode_is_private": permissions is None or permissions & 0o077 == 0,
        "gcloud_binary_present": bool(gcloud),
        "gcloud_active_account_present": gcloud_account_present,
        "gcloud_project": gcloud_project,
        "cuda": cuda,
        "software": software_versions(),
        "ready_for_local_contract_tests": True,
        "ready_for_gpu_smoke": bool(
            (os.environ.get("HF_TOKEN") or _parse_env_value(env_path, "HF_TOKEN"))
            and cuda["available"]
        ),
    }


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    try:
        import numpy
        numpy.random.seed(int(seed))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def torch_dtype(torch_module, name: str):
    normalized = str(name).lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch_module.float16
    if normalized in {"float32", "fp32"}:
        return torch_module.float32
    raise ContractError(f"unsupported dtype: {name}")


def load_tokenizer(config: dict, model_key: str):
    if model_key not in config["models"]:
        raise ContractError(f"unknown model key: {model_key}")
    load_hf_token()
    from transformers import AutoTokenizer
    record = config["models"][model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        record["model_id"],
        revision=record["tokenizer_revision"],
        trust_remote_code=bool(record.get("trust_remote_code", False)),
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    observed = select_decision_tokens(tokenizer)
    if observed["safe_id"] != record["safe_token_id"]:
        raise ContractError(f"safe-token drift for {model_key}")
    if observed["unsafe_id"] != record["unsafe_token_id"]:
        raise ContractError(f"unsafe-token drift for {model_key}")
    return tokenizer, observed


def load_base_model(config: dict, model_key: str, *, device: str):
    load_hf_token()
    import torch
    from transformers import AutoModelForCausalLM
    record = config["models"][model_key]
    model = AutoModelForCausalLM.from_pretrained(
        record["model_id"],
        revision=record["model_revision"],
        trust_remote_code=bool(record.get("trust_remote_code", False)),
        dtype=torch_dtype(torch, record["dtype"]),
        token=os.environ.get("HF_TOKEN"),
    )
    model.config.use_cache = False
    model.to(device)
    return model


def prepare_prompt_cache(
    *, config: dict, model_key: str, manifest_path: str | Path, out_path: str | Path
) -> dict:
    """Render once, then freeze prompt token IDs to eliminate date-template drift."""
    if config["prompt"]["system"] != SYSTEM_PROMPT:
        raise ContractError("configured system prompt differs from the runtime prompt")
    tokenizer, decision = load_tokenizer(config, model_key)
    manifest_file = output_path(manifest_path)
    manifest = read_jsonl(manifest_file)
    rows = []
    fingerprints = []
    for source_row in manifest:
        sample_id, source, gold, family_id, content_hash = row_identity(source_row)
        rendered, truncation = budgeted_prompt(
            tokenizer,
            row_text(source_row),
            max_length=int(config["prompt"]["max_length"]),
            reserved_tokens=2,
        )
        input_ids = list(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        prompt_hash = canonical_sha256({"input_ids": input_ids})
        fingerprints.append({"sample_id": sample_id, "prompt_sha256": prompt_hash})
        rows.append({
            "sample_id": sample_id,
            "source": source,
            "gold": gold,
            "family_id": family_id,
            "content_sha256": content_hash,
            "input_ids": input_ids,
            "prompt_sha256": prompt_hash,
            "truncated": bool(truncation["truncated"]),
        })
    write_jsonl(out_path, rows)
    metadata = {
        "schema_version": 1,
        "kind": "frozen_prompt_token_cache",
        "created_utc": utcnow(),
        "model_key": model_key,
        "model_id": config["models"][model_key]["model_id"],
        "tokenizer_revision": config["models"][model_key]["tokenizer_revision"],
        "manifest_path": manifest_file.relative_to(output_path(".")).as_posix(),
        "manifest_sha256": sha256_file(manifest_file),
        "rows": len(rows),
        "decision_tokens": decision,
        "template_probe_sha256": prompt_template_sha256(tokenizer),
        "ordered_prompt_sha256": sha256_ordered(fingerprints),
        "cache_sha256": sha256_file(output_path(out_path)),
        "dynamic_template_rerender_allowed_after_lock": False,
    }
    write_json(f"{out_path}.metadata.json", metadata)
    return metadata


def validate_prompt_cache(
    *, config: dict, model_key: str, manifest_path: str | Path, cache_path: str | Path
) -> tuple[list[dict], dict]:
    manifest_file = output_path(manifest_path)
    manifest = read_jsonl(manifest_file)
    cache = read_jsonl(output_path(cache_path))
    metadata_path = output_path(f"{cache_path}.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(output_path(cache_path)) != metadata.get("cache_sha256"):
        raise ContractError("prompt cache bytes drifted")
    if sha256_file(manifest_file) != metadata.get("manifest_sha256"):
        raise ContractError("prompt cache was built from another manifest")
    if metadata.get("model_key") != model_key:
        raise ContractError("prompt cache model key mismatch")
    expected = [row_identity(row) for row in manifest]
    observed = [row_identity(row) for row in cache]
    if expected != observed:
        raise ContractError("prompt cache identities/order differ from manifest")
    if any(not row.get("input_ids") for row in cache):
        raise ContractError("prompt cache contains an empty token sequence")
    return cache, metadata
