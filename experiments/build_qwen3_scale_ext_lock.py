#!/usr/bin/env python
"""Seal an *extension* lock adding the Qwen3 scale ladder (8B, 32B) to the panel spec.

    python experiments/build_qwen3_scale_ext_lock.py

WHY A SIDECAR LOCK, AND WHY IT CANNOT BE VERIFIED. Act~I's estimand is literally
`fixed_panel_mean over 4 checkpoints`, and the fixed panel is enforced *in code*:
`_verify_strict_lock_structure` asserts `set(models) == set(MODEL_KEYS)`, where MODEL_KEYS
comes from `paper_a_common.MODEL_PANEL`. That module's sha256 is recorded in
`RELEASE.json`, and `configs/paper_a_sft.yaml` is bound by `verify-lock`. So a fifth
checkpoint cannot be added anywhere without either changing the locked estimand or breaking
release verification -- which means no extended lock can pass `C.load_lock`, by design.

This sidecar therefore exists to be consumed *directly as a dict* by the canonical
primitives (`train_one_cell`, `score_bundle`, `base_run_meta`), all of which resolve their
model spec through `lock_model_panel(lock)` and do not re-verify. The extension runner
documents that it deliberately bypasses `verify_lock`. Nothing bound to the release is
touched: this script asserts the released lock's self-hash is unchanged before exiting.

WHAT IS COPIED VERBATIM. Recipe, prompt spec, data/order seeds, operating point, manifest
paths and `train_manifest_sha256`. That byte-level parity is the entire reason an 8B or 32B
number is comparable to the four panel checkpoints: same 1,200 training rows, same LoRA
recipe, same margin and calibration primitives.

A PROPERTY WORTH NOTING. Qwen3-4B, Qwen3-8B and Qwen3-32B all render the frozen guard
prompt to the *same* `prompt_template_sha256` (d5547c01...) and select the same single-token
decision pair (` safe`=6092 / ` unsafe`=19860). Across the ladder the prompt is byte-identical,
so base scale is the only quantity varying -- verified before sealing, not assumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

SRC_LOCK = REPO / "artifacts/paper_a_sft_v2/LOCK.json"
OUT_DIR = REPO / "artifacts/qwen3_scale_ext"
OUT_LOCK = OUT_DIR / "LOCK.json"

# Shared by the whole Qwen3 family; verified against each tokenizer at the pinned revision.
QWEN3_TEMPLATE_SHA = "d5547c019ac90e781429e280d7366d2c796e334514d04bc87e49b98509da8795"

EXT_MODELS = {
    "qwen3_8b": {
        "model_id": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "template_sha256": QWEN3_TEMPLATE_SHA,
        "weights_gb": 16.4,
    },
    "qwen3_32b": {
        "model_id": "Qwen/Qwen3-32B",
        "model_revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "template_sha256": QWEN3_TEMPLATE_SHA,
        "weights_gb": 65.5,
    },
}


def build() -> dict:
    import paper_a_common as C

    lock = json.loads(SRC_LOCK.read_text())
    src_sha = lock["lock_sha256"]

    models = dict(lock.get("models") or {})
    prompt = dict(lock.get("prompt") or {})
    per_model = dict(prompt.get("per_model_template_sha256") or {})
    for key, spec in EXT_MODELS.items():
        models[key] = {
            "model_id": spec["model_id"],
            "model_revision": spec["model_revision"],
            "tokenizer_revision": spec["model_revision"],
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "attn_implementation": None,
        }
        per_model[key] = spec["template_sha256"]
    prompt["per_model_template_sha256"] = per_model
    lock["models"] = models
    lock["prompt"] = prompt

    lock["n_checkpoints"] = len(models)
    lock["study_id"] = "qwen3_scale_extension"
    # Describes the lock *contract*, and the verifier admits only "final" or
    # "development_unverified". This sidecar is fully specified and sealed, so "final" is
    # accurate; what marks the runs non-canonical is run_kind="nonfinal" on every cell plus
    # study_id and extension_of here -- the same combination the KL-SFT sweep used.
    lock["finalization_status"] = "final"
    lock["artifact_paths"] = {**(lock.get("artifact_paths") or {}),
                              "root": "artifacts/qwen3_scale_ext",
                              "runs": "artifacts/qwen3_scale_ext/runs",
                              "scores": "artifacts/qwen3_scale_ext/scores"}
    lock["extension_of"] = {
        "path": "artifacts/paper_a_sft_v2/LOCK.json",
        "lock_sha256": src_sha,
        "added_model_keys": sorted(EXT_MODELS),
        "note": ("Sidecar lock for a scale-ladder extension. Recipe, prompt spec, seeds, "
                 "manifests and train_manifest_sha256 are copied verbatim from the release "
                 "lock so extension checkpoints are trained and scored under identical "
                 "contracts. The release lock and every file it binds are unmodified. This "
                 "lock is NOT claim-bearing for Act I, cannot pass "
                 "_verify_strict_lock_structure (which pins the four-checkpoint panel in "
                 "code), and must never be given to the canonical analyzer."),
    }

    lock.pop("lock_sha256", None)
    lock["lock_sha256"] = C.canonical_obj_sha256(lock)
    return lock


def main() -> int:
    import paper_a_common as C

    if not SRC_LOCK.is_file():
        print(f"missing {SRC_LOCK}", file=sys.stderr)
        return 1
    lock = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")

    def selfhash(obj):
        return C.canonical_obj_sha256({k: v for k, v in obj.items() if k != "lock_sha256"})

    written = json.loads(OUT_LOCK.read_text())
    assert selfhash(written) == written["lock_sha256"], "sidecar self-hash does not verify"
    released = json.loads(SRC_LOCK.read_text())
    assert selfhash(released) == released["lock_sha256"], "released LOCK.json self-hash broke!"
    assert released["lock_sha256"] == lock["extension_of"]["lock_sha256"]

    print(f"wrote {OUT_LOCK.relative_to(REPO)}")
    print(f"  study_id       {lock['study_id']}")
    print(f"  models         {', '.join(sorted(lock['models']))}")
    print(f"  added          {', '.join(lock['extension_of']['added_model_keys'])}")
    print(f"  lock_sha256    {lock['lock_sha256'][:16]}…")
    print(f"  extension_of   {lock['extension_of']['lock_sha256'][:16]}…")
    print(f"  train_manifest {lock['train_manifest_sha256'][:16]}… (copied verbatim)")
    print("  released LOCK.json verified byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
