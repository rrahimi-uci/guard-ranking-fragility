# Environment record

Phase 4 requires a pinned runtime for the environment that produced a claimed result.
**No result is claimed here**, so this records what the pilot actually ran on rather
than certifying a release environment.

`gpu-requirements.txt` is a **partial record, not a lock**, and says so in its header:
torch/CUDA came from the VM image rather than pip, `accelerate` was installed unpinned
and its resolved version was never captured, and the pilot A100s have all been deleted,
so no `pip freeze` is recoverable. Completing it means re-provisioning and re-running --
a new experiment with its own authorization. The practical consequence: **the GCS
development artefacts cannot be exactly reproduced from this record.** They are
development-only evidence and nothing depends on reproducing them.

| Layer | Value |
|---|---|
| GPU VM image | `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` (deeplearning-platform-release) |
| Machine type | `a2-highgpu-1g`, NVIDIA A100-SXM4-40GB |
| Python (VM) | 3.10, image default |
| torch | 2.9.1+cu129 |
| Installed | `transformers==5.12.1`, `peft==0.19.1`, `accelerate` |
| Removed | `torchaudio` — the image build's ABI does not load, and transformers 5.x imports it transitively via `loss_rnnt` |
| Local analysis | repo `.venv`, Python 3.14.4 — CPU contract tests only, never a release environment |

Backbones are pinned by revision in `config/study.json`:
`Qwen/Qwen2.5-1.5B-Instruct@989aa798...`, `HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2e...`.

## Cloud SDK

Deliberately **not** vendored here. The predecessor tree carries a 463 MB
`build/tooling/google-cloud-sdk/`, which Phase 5 requires be installed outside the
repository. `cloud/*.sh` therefore expect `gcloud`/`gsutil` on `PATH`.

## Outputs

New local execution routes to ignored `<repo>/runs/paper-c-specialize-align-mortgage-v1/`,
not under this package -- and that is now enforced rather than requested.
`contracts.runs_root()` resolves it from the repository marker, and `output_path()`
admits exactly two roots: this workspace (so committed development paths keep resolving)
and that runs directory.

An earlier version of this file promised the routing while the contract *rejected* it:
an absolute repo-level `runs/` path raised `ContractError`, and a relative `runs/x`
silently landed inside the package instead. Four tests in `tests/test_contracts.py` now
cover the admitted path, the two-root boundary, non-widening of an explicit `root=`, and
the unchanged relative-path behaviour.

Existing development artefacts are inventoried in
`provenance/EXTERNAL_OBJECT_MANIFEST.json` and remain development-only.
