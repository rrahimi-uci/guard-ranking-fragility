# Environment record

Phase 4 requires a pinned runtime for the environment that produced a claimed result.
**No result is claimed here**, so this records what the pilot actually ran on rather
than certifying a release environment.

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

New local execution belongs in ignored `runs/paper-c-specialize-align-mortgage-v1/`,
not under this package. Existing development artefacts are inventoried in
`provenance/EXTERNAL_OBJECT_MANIFEST.json` and remain development-only.
