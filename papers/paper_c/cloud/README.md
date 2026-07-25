# Paper C GCP runbook

This runbook is intentionally separate from the repository's older fleet
launchers. It never copies `.env`, never injects `OPENAI_API_KEY`, and never puts
`HF_TOKEN` directly in VM metadata.

## Security model

1. Authenticate `gcloud` locally with your user account.
2. Use a dedicated Paper C VM service account.
3. Grant it only:
   - read/write access to the Paper C bucket prefix;
   - Secret Manager access to one Hugging Face token secret.
4. The startup script reads that secret into the process environment, warms the
   exact model cache, unsets it after the run, and never prints it.
5. `OPENAI_API_KEY` is unused and must not be sent to the VM.
6. The smoke VM has a two-hour hard limit, deletes itself on termination, and
   uploads best-effort failure artifacts through an early exit trap.
7. The guest verifies the uploaded bundle checksum before extraction, installs
   `environment/gpu-requirements.txt`, and proves CUDA visibility before a model
   is downloaded.

The repository `.env` currently supplies an HF token for local development but
does not contain GCP credentials. GCP authentication must come from `gcloud` and
the VM service account/ADC.

## Required non-secret variables

```bash
export PAPER_C_GCP_PROJECT=...
export PAPER_C_GCS_BUCKET=gs://.../paper-c
export PAPER_C_GCP_SERVICE_ACCOUNT=paper-c-runner@...iam.gserviceaccount.com
export PAPER_C_HF_SECRET=paper-c-hf-token
```

Run `bash cloud/preflight.sh`, then build/upload a bundle. Launching a VM requires
the explicit cost acknowledgement `PAPER_C_ACKNOWLEDGE_COST=YES`.

`a2-highgpu-1g` includes its A100; the launcher intentionally does not attach a
second accelerator. The VM uses `--instance-termination-action=DELETE`, and the
boot disk is auto-deleted with it. `monitor_smoke.sh` downloads and verifies the
result archive, then explicitly deletes the instance and attached boot disk.

The first authorized compute job is one three-objective smoke cell. Do not
launch the 20 Stage-1 plus 120 Stage-2 panel until the smoke artifacts pass,
the design lock is valid, reference/development scoring is complete, and the
post-selection lock implementation has been exercised on the full inventory.
