#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for command_name in gcloud tar shasum; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "missing required command: $command_name" >&2
    exit 2
  fi
done

: "${PAPER_C_GCP_PROJECT:?set PAPER_C_GCP_PROJECT}"
: "${PAPER_C_GCS_BUCKET:?set PAPER_C_GCS_BUCKET}"
: "${PAPER_C_GCP_SERVICE_ACCOUNT:?set PAPER_C_GCP_SERVICE_ACCOUNT}"
: "${PAPER_C_HF_SECRET:?set PAPER_C_HF_SECRET}"

ZONE="${PAPER_C_GCP_ZONE:-us-central1-a}"
REGION="${ZONE%-*}"
MACHINE="${PAPER_C_GCP_MACHINE_TYPE:-a2-highgpu-1g}"
IMAGE="${PAPER_C_GCP_IMAGE:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260722}"
IMAGE_PROJECT="${PAPER_C_GCP_IMAGE_PROJECT:-deeplearning-platform-release}"
PROTOCOL_LOCK="${PAPER_C_PROTOCOL_LOCK:-artifacts/locks/PROTOCOL_LOCK_SUPERSEDING_009.json}"

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -n 1)"
if [[ -z "$ACTIVE_ACCOUNT" ]]; then
  echo "gcloud has no active account" >&2
  exit 2
fi

gcloud projects describe "$PAPER_C_GCP_PROJECT" --format='value(projectId)' >/dev/null
gcloud iam service-accounts describe "$PAPER_C_GCP_SERVICE_ACCOUNT" \
  --project="$PAPER_C_GCP_PROJECT" --format='value(email)' >/dev/null
gcloud storage buckets describe "$PAPER_C_GCS_BUCKET" \
  --format='value(name)' >/dev/null
gcloud secrets describe "$PAPER_C_HF_SECRET" \
  --project="$PAPER_C_GCP_PROJECT" --format='value(name)' >/dev/null
ENABLED_SECRET_VERSION="$(gcloud secrets versions list "$PAPER_C_HF_SECRET" \
  --project="$PAPER_C_GCP_PROJECT" --filter='state=ENABLED' --limit=1 \
  --format='value(name)')"
if [[ -z "$ENABLED_SECRET_VERSION" ]]; then
  echo "Paper C Hugging Face secret has no enabled version" >&2
  exit 2
fi

gcloud compute machine-types describe "$MACHINE" --zone="$ZONE" \
  --project="$PAPER_C_GCP_PROJECT" --format='value(name)' >/dev/null
gcloud compute images describe "$IMAGE" --project="$IMAGE_PROJECT" \
  --format='value(name)' >/dev/null
QUOTAS="$(gcloud compute regions describe "$REGION" \
  --project="$PAPER_C_GCP_PROJECT" --flatten='quotas[]' \
  --format='value(quotas.metric,quotas.limit)')"
A100_QUOTA="$(awk '$1 == "NVIDIA_A100_GPUS" {print $2}' <<< "$QUOTAS")"
A2_CPU_QUOTA="$(awk '$1 == "A2_CPUS" {print $2}' <<< "$QUOTAS")"
if [[ -z "$A100_QUOTA" || -z "$A2_CPU_QUOTA" ]]; then
  echo "required A100 or A2 CPU quota is unavailable in $REGION" >&2
  exit 2
fi

PYTHONPATH="$ROOT/src" python3 -m paper_c validate-config --config config/study.json
PYTHONPATH="$ROOT/src" python3 -m paper_c validate-config --config config/smoke.json
PYTHONPATH="$ROOT/src" python3 -m paper_c validate-lock --lock "$PROTOCOL_LOCK"
echo "preflight valid; A100 quota=$A100_QUOTA, A2 CPU quota=$A2_CPU_QUOTA; no resource was created"
