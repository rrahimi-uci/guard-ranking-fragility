#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${PAPER_C_ACKNOWLEDGE_COST:?set PAPER_C_ACKNOWLEDGE_COST=YES to create a billed VM}"
if [[ "$PAPER_C_ACKNOWLEDGE_COST" != "YES" ]]; then
  echo "PAPER_C_ACKNOWLEDGE_COST must equal YES" >&2
  exit 2
fi
: "${PAPER_C_GCP_PROJECT:?set PAPER_C_GCP_PROJECT}"
: "${PAPER_C_GCS_BUCKET:?set PAPER_C_GCS_BUCKET}"
: "${PAPER_C_GCP_SERVICE_ACCOUNT:?set PAPER_C_GCP_SERVICE_ACCOUNT}"
: "${PAPER_C_HF_SECRET:?set PAPER_C_HF_SECRET}"

MODEL_KEY="${PAPER_C_MODEL_KEY:-qwen25_15b}"
SEED="${PAPER_C_SEED:-42}"
ZONE="${PAPER_C_GCP_ZONE:-us-central1-a}"
MACHINE="${PAPER_C_GCP_MACHINE_TYPE:-a2-highgpu-1g}"
IMAGE="${PAPER_C_GCP_IMAGE:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260722}"
IMAGE_PROJECT="${PAPER_C_GCP_IMAGE_PROJECT:-deeplearning-platform-release}"
RUN_ID="paper-c-smoke-${MODEL_KEY//_/-}-${SEED}-$(date -u +%Y%m%d%H%M%S)"
BUNDLE_URI="$PAPER_C_GCS_BUCKET/bundles/$RUN_ID.tar.gz"

bash cloud/preflight.sh
bash cloud/build_bundle.sh
gcloud storage cp cloud/bundle/paper-c-smoke.tar.gz "$BUNDLE_URI"
gcloud storage cp cloud/bundle/paper-c-smoke.tar.gz.sha256 "$BUNDLE_URI.sha256"

gcloud compute instances create "$RUN_ID" \
  --project="$PAPER_C_GCP_PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --max-run-duration=2h \
  --instance-termination-action=DELETE \
  --no-restart-on-failure \
  --image="$IMAGE" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size=250GB \
  --service-account="$PAPER_C_GCP_SERVICE_ACCOUNT" \
  --scopes=cloud-platform \
  --labels=study=paper-c,phase=smoke \
  --metadata="paper-c-bundle=$BUNDLE_URI,paper-c-bucket=$PAPER_C_GCS_BUCKET,paper-c-hf-secret=$PAPER_C_HF_SECRET,paper-c-project=$PAPER_C_GCP_PROJECT,paper-c-model=$MODEL_KEY,paper-c-seed=$SEED" \
  --metadata-from-file="startup-script=$ROOT/cloud/startup.sh"

echo "created $RUN_ID; monitoring until artifacts arrive or the VM exits"
bash cloud/monitor_smoke.sh "$RUN_ID"
