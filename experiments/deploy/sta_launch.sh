#!/bin/bash
# Reproducible fleet launch for the starting-type adaptation study (proposal papers/unified-report).
# One VM per checkpoint; GPU-sized by params (>=3B -> A100-80GB a2-ultragpu-1g; <=2B -> A100-40GB
# a2-highgpu-1g). Each VM runs experiments/deploy/sta_startup.sh: preflight -> train (11 cells:
# 1 unmodified + 5 sft + 5 kl_sft@primary_beta) -> eval (per-checkpoint parquet) -> upload -> stop.
#
# Prereqs: gcloud auth; a HF_TOKEN file (gated gemma/llama/ai2 licenses already accepted) at
# $HF_TOKEN_FILE; the bundle uploaded to $BUCKET/bundle.tar.gz (see build step below).
#
#   export BUCKET=gs://jazztest-bucket/sta HF_TOKEN_FILE=/tmp/hftok
#   # build+upload bundle (from repo root):
#   tar czf /tmp/sta_bundle.tar.gz experiments/*.py configs/*.yaml \
#       artifacts/paper_a_sft_v2/manifests artifacts/paper_a_sft_v2/LOCK.json \
#       requirements-starting-type-adaptation.txt
#   gsutil cp /tmp/sta_bundle.tar.gz $BUCKET/bundle.tar.gz
#   bash experiments/deploy/sta_launch.sh            # launches all 10
#   bash experiments/deploy/sta_launch.sh KEY1 KEY2  # launches a subset
set -euo pipefail
PROJECT=${PROJECT:-jazzx-gcp-poc-1}
BUCKET=${BUCKET:-gs://jazztest-bucket/sta}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-/tmp/hftok}
IMAGE=${IMAGE:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260713}
IMAGE_PROJECT=${IMAGE_PROJECT:-deeplearning-platform-release}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# checkpoint -> "machine:accel:zone"  (>=3B on 80GB; small on 40GB)
declare -A PLAN=(
  [qwen25_15b]="a2-highgpu-1g:nvidia-tesla-a100:us-central1-f"
  [smollm2_17b]="a2-highgpu-1g:nvidia-tesla-a100:us-central1-f"
  [qwen3guard_gen_06b]="a2-highgpu-1g:nvidia-tesla-a100:us-central1-f"
  [llama_guard_3_1b]="a2-highgpu-1g:nvidia-tesla-a100:us-central1-f"
  [shieldgemma_2b]="a2-highgpu-1g:nvidia-tesla-a100:us-central1-c"
  [granite_guardian_31_2b]="a2-highgpu-1g:nvidia-tesla-a100:us-central1-c"
  [smollm3_3b]="a2-ultragpu-1g:nvidia-a100-80gb:us-central1-a"
  [qwen3_4b]="a2-ultragpu-1g:nvidia-a100-80gb:us-central1-a"
  [qwen3guard_gen_4b]="a2-ultragpu-1g:nvidia-a100-80gb:us-central1-a"
  [wildguard_7b]="a2-ultragpu-1g:nvidia-a100-80gb:us-central1-b"
)
KEYS=("$@"); [ ${#KEYS[@]} -eq 0 ] && KEYS=("${!PLAN[@]}")
for MK in "${KEYS[@]}"; do
  IFS=: read -r MACHINE ACCEL ZONE <<< "${PLAN[$MK]}"
  VM="sta-$(echo "$MK" | tr '_' '-')"
  echo "=== launching $VM ($MK) $MACHINE/$ACCEL @ $ZONE ==="
  gcloud compute instances create "$VM" --project="$PROJECT" --zone="$ZONE" \
    --machine-type="$MACHINE" --accelerator="type=$ACCEL,count=1" \
    --maintenance-policy=TERMINATE --provisioning-model=STANDARD \
    --image="$IMAGE" --image-project="$IMAGE_PROJECT" \
    --boot-disk-size=200GB --scopes=cloud-platform \
    --metadata="model-key=$MK,bucket=$BUCKET" \
    --metadata-from-file="startup-script=$HERE/sta_startup.sh,hf-token=$HF_TOKEN_FILE" \
    || echo "  [WARN] $VM launch failed (capacity?) -- retry a different zone"
done
