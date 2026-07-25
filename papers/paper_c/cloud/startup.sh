#!/usr/bin/env bash
set -Eeuo pipefail

WORKDIR="/opt/paper_c"
LOG="$WORKDIR/cloud-run.log"
RUN_BUCKET=""
RUN_ID="$(hostname)"

mkdir -p "$WORKDIR/artifacts/smoke"
exec > >(tee -a "$LOG") 2>&1

cleanup() {
  local run_status=$?
  trap - EXIT
  set +e
  unset HF_TOKEN
  printf '{"guest_exit_status":%d}\n' "$run_status" \
    > "$WORKDIR/artifacts/smoke/guest_exit_status.json"
  if [[ -n "$RUN_BUCKET" ]]; then
    tar -czf "$WORKDIR/$RUN_ID-results.tar.gz" -C "$WORKDIR" \
      artifacts/smoke cloud-run.log
    RESULT_SHA256="$(sha256sum "$WORKDIR/$RUN_ID-results.tar.gz" | awk '{print $1}')"
    printf '%s  %s\n' "$RESULT_SHA256" "$RUN_ID-results.tar.gz" \
      > "$WORKDIR/$RUN_ID-results.tar.gz.sha256"
    gcloud storage cp "$WORKDIR/$RUN_ID-results.tar.gz" \
      "$RUN_BUCKET/runs/$RUN_ID/" || true
    gcloud storage cp "$WORKDIR/$RUN_ID-results.tar.gz.sha256" \
      "$RUN_BUCKET/runs/$RUN_ID/" || true
  fi
  sync
  shutdown -h now || true
  exit "$run_status"
}
trap cleanup EXIT

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

BUNDLE_URI="$(metadata paper-c-bundle)"
RUN_BUCKET="$(metadata paper-c-bucket)"
HF_SECRET="$(metadata paper-c-hf-secret)"
PROJECT="$(metadata paper-c-project)"
MODEL_KEY="$(metadata paper-c-model)"
SEED="$(metadata paper-c-seed)"
RUN_ID="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name)"
echo "starting Paper C smoke $RUN_ID"

gcloud storage cp "$BUNDLE_URI" "$WORKDIR/paper-c-smoke.tar.gz"
gcloud storage cp "$BUNDLE_URI.sha256" "$WORKDIR/paper-c-smoke.tar.gz.sha256"
(
  cd "$WORKDIR"
  sha256sum -c paper-c-smoke.tar.gz.sha256
)
tar -xzf "$WORKDIR/paper-c-smoke.tar.gz" -C "$WORKDIR"
cd "$WORKDIR"

python3 -m pip install --disable-pip-version-check \
  --requirement environment/gpu-requirements.txt
# The pinned DLVM currently carries torchaudio 2.11 beside torch 2.9.1. Paper C
# has no audio path, and removing it prevents Transformers from importing an
# ABI-incompatible optional extension.
python3 -m pip uninstall --yes torchaudio

nvidia-smi | tee artifacts/smoke/nvidia-smi.txt
python3 -c 'import torch; assert torch.__version__.startswith("2.9."); assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
python3 -m pip freeze > artifacts/smoke/python-freeze.txt

HF_TOKEN="$(gcloud secrets versions access latest --secret="$HF_SECRET" --project="$PROJECT")"
if [[ -z "$HF_TOKEN" ]]; then
  echo "Paper C Hugging Face secret is empty" >&2
  exit 2
fi
export HF_TOKEN

bash cloud/run_smoke.sh "$MODEL_KEY" "$SEED"
echo "smoke completed"
