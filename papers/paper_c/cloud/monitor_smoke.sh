#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${1:?run id required}"
: "${PAPER_C_GCP_PROJECT:?set PAPER_C_GCP_PROJECT}"
: "${PAPER_C_GCS_BUCKET:?set PAPER_C_GCS_BUCKET}"
ZONE="${PAPER_C_GCP_ZONE:-us-central1-a}"
RESULT_URI="$PAPER_C_GCS_BUCKET/runs/$RUN_ID/$RUN_ID-results.tar.gz"
RESULT_SHA_URI="$RESULT_URI.sha256"
LOCAL_RESULT_DIR="artifacts/cloud/$RUN_ID"
INSTANCE_PRESENT=1

mkdir -p "$LOCAL_RESULT_DIR"

delete_instance() {
  if [[ "$INSTANCE_PRESENT" -eq 1 ]]; then
    gcloud compute instances delete "$RUN_ID" \
      --project="$PAPER_C_GCP_PROJECT" --zone="$ZONE" \
      --delete-disks=all --quiet || true
    INSTANCE_PRESENT=0
  fi
}
trap delete_instance EXIT INT TERM

for _ in {1..260}; do
  if gcloud storage ls "$RESULT_URI" >/dev/null 2>&1; then
    gcloud storage cp "$RESULT_URI" "$LOCAL_RESULT_DIR/"
    gcloud storage cp "$RESULT_SHA_URI" "$LOCAL_RESULT_DIR/"
    (
      cd "$LOCAL_RESULT_DIR"
      shasum -a 256 -c "$RUN_ID-results.tar.gz.sha256"
    )
    delete_instance
    trap - EXIT INT TERM
    echo "verified result archive at $LOCAL_RESULT_DIR"
    exit 0
  fi

  STATUS="$(gcloud compute instances describe "$RUN_ID" \
    --project="$PAPER_C_GCP_PROJECT" --zone="$ZONE" \
    --format='value(status)' 2>/dev/null || true)"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $RUN_ID status=${STATUS:-not-found}"
  if [[ -z "$STATUS" ]]; then
    INSTANCE_PRESENT=0
    echo "instance disappeared before a result archive was found" >&2
    exit 1
  fi
  if [[ "$STATUS" == "TERMINATED" ]]; then
    sleep 15
    continue
  fi
  sleep 30
done

echo "monitor timed out before a result archive was found" >&2
exit 1
