#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p cloud/bundle

ARCHIVE="cloud/bundle/paper-c-smoke.tar.gz"
CHECKSUM="$ARCHIVE.sha256"

if [[ ! -f inputs/INPUT_MANIFEST.json || ! -f inputs/manifests/train.jsonl ]]; then
  echo "isolated inputs are missing; run make bootstrap first" >&2
  exit 2
fi

COPYFILE_DISABLE=1 tar \
  --exclude='./.env' \
  --exclude='./build' \
  --exclude='./artifacts/runs' \
  --exclude='./artifacts/smoke' \
  --exclude='./cloud/bundle' \
  --exclude='*/__pycache__' \
  -czf "$ARCHIVE" .

if tar -tzf "$ARCHIVE" | grep -Eq '(^|/)\.env$'; then
  echo "bundle unexpectedly contains .env" >&2
  exit 2
fi
ARCHIVE_SHA256="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
printf '%s  %s\n' "$ARCHIVE_SHA256" "$(basename "$ARCHIVE")" > "$CHECKSUM"
echo "created $ARCHIVE"
