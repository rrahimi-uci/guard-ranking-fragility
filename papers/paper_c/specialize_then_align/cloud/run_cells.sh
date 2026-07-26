#!/usr/bin/env bash
# Startup script for a Paper C v2 training VM.
#
# Downloads the workspace bundle, runs the cells named in CELLS, uploads the
# adapters and run records, then deletes the instance.  It self-deletes on *every*
# exit path, including failure, because v1's cost overruns came from VMs that
# outlived a crashed run.
set -uo pipefail

BUCKET="$(curl -sf -H Metadata-Flavor:Google http://metadata/computeMetadata/v1/instance/attributes/bucket)"
PREFIX="$(curl -sf -H Metadata-Flavor:Google http://metadata/computeMetadata/v1/instance/attributes/prefix)"
CELLS="$(curl -sf -H Metadata-Flavor:Google http://metadata/computeMetadata/v1/instance/attributes/cells)"
NAME="$(curl -sf -H Metadata-Flavor:Google http://metadata/computeMetadata/v1/instance/name)"
ZONE="$(curl -sf -H Metadata-Flavor:Google http://metadata/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')"

LOG=/var/log/paper_c.log
exec > >(tee -a "$LOG") 2>&1
echo "=== paper_c v2 cell runner: $NAME zone=$ZONE cells=$CELLS ==="

finish() {
  code=$?
  echo "=== exit code $code; uploading log and deleting instance ==="
  gsutil -q cp "$LOG" "gs://${BUCKET}/${PREFIX}/logs/${NAME}.log" || true
  # Prefer deletion, but the runner's service account may lack compute.instances.delete.
  # Halting is always permitted and stops compute billing, so it is the fallback.
  gcloud -q compute instances delete "$NAME" --zone "$ZONE" || {
    echo "delete denied; halting instead"; shutdown -h now
  }
}
trap finish EXIT

nvidia-smi || echo "WARNING: nvidia-smi unavailable"

cd /opt
gsutil -q cp "gs://${BUCKET}/${PREFIX}/bundle.tar.gz" . || exit 1
mkdir -p work && tar xzf bundle.tar.gz -C work
cd work

PY=python3
$PY -m pip install -q --upgrade pip
# transformers 5.x imports torchaudio transitively (loss_rnnt); the image's build has
# a mismatched ABI and raises on load.  No audio path is used here, so remove it.
$PY -m pip uninstall -q -y torchaudio || true
$PY -m pip install -q "transformers==5.12.1" "peft==0.19.1" "accelerate" || exit 1
$PY - <<'PROBE'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PROBE

# Stage already-completed cells (the references) so specialist cells can continue
# from them.  -n means existing local copies are never overwritten.
mkdir -p artifacts/cells
gsutil -q -m cp -n -r "gs://${BUCKET}/${PREFIX}/cells/*" artifacts/cells/ 2>/dev/null || true
echo "staged $(ls artifacts/cells 2>/dev/null | wc -l) existing cells"

IFS=',' read -ra LIST <<< "$CELLS"
for cell in "${LIST[@]}"; do
  # a "propose:<backbone>:<panel>" entry runs candidate generation + pair building
  if [[ "$cell" == propose:* ]]; then
    IFS=':' read -r _ bb panel <<< "$cell"
    echo "=== propose $bb $panel ==="
    $PY tools/run_propose.py "$bb" "$panel" || { echo "PROPOSE FAILED: $bb $panel"; continue; }
    gsutil -q -m cp -r "artifacts/pairs/${panel}__${bb}" "gs://${BUCKET}/${PREFIX}/pairs/" \
      && echo "UPLOADED pairs ${panel}__${bb}" || echo "UPLOAD FAILED pairs"
    continue
  fi
  echo "=== cell $cell ==="
  $PY tools/run_one_cell.py "$cell" || { echo "CELL FAILED: $cell"; continue; }
  # run_one_cell.py sanitises "::" to "__" for the directory name; upload must match.
  safe="${cell//::/__}"
  gsutil -q -m cp -r "artifacts/cells/${safe}" "gs://${BUCKET}/${PREFIX}/cells/" \
    && echo "UPLOADED ${safe}" || echo "UPLOAD FAILED ${safe}"
done

echo "=== all cells attempted ==="
