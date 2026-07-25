#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${1:?model key required}"
SEED="${2:?seed required}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"

CONFIG=config/smoke.json
TRAIN=inputs/manifests/train.jsonl
PROMPTS="artifacts/smoke/prompts/$MODEL_KEY.train.jsonl"
PARTITION=artifacts/smoke/stage2_partition.jsonl
STAGE1="artifacts/smoke/stage1/$MODEL_KEY/seed_$SEED"
REFERENCE="artifacts/smoke/reference/$MODEL_KEY/seed_$SEED.jsonl"
SELECTION="artifacts/smoke/selections/$MODEL_KEY/seed_$SEED.jsonl"

python3 -m paper_c validate-config --config "$CONFIG"
python3 -m paper_c prepare-prompts --config "$CONFIG" --model-key "$MODEL_KEY" \
  --manifest "$TRAIN" --out "$PROMPTS"
python3 -m paper_c partition --config "$CONFIG" --manifest "$TRAIN" --out "$PARTITION"
python3 -m paper_c train-stage1 --config "$CONFIG" --model-key "$MODEL_KEY" --seed "$SEED" \
  --manifest "$TRAIN" --prompt-cache "$PROMPTS" --out "$STAGE1"

# The exact tokenizer/model revision is now cached. Remove the secret from this
# process and make every later load offline-only.
unset HF_TOKEN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 -m paper_c score --config "$CONFIG" --mode reference --model-key "$MODEL_KEY" \
  --condition stage1 --manifest "$TRAIN" --prompt-cache "$PROMPTS" \
  --adapter "$STAGE1/adapter" --out "$REFERENCE"
python3 -m paper_c select --config "$CONFIG" --partition "$PARTITION" --reference "$REFERENCE" \
  --model-key "$MODEL_KEY" --seed "$SEED" --out "$SELECTION"

for objective in verdict_ce pair_ce dpo; do
  python3 -m paper_c train-stage2 --config "$CONFIG" --model-key "$MODEL_KEY" --seed "$SEED" \
    --objective "$objective" --sampler uncertain --manifest "$TRAIN" --prompt-cache "$PROMPTS" \
    --selection "$SELECTION" --reference "$REFERENCE" --stage1-adapter "$STAGE1/adapter" \
    --out "artifacts/smoke/stage2/$MODEL_KEY/seed_$SEED/uncertain/$objective"
done

python3 -m paper_c smoke-audit --config "$CONFIG" --root \
  "artifacts/smoke/stage2/$MODEL_KEY/seed_$SEED/uncertain"
