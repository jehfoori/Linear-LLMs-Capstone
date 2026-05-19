#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: scripts/remote_run.sh <dataset.jsonl> <model_config.yaml> <out_dir>"
  exit 2
fi

DATASET="$1"
MODEL_CONFIG="$2"
OUT_DIR="$3"

python -m niah.cli.evaluate \
  --dataset "$DATASET" \
  --model-config "$MODEL_CONFIG" \
  --out "$OUT_DIR"

python -m niah.cli.analyze_results --run-dir "$OUT_DIR"
