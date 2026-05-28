#!/usr/bin/env bash
set -euo pipefail

MAMBA_PY=${MAMBA_PY:-/workspace/.venv-mamba-official23/bin/python}
RESULT_ROOT=${RESULT_ROOT:-results/state_spaces_mamba_family_primary}
TOKENIZER_ID=${TOKENIZER_ID:-EleutherAI/gpt-neox-20b}
RUN_MODE=${RUN_MODE:-all}

DATASET_CONFIG=configs/dataset_state_spaces_mamba_family_primary_n30_l2048_8192.yaml
DATASET_PATH=datasets/state_spaces_mamba_family_primary_n30_l2048_8192_neox_tok.jsonl

MODEL_NAMES=(mamba_2_8b mamba2_2_7b mamba2attn_2_7b)
MODEL_CONFIGS=(
  configs/model_state_spaces_mamba_2_8b.yaml
  configs/model_state_spaces_mamba2_2_7b.yaml
  configs/model_state_spaces_mamba2attn_2_7b.yaml
)

SMOKE_EXAMPLE_IDS=(
  passkey_distractors_d0_2048_0000
  passkey_distractors_d5_2048_0000
  passkey_distractors_d10_2048_0000
)

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/comparisons"
LOG_PATH="$RESULT_ROOT/logs/run.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "Started State Spaces Mamba-family primary run at $(date -Is)"
echo "Run mode: $RUN_MODE"
echo "Result root: $RESULT_ROOT"
echo "Tokenizer: $TOKENIZER_ID"
git rev-parse --short HEAD || true
git status --short || true

generate_dataset() {
  echo
  echo "Generating $DATASET_PATH from $DATASET_CONFIG"
  "$MAMBA_PY" -m niah.cli.generate_dataset \
    --config "$DATASET_CONFIG" \
    --tokenizer-id "$TOKENIZER_ID" \
    --out "$DATASET_PATH"
}

run_eval() {
  local dataset_path=$1
  local model_config=$2
  local out_dir=$3
  shift 3

  if [[ -f "$out_dir/summary.csv" ]]; then
    echo
    echo "Skipping completed run: $out_dir"
    return
  fi

  echo
  echo "Evaluating $model_config on $dataset_path"
  rm -rf "$out_dir"
  "$MAMBA_PY" -m niah.cli.evaluate \
    --dataset "$dataset_path" \
    --model-config "$model_config" \
    --out "$out_dir" \
    "$@"
  "$MAMBA_PY" -m niah.cli.analyze_results --run-dir "$out_dir"
}

run_smoke() {
  echo
  echo "Running smoke checks"
  local idx
  for idx in "${!MODEL_NAMES[@]}"; do
    run_eval \
      "$DATASET_PATH" \
      "${MODEL_CONFIGS[$idx]}" \
      "$RESULT_ROOT/smoke/${MODEL_NAMES[$idx]}" \
      --example-ids "${SMOKE_EXAMPLE_IDS[@]}"
  done
}

run_full() {
  echo
  echo "Running full primary matrix"
  local idx
  for idx in "${!MODEL_NAMES[@]}"; do
    run_eval \
      "$DATASET_PATH" \
      "${MODEL_CONFIGS[$idx]}" \
      "$RESULT_ROOT/full/${MODEL_NAMES[$idx]}"
  done

  echo
  echo "Comparing full matrix"
  rm -rf "$RESULT_ROOT/comparisons/full"
  "$MAMBA_PY" -m niah.cli.compare_runs \
    --run-dirs \
    "$RESULT_ROOT/full/mamba_2_8b" \
    "$RESULT_ROOT/full/mamba2_2_7b" \
    "$RESULT_ROOT/full/mamba2attn_2_7b" \
    --out "$RESULT_ROOT/comparisons/full"
}

generate_dataset

case "$RUN_MODE" in
  smoke)
    run_smoke
    ;;
  full)
    run_full
    ;;
  all)
    run_smoke
    run_full
    ;;
  *)
    echo "Unknown RUN_MODE: $RUN_MODE" >&2
    exit 1
    ;;
esac

echo
echo "Finished State Spaces Mamba-family primary run at $(date -Is)"
