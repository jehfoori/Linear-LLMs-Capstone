#!/usr/bin/env bash
set -euo pipefail

MAMBA_PY=${MAMBA_PY:-/workspace/.venv-mamba-official23/bin/python}
RESULT_ROOT=${RESULT_ROOT:-results/state_spaces_2p7b_probe}
TOKENIZER_ID=${TOKENIZER_ID:-EleutherAI/gpt-neox-20b}

SINGLE_DATASET=datasets/state_spaces_probe_single_n10_l2048_8192_neox_tok.jsonl
MULTIKEY_DATASET=datasets/state_spaces_probe_multikey_d20_n10_l2048_8192_neox_tok.jsonl

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/comparisons"
LOG_PATH="$RESULT_ROOT/logs/run.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "Started State Spaces 2.7B probe at $(date -Is)"
echo "Result root: $RESULT_ROOT"
echo "Tokenizer: $TOKENIZER_ID"
git rev-parse --short HEAD || true
git status --short || true

generate_dataset() {
  local config_path=$1
  local out_path=$2
  echo
  echo "Generating $out_path from $config_path"
  "$MAMBA_PY" -m niah.cli.generate_dataset \
    --config "$config_path" \
    --tokenizer-id "$TOKENIZER_ID" \
    --out "$out_path"
}

run_eval() {
  local dataset_path=$1
  local model_config=$2
  local out_dir=$3
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
    --out "$out_dir"
  "$MAMBA_PY" -m niah.cli.analyze_results --run-dir "$out_dir"
}

compare_task() {
  local task=$1
  rm -rf "$RESULT_ROOT/comparisons/$task"
  "$MAMBA_PY" -m niah.cli.compare_runs \
    --run-dirs \
    "$RESULT_ROOT/$task/transformerpp_2_7b" \
    "$RESULT_ROOT/$task/mamba2_2_7b" \
    "$RESULT_ROOT/$task/mamba2attn_2_7b" \
    --out "$RESULT_ROOT/comparisons/$task"
}

generate_dataset configs/dataset_state_spaces_probe_single_n10_l2048_8192.yaml "$SINGLE_DATASET"
generate_dataset configs/dataset_state_spaces_probe_multikey_d20_n10_l2048_8192.yaml "$MULTIKEY_DATASET"

for task in single_needle multi_key; do
  case "$task" in
    single_needle) dataset_path=$SINGLE_DATASET ;;
    multi_key) dataset_path=$MULTIKEY_DATASET ;;
    *) echo "Unknown task: $task" >&2; exit 1 ;;
  esac

  run_eval "$dataset_path" configs/model_state_spaces_transformerpp_2_7b.yaml "$RESULT_ROOT/$task/transformerpp_2_7b"
  run_eval "$dataset_path" configs/model_state_spaces_mamba2_2_7b.yaml "$RESULT_ROOT/$task/mamba2_2_7b"
  run_eval "$dataset_path" configs/model_state_spaces_mamba2attn_2_7b.yaml "$RESULT_ROOT/$task/mamba2attn_2_7b"

  echo
  echo "Comparing task: $task"
  compare_task "$task"
done

echo
echo "Finished State Spaces 2.7B probe at $(date -Is)"
