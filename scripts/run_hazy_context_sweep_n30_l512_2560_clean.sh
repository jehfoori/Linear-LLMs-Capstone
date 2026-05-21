#!/usr/bin/env bash
set -euo pipefail

BASED_PY=${BASED_PY:-/workspace/.venv-based/bin/python}
RESULT_ROOT=${RESULT_ROOT:-results/hazy_context_sweep_n30_l512_2560_clean}
TOKENIZER_ID=${TOKENIZER_ID:-gpt2}

SINGLE_DATASET=datasets/context_sweep_single_n30_l512_2560_gpt2tok.jsonl
MULTIKEY_DATASET=datasets/context_sweep_multikey_d20_n30_l512_2560_gpt2tok.jsonl

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/comparisons"
LOG_PATH="$RESULT_ROOT/logs/run.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "Started clean HazyResearch context sweep n30 run at $(date -Is)"
echo "Result root: $RESULT_ROOT"
echo "Tokenizer: $TOKENIZER_ID"
git rev-parse --short HEAD || true
git status --short || true

generate_dataset() {
  local config_path=$1
  local out_path=$2
  echo
  echo "Generating $out_path from $config_path"
  "$BASED_PY" -m niah.cli.generate_dataset \
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
  "$BASED_PY" -m niah.cli.evaluate \
    --dataset "$dataset_path" \
    --model-config "$model_config" \
    --out "$out_dir"
  "$BASED_PY" -m niah.cli.analyze_results --run-dir "$out_dir"
}

compare_task() {
  local task=$1
  rm -rf "$RESULT_ROOT/comparisons/$task"
  "$BASED_PY" -m niah.cli.compare_runs \
    --run-dirs \
    "$RESULT_ROOT/$task/attn_360m" \
    "$RESULT_ROOT/$task/mamba_360m" \
    "$RESULT_ROOT/$task/based_360m_fp32" \
    "$RESULT_ROOT/$task/attn_1b" \
    "$RESULT_ROOT/$task/mamba_1b" \
    "$RESULT_ROOT/$task/based_1b_fp32" \
    --out "$RESULT_ROOT/comparisons/$task"
}

generate_dataset configs/dataset_context_sweep_single_n30_l512_2560.yaml "$SINGLE_DATASET"
generate_dataset configs/dataset_context_sweep_multikey_d20_n30_l512_2560.yaml "$MULTIKEY_DATASET"

for task in single_needle multi_key; do
  case "$task" in
    single_needle) dataset_path=$SINGLE_DATASET ;;
    multi_key) dataset_path=$MULTIKEY_DATASET ;;
    *) echo "Unknown task: $task" >&2; exit 1 ;;
  esac

  run_eval "$dataset_path" configs/model_attn_360m_hazy.yaml "$RESULT_ROOT/$task/attn_360m"
  run_eval "$dataset_path" configs/model_mamba_360m_hazy.yaml "$RESULT_ROOT/$task/mamba_360m"
  run_eval "$dataset_path" configs/model_based_360m_hazy_fp32.yaml "$RESULT_ROOT/$task/based_360m_fp32"
  run_eval "$dataset_path" configs/model_attn_1b_hazy.yaml "$RESULT_ROOT/$task/attn_1b"
  run_eval "$dataset_path" configs/model_mamba_1b_hazy.yaml "$RESULT_ROOT/$task/mamba_1b"
  run_eval "$dataset_path" configs/model_based_1b_hazy_fp32.yaml "$RESULT_ROOT/$task/based_1b_fp32"

  echo
  echo "Comparing task: $task"
  compare_task "$task"
done

echo
echo "Finished clean HazyResearch context sweep n30 run at $(date -Is)"
