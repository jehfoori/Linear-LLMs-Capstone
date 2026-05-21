#!/usr/bin/env bash
set -euo pipefail

BASED_PY=${BASED_PY:-/workspace/.venv-based/bin/python}
RESULT_ROOT=${RESULT_ROOT:-results/hazy_niah_n30_l1024_2048}
TOKENIZER_ID=${TOKENIZER_ID:-gpt2}

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/comparisons"
LOG_PATH="$RESULT_ROOT/logs/run.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "Started HazyResearch NIAH n30 run at $(date -Is)"
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
  echo
  echo "Evaluating $model_config on $dataset_path"
  "$BASED_PY" -m niah.cli.evaluate \
    --dataset "$dataset_path" \
    --model-config "$model_config" \
    --out "$out_dir"
  "$BASED_PY" -m niah.cli.analyze_results --run-dir "$out_dir"
}

compare_setting() {
  local setting=$1
  "$BASED_PY" -m niah.cli.compare_runs \
    --run-dirs \
    "$RESULT_ROOT/$setting/based_360m" \
    "$RESULT_ROOT/$setting/mamba_360m" \
    "$RESULT_ROOT/$setting/attn_360m" \
    "$RESULT_ROOT/$setting/based_1b" \
    "$RESULT_ROOT/$setting/mamba_1b" \
    "$RESULT_ROOT/$setting/attn_1b" \
    --out "$RESULT_ROOT/comparisons/$setting"
}

generate_dataset configs/dataset_hazy_single_n30_l1024_2048.yaml datasets/hazy_single_n30_l1024_2048_gpt2tok.jsonl
generate_dataset configs/dataset_hazy_d5_n30_l1024_2048.yaml datasets/hazy_d5_n30_l1024_2048_gpt2tok.jsonl
generate_dataset configs/dataset_hazy_d20_n30_l1024_2048.yaml datasets/hazy_d20_n30_l1024_2048_gpt2tok.jsonl

for setting in single d5 d20; do
  case "$setting" in
    single) dataset_path=datasets/hazy_single_n30_l1024_2048_gpt2tok.jsonl ;;
    d5) dataset_path=datasets/hazy_d5_n30_l1024_2048_gpt2tok.jsonl ;;
    d20) dataset_path=datasets/hazy_d20_n30_l1024_2048_gpt2tok.jsonl ;;
    *) echo "Unknown setting: $setting" >&2; exit 1 ;;
  esac

  run_eval "$dataset_path" configs/model_based_360m_hazy.yaml "$RESULT_ROOT/$setting/based_360m"
  run_eval "$dataset_path" configs/model_mamba_360m_hazy.yaml "$RESULT_ROOT/$setting/mamba_360m"
  run_eval "$dataset_path" configs/model_attn_360m_hazy.yaml "$RESULT_ROOT/$setting/attn_360m"
  run_eval "$dataset_path" configs/model_based_1b_hazy.yaml "$RESULT_ROOT/$setting/based_1b"
  run_eval "$dataset_path" configs/model_mamba_1b_hazy.yaml "$RESULT_ROOT/$setting/mamba_1b"
  run_eval "$dataset_path" configs/model_attn_1b_hazy.yaml "$RESULT_ROOT/$setting/attn_1b"

  echo
  echo "Comparing setting: $setting"
  compare_setting "$setting"
done

echo
echo "Finished HazyResearch NIAH n30 run at $(date -Is)"
