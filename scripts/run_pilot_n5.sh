#!/usr/bin/env bash
set -euo pipefail

MAMBA_PY=${MAMBA_PY:-/workspace/.venv-mamba-official23/bin/python}
GDN_PY=${GDN_PY:-/workspace/.venv-gdn/bin/python}
RESULT_ROOT=${RESULT_ROOT:-results/pilot_n5}

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/comparisons"
LOG_PATH="$RESULT_ROOT/logs/run.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "Started pilot run at $(date -Is)"
echo "Result root: $RESULT_ROOT"
git rev-parse --short HEAD || true
git status --short || true

generate_dataset() {
  local config_path=$1
  local out_path=$2
  echo
  echo "Generating $out_path from $config_path"
  "$MAMBA_PY" -m niah.cli.generate_dataset \
    --config "$config_path" \
    --out "$out_path"
}

run_eval() {
  local py=$1
  local dataset_path=$2
  local model_config=$3
  local out_dir=$4
  echo
  echo "Evaluating $model_config on $dataset_path"
  "$py" -m niah.cli.evaluate \
    --dataset "$dataset_path" \
    --model-config "$model_config" \
    --out "$out_dir"
  "$MAMBA_PY" -m niah.cli.analyze_results --run-dir "$out_dir"
}

compare_setting() {
  local setting=$1
  "$MAMBA_PY" -m niah.cli.compare_runs \
    --run-dirs \
    "$RESULT_ROOT/$setting/mamba2_370m" \
    "$RESULT_ROOT/$setting/mamba2_1_3b" \
    "$RESULT_ROOT/$setting/gdn_340m_map" \
    "$RESULT_ROOT/$setting/gdn_1_3b_map" \
    --out "$RESULT_ROOT/comparisons/$setting"
}

generate_dataset configs/dataset_pilot_single_n5.yaml datasets/pilot_single_n5.jsonl
generate_dataset configs/dataset_pilot_d5_n5.yaml datasets/pilot_d5_n5.jsonl
generate_dataset configs/dataset_pilot_d20_n5.yaml datasets/pilot_d20_n5.jsonl

for setting in single d5 d20; do
  case "$setting" in
    single) dataset_path=datasets/pilot_single_n5.jsonl ;;
    d5) dataset_path=datasets/pilot_d5_n5.jsonl ;;
    d20) dataset_path=datasets/pilot_d20_n5.jsonl ;;
    *) echo "Unknown setting: $setting" >&2; exit 1 ;;
  esac

  run_eval "$MAMBA_PY" "$dataset_path" configs/model_mamba2_official.yaml "$RESULT_ROOT/$setting/mamba2_370m"
  run_eval "$MAMBA_PY" "$dataset_path" configs/model_mamba2_1_3b_official.yaml "$RESULT_ROOT/$setting/mamba2_1_3b"
  run_eval "$GDN_PY" "$dataset_path" configs/model_gated_deltanet_340m_map_pure.yaml "$RESULT_ROOT/$setting/gdn_340m_map"
  run_eval "$GDN_PY" "$dataset_path" configs/model_gated_deltanet_1_3b_map_pure.yaml "$RESULT_ROOT/$setting/gdn_1_3b_map"

  echo
  echo "Comparing setting: $setting"
  compare_setting "$setting"
done

echo
echo "Finished pilot run at $(date -Is)"
