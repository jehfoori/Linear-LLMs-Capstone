#!/usr/bin/env bash
set -euo pipefail

BASED_PY=${BASED_PY:-/workspace/.venv-based/bin/python}
SOURCE_RESULT_ROOT=${SOURCE_RESULT_ROOT:-results/hazy_context_sweep_n20_l512_2560}
RESULT_ROOT=${RESULT_ROOT:-results/hazy_context_sweep_n20_l512_2560_fp32_based360}
MODEL_CONFIG=${MODEL_CONFIG:-configs/model_based_360m_hazy_fp32.yaml}

LENGTHS=(512 768 1024 1280 1536 1792 2048 2560)

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/chunks" "$RESULT_ROOT/comparisons_patched"
LOG_PATH="$RESULT_ROOT/logs/run.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "Started Based 360M fp32 chunked context rerun at $(date -Is)"
echo "Result root: $RESULT_ROOT"
echo "Source result root: $SOURCE_RESULT_ROOT"
git rev-parse --short HEAD || true
git status --short || true

run_length_chunk() {
  local task=$1
  local dataset=$2
  local id_prefix=$3
  local length=$4
  local out_dir="$RESULT_ROOT/chunks/$task/l$length"
  local ids=()

  for index in $(seq 0 19); do
    ids+=("$(printf "%s_%s_%04d" "$id_prefix" "$length" "$index")")
  done

  echo
  echo "Evaluating $task length=$length in fresh process"
  "$BASED_PY" -m niah.cli.evaluate \
    --dataset "$dataset" \
    --model-config "$MODEL_CONFIG" \
    --out "$out_dir" \
    --example-ids "${ids[@]}"
  "$BASED_PY" -m niah.cli.analyze_results --run-dir "$out_dir"
}

aggregate_task() {
  local task=$1
  local dataset=$2
  local out_dir="$RESULT_ROOT/$task/based_360m_fp32"

  echo
  echo "Aggregating chunked runs for $task"
  "$BASED_PY" - "$RESULT_ROOT/chunks/$task" "$out_dir" "$dataset" "$MODEL_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from niah.analyze import dataset_manifest
from niah.data import load_config, write_json

chunk_root = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
dataset_path = sys.argv[3]
model_config_path = sys.argv[4]
out_dir.mkdir(parents=True, exist_ok=True)

prediction_rows = []
for path in sorted(chunk_root.glob("l*/predictions.jsonl")):
    with path.open("r", encoding="utf-8") as handle:
        prediction_rows.extend(json.loads(line) for line in handle if line.strip())

prediction_rows.sort(key=lambda row: (int(row["target_length"]), str(row["example_id"])))
with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
    for row in prediction_rows:
        handle.write(json.dumps(row, sort_keys=True) + "\n")

first_env = next(chunk_root.glob("l*/environment.json"))
first_load = next(chunk_root.glob("l*/model_load_report.json"))
(out_dir / "environment.json").write_text(first_env.read_text(encoding="utf-8"), encoding="utf-8")
(out_dir / "model_load_report.json").write_text(first_load.read_text(encoding="utf-8"), encoding="utf-8")
write_json(
    out_dir / "manifest.json",
    {
        **dataset_manifest(dataset_path),
        "model_config_path": model_config_path,
        "model_config": load_config(model_config_path),
        "evaluation_subset": {
            "requested_example_ids": "chunked_by_length",
            "limit": None,
            "num_selected_examples": len(prediction_rows),
            "num_total_examples": len(prediction_rows),
        },
        "generation": {
            "max_new_tokens": load_config(model_config_path).get("max_new_tokens", 8),
            "do_sample": False,
            "use_cache": load_config(model_config_path).get("use_cache", True),
            "decode_strategy": load_config(model_config_path).get("decode_strategy"),
        },
        "scoring": {"method": "first_number_exact_match"},
        "chunked_fresh_process_per_length": True,
    },
)
print(f"Wrote {len(prediction_rows)} aggregate predictions to {out_dir}")
PY
  "$BASED_PY" -m niah.cli.analyze_results --run-dir "$out_dir"
}

compare_patched() {
  local task=$1
  "$BASED_PY" -m niah.cli.compare_runs \
    --run-dirs \
    "$RESULT_ROOT/$task/based_360m_fp32" \
    "$SOURCE_RESULT_ROOT/$task/mamba_360m" \
    "$SOURCE_RESULT_ROOT/$task/attn_360m" \
    "$SOURCE_RESULT_ROOT/$task/based_1b" \
    "$SOURCE_RESULT_ROOT/$task/mamba_1b" \
    "$SOURCE_RESULT_ROOT/$task/attn_1b" \
    --out "$RESULT_ROOT/comparisons_patched/$task"
}

for length in "${LENGTHS[@]}"; do
  run_length_chunk single_needle datasets/context_sweep_single_n20_l512_2560_gpt2tok.jsonl passkey_single "$length"
done
aggregate_task single_needle datasets/context_sweep_single_n20_l512_2560_gpt2tok.jsonl
compare_patched single_needle

for length in "${LENGTHS[@]}"; do
  run_length_chunk multi_key datasets/context_sweep_multikey_d20_n20_l512_2560_gpt2tok.jsonl passkey_distractors "$length"
done
aggregate_task multi_key datasets/context_sweep_multikey_d20_n20_l512_2560_gpt2tok.jsonl
compare_patched multi_key

echo
echo "Finished Based 360M fp32 chunked context rerun at $(date -Is)"
