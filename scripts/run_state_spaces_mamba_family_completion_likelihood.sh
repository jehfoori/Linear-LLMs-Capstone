#!/usr/bin/env bash
set -euo pipefail

MAMBA_PY=${MAMBA_PY:-/workspace/.venv-mamba-official23/bin/python}
RESULT_ROOT=${RESULT_ROOT:-results/state_spaces_mamba_family_completion_likelihood}
RUN_MODE=${RUN_MODE:-all}

SMOKE_EXAMPLE_IDS=(
  passkey_distractors_d0_2048_0000
  passkey_distractors_d5_2048_0000
  passkey_distractors_d10_2048_0000
)

mkdir -p "$RESULT_ROOT/logs"
LOG_PATH="$RESULT_ROOT/logs/run.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "Started State Spaces Mamba-family completion-likelihood run at $(date -Is)"
echo "Run mode: $RUN_MODE"
echo "Result root: $RESULT_ROOT"
git rev-parse --short HEAD || true
git status --short || true

run_smoke() {
  echo
  echo "Running completion-likelihood smoke checks"
  rm -rf "$RESULT_ROOT/smoke"
  "$MAMBA_PY" scripts/run_state_spaces_mamba_family_likelihood.py \
    --out "$RESULT_ROOT/smoke" \
    --prompt-style passkey_completion \
    --example-ids "${SMOKE_EXAMPLE_IDS[@]}"
}

run_full() {
  echo
  echo "Running full completion-likelihood matrix"
  rm -rf "$RESULT_ROOT/full"
  "$MAMBA_PY" scripts/run_state_spaces_mamba_family_likelihood.py \
    --out "$RESULT_ROOT/full" \
    --prompt-style passkey_completion
}

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
echo "Finished State Spaces Mamba-family completion-likelihood run at $(date -Is)"
