#!/usr/bin/env bash
set -euo pipefail

MAMBA_PY=${MAMBA_PY:-/workspace/.venv-mamba-official23/bin/python}
RESULT_ROOT=${RESULT_ROOT:-results/state_spaces_query_position_diagnostic}

mkdir -p "$RESULT_ROOT/logs"
LOG_PATH="$RESULT_ROOT/logs/run.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "Started State Spaces query/position diagnostic at $(date -Is)"
echo "Result root: $RESULT_ROOT"
git rev-parse --short HEAD || true
git status --short || true

"$MAMBA_PY" scripts/diagnostics/run_state_spaces_query_position_diagnostic.py --out "$RESULT_ROOT"

echo
echo "Finished State Spaces query/position diagnostic at $(date -Is)"
