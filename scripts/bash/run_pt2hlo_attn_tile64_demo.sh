#!/usr/bin/env bash
# Fork note (MLIR-hardware-analysis): default REFERENCE_HLO is workloads/gemmini_anchor_attention_tile64.hlo
# (Gemmini anchor attention). Upstream used attention_tile64_workable.hlo — removed; see ACT_CALIBRATION_FORK_NOTES.md.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/opt/miniconda/envs/act/bin/python3" ]]; then
    PYTHON_BIN="/opt/miniconda/envs/act/bin/python3"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

PT2HLO_OUT="${PT2HLO_OUT:-$ROOT_DIR/pt2hlo/out_attention_core64}"
WORKLOAD_SPECS="${WORKLOAD_SPECS:-64,64:bfloat16;64,64:bfloat16;64,64:bfloat16}"
REFERENCE_HLO="${REFERENCE_HLO:-$ROOT_DIR/workloads/gemmini_anchor_attention_tile64.hlo}"
MODEL_FILE="${MODEL_FILE:-$ROOT_DIR/pt2hlo/examples/attention_core64.py}"
MODEL_ENTRY="${MODEL_ENTRY:-build_model}"

if ! "$PYTHON_BIN" -c "import torch_xla" >/dev/null 2>&1; then
  echo "Error: torch_xla is not available in $PYTHON_BIN" >&2
  echo "Use a Python environment with compatible torch/torch_xla packages installed." >&2
  exit 1
fi

mkdir -p "$PT2HLO_OUT"
IFS='|' read -r -a WORKLOADS <<< "$WORKLOAD_SPECS"

PT_ARGS=(
  "$ROOT_DIR/pt2hlo/pt2hlo.py"
  --model-file "$MODEL_FILE"
  --model-entry "$MODEL_ENTRY"
  --isa-profile attn_tile64
  --allow-ops-from-hlo "$REFERENCE_HLO"
  --strict-ops
  --output-dir "$PT2HLO_OUT"
)
for workload in "${WORKLOADS[@]}"; do
  PT_ARGS+=(--workload "$workload")
done

echo "Exporting PyTorch workloads to HLO with $PYTHON_BIN"
"$PYTHON_BIN" "${PT_ARGS[@]}"

echo "Wrote HLO workloads to: $PT2HLO_OUT"
