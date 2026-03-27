#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/opt/miniconda/envs/act/bin/python3" ]]; then
    PYTHON_BIN="/opt/miniconda/envs/act/bin/python3"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

PT2HLO_OUT="${PT2HLO_OUT:-$ROOT_DIR/pt2hlo/out_qkv_dse_demo}"
WORKLOAD_SPECS="${WORKLOAD_SPECS:-64,64:bfloat16|128,64:bfloat16|256,64:bfloat16|512,64:bfloat16}"

if ! "$PYTHON_BIN" -c "import torch_xla" >/dev/null 2>&1; then
  echo "Error: torch_xla is not available in $PYTHON_BIN" >&2
  echo "Rebuild the ACT Docker image so the act environment includes torch/torch_xla, or set PYTHON_BIN explicitly." >&2
  exit 1
fi

mkdir -p "$PT2HLO_OUT"
IFS='|' read -r -a WORKLOADS <<< "$WORKLOAD_SPECS"

PT_ARGS=(
  "$ROOT_DIR/pt2hlo/pt2hlo.py"
  --model-file "$ROOT_DIR/pt2hlo/examples/qkv_dse_demo_block.py"
  --model-entry build_model
  --isa-profile qkv_dse
  --strict-ops
  --output-dir "$PT2HLO_OUT"
)
for workload in "${WORKLOADS[@]}"; do
  PT_ARGS+=(--workload "$workload")
done

echo "Exporting PyTorch workloads to HLO with $PYTHON_BIN"
"$PYTHON_BIN" "${PT_ARGS[@]}"

echo "Wrote HLO workloads to: $PT2HLO_OUT"
