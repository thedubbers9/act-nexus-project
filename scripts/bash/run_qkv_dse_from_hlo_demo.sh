#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

BACKEND_BIN="${BACKEND_BIN:-$ROOT_DIR/backends/QKV_DSE}"
BACKEND_DIR="${BACKEND_DIR:-$ROOT_DIR/targets/QKV_DSE/backend}"
PT2HLO_OUT="${PT2HLO_OUT:-$ROOT_DIR/pt2hlo/out_qkv_dse_demo}"
COMPILED_DIR="${COMPILED_DIR:-$ROOT_DIR/asm/qkv_dse_demo}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/log/qkv_dse_demo}"
PLOT_OUT="${PLOT_OUT:-$ROOT_DIR/demo_output/qkv_dse_demo}"
HW_RESOURCE_CONFIG="${HW_RESOURCE_CONFIG:-$ROOT_DIR/phase1_dse/dse/config/primitive_hw_config.json}"
ISA_COST_REFRESH="${ISA_COST_REFRESH:-1}"

mkdir -p "$COMPILED_DIR" "$LOG_ROOT" "$PLOT_OUT"

if [[ "$ISA_COST_REFRESH" == "1" ]]; then
  SKIP_REGEN=1 HW_RESOURCE_CONFIG="$HW_RESOURCE_CONFIG" "$ROOT_DIR/scripts/bash/run_qkv_dse_primitives.sh"
fi

mapfile -t HLOS < <(find "$PT2HLO_OUT" -maxdepth 1 -name 'workload_*.hlo' | sort)
if [[ "${#HLOS[@]}" -eq 0 ]]; then
  echo "Error: no HLO workloads found in $PT2HLO_OUT" >&2
  echo "Run run_pt2hlo_qkv_dse_demo.sh first." >&2
  exit 1
fi

echo "Compiling HLO workloads with QKV_DSE backend"
for hlo_path in "${HLOS[@]}"; do
  stem="$(basename "$hlo_path" .hlo)"
  label="${stem/workload_/demo_}"
  out_py="$COMPILED_DIR/${label}.py"
  log_dir="$LOG_ROOT/${label}"
  mkdir -p "$log_dir"
  "$BACKEND_BIN" --input "$hlo_path" --output "$out_py" --log "$log_dir"
done

echo "Plotting ISA cost summaries"
"$PYTHON_BIN" "$ROOT_DIR/plot_isa_workload_costs.py" \
  --backend-dir "$BACKEND_DIR" \
  --compiled-dir "$COMPILED_DIR" \
  --out-dir "$PLOT_OUT"

echo "Compiled kernels: $COMPILED_DIR"
echo "Plots + summary: $PLOT_OUT"
