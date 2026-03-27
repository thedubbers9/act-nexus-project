#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

BACKEND_BIN="${BACKEND_BIN:-$ROOT_DIR/backends/ATTN_TILE64}"
BACKEND_DIR="${BACKEND_DIR:-$ROOT_DIR/targets/ATTN_TILE64/backend}"
PT2HLO_OUT="${PT2HLO_OUT:-$ROOT_DIR/pt2hlo/out_attention_core64}"
COMPILED_DIR="${COMPILED_DIR:-$ROOT_DIR/asm/attn_tile64_demo}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/log/attn_tile64_demo}"
PLOT_OUT="${PLOT_OUT:-$ROOT_DIR/demo_output/attn_tile64_demo}"
HW_RESOURCE_CONFIG="${HW_RESOURCE_CONFIG:-$ROOT_DIR/dse/config/primitive_hw_config.json}"
ISA_COST_REFRESH="${ISA_COST_REFRESH:-1}"
ORTOOLS_LIB_DIR="${ORTOOLS_LIB_DIR:-/opt/ortools/lib}"

mkdir -p "$COMPILED_DIR" "$LOG_ROOT" "$PLOT_OUT"

if [[ -d "$ORTOOLS_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$ORTOOLS_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ "$ISA_COST_REFRESH" == "1" ]]; then
  HW_RESOURCE_CONFIG="$HW_RESOURCE_CONFIG" bash "$ROOT_DIR/run_attn_tile64_primitives.sh"
fi

mapfile -t HLOS < <(find "$PT2HLO_OUT" -maxdepth 1 -name 'workload_*.hlo' | sort)
if [[ "${#HLOS[@]}" -eq 0 ]]; then
  echo "Error: no HLO workloads found in $PT2HLO_OUT" >&2
  echo "Run run_pt2hlo_attn_tile64_demo.sh first." >&2
  exit 1
fi

if [[ ! -x "$BACKEND_BIN" ]]; then
  echo "Error: backend binary not found or not executable: $BACKEND_BIN" >&2
  echo "Generate the backend first via run_attn_tile64_primitives.sh." >&2
  exit 1
fi

echo "Compiling HLO workloads with ATTN_TILE64 backend"
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

first_candidate_dir=""
for log_dir in "$LOG_ROOT"/*; do
  if [[ -d "$log_dir" ]]; then
    first_candidate_dir="$log_dir"
    break
  fi
done

if [[ -n "$first_candidate_dir" ]]; then
  echo "Plotting candidate ISA cost summaries from $first_candidate_dir"
  "$PYTHON_BIN" "$ROOT_DIR/plot_isa_workload_costs.py" \
    --backend-dir "$BACKEND_DIR" \
    --candidate-dir "$first_candidate_dir" \
    --out-dir "$PLOT_OUT"
fi

echo "Compiled kernels: $COMPILED_DIR"
echo "Plots + summary: $PLOT_OUT"
