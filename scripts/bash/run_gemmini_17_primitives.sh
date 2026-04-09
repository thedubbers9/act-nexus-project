#!/usr/bin/env bash
# Generate taidl_instruction_costs.json for GEMMINI_17 (enables per-op ISA energy in
# plot_isa_workload_costs.py — same pipeline as run_attn_tile64_primitives.sh).
#
# Run from anywhere; resolves ACT repo root as parent of scripts/.
#
#   bash scripts/bash/run_gemmini_17_primitives.sh
#
# Env:
#   SKIP_REGEN=1  — do not re-run isa_examples/GEMMINI_17.py
#   HW_RESOURCE_CONFIG — default dse/config/primitive_hw_config.json

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

ISA_FILE="${ISA_FILE:-$ROOT_DIR/isa_examples/GEMMINI_17.py}"
BACKEND_DIR="${BACKEND_DIR:-$ROOT_DIR/targets/GEMMINI_17/backend}"
PRIMITIVE_JSON="${PRIMITIVE_JSON:-$BACKEND_DIR/taidl_primitive_nodes.json}"
DETAIL_CSV="${DETAIL_CSV:-$BACKEND_DIR/taidl_primitive_nodes.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-$BACKEND_DIR/taidl_primitive_summary.csv}"
HW_RESOURCE_CONFIG="${HW_RESOURCE_CONFIG:-$ROOT_DIR/dse/config/primitive_hw_config.json}"
ESTIMATE_DETAIL_CSV="${ESTIMATE_DETAIL_CSV:-$BACKEND_DIR/taidl_primitive_estimates.csv}"
ESTIMATE_SUMMARY_CSV="${ESTIMATE_SUMMARY_CSV:-$BACKEND_DIR/taidl_instruction_estimates.csv}"
ESTIMATE_SUMMARY_JSON="${ESTIMATE_SUMMARY_JSON:-$BACKEND_DIR/taidl_instruction_costs.json}"
SKIP_REGEN="${SKIP_REGEN:-0}"

if [[ "$SKIP_REGEN" != "1" ]]; then
  if ! "$PYTHON_BIN" -c "import antlr4" >/dev/null 2>&1; then
    echo "Error: Python '$PYTHON_BIN' needs the 'antlr4' module (ACT environment)." >&2
    exit 1
  fi
  echo "[1/2] Regenerating GEMMINI_17 backend from $ISA_FILE"
  (cd "$ROOT_DIR" && export PYTHONPATH="$ROOT_DIR" && "$PYTHON_BIN" "$ISA_FILE")
else
  echo "[1/2] Skipping backend regeneration (SKIP_REGEN=1)"
fi

if [[ ! -f "$PRIMITIVE_JSON" ]]; then
  echo "Error: missing $PRIMITIVE_JSON — run without SKIP_REGEN=1 first." >&2
  exit 1
fi
if [[ ! -f "$HW_RESOURCE_CONFIG" ]]; then
  echo "Error: missing $HW_RESOURCE_CONFIG" >&2
  exit 1
fi

echo "[2/3] Exporting primitive-node CSV files"
"$PYTHON_BIN" "$ROOT_DIR/export_primitive_nodes_csv.py" \
  --input "$PRIMITIVE_JSON" \
  --detail_csv "$DETAIL_CSV" \
  --summary_csv "$SUMMARY_CSV"

echo "[3/3] Estimating per-instruction energy (writes taidl_instruction_costs.json)"
"$PYTHON_BIN" "$ROOT_DIR/estimate_primitive_resources.py" \
  --input "$PRIMITIVE_JSON" \
  --hw_config "$HW_RESOURCE_CONFIG" \
  --detail_csv "$ESTIMATE_DETAIL_CSV" \
  --summary_csv "$ESTIMATE_SUMMARY_CSV" \
  --summary_json "$ESTIMATE_SUMMARY_JSON"

echo
echo "Cost model for plot_isa_workload_costs.py:"
echo "  $ESTIMATE_SUMMARY_JSON"
