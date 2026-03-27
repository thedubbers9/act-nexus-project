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

ISA_FILE="${ISA_FILE:-$ROOT_DIR/ATTN_TILE64.py}"
BACKEND_DIR="${BACKEND_DIR:-$ROOT_DIR/targets/ATTN_TILE64/backend}"
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
    echo "Error: Python runtime '$PYTHON_BIN' does not have the 'antlr4' module." >&2
    echo "Run this script inside the ACT container/environment where ATTN_TILE64.py already works." >&2
    exit 1
  fi
  echo "[1/2] Regenerating ATTN_TILE64 backend artifacts from $ISA_FILE"
  (cd "$ROOT_DIR" && "$PYTHON_BIN" "$ISA_FILE")
else
  echo "[1/2] Skipping backend regeneration because SKIP_REGEN=1"
fi

if [[ ! -f "$PRIMITIVE_JSON" ]]; then
  echo "Error: primitive JSON not found: $PRIMITIVE_JSON" >&2
  echo "Run without SKIP_REGEN=1, or verify ATTN_TILE64 backend generation succeeded." >&2
  exit 1
fi
if [[ ! -f "$HW_RESOURCE_CONFIG" ]]; then
  echo "Error: hardware resource config not found: $HW_RESOURCE_CONFIG" >&2
  exit 1
fi

echo "[2/3] Exporting primitive-node CSV files"
"$PYTHON_BIN" "$ROOT_DIR/export_primitive_nodes_csv.py" \
  --input "$PRIMITIVE_JSON" \
  --detail_csv "$DETAIL_CSV" \
  --summary_csv "$SUMMARY_CSV"

echo "[3/3] Estimating primitive resources from hardware config"
"$PYTHON_BIN" "$ROOT_DIR/estimate_primitive_resources.py" \
  --input "$PRIMITIVE_JSON" \
  --hw_config "$HW_RESOURCE_CONFIG" \
  --detail_csv "$ESTIMATE_DETAIL_CSV" \
  --summary_csv "$ESTIMATE_SUMMARY_CSV" \
  --summary_json "$ESTIMATE_SUMMARY_JSON"

echo
echo "Artifacts:"
echo "  Primitive JSON:   $PRIMITIVE_JSON"
echo "  Detailed CSV:     $DETAIL_CSV"
echo "  Summary CSV:      $SUMMARY_CSV"
echo "  HW config:        $HW_RESOURCE_CONFIG"
echo "  Estimate CSV:     $ESTIMATE_DETAIL_CSV"
echo "  Estimate summary: $ESTIMATE_SUMMARY_CSV"
echo "  Cost JSON:        $ESTIMATE_SUMMARY_JSON"
