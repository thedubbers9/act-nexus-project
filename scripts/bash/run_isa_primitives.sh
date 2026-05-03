#!/usr/bin/env bash
# Generic: regenerate backend from TAIDL (optional) + export primitive CSVs + estimate instruction costs.
#
# Usage (from anywhere):
#   bash scripts/bash/run_isa_primitives.sh --isa-name GEMMINI_17
#   bash scripts/bash/run_isa_primitives.sh --isa /path/to/MyISA.py --backend-dir /path/to/targets/MyISA/backend
#
# With --isa-name <NAME> (ACT repo layout):
#   ISA file    -> <act-root>/isa_examples/<NAME>.py
#   Backend dir -> <act-root>/targets/<NAME>/backend
#
# Environment (optional overrides):
#   PYTHON_BIN, SKIP_REGEN=1, HW_RESOURCE_CONFIG, PRIMITIVE_JSON, CSV/JSON output paths
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

ISA_FILE=""
BACKEND_DIR=""
ISA_NAME=""

usage() {
  cat <<EOF >&2
Usage: $(basename "$0") (--isa-name <NAME> | --isa <path/to/ISA.py>) [--backend-dir <path>] [--hw-config <path>]

  --isa-name <NAME>     Use isa_examples/<NAME>.py and targets/<NAME>/backend.
  --isa <path>         Explicit TAIDL ISA Python file.
  --backend-dir <path> Override backend directory (default: from --isa-name or basename of --isa).
  --hw-config <path>    Override hardware JSON (default: phase1_dse/dse/config/primitive_hw_config.json).

Examples:
  $(basename "$0") --isa-name GEMMINI_17
  $(basename "$0") --isa "\$PWD/isa_examples/ATTN_TILE64.py"

Env: SKIP_REGEN=1 skips running the ISA .py; PYTHON_BIN selects interpreter.
EOF
  exit 1
}

HW_RESOURCE_CONFIG="${HW_RESOURCE_CONFIG:-${ACT_ROOT}/phase1_dse/dse/config/primitive_hw_config.json}"
SKIP_REGEN="${SKIP_REGEN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --isa-name)
      [[ $# -ge 2 ]] || usage
      ISA_NAME="$2"
      shift 2
      ;;
    --isa)
      [[ $# -ge 2 ]] || usage
      ISA_FILE="$2"
      shift 2
      ;;
    --backend-dir)
      [[ $# -ge 2 ]] || usage
      BACKEND_DIR="$2"
      shift 2
      ;;
    --hw-config)
      [[ $# -ge 2 ]] || usage
      HW_RESOURCE_CONFIG="$2"
      shift 2
      ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

if [[ -n "$ISA_NAME" && -n "$ISA_FILE" ]]; then
  echo "Error: use only one of --isa-name or --isa" >&2
  exit 1
fi
if [[ -z "$ISA_NAME" && -z "$ISA_FILE" ]]; then
  usage
fi

if [[ -n "$ISA_NAME" ]]; then
  ISA_FILE="${ACT_ROOT}/isa_examples/${ISA_NAME}.py"
fi

ISA_FILE="$(cd "$(dirname "$ISA_FILE")" && pwd)/$(basename "$ISA_FILE")"

if [[ ! -f "$ISA_FILE" ]]; then
  echo "Error: ISA file not found: $ISA_FILE" >&2
  exit 1
fi

if [[ -z "$BACKEND_DIR" ]]; then
  if [[ -n "$ISA_NAME" ]]; then
    BACKEND_DIR="${ACT_ROOT}/targets/${ISA_NAME}/backend"
  else
    base="$(basename "$ISA_FILE" .py)"
    BACKEND_DIR="${ACT_ROOT}/targets/${base}/backend"
  fi
fi

PRIMITIVE_JSON="${PRIMITIVE_JSON:-$BACKEND_DIR/taidl_primitive_nodes.json}"
DETAIL_CSV="${DETAIL_CSV:-$BACKEND_DIR/taidl_primitive_nodes.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-$BACKEND_DIR/taidl_primitive_summary.csv}"
ESTIMATE_DETAIL_CSV="${ESTIMATE_DETAIL_CSV:-$BACKEND_DIR/taidl_primitive_estimates.csv}"
ESTIMATE_SUMMARY_CSV="${ESTIMATE_SUMMARY_CSV:-$BACKEND_DIR/taidl_instruction_estimates.csv}"
ESTIMATE_SUMMARY_JSON="${ESTIMATE_SUMMARY_JSON:-$BACKEND_DIR/taidl_instruction_costs.json}"

if [[ "$SKIP_REGEN" != "1" ]]; then
  if ! "$PYTHON_BIN" -c "import antlr4" >/dev/null 2>&1; then
    echo "Error: Python '$PYTHON_BIN' needs the 'antlr4' module (run inside ACT conda env)." >&2
    exit 1
  fi
  echo "[1/3] Regenerating backend from $ISA_FILE"
  (cd "$ACT_ROOT" && export PYTHONPATH="${ACT_ROOT}${PYTHONPATH:+:$PYTHONPATH}" && "$PYTHON_BIN" "$ISA_FILE")
else
  echo "[1/3] Skipping ISA regeneration (SKIP_REGEN=1)"
fi

if [[ ! -f "$PRIMITIVE_JSON" ]]; then
  echo "Error: missing $PRIMITIVE_JSON — run without SKIP_REGEN=1 first." >&2
  exit 1
fi
if [[ ! -f "$HW_RESOURCE_CONFIG" ]]; then
  echo "Error: missing HW config: $HW_RESOURCE_CONFIG" >&2
  exit 1
fi

echo "[2/3] Exporting primitive-node CSV files"
"$PYTHON_BIN" "$ACT_ROOT/export_primitive_nodes_csv.py" \
  --input "$PRIMITIVE_JSON" \
  --detail_csv "$DETAIL_CSV" \
  --summary_csv "$SUMMARY_CSV"

echo "[3/3] Estimating per-instruction energy (writes taidl_instruction_costs.json)"
"$PYTHON_BIN" "$ACT_ROOT/estimate_primitive_resources.py" \
  --input "$PRIMITIVE_JSON" \
  --hw_config "$HW_RESOURCE_CONFIG" \
  --detail_csv "$ESTIMATE_DETAIL_CSV" \
  --summary_csv "$ESTIMATE_SUMMARY_CSV" \
  --summary_json "$ESTIMATE_SUMMARY_JSON"

echo
echo "ACT_ROOT=$ACT_ROOT"
echo "Cost JSON for plot_isa_workload_costs.py: $ESTIMATE_SUMMARY_JSON"
