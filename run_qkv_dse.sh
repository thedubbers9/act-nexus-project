#!/usr/bin/env bash
set -euo pipefail

# Optional overrides:
#   BACKEND_BIN=./backends/QKV ./run_qkv_dse.sh
#   INPUT_HLO=my_input.hlo OUTPUT_PY=asm/out.py ./run_qkv_dse.sh
BACKEND_BIN="${BACKEND_BIN:-./backends/QKV_DSE}"
INPUT_HLO="${INPUT_HLO:-workloads/llm_compute_bound_large_workable.hlo}"
OUTPUT_PY="${OUTPUT_PY:-asm/compiled_qkv.py}"
LOG_DIR="${LOG_DIR:-log/qkv_compile}"
DSE_HW_CONFIG="${DSE_HW_CONFIG:-dse/config/config.yaml}"
DSE_TARGETS="${DSE_TARGETS:-dse/config/targets.yaml}"
DSE_OUT="${DSE_OUT:-dse/output}"
PLOT_OUT="${PLOT_OUT:-$DSE_OUT/plot_from_log}"

mkdir -p "$(dirname "$OUTPUT_PY")" "$LOG_DIR" "$DSE_OUT" "$PLOT_OUT"

if [[ ! -x "$BACKEND_BIN" ]]; then
  echo "Error: backend binary not found or not executable: $BACKEND_BIN" >&2
  exit 1
fi
if [[ ! -f "$INPUT_HLO" ]]; then
  echo "Error: input file not found: $INPUT_HLO" >&2
  exit 1
fi
if [[ ! -f "$DSE_HW_CONFIG" ]]; then
  echo "Error: DSE hw config not found: $DSE_HW_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$DSE_TARGETS" ]]; then
  echo "Error: DSE targets file not found: $DSE_TARGETS" >&2
  exit 1
fi

"$BACKEND_BIN" \
  --input "$INPUT_HLO" \
  --output "$OUTPUT_PY" \
  --log "$LOG_DIR" \
  --pre-schedule-dse \
  --dse-hw-config "$DSE_HW_CONFIG" \
  --dse-targets "$DSE_TARGETS" \
  --dse-out "$DSE_OUT"

# Also produce consolidated plots from dumped .pii candidates.
python3 -m dse.src.forward_bound \
  --input "$LOG_DIR" \
  --input_mode pii \
  --hw_config "$DSE_HW_CONFIG" \
  --targets "$DSE_TARGETS" \
  --out "$PLOT_OUT" \
  --plot
