#!/usr/bin/env bash
# Hardware-mapping + cost-model static energy on .pii (no roofline / no pre-schedule hook).
# Uses: phase1_dse/dse/src/energy_workload.py + primitive_hw_config.json +
#       docs/hardware_mapping_interface_package/final_mapping.json (via energy_estimate).
#
# Run from ACT repo root:
#   bash phase1_dse/dse/scripts/run_gemmini_dse_demo.sh
#
# Roofline / forward_bound / multi-workload compile drivers were moved to:
#   phase1_dse/_archived_pii_roofline/

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=".:${ROOT}/phase1_dse"

PII="${PII:-phase1_dse/dse/examples/qkv_like.pii}"
OUT="${OUT:-phase1_dse/dse/output/gemmini_energy_demo}"
mkdir -p "$OUT"

PLOT_FLAG=()
if [[ "${NO_PLOT:-0}" != "1" ]]; then
  PLOT_FLAG=(--plot)
fi

echo "== Static energy (primitive_hw_config + hardware mapping / fused patterns) =="
python3 -m dse.energy_workload \
  --input "$PII" \
  --hw_config phase1_dse/dse/config/primitive_hw_config.json \
  --out "$OUT" \
  "${PLOT_FLAG[@]}"

if [[ -n "${ISA_COST_LOG_DIR:-}" ]]; then
  ISA_BACKEND_DIR="${ISA_BACKEND_DIR:-targets/GEMMINI_17/backend}"
  ISA_COST_OUT="${ISA_COST_OUT:-demo_output/gemmini_isa_cost}"
  if [[ ! -d "$ISA_BACKEND_DIR/python/cost" ]]; then
    echo "Warning: skip ISA cost plots — missing $ISA_BACKEND_DIR/python/cost (run isa_examples/GEMMINI_17.py first)" >&2
  elif compgen -G "$ISA_COST_LOG_DIR/*.py" > /dev/null; then
    echo "== ISA op-energy plots (lowest-energy style comparison across *.py candidates) =="
    mkdir -p "$ISA_COST_OUT"
    python3 plot_isa_workload_costs.py \
      --backend-dir "$ISA_BACKEND_DIR" \
      --candidate-dir "$ISA_COST_LOG_DIR" \
      --out-dir "$ISA_COST_OUT"
  else
    echo "Warning: ISA_COST_LOG_DIR=$ISA_COST_LOG_DIR has no *.py candidates; skip" >&2
  fi
fi

echo "Done. Outputs under $OUT (energy_detail_*.json, candidate_energy.csv, plots/)."
