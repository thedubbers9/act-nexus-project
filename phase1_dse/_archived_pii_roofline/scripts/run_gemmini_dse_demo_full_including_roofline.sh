#!/usr/bin/env bash
# =============================================================================
# Phase-1 DSE demo (optional analytical pass — NOT used for Gemmini↔PrimeTime calibration).
# Roofline-style forward bound on .pii + static pJ energy + optional ISA cost plots.
# Package location: phase1_dse/dse/ (see phase1_dse/README.md).
# =============================================================================
# Run from ACT repo root:
#   bash phase1_dse/dse/scripts/run_gemmini_dse_demo.sh
# Env:
#   PII      — .pii file or directory (default: phase1_dse/dse/examples/qkv_like.pii)
#   OUT      — output directory (default: phase1_dse/dse/output/gemmini_dse_demo)
#   NO_PLOT  — set to 1 to skip matplotlib figures
#   ISA_COST_LOG_DIR — if set (e.g. log/gemmini_attention64), run plot_isa_workload_costs.py
#   ISA_BACKEND_DIR — default targets/GEMMINI_17/backend
#   ISA_COST_OUT    — output dir for those plots (default: demo_output/gemmini_isa_cost)

set -euo pipefail

# phase1_dse/dse/scripts -> …/phase1_dse/dse -> …/phase1_dse -> ACT repo root
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=".:${ROOT}/phase1_dse"

PII="${PII:-phase1_dse/dse/examples/qkv_like.pii}"
OUT="${OUT:-phase1_dse/dse/output/gemmini_dse_demo}"
mkdir -p "$OUT"

PLOT_FLAG=()
if [[ "${NO_PLOT:-0}" != "1" ]]; then
  PLOT_FLAG=(--plot)
fi

echo "== Forward bound (latency vs bandwidth, demo outputs) =="
python3 -m dse.forward_bound \
  --input "$PII" \
  --hw_config phase1_dse/dse/config/config.yaml \
  --out "$OUT" \
  "${PLOT_FLAG[@]}"

echo "== Static energy (primitive_hw_config + ISA→class rules) =="
python3 -m dse.energy_workload \
  --input "$PII" \
  --hw_config phase1_dse/dse/config/primitive_hw_config.json \
  --out "$OUT" \
  "${PLOT_FLAG[@]}"

if [[ "${NO_PLOT:-0}" != "1" ]]; then
  echo "== Extra plots (same inputs as plot_from_log demo) =="
  python3 phase1_dse/dse/src/plots_.py --in_dir "$OUT" --out_dir "$OUT/plots_meeting"
fi

if [[ -n "${ISA_COST_LOG_DIR:-}" ]]; then
  ISA_BACKEND_DIR="${ISA_BACKEND_DIR:-targets/GEMMINI_17/backend}"
  ISA_COST_OUT="${ISA_COST_OUT:-demo_output/gemmini_isa_cost}"
  if [[ ! -d "$ISA_BACKEND_DIR/python/cost" ]]; then
    echo "Warning: skip ISA cost plots — missing $ISA_BACKEND_DIR/python/cost (run isa_examples/GEMMINI_17.py first)" >&2
  elif compgen -G "$ISA_COST_LOG_DIR/*.py" > /dev/null; then
    echo "== ISA op-energy plots (plot_isa_workload_costs.py, attn_tile64_demo style) =="
    mkdir -p "$ISA_COST_OUT"
    python3 plot_isa_workload_costs.py \
      --backend-dir "$ISA_BACKEND_DIR" \
      --candidate-dir "$ISA_COST_LOG_DIR" \
      --out-dir "$ISA_COST_OUT"
  else
    echo "Warning: ISA_COST_LOG_DIR=$ISA_COST_LOG_DIR has no *.py candidates; skip ISA cost plots" >&2
  fi
fi

echo "Done. Key outputs under $OUT : summary.json, frontier.csv, candidate_energy.csv, plots/"
