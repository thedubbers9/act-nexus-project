#!/usr/bin/env bash
set -euo pipefail

# Compile multiple ISA-compatible workloads and generate combined bound plots.
# Usage:
#   ./run_multi_workload_dse.sh
# Optional env overrides:
#   BACKEND=./backends/QKV_DSE
#   HW_CFG=dse/config/config.yaml
#   TARGETS_CFG=dse/config/targets.yaml
#   SLA_TARGET=1e-5
#   FEAS_TARGET=1e-5

BACKEND="${BACKEND:-./backends/QKV_DSE}"
HW_CFG="${HW_CFG:-dse/config/config.yaml}"
TARGETS_CFG="${TARGETS_CFG:-dse/config/targets.yaml}"
SLA_TARGET="${SLA_TARGET:-1e-5}"
FEAS_TARGET="${FEAS_TARGET:-1e-5}"

WORKLOADS=(
  "workloads/llm_mixed_attention_large_workable.hlo"
  
)

declare -a DSE_DIRS=()
mkdir -p asm log dse/output

for workload in "${WORKLOADS[@]}"; do
  name="$(basename "${workload}" .hlo)"
  out_py="asm/compiled_${name}.py"
  log_dir="log/${name}_compile"
  dse_out="dse/output/${name}"

  echo "==> Compiling ${workload}"
  "${BACKEND}" \
    --input "${workload}" \
    --output "${out_py}" \
    --log "${log_dir}" \
    --pre-schedule-dse \
    --dse-hw-config "${HW_CFG}" \
    --dse-targets "${TARGETS_CFG}" \
    --dse-out "${dse_out}"

  DSE_DIRS+=("${dse_out}")
done

joined_dirs="$(IFS=,; echo "${DSE_DIRS[*]}")"

echo "==> Generating multi-workload charts"
python3 dse/src/plots_.py \
  --in_dir "${DSE_DIRS[0]}" \
  --out_dir dse/output/plots_multi_workload \
  --workload_dirs "${joined_dirs}" \
  --sla_target_s "${SLA_TARGET}" \
  --feas_target_s "${FEAS_TARGET}"

echo "Done. Charts: dse/output/plots_multi_workload"
