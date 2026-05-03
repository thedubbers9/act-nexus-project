#!/usr/bin/env bash
# Create a new Phase F run bundle from _template under MLIR-hardware-analysis/docs/calibration/run_bundles/.
# Usage: init_phase_f_run_bundle.sh YYYYMMDD workload_slug dut_scope_slug
# Example: init_phase_f_run_bundle.sh 20260419 matmul_64x64 chipyard_full_active_matmul_20us
#
# Run from anywhere (paths are derived from this script's location).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cursor_helpers -> scripts -> dse -> phase1_dse -> act -> submodule -> MLIR-hardware-analysis (6 parents)
MLIR_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../" && pwd)"
TEMPLATE="${MLIR_ROOT}/docs/calibration/run_bundles/_template"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 YYYYMMDD workload_slug dut_scope_slug" >&2
  exit 1
fi

DATE="$1"
WORKLOAD="$2"
DUT="$3"
NAME="${DATE}_${WORKLOAD}_${DUT}"
DEST="${MLIR_ROOT}/docs/calibration/run_bundles/${NAME}"

if [[ ! -d "${TEMPLATE}" ]]; then
  echo "Missing template: ${TEMPLATE}" >&2
  exit 1
fi

if [[ -e "${DEST}" ]]; then
  echo "Refusing to overwrite: ${DEST}" >&2
  exit 1
fi

mkdir -p "${DEST}"
cp -a "${TEMPLATE}/." "${DEST}/"
sed -i "s/\"dut_scope\": null/\"dut_scope\": \"${DUT}\"/" "${DEST}/run_manifest.json" 2>/dev/null || \
  perl -pi -e "s/\"dut_scope\": null/\"dut_scope\": \"${DUT}\"/" "${DEST}/run_manifest.json"
sed -i "s/\"workload_label\": null/\"workload_label\": \"${WORKLOAD}\"/" "${DEST}/run_manifest.json" 2>/dev/null || \
  perl -pi -e "s/\"workload_label\": null/\"workload_label\": \"${WORKLOAD}\"/" "${DEST}/run_manifest.json"

echo "Created ${DEST}"
echo "Edit run_manifest.json (paths, git_sha) before EDA."
