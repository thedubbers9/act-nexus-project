#!/usr/bin/env bash
# Create a new Phase F run bundle from _template.
# Usage: init_phase_f_run_bundle.sh YYYYMMDD workload_slug dut_scope_slug
# Example: init_phase_f_run_bundle.sh 20260419 matmul_64x64 chipyard_full_active_matmul_20us
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMODULE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE="${SUBMODULE_ROOT}/.cursor/docs/artifacts/runs/_template"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 YYYYMMDD workload_slug dut_scope_slug" >&2
  exit 1
fi

DATE="$1"
WORKLOAD="$2"
DUT="$3"
NAME="${DATE}_${WORKLOAD}_${DUT}"
DEST="${SUBMODULE_ROOT}/.cursor/docs/artifacts/runs/${NAME}"

if [[ -e "${DEST}" ]]; then
  echo "Refusing to overwrite: ${DEST}" >&2
  exit 1
fi

mkdir -p "${DEST}"
cp -a "${TEMPLATE}/." "${DEST}/"
sed -i "s/\"dut_scope\": null/\"dut_scope\": \"${DUT}\"/" "${DEST}/run_manifest.json" 2>/dev/null || \
  perl -pi -e "s/\"dut_scope\": null/\"dut_scope\": \"${DUT}\"/" "${DEST}/run_manifest.json"
# workload_label — best-effort from slug
sed -i "s/\"workload_label\": null/\"workload_label\": \"${WORKLOAD}\"/" "${DEST}/run_manifest.json" 2>/dev/null || \
  perl -pi -e "s/\"workload_label\": null/\"workload_label\": \"${WORKLOAD}\"/" "${DEST}/run_manifest.json"

echo "Created ${DEST}"
echo "Edit run_manifest.json (paths, git_sha) before EDA."
