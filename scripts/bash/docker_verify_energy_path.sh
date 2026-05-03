#!/usr/bin/env bash
# Non-interactive smoke test: ACT Docker image + Phase-1 static energy CLI.
# Run from anywhere; mounts the ACT repo root at /workspace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_MOUNT="/workspace"
OUT_DIR="${CONTAINER_MOUNT}/.docker_verify_energy_out"

ARCH="$(uname -m)"
if [ "$ARCH" = "x86_64" ]; then
  IMAGE_NAME="devanshdvj/act:latest-amd64"
elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
  IMAGE_NAME="devanshdvj/act:latest-arm64"
else
  echo "Error: Unsupported architecture: $ARCH"
  exit 1
fi

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Image ${IMAGE_NAME} not found. Build it first:"
  echo "  cd ${ACT_ROOT}/scripts/bash && ./docker_build.sh --rebuild"
  echo "Or pull a published image if your registry has it."
  exit 1
fi

INNER_CMD="set -euo pipefail
source /opt/miniconda/etc/profile.d/conda.sh
conda activate act
export PYTHONPATH=.:phase1_dse
python3 -m dse.energy_workload --help >/dev/null
rm -rf ${OUT_DIR} && mkdir -p ${OUT_DIR}
python3 -m dse.energy_workload \\
  --input phase1_dse/dse/examples/qkv_like.pii \\
  --hw_config phase1_dse/dse/config/primitive_hw_config.json \\
  --out ${OUT_DIR}
test -f ${OUT_DIR}/energy_summary.json
echo OK: dse.energy_workload completed inside container.
"

docker run --rm \
  --entrypoint bash \
  -v "${ACT_ROOT}:${CONTAINER_MOUNT}:rw" \
  -w "${CONTAINER_MOUNT}" \
  "${IMAGE_NAME}" \
  -lc "${INNER_CMD}"
