#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$ROOT_DIR/run_pt2hlo_qkv_dse_demo.sh"
bash "$ROOT_DIR/run_qkv_dse_from_hlo_demo.sh"
