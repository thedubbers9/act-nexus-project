#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bash "$ROOT_DIR/scripts/bash/run_pt2hlo_attn_tile64_demo.sh"
bash "$ROOT_DIR/scripts/bash/run_attn_tile64_from_hlo_demo.sh"
