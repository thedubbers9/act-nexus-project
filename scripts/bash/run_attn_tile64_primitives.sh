#!/usr/bin/env bash
# Thin wrapper: see run_isa_primitives.sh (--isa-name ATTN_TILE64).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_isa_primitives.sh" --isa-name ATTN_TILE64 "$@"
