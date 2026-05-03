#!/usr/bin/env bash
# Run PrimeTime post-sanity TCL on a completed synopsys-pt-power step.
# Usage:
#   PT_POWER_STEP=/path/to/build/.../24-synopsys-pt-power \
#     bash run_pt_post_sanity.sh
#
# Requires: pt_shell on PATH, Synopsys env sourced (e.g. source_synopsys_cmu.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TCL="${SCRIPT_DIR}/pt_post_sanity.tcl"

if [[ -z "${PT_POWER_STEP:-}" ]]; then
  echo "Set PT_POWER_STEP to the synopsys-pt-power step directory (contains outputs/primetime.session)." >&2
  exit 1
fi

if [[ ! -f "${PT_POWER_STEP}/outputs/primetime.session" ]]; then
  echo "ERROR: ${PT_POWER_STEP}/outputs/primetime.session not found." >&2
  exit 1
fi

cd "${PT_POWER_STEP}"
echo "Running pt_shell in $(pwd)" >&2
pt_shell -f "${TCL}"
