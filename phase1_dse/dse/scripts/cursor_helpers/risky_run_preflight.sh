#!/usr/bin/env bash
# Preflight for risky mflowgen/VCS steps. Exit non-zero = do not launch.
# Usage:
#   RUN_GUARD_STATUS=/path/to/run_guard_status.json \
#   DESIGN_ROOT=/path/to/gemmini_chipyard_locked_bf16 \
#   BUILD_VCS_DIR=/path/to/build/.../20-synopsys-vcs-sim \
#   ./risky_run_preflight.sh
#
# All paths optional; checks that are skipped print [skip].

set -euo pipefail

MIN_FREE_KB=${MIN_FREE_KB:-31457280}
MONITOR_MAX_AGE_SEC=${MONITOR_MAX_AGE_SEC:-60}

warn() { echo "[preflight] $*" >&2; }
die() { echo "[preflight] FAIL: $*" >&2; exit 1; }

# --- Free space on /scratch/krish ---
if [[ -d /scratch/krish ]]; then
  avail=$(df -Pk /scratch/krish 2>/dev/null | awk 'NR==2 {print $4}')
  if [[ -n "${avail}" ]] && [[ "${avail}" -lt "${MIN_FREE_KB}" ]]; then
    die "free space on /scratch/krish (${avail} KB) < ${MIN_FREE_KB} KB"
  fi
  warn "free space OK on /scratch/krish (${avail:-unknown} KB avail)"
else
  warn "[skip] /scratch/krish not found"
fi

# --- Monitor heartbeat (optional) ---
if [[ -n "${RUN_GUARD_STATUS:-}" ]] && [[ -f "${RUN_GUARD_STATUS}" ]]; then
  now=$(date +%s)
  mtime=$(stat -c %Y "${RUN_GUARD_STATUS}" 2>/dev/null || stat -f %m "${RUN_GUARD_STATUS}" 2>/dev/null || echo 0)
  age=$((now - mtime))
  if [[ "${age}" -gt "${MONITOR_MAX_AGE_SEC}" ]]; then
    die "run_guard_status.json too old (${age}s > ${MONITOR_MAX_AGE_SEC}s): ${RUN_GUARD_STATUS}"
  fi
  warn "monitor heartbeat OK (${age}s): ${RUN_GUARD_STATUS}"
else
  warn "[skip] RUN_GUARD_STATUS not set or missing — monitor heartbeat not verified"
fi

# --- Scoped dump-pattern grep ---
if command -v rg >/dev/null 2>&1; then
  dump_scan() {
    local root="$1"
    rg -l --no-messages -S -e '\$dumpfile' -e '\$dumpvars' -e 'dump_vcd' -e 'fsdb' "$root" 2>/dev/null | head -n 1 | grep -q .
  }
  if [[ -n "${DESIGN_ROOT:-}" ]] && [[ -d "${DESIGN_ROOT}" ]]; then
    if dump_scan "${DESIGN_ROOT}"; then
      die "dump pattern matched under DESIGN_ROOT=${DESIGN_ROOT} — review TB/TCL"
    fi
    warn "no dump pattern hits in DESIGN_ROOT=${DESIGN_ROOT}"
  else
    warn "[skip] DESIGN_ROOT not set — dump grep skipped"
  fi
  if [[ -n "${BUILD_VCS_DIR:-}" ]] && [[ -d "${BUILD_VCS_DIR}" ]]; then
    if dump_scan "${BUILD_VCS_DIR}"; then
      die "dump pattern matched under BUILD_VCS_DIR=${BUILD_VCS_DIR}"
    fi
    warn "no dump pattern hits in BUILD_VCS_DIR=${BUILD_VCS_DIR}"
  else
    warn "[skip] BUILD_VCS_DIR not set — VCS-step grep skipped"
  fi
else
  warn "[skip] ripgrep (rg) not installed — run manual grep for dump patterns"
fi

warn "preflight OK"
exit 0
