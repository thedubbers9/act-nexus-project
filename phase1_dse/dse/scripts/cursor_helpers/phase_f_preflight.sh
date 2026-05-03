#!/usr/bin/env bash
# Phase F wrapper — runs risky_run_preflight.sh with the same env vars.
# See PHASE_F_RUNBOOK.md §2.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/risky_run_preflight.sh" "$@"
