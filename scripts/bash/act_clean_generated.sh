#!/usr/bin/env bash
# Cleans Phase-1 DSE outputs under phase1_dse/dse/ plus asm/log. Run from repo via bash scripts/bash/act_clean_generated.sh
set -euo pipefail

# Dry-run by default. Use --yes to actually delete.
# Optional: --include-old-script also removes run_dse_forward_bound.sh.

DO_DELETE=0
INCLUDE_OLD_SCRIPT=0

for arg in "$@"; do
  case "$arg" in
    --yes) DO_DELETE=1 ;;
    --include-old-script) INCLUDE_OLD_SCRIPT=1 ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: $0 [--yes] [--include-old-script]" >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

paths=(
  "asm"
  "log"
  "phase1_dse/dse/output"
  "phase1_dse/dse/__pycache__"
  "phase1_dse/dse/src/__pycache__"
  "phase1_dse/dse/tests/__pycache__"
)

if [[ "$INCLUDE_OLD_SCRIPT" -eq 1 ]]; then
  paths+=("run_dse_forward_bound.sh")
fi

existing=()
for p in "${paths[@]}"; do
  if [[ -e "$p" ]]; then
    existing+=("$p")
  fi
done

if [[ "${#existing[@]}" -eq 0 ]]; then
  echo "Nothing to clean."
  exit 0
fi

echo "Cleanup targets:"
for p in "${existing[@]}"; do
  echo "  - $p"
done

if [[ "$DO_DELETE" -eq 0 ]]; then
  echo
  echo "Dry-run only. Re-run with --yes to delete."
  exit 0
fi

rm -rf "${existing[@]}"
echo "Cleanup complete."
