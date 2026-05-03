# Cursor helpers (PrimeTime / run bundles / ACT vs PT)

**Canonical location:** `submodule/act/phase1_dse/dse/scripts/cursor_helpers/` inside the ACT submodule (tracked in git).

These are **shell + Python** helpers for PrimeTime bundles, PT hierarchy parsing, preflight checks, and scaling tables. **`eda/`** holds Tcl + `run_pt_post_sanity.sh` for optional PT post-checks.

| File | Role (short) |
|------|----------------|
| `act_pt_term_bucket_v1.py` | PT term split + bucket table + JSON for calibration workflows |
| `parse_pt_hier_to_buckets.py` | Map PrimeTime hierarchy text to bucket JSON |
| `plot_act_pt_matmul_200us.py` | Matplotlib plots for ACT vs PT matmul-style comparisons |
| `bundle_scaling_table.py` | Scaling table over `docs/calibration/run_bundles/**/pt_comparison_bundle.json` |
| `deep_dive_bundle_audit.py` | Audit helper over saved bundle JSON |
| `init_phase_f_run_bundle.sh` | Create `docs/calibration/run_bundles/<name>/` from the tracked `_template` |
| `phase_f_preflight.sh` | Thin wrapper / preflight hook |
| `risky_run_preflight.sh` | Guardrails before long runs |
| `eda/` | `pt_post_sanity.tcl`, `run_pt_post_sanity.sh` |

**Calibration markdown** lives in the parent MLIR repo: **`docs/calibration/`** (index + `run_bundles/_template`), not under this directory.
