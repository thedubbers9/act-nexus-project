# Cursor helper scripts (ported)

These files are copies of `MLIR-hardware-analysis/submodule/.cursor/scripts/` (shell + Python helpers for PrimeTime bundles, PT hierarchy parsing, and preflight checks). They live under `phase1_dse/dse/scripts/` so the ACT + DSE tree carries the same tooling.

**Not copied:** `eda/` (Tcl + `run_pt_post_sanity.sh`) — still under `submodule/.cursor/scripts/eda/`.

| File | Role (short) |
|------|----------------|
| `act_pt_term_bucket_v1.py` | PT term split + bucket table + JSON for calibration workflows |
| `parse_pt_hier_to_buckets.py` | Map PrimeTime hierarchy text to bucket JSON |
| `plot_act_pt_matmul_200us.py` | Matplotlib plots for ACT vs PT matmul-style comparisons |
| `bundle_scaling_table.py` | Scaling table helper for run bundles |
| `deep_dive_bundle_audit.py` | Audit helper over saved bundle JSON |
| `init_phase_f_run_bundle.sh` | Initialize a phase-F style run bundle layout |
| `phase_f_preflight.sh` | Thin wrapper / preflight hook |
| `risky_run_preflight.sh` | Guardrails before long runs |

Re-sync from `.cursor/scripts` if those versions change (run from this directory):

```bash
cp -a ../../../../../.cursor/scripts/*.sh ../../../../../.cursor/scripts/*.py .
```
