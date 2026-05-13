# Cursor helpers (legacy PrimeTime / run-bundle helpers)

**Canonical location:** `submodule/act/phase1_dse/dse/scripts/cursor_helpers/` inside the ACT submodule (tracked in git).

These are optional helper scripts from the earlier PrimeTime/run-bundle workflow.
They are **not required** to regenerate the current two presentation graphs in
`submodule/act/out/graphs`.

The current graph path is:

- `scripts/make_pt_vs_act_energy_breakdown.py`
- `scripts/make_candidate_energy_profiles_gemm64.py`
- source CSVs under `out/graphs/source_data/`

The helpers below are kept only for ad-hoc PT hierarchy parsing, old bundle
audits, and preflight checks. Some of them reference the older parent-repo
`docs/calibration/` layout, which is not part of the current presentation graph
bundle.

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
