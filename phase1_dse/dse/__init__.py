"""Phase-1 DSE package (import name: `dse`).

This tree lives under `act/phase1_dse/dse/` in the MLIR-hardware-analysis fork.
The maintained path is **static energy on `.pii`** (`dse.energy_workload`,
`energy_estimate`, `parse_pii`, `features`) plus hardware-mapping metadata.
It is not the PrimeTime-matched calibration pipeline; see
`estimate_primitive_resources.py` and `plot_isa_workload_costs.py` at the ACT
repo root for that.

The former roofline hook (`python -m dse.forward_bound`, `--pre-schedule-dse`)
is **archived** under `phase1_dse/_archived_pii_roofline/`. Backends may still
accept `--pre-schedule-dse` but the hook is a no-op that prints a skip notice.

Demos set `PYTHONPATH` to include `phase1_dse` so this package imports as top-level `dse`.
"""
