# Archived bash scripts

These are **not** required for the minimal **hardware JSON → primitive rollup → `taidl_instruction_costs.json` → plots** flow (`run_isa_primitives.sh`) or for **`dse.energy_workload`** on `.pii`. They remain supported for tutorials and end-to-end HLO / PT2HLO demos.

| Script | Role |
|--------|------|
| `micro25_*.sh` | MICRO'25 tutorial helpers; `tutorials/micro25/*.sh` exec copies here. |
| `run_pt2hlo_*.sh`, `run_*_pt2hlo_demo.sh`, `run_*_from_hlo_demo.sh` | Orchestrate PyTorch→HLO, backend compile, optional primitive refresh, and `plot_isa_workload_costs.py`. |

From ACT root, example:

```bash
bash scripts/bash/archive/run_attn_tile64_pt2hlo_demo.sh
```

Paths inside these scripts assume this **`archive/`** location (`ROOT_DIR` is three levels above each script).
