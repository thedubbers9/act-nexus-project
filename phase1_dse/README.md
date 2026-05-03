# Phase-1 DSE (isolated tree)

This folder holds the **Phase-1 design-space exploration (DSE)** Python package and assets. It is **optional** for the **Gemmini ↔ PrimeTime ↔ ACT** calibration workflow in `MLIR-hardware-analysis` (that path uses `estimate_primitive_resources.py`, `primitive_hw_config_micro.json`, and `plot_isa_workload_costs.py` instead).

## What ships in this fork

| Path | Role |
|------|------|
| `dse/` | Python package `dse`: **static energy on `.pii`** (`energy_workload` / `energy_estimate`), PII parsing, hardware-mapping metadata, tests, `primitive_hw_config.json`. |
| `dse/scripts/run_gemmini_dse_demo.sh` | One-shot demo: `python3 -m dse.energy_workload` (+ optional plots). |
| `_archived_pii_roofline/` | Archived **roofline / `dse.forward_bound`** hook, old YAML configs, and compile-driver scripts. See `_archived_pii_roofline/README.md`. |

Compiler flag **`--pre-schedule-dse`** is a **deprecated no-op** (prints a skip notice). Roofline code was removed from the live import path; restore from `_archived_pii_roofline/` if you need it.

## How Python finds `dse`

Run from the **ACT repository root**:

```bash
export PYTHONPATH=".:$(pwd)/phase1_dse"
python3 -m dse.energy_workload --help
```

## Demo

```bash
cd /path/to/submodule/act
bash phase1_dse/dse/scripts/run_gemmini_dse_demo.sh
```

## Why this was moved under `phase1_dse/`

To separate **pre-schedule analytical DSE** (now mostly archived here) from the **calibration-grade ACT energy** pipeline and reduce confusion about which tools participate in PrimeTime comparisons.

## Full runbook (Docker + ACT + workloads + plots)

See **[ENERGY_ESTIMATION_RUNBOOK.md](ENERGY_ESTIMATION_RUNBOOK.md)**.
