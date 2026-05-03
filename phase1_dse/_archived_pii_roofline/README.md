# Archived: PII roofline / pre-schedule characterization

This directory holds the **pre-schedule `.pii` hook** and **roofline-style**
characterization code that used to run as `python3 -m dse.forward_bound` from
generated backends (`--pre-schedule-dse`).

## Why archived

The MLIR-hardware-analysis fork keeps the **hardware mapping → cost model →
static energy on `.pii`** path (`dse.energy_workload` / `energy_estimate.py`)
and does **not** maintain the compile-time roofline hook.

## Contents

| Path | Former role |
|------|-------------|
| `forward_bound.py`, `src/forward_bound.py` | CLI + roofline / latency vs bandwidth sweep on `.pii` |
| `model_forward.py`, `src/model_forward.py` | Forward model / frontier math |
| `src/plots_.py` | Extra meeting-style plots over forward_bound outputs |
| `config/config.yaml`, `config/targets.yaml` | Hardware peak + sweep / SLA targets for forward_bound |
| `tests/test_forward_bound_*.py`, `test_model_forward.py` | Unit tests for the above |
| `scripts/run_gemmini_dse_demo_full_including_roofline.sh` | Old demo: roofline + energy + plots_.py |
| `scripts/run_multi_workload_dse.sh` | Compile HLO with `--pre-schedule-dse` + multi-workload charts |
| `scripts/run_qkv_dse.sh` | QKV compile + forward_bound replay |

## Restoring

1. Move files back into `phase1_dse/dse/` mirroring original paths.
2. Restore `run_pre_schedule_dse` in `targets/*/backend/src/pipeline.rs` (and `generators/backend/generic`) from git history.
3. Re-install bash scripts under `scripts/bash/` if desired.

## Active replacement

- Static energy demo: `phase1_dse/dse/scripts/run_gemmini_dse_demo.sh` (energy only).
- Compiler hook: `run_pre_schedule_dse` is a **no-op** that prints a skip notice.
