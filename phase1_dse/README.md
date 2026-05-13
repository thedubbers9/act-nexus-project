# Phase-1 ACT Energy Documentation

This folder contains the documentation for the current ACT energy-estimation
flow used in this fork.

The story is intentionally split into only three active documents:

| Document | Purpose |
| --- | --- |
| `README.md` | This overview: what to read first and how the pieces fit together. |
| `ENERGY_ESTIMATION.md` | The detailed energy-estimation flow: `.pii` input, Python code path, config files, action counts, outputs, and a GEMM example. |
| `HARDWARE_INTERFACE_ENERGY.md` | The hardware-interface flow: semantic primitive -> Gemmini realization -> hardware actions -> energy buckets. |

## Current Flow

ACT produces candidate `.pii` programs. The energy estimator reads those files,
maps each instruction to a Gemmini hardware realization, expands tiled workloads
into hardware action counts, and multiplies those counts by measured or proxy
energy coefficients.

```text
HLO workload
  -> ACT backend candidates
  -> .pii candidate files
  -> hardware-interface realization selection
  -> Gemmini action counts
  -> energy coefficients
  -> candidate energy ranking + validation graphs
```

For example, a 64x64 GEMM is not charged as one abstract `gemm`. The hardware
interface maps it to `dot -> loop_ws_tiled`, then the estimator counts the
repeated 8x8 mesh passes, SPAD movement, ACC chunks, and command activity.

---

## Where the Code Lives

| Path | Role |
| --- | --- |
| `dse/src/parse_pii.py` | Parses `.pii` files into instruction records. |
| `dse/src/energy_estimate.py` | Main energy estimator and Gemmini schedule/action expansion. |
| `dse/src/energy_workload.py` | CLI driver for one `.pii` file or a directory of candidates. |
| `dse/src/features.py` | Helper for shape/op/byte terms used by the estimator. |
| `dse/config/primitive_hw_config_micro.json` | Current Gemmini energy coefficients and schedule-event model knobs. |
| `dse/hardware_interface/hardware_mapping_interface_package/final_mapping.json` | Runtime primitive-to-realization mapping. |
| `estimate_primitive_resources.py` | Helper for primitive-resource summaries from JSON descriptions. |
| `plot_isa_workload_costs.py` | Helper for backend ISA instruction-cost plots. |

---

## Basic Command

Run this from the ACT repo root:

```bash
export PYTHONPATH=".:phase1_dse"
python3 -m dse.energy_workload \
  --input log/demo_matmul/run \
  --hw_config phase1_dse/dse/config/primitive_hw_config_micro.json \
  --mapping_json phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/final_mapping.json \
  --out demo_output/energy_matmul_all \
  --plot
```

Main outputs:

| Output | Meaning |
| --- | --- |
| `energy_summary.json` | Overall result for all candidates. |
| `candidate_energy.csv` | One row per candidate with total energy and breakdown fields. |
| `energy_detail_*.json` | Per-instruction primitive, realization, schedule events, and energy. |
| `plots/energy_by_cost_tag_*.png` | Optional plot output when `--plot` is enabled. |

Read `ENERGY_ESTIMATION.md` next for the code-level walkthrough, then
`HARDWARE_INTERFACE_ENERGY.md` for the realization mapping details.
