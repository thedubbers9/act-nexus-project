# Hardware Interface and Realization-First Energy (end-to-end)

This document explains the hardware-interface layer used for energy estimation.  
Paths are from the ACT repository root (`submodule/act` inside MLIR-hardware-analysis).

For container compile runbooks and ISA primitive refresh, see `ENERGY_ESTIMATION.md` in this same folder.

---

## 1) What the hardware interface is

TAIDL tells us the semantic meaning of an instruction, but not which exact hardware block realizes it.
The hardware interface is the layer between instruction semantics and cost estimation.

It answers this question for each instruction:

- which IP blocks execute it on the target hardware,
- which realization is selected,
- and what cost should be charged.

Conceptually:

`TAIDL instruction semantics -> semantic primitives -> hardware interface (+ target hardware mapping) -> per-ISA instruction cost`

---

## 2) Realization-first flow used in code

The runtime estimator follows this realization-first flow:

1. Parse `.pii` instructions.
2. Decompose into semantic primitives (op names + attrs).
3. Check fused pattern overrides.
4. If no fused match, use primitive fallback/default mapping.
5. Select one realization per primitive.
6. Map realization to a `cost_tag`.
7. Compute energy using op/byte proxies and cost coefficients.
8. Sum into per-instruction and per-candidate totals.

This is the same decomposition shown in the hardware-interface breakdown slides:

- semantic primitives,
- fused ops override rules,
- fallback predictor/default primitive-to-IP rules,
- final ISA instruction cost estimate.

---

## 3) Primitive mapping (Gemmini target)

### 3.1 Compute-side examples

Gemmini is the first target used to instantiate the interface.
A primitive can map to different hardware realizations depending on fusion, storage location (DRAM vs SPAD vs ACC), and shape.

Representative compute mappings:


| Primitive                                                               | Example realizations                                           |
| ----------------------------------------------------------------------- | -------------------------------------------------------------- |
| `dot`                                                                   | `systolic_gemm`, `systolic_gemm_transposed`, `loop_ws_tiled`   |
| `reduce_add`, `add`, `minimum`, `xor`, `subtract`, `multiply`, `divide` | on-chip/acc path when available, otherwise `standalone_rocket` |
| `maximum`                                                               | `relu_activation` or `standalone_rocket`                       |


### 3.2 Data-side examples

Representative data mappings:


| Primitive | Example realizations                |
| --------- | ----------------------------------- |
| `copy`    | `onchip_move`, `dma_offchip`        |
| `reshape` | `logical_view`, `physical_relayout` |


Interpretation:

- `dma_offchip`: cost driven by off-chip bytes and DMA path.
- `onchip_move`: tile stays in accelerator fabric (for example SPAD <-> ACC), no DRAM traffic.
- `logical_view`: no physical byte movement.
- `physical_relayout`: explicit on-chip movement/reorder.

---

## 4) Systolic-array worked example

This section walks through a concrete systolic-array case end-to-end.

### Case A: `gemm`-like op (maps to `dot`)

Given a parsed instruction like `gemm`:

1. **Normalize primitive**
  - `gemm` is normalized to primitive `dot`.
2. **Resolve realization**
  - If output shape has any dimension `> 8` (current DIM-aware heuristic), choose `loop_ws_tiled`.
  - Otherwise choose default `systolic_gemm`.
  - `systolic_gemm_transposed` exists in mapping and can be selected by richer rules; current code mainly uses default + shape heuristic.
3. **Map to cost tag**
  - `systolic_gemm -> tensor_compute`
  - `loop_ws_tiled -> tensor_compute_tiled`
4. **Charge energy**
  - `features._instruction_contrib` provides `ops` and `local/hbm` bytes.
  - Estimator applies: `energy = ops * energy_per_op_pj + bytes * energy_per_byte_pj` using the selected cost-tag row.

### Case B: `gemm_acc`-style behavior

For `gemm_acc`, the primitive still normalizes to `dot` for mesh compute charge.
Accumulator-side behavior is represented through realization/cost-tag mapping and feature terms; depending on the selected realization and mapping tables, cost is charged as mesh compute, tiled compute, or on-chip compute/fallback paths.

---

## 5) Files and responsibilities

### Core estimator files


| File                                    | Role                                                                                                 |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `phase1_dse/dse/src/parse_pii.py`       | Parse `.pii` lines into instruction records (`name`, `attrs`, shape, dtype, line).                   |
| `phase1_dse/dse/src/features.py`        | Compute op/byte proxies (`ops`, `local`, `hbm_read`, `hbm_write`) per instruction.                   |
| `phase1_dse/dse/src/energy_estimate.py` | Realization-first resolver and energy arithmetic (`primitive -> realization -> cost_tag -> energy`). |
| `phase1_dse/dse/src/energy_workload.py` | CLI driver for batch runs, outputs summary/detail JSON and CSV.                                      |


### Mapping and coefficient files


| File                                                                                              | Role                                                                     |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/final_mapping.json`         | Runtime primitive-to-realization mapping, fused patterns, and cost tags. |
| `phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/README.md`                  | Mapping package schema and regeneration notes.                           |
| `phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/primitive_realizations.csv` | Primitive realization source table.                                      |
| `phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/realization_ip_flow.csv`    | Per-realization ordered IP flow.                                         |
| `phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/realization_cost_tags.csv`  | Per-realization cost-tag assignment hints.                               |
| `.cursor/primitive_hw_config_micro.json`                                                          | Cost coefficients keyed by `cost_tags` (project micro profile).          |
| `phase1_dse/dse/config/primitive_hw_config.json`                                                  | Legacy/portable config in ACT tree (compatibility path still supported). |


---

## 6) Step-by-step workflow (how to run)

1. Prepare `.pii` candidate(s) from ACT backend.
2. Confirm mapping file (`final_mapping.json`) and hw config (`primitive_hw_config*.json`) to use.
3. Run:

```bash
export PYTHONPATH="phase1_dse:${PYTHONPATH}"
python3 -m dse.energy_workload   --input phase1_dse/dse/examples/qkv_like.pii   --hw_config /scratch/krish/MLIR-hardware-analysis/submodule/.cursor/primitive_hw_config_micro.json   --out phase1_dse/dse/output/my_energy_run   --mapping_json phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/final_mapping.json
```

1. Inspect outputs:
  - `energy_summary.json` (candidate totals),
  - `energy_detail_*.json` (per-instruction primitive, realization, cost tag, and energy),
  - `candidate_energy.csv`.
2. Validate selected realizations for expected ops (especially `dot` and fused softmax-like patterns).

---

