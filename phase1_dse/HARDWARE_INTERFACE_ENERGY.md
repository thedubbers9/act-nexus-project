# Hardware Interface and Realization-First Energy (end-to-end)

This document explains the hardware-interface layer used for energy estimation.  
Paths are from the ACT repository root (`submodule/act` inside MLIR-hardware-analysis).

For the end-to-end energy-estimation code path, see `ENERGY_ESTIMATION.md` in
this same folder.

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
7. Compute energy using realization action counts and cost coefficients.
8. Sum into per-instruction and per-candidate totals.

This is the same decomposition shown in the hardware-interface breakdown slides:

- semantic primitives,
- fused ops override rules,
- fallback predictor/default primitive-to-IP rules,
- final ISA instruction cost estimate.

---

## 3) Primitive mapping (Gemmini target)

The mapping is target-specific. For this project, the target is Gemmini, so the
interface describes how TAIDL/ACT primitives can be realized by Gemmini hardware
blocks such as the systolic array, scratchpad, accumulator, DMA path, or Rocket
fallback.

The runtime mapping used by the estimator is:

`phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/final_mapping.json`

That JSON is formed from three source tables:

| Source file | What it describes |
| --- | --- |
| `primitive_realizations.csv` | Which realizations are legal for each semantic primitive. |
| `realization_ip_flow.csv` | Which hardware blocks are activated by each realization. |
| `realization_cost_tags.csv` | Which cost bucket/coefficient family each realization should use. |

The estimator uses this mapping in two steps:

1. Normalize the primitive name, for example `gemm`, `matmul`, and `dot` all
   become the canonical primitive `dot`.
2. Select the realization, for example small dot products can use
   `systolic_gemm`, while a 64x64 GEMM on an 8x8 array selects
   `loop_ws_tiled`.

After this, the realization is expanded into hardware actions and energy is
charged from the coefficient table.

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

### 3.3 Current realization inventory

This is the current realization inventory in the Gemmini mapping. The important
thing is that a primitive is not forced to have one cost forever; it can choose a
realization based on shape, fusion, and where the data lives.

| Primitive | Realizations in the mapping |
| --- | --- |
| `add` | `fused_bias_in_matmul`, `acc_accumulate`, `standalone_rocket` |
| `bitcast_convert` | `logical_view`, `physical_copy` |
| `broadcast` | `virtual_broadcast`, `materialized_broadcast`, `fused_broadcast` |
| `concatenate` | `dma_staging` |
| `constant` | `host_to_device` |
| `convert` | `mvin_scale`, `acc_scale_read`, `standalone_rocket` |
| `copy` | `dma_offchip`, `onchip_move` |
| `divide` | `standalone_rocket`, `fused_quant_scale` |
| `dot` | `systolic_gemm`, `systolic_gemm_transposed`, `loop_ws_tiled` |
| `dynamic_update_slice` | `standalone_rocket` |
| `exp` | `config_norm_path`, `standalone_rocket` |
| `exponential` | `config_norm_path`, `standalone_rocket` |
| `maximum` | `relu_activation`, `maxpool_on_store`, `standalone_rocket` |
| `minimum` | `standalone_rocket` |
| `multiply` | `standalone_rocket` |
| `parameter` | `host_to_device` |
| `reduce` | `acc_partial_passes`, `standalone_rocket` |
| `reduce_add` | `acc_partial_passes`, `standalone_rocket` |
| `reshape` | `logical_view`, `physical_relayout` |
| `select_eq_var` | `standalone_rocket` |
| `select_lt` | `standalone_rocket`, `fused_relu_like` |
| `slice` | `affine_addressing`, `dma_extract` |
| `subtract` | `acc_path`, `standalone_rocket` |
| `transpose` | `folded_into_matmul`, `explicit_relayout` |
| `xor` | `standalone_rocket` |

The mapping also contains fused-pattern entries. These are used when several
primitive operations should be treated as one hardware action rather than as
separate unrelated operations:

| Fused pattern | Purpose |
| --- | --- |
| `matmul_bias` | Fold bias add into the matmul/accumulator path. |
| `matmul_bias_relu` | Fold matmul, bias, and ReLU into one fused Gemmini path. |
| `matmul_relu` | Fold activation into matmul output handling. |
| `softmax_fused` | Recognize a softmax-like sequence as one fused higher-level path. |
| `exp_reduce_fused` | Recognize exp + reduction style normalization. |
| `conv_ws_fused` | Recognize convolution lowered to a weight-stationary style schedule. |

### 3.4 Realizations used in the current Gemmini graphs

The presentation graphs use a small subset of the full mapping:

| Workload | Main semantic primitive | Selected realization | Hardware actions charged |
| --- | --- | --- | --- |
| `GEMM_64x64` | `dot` | `loop_ws_tiled` | 8x8 mesh passes, SPAD movement, ACC read/write/RMW chunks, command hooks. |
| `MAC_64x64` | `dot` plus accumulated `C` input | `loop_ws_tiled` | Same mesh schedule as GEMM, plus extra preload/SPAD/ACC activity for `A * B + C`. |
| `ADD_64x64` | `add` | `acc_accumulate` / accumulator-style path | SPAD/ACC movement and accumulator activity, not useful mesh MAC work. |
| candidate graph | multiple PII candidates | selected per primitive | Each candidate is rescored by the same action-count model, then ranked by ACT-estimated energy. |

This is why the hardware interface matters: ACT can compare candidate PII graphs
using target-aware hardware costs instead of treating every semantic primitive as
an abstract software operation.

---

## 4) Systolic-array worked example

This is the main example used in the presentation: `GEMM_64x64` on an 8x8
Gemmini array.

Semantic operation:

`D = A * B`

Hardware-interface decision:

`gemm -> dot -> loop_ws_tiled`

The condition is that the operation is larger than the physical 8x8 array, so
one semantic GEMM cannot be represented as one mesh action. The realization
expands it into repeated 8x8 actions.

### 4.1 Tile counts

For a 64x64 GEMM on an 8x8 array:

- The 64x64 output is split into `8 x 8 = 64` output tiles.
- Each output tile needs 8 K-dimension tile passes.
- Total mesh compute passes are `64 output tiles x 8 K passes = 512`.

This gives the command/action counts used by both the full-chip testbench and
the ACT model:

| Action | Count | Meaning |
| --- | ---: | --- |
| compute | 512 | One 8x8 mesh pass per output tile per K tile. |
| mvin | 1024 | Load A and B operand tiles for each compute pass. |
| mvout | 64 | Write each final output tile once. |
| config | 1536 | Current probe emits 3 config commands per compute pass. |
| flush | 513 | One flush per compute pass plus final drain. |

The important part is that the hardware interface exposes these counts instead
of hiding the whole workload behind one abstract `gemm` cost.

### 4.2 Energy coefficients

The action counts are multiplied by measured or proxy energy coefficients:

| Hardware bucket | Current coefficient source |
| --- | --- |
| MAC array | Isolated mesh IP slope, about `2601.7 pJ` per 8x8 mesh pass. |
| SPAD / on-chip movement | SRAM/SPAD byte proxy, currently `19 pJ/byte`. |
| ACC data activity | Isolated ACC IP delta slope plus the enabled ACC active-envelope term. |
| Control / command activity | Optional linear command-count model used for presentation graphs. |

So the ACT energy is conceptually:

```text
E_ACT =
  mesh_passes        * E_mesh_pass
+ spad_bytes         * E_spad_byte
+ acc_read_chunks    * E_acc_read_chunk
+ acc_write_chunks   * E_acc_write_chunk
+ acc_rmw_chunks     * E_acc_rmw_chunk
+ acc_active_cycles  * E_acc_active_cycle
+ command_counts     * E_command
```

For the MAC array part of `GEMM_64x64`:

```text
E_mesh = 512 * 2601.7 pJ
       = 1.332 uJ
```

### 4.3 ACC data energy vs ACC clocked energy

The ACC bucket needs one extra clarification.

The isolated ACC IP can be interpreted in two ways:

| ACC interpretation | What it means | When to use it |
| --- | --- | --- |
| data-action slope | Difference between read/write/RMW activity and idle, per 32B chunk. | Best for action-count-only ACT modeling. |
| clocked active envelope | The whole clocked active window around ACC activity. | Useful when trying to match full-chip PrimeTime dynamic buckets. |

The current presentation graph includes both ACC pieces: the idle-subtracted
data-action slope and the ACC active envelope. This is deliberate because the
PrimeTime ACC bucket includes clocked hierarchy activity. If we later want a
strict action-count-only graph, we can disable the active envelope in the config.

---

## 5) Files and responsibilities

### Core estimator files


| File                                    | Role                                                                                                 |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `phase1_dse/dse/src/parse_pii.py`       | Parse `.pii` lines into instruction records (`name`, `attrs`, shape, dtype, line).                   |
| `phase1_dse/dse/src/features.py`        | Helper for shape, op-count, and byte-count terms used by the estimator.                              |
| `phase1_dse/dse/src/energy_estimate.py` | Main realization resolver and Gemmini schedule/action energy model.                                  |
| `phase1_dse/dse/src/energy_workload.py` | CLI driver for batch runs, outputs summary/detail JSON and CSV.                                      |


### Mapping and coefficient files


| File                                                                                              | Role                                                                     |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/final_mapping.json`         | Runtime primitive-to-realization mapping, fused patterns, and cost tags. |
| `phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/README.md`                  | Mapping package schema and regeneration notes.                           |
| `phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/primitive_realizations.csv` | Primitive realization source table.                                      |
| `phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/realization_ip_flow.csv`    | Per-realization ordered IP flow.                                         |
| `phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/realization_cost_tags.csv`  | Per-realization cost-tag assignment hints.                               |
| `phase1_dse/dse/config/primitive_hw_config_micro.json`                                            | Current Gemmini micro energy config used by the active graphs.           |
| `phase1_dse/dse/config/primitive_hw_config.json`                                                  | Legacy/portable config in ACT tree (compatibility path still supported). |


---

## 6) Step-by-step workflow (how to run)

1. Prepare `.pii` candidate(s) from ACT backend.
2. Confirm mapping file (`final_mapping.json`) and hw config (`primitive_hw_config*.json`) to use.
3. Run:

```bash
export PYTHONPATH="phase1_dse:${PYTHONPATH}"
python3 -m dse.energy_workload \
  --input phase1_dse/dse/examples/qkv_like.pii \
  --hw_config phase1_dse/dse/config/primitive_hw_config_micro.json \
  --out phase1_dse/dse/output/my_energy_run \
  --mapping_json phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/final_mapping.json
```

1. Inspect outputs:
  - `energy_summary.json` (candidate totals),
  - `energy_detail_*.json` (per-instruction primitive, realization, cost tag, and energy),
  - `candidate_energy.csv`.
2. Validate selected realizations for expected ops, especially `dot`,
   `loop_ws_tiled`, `acc_accumulate`, and fused matmul patterns.

---
