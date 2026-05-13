# ACT Energy Estimation

This document explains the current energy-estimation flow used by this fork.
It focuses only on the active `.pii` energy path and the Gemmini action-count
model used for the presentation graphs.

Read with:

- `README.md` for the high-level entry point.
- `HARDWARE_INTERFACE_ENERGY.md` for the primitive-to-realization mapping.

---

## 1) What Goes In and What Comes Out

Input:

```text
ACT candidate .pii files
```

Output:

```text
energy_summary.json
candidate_energy.csv
energy_detail_*.json
optional plots/
```

The estimator does not rerun physical design and does not rerun PrimeTime. It
reads saved ACT candidate programs and estimates energy from the operations and
hardware actions implied by those programs.

---

## 2) Main Command

Run from the ACT repo root:

```bash
export PYTHONPATH=".:phase1_dse"
python3 -m dse.energy_workload \
  --input log/demo_matmul/run \
  --hw_config phase1_dse/dse/config/primitive_hw_config_micro.json \
  --mapping_json phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/final_mapping.json \
  --out demo_output/energy_matmul_all \
  --plot
```

`--input` can be one `.pii` file or a directory of candidate `.pii` files.

---

## 3) Source Files

| File | What it does |
| --- | --- |
| `dse/src/parse_pii.py` | Parses ACT `.pii` text into `CandidateProgram` and `InstructionCall` objects. |
| `dse/src/energy_workload.py` | CLI wrapper. Loads candidates, calls the estimator, writes JSON/CSV/plots. |
| `dse/src/energy_estimate.py` | Main estimator. Selects realizations, expands Gemmini actions, and sums energy. |
| `dse/src/features.py` | Helper for shape, width, op-count, and byte-count terms used by `energy_estimate.py`. |
| `dse/config/primitive_hw_config_micro.json` | Current Gemmini coefficients and schedule-event model settings. |
| `dse/hardware_interface/hardware_mapping_interface_package/final_mapping.json` | Runtime primitive-to-realization mapping. |

Supporting scripts:

| File | Role |
| --- | --- |
| `estimate_primitive_resources.py` | Produces primitive-resource summaries from primitive JSON descriptions. |
| `plot_isa_workload_costs.py` | Plots backend ISA instruction-cost breakdowns. |
| `scripts/make_candidate_energy_profiles_gemm64.py` | Builds candidate energy profile plots for GEMM64 candidates. |

---

## 4) Code Path

`energy_workload.py` is the entry point:

```text
energy_workload.py
  -> parse_pii_dir(...)
      -> parse_pii.py
  -> estimate_program(...)
      -> energy_estimate.py
          -> _normalize_primitive_name(...)
          -> _resolve_mapping(...)
          -> _gemmini_schedule_events_for_instruction(...)
          -> _add_gemmini_schedule_event_memory(...)
          -> _add_gemmini_accumulator_events(...)
  -> write energy_summary.json
  -> write candidate_energy.csv
  -> write energy_detail_*.json
```

Step by step:

1. `parse_pii.py` reads every `.pii` line and records the op name, shape, dtype,
   buffer, offset, children, and source line number.
2. `energy_estimate.py` normalizes op names into semantic primitives. For
   example, `gemm`, `gemm_acc`, `matmul8`, and `dot` map to primitive `dot`.
3. The estimator reads `final_mapping.json` and selects a Gemmini realization
   for the primitive.
4. For Gemmini tiled workloads, the realization expands into hardware schedule
   events such as mesh tile passes, SPAD bytes, ACC chunks, config commands,
   flush commands, and DMA-style movement.
5. The estimator reads coefficients from `primitive_hw_config_micro.json`.
6. Each energy term is added into `by_cost_tag_pj`, `by_realization_pj`, and the
   per-instruction detail JSON.

---

## 5) Current Energy Buckets

The current Gemmini model is an action-count model:

```text
E_ACT =
  mesh_passes      * E_mesh_pass
+ spad_bytes       * E_spad_byte
+ acc_read_chunks  * E_acc_read_chunk
+ acc_write_chunks * E_acc_write_chunk
+ acc_rmw_chunks   * E_acc_rmw_chunk
+ command_counts   * E_command
```

Current buckets:

| Bucket | How ACT estimates it |
| --- | --- |
| MAC Array | Count 8x8 mesh passes and multiply by the mesh tile coefficient. |
| SPAD / on-chip movement | Count local/SPAD bytes and multiply by the SPAD byte proxy. |
| ACC | Count ACC read/write/RMW 32B chunks, then charge isolated ACC data-slope energy plus the enabled ACC active-envelope term. |
| Control / command activity | Optional linear command-count model used by the latest presentation graph. |
| Off-chip memory | Excluded for PrimeTime comparison because full-chip PrimeTime does not model external DRAM/HBM. |

The current SPAD byte proxy is `19 pJ/byte`. This is a proxy for on-chip SRAM
movement, not a full SPAD hierarchy model. The remaining full-chip gap is mostly
SPAD active-window/control overhead, ACC active hierarchy overhead, execute
controller activity, command queues, and DMA/load-store controller activity.

---

## 6) ACC Modeling Note

The ACC model has two possible interpretations:

| Term | Meaning |
| --- | --- |
| ACC data slope | Idle-subtracted read/write/RMW energy per 32B chunk from the isolated ACC IP sweep. |
| ACC clocked envelope | Clocked active-window energy for the ACC hierarchy while the IP is active. |

For a clean action-count-only ACT model, the data slope is the safest term
because it is comparable to the mesh action slope. For the latest
presentation-facing PT-vs-ACT graph, we intentionally include the ACC active
envelope as well, because the PrimeTime ACC bucket includes clocked ACC
hierarchy activity. This is why the plotted ACT ACC bucket is much larger than
the idle-subtracted data slope alone.

The coefficient values are read from:

```text
dse/config/primitive_hw_config_micro.json
```

Look under:

```text
gemmini_accumulator_event_model.energy_pj_per_tile
gemmini_accumulator_event_model.active_envelope
```

---

## 7) Worked Example: GEMM 64x64

Semantic workload:

```text
D = A * B
```

Primitive and realization:

```text
gemm -> dot -> loop_ws_tiled
```

Why `loop_ws_tiled`?

```text
64x64 GEMM is larger than the physical 8x8 Gemmini array.
```

Tile math:

```text
M tiles = 64 / 8 = 8
N tiles = 64 / 8 = 8
K tiles = 64 / 8 = 8

output tiles     = 8 * 8     = 64
mesh tile passes = 8 * 8 * 8 = 512
```

Main action counts:

| Action | Count | Meaning |
| --- | ---: | --- |
| compute | 512 | One 8x8 mesh pass per output tile per K tile. |
| mvin | 1024 | Load A and B tiles for each compute pass. |
| mvout | 64 | Write each final output tile once. |
| config | 1536 | Three config commands per compute pass in the current probe. |
| flush | 513 | One flush per compute pass plus final drain. |

Mesh energy example:

```text
E_mesh = 512 * E_mesh_pass
       = 512 * 2601.7 pJ
       = 1.332 uJ
```

The full ACT estimate then adds SPAD bytes, ACC chunk events, and command/control
terms from the same schedule.

---

## 8) Worked Example: MAC 64x64

Semantic workload:

```text
D = A * B + C
```

Primitive and realization:

```text
gemm_acc -> dot -> loop_ws_tiled
```

MAC has the same 512 mesh compute passes as GEMM, but it carries extra C/preload
traffic and more ACC read-modify-write behavior. That is why MAC has similar MAC
array energy but higher SPAD/ACC/control energy than GEMM.

---

## 9) Candidate Energy Ranking

For candidate plots, ACT repeats the same process for every candidate `.pii`
file in a log directory:

```text
candidate_0.pii -> total energy
candidate_1.pii -> total energy
candidate_2.pii -> total energy
...
```

The compiler can then choose the lowest-energy candidate, or the analysis script
can plot unique candidate energy profiles for presentation.

This is the important compiler-facing point: the energy model is not just a
post-processing graph. It can act as a cost function for candidate selection.
