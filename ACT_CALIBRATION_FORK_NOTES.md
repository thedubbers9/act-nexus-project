# ACT fork notes (MLIR-hardware-analysis / `submodule/act`)

Edits below are **local to this workspace** relative to a vanilla ACT checkout. They exist to align demos with Gemmini anchor workloads and to relocate Phase-1 DSE.

## Phase-1 DSE relocation

- The `dse/` package was moved to **`phase1_dse/dse/`**.
- Backend `pipeline.rs` (GEMMINI_17, ATTN_TILE64, Gemmini, QKV_DSE, and `generators/backend/generic`) sets **`PYTHONPATH`** to include `phase1_dse` and defaults **`--dse-hw-config`** to `phase1_dse/dse/config/config.yaml`.
- Bash helpers under `scripts/bash/` were updated to use **`phase1_dse/dse/config/...`** paths where they previously referenced `dse/config/...`.
- **`run_gemmini_17_primitives.sh`** is **not** Phase-1 DSE; it generates `taidl_instruction_costs.json` for calibration. Its **default** `HW_RESOURCE_CONFIG` now points at the JSON copy under `phase1_dse/dse/config/`; override with **`../.cursor/primitive_hw_config_micro.json`** when running from the MLIR-hardware-analysis tree.

## Workload / pt2hlo demo (local)

- **`scripts/bash/run_pt2hlo_attn_tile64_demo.sh`**: default `REFERENCE_HLO` was switched to `workloads/gemmini_anchor_attention_tile64.hlo` (Gemmini anchor attention). **`workloads/attention_tile64_workable.hlo`** was removed as superseded.
- Related small doc edits: `isa_examples/docs_attention_tile64.md`, `pt2hlo/README.md`.

Regenerate or revert these if you need strict upstream demo paths.
