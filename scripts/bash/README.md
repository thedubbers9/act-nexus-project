# Bash Scripts

This folder holds shell entrypoints for **ISA primitive / instruction-cost refresh**, **Docker (ACT image)**, and **generated-artifact cleanup**. HLO / PT2HLO demo drivers and MICRO'25 helpers live under **`archive/`** (still runnable; paths updated).

## Groups

- **`run_isa_primitives.sh`**: generic ISA → primitive CSV → `taidl_instruction_costs.json` (use this for **`plot_isa_workload_costs.py`** and the hardware-config → per-instruction energy path).
- **`run_gemmini_17_primitives.sh`**, **`run_attn_tile64_primitives.sh`**, **`run_qkv_dse_primitives.sh`**: thin wrappers around `run_isa_primitives.sh`.
- **`docker_*`**, **`docker_verify_energy_path.sh`**: container build/run/cleanup and a non-interactive check that `python3 -m dse.energy_workload` runs in the image.
- **`act_clean_generated.sh`**: remove common generated outputs under `phase1_dse/` and related dirs (see script header).
- **`archive/`**: PT2HLO / compile-and-plot demo chains and **`micro25_*`** tutorial scripts (see `archive/README.md`).

## ISA primitive / cost refresh

Regenerate backend from TAIDL (unless `SKIP_REGEN=1`), export primitive node CSVs, and write `taidl_instruction_costs.json`:

```bash
# From ACT repo root (script resolves act root):
bash scripts/bash/run_isa_primitives.sh --isa-name GEMMINI_17

bash scripts/bash/run_isa_primitives.sh \
  --isa /path/to/MyISA.py \
  --backend-dir /path/to/targets/MyISA/backend

HW_RESOURCE_CONFIG="$PWD/phase1_dse/dse/config/primitive_hw_config_micro.json" \
  bash scripts/bash/run_isa_primitives.sh --isa-name GEMMINI_17
```

Environment overrides (all optional): `PYTHON_BIN`, `SKIP_REGEN=1`, `HW_RESOURCE_CONFIG`, `PRIMITIVE_JSON`, and the `*_CSV` / `*_JSON` paths used by `export_primitive_nodes_csv.py` / `estimate_primitive_resources.py`.

## Phase-1 DSE (optional)

- Package: `phase1_dse/dse/` (`dse`).
- Example demo: `bash phase1_dse/dse/scripts/run_gemmini_dse_demo.sh` from ACT repo root.
- Calibration notes: `phase1_dse/README.md`, `ACT_CALIBRATION_FORK_NOTES.md`.

## Docker: verify static energy path

```bash
cd /path/to/act/scripts/bash
./docker_verify_energy_path.sh
```

Runs `python3 -m dse.energy_workload` on `phase1_dse/dse/examples/qkv_like.pii` inside the container.

**Rootless Podman:** host subuid/subgid ranges may be required for `docker build` / `docker pull`; see site policy if builds fail with ownership or ID-mapping errors.
