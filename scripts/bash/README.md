# Bash Scripts

This folder centralizes shell scripts used across the ACT repo so the interface/demo flow is easier to browse.

## Groups
- `run_*`: workload, demo, and DSE entrypoints.
- `run_isa_primitives.sh`: **generic** ISA → primitive CSV → `taidl_instruction_costs.json` (preferred).
- `run_gemmini_17_primitives.sh`, `run_attn_tile64_primitives.sh`, `run_qkv_dse_primitives.sh`: thin wrappers around `run_isa_primitives.sh` (same env vars as before).
- `docker_*`: container build/run/cleanup helpers.
- `micro25_*`: MICRO'25 artifact/helper scripts.
- `act_clean_generated.sh`: generated-artifact cleanup helper.

## ISA primitive / cost refresh (one script)

Regenerate backend from TAIDL (unless `SKIP_REGEN=1`), export primitive node CSVs, and write `taidl_instruction_costs.json` for **`plot_isa_workload_costs.py`**:

```bash
# From ACT repo root (or any cwd — script resolves act root):
bash scripts/bash/run_isa_primitives.sh --isa-name GEMMINI_17

# Explicit paths (non-standard layout or out-of-tree ISA):
bash scripts/bash/run_isa_primitives.sh \
  --isa /path/to/MyISA.py \
  --backend-dir /path/to/targets/MyISA/backend

# Optional hardware JSON (e.g. PrimeTime-matched micro config):
HW_RESOURCE_CONFIG="$PWD/../.cursor/primitive_hw_config_micro.json" \
  bash scripts/bash/run_isa_primitives.sh --isa-name GEMMINI_17
```

Environment overrides (all optional): `PYTHON_BIN`, `SKIP_REGEN=1`, `HW_RESOURCE_CONFIG`, `PRIMITIVE_JSON`, and the various `*_CSV` / `*_JSON` paths used by `export_primitive_nodes_csv.py` / `estimate_primitive_resources.py`.

## Notes
- These are copied here for discoverability and slide/demo prep.
- Original scripts may still exist in their source locations.
- Prefer this folder when you want a single place to browse runnable shell entrypoints.
- **`run_*` demo drivers** (`run_attn_tile64_from_hlo_demo.sh`, `run_pt2hlo_*`, etc.) are **not** duplicates of `run_isa_primitives.sh`: they orchestrate HLO → backend → plots for specific paper/demo flows.

## Phase-1 DSE (optional)

- Code and configs: `phase1_dse/dse/` (Python package `dse`).
- One-shot demo: `bash phase1_dse/dse/scripts/run_gemmini_dse_demo.sh` from ACT repo root.
- **Not** the PrimeTime calibration path; see `phase1_dse/README.md` and `ACT_CALIBRATION_FORK_NOTES.md`.

## Docker: verify static energy path (non-interactive)

After the ACT image exists (`./docker_build.sh --rebuild` the first time, or `docker pull` if you use a registry copy), run:

```bash
cd /path/to/act/scripts/bash
./docker_verify_energy_path.sh
```

This runs `python3 -m dse.energy_workload` on `phase1_dse/dse/examples/qkv_like.pii` inside the container (no TTY required).

**Rootless Podman:** if `docker pull` / `docker build` fails with `insufficient UIDs or GIDs` or `lchown /etc/gshadow`, the host user needs subordinate ID ranges in `/etc/subuid` and `/etc/subgid`, or use rootful Docker / rootful Podman per your site policy. The verification script cannot fix that from the repo alone.
