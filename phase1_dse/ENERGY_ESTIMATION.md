# Energy estimation + ACT compile runbook

This fork’s workflow has two layers:

1. **ACT compiler backends** (Rust + OR-Tools): compile `.hlo` → generated Python kernel + **PII** logs (candidate programs).
2. **Static energy on `.pii`** (Python `dse` package): cost model + `primitive_hw_config.json` → **pJ** (+ optional plots).

Use the **MICRO’25 tutorial container** for (1) on cluster machines where the checked-in `backends/`* binaries match the image (glibc, OR-Tools). Use **host Python** for (2) when you only need energy on saved `.pii` files.

---

## A. One-time: Podman storage + pull tutorial image

From the **MLIR-hardware-analysis repo root** (where `setup_act_docker.sh` lives):

```bash
bash ./setup_act_docker.sh
```

That writes `~/.config/containers/storage.conf` (graph under `<repo>/.act-podman/`), runs `podman system migrate`, then `submodule/act/tutorials/micro25/docker.sh --setup`.

### Registry menu (“Please select an image”)

If Podman asks you to pick a registry, choose the line that starts with `**docker.io/devanshdvj/act-tutorials**` (often the **last** option). Do **not** use `registry.access.redhat.com` or `registry.redhat.io` for this public image.

If pulls still fail with `/etc/gshadow` / subuid errors, ask admins for `**/etc/subuid` + `/etc/subgid`**, or run on a laptop with Docker Desktop.

---

## B. Start the ACT tutorial shell (compile-capable image)

```bash
cd submodule/act/tutorials/micro25
./docker.sh --compile
```

Inside the container you should see something like `(act) root@…:/workspace#`. Here `**/workspace**` is the `**submodule/act**` tree mounted read/write.

```bash
conda activate act   # if needed
cd /workspace
```

---

## C. Which ISAs / backends exist (this repo)

Prebuilt binaries live in `**/workspace/backends/**` (host: `submodule/act/backends/`). TAIDL sources are under `**isa_examples/**`; generated Rust/C++ lives under `**targets/<ISA>/backend/**`.


| Backend binary    | TAIDL spec                    | `targets/…`                   | What it is                                                                                                                                                             |
| ----------------- | ----------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**GEMMINI_17**`  | `isa_examples/GEMMINI_17.py`  | `targets/GEMMINI_17/backend`  | Gemmini-style **16-instruction** ISA: scratchpad/accumulator, **mvin/mvout**, mesh **GEMM**, norms, etc. Good default for **matmul** and many **Gemmini-shaped** HLOs. |
| `**ATTN_TILE64`** | `isa_examples/ATTN_TILE64.py` | `targets/ATTN_TILE64/backend` | **64×64 attention tile**: scratchpad + matmul + **softmax** unit; HLO subset aimed at **scaled dot-product attention core** (Q, K, V → O).                             |
| `**QKV_DSE`**     | `isa_examples/QKV_DSE.py`     | `targets/QKV_DSE/backend`     | **QKV variant for DSE**: larger on-chip buffers (`d1`/`d2` **4096×64**), separate backend name from tutorial `**QKV`** so benchmarks do not collide.                   |


**Not a separate binary here:** tutorial `**QKV`** (`isa_examples/QKV.py`) — generate with `python QKV.py` per MICRO exercise if you need `backends/QKV`.

**Regenerate primitive / ISA cost JSON** (host or container, from `act/` root):

```bash
bash scripts/bash/run_isa_primitives.sh --isa-name GEMMINI_17
bash scripts/bash/run_isa_primitives.sh --isa-name ATTN_TILE64
bash scripts/bash/run_isa_primitives.sh --isa-name QKV_DSE
```

For calibration-style HW JSON, point `HW_RESOURCE_CONFIG` at `MLIR-hardware-analysis/submodule/.cursor/primitive_hw_config_micro.json` (see `ACT_CALIBRATION_FORK_NOTES.md`).

---

## D. Compile an HLO with a backend (ACT)

Pattern (inside `**/workspace**`):

```bash
mkdir -p asm/my_run log/my_run
./backends/<ISA_NAME> \
  --input workloads/<file>.hlo \
  --output asm/my_run/kernel.py \
  --log log/my_run
```

**Example (Gemmini, smallest workload):**

```bash
mkdir -p asm/demo_matmul log/demo_matmul
./backends/GEMMINI_17 \
  --input workloads/matmul_64x64.hlo \
  --output asm/demo_matmul/matmul_64x64.py \
  --log log/demo_matmul/run
```

**Example (attention tile, ATTN backend):**

```bash
mkdir -p asm/demo_attn log/demo_attn
./backends/ATTN_TILE64 \
  --input workloads/attention_64x64.hlo \
  --output asm/demo_attn/out.py \
  --log log/demo_attn/run
```

### Where outputs go


| Artifact                     | Location                                                                                                                                                                                      |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Chosen kernel (Python)**   | Path you passed to `--output` (e.g. `asm/.../kernel.py`).                                                                                                                                     |
| **Compile log + candidates** | `--log` directory: for each candidate index `**N`**, you get `**N.py`** (ASM candidate) and `**N.pii**` (pre-schedule program text). The backend copies the **best** candidate to `--output`. |
| **Metadata**                 | Often `metadata.json` under the log dir (depends on backend).                                                                                                                                 |


---

## E. Workloads (`workloads/*.hlo`) — full list

All paths relative to `**act/`** root (`/workspace` in container).


| File                                        | Summary                                                              |
| ------------------------------------------- | -------------------------------------------------------------------- |
| `matmul_64x64.hlo`                          | Single **64×64 bf16 matmul** (smallest smoke test).                  |
| `attention_64x64.hlo`                       | **64×64** scaled dot-product style attention (Q,K,V → O).            |
| `gemmini_anchor_attention_tile64.hlo`       | Attention tile variant aligned with Gemmini anchor experiments.      |
| `attention_tile64_long_manual.hlo`          | **Longer** manual attention-tile graph (more parameters / ops).      |
| `llm_compute_bound.hlo`                     | **Compute-heavy** wide sequence (`bf16[2048,64]` stream) + tile ops. |
| `llm_mem_bound.hlo`                         | **Memory-heavy** residual-style ops at long sequence length.         |
| `llm_mem_bound_kv_merge_large_workable.hlo` | **KV-merge / memory** style large tensors, workable for compile.     |
| `llm_mixed_attention_large_workable.hlo`    | **Mixed** attention + large sequence block.                          |
| `llm_layer_kernel_tile64_anchor.hlo`        | **Layer-style** kernel with multiple **64×64** tiles.                |
| `transformer_block64_medium_manual.hlo`     | **Manual** medium transformer-style block at 64-wide tiles.          |
| `transformer_block64_rich_manual.hlo`       | **Richer** manual transformer block (more tensors).                  |


Pick the backend (**GEMMINI_17** vs **ATTN_TILE64** vs **QKV_DSE**) to match the **HLO ops** your file uses; if compilation errors with “no rewrite”, try the other ISA or simplify the HLO.

---

## F. Static energy (pJ) on `.pii` — `dse.energy_workload`

From `**act/`** root on **any** machine with Python deps:

```bash
export PYTHONPATH=".:phase1_dse"
python3 -m dse.energy_workload \
  --input log/demo_matmul/run/0.pii \
  --hw_config phase1_dse/dse/config/primitive_hw_config.json \
  --out demo_output/energy_run \
  --plot
```

- `**--input**`: one `.pii` file or a directory of candidates.
- `**--hw_config**`: `primitive_hw_config.json` (abstraction-class energies).
- `**--mapping_json**`: optional `phase1_dse/dse/docs/hardware_mapping_interface_package/final_mapping.json` (defaulted inside code if present).
- `**--plot**`: writes `demo_output/energy_run/plots/energy_by_class_<stem>.png`.
- **JSON/CSV**: `energy_summary.json`, `candidate_energy.csv`, `energy_detail_*.json` under `--out`.

---

## G. ISA energy plots (instruction breakdown) — `plot_isa_workload_costs.py`

Runs from `**act/`** root; uses the backend’s `**python/cost/model.py`**.

**Workload view** (one compiled `.py` per row, e.g. single output kernel):

```bash
python3 plot_isa_workload_costs.py \
  --backend-dir targets/GEMMINI_17/backend \
  --compiled-dir asm/demo_matmul \
  --out-dir demo_output/isa_plots_workload
```

**Candidate view** (every `N.py` under a compile log dir):

```bash
python3 plot_isa_workload_costs.py \
  --backend-dir targets/GEMMINI_17/backend \
  --candidate-dir log/demo_matmul/run \
  --out-dir demo_output/isa_plots_candidates
```

**Both** in one output directory:

```bash
python3 plot_isa_workload_costs.py \
  --backend-dir targets/ATTN_TILE64/backend \
  --compiled-dir asm/demo_attn \
  --candidate-dir log/demo_attn/run \
  --out-dir demo_output/isa_plots_attn
```

Outputs typically include:

- `*_total_isa_energy.png`, `*_instruction_energy_breakdown.png`, zoom / delta charts  
- `*_cost_summary.csv` / `.json`

---

## Quick reference paths (host, from MLIR repo)


| What                 | Path                                                             |
| -------------------- | ---------------------------------------------------------------- |
| Setup script         | `MLIR-hardware-analysis/setup_act_docker.sh`                     |
| ACT root             | `MLIR-hardware-analysis/submodule/act/`                          |
| Tutorial             | `…/submodule/act/tutorials/micro25/docker.sh`                    |
| Backends             | `…/submodule/act/backends/{GEMMINI_17,ATTN_TILE64,QKV_DSE}`      |
| Workloads            | `…/submodule/act/workloads/*.hlo`                                |
| DSE / energy package | `…/submodule/act/phase1_dse/dse/`                                |
| HW JSON (default)    | `…/submodule/act/phase1_dse/dse/config/primitive_hw_config.json` |
| Plot script          | `…/submodule/act/plot_isa_workload_costs.py`                     |


