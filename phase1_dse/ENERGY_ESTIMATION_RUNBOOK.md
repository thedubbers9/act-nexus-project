# ENERGY_ESTIMATION_RUNBOOK

Command-first runbook for ACT compile + Phase-1 static energy estimation.

This file intentionally focuses on execution steps and commands.
For hardware-interface concepts and realization mapping details, see:
`HARDWARE_INTERFACE_ENERGY.md`.

---

## 0) Preconditions

From repo root:

```bash
cd /scratch/krish/MLIR-hardware-analysis
git submodule update --init --recursive
```

Set up container image/storage once:

```bash
bash ./setup_act_docker.sh
```

---

## 1) Launch ACT container

```bash
cd /scratch/krish/MLIR-hardware-analysis/submodule/act/tutorials/micro25
./docker.sh --compile
```

Inside container:

```bash
conda activate act
cd /workspace
```

---

## 2) Compile HLO with ACT backend

### Example A: GEMMINI_17 + matmul

```bash
cd /workspace
mkdir -p asm/demo_matmul log/demo_matmul
./backends/GEMMINI_17   --input workloads/matmul_64x64.hlo   --output asm/demo_matmul/matmul_64x64.py   --log log/demo_matmul/run
```

### Example B: ATTN_TILE64 + attention

```bash
cd /workspace
mkdir -p asm/demo_attn log/demo_attn
./backends/ATTN_TILE64   --input workloads/attention_64x64.hlo   --output asm/demo_attn/attention_64x64.py   --log log/demo_attn/run
```

Outputs:

- chosen kernel: `asm/.../*.py`
- candidate logs: `log/.../run/*.py`, `log/.../run/*.pii`

---

## 3) Run static energy estimation on `.pii`

### Single candidate file

```bash
cd /workspace
export PYTHONPATH=".:phase1_dse"
python3 -m dse.energy_workload   --input log/demo_matmul/run/0.pii   --hw_config /scratch/krish/MLIR-hardware-analysis/submodule/.cursor/primitive_hw_config_micro.json   --out demo_output/energy_matmul   --mapping_json phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/final_mapping.json
```

### Whole candidate directory

```bash
cd /workspace
export PYTHONPATH=".:phase1_dse"
python3 -m dse.energy_workload   --input log/demo_matmul/run   --hw_config /scratch/krish/MLIR-hardware-analysis/submodule/.cursor/primitive_hw_config_micro.json   --out demo_output/energy_matmul_all   --mapping_json phase1_dse/dse/hardware_interface/hardware_mapping_interface_package/final_mapping.json   --plot
```

Outputs under `--out`:

- `energy_summary.json`
- `candidate_energy.csv`
- `energy_detail_*.json`
- `plots/energy_by_cost_tag_*.png` (if `--plot`)

---

## 4) (Optional) Refresh ISA primitive cost artifacts

```bash
cd /workspace
bash scripts/bash/run_isa_primitives.sh --isa-name GEMMINI_17
bash scripts/bash/run_isa_primitives.sh --isa-name ATTN_TILE64
bash scripts/bash/run_isa_primitives.sh --isa-name QKV_DSE
```

Use micro hw config explicitly:

```bash
cd /workspace
HW_RESOURCE_CONFIG="/scratch/krish/MLIR-hardware-analysis/submodule/.cursor/primitive_hw_config_micro.json"   bash scripts/bash/run_isa_primitives.sh --isa-name GEMMINI_17
```

---

## 5) (Optional) Plot ISA instruction cost breakdown

```bash
cd /workspace
python3 plot_isa_workload_costs.py   --backend-dir targets/GEMMINI_17/backend   --compiled-dir asm/demo_matmul   --candidate-dir log/demo_matmul/run   --out-dir demo_output/isa_plots_demo
```

---

## 6) Quick troubleshooting

### Backend binary not runnable / dependency mismatch

Use the tutorial container (`./docker.sh --compile`) and run from `/workspace`.

### `No module named dse`

Set Python path before calling `dse.energy_workload`:

```bash
export PYTHONPATH=".:phase1_dse"
```

### No `.pii` files found

Check your compile `--log` directory, then point `--input` at either:

- one `.pii` file, or
- a directory that contains `*.pii`.
