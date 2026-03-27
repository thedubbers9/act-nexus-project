# ATTN_TILE64

`ATTN_TILE64` is a small transformer-attention accelerator definition intended
to be realistic enough for current ML systems work while still being manageable
inside the existing ACT/TAIDL flow.

## Workload family

The target workload is the attention core on one fixed tile:

`O = softmax(Q @ K^T) @ V`

with all operands shaped as `bf16[64,64]`.

This is intentionally:

- current and popular: transformer attention is the dominant LLM primitive
- non-dummy: it includes both contraction and normalization
- manageable: it stays within ACT's current comfortable HLO subset

## Why this first version is limited

The current ACT backend path in this repository already has a good path for:

- `dot`
- `transpose`
- `exponential`
- `reduce` with add region
- `broadcast`
- `divide`
- `add`

This version therefore avoids features that would likely require follow-on
parser and rewrite work, such as:

- numerically stabilized softmax via reduce-max/subtract
- LayerNorm / RMSNorm
- GELU / SwiGLU
- masking
- dynamic sequence length

## Files

- ISA spec: [ATTN_TILE64.py](/scratch/krish/MLIR-hardware-analysis/submodule/act/ATTN_TILE64.py)
- Reference HLO: [attention_tile64_workable.hlo](/scratch/krish/MLIR-hardware-analysis/submodule/act/workloads/attention_tile64_workable.hlo)

## Suggested next steps

1. Generate the backend with `python ATTN_TILE64.py`
2. Try compiling `workloads/attention_tile64_workable.hlo`
3. Once that path is solid, add:
   - stable softmax
   - scaling
   - residual add
   - projection GEMMs
4. After that, consider a v2 ISA with explicit norm support

## Demo commands

If you want the end-to-end ACT demo to use `ATTN_TILE64` instead of the older
`QKV_DSE` path, use the `ATTN_TILE64` scripts directly.

PyTorch to HLO:

```bash
cd /workspace
bash run_pt2hlo_attn_tile64_demo.sh
```

Backend generation, primitive-cost refresh, HLO compile, and plots:

```bash
cd /workspace
bash run_attn_tile64_from_hlo_demo.sh
```

Whole flow:

```bash
cd /workspace
bash run_attn_tile64_pt2hlo_demo.sh
```

Outputs land in:

- `pt2hlo/out_attention_core64`
- `targets/ATTN_TILE64/backend`
- `asm/attn_tile64_demo`
- `demo_output/attn_tile64_demo`

## Richer demo workload

If you want a slightly more complex tile workload with a broader mix of
instructions, use:

- [attention_block64.py](/scratch/krish/MLIR-hardware-analysis/submodule/act/pt2hlo/examples/attention_block64.py)

This computes:

`softmax(Q @ K^T) @ V @ Proj + Residual`

with all tensors shaped as `bf16[64,64]`.

PyTorch to HLO:

```bash
cd /workspace
MODEL_FILE=/workspace/pt2hlo/examples/attention_block64.py \
PT2HLO_OUT=/workspace/pt2hlo/out_attention_block64 \
WORKLOAD_SPECS="64,64:bfloat16;64,64:bfloat16;64,64:bfloat16;64,64:bfloat16;64,64:bfloat16" \
bash run_pt2hlo_attn_tile64_demo.sh
```

Compile and plot:

```bash
cd /workspace
PYTHON_BIN=/opt/miniconda/envs/act/bin/python3 \
PT2HLO_OUT=/workspace/pt2hlo/out_attention_block64 \
COMPILED_DIR=/workspace/asm/attn_tile64_block_demo \
LOG_ROOT=/workspace/log/attn_tile64_block_demo \
PLOT_OUT=/workspace/demo_output/attn_tile64_block_demo \
bash run_attn_tile64_from_hlo_demo.sh
```
