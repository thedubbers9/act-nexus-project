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
