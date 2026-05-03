# pt2hlo workflow

This folder provides a small CLI that converts PyTorch models into XLA HLO text workloads.

The exporter now creates sample inputs on the host first and only then moves them
to XLA. This avoids accidentally capturing random input generation ops such as
`rng-bit-generator` into the emitted HLO.

## What you get

- `pt2hlo.py`: main CLI.
- `examples/simple_mlp.py`: runnable example model module.
- `examples/attention_core64.py`: real attention-core example aligned with `ATTN_TILE64`.
- `examples/qkv_dse_attention.py`: backend-friendly LLM-style attention skeleton.
- `requirements.txt`: Python package hints.

## Requirements

Use a Python environment with compatible versions of:

- `torch`
- `torch-xla`

Install from `requirements.txt` and pin versions as needed for your platform.

## Input format

The script accepts one or more `--input` flags.

Format:

`--input "<shape>:<dtype>"`

Examples:

- `--input "1,128:float32"`
- `--input "32,3,224,224:float32"`
- `--input "1,64:int64"`

If your model has multiple positional inputs, provide multiple `--input` flags in order.

## Basic usage

```bash
cd /scratch/krish/MLIR-hardware-analysis/submodule/act/pt2hlo
python pt2hlo.py \
  --model-file examples/simple_mlp.py \
  --model-entry build_model \
  --input "1,128:float32" \
  --output-dir out
```

This writes:

- `out/workload_000.hlo`
- `out/workload_000.meta.json`

## Multiple workloads

You can pass several `--workload` flags, each containing a semicolon-delimited input group.

Each group is a full call signature for one model run.

```bash
python pt2hlo.py \
  --model-file examples/simple_mlp.py \
  --model-entry build_model \
  --workload "1,128:float32" \
  --workload "4,128:float32" \
  --output-dir out
```

## Tiny LLM-style example workload

Use the provided transformer block in `examples/tiny_llm_block.py`.

Input shape follows `[batch, seq_len, hidden_size]` with hidden size `128`.

Single workload:

```bash
python pt2hlo.py \
  --model-file examples/tiny_llm_block.py \
  --model-entry build_model \
  --input "1,128,128:float32" \
  --output-dir out_tiny_llm
```

Multiple sequence lengths:

```bash
python pt2hlo.py \
  --model-file examples/tiny_llm_block.py \
  --model-entry build_model \
  --workload "1,64,128:float32" \
  --workload "1,128,128:float32" \
  --workload "1,256,128:float32" \
  --output-dir out_tiny_llm_multi
```

## Model entry behavior

`--model-entry` should point to one symbol in your `--model-file`:

- a `torch.nn.Module` instance
- a `torch.nn.Module` subclass (instantiated with no args)
- a callable returning a `torch.nn.Module`

If you need constructor args, create a small factory function in your model file and use that as `--model-entry`.

## ISA-aware op validation

`pt2hlo.py` can check generated HLO ops against an allowed set before you feed ACT.

Useful flags:

- `--isa-profile qkv_dse`: built-in conservative allowlist for current `QKV_DSE` workload style.
- `--allow-ops-from-hlo <path>`: infer allowed ops from a known-working HLO file.
- `--allow-ops-file <path>`: custom allowlist (one op per line).
- `--strict-ops`: fail export if unsupported ops are found.

Example using ACT's known workload as reference:

```bash
python pt2hlo.py \
  --model-file examples/simple_mlp.py \
  --model-entry build_model \
  --input "1,128:float32" \
  --allow-ops-from-hlo ../workloads/llm_mixed_attention_large_workable.hlo \
  --strict-ops \
  --output-dir out_checked
```

Example using built-in `qkv_dse` profile:

```bash
python pt2hlo.py \
  --model-file examples/tiny_llm_block.py \
  --model-entry build_model \
  --input "1,128,128:float32" \
  --isa-profile qkv_dse \
  --strict-ops \
  --output-dir out_tiny_llm_checked
```

If unsupported ops appear, the script prints their names and line numbers in the generated HLO.
It also prints compatibility hints for common sources such as:

- device-side random input generation
- `LayerNorm` lowering
- `GELU` lowering
- numerically stable softmax lowering

## Backend-friendly LLM-style workload

If you want a PyTorch example that is intentionally shaped to stay close to the
current `QKV_DSE` ISA, use `examples/qkv_dse_attention.py`.

```bash
python pt2hlo.py \
  --model-file examples/qkv_dse_attention.py \
  --model-entry build_model \
  --input "512,64:float32" \
  --isa-profile qkv_dse \
  --strict-ops \
  --output-dir out_qkv_dse_attention
```

This example avoids:

- `LayerNorm`
- `GELU`
- numerically stabilized softmax
- head split/merge transpose chains

## Attention core for PyTorch -> HLO -> ACT

If you want the cleanest end-to-end path from PyTorch into an ACT workload,
use `examples/attention_core64.py`.

This example is a real attention core with three inputs:

- `q`: `[64,64]`
- `k`: `[64,64]`
- `v`: `[64,64]`

and computes:

`softmax(q @ k^T) @ v`

It intentionally uses a manual `exp / sum / divide` softmax instead of
`torch.softmax(...)` so the exported HLO stays close to the current ACT
reference workload in `../workloads/gemmini_anchor_attention_tile64.hlo`.

Example:

```bash
python pt2hlo.py \
  --model-file examples/attention_core64.py \
  --model-entry build_model \
  --input "64,64:bfloat16" \
  --input "64,64:bfloat16" \
  --input "64,64:bfloat16" \
  --isa-profile attn_tile64 \
  --strict-ops \
  --output-dir out_attention_core64
```

If you want the strongest check, validate against the known workable HLO
instead of only the built-in profile:

```bash
python pt2hlo.py \
  --model-file examples/attention_core64.py \
  --model-entry build_model \
  --input "64,64:bfloat16" \
  --input "64,64:bfloat16" \
  --input "64,64:bfloat16" \
  --allow-ops-from-hlo ../workloads/gemmini_anchor_attention_tile64.hlo \
  --strict-ops \
  --output-dir out_attention_core64_checked
```

This is the intended minimal path when the goal is:

`PyTorch -> HLO -> ACT backend`
