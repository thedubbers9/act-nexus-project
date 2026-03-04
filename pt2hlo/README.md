# pt2hlo workflow

This folder provides a small CLI that converts PyTorch models into XLA HLO text workloads.

## What you get

- `pt2hlo.py`: main CLI.
- `examples/simple_mlp.py`: runnable example model module.
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
