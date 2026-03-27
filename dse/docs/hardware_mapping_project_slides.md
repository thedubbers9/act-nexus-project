# Slide 1: Problem

## ACT today

- TAIDL tells us what an instruction means
- `primitive_hw_config.json` tells us what calibrated hardware buckets cost
- but we are still missing the clean bridge between the two

## Core problem

- semantic primitives are not always the same as realized hardware
- `dot` may execute on a systolic MAC array
- `softmax` may execute as a fused IP
- partial fusion is common in real accelerators

---

# Slide 2: Why direct primitive-to-hardware mapping breaks

## Example: GEMM

Semantic view:

```text
dot(A, B)
```

Bad hardware interpretation:

```text
dot -> mul + add -> multiplier IP + adder IP
```

Correct hardware interpretation:

```text
dot -> systolic_array_ip -> tensor_compute
```

---

# Slide 3: Real accelerator behavior

Recent AI accelerators usually have some combination of:

- systolic or matrix engine
- vector / elementwise datapath
- DMA / movement hardware
- specialized normalization / softmax / quantization hardware

That means:

- one semantic instruction may use one fused IP
- one semantic instruction may use multiple IPs
- several semantic primitives may share one hardware engine

---

# Slide 4: Proposed interface

## User-facing mapping file

```text
primitive_to_ip
fused_patterns
ip_costs
```

### Meaning

- `primitive_to_ip`
  - fallback mapping for unfused primitives
- `fused_patterns`
  - pattern overrides for fused hardware
- `ip_costs`
  - bridge from user IP names to calibrated abstraction classes

---

# Slide 5: End-to-end stack

```text
TAIDL instruction
    ->
semantic primitive graph
    ->
mapping interface
    ->
realized IP assignments
    ->
abstraction class lookup
    ->
calibrated cost from primitive_hw_config.json
```

## Key benefit

We keep TAIDL semantic while still letting the hardware designer express fusion honestly.

---

# Slide 6: Real-world example

## Accelerator

- systolic array for GEMM
- dedicated softmax IP
- DMA
- small vector datapath

## Mapping

```text
dot -> systolic_array_ip -> tensor_compute
copy -> dma_ip -> contiguous_move
softmax pattern -> softmax_ip -> special_math
```

This matches real accelerator design much better than forcing every operation into tiny arithmetic IPs.

---

# Slide 7: Partial fusion example

## User hardware

- fused `exponential + reduce_add`
- standalone `broadcast`
- standalone `divide`

## Realized mapping

```text
exponential + reduce_add -> exp_reduce_ip
broadcast                -> layout_or_dma_ip
divide                   -> vector_div_ip
```

## Why it matters

This is the difference between a useful co-design interface and a brittle cost spreadsheet.

---

# Slide 8: What we already have

## Calibrated abstraction buckets

- `tensor_compute`
- `special_math`
- `contiguous_move`
- `vector_compute_add`
- `vector_compute_mul`

These already have first-pass calibrated numbers in:

- `dse/config/primitive_hw_config.json`

So the missing piece is not the idea of cost. The missing piece is the mapping mechanism.

---

# Slide 9: Proposed implementation plan

## Phase 1

- document the interface
- provide example mapping JSON

## Phase 2

- add executable mapper pass
- apply fused patterns first
- apply primitive fallback second

## Phase 3

- validate IP names, abstraction classes, and overlaps

## Phase 4

- use realized IP assignments for cost estimation

---

# Slide 10: Project value

## What this gives ACT

- semantic ISA description in TAIDL
- user-editable hardware realization interface
- RTL-calibrated hardware cost model

## Project message

ACT is not just a cost estimator.

It is a co-design interface:

- describe instruction meaning
- describe hardware realization
- estimate cost with calibrated hardware abstractions

That is a stronger and more defensible project story.
