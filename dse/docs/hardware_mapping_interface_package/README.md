# Hardware Mapping Interface Package

This folder is a compact handoff package for the ACT hardware mapping interface.

It explains the layer between:

- TAIDL semantic instructions
- primitive decomposition
- optional fused hardware realization
- calibrated hardware-cost lookup

The goal is to answer one practical question:

> Given a semantic instruction stream, which hardware IP executes each part, and which calibrated cost bucket should ACT charge?

## Files in this folder

- [interface_mapping.csv](/scratch/krish/MLIR-hardware-analysis/submodule/act/dse/docs/hardware_mapping_interface_package/interface_mapping.csv)
  - A flat mapping table showing:
    - primitive fallback rules
    - fused pattern overrides
    - the final abstraction class charged by the estimator

- [interface_layer_meanings.csv](/scratch/krish/MLIR-hardware-analysis/submodule/act/dse/docs/hardware_mapping_interface_package/interface_layer_meanings.csv)
  - A glossary-style table explaining what each interface layer means and why it exists.

## Conceptual stack

```text
TAIDL instruction
    ->
semantic primitive graph
    ->
mapping interface
    |- primitive fallback mapping
    |- fused pattern overrides
    ->
hardware IP assignment
    ->
abstraction class lookup
    ->
calibrated cost from primitive_hw_config.json
```

## How to read the interface

### 1. TAIDL stays semantic

TAIDL describes the meaning of an instruction, not the implementation shape.

Examples:

- `gemm` means matrix multiplication semantics
- `softmax` means a semantic sequence like `exponential -> reduce_add -> broadcast -> divide`

### 2. Primitive decomposition exposes the work

The semantic instruction is lowered into primitive operations so ACT can reason about:

- fusion opportunities
- hardware assignment
- cost accounting

### 3. Primitive fallback mapping gives a default hardware home

Each primitive gets a default hardware IP if no fusion rule applies.

Examples:

- `dot -> systolic_array_ip`
- `divide -> vector_div_ip`
- `reduce_add -> reduction_tree_ip`

### 4. Fused patterns override the fallback mapping

If the accelerator has a fused block, the interface can replace multiple primitives with one realized IP.

Example:

- `exponential + reduce_add + broadcast + divide -> softmax_ip`

This is how ACT can model:

- unfused implementations
- partially fused implementations
- fully fused accelerator blocks

without changing TAIDL semantics.

### 5. IP names are mapped to abstraction classes

The user-facing IP name is not charged directly.
Instead, it points to an abstraction class in `primitive_hw_config.json`.

Examples:

- `systolic_array_ip -> tensor_compute`
- `vector_div_ip -> vector_compute_div`
- `softmax_ip -> special_math`

This keeps the interface readable for hardware designers while preserving a stable estimator backend.

## Why this layer exists

Without this interface, ACT would have to assume:

- every instruction maps directly to one fixed hardware class

That breaks down as soon as:

- one instruction decomposes into several primitives
- hardware fuses some but not all primitives
- different accelerators realize the same semantics differently

This interface solves that problem cleanly by separating:

1. semantic meaning
2. realization choice
3. calibrated hardware cost

## Recommended workflow

1. Define TAIDL semantics.
2. Decompose each instruction into semantic primitives.
3. Edit the mapping interface:
   - fallback primitive mappings
   - fused pattern overrides
   - IP-to-abstraction-class links
4. Calibrate the abstraction classes in `primitive_hw_config.json`.
5. Run ACT cost estimation and plotting.

## Related source files

- [hardware_mapping_interface_spec.md](/scratch/krish/MLIR-hardware-analysis/submodule/act/dse/docs/hardware_mapping_interface_spec.md)
- [hardware_mapping_interface_example.json](/scratch/krish/MLIR-hardware-analysis/submodule/act/dse/docs/hardware_mapping_interface_example.json)
- [primitive_hw_config.json](/scratch/krish/MLIR-hardware-analysis/submodule/act/dse/config/primitive_hw_config.json)

## Short takeaway

The interface package exists so ACT can say:

> "This semantic instruction decomposes into these primitives, these primitives are realized by these hardware IPs, and those IPs charge these calibrated abstraction-class costs."

That is the whole contract in one sentence.
