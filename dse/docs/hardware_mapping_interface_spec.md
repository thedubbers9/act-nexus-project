# Hardware Mapping Interface Spec

This document defines the interface between:

- TAIDL instruction semantics
- semantic primitive decomposition
- optional fusion / execution mapping
- calibrated hardware/IP cost lookup

The goal is to let ACT answer:

> Given a semantic program, which hardware IPs execute it, and what cost should be charged?

This spec is intentionally practical. It is designed for accelerator designers, not just compiler researchers.

---

## 1. Problem statement

TAIDL semantics tell us what an instruction means.

Example:

- `gemm` means `dot`
- `softmax` may mean `exponential -> reduce_add -> broadcast -> divide`

But hardware often executes those semantics in a different shape:

- `dot` may execute on one systolic MAC array
- `softmax` may execute on one fused softmax IP
- `exponential + reduce_add` may be fused, while `broadcast` and `divide` remain separate

So the ACT cost model needs a user-editable layer between semantics and cost.

---

## 2. Design goals

The interface must:

1. keep TAIDL semantic
2. support unfused primitive-by-primitive mapping
3. support partial fusion
4. support full fusion
5. attach cost to hardware IPs, not only to raw primitives
6. remain understandable to a hardware designer editing a JSON file

---

## 3. Conceptual stack

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

The key point is:

- TAIDL remains semantic
- the mapping interface expresses how a particular accelerator realizes those semantics

---

## 4. User-facing interface

The recommended user-facing file has three sections:

```json
{
  "primitive_to_ip": {},
  "fused_patterns": [],
  "ip_costs": {}
}
```

### 4.1 `primitive_to_ip`

Fallback mapping for individual primitive nodes when no fused pattern matches.

Example:

```json
{
  "primitive_to_ip": {
    "dot": "systolic_array_ip",
    "ewise_add": "vector_add_ip",
    "ewise_mul": "vector_mul_ip",
    "divide": "vector_div_ip",
    "reduce_add": "reduction_tree_ip",
    "exponential": "exp_ip",
    "broadcast": "layout_or_dma_ip"
  }
}
```

### 4.2 `fused_patterns`

Higher-priority pattern overrides.

Each pattern says:

- which primitive sequence or set we are matching
- which hardware IP should replace those individual mappings

Example:

```json
{
  "fused_patterns": [
    {
      "name": "softmax_fused",
      "match": ["exponential", "reduce_add", "broadcast", "divide"],
      "ip": "softmax_ip",
      "priority": 100
    }
  ]
}
```

### 4.3 `ip_costs`

Maps an IP name to the calibrated abstraction class used by the cost model.

Example:

```json
{
  "ip_costs": {
    "systolic_array_ip": {
      "abstraction_class": "tensor_compute"
    },
    "vector_add_ip": {
      "abstraction_class": "vector_compute_add"
    },
    "vector_mul_ip": {
      "abstraction_class": "vector_compute_mul"
    },
    "softmax_ip": {
      "abstraction_class": "special_math"
    }
  }
}
```

---

## 5. Meaning of each layer

### TAIDL semantics

The mathematical meaning of the instruction.

Examples:

- `dot`
- `exponential`
- `reduce_add`
- `divide`

### Primitive fallback mapping

The hardware assignment used when no special fusion is declared.

Examples:

- `dot -> systolic_array_ip`
- `divide -> vector_div_ip`

### Fused patterns

Declared hardware shortcuts that replace multiple semantic primitives with one realized IP assignment.

Examples:

- `exponential + reduce_add + broadcast + divide -> softmax_ip`
- `exponential + reduce_add -> exp_reduce_ip`

### IP cost mapping

The bridge from a user-facing IP name to a calibrated hardware abstraction class.

Examples:

- `systolic_array_ip -> tensor_compute`
- `softmax_ip -> special_math`

---

## 6. Matching rules

This is the part that makes the interface deterministic instead of hand-wavy.

### Rule 1: Fused patterns run before fallback mapping

The mapper first tries to match `fused_patterns`.

Any unmatched primitives then use `primitive_to_ip`.

### Rule 2: Higher priority wins

If two patterns overlap, the larger `priority` wins.

### Rule 3: No primitive node may belong to two final matches

Each primitive instance is assigned to exactly one realized hardware mapping:

- one fused IP
- or one fallback IP

### Rule 4: Exact primitive names only

The `match` field refers to canonical semantic primitive names after decomposition.

Examples:

- `dot`
- `exponential`
- `reduce_add`
- `broadcast`
- `divide`

### Rule 5: Validation is required

Before running estimation, the tool should check:

- every referenced IP exists in `ip_costs`
- every `abstraction_class` exists in `primitive_hw_config.json`
- every primitive node is assigned exactly once
- no pattern leaves an ambiguous overlap

---

## 7. Real-world examples

### Example A: GEMM on a systolic array

TAIDL semantics:

```text
dot(%A, %B)
```

Mapping:

```text
dot -> systolic_array_ip -> tensor_compute
```

This is the important point:

- do not force `dot` into standalone `mul` + `add` IPs
- the realized hardware is the systolic MAC array

### Example B: Fully fused softmax

TAIDL semantics:

```text
exponential
reduce_add
broadcast
divide
```

Pattern:

```json
{
  "name": "softmax_fused",
  "match": ["exponential", "reduce_add", "broadcast", "divide"],
  "ip": "softmax_ip",
  "priority": 100
}
```

Realized mapping:

```text
softmax subgraph -> softmax_ip -> special_math
```

Cost behavior:

- use the softmax IP cost
- do not sum four separate primitive costs

### Example C: Partially fused softmax

Suppose the accelerator has:

- fused `exponential + reduce_add`
- standalone `broadcast`
- standalone `divide`

Then:

```json
{
  "primitive_to_ip": {
    "broadcast": "layout_or_dma_ip",
    "divide": "vector_div_ip"
  },
  "fused_patterns": [
    {
      "name": "exp_reduce_fused",
      "match": ["exponential", "reduce_add"],
      "ip": "exp_reduce_ip",
      "priority": 80
    }
  ]
}
```

Realized mapping:

```text
exponential + reduce_add -> exp_reduce_ip
broadcast                -> layout_or_dma_ip
divide                   -> vector_div_ip
```

That is the behavior the interface must make possible.

---

## 8. Recommended schema

This is the recommended schema for the first implementation.

```json
{
  "version": 1,
  "primitive_to_ip": {
    "<primitive_name>": "<ip_name>"
  },
  "fused_patterns": [
    {
      "name": "<pattern_name>",
      "match": ["<primitive_name>", "<primitive_name>"],
      "ip": "<ip_name>",
      "priority": 0
    }
  ],
  "ip_costs": {
    "<ip_name>": {
      "abstraction_class": "<abstraction_class_name>",
      "notes": "<freeform text>"
    }
  }
}
```

### Required fields

- `primitive_to_ip`
- `fused_patterns`
- `ip_costs`
- each `ip_costs.*.abstraction_class`

### Optional fields

- `version`
- `priority`
- `notes`

---

## 9. Recommended implementation plan

### Phase 1: Documentation and example spec

Done by:

- `dse/README.md`
- `dse/docs/hardware_mapping_interface_example.json`

### Phase 2: Executable mapper pass

Add a pass that:

1. reads semantic primitive nodes
2. applies `fused_patterns`
3. applies `primitive_to_ip` fallback
4. emits realized IP assignments
5. looks up `abstraction_class` through `ip_costs`

### Phase 3: Validation

Add a validator that checks:

- unresolved primitives
- unresolved IP names
- unknown abstraction classes
- overlapping pattern matches

### Phase 4: Cost estimation

Use realized IP assignments as the basis for:

- area
- energy
- throughput
- movement

instead of assuming primitive-to-abstraction is always one-to-one.

---

## 10. Project position

This interface is a good fit for ACT because it lets the project be:

- semantic on the compiler side
- hardware-aware on the cost-model side
- flexible for different accelerators

It also creates a much better user story:

> "Describe your instruction semantics in TAIDL.  
> Describe your hardware realization in the mapping file.  
> ACT computes cost using calibrated hardware abstractions."

That is a much stronger project interface than forcing a direct primitive-to-hardware assumption.
