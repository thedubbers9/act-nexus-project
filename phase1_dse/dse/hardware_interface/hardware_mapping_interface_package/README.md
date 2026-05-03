# Final Mapping — Realization-First Hardware Interface

## What this is

A concrete mapping from every TAIDL semantic primitive to physical hardware
blocks on Gemmini (and generalizable to other accelerators).

The key idea: **primitive → realization → IP flow → cost tag**.

A single primitive (e.g. `broadcast`) can have multiple realizations
(`virtual_broadcast`, `materialized_broadcast`, `fused_broadcast`), each with
its own ordered IP pipeline and cost accounting.  Fused patterns override
per-op fallback when a recognized multi-primitive sequence is detected.

## Files

| File | Purpose |
|------|---------|
| `primitive_realizations.csv` | Each TAIDL primitive and its possible realizations, with selection condition and Gemmini rationale |
| `realization_ip_flow.csv` | Ordered IP pipeline steps for each realization |
| `realization_cost_tags.csv` | Accounting tag and formula hint for each realization |
| `fused_patterns.json` | Fusion overrides that replace multiple primitives with one combined realization |
| `generate_final_mapping.py` | Generates `final_mapping.xlsx` and `final_mapping.json` from the CSVs+JSON |
| `final_mapping.xlsx` | Human-readable multi-sheet workbook (generated) |
| `final_mapping.json` | Machine-readable interface (generated) |

## Schema

### primitive_realizations.csv

```
primitive           — TAIDL primitive name
realization_id      — unique realization name
is_default          — 1 if this is the default choice for the primitive
condition           — when to select this realization (human-readable predicate)
gemmini_rationale   — why this mapping exists on Gemmini hardware
```

### realization_ip_flow.csv

```
realization_id — links back to primitive_realizations
step           — ordered step number (1, 2, 3, ...)
ip_block       — Gemmini IP block name (e.g. systolic_mesh, dma_engine, acc_adder)
direction      — read / write / compute / control / absorbed / metadata
notes          — clarification
```

### realization_cost_tags.csv

```
realization_id    — links back
cost_tag          — stable accounting bucket (tensor_compute, onchip_movement, etc.)
cost_formula_hint — parametric formula skeleton
notes             — clarification
```

### fused_patterns.json

```json
{
  "fused_patterns": [
    {
      "name":            "softmax_fused",
      "match":           ["exponential", "reduce_add", "broadcast", "divide"],
      "realization_id":  "fused_softmax",
      "ip_flow":         [{"step": 1, "ip_block": "...", "direction": "..."}],
      "cost_tag":        "special_function",
      "priority":        100,
      "notes":           "..."
    }
  ]
}
```

Patterns are tried highest-priority first.  Any primitive consumed by a fused
pattern is NOT also looked up in per-op fallback.

## Pipeline (how to use it)

```
TAIDL instruction semantics
    ↓
primitive decomposition (e.g. softmax → exp + reduce_add + broadcast + divide)
    ↓
normalization (canonical op names that match CSV keys)
    ↓
fusion check: try fused_patterns highest-priority-first
    ↓
per-op fallback: for unmatched primitives, pick realization from
                 primitive_realizations (default unless condition selects alt)
    ↓
IP flow assignment: look up ordered IP steps for chosen realization
    ↓
cost aggregation: use cost_tag × hardware params
```

## Decision rules for ambiguous primitives

### broadcast
- **virtual_broadcast** (default): single consumer, immediately consumed, no materialization
- **materialized_broadcast**: result must physically exist as a dense expanded tensor
- **fused_broadcast**: absorbed inside a fused_pattern (e.g. softmax)

### reshape
- **logical_view** (default): metadata-only, compatible strides, cost = 0
- **physical_relayout**: lowering requires actual byte reorder

### transpose
- **explicit_relayout** (default): standalone transpose is almost always physical
- **folded_into_matmul**: absorbed by following dot via config_ex transpose bits

### convert
- **standalone_rocket** (default): true dtype cast on CPU
- **mvin_scale**: scale during DRAM→SPAD load
- **acc_scale_read**: scale on ACC read-down

## Regenerating outputs

```bash
# From this directory (…/hardware_mapping_interface_package/):
python3 generate_final_mapping.py
```

Requires `openpyxl` (`python3 -m pip install --user openpyxl`).
