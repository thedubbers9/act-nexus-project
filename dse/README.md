# DSE  (ACT Phase-1, Pre-Schedule)

This guide explains the DSE code in `dse/`.

It covers:
- What this DSE pass is doing
- Where it hooks into the ACT compiler
- What the inputs mean
- What each main function does
- A full worked example using `log/mem_bound_compile/0.pii`
- How to read final outputs

---

## 1) What is this DSE pass?

`DSE` means **Design Space Exploration**.

In this project, the Phase-1 DSE is a **fast analytical bound pass**. It does not try to optimize schedules or pick hardware. Instead, it answers:

- Given a program candidate, a bandwidth, and a compute peak:
  - What is the **best-case (lower-bound) latency**?
  - Is the bottleneck memory or compute?
- Given a target latency:
  - What **minimum bandwidth** is required?

Important:
- This pass runs on **pre-schedule `.pii` candidates**.
- It is intended as a fast feasibility check, not a full cycle-accurate model.

---

## 2) Where this connects to ACT compiler

Hook location in generated backend:
- `targets/QKV/backend/src/pipeline.rs`

The integration is:
1. Compiler extracts a candidate graph.
2. Compiler saves candidate as `<id>.pii`.
3. Before schedule/allocation bridge, compiler calls DSE.
4. DSE writes analysis files; normal compilation continues.

Relevant code points:
- CLI flags parse: `targets/QKV/backend/src/pipeline.rs:343`
- DSE launcher: `targets/QKV/backend/src/pipeline.rs:204`
- Hook call after `pii.save(...)`: `targets/QKV/backend/src/pipeline.rs:450`
- Hook call before `cpp_bridge(...)`: `targets/QKV/backend/src/pipeline.rs:464`

The compiler launches:

```bash
python3 -m dse.forward_bound \
  --input <path_to_pii> \
  --input_mode pii \
  --hw_config dse/config.yaml \
  --out <dse_output_dir> \
  [--targets ...]
```

---

## 3) Inputs to DSE and what they mean

## Input A: `.pii` candidate file
Example:
- `log/mem_bound_compile/0.pii`

This is a text form of one extracted program candidate.
Each line is one IR node/op.

Example line:

```text
t1: D1[-1] = bf16[64,64] load_cm[rows='64'](t0)
```

Read it as:
- `t1`: node id
- `D1[-1]`: destination buffer and offset
- `bf16[64,64]`: output type and shape
- `load_cm`: operation
- `[rows='64']`: attributes
- `(t0)`: dependency (input node)

## Input B: hardware/sweep config
File:
- `dse/config.yaml`

Provides:
- `hardware.peak_compute_ops_per_s`
- bandwidth sweep range (`start`, `stop`, `num`, `scale`)

## Input C: target latency list (optional)
File:
- `dse/examples/targets.yaml`

Provides:
- `targets.latencies_s`

Used for inverse query:
- "To hit latency `T*`, what bandwidth is minimally required?"

---

## 4) End-to-end flow chart

```text
ACT compiler (pipeline.rs)
    |
    |  saves candidate as .pii
    v
python -m dse.forward_bound
    |
    +--> parse_pii.py
    |      parse text .pii into structured instructions
    |
    +--> features.py
    |      convert instructions -> bytes/ops features
    |
    +--> model_forward.py
    |      compute latency lower bounds + bottleneck
    |      compute inverse bandwidth bounds
    |
    +--> plot.py (optional)
    |      generate latency-vs-bandwidth chart
    |
    v
CSV/JSON outputs per candidate
```

---
## Deep dive: `attrs`, `bytes_per_elem`, and feature math logic

This section explains the two most important concepts in feature extraction:
- what instruction `attrs` are
- why `bytes_per_elem` exists and how formulas use it

### What are `attrs`?

In a `.pii` line, ops can carry attributes inside brackets:

```text
load_cm[rows='64']
```

Parser behavior (`parse_pii.py`):
- Extracts op name: `load_cm`
- Extracts key/value attributes: `rows='64'`
- Converts numeric text to number type (`"64"` -> `64`)
- Normalizes `rows` to `n` for convenience:
  - if `rows` exists and `n` not present, set `n = rows`

So later in `features.py`, most formulas read:
- `n` (row count)
- `_shape` (shape list like `[64,64]`)
- `_buffer`, `_dtype` (context fields added by parser)

### What is `bytes_per_elem`?

In `extract_features(...)`, you will see:

```python
bytes_per_elem = 2
```

Reason:
- QKV data path is modeled around `bf16` tensor tiles.
- `bf16` is 16 bits = 2 bytes.
- For Phase-1 MVP, bytes are estimated with this fixed element size.

Important assumption:
- This is a modeling simplification.
- It is used for consistency/trend analysis, not exact byte-accurate protocol modeling for every dtype conversion.

### Why formulas look like `n * width * bytes_per_elem`

For data movement ops (load/store), the model estimates:
- number of elements moved
- multiplied by bytes per element

For a tile with `n` rows and `width` columns:
- elements moved = `n * width`
- bytes moved = `n * width * bytes_per_elem`

That is why load/store formulas use this structure.

### Per-op logic and reasoning

`_instruction_contrib(ins, width, bytes_per_elem)` applies one rule by op type:

1. `load_rm`, `load_cm`
- Interpreted as HBM -> on-chip movement
- Adds HBM read bytes
- Adds local bytes (same amount) as local traffic proxy
- Formula:
  - `hbm_read = n * w * bytes_per_elem`

2. `store_rm`, `store_cm`
- Interpreted as on-chip -> HBM movement
- Adds HBM write bytes
- Adds local bytes
- Formula:
  - `hbm_write = n * w * bytes_per_elem`

3. `gemm`
- Interpreted as compute-dominant op
- Adds compute ops proxy:
  - `ops = 2 * m * k * n`
- Why `2 * m * k * n`:
  - each multiply-accumulate is roughly 2 floating-point ops
- Adds local byte proxy for matrix operands/results

4. `softmax`
- Adds approximate compute ops:
  - includes exp/reduction/divide style work proxy
- Adds local traffic proxy

5. `mov`
- No HBM movement counted
- Adds local movement proxy only

6. Allowed-but-ignored ops (`Var`, `transpose`, etc.)
- Marked as known so parser doesn’t flag them unknown
- Contribute zero in current MVP accounting

7. Unknown ops
- Marked unknown and listed in output
- Optional fallback: if unknown op writes to `HBM`, estimate write bytes from output shape

### Why there are local bytes and HBM bytes separately

- `hbm_bytes` measures off-chip traffic pressure (usually the main bottleneck in memory-bound cases)
- `local_bytes` tracks on-chip movement proxy

In current model, latency bound uses:
- `hbm_bytes` and `compute_ops`
- `local_bytes` is currently informational (useful for future model upgrades)

### Current simplifications to keep in mind

- Fixed `bytes_per_elem = 2` for MVP.
- Some ops are intentionally ignored or approximated.
- Formulas are designed for fast trend/bound analysis, not cycle-accurate timing.
- This is why results should be read as:
  - "required lower bound / dominant regime"
  - not exact final runtime prediction.

---

## 6) Worked example: `log/mem_bound_compile/0.pii`

Input `.pii`:

```text
t0: HBM[0] = u8[8192] Var['%q']()
t1: D1[-1] = bf16[64,64] load_cm[rows='64'](t0)
t2: HBM[24576] = u8[8192] store_cm[rows='64'](t1)
```

How parser decodes:
- Line 1:
  - op = `Var`, buffer = `HBM`, shape = `[8192]`
- Line 2:
  - op = `load_cm`, attrs includes `n=64`, shape `[64,64]`
- Line 3:
  - op = `store_cm`, attrs includes `n=64`, shape `[8192]` text with store semantics

Feature extraction math (`bytes_per_elem = 2`):

- `load_cm` contribution:
  - `hbm_read = n * width * 2 = 64 * 64 * 2 = 8192`
- `store_cm` contribution:
  - `hbm_write = n * width * 2 = 64 * 64 * 2 = 8192`
- `Var` contributes no bytes/ops in current model.

Why this makes sense for this candidate:
- Program only loads one tile and stores one tile.
- No `gemm` or `softmax` in this candidate.
- So compute proxy is zero and traffic is purely memory movement.

Totals in `candidate_features.csv`:
- `hbm_read_bytes = 8192`
- `hbm_write_bytes = 8192`
- `hbm_bytes = 16384`
- `compute_ops = 0`
- `intensity_ops_per_hbm_byte = 0 / 16384 = 0`

These match:
- `dse_out_mem_bound/pii_0/candidate_features.csv`

Forward model at one bandwidth example:
- If `BW = 1e8 B/s` and `peak_compute = 1e11 ops/s`:
  - `T_mem = 16384 / 1e8 = 1.6384e-4 s`
  - `T_cmp = 0 / 1e11 = 0`
  - `T_lb = max(T_mem, T_cmp) = 1.6384e-4 s`
  - bottleneck = `memory`

This matches first row of:
- `dse_out_mem_bound/pii_0/forward_bounds.csv`

Inverse query example:
- target latency `1e-5 s`
- required bandwidth:
  - `BW_min = hbm_bytes / target = 16384 / 1e-5 = 1.6384e9 B/s`

This matches `summary.json` inverse section.

---

## 7) How to read final outputs

Outputs per candidate directory (for example `dse_out_mem_bound/pii_0/`):

- `candidate_features.csv`
  - Static program features from `.pii`.
  - Best file to check if parser/feature formulas make sense.

- `forward_bounds.csv`
  - One row per `(candidate, bandwidth)` with:
    - `t_mem_lb_s`
    - `t_cmp_lb_s`
    - `t_lb_s`
    - `bottleneck`

- `frontier.csv`
  - For each bandwidth point, the best candidate only.
  - If there is one candidate, this is effectively same trend as `forward_bounds.csv`.

- `summary.json`
  - Top-level report:
    - hardware sweep used
    - bottleneck trend across sweep
    - inverse bandwidth bounds for target latencies
    - file paths to outputs
    - plot status

- `plots/latency_vs_bw.png` (if enabled and matplotlib available)
  - Visual trend of latency lower bound vs bandwidth.

---

## 8) Quick mental model

Think of this pass as:

1. Read candidate instructions (`.pii`).
2. Estimate data movement bytes and compute work.
3. Ask: under assumed hardware, what is unavoidable best-case time?
4. Classify whether memory or compute is the fundamental limiter.

This is why it is useful early in the compiler flow:
- fast
- schedule-agnostic
- helps reason about feasibility and bottlenecks before deep optimization.

---

## 9) Current limitations (important)

- This is a Phase-1 MVP model, not full hardware simulation.
- Op accounting is currently focused on QKV-like patterns.
- Energy/power and full memory-capacity constraints are not modeled yet.
- Some ops are intentionally ignored or approximated.

Use results as **bounds and trends**, not exact cycle predictions.
