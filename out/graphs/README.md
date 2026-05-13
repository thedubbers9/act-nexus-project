# ACT Presentation Graphs

This folder keeps only the presentation-facing graph artifacts and the minimal
CSV files needed to inspect the plotted values.

## Graphs

- `candidate_energy_profiles_gemm64.{png,pdf}`
  GEMM 64x64 ACT candidate energy groups.
- `pt_vs_act_energy_breakdown.{png,pdf}`
  PrimeTime dynamic energy vs ACT estimated energy for GEMM, MAC, and ADD.

## CSVs

- `candidate_energy_profiles_gemm64_groups.csv`
- `candidate_energy_profiles_gemm64_summary.csv`
- `pt_vs_act_energy_breakdown_summary.csv`

The graph-generation scripts live in:

- `scripts/make_candidate_energy_profiles_gemm64.py`
- `scripts/make_pt_vs_act_energy_breakdown.py`

Both scripts use paths relative to the ACT checkout and read preserved source
CSVs from `out/graphs/source_data`.
