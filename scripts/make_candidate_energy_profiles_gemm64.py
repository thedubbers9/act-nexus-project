#!/usr/bin/env python3
"""Generate the GEMM 64x64 candidate-energy graph.

Run from anywhere inside or outside the ACT checkout:

    python scripts/make_candidate_energy_profiles_gemm64.py

The script is intentionally ACT-local. It reads the preserved source CSVs in
`out/graphs/source_data` and writes only presentation-facing graph artifacts to
`out/graphs`.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ACT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ACT_ROOT / "out" / "graphs"
SOURCE_DIR = OUT_DIR / "source_data"
GROUPS_CSV = SOURCE_DIR / "candidate_energy_profiles_gemm64_groups.csv"
SUMMARY_CSV = SOURCE_DIR / "candidate_energy_profiles_gemm64_summary.csv"


def _float(value, default=0.0):
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _load_groups():
    groups = []
    with GROUPS_CSV.open() as f:
        for row in csv.DictReader(f):
            groups.append(
                {
                    "group": int(row["group"]),
                    "energy_uJ": _float(row["energy_uJ"]),
                    "num_candidates": int(row["num_candidates"]),
                    "candidate_ids": row["candidate_ids"],
                }
            )
    return groups


def _plot(groups):
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    xs = list(range(len(groups)))
    energies = [group["energy_uJ"] for group in groups]

    ax.bar(
        xs,
        energies,
        color="#4f83a1",
        edgecolor="#1f2933",
        linewidth=0.8,
        width=0.76,
    )

    y_min = 11.0
    y_max = 15.0
    label_pad = (y_max - y_min) * 0.025
    for x, energy in zip(xs, energies):
        if energy > y_max:
            label = f"{energy:.1f}^"
            y = y_max - label_pad
            va = "top"
        elif energy < y_min:
            label = f"{energy:.1f}v"
            y = y_min + label_pad
            va = "bottom"
        else:
            label = f"{energy:.1f}"
            y = energy + label_pad
            va = "bottom"
        ax.text(x, y, label, ha="center", va=va, fontsize=9.2)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"G{group['group']}" for group in groups], fontsize=9)
    ax.set_ylabel("ACT estimated energy (uJ)", fontsize=12)
    ax.set_xlabel("Unique PII candidate energy groups", fontsize=11)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)

    fig.text(
        0.5,
        0.965,
        "GEMM 64x64 Candidate Energy Profiles",
        ha="center",
        va="top",
        fontsize=15.5,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.19)

    png = OUT_DIR / "candidate_energy_profiles_gemm64.png"
    pdf = OUT_DIR / "candidate_energy_profiles_gemm64.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _copy_csvs():
    (OUT_DIR / "candidate_energy_profiles_gemm64_groups.csv").write_text(GROUPS_CSV.read_text())
    if SUMMARY_CSV.exists():
        (OUT_DIR / "candidate_energy_profiles_gemm64_summary.csv").write_text(SUMMARY_CSV.read_text())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = _load_groups()
    png, pdf = _plot(groups)
    _copy_csvs()
    print(f"Wrote {png.relative_to(ACT_ROOT)}")
    print(f"Wrote {pdf.relative_to(ACT_ROOT)}")
    print(f"Wrote {(OUT_DIR / 'candidate_energy_profiles_gemm64_groups.csv').relative_to(ACT_ROOT)}")
    if SUMMARY_CSV.exists():
        print(f"Wrote {(OUT_DIR / 'candidate_energy_profiles_gemm64_summary.csv').relative_to(ACT_ROOT)}")


if __name__ == "__main__":
    main()
