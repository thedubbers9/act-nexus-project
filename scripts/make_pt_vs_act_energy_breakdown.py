#!/usr/bin/env python3
"""Generate the PrimeTime-vs-ACT stacked energy graph.

Run from anywhere inside or outside the ACT checkout:

    python scripts/make_pt_vs_act_energy_breakdown.py

The graph is absolute energy in uJ. It is not normalized and not zoomed.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ACT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ACT_ROOT / "out" / "graphs"
SOURCE_CSV = OUT_DIR / "source_data" / "pt_vs_act_energy_breakdown_source.csv"

CATEGORIES = ["MAC Array", "ACC", "Scratchpad", "Control Logic"]
COLORS = {
    "MAC Array": "#2a9d8f",
    "ACC": "#e76f51",
    "Scratchpad": "#4f83a1",
    "Control Logic": "#8d99ae",
}


def _float(value, default=0.0):
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _load_rows():
    rows = []
    with SOURCE_CSV.open() as f:
        for row in csv.DictReader(f):
            out = dict(row)
            for key in (*CATEGORIES, "total"):
                out[key] = _float(out[key])
            rows.append(out)
    return rows


def _plot(rows):
    fig, ax = plt.subplots(figsize=(14.4, 5.8))

    workloads = []
    for row in rows:
        if row["workload"] not in [item[0] for item in workloads]:
            workloads.append((row["workload"], row["display"], row["formula"]))

    x_positions = []
    x_labels = []
    group_centers = []
    group_labels = []
    formula_labels = []

    start = 0.0
    bar_width = 0.78
    intra_gap = 0.92
    group_gap = 1.65

    for _workload, display, formula in workloads:
        pt_x = start
        act_x = start + intra_gap
        x_positions.extend([pt_x, act_x])
        x_labels.extend(["PT", "ACT"])
        group_centers.append((pt_x + act_x) / 2.0)
        group_labels.append(display)
        formula_labels.append(formula)
        start = act_x + group_gap

    bottoms = [0.0] * len(rows)
    legend_handles = []
    legend_labels = []
    for category in CATEGORIES:
        values = [row[category] for row in rows]
        bars = ax.bar(
            x_positions,
            values,
            width=bar_width,
            bottom=bottoms,
            color=COLORS[category],
            edgecolor="white",
            linewidth=0.6,
            label=category,
        )
        legend_handles.append(bars[0])
        legend_labels.append(category)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    y_top = max(row["total"] for row in rows) * 1.10
    label_pad = y_top * 0.025
    for x, row in zip(x_positions, rows):
        ax.text(x, row["total"] + label_pad, f"{row['total']:.1f}", ha="center", va="bottom", fontsize=9.5)

    fig.text(0.5, 0.965, "PrimeTime vs ACT Energy Breakdown", ha="center", va="top", fontsize=15.5)
    ax.set_ylabel("Energy (uJ)", fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=10.5)
    ax.set_ylim(0.0, y_top)
    ax.grid(axis="y", linestyle=":", linewidth=0.9, alpha=0.45)

    for center, label, formula in zip(group_centers, group_labels, formula_labels):
        ax.text(
            center,
            -0.115,
            f"{label}\n{formula}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9.6,
        )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=4,
        frameon=False,
        fontsize=9.8,
        handlelength=1.8,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.78, bottom=0.205)

    png = OUT_DIR / "pt_vs_act_energy_breakdown.png"
    pdf = OUT_DIR / "pt_vs_act_energy_breakdown.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _write_summary(rows):
    summary = OUT_DIR / "pt_vs_act_energy_breakdown_summary.csv"
    fields = ["workload", "display", "formula", "bar_label", *CATEGORIES, "total"]
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    png, pdf = _plot(rows)
    summary = _write_summary(rows)
    print(f"Wrote {png.relative_to(ACT_ROOT)}")
    print(f"Wrote {pdf.relative_to(ACT_ROOT)}")
    print(f"Wrote {summary.relative_to(ACT_ROOT)}")


if __name__ == "__main__":
    main()
