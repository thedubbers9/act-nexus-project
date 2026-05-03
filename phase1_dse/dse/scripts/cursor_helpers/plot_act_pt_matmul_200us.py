#!/usr/bin/env python3
"""Plot ACT vs PrimeTime (ground truth) from pt_comparison_bundle + v1 term JSON.

Writes PNG (+ optional PDF) next to the bundle. Requires matplotlib.

  python3 plot_act_pt_matmul_200us.py [--bundle DIR_OR_JSON]
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bundle",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "artifacts"
        / "runs"
        / "20260420_matmul_64x64_chipyard_full_active_matmul_200us"
        / "pt_comparison_bundle.json",
        help="Path to pt_comparison_bundle.json or its parent directory",
    )
    args = ap.parse_args()

    bundle_path = args.bundle
    if bundle_path.is_dir():
        bundle_path = bundle_path / "pt_comparison_bundle.json"
    v1_path = bundle_path.parent / (bundle_path.stem + "_act_pt_v1_term_buckets.json")

    bundle = json.loads(bundle_path.read_text())
    v1 = json.loads(v1_path.read_text()) if v1_path.is_file() else None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required: python3 -m pip install --user matplotlib", file=sys.stderr)
        sys.exit(1)

    pt_total = float(bundle["window"]["energy_uj"])
    act_total = float(bundle["act"]["total_energy_uj"])
    pct = float(bundle["act"]["pct_error_vs_pt"])
    wl = bundle.get("workload_label", "")
    dut = bundle.get("dut_scope", "")
    cap_us = float(bundle["window"]["capture_ns"]) / 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left: total window energy ---
    ax = axes[0]
    names = ["PrimeTime\n(ground truth)", "ACT\n(scaled window)"]
    vals = [pt_total, act_total]
    colors = ["#2ca02c", "#ff7f0e"]
    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Energy (µJ)")
    ax.set_title("Total Gemmini window energy ({} µs)".format(cap_us))
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, "{:.2f}".format(v), ha="center", va="bottom", fontsize=10)
    ax.text(
        0.5,
        0.02,
        "Δ ≈ {:+.1f}% vs PT".format(pct),
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )

    # --- Right: per-bucket (PT vs ACT v1) ---
    ax = axes[1]
    if v1:
        order = ["mesh", "scratchpad", "controller_dma", "glue"]
        pt_b = v1["pt_window_bucket_uj"]
        act_b = v1["act_window_bucket_uj"]
        pt_vals = [float(pt_b.get(k) or 0) for k in order]
        act_vals = [float(act_b.get(k) or 0) for k in order]
        labels = [k.replace("_", "\n") for k in order]
        w = 0.36
        xi = np.arange(len(order))
        ax.bar(xi - w / 2, pt_vals, w, label="PrimeTime", color="#2ca02c", edgecolor="black", linewidth=0.5)
        ax.bar(xi + w / 2, act_vals, w, label="ACT (v1 term map)", color="#ff7f0e", edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Energy (µJ)")
        ax.set_title("Per-bucket energy (same {} µs window)".format(cap_us))
        ax.set_xticks(xi)
        ax.set_xticklabels(labels, fontsize=8)
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Missing {}\n(run act_pt_term_bucket_v1.py)".format(v1_path.name), ha="center", va="center")
        ax.axis("off")

    fig.suptitle(
        "matmul_64x64 — ACT vs PrimeTime ground truth\n{} / {}".format(wl, dut),
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()

    out_png = bundle_path.parent / "act_pt_comparison_matmul_200us.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    print("Wrote {}".format(out_png))
    plt.close()


if __name__ == "__main__":
    main()
