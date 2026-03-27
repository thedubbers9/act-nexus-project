#!/usr/bin/env python3
"""Plot total and per-instruction ISA energy for compiled ACT workloads."""

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_cost_module(backend_dir: Path):
    model_path = backend_dir / "python" / "cost" / "model.py"
    spec = importlib.util.spec_from_file_location("act_backend_cost_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load cost model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summaries(cost_module, compiled_paths):
    rows = []
    instruction_names = set()
    for path in compiled_paths:
        summary = cost_module.summarize_cost(str(path))
        row = {
            "label": path.stem,
            "path": str(path),
            "total_energy_pj": float(summary["total_energy_pj"]),
            "used_fallback": bool(summary["used_fallback"]),
            "instruction_breakdown": summary["instruction_breakdown"],
        }
        instruction_names.update(summary["instruction_breakdown"].keys())
        rows.append(row)
    return rows, sorted(instruction_names)


def _natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.stem)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def _write_csv(rows, instruction_names, out_csv: Path):
    fieldnames = ["label", "path", "total_energy_pj", "used_fallback"] + [
        f"{name}_energy_pj" for name in instruction_names
    ] + [f"{name}_calls" for name in instruction_names]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {
                "label": row["label"],
                "path": row["path"],
                "total_energy_pj": row["total_energy_pj"],
                "used_fallback": row["used_fallback"],
            }
            for name in instruction_names:
                record = row["instruction_breakdown"].get(name, {})
                flat[f"{name}_energy_pj"] = record.get("total_energy_pj", 0.0)
                flat[f"{name}_calls"] = record.get("calls", 0)
            writer.writerow(flat)


def _plot_total(rows, out_path: Path):
    labels = [row["label"] for row in rows]
    totals = [row["total_energy_pj"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, totals, color="#2b6cb0")
    ax.set_ylabel("Total ISA energy (pJ)")
    ax.set_title("Total compiled ISA energy by workload")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for bar, total in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{total:.0f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_stacked(rows, instruction_names, out_path: Path):
    labels = [row["label"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = [0.0] * len(rows)
    for name in instruction_names:
        values = [
            row["instruction_breakdown"].get(name, {}).get("total_energy_pj", 0.0)
            for row in rows
        ]
        ax.bar(labels, values, bottom=bottom, label=name)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_ylabel("Energy (pJ)")
    ax.set_title("ISA energy breakdown by instruction")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_outputs(rows, instruction_names, out_dir: Path, prefix: str, total_title: str, stacked_title: str):
    json_path = out_dir / f"{prefix}_cost_summary.json"
    csv_path = out_dir / f"{prefix}_cost_summary.csv"
    total_path = out_dir / f"{prefix}_total_isa_energy.png"
    normalized_total_path = out_dir / f"{prefix}_total_isa_energy_delta_from_best_pct.png"
    zoomed_total_path = out_dir / f"{prefix}_total_isa_energy_zoomed.png"
    stacked_path = out_dir / f"{prefix}_instruction_energy_breakdown.png"
    best_ops_raw_path = out_dir / f"{prefix}_best_candidate_op_energy.png"
    best_ops_norm_path = out_dir / f"{prefix}_best_candidate_op_energy_normalized_pct.png"

    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    _write_csv(rows, instruction_names, csv_path)

    labels = [row["label"] for row in rows]
    totals = [row["total_energy_pj"] for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 4.5))
    bars = ax.bar(labels, totals, color="#2b6cb0")
    ax.set_ylabel("Total ISA energy (pJ)")
    ax.set_title(total_title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for bar, total in zip(bars, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{total:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90 if len(labels) > 12 else 0,
        )
    if len(labels) > 10:
        ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(total_path, dpi=180)
    plt.close(fig)

    best_total = min(totals) if totals else 1.0
    normalized_totals = [
        ((total - best_total) / best_total) * 100.0 if best_total else 0.0
        for total in totals
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 4.5))
    bars = ax.bar(labels, normalized_totals, color="#dd6b20")
    ax.set_ylabel("Delta from best candidate (%)")
    ax.set_title(f"{total_title} (Best Candidate = 0%)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for bar, value in zip(bars, normalized_totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90 if len(labels) > 12 else 0,
        )
    if len(labels) > 10:
        ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(normalized_total_path, dpi=180)
    plt.close(fig)

    energy_span = max(totals) - best_total if totals else 0.0
    pad = max(energy_span * 0.15, best_total * 0.002 if best_total else 1.0)
    ymin = max(0.0, best_total - pad)
    ymax = max(totals) + pad if totals else 1.0
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 4.5))
    bars = ax.bar(labels, totals, color="#2f855a")
    ax.set_ylabel("Total ISA energy (pJ)")
    ax.set_title(f"{total_title} (Zoomed Around Best Candidate)")
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for bar, total in zip(bars, totals):
        delta = total - best_total
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            total,
            f"+{delta:.0f}" if delta > 0 else "best",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90 if len(labels) > 12 else 0,
        )
    if len(labels) > 10:
        ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(zoomed_total_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.65), 5))
    bottom = [0.0] * len(rows)
    for name in instruction_names:
        values = [
            row["instruction_breakdown"].get(name, {}).get("total_energy_pj", 0.0)
            for row in rows
        ]
        ax.bar(labels, values, bottom=bottom, label=name)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_ylabel("Energy (pJ)")
    ax.set_title(stacked_title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    if len(labels) > 10:
        ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(stacked_path, dpi=180)
    plt.close(fig)

    best_row = min(rows, key=lambda row: row["total_energy_pj"])
    best_breakdown = best_row["instruction_breakdown"]
    best_items = sorted(
        (
            (
                name,
                record.get("total_energy_pj", 0.0),
                record.get("calls", 0),
            )
            for name, record in best_breakdown.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if best_items:
        op_labels = [item[0] for item in best_items]
        op_values = [item[1] for item in best_items]
        op_calls = [item[2] for item in best_items]

        fig, ax = plt.subplots(figsize=(max(8, len(op_labels) * 1.1), 4.8))
        bars = ax.bar(op_labels, op_values, color="#3182ce")
        ax.set_ylabel("Energy (pJ)")
        ax.set_title(f"Best Candidate Op Energy: {best_row['label']}")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for bar, value, calls in zip(bars, op_values, op_calls):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.0f}\n({calls}x)",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        if len(op_labels) > 6:
            ax.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        fig.savefig(best_ops_raw_path, dpi=180)
        plt.close(fig)

        total_best = best_row["total_energy_pj"] or 1.0
        op_pct = [(value / total_best) * 100.0 for value in op_values]
        fig, ax = plt.subplots(figsize=(max(8, len(op_labels) * 1.1), 4.8))
        bars = ax.bar(op_labels, op_pct, color="#d69e2e")
        ax.set_ylabel("Percent of best candidate total energy (%)")
        ax.set_title(f"Best Candidate Op Energy Share: {best_row['label']}")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for bar, value, raw_value in zip(bars, op_pct, op_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        if len(op_labels) > 6:
            ax.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        fig.savefig(best_ops_norm_path, dpi=180)
        plt.close(fig)

    print(f"Wrote summary JSON: {json_path}")
    print(f"Wrote summary CSV:  {csv_path}")
    print(f"Wrote total plot:   {total_path}")
    print(f"Wrote norm plot:    {normalized_total_path}")
    print(f"Wrote zoom plot:    {zoomed_total_path}")
    print(f"Wrote stacked plot: {stacked_path}")
    if best_items:
        print(f"Wrote best-op plot: {best_ops_raw_path}")
        print(f"Wrote best-op pct:  {best_ops_norm_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-dir", required=True, help="Path to targets/<ISA>/backend")
    parser.add_argument("--compiled-dir", help="Directory with compiled workload .py files")
    parser.add_argument("--candidate-dir", help="Directory with candidate .py files, e.g. log/<run>/demo_000")
    parser.add_argument("--out-dir", required=True, help="Directory for CSV/JSON/PNG outputs")
    args = parser.parse_args()

    backend_dir = Path(args.backend_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.compiled_dir and not args.candidate_dir:
        raise SystemExit("Provide at least one of --compiled-dir or --candidate-dir")

    cost_module = _load_cost_module(backend_dir)

    if args.compiled_dir:
        compiled_dir = Path(args.compiled_dir).resolve()
        compiled_paths = sorted(
            (p for p in compiled_dir.glob("*.py") if p.is_file()),
            key=_natural_sort_key,
        )
        if not compiled_paths:
            raise SystemExit(f"No compiled workload .py files found in {compiled_dir}")
        rows, instruction_names = _summaries(cost_module, compiled_paths)
        _write_outputs(
            rows,
            instruction_names,
            out_dir,
            prefix="workload",
            total_title="Total compiled ISA energy by workload",
            stacked_title="ISA energy breakdown by instruction",
        )

    if args.candidate_dir:
        candidate_dir = Path(args.candidate_dir).resolve()
        candidate_paths = sorted(
            (p for p in candidate_dir.glob("*.py") if p.is_file()),
            key=_natural_sort_key,
        )
        if not candidate_paths:
            raise SystemExit(f"No candidate .py files found in {candidate_dir}")
        rows, instruction_names = _summaries(cost_module, candidate_paths)
        _write_outputs(
            rows,
            instruction_names,
            out_dir,
            prefix="candidate",
            total_title="Total ISA energy by candidate kernel",
            stacked_title="Candidate ISA energy breakdown by instruction",
        )


if __name__ == "__main__":
    main()
