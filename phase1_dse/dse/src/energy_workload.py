"""CLI: estimate static energy (pJ) for Gemmini-style .pii workloads."""

import argparse
import csv
import json
from pathlib import Path

from .energy_estimate import estimate_program, load_mapping_meta
from .parse_pii import parse_pii_dir
from .plot import plot_energy_by_class


def _write_per_candidate_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = ["candidate", "kernel_name", "total_energy_pj", "by_cost_tag_pj", "by_realization_pj"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "candidate": r["candidate"],
                    "kernel_name": r["kernel_name"],
                    "total_energy_pj": r["total_energy_pj"],
                    "by_cost_tag_pj": json.dumps(r["by_cost_tag_pj"]),
                    "by_realization_pj": json.dumps(r["by_realization_pj"]),
                }
            )


def run_energy_workload(input_path, hw_config_path, out_dir, mapping_json_path, with_plot):
    programs = parse_pii_dir(input_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = load_mapping_meta(mapping_json_path)
    summaries = []
    detail_paths = []

    for prog in programs:
        est = estimate_program(prog, hw_config_path, mapping_json_path)
        est["mapping_interface"] = est.get("mapping_interface") or meta
        summaries.append(
            {
                "candidate": est["candidate"],
                "kernel_name": est["kernel_name"],
                "total_energy_pj": est["total_energy_pj"],
                "by_cost_tag_pj": est["by_cost_tag_pj"],
                "by_realization_pj": est["by_realization_pj"],
            }
        )

        stem = Path(est["candidate"]).stem
        detail = out / "energy_detail_{}.json".format(stem)
        detail.write_text(json.dumps(est, indent=2))
        detail_paths.append(str(detail))

        if with_plot:
            plot_path = out / "plots" / "energy_by_cost_tag_{}.png".format(stem)
            plot_energy_by_class(
                est["by_cost_tag_pj"],
                plot_path,
                title="Energy (pJ) by cost tag — {}".format(stem),
            )

    summary = {
        "status": "ok",
        "num_candidates": len(programs),
        "hw_config": str(hw_config_path),
        "mapping_interface": meta,
        "candidates": summaries,
        "detail_json": detail_paths,
        "csv": str(out / "candidate_energy.csv"),
    }
    _write_per_candidate_csv(summaries, out / "candidate_energy.csv")
    (out / "energy_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Static energy (pJ) from .pii + realization mapping + primitive hw cost tags"
    )
    parser.add_argument("--input", required=True, help=".pii file or directory of candidates")
    parser.add_argument(
        "--hw_config",
        required=True,
        help="primitive hw config (`cost_tags` preferred; legacy `abstraction_classes` still supported)",
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--mapping_json",
        default=None,
        help="Optional final_mapping.json (used for realization selection)",
    )
    parser.add_argument("--plot", action="store_true", help="Write plots/energy_by_cost_tag_*.png")

    args = parser.parse_args()
    summary = run_energy_workload(
        args.input,
        args.hw_config,
        args.out,
        args.mapping_json,
        args.plot,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
