"""Generate meeting-ready plots from DSE CSV outputs.

Outputs:
- latency_vs_bw_frontier.png
- required_bw_vs_target_latency.png (optional, when --workload_dirs is provided)
- required_resources_fixed_sla.png (optional, when --workload_dirs is provided)
- required_bw_fixed_sla.png (optional, when --workload_dirs is provided)
- required_compute_fixed_sla.png (optional, when --workload_dirs is provided)
- feasibility_map_fixed_latency_table.csv (optional, when --workload_dirs is provided)
- feasibility_map_fixed_latency_table.md (optional, when --workload_dirs is provided)
- multi_workload_required_bounds.csv (optional, when --workload_dirs is provided)
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

BYTES_PER_GB = 1.0e9
OPS_PER_TOPS = 1.0e12

WORKLOAD_LABELS = {
    "llm_mem_bound_large_workable": "LLM Memory-Bound Residual Chain",
    "llm_mem_bound_kv_merge_large_workable": "LLM Memory-Bound KV Merge",
    "llm_mixed_attention_large_workable": "LLM Mixed Attention Block",
    "llm_compute_bound_large_workable": "LLM Compute-Bound GEMM Chain",
}

WORKLOAD_DESCRIPTIONS = {
    "LLM Memory-Bound Residual Chain": "Large-token residual/add chain; low arithmetic intensity, memory traffic dominates.",
    "LLM Memory-Bound KV Merge": "KV-cache-style tensor merge with repeated elementwise updates; memory-heavy behavior.",
    "LLM Mixed Attention Block": "Attention-like block with projections + softmax + residual path; mixed memory/compute behavior.",
    "LLM Compute-Bound GEMM Chain": "Large matrix-multiply chain with residual connection; compute-heavy behavior.",
}


def _read_csv(path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(rows, key):
    for r in rows:
        if key in r and r[key] not in (None, ""):
            r[key] = float(r[key])
    return rows


def _apply_style(plt):
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#f8fafc"
    plt.rcParams["axes.edgecolor"] = "#cbd5e1"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["font.size"] = 10


def _apply_log_ticks(ax, x_log=False, y_log=False):
    from matplotlib.ticker import LogLocator, NullFormatter

    if x_log:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if y_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
        ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.2)


def _pretty_workload_name(raw):
    return WORKLOAD_LABELS.get(raw, raw.replace("_", " ").title())


def _format_latency_us(seconds):
    if seconds is None:
        return "N/A"
    return "{:.2f} us".format(float(seconds) * 1.0e6)


def _write_workload_descriptions(workloads, out_path):
    lines = ["# Workload Descriptions", ""]
    for name in sorted(workloads.keys()):
        desc = WORKLOAD_DESCRIPTIONS.get(name, "Description not provided.")
        lines.append("- **{}**: {}".format(name, desc))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def _load_workload_outputs(workload_dirs):
    workloads = {}
    for wd in workload_dirs:
        path = Path(wd)
        if not path.exists():
            continue

        forward_csv = path / "forward_bounds.csv"
        frontier_csv = path / "frontier.csv"
        inverse_csv = path / "inverse_bandwidth_bounds.csv"
        feature_csv = path / "candidate_features.csv"
        if not (forward_csv.exists() and frontier_csv.exists() and inverse_csv.exists() and feature_csv.exists()):
            continue

        forward_rows = _read_csv(forward_csv)
        for key in [
            "hbm_bytes",
            "compute_ops",
            "intensity_ops_per_hbm_byte",
            "bandwidth_bytes_per_s",
            "peak_compute_ops_per_s",
            "t_mem_lb_s",
            "t_cmp_lb_s",
            "t_lb_s",
        ]:
            forward_rows = _as_float(forward_rows, key)

        frontier_rows = _read_csv(frontier_csv)
        for key in [
            "hbm_bytes",
            "compute_ops",
            "intensity_ops_per_hbm_byte",
            "bandwidth_bytes_per_s",
            "peak_compute_ops_per_s",
            "t_mem_lb_s",
            "t_cmp_lb_s",
            "t_lb_s",
        ]:
            frontier_rows = _as_float(frontier_rows, key)

        inverse_rows = _read_csv(inverse_csv)
        for key in [
            "target_latency_s",
            "min_required_bandwidth_bytes_per_s",
            "t_cmp_lb_s",
            "required_bandwidth_bytes_per_s",
        ]:
            inverse_rows = _as_float(inverse_rows, key)

        feature_rows = _read_csv(feature_csv)
        for key in [
            "instruction_count",
            "hbm_read_bytes",
            "hbm_write_bytes",
            "hbm_bytes",
            "local_bytes",
            "compute_ops",
            "intensity_ops_per_hbm_byte",
            "unknown_instruction_count",
        ]:
            feature_rows = _as_float(feature_rows, key)

        # For per-PII dirs (e.g., .../<workload>/pii_1), use parent workload name
        # so multi-workload plots don't collapse all inputs into "pii_1".
        base_label = path.name
        if base_label.startswith("pii_") and path.parent is not None:
            parent_name = path.parent.name
            if parent_name:
                base_label = parent_name
        label = _pretty_workload_name(base_label)
        suffix = 2
        while label in workloads:
            label = "{}_{}".format(base_label, suffix)
            suffix += 1

        workloads[label] = {
            "path": str(path),
            "forward": forward_rows,
            "frontier": frontier_rows,
            "inverse": inverse_rows,
            "features": feature_rows,
        }
    return workloads


def _dedup_inverse_targets(inverse_rows):
    by_target = {}
    for r in inverse_rows:
        target = r.get("target_latency_s")
        if target in (None, ""):
            continue
        t = float(target)
        cur = by_target.get(t)
        if cur is None:
            by_target[t] = r
            continue
        cur_bw = cur.get("min_required_bandwidth_bytes_per_s")
        new_bw = r.get("min_required_bandwidth_bytes_per_s")
        if cur_bw in (None, "") and new_bw not in (None, ""):
            by_target[t] = r
    return by_target


def _build_multi_workload_bounds(workloads):
    rows = []
    for workload, data in sorted(workloads.items()):
        inverse_map = _dedup_inverse_targets(data["inverse"])
        hbm_bytes_vals = [
            float(r["hbm_bytes"])
            for r in data["features"]
            if r.get("hbm_bytes") not in (None, "")
        ]
        min_hbm_bytes = min(hbm_bytes_vals) if hbm_bytes_vals else None
        compute_ops_vals = [
            float(r["compute_ops"])
            for r in data["features"]
            if r.get("compute_ops") not in (None, "")
        ]
        min_compute_ops = min(compute_ops_vals) if compute_ops_vals else None

        for target in sorted(inverse_map.keys()):
            inv = inverse_map[target]
            min_bw = inv.get("min_required_bandwidth_bytes_per_s")
            if min_bw in (None, "") and min_hbm_bytes is not None and target > 0:
                # Fallback lower bound from minimum HBM movement, even if compute-limited.
                min_bw = min_hbm_bytes / target
            status = inv.get("status", "")
            best_candidate = inv.get("best_candidate", "")
            req_compute = None
            if min_compute_ops is not None and target > 0:
                req_compute = min_compute_ops / target
            rows.append(
                {
                    "workload": workload,
                    "target_latency_s": target,
                    "status": status,
                    "best_candidate_for_bw": best_candidate,
                    "min_required_bandwidth_bytes_per_s": min_bw,
                    "min_required_compute_ops_per_s": req_compute,
                    "min_required_bandwidth_gb_per_s": None
                    if min_bw in (None, "")
                    else float(min_bw) / BYTES_PER_GB,
                    "min_required_compute_bf16_tops": None
                    if req_compute in (None, "")
                    else float(req_compute) / OPS_PER_TOPS,
                }
            )
    return rows


def _write_rows_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if not rows_list:
        path.write_text("")
        return
    keys = []
    seen = set()
    for r in rows_list:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows_list:
            w.writerow(r)


def _plot_multi_required_curve(rows, y_key, y_label, title, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    _apply_style(plt)
    grouped = defaultdict(list)
    for r in rows:
        yv = r.get(y_key)
        if yv in (None, ""):
            continue
        xv = float(r["target_latency_s"]) * 1.0e6
        y = float(yv)
        if xv <= 0 or y <= 0 or not math.isfinite(xv) or not math.isfinite(y):
            continue
        grouped[r["workload"]].append((xv, y, r.get("status", "")))

    if not grouped:
        return False

    plt.figure(figsize=(8.6, 5.2))
    for workload, pts in sorted(grouped.items()):
        pts = sorted(pts, key=lambda x: x[0])
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        plt.plot(x, y, marker="o", linewidth=2.0, markersize=4, label=workload)

    ax = plt.gca()
    # For a single-workload view, use a linear Y axis with minor ticks for readability.
    # For multi-workload comparisons, keep log scaling.
    if len(grouped) == 1:
        ax.set_yscale("linear")
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.2)
    else:
        _apply_log_ticks(ax, x_log=False, y_log=True)
    plt.xlabel("Target latency (us)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180)
    plt.close()
    return True


def _plot_fixed_sla_required_resources(rows, out_path, requested_target=None):
    import matplotlib.pyplot as plt

    _apply_style(plt)
    target = _pick_target(rows, requested_target)
    if target is None:
        return False

    selected = []
    for r in rows:
        t = r.get("target_latency_s")
        if t in (None, ""):
            continue
        if abs(float(t) - target) > 1e-18:
            continue
        bw = r.get("min_required_bandwidth_bytes_per_s")
        cmpv = r.get("min_required_compute_ops_per_s")
        if bw in (None, "") or cmpv in (None, ""):
            continue
        bwf = float(bw)
        cmpf = float(cmpv)
        if bwf <= 0 or cmpf <= 0:
            continue
        selected.append((r["workload"], bwf, cmpf))

    if not selected:
        return False

    selected.sort(key=lambda x: x[0])
    workloads = [x[0] for x in selected]
    bw_vals = [x[1] / BYTES_PER_GB for x in selected]
    cmp_vals = [x[2] / OPS_PER_TOPS for x in selected]
    xs = list(range(len(workloads)))

    fig, axes = plt.subplots(2, 1, figsize=(max(8.6, 1.45 * len(workloads)), 6.8), sharex=True)
    axes[0].bar(xs, bw_vals, color="#0284c7", alpha=0.9)
    _apply_log_ticks(axes[0], y_log=True)
    axes[0].set_ylabel("BW_min (GB/s)")
    axes[0].set_title("Fixed-Latency Resource Requirements (target = {})".format(_format_latency_us(target)))

    axes[1].bar(xs, cmp_vals, color="#ea580c", alpha=0.9)
    _apply_log_ticks(axes[1], y_log=True)
    axes[1].set_ylabel("Compute_min (BF16 TOPS)")
    axes[1].set_xlabel("Workload")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(workloads, rotation=20, ha="right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)
    return True


def _plot_fixed_sla_single_metric(rows, out_path, requested_target=None, metric="bw"):
    import matplotlib.pyplot as plt

    _apply_style(plt)
    target = _pick_target(rows, requested_target)
    if target is None:
        return False

    selected = []
    for r in rows:
        t = r.get("target_latency_s")
        if t in (None, ""):
            continue
        if abs(float(t) - target) > 1e-18:
            continue
        bw = r.get("min_required_bandwidth_bytes_per_s")
        cmpv = r.get("min_required_compute_ops_per_s")
        if bw in (None, "") or cmpv in (None, ""):
            continue
        bwf = float(bw)
        cmpf = float(cmpv)
        if bwf <= 0 or cmpf <= 0:
            continue
        selected.append((r["workload"], bwf, cmpf))

    if not selected:
        return False

    selected.sort(key=lambda x: x[0])
    workloads = [x[0] for x in selected]
    xs = list(range(len(workloads)))

    if metric == "compute":
        vals = [x[2] / OPS_PER_TOPS for x in selected]
        ylabel = "Compute_min (BF16 TOPS)"
        color = "#ea580c"
        title = "Fixed-Latency Compute Requirement (target = {})".format(_format_latency_us(target))
    else:
        vals = [x[1] / BYTES_PER_GB for x in selected]
        ylabel = "BW_min (GB/s)"
        color = "#0284c7"
        title = "Fixed-Latency Bandwidth Requirement (target = {})".format(_format_latency_us(target))

    plt.figure(figsize=(max(8.6, 1.45 * len(workloads)), 4.2))
    plt.bar(xs, vals, color=color, alpha=0.9)
    ax = plt.gca()
    _apply_log_ticks(ax, y_log=True)
    plt.ylabel(ylabel)
    plt.xlabel("Workload")
    plt.xticks(xs, workloads, rotation=20, ha="right")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180)
    plt.close()
    return True


def _representative_intensity(frontier_rows):
    if not frontier_rows:
        return None
    counts = defaultdict(int)
    intensity = {}
    for r in frontier_rows:
        cand = r.get("candidate")
        counts[cand] += 1
        if cand not in intensity and r.get("intensity_ops_per_hbm_byte") not in (None, ""):
            intensity[cand] = float(r["intensity_ops_per_hbm_byte"])
    winner = max(sorted(counts.keys()), key=lambda c: counts[c])
    return intensity.get(winner)


def _plot_operational_intensity_by_workload(workloads, out_path):
    import matplotlib.pyplot as plt

    _apply_style(plt)
    names = sorted(workloads.keys())
    if not names:
        return False

    plt.figure(figsize=(max(8.4, 1.6 * len(names)), 5.2))
    xs = list(range(len(names)))

    for i, name in enumerate(names):
        feats = workloads[name]["features"]
        vals = []
        for r in feats:
            v = r.get("intensity_ops_per_hbm_byte")
            if v in (None, ""):
                continue
            vf = float(v)
            if vf <= 0 or not math.isfinite(vf):
                continue
            vals.append(vf)
        if vals:
            plt.scatter([i] * len(vals), vals, color="#94a3b8", alpha=0.6, s=28, label=None)

        rep = _representative_intensity(workloads[name]["frontier"])
        if rep is not None and rep > 0 and math.isfinite(rep):
            plt.scatter([i], [rep], color="#0f172a", s=85, marker="D", label=None)

    plt.yscale("log")
    plt.xticks(xs, names, rotation=20, ha="right")
    plt.ylabel("Operational intensity (ops / HBM byte)")
    plt.xlabel("Workload")
    plt.title("Operational Intensity by Workload")
    # Add lightweight legend proxies for marker meanings.
    plt.scatter([], [], color="#94a3b8", s=28, label="all candidates")
    plt.scatter([], [], color="#0f172a", s=85, marker="D", label="frontier representative")
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180)
    plt.close()
    return True


def _pick_target(rows, requested_target):
    targets = sorted(set(float(r["target_latency_s"]) for r in rows if r.get("target_latency_s") not in (None, "")))
    if not targets:
        return None
    if requested_target is None:
        return targets[len(targets) // 2]
    return min(targets, key=lambda t: abs(t - requested_target))


def _pareto_frontier(points):
    pts = []
    seen = set()
    for bw, cmpv, cand in points:
        key = (bw, cmpv, cand)
        if key not in seen:
            seen.add(key)
            pts.append((bw, cmpv, cand))
    out = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            q_dominates_p = (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1])
            if q_dominates_p:
                dominated = True
                break
        if not dominated:
            out.append(p)
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    return out


def _build_feasibility_points(workloads, target_latency_s):
    rows = []
    if target_latency_s is None or target_latency_s <= 0:
        return rows
    for workload, data in sorted(workloads.items()):
        for r in data["features"]:
            hbm = r.get("hbm_bytes")
            ops = r.get("compute_ops")
            if hbm in (None, "") or ops in (None, ""):
                continue
            bw_req = float(hbm) / target_latency_s
            cmp_req = float(ops) / target_latency_s
            rows.append(
                {
                    "workload": workload,
                    "candidate": r.get("candidate", ""),
                    "target_latency_s": target_latency_s,
                    "required_bandwidth_bytes_per_s": bw_req,
                    "required_compute_ops_per_s": cmp_req,
                }
            )
    return rows


def _plot_feasibility_map_fixed_latency(feas_rows, target_latency_s, out_path):
    import matplotlib.pyplot as plt

    _apply_style(plt)
    if not feas_rows:
        return False

    grouped = defaultdict(list)
    for r in feas_rows:
        bw = r.get("required_bandwidth_bytes_per_s")
        cmpv = r.get("required_compute_ops_per_s")
        if bw in (None, "") or cmpv in (None, ""):
            continue
        bwf = float(bw)
        cmpf = float(cmpv)
        if bwf <= 0 or cmpf <= 0:
            continue
        grouped[r["workload"]].append((bwf, cmpf, r.get("candidate", "")))
    if not grouped:
        return False

    plt.figure(figsize=(8.8, 5.4))
    for workload, pts in sorted(grouped.items()):
        pareto = _pareto_frontier(pts)
        x = [p[0] for p in pareto]
        y = [p[1] for p in pareto]
        plt.plot(x, y, marker="o", linewidth=2.0, markersize=4, label=workload)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Required bandwidth (bytes/s)")
    plt.ylabel("Required peak compute (ops/s)")
    plt.title("Feasibility Map at Fixed Target Latency ({:.3e} s)".format(target_latency_s))
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180)
    plt.close()
    return True


def _write_feasibility_map_table(feas_rows, target_latency_s, out_csv_path):
    grouped = defaultdict(list)
    for r in feas_rows:
        bw = r.get("required_bandwidth_bytes_per_s")
        cmpv = r.get("required_compute_ops_per_s")
        if bw in (None, "") or cmpv in (None, ""):
            continue
        bwf = float(bw)
        cmpf = float(cmpv)
        if bwf <= 0 or cmpf <= 0:
            continue
        grouped[r["workload"]].append((bwf, cmpf, r.get("candidate", "")))

    rows = []
    for workload, pts in sorted(grouped.items()):
        pareto = _pareto_frontier(pts)
        if not pareto:
            continue

        bw_best = min(pareto, key=lambda p: (p[0], p[1], p[2]))
        cmp_best = min(pareto, key=lambda p: (p[1], p[0], p[2]))

        rows.append(
            {
                "workload": workload,
                "target_latency_s": target_latency_s,
                "target_latency_us": target_latency_s * 1.0e6,
                "pareto_points": len(pareto),
                "min_required_bandwidth_bytes_per_s": bw_best[0],
                "min_required_bandwidth_gb_per_s": bw_best[0] / BYTES_PER_GB,
                "candidate_at_min_bw": bw_best[2],
                "min_required_compute_ops_per_s": cmp_best[1],
                "min_required_compute_bf16_tops": cmp_best[1] / OPS_PER_TOPS,
                "candidate_at_min_compute": cmp_best[2],
            }
        )

    _write_rows_csv(out_csv_path, rows)

    md_path = out_csv_path.with_suffix(".md")
    if not rows:
        md_path.write_text("No feasibility points available.\n")
        return False

    lines = []
    lines.append("| workload | target_latency_us | pareto_points | BW_min (GB/s) | candidate_at_min_bw | Compute_min (BF16 TOPS) | candidate_at_min_compute |")
    lines.append("|---|---:|---:|---:|---|---:|---|")
    for r in rows:
        lines.append(
            "| {w} | {t_us:.2f} | {p} | {bw_gb:.3f} | {cbw} | {cmp_tops:.3f} | {ccmp} |".format(
                w=r["workload"],
                t_us=float(r["target_latency_us"]),
                p=int(r["pareto_points"]),
                bw_gb=float(r["min_required_bandwidth_gb_per_s"]),
                cbw=r["candidate_at_min_bw"],
                cmp_tops=float(r["min_required_compute_bf16_tops"]),
                ccmp=r["candidate_at_min_compute"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n")
    return True


def _plot_latency_with_frontier(forward_rows, frontier_rows, out_path):
    import matplotlib.pyplot as plt

    _apply_style(plt)
    by_candidate = defaultdict(list)
    for r in forward_rows:
        by_candidate[r["candidate"]].append((r["bandwidth_bytes_per_s"], r["t_lb_s"]))

    plt.figure(figsize=(8.2, 5.0))
    for _, pts in by_candidate.items():
        pts = sorted(pts, key=lambda x: x[0])
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        plt.plot(x, y, alpha=0.2, linewidth=1.0, color="#475569")

    frontier_sorted = sorted(frontier_rows, key=lambda r: r["bandwidth_bytes_per_s"])
    fx = [r["bandwidth_bytes_per_s"] for r in frontier_sorted]
    fy = [r["t_lb_s"] for r in frontier_sorted]
    plt.plot(fx, fy, color="#0f172a", linewidth=2.2, marker="o", markersize=3, label="frontier")

    ax = plt.gca()
    _apply_log_ticks(ax, x_log=True, y_log=True)
    plt.xlabel("Bandwidth (bytes/s)")
    plt.ylabel("Latency lower bound (s)")
    plt.title("Lower-Bound Latency vs Bandwidth (Roofline-Like)")
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180)
    plt.close()


def _plot_bottleneck_regime(forward_rows, out_path):
    import matplotlib.pyplot as plt

    _apply_style(plt)
    colors = {"memory": "#dc2626", "compute": "#2563eb", "balanced": "#16a34a"}
    grouped = defaultdict(list)
    for r in forward_rows:
        grouped[r["bottleneck"]].append(r)

    plt.figure(figsize=(8.2, 5.0))
    for bottleneck, rows in grouped.items():
        x = [r["bandwidth_bytes_per_s"] for r in rows]
        y = [r["t_lb_s"] for r in rows]
        plt.scatter(
            x,
            y,
            s=14,
            alpha=0.65,
            color=colors.get(bottleneck, "#64748b"),
            label=bottleneck,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Bandwidth (bytes/s)")
    plt.ylabel("Latency lower bound (s)")
    plt.title("Bottleneck Regime Map")
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180)
    plt.close()


def _plot_required_bw_vs_target(inverse_rows, out_path):
    import matplotlib.pyplot as plt

    _apply_style(plt)
    feasible = []
    seen_targets = set()
    for r in inverse_rows:
        status = r.get("status")
        if status != "feasible":
            continue
        target = r.get("target_latency_s")
        bw = r.get("min_required_bandwidth_bytes_per_s")
        if target in seen_targets:
            continue
        if target is None or bw in (None, ""):
            continue
        seen_targets.add(target)
        feasible.append((float(target), float(bw)))

    feasible.sort(key=lambda x: x[0])
    if not feasible:
        return False

    x = [p[0] for p in feasible]
    y = [p[1] for p in feasible]

    plt.figure(figsize=(8.2, 5.0))
    plt.plot(x, y, color="#f97316", marker="o", linewidth=2.0, markersize=4)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Target latency (s)")
    plt.ylabel("Min required bandwidth (bytes/s)")
    plt.title("Inverse DSE: Required Bandwidth vs Target Latency")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180)
    plt.close()
    return True


def _pick_sla_target(inverse_rows, requested_target):
    targets = sorted(
        set(float(r["target_latency_s"]) for r in inverse_rows if r.get("target_latency_s") not in (None, ""))
    )
    if not targets:
        return None
    if requested_target is None:
        # Default to the middle target for a stable, representative fixed-SLA chart.
        return targets[len(targets) // 2]
    return min(targets, key=lambda t: abs(t - requested_target))


def _plot_required_bw_fixed_sla(inverse_rows, out_path, requested_target=None):
    import matplotlib.pyplot as plt

    _apply_style(plt)
    target = _pick_sla_target(inverse_rows, requested_target)
    if target is None:
        return False, None

    # Keep one row per candidate at this target.
    by_candidate = {}
    for r in inverse_rows:
        if r.get("target_latency_s") in (None, ""):
            continue
        t = float(r["target_latency_s"])
        if abs(t - target) > 1e-18:
            continue
        cand = r.get("candidate", "")
        if cand and cand not in by_candidate:
            by_candidate[cand] = r

    if not by_candidate:
        return False, target

    candidates = sorted(by_candidate.keys())
    yvals = []
    colors = []
    labels = []

    # Use finite values only for bar heights; keep infeasible as zero + label.
    finite_vals = []
    for cand in candidates:
        row = by_candidate[cand]
        bw = row.get("required_bandwidth_bytes_per_s")
        feasible = str(row.get("feasible_with_infinite_bandwidth", "")).lower() == "true"
        if bw in (None, ""):
            yvals.append(0.0)
            colors.append("#94a3b8")
            labels.append("infeasible")
        else:
            v = float(bw)
            yvals.append(v)
            finite_vals.append(v)
            colors.append("#16a34a" if feasible else "#94a3b8")
            labels.append("feasible" if feasible else "infeasible")

    if not finite_vals:
        return False, target

    floor = min(finite_vals) * 0.6
    yvals = [v if v > 0 else floor for v in yvals]

    plt.figure(figsize=(8.2, 4.8))
    xs = list(range(len(candidates)))
    plt.bar(xs, yvals, color=colors, alpha=0.85)
    plt.yscale("log")
    plt.xticks(xs, candidates, rotation=25, ha="right")
    plt.xlabel("Candidate")
    plt.ylabel("Required bandwidth (bytes/s)")
    plt.title("Required Bandwidth at Fixed SLA (target = {:.3e} s)".format(target))

    ymax = max(finite_vals)
    # Annotate infeasible bars near the top for readability.
    for i, text in enumerate(labels):
        if text == "infeasible":
            plt.text(i, ymax * 0.85, text, rotation=90, va="top", ha="center", fontsize=8, color="#334155")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180)
    plt.close()
    return True, target


def _plot_bottleneck_regime_multi_workload(workload_frontiers, out_path):
    import matplotlib.pyplot as plt

    _apply_style(plt)
    if not workload_frontiers:
        return False

    colors = {"memory": "#dc2626", "compute": "#2563eb", "balanced": "#16a34a"}
    names = sorted(workload_frontiers.keys())

    plt.figure(figsize=(9.0, max(3.6, 0.85 * len(names) + 1.8)))
    for yi, name in enumerate(names):
        rows = sorted(workload_frontiers[name], key=lambda r: r["bandwidth_bytes_per_s"])
        for r in rows:
            c = colors.get(r.get("bottleneck", ""), "#64748b")
            plt.scatter(r["bandwidth_bytes_per_s"], yi, color=c, s=20, alpha=0.9)

    plt.xscale("log")
    plt.yticks(list(range(len(names))), names)
    plt.xlabel("Bandwidth (bytes/s)")
    plt.ylabel("Workload")
    plt.title("Bottleneck Regime Across Workloads")

    legend_handles = []
    for label, color in colors.items():
        legend_handles.append(plt.Line2D([], [], marker="o", linestyle="", color=color, label=label))
    plt.legend(handles=legend_handles, title="Bottleneck", loc="best")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180)
    plt.close()
    return True


def _plot_frontier_decomposition(frontier_rows, out_path):
    import matplotlib.pyplot as plt

    _apply_style(plt)
    rows = sorted(frontier_rows, key=lambda r: r["bandwidth_bytes_per_s"])
    bw = [r["bandwidth_bytes_per_s"] for r in rows]
    t_lb = [r["t_lb_s"] for r in rows]
    t_mem = [r["t_mem_lb_s"] for r in rows]
    t_cmp = [r["t_cmp_lb_s"] for r in rows]

    plt.figure(figsize=(8.2, 5.0))
    plt.plot(bw, t_lb, color="#0f172a", linewidth=2.3, label="t_lb")
    plt.plot(bw, t_mem, color="#dc2626", linewidth=1.8, linestyle="--", label="t_mem_lb")
    plt.plot(bw, t_cmp, color="#2563eb", linewidth=1.8, linestyle="--", label="t_cmp_lb")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Bandwidth (bytes/s)")
    plt.ylabel("Latency (s)")
    plt.title("Frontier Latency Decomposition")
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180)
    plt.close()


def _plot_candidate_tradeoff_map(forward_rows, frontier_rows, out_path):
    import matplotlib.pyplot as plt

    _apply_style(plt)
    by_candidate = {}
    for r in forward_rows:
        cand = r["candidate"]
        if cand not in by_candidate:
            by_candidate[cand] = {
                "hbm_bytes": r["hbm_bytes"],
                "compute_ops": r["compute_ops"],
                "intensity": r["intensity_ops_per_hbm_byte"],
            }

    wins = defaultdict(int)
    for r in frontier_rows:
        wins[r["candidate"]] += 1

    x = []
    y = []
    s = []
    labels = []
    for cand, v in sorted(by_candidate.items()):
        x.append(v["hbm_bytes"])
        y.append(v["compute_ops"])
        s.append(20 + 35 * wins.get(cand, 0))
        labels.append(cand)

    plt.figure(figsize=(8.2, 5.0))
    plt.scatter(x, y, s=s, c="#0ea5e9", alpha=0.75, edgecolors="#0f172a", linewidths=0.6)
    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]), fontsize=8, alpha=0.85)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("HBM bytes (feature)")
    plt.ylabel("Compute ops (feature)")
    plt.title("Candidate Tradeoff Map (bubble size = frontier wins)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180)
    plt.close()


def _plot_winner_regions(frontier_rows, out_path):
    import matplotlib.pyplot as plt

    _apply_style(plt)
    rows = sorted(frontier_rows, key=lambda r: r["bandwidth_bytes_per_s"])
    candidates = sorted({r["candidate"] for r in rows})
    idx = dict((c, i) for i, c in enumerate(candidates))
    x = [r["bandwidth_bytes_per_s"] for r in rows]
    y = [idx[r["candidate"]] for r in rows]

    plt.figure(figsize=(8.2, 3.6))
    plt.step(x, y, where="post", color="#7c3aed", linewidth=2.0)
    plt.scatter(x, y, s=18, color="#7c3aed")
    plt.xscale("log")
    plt.yticks(list(range(len(candidates))), candidates)
    plt.xlabel("Bandwidth (bytes/s)")
    plt.ylabel("Frontier winner")
    plt.title("Winner Candidate Regions Across Bandwidth")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180)
    plt.close()


def _write_frontier_switch_table(frontier_rows, out_path):
    rows = sorted(frontier_rows, key=lambda r: r["bandwidth_bytes_per_s"])
    if not rows:
        out_path.write_text("")
        return

    segments = []
    cur_candidate = rows[0]["candidate"]
    start_bw = rows[0]["bandwidth_bytes_per_s"]
    end_bw = rows[0]["bandwidth_bytes_per_s"]

    for r in rows[1:]:
        cand = r["candidate"]
        bw = r["bandwidth_bytes_per_s"]
        if cand == cur_candidate:
            end_bw = bw
            continue
        segments.append((cur_candidate, start_bw, end_bw))
        cur_candidate = cand
        start_bw = bw
        end_bw = bw
    segments.append((cur_candidate, start_bw, end_bw))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "start_bw_bytes_per_s", "end_bw_bytes_per_s"])
        for cand, start, end in segments:
            w.writerow([cand, "{:.6e}".format(start), "{:.6e}".format(end)])


def main():
    parser = argparse.ArgumentParser(
        description="Generate minimal DSE plots: basic roofline-like + multi-workload bounds."
    )
    parser.add_argument(
        "--in_dir",
        default="dse/output/plot_from_log",
        help="Directory containing forward_bounds.csv/frontier.csv/inverse_bandwidth_bounds.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="dse/output/plot_from_log/plots_meeting",
        help="Directory to write generated plots and CSV table",
    )
    parser.add_argument(
        "--sla_target_s",
        type=float,
        default=None,
        help="Optional fixed latency target (s) for required_resources_fixed_sla chart",
    )
    parser.add_argument(
        "--workload_dirs",
        default=None,
        help="Optional comma-separated list of DSE output dirs for multi-workload bounds chart",
    )
    parser.add_argument(
        "--feas_target_s",
        type=float,
        default=None,
        help="Deprecated: kept for CLI compatibility (ignored).",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    forward_csv = in_dir / "forward_bounds.csv"
    frontier_csv = in_dir / "frontier.csv"
    inverse_csv = in_dir / "inverse_bandwidth_bounds.csv"

    for p in [forward_csv, frontier_csv, inverse_csv]:
        if not p.exists():
            raise FileNotFoundError("Missing required input file: {}".format(p))

    forward_rows = _read_csv(forward_csv)
    for key in [
        "hbm_bytes",
        "compute_ops",
        "intensity_ops_per_hbm_byte",
        "bandwidth_bytes_per_s",
        "peak_compute_ops_per_s",
        "t_mem_lb_s",
        "t_cmp_lb_s",
        "t_lb_s",
    ]:
        forward_rows = _as_float(forward_rows, key)

    frontier_rows = _read_csv(frontier_csv)
    for key in [
        "hbm_bytes",
        "compute_ops",
        "intensity_ops_per_hbm_byte",
        "bandwidth_bytes_per_s",
        "peak_compute_ops_per_s",
        "t_mem_lb_s",
        "t_cmp_lb_s",
        "t_lb_s",
    ]:
        frontier_rows = _as_float(frontier_rows, key)

    _plot_latency_with_frontier(forward_rows, frontier_rows, out_dir / "latency_vs_bw_frontier.png")

    if args.workload_dirs:
        dirs = [d.strip() for d in args.workload_dirs.split(",") if d.strip()]
        workloads = _load_workload_outputs(dirs)

        multi_bounds = _build_multi_workload_bounds(workloads)
        _write_rows_csv(out_dir / "multi_workload_required_bounds.csv", multi_bounds)

        _plot_multi_required_curve(
            multi_bounds,
            y_key="min_required_bandwidth_gb_per_s",
            y_label="Min required bandwidth (GB/s)",
            title="Bandwidth Lower Bound vs Latency Target",
            out_path=out_dir / "required_bw_vs_target_latency.png",
        )
        legacy = out_dir / "required_bw_vs_target_latency_multi.png"
        if legacy.exists():
            legacy.unlink()
        _plot_fixed_sla_required_resources(
            multi_bounds,
            out_dir / "required_resources_fixed_sla.png",
            requested_target=args.sla_target_s,
        )
        _plot_fixed_sla_single_metric(
            multi_bounds,
            out_dir / "required_bw_fixed_sla.png",
            requested_target=args.sla_target_s,
            metric="bw",
        )
        _plot_fixed_sla_single_metric(
            multi_bounds,
            out_dir / "required_compute_fixed_sla.png",
            requested_target=args.sla_target_s,
            metric="compute",
        )
        fixed_target = _pick_target(multi_bounds, args.sla_target_s)
        feas_rows = _build_feasibility_points(workloads, fixed_target)
        _write_feasibility_map_table(
            feas_rows,
            fixed_target,
            out_dir / "feasibility_map_fixed_latency_table.csv",
        )
        _write_workload_descriptions(workloads, out_dir / "workload_descriptions.md")

    print("Wrote minimal plot bundle to {}".format(out_dir))


if __name__ == "__main__":
    main()
