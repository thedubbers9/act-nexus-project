"""Plot utilities for DSE forward-bound outputs."""

from pathlib import Path


def plot_latency_vs_bw(frontier_rows, out_path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    rows = sorted(list(frontier_rows), key=lambda r: r.bandwidth_bytes_per_s)
    if not rows:
        return False

    x = [r.bandwidth_bytes_per_s for r in rows]
    y = [r.t_lb_s for r in rows]
    c = [r.bottleneck for r in rows]

    colors = {"memory": "tab:red", "compute": "tab:blue", "balanced": "tab:green"}
    point_colors = [colors.get(tag, "tab:gray") for tag in c]

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, color="black", linewidth=1.0, alpha=0.6)
    plt.scatter(x, y, c=point_colors, s=22)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Bandwidth (bytes/s)")
    plt.ylabel("Latency lower bound (s)")
    plt.title("Forward-Pass Lower Bound Trend vs Bandwidth")
    plt.grid(True, which="both", linestyle="--", alpha=0.25)

    legend_handles = []
    for label, color in colors.items():
        h = plt.Line2D([], [], marker="o", linestyle="", color=color, label=label)
        legend_handles.append(h)
    plt.legend(handles=legend_handles, title="Bottleneck", loc="best")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), dpi=180)
    plt.close()
    return True


def plot_required_bw_vs_target_latency(inverse_rows, out_path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    pairs = []
    for row in inverse_rows:
        if row.get("status") != "feasible":
            continue
        target = row.get("target_latency_s")
        bw = row.get("min_required_bandwidth_bytes_per_s")
        if target is None or bw is None:
            continue
        pairs.append((float(target), float(bw)))

    if not pairs:
        return False

    pairs = sorted(pairs, key=lambda x: x[0])
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, color="tab:orange", marker="o", linewidth=1.4, markersize=4)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Target latency (s)")
    plt.ylabel("Min required bandwidth (bytes/s)")
    plt.title("Inverse DSE: Required Bandwidth vs Target Latency")
    plt.grid(True, which="both", linestyle="--", alpha=0.25)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), dpi=180)
    plt.close()
    return True
