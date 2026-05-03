"""Forward-pass analytical lower-bound model for ACT Phase-1 MVP."""


class EvalRow(object):
    def __init__(
        self,
        candidate,
        kernel_name,
        hbm_bytes,
        compute_ops,
        intensity_ops_per_hbm_byte,
        bandwidth_bytes_per_s,
        peak_compute_ops_per_s,
        t_mem_lb_s,
        t_cmp_lb_s,
        t_lb_s,
        bottleneck,
    ):
        self.candidate = candidate
        self.kernel_name = kernel_name
        self.hbm_bytes = hbm_bytes
        self.compute_ops = compute_ops
        self.intensity_ops_per_hbm_byte = intensity_ops_per_hbm_byte
        self.bandwidth_bytes_per_s = bandwidth_bytes_per_s
        self.peak_compute_ops_per_s = peak_compute_ops_per_s
        self.t_mem_lb_s = t_mem_lb_s
        self.t_cmp_lb_s = t_cmp_lb_s
        self.t_lb_s = t_lb_s
        self.bottleneck = bottleneck

    def to_dict(self):
        return {
            "candidate": self.candidate,
            "kernel_name": self.kernel_name,
            "hbm_bytes": self.hbm_bytes,
            "compute_ops": self.compute_ops,
            "intensity_ops_per_hbm_byte": self.intensity_ops_per_hbm_byte,
            "bandwidth_bytes_per_s": self.bandwidth_bytes_per_s,
            "peak_compute_ops_per_s": self.peak_compute_ops_per_s,
            "t_mem_lb_s": self.t_mem_lb_s,
            "t_cmp_lb_s": self.t_cmp_lb_s,
            "t_lb_s": self.t_lb_s,
            "bottleneck": self.bottleneck,
        }


def evaluate_row(feature, bandwidth_bytes_per_s, peak_compute_ops_per_s):
    if bandwidth_bytes_per_s <= 0:
        raise ValueError("bandwidth_bytes_per_s must be positive")
    if peak_compute_ops_per_s <= 0:
        raise ValueError("peak_compute_ops_per_s must be positive")

    t_mem = feature.hbm_bytes / bandwidth_bytes_per_s
    t_cmp = feature.compute_ops / peak_compute_ops_per_s
    t_lb = max(t_mem, t_cmp)

    if abs(t_mem - t_cmp) <= 1e-18:
        bottleneck = "balanced"
    elif t_mem > t_cmp:
        bottleneck = "memory"
    else:
        bottleneck = "compute"

    return EvalRow(
        candidate=feature.candidate,
        kernel_name=feature.kernel_name,
        hbm_bytes=feature.hbm_bytes,
        compute_ops=feature.compute_ops,
        intensity_ops_per_hbm_byte=feature.intensity_ops_per_hbm_byte,
        bandwidth_bytes_per_s=bandwidth_bytes_per_s,
        peak_compute_ops_per_s=peak_compute_ops_per_s,
        t_mem_lb_s=t_mem,
        t_cmp_lb_s=t_cmp,
        t_lb_s=t_lb,
        bottleneck=bottleneck,
    )


def evaluate_sweep(features, bandwidth_values, peak_compute_ops_per_s):
    rows = []
    for bw in bandwidth_values:
        for feat in features:
            rows.append(evaluate_row(feat, bw, peak_compute_ops_per_s))
    return rows


def frontier(rows):
    by_bw = {}
    for row in rows:
        by_bw.setdefault(row.bandwidth_bytes_per_s, []).append(row)

    out = []
    for bw in sorted(by_bw.keys()):
        candidates = by_bw[bw]
        best = min(candidates, key=lambda r: (r.t_lb_s, r.hbm_bytes, r.compute_ops, r.candidate))
        out.append(best)
    return out


def inverse_bandwidth_bounds(features, target_latencies_s, peak_compute_ops_per_s):
    targets = list(target_latencies_s)
    feats = list(features)
    if any(t <= 0 for t in targets):
        raise ValueError("All target latencies must be positive")

    results = []

    for target in targets:
        per_candidate = []
        for feat in feats:
            t_cmp = feat.compute_ops / peak_compute_ops_per_s
            feasible = target >= t_cmp
            bw_min = (feat.hbm_bytes / target) if feasible else None

            per_candidate.append(
                {
                    "candidate": feat.candidate,
                    "target_latency_s": target,
                    "t_cmp_lb_s": t_cmp,
                    "required_bandwidth_bytes_per_s": bw_min,
                    "feasible_with_infinite_bandwidth": feasible,
                }
            )

        feasible_rows = [r for r in per_candidate if r["required_bandwidth_bytes_per_s"] is not None]
        if feasible_rows:
            best = min(feasible_rows, key=lambda r: r["required_bandwidth_bytes_per_s"])
            status = "feasible"
        else:
            best = None
            status = "compute_limited"

        results.append(
            {
                "target_latency_s": target,
                "status": status,
                "best_candidate": None if best is None else best["candidate"],
                "min_required_bandwidth_bytes_per_s": None
                if best is None
                else best["required_bandwidth_bytes_per_s"],
                "candidates": per_candidate,
            }
        )

    return results
