"""CLI for forward-pass memory-bound feasibility analysis on ACT artifacts."""

import argparse
import csv
import json
import math
from pathlib import Path

from .features import extract_feature_table
from .model_forward import evaluate_sweep, frontier, inverse_bandwidth_bounds
from .parse_pii import parse_pii_dir
from .plot import plot_latency_vs_bw, plot_required_bw_vs_target_latency


FUTURE_HOOK_HINT = (
    "Preferred pre-schedule callpoint: generators/backend/generic/src/pipeline.rs, "
    "inside Phase-1 extractor loop after pii.save(...) and before cpp_bridge(...) scheduling/allocation."
)


def _coerce_scalar(text):
    s = text.strip()
    if not s:
        return ""
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    lo = s.lower()
    if lo == "true":
        return True
    if lo == "false":
        return False
    if lo == "null" or lo == "none":
        return None
    try:
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return s


def _normalize_data(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = _normalize_data(v)
        return out
    if isinstance(obj, list):
        return [_normalize_data(x) for x in obj]
    if isinstance(obj, str):
        return _coerce_scalar(obj)
    return obj


def _tokenize_yaml_subset(text):
    tokens = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        # Keep parser strict/simple: comments only when line starts with '#'.
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if "\t" in line:
            raise ValueError("Tabs are not supported in YAML fallback parser (line {})".format(lineno))
        tokens.append((indent, stripped, lineno))
    return tokens


def _parse_yaml_subset_node(tokens, idx, indent):
    if idx >= len(tokens):
        return {}, idx
    tok_indent, tok_text, tok_line = tokens[idx]
    if tok_indent != indent:
        raise ValueError("Invalid indentation at line {}: expected {}, got {}".format(tok_line, indent, tok_indent))

    # Parse a list block.
    if tok_text.startswith("- "):
        out = []
        while idx < len(tokens):
            cur_indent, cur_text, cur_line = tokens[idx]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise ValueError("Unexpected indentation at line {}".format(cur_line))
            if not cur_text.startswith("- "):
                raise ValueError("Mixed list/map indentation at line {}".format(cur_line))

            item = cur_text[2:].strip()
            idx += 1
            if item:
                out.append(_coerce_scalar(item))
            else:
                if idx < len(tokens) and tokens[idx][0] > indent:
                    child_indent = tokens[idx][0]
                    child, idx = _parse_yaml_subset_node(tokens, idx, child_indent)
                    out.append(child)
                else:
                    out.append(None)
        return out, idx

    # Parse a map block.
    out = {}
    while idx < len(tokens):
        cur_indent, cur_text, cur_line = tokens[idx]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise ValueError("Unexpected indentation at line {}".format(cur_line))
        if cur_text.startswith("- "):
            raise ValueError("Unexpected list item at line {}".format(cur_line))
        if ":" not in cur_text:
            raise ValueError("Expected key:value mapping at line {}".format(cur_line))

        key, rhs = cur_text.split(":", 1)
        key = key.strip()
        rhs = rhs.strip()
        idx += 1

        if rhs:
            out[key] = _coerce_scalar(rhs)
        else:
            if idx < len(tokens) and tokens[idx][0] > indent:
                child_indent = tokens[idx][0]
                child, idx = _parse_yaml_subset_node(tokens, idx, child_indent)
                out[key] = child
            else:
                out[key] = {}
    return out, idx


def _load_yaml_with_fallback(text):
    try:
        import yaml

        data = yaml.safe_load(text)
        if data is None:
            return {}
        return _normalize_data(data)
    except Exception:
        tokens = _tokenize_yaml_subset(text)
        if not tokens:
            return {}
        root_indent = tokens[0][0]
        data, idx = _parse_yaml_subset_node(tokens, 0, root_indent)
        if idx != len(tokens):
            line = tokens[idx][2]
            raise ValueError("Unparsed YAML content near line {}".format(line))
        return _normalize_data(data)


def _read_structured(path):
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise IOError("Config file not found: {}".format(p))
    txt = p.read_text()

    if p.suffix.lower() == ".json":
        return json.loads(txt)

    try:
        data = _load_yaml_with_fallback(txt)
        if not isinstance(data, dict):
            raise ValueError("Expected mapping at root of {}".format(p))
        return data
    except Exception as e:
        raise ValueError("Failed to parse config {}: {}".format(p, e))


def _linspace(start, stop, num):
    if num <= 1:
        return [start]
    step = (stop - start) / float(num - 1)
    return [start + i * step for i in range(num)]


def _logspace(start, stop, num):
    if start <= 0 or stop <= 0:
        raise ValueError("log sweep requires positive start/stop")
    if num <= 1:
        return [start]
    ls = math.log10(start)
    le = math.log10(stop)
    return [10 ** x for x in _linspace(ls, le, num)]


def _bandwidth_values(config):
    sweep_cfg = ((config.get("sweep") or {}).get("bandwidth") or {})
    if isinstance(sweep_cfg, dict) and "values" in sweep_cfg:
        vals = [float(v) for v in sweep_cfg["values"]]
        if not vals:
            raise ValueError("sweep.bandwidth.values is empty")
        return vals

    hw_bw = (config.get("hardware") or {}).get("bandwidth_bytes_per_s")

    if not sweep_cfg:
        if hw_bw is None:
            raise ValueError(
                "No bandwidth sweep provided and hardware.bandwidth_bytes_per_s missing"
            )
        return [float(hw_bw)]

    start = float(sweep_cfg.get("start"))
    stop = float(sweep_cfg.get("stop"))
    num = int(sweep_cfg.get("num", 1))
    scale = str(sweep_cfg.get("scale", "linear")).lower()

    if scale == "log":
        return _logspace(start, stop, num)
    if scale == "linear":
        return _linspace(start, stop, num)
    raise ValueError("Unknown sweep.bandwidth.scale: {}".format(scale))


def _target_latencies(base_cfg, override_cfg):
    merged = {}
    merged.update(base_cfg)
    if override_cfg:
        merged.update(override_cfg)

    targets = merged.get("targets", {})
    vals = targets.get("latencies_s", [])
    return [float(v) for v in vals]


def _write_csv(path, rows):
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        path.write_text("")
        return

    fieldnames = []
    seen = set()
    for row in rows_list:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def _flatten_inverse_rows(inverse_rows):
    rows = []
    for block in inverse_rows:
        target = block.get("target_latency_s")
        status = block.get("status")
        best = block.get("best_candidate")
        min_bw = block.get("min_required_bandwidth_bytes_per_s")
        for cand in block.get("candidates", []):
            rows.append(
                {
                    "target_latency_s": target,
                    "status": status,
                    "best_candidate": best,
                    "min_required_bandwidth_bytes_per_s": min_bw,
                    "candidate": cand.get("candidate"),
                    "t_cmp_lb_s": cand.get("t_cmp_lb_s"),
                    "required_bandwidth_bytes_per_s": cand.get(
                        "required_bandwidth_bytes_per_s"
                    ),
                    "feasible_with_infinite_bandwidth": cand.get(
                        "feasible_with_infinite_bandwidth"
                    ),
                }
            )
    return rows

'''Sees input_mode=pii Calls 0.pii.'''
def _parse_candidates(input_path, input_mode):
    p = Path(input_path)

    if input_mode == "pii" or input_mode == "auto":
        return parse_pii_dir(p), "pii"
    raise ValueError("Unsupported input_mode: {} (allowed: auto, pii)".format(input_mode))


def run_forward_bound(
    input_path,
    hw_config_path,
    targets_config_path,
    out_dir,
    with_plot,
    input_mode,
):
    #Loads hardware/sweep config.yaml into a Python dict.
    hw_cfg = _read_structured(hw_config_path)

    #Loads target latency overrides into a dict (or {} if None). (hardware section)
    targets_override = _read_structured(targets_config_path)

    #Pulls peak compute number from config and converts to float.
    peak_compute = float((hw_cfg.get("hardware") or {}).get("peak_compute_ops_per_s"))
    if peak_compute <= 0:
        raise ValueError("hardware.peak_compute_ops_per_s must be positive")

    #
    bw_values = _bandwidth_values(hw_cfg)
    target_latencies = _target_latencies(hw_cfg, targets_override)

    programs, mode_used = _parse_candidates(input_path, input_mode)
    features = extract_feature_table(programs)

    eval_rows = evaluate_sweep(features, bw_values, peak_compute)
    frontier_rows = frontier(eval_rows)
    inverse_rows = inverse_bandwidth_bounds(features, target_latencies, peak_compute)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_csv(out / "forward_bounds.csv", [r.to_dict() for r in eval_rows])
    _write_csv(out / "frontier.csv", [r.to_dict() for r in frontier_rows])
    _write_csv(out / "candidate_features.csv", [r.to_dict() for r in features])
    _write_csv(out / "inverse_bandwidth_bounds.csv", _flatten_inverse_rows(inverse_rows))

    plot_path = out / "plots" / "latency_vs_bw.png"
    inverse_plot_path = out / "plots" / "required_bw_vs_target_latency.png"
    plot_written = False
    inverse_plot_written = False
    if with_plot:
        plot_written = plot_latency_vs_bw(frontier_rows, plot_path)
        inverse_plot_written = plot_required_bw_vs_target_latency(
            inverse_rows, inverse_plot_path
        )

    bw_to_bottleneck = [
        {
            "bandwidth_bytes_per_s": r.bandwidth_bytes_per_s,
            "candidate": r.candidate,
            "bottleneck": r.bottleneck,
            "t_lb_s": r.t_lb_s,
        }
        for r in frontier_rows
    ]

    summary = {
        "status": "ok",
        "input_path": str(input_path),
        "input_mode_used": mode_used,
        "num_candidates": len(features),
        "num_eval_rows": len(eval_rows),
        "hardware": {
            "peak_compute_ops_per_s": peak_compute,
            "bandwidth_sweep_bytes_per_s": bw_values,
        },
        "targets": {"latencies_s": target_latencies},
        "future_hook_hint": FUTURE_HOOK_HINT,
        "frontier_bottlenecks": bw_to_bottleneck,
        "inverse_bandwidth_bounds": inverse_rows,
        "outputs": {
            "forward_bounds_csv": str(out / "forward_bounds.csv"),
            "frontier_csv": str(out / "frontier.csv"),
            "candidate_features_csv": str(out / "candidate_features.csv"),
            "inverse_bandwidth_bounds_csv": str(out / "inverse_bandwidth_bounds.csv"),
            "plot_written": plot_written,
            "plot_path": str(plot_path),
            "inverse_plot_written": inverse_plot_written,
            "inverse_plot_path": str(inverse_plot_path),
        },
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="ACT Phase-1 forward-pass feasibility analysis (memory-bound MVP)"
    )
    parser.add_argument("--input", required=True, help="Candidate file/dir (.pii pre-schedule only)")
    parser.add_argument("--input_mode", default="auto", choices=["auto", "pii"], help="Phase-1 only: auto/ pii parse .pii candidates")
    parser.add_argument("--hw_config", required=True, help="YAML/JSON hardware+sweep config")
    parser.add_argument("--targets", default=None, help="Optional YAML/JSON target-latency override")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Generate optional latency-vs-bandwidth plot")

    args = parser.parse_args()

    summary = run_forward_bound(
        input_path=args.input,
        hw_config_path=args.hw_config,
        targets_config_path=args.targets,
        out_dir=args.out,
        with_plot=args.plot,
        input_mode=args.input_mode,
    )

    print(json.dumps({"status": summary["status"], "input_mode_used": summary["input_mode_used"], "outputs": summary["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
