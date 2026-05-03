#!/usr/bin/env python3
"""parse_pt_hier_to_buckets.py

Parse a Synopsys PrimeTime ``Gemmini.power.hier.rpt`` and roll the top-level
children of the ``Gemmini`` design into a small set of logical IP buckets used
by ACT calibration.

The mapping is locked for Experiment 1 (see ``EXPERIMENT_1_PLAN.md`` §3):

    mesh           -> ex_controller         (proxy for compute_path)
    accumulator    -> any leaf containing "Accumulator"
    scratchpad     -> rest of spad
    controller_dma -> load_controller, store_controller, tlb,
                      and anything inside spad named *Stream{Reader,Writer}* or *xbar*
    glue           -> reservation_station, mod (LoopConv), mod_1 (LoopMatmul),
                      raw_cmd_q, unrolled_cmd_q, req_arb*, counters, im2col

Power columns in the report are watts. For a 20 us PT capture window we report:

    energy_uj_per_bucket = total_power_w * 20.0    (20 us = 20.0 us)

Usage:
    parse_pt_hier_to_buckets.py <path/to/Gemmini.power.hier.rpt> \
        [--window-us 20.0] [--out grouped_buckets.json]

Writes JSON to stdout if --out not given.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROW_RE = re.compile(
    r"^(\s*)(\S+)\s*\(([^)]+)\)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$"
)

BUCKET_RULES = [
    ("controller_dma", re.compile(r"^(load_controller|store_controller|tlb)$")),
    ("glue", re.compile(r"^(reservation_station|mod|mod_1|raw_cmd_q|unrolled_cmd_q|req_arb.*|counters|im2col|reservation_station_completed_arb)$")),
    ("mesh", re.compile(r"^ex_controller$")),
]


def classify_top_child(inst):
    if inst == "spad":
        return "scratchpad"
    for bucket, pat in BUCKET_RULES:
        if pat.match(inst):
            return bucket
    return "glue"


CHIP_LINE_RE = re.compile(r"^Gemmini\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$")


def parse_chip_total(rpt_path):
    for line in rpt_path.read_text().splitlines():
        m = CHIP_LINE_RE.match(line)
        if m:
            return float(m.group(4))
    return None


def parse_top_children(rpt_path):
    rows = []
    in_table = False
    for line in rpt_path.read_text().splitlines():
        if line.lstrip().startswith("Gemmini ") and not in_table:
            in_table = True
            continue
        if not in_table:
            continue
        if line.strip() == "":
            if rows:
                break
            continue
        m = ROW_RE.match(line)
        if not m:
            continue
        indent, inst, design = m.group(1), m.group(2), m.group(3)
        if len(indent) != 2:
            continue
        rows.append({
            "instance": inst,
            "design": design,
            "int_w": float(m.group(4)),
            "switch_w": float(m.group(5)),
            "leak_w": float(m.group(6)),
            "total_w": float(m.group(7)),
            "pct": float(m.group(8)),
        })
    return rows


def bucket_energies(rows, window_us):
    out = {
        "mesh": {"power_w": 0.0, "energy_uj": 0.0, "instances": []},
        "accumulator": {"power_w": 0.0, "energy_uj": 0.0, "instances": []},
        "scratchpad": {"power_w": 0.0, "energy_uj": 0.0, "instances": []},
        "controller_dma": {"power_w": 0.0, "energy_uj": 0.0, "instances": []},
        "glue": {"power_w": 0.0, "energy_uj": 0.0, "instances": []},
    }
    for r in rows:
        b = classify_top_child(str(r["instance"]))
        out[b]["power_w"] += float(r["total_w"])
        out[b]["instances"].append(f"{r['instance']} ({r['design']})")
    for b in out.values():
        b["energy_uj"] = b["power_w"] * window_us
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rpt", type=Path, help="Path to Gemmini.power.hier.rpt")
    ap.add_argument("--window-us", type=float, default=20.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = parse_top_children(args.rpt)
    if not rows:
        print("ERROR: no top-level Gemmini children parsed", file=sys.stderr)
        return 2
    buckets = bucket_energies(rows, args.window_us)
    bucketed_w = sum(b["power_w"] for b in buckets.values())
    chip_total_w = parse_chip_total(args.rpt)
    residual_w = chip_total_w - bucketed_w if chip_total_w is not None else None
    payload = {
        "source_rpt": str(args.rpt),
        "window_us": args.window_us,
        "totals": {
            "chip_total_w": chip_total_w,
            "chip_total_energy_uj": (chip_total_w * args.window_us) if chip_total_w is not None else None,
            "bucketed_w": bucketed_w,
            "bucketed_energy_uj": bucketed_w * args.window_us,
            "residual_w": residual_w,
            "residual_energy_uj": (residual_w * args.window_us) if residual_w is not None else None,
            "residual_note": "Difference between Gemmini-line total and sum of top-level children. Typically clock distribution, parasitics, or design-top wiring not under any named child.",
        },
        "buckets": buckets,
        "raw_top_children": rows,
    }
    out_text = json.dumps(payload, indent=2)
    if args.out is None:
        print(out_text)
    else:
        args.out.write_text(out_text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
