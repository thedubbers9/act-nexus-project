#!/usr/bin/env python3
"""Deep Dive diagnostic helpers using pt_comparison_bundle.json (no VCS/PT required).

Implements checklist items that are JSON-only:
  Step 1 — Implied pJ per tensor op / per MAC from ``act.instruction_breakdown_pj`` + op counts
           (same geometry as gemmini_matmul_pt_matched / bundle mapping_note).
  Step 4 — Window vs SAIF cycles, epoch scaling, ACT vs PT totals, crude per-cycle energy sanity.

Does NOT replace PT report_switching_activity, leakage splits, or GLS debug.

Usage:
  python3 deep_dive_bundle_audit.py [--bundle PATH] [--width N]
"""

import argparse
import json
import sys
from pathlib import Path


def _act_root():
    return Path(__file__).resolve().parents[2] / "act"


def _gemm_ops_local(width, bytes_per_elem):
    sys.path.insert(0, str(_act_root()))
    from dse.src.features import _instruction_contrib
    from types import SimpleNamespace

    ins = SimpleNamespace(
        name="gemm",
        attrs={"n": 8, "_shape": (8, width)},
    )
    c = _instruction_contrib(ins, width, bytes_per_elem)
    return c["ops"], c["local"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bundle",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "artifacts"
        / "runs"
        / "20260420_matmul_64x64_chipyard_full_active_matmul_200us"
        / "pt_comparison_bundle.json",
    )
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--bytes-per-elem", type=int, default=2)
    args = ap.parse_args()

    data = json.loads(args.bundle.read_text())
    act = data.get("act") or {}
    br = act.get("instruction_breakdown_pj") or {}
    win = data.get("window") or {}
    ptm = act.get("pt_window_event_model") or {}
    saif = (ptm.get("measured_from_saif") or {}) if ptm else {}

    capture_ns = win.get("capture_ns")
    pt_e_uj = win.get("energy_uj")
    pt_p_w = win.get("total_power_w")
    saif_cyc = ptm.get("saif_capture_cycles")
    epochs = saif.get("inferred_completed_rom_epochs")

    gemm_pj = float(br.get("gemm", 0))
    ops, local = _gemm_ops_local(args.width, args.bytes_per_elem)
    macs = ops / 2.0
    pj_per_op = gemm_pj / ops if ops else float("nan")
    pj_per_mac = gemm_pj / macs if macs else float("nan")

    act_total = act.get("total_energy_uj")
    pct = act.get("pct_error_vs_pt")

    out = {
        "bundle": str(args.bundle),
        "step1_implied_gemm_energy": {
            "gemm_total_pj_from_bundle": gemm_pj,
            "assumed_gemm_ops_tensor": ops,
            "assumed_gemm_local_bytes": local,
            "implied_pj_per_tensor_op": pj_per_op,
            "implied_pj_per_mac_if_mac_is_op_over_2": pj_per_mac,
            "sanity_note": "Horowitz-scale int8 MAC is often ~0.02–0.3 pJ (node-dependent); "
            "this is a coarse flag for unit/scale mistakes, not signoff.",
            "flag_if_pj_per_mac_outside_0.001_to_10": not (0.001 <= pj_per_mac <= 10.0)
            if pj_per_mac == pj_per_mac
            else True,
        },
        "step4_window_epoch": {
            "capture_ns": capture_ns,
            "pt_window_energy_uj": pt_e_uj,
            "pt_total_power_w": pt_p_w,
            "saif_capture_cycles": saif_cyc,
            "inferred_completed_rom_epochs": epochs,
            "act_total_energy_uj_reported": act_total,
            "pct_error_vs_pt": pct,
        },
    }

    if capture_ns and pt_e_uj:
        out["step4_window_epoch"]["pt_implied_avg_power_mw"] = (
            float(pt_e_uj) / float(capture_ns) * 1e6
        )
    if saif_cyc and epochs and float(epochs) > 0:
        out["step4_window_epoch"]["saif_cycles_per_inferred_epoch"] = float(saif_cyc) / float(
            epochs
        )

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
