#!/usr/bin/env python3
"""ACT vs PT bucket comparison v1: split tensor_compute into op vs byte terms (mesh vs scratchpad).

Uses the same op/local accounting as ``act/dse/src/features._instruction_contrib``
and tensor_compute rates from ``primitive_hw_config_micro.json``.
Per-instruction energies default to pt_comparison_bundle.json ``act.instruction_breakdown_pj``;
the gemm instruction is split between PT buckets using the op:byte ratio from
``primitive_hw_config_micro.json`` tensor_compute (energy_per_op_pj / energy_per_byte_pj).

Run from anywhere:
  python3 act_pt_term_bucket_v1.py [--bundle PATH] [--hw PATH]

Writes JSON next to the bundle: ``<bundle_stem>_act_pt_v1_term_buckets.json``
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _repo_act_root() -> Path:
    return Path(__file__).resolve().parents[2] / "act"


def _import_act_dse():
    act = _repo_act_root()
    sys.path.insert(0, str(act))
    from dse.src.energy_estimate import _load_abstraction_classes
    from dse.src.features import _instruction_contrib

    return _load_abstraction_classes, _instruction_contrib


def _pt_buckets_from_act_epoch(
    bundle: dict,
    classes: dict,
    _instruction_contrib,
    width: int,
    bytes_per_elem: int,
) -> dict:
    """Return per-bucket pJ for one logical epoch (ACT proxy → PT names)."""
    act = bundle.get("act") or {}
    br = act.get("instruction_breakdown_pj") or {}
    if not br:
        raise ValueError("bundle.act.instruction_breakdown_pj missing")

    # Instruction sequence must match workload_py kernel (see mapping_note).
    seq = [
        ("load_rm", {"n": 8, "_shape": (8, width)}),
        ("load_cm", {"n": 8, "_shape": (8, width)}),
        ("gemm", {"n": 8, "_shape": (8, width)}),
        ("mov", {"n": 8, "_shape": (8, width)}),
        ("store_rm", {"n": 8, "_shape": (8, width)}),
    ]
    for name, _ in seq:
        if name not in br:
            raise KeyError(f"instruction_breakdown_pj missing {name}")

    mesh = 0.0
    scratchpad = 0.0
    controller_dma = 0.0
    glue = 0.0
    accumulator = 0.0

    tc_row = classes.get("tensor_compute") or classes["uncategorized"]

    for name, attrs in seq:
        ins = SimpleNamespace(name=name, attrs=attrs)
        E = float(br[name])
        c = _instruction_contrib(ins, width, bytes_per_elem)

        if name.startswith("load_") or name.startswith("store_"):
            controller_dma += E
            continue
        if name.startswith("mov"):
            scratchpad += E
            continue
        if name in ("gemm", "gemm_acc"):
            e_op = float(c["ops"]) * float(tc_row.get("energy_per_op_pj") or 0)
            e_b = float(c["local"]) * float(tc_row.get("energy_per_byte_pj") or 0)
            denom = e_op + e_b
            if denom <= 0:
                mesh += E
                continue
            mesh += E * (e_op / denom)
            scratchpad += E * (e_b / denom)
            continue

        glue += E

    return {
        "mesh_pj": mesh,
        "scratchpad_pj": scratchpad,
        "controller_dma_pj": controller_dma,
        "glue_pj": glue,
        "accumulator_pj": accumulator,
    }


def main() -> None:
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
        help="pt_comparison_bundle.json",
    )
    ap.add_argument(
        "--hw",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "primitive_hw_config_micro.json",
        help="primitive_hw_config_micro.json (must match bundle.act.hw_config)",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=64,
        help="Gemmini K-dimension / operand width for features._instruction_contrib (default 64 for tile8 OS path in bundle).",
    )
    ap.add_argument(
        "--bytes-per-elem",
        type=int,
        default=2,
        help="bf16 = 2",
    )
    args = ap.parse_args()

    _load_abstraction_classes, _instruction_contrib = _import_act_dse()
    bundle = json.loads(args.bundle.read_text())
    classes = _load_abstraction_classes(str(args.hw))

    act_meta = bundle.get("act") or {}
    epochs = (act_meta.get("pt_window_event_model") or {}).get("measured_from_saif") or {}
    epoch_count = float(epochs.get("inferred_completed_rom_epochs") or act_meta.get("epoch_count") or 0)
    if epoch_count <= 0:
        raise ValueError("Could not infer epoch_count from bundle")

    one_epoch = _pt_buckets_from_act_epoch(
        bundle, classes, _instruction_contrib, args.width, args.bytes_per_elem
    )

    pt_uj = bundle.get("grouped_buckets_uj") or {}
    window_ns = (bundle.get("window") or {}).get("capture_ns")

    out = {
        "schema": "act_pt_v1_term_bucket_report_v1",
        "bundle": str(args.bundle),
        "hw_config": str(args.hw),
        "assumptions": {
            "width_for_features": args.width,
            "bytes_per_elem": args.bytes_per_elem,
            "gemm_split": "tensor_compute energy split: ops*energy_per_op_pj → mesh; local_bytes*energy_per_byte_pj → scratchpad; ratios applied to bundle gemm pJ total.",
            "dma_instructions": "load_rm, load_cm, store_rm → controller_dma; mov → scratchpad (contiguous_move local).",
            "epoch_scaling": "window_act_bucket_uj = one_epoch_bucket_pj * epoch_count / 1e6",
        },
        "window": {
            "capture_ns": window_ns,
            "epoch_count_open_loop": epoch_count,
            "saif_capture_cycles": (act_meta.get("pt_window_event_model") or {}).get("saif_capture_cycles"),
        },
        "act_one_epoch_bucket_pj": one_epoch,
        "act_window_bucket_uj": {
            k.replace("_pj", ""): one_epoch[k] * epoch_count / 1e6 for k in one_epoch
        },
        "pt_window_bucket_uj": {b: pt_uj.get(b) for b in ("mesh", "scratchpad", "controller_dma", "glue", "accumulator")},
        "ratio_act_over_pt_window": {},
    }

    aw = out["act_window_bucket_uj"]
    for b in ("mesh", "scratchpad", "controller_dma", "glue", "accumulator"):
        ptv = pt_uj.get(b)
        av = aw.get(b)
        if ptv is None or av is None:
            out["ratio_act_over_pt_window"][b] = None
        elif float(ptv) == 0.0:
            out["ratio_act_over_pt_window"][b] = None if float(av) == 0.0 else float("inf")
        else:
            out["ratio_act_over_pt_window"][b] = float(av) / float(ptv)

    out_path = args.bundle.with_name(args.bundle.stem + "_act_pt_v1_term_buckets.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
