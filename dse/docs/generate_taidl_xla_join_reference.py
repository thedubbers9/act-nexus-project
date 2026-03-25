#!/usr/bin/env python3
"""Join TAIDL primitive reference with broader XLA/HLO op reference."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TAIDL_CSV = ROOT / "taidl_primitive_reference.csv"
XLA_CSV = ROOT / "xla_hlo_operation_reference.csv"
OUT_CSV = ROOT / "taidl_xla_join_reference.csv"


TAIDL_TO_XLA = {
    "reshape": "Reshape",
    "convert": "ConvertElementType",
    "copy": "Copy",
    "exp": "Exp",
    "concatenate": "ConcatInDim (Concatenate)",
    "bitcast_convert": "BitcastConvertType",
    "transpose": "Transpose",
    "slice": "Slice",
    "dot": "Dot",
    "constant": "Constant",
    "broadcast": "Broadcast",
    "maximum": "Max",
    "minimum": "Min",
    "select_lt": "Select",
    "select_eq_var": "Select",
    "xor": "Xor",
    "add": "Add",
    "dynamic_update_slice": "DynamicUpdateSlice",
    "subtract": "Sub",
    "multiply": "Mul",
    "divide": "Div",
    "reduce": "Reduce",
    "parameter": "Parameter",
    "reduce_add": "Reduce",
    "exponential": "Exp",
}


ABSTRACTION_TO_RESOURCE = {
    "input_literal": ("metadata_only", "input_port_or_constant_store"),
    "layout_metadata": ("metadata_only", "logical_view_or_typecast"),
    "data_movement": ("local_move_unit", "scratchpad_dma_or_tensor_mover"),
    "data_movement_update": ("local_move_unit", "scratchpad_update_engine"),
    "tensor_compute": ("tensor_matmul_unit", "systolic_array_or_tensor_core"),
    "elementwise_alu": ("vector_alu", "simd_lane_cluster"),
    "elementwise_special": ("vector_sfu", "sfu_block"),
    "reduction": ("reduction_unit", "vector_reduce_tree"),
    "predication_select": ("vector_alu", "simd_lane_cluster_or_predicate_unit"),
    "collective": ("collective_unit", "collective_network_fabric"),
    "control_or_token": ("control_unit", "dispatcher_or_token_tracker"),
    "tuple_structural": ("metadata_only", "tuple_metadata_handler"),
    "rng": ("special_function_unit", "rng_engine"),
    "uncategorized": ("unknown_resource", "unmapped"),
}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    taidl_rows = read_csv(TAIDL_CSV)
    xla_rows = read_csv(XLA_CSV)
    xla_index = {row["xla_hlo_op"]: row for row in xla_rows}

    fieldnames = [
        "taidl_primitive",
        "taidl_status",
        "canonical_xla_op_from_taidl",
        "matched_xla_hlo_op",
        "exists_in_full_xla_csv",
        "taidl_description",
        "xla_description",
        "suggested_abstraction_group",
        "suggested_resource_class",
        "suggested_functional_unit",
        "taidl_web_source_url",
        "xla_source_url",
        "repo_source",
        "notes",
    ]

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in taidl_rows:
            primitive = row["taidl_primitive"]
            matched_xla = TAIDL_TO_XLA.get(primitive, "")
            xla_row = xla_index.get(matched_xla)
            abstraction = row["suggested_abstraction_group"]
            resource_class, functional_unit = ABSTRACTION_TO_RESOURCE.get(
                abstraction, ("unknown_resource", "unmapped")
            )

            note_parts = []
            if row["status"] != "standard_xla":
                note_parts.append("TAIDL-specific alias/helper")
            if primitive in ("select_lt", "select_eq_var"):
                note_parts.append("Maps to XLA Select plus compare logic; not a one-to-one native XLA op")
            if primitive == "reduce_add":
                note_parts.append("Maps to XLA Reduce with add combiner")
            if primitive in ("exp", "exponential"):
                note_parts.append("Both TAIDL spellings map to XLA Exp")

            writer.writerow(
                {
                    "taidl_primitive": primitive,
                    "taidl_status": row["status"],
                    "canonical_xla_op_from_taidl": row["canonical_xla_hlo_op"],
                    "matched_xla_hlo_op": matched_xla,
                    "exists_in_full_xla_csv": bool(xla_row),
                    "taidl_description": row["description"],
                    "xla_description": xla_row["description"] if xla_row else "",
                    "suggested_abstraction_group": abstraction,
                    "suggested_resource_class": resource_class,
                    "suggested_functional_unit": functional_unit,
                    "taidl_web_source_url": row["web_source_url"],
                    "xla_source_url": xla_row["source_url"] if xla_row else "",
                    "repo_source": row["repo_source"],
                    "notes": "; ".join(note_parts),
                }
            )

    print("Wrote {} rows to {}".format(len(taidl_rows), OUT_CSV))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
