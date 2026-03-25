#!/usr/bin/env python3
"""Export normalized primitive-node JSON to CSV files."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DETAIL_FIELDS = [
    "instruction",
    "node_index",
    "id",
    "op",
    "abstraction_class",
    "source_op",
    "inputs",
    "shape",
    "dtype",
    "attrs",
    "is_root",
]


SUMMARY_FIELDS = [
    "instruction",
    "abstraction_class",
    "primitive_op",
    "count",
]


def _load_nodes(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


_LEGACY_OP_TO_ABSTRACTION = {
    "input": "input_literal",
    "constant": "input_literal",
    "reshape": "logical_view",
    "bitcast": "logical_view",
    "convert": "logical_view",
    "broadcast": "logical_view",
    "transpose": "contiguous_move",
    "copy": "contiguous_move",
    "slice": "contiguous_move",
    "concat": "contiguous_move",
    "dynamic_update_slice": "contiguous_move",
    "matmul": "tensor_compute",
    "ewise_exp": "special_math",
    "reduce_sum": "reduction",
    "reduce_generic": "reduction",
    "ewise_div": "vector_compute",
    "ewise_add": "vector_compute",
    "ewise_sub": "vector_compute",
    "ewise_mul": "vector_compute",
    "ewise_max": "vector_compute",
    "ewise_min": "vector_compute",
    "ewise_xor": "vector_compute",
    "select_lt": "predication_select",
    "select_eq_var": "predication_select",
}


def _node_abstraction(node: dict) -> str:
    if node.get("abstraction_class"):
        return node["abstraction_class"]
    op = node.get("op", "")
    if op in _LEGACY_OP_TO_ABSTRACTION:
        return _LEGACY_OP_TO_ABSTRACTION[op]
    return node.get("resource_class", "")


def _write_detail_csv(nodes_by_instruction: dict, out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        writer.writeheader()

        for instruction, nodes in nodes_by_instruction.items():
            for node_index, node in enumerate(nodes):
                writer.writerow(
                    {
                        "instruction": instruction,
                        "node_index": node_index,
                        "id": node.get("id", ""),
                        "op": node.get("op", ""),
                        "abstraction_class": _node_abstraction(node),
                        "source_op": node.get("source_op", ""),
                        "inputs": "|".join(node.get("inputs", [])),
                        "shape": "|".join(node.get("shape", [])),
                        "dtype": node.get("dtype", ""),
                        "attrs": json.dumps(node.get("attrs", {}), sort_keys=True),
                        "is_root": node.get("is_root", False),
                    }
                )


def _write_summary_csv(nodes_by_instruction: dict, out_path: Path) -> None:
    grouped = defaultdict(Counter)

    for instruction, nodes in nodes_by_instruction.items():
        for node in nodes:
            key = (_node_abstraction(node), node.get("op", ""))
            grouped[instruction][key] += 1

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        for instruction in sorted(grouped.keys()):
            for (abstraction_class, primitive_op), count in sorted(grouped[instruction].items()):
                writer.writerow(
                    {
                        "instruction": instruction,
                        "abstraction_class": abstraction_class,
                        "primitive_op": primitive_op,
                        "count": count,
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to taidl_primitive_nodes.json")
    parser.add_argument("--detail_csv", required=True, help="Output path for detailed node CSV")
    parser.add_argument("--summary_csv", required=True, help="Output path for summary primitive CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    detail_csv = Path(args.detail_csv)
    summary_csv = Path(args.summary_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Primitive JSON not found: {input_path}")

    detail_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    nodes_by_instruction = _load_nodes(input_path)
    _write_detail_csv(nodes_by_instruction, detail_csv)
    _write_summary_csv(nodes_by_instruction, summary_csv)

    print(f"Wrote detailed primitive CSV: {detail_csv}")
    print(f"Wrote primitive summary CSV: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
