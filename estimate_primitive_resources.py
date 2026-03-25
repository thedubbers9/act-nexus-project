#!/usr/bin/env python3
"""Estimate primitive-level work, bytes, energy, and configured area."""

import argparse
import ast
import csv
import json
from collections import OrderedDict
from pathlib import Path


DETAIL_FIELDS = [
    "instruction",
    "node_index",
    "id",
    "op",
    "hardware_abstraction_class",
    "configured_resource_class",
    "implementation",
    "source_op",
    "inputs",
    "shape",
    "dtype",
    "attrs",
    "compute_ops_formula",
    "read_bytes_formula",
    "write_bytes_formula",
    "energy_pj_formula",
    "configured_unit_count",
    "configured_area_per_unit_mm2",
    "configured_total_area_mm2",
    "configured_throughput_ops_per_cycle",
    "configured_bandwidth_bytes_per_cycle",
]


SUMMARY_FIELDS = [
    "instruction",
    "compute_ops_formula",
    "read_bytes_formula",
    "write_bytes_formula",
    "energy_pj_formula",
    "configured_area_mm2",
    "abstractions_used",
    "resource_classes_used",
    "implementations_used",
]


def _load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def _safe_eval_numeric(expr: str):
    if expr in ("", None):
        return None
    try:
        node = ast.parse(str(expr), mode="eval")
    except Exception:
        return None

    def _eval(cur):
        if isinstance(cur, ast.Expression):
            return _eval(cur.body)
        if isinstance(cur, ast.Constant) and isinstance(cur.value, (int, float)):
            return float(cur.value)
        if isinstance(cur, ast.Num):
            return float(cur.n)
        if isinstance(cur, ast.UnaryOp) and isinstance(cur.op, (ast.UAdd, ast.USub)):
            value = _eval(cur.operand)
            return value if isinstance(cur.op, ast.UAdd) else -value
        if isinstance(cur, ast.BinOp):
            lhs = _eval(cur.left)
            rhs = _eval(cur.right)
            if isinstance(cur.op, ast.Add):
                return lhs + rhs
            if isinstance(cur.op, ast.Sub):
                return lhs - rhs
            if isinstance(cur.op, ast.Mult):
                return lhs * rhs
            if isinstance(cur.op, ast.Div):
                return lhs / rhs
        raise ValueError("non-numeric expression")

    try:
        return _eval(node)
    except Exception:
        return None


def _is_number(value):
    try:
        float(value)
        return True
    except Exception:
        return False


def _fmt_num(value):
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _mul_terms(terms):
    numeric = 1.0
    symbolic = []
    for term in terms:
        if term in ("", None):
            continue
        if isinstance(term, (int, float)):
            numeric *= float(term)
        elif _is_number(term):
            numeric *= float(term)
        else:
            text = str(term).strip()
            if any(op in text for op in (" + ", " - ", " / ")) and not (
                text.startswith("(") and text.endswith(")")
            ):
                text = f"({text})"
            symbolic.append(text)

    if numeric == 0.0:
        return "0"

    pieces = []
    if numeric != 1.0 or not symbolic:
        pieces.append(_fmt_num(numeric))
    pieces.extend(symbolic)
    return " * ".join(pieces) if pieces else "0"


def _add_terms(terms):
    clean = [term for term in terms if term not in ("", None, "0", 0, 0.0)]
    if not clean:
        return "0"
    return " + ".join(str(term) for term in clean)


def _sub_terms(lhs, rhs):
    if rhs in ("0", 0, 0.0, None, ""):
        return lhs
    if lhs in ("0", 0, 0.0, None, ""):
        return f"-({rhs})"
    return f"({lhs}) - ({rhs})"


def _shape_elements(shape):
    if not shape:
        return "1"
    return _mul_terms(shape)


def _bytes_formula(shape, bytes_per_elem):
    return _mul_terms([_shape_elements(shape), bytes_per_elem])


def _class_cfg(hw_cfg, abstraction_class):
    return (hw_cfg.get("abstraction_classes") or {}).get(abstraction_class, {})


def _node_map(nodes):
    return {node.get("id"): node for node in nodes}


def _input_node(node, node_lookup, index):
    inputs = node.get("inputs") or []
    if index >= len(inputs):
        return None
    return node_lookup.get(inputs[index])


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


def _node_abstraction(node):
    if node.get("abstraction_class"):
        return node["abstraction_class"]
    op = node.get("op", "")
    if op in _LEGACY_OP_TO_ABSTRACTION:
        return _LEGACY_OP_TO_ABSTRACTION[op]
    return node.get("resource_class", "uncategorized")


def _estimate_node(node, node_lookup, hw_cfg):
    abstraction_class = _node_abstraction(node)
    cfg = _class_cfg(hw_cfg, abstraction_class)
    dtype = node.get("dtype", "")
    bytes_per_elem = (hw_cfg.get("bytes_per_dtype") or {}).get(dtype, 0)
    shape = node.get("shape") or []
    op = node.get("op", "")

    compute_ops = "0"
    read_bytes = "0"
    write_bytes = "0"

    if op in ("input", "constant", "reshape", "bitcast", "convert"):
        pass
    elif op in ("copy", "transpose", "slice", "concat", "dynamic_update_slice"):
        moved = _bytes_formula(shape, bytes_per_elem)
        read_bytes = moved
        write_bytes = moved
    elif op == "broadcast":
        if cfg.get("materialize_outputs", False):
            moved = _bytes_formula(shape, bytes_per_elem)
            read_bytes = moved
            write_bytes = moved
    elif op in (
        "ewise_exp",
        "ewise_add",
        "ewise_sub",
        "ewise_mul",
        "ewise_div",
        "ewise_max",
        "ewise_min",
        "ewise_xor",
        "select_lt",
        "select_eq_var",
    ):
        out_elems = _shape_elements(shape)
        compute_ops = out_elems
        input_count = max(len(node.get("inputs") or []), 1)
        read_bytes = _mul_terms([input_count, out_elems, bytes_per_elem])
        write_bytes = _bytes_formula(shape, bytes_per_elem)
    elif op in ("reduce_sum", "reduce_generic"):
        input0 = _input_node(node, node_lookup, 0)
        input_shape = input0.get("shape", []) if input0 else shape
        input_elems = _shape_elements(input_shape)
        output_elems = _shape_elements(shape)
        compute_ops = _sub_terms(input_elems, output_elems)
        read_bytes = _bytes_formula(input_shape, bytes_per_elem)
        write_bytes = _bytes_formula(shape, bytes_per_elem)
    elif op == "matmul":
        input0 = _input_node(node, node_lookup, 0)
        input1 = _input_node(node, node_lookup, 1)
        a_shape = input0.get("shape", []) if input0 else []
        b_shape = input1.get("shape", []) if input1 else []
        c_shape = shape
        if len(c_shape) >= 2:
            m_dim = c_shape[0]
            n_dim = c_shape[1]
        else:
            m_dim = "M"
            n_dim = "N"
        if len(a_shape) >= 2:
            k_dim = a_shape[1]
        elif len(b_shape) >= 1:
            k_dim = b_shape[0]
        else:
            k_dim = "K"
        compute_ops = _mul_terms([2, m_dim, k_dim, n_dim])
        read_terms = []
        if a_shape:
            read_terms.append(_bytes_formula(a_shape, bytes_per_elem))
        if b_shape:
            read_terms.append(_bytes_formula(b_shape, bytes_per_elem))
        read_bytes = _add_terms(read_terms)
        write_bytes = _bytes_formula(c_shape, bytes_per_elem)

    energy_terms = []
    energy_per_op = cfg.get("energy_per_op_pj")
    energy_per_byte = cfg.get("energy_per_byte_pj")
    if energy_per_op not in (None, 0, 0.0, "0"):
        energy_terms.append(_mul_terms([compute_ops, energy_per_op]))
    moved_total = _add_terms([read_bytes, write_bytes])
    if energy_per_byte not in (None, 0, 0.0, "0"):
        energy_terms.append(_mul_terms([moved_total, energy_per_byte]))
    energy_formula = _add_terms(energy_terms)

    unit_count = cfg.get("unit_count", 0)
    area_per_unit = cfg.get("area_per_unit_mm2", 0.0)
    total_area = unit_count * area_per_unit
    implementation = cfg.get("implementation", abstraction_class)
    resource_class = cfg.get("resource_class", abstraction_class)
    throughput_ops = cfg.get("throughput_ops_per_cycle", "")
    bandwidth_bytes = cfg.get("bandwidth_bytes_per_cycle", "")

    return {
        "hardware_abstraction_class": abstraction_class,
        "configured_resource_class": resource_class,
        "implementation": implementation,
        "compute_ops_formula": compute_ops,
        "read_bytes_formula": read_bytes,
        "write_bytes_formula": write_bytes,
        "energy_pj_formula": energy_formula,
        "configured_unit_count": unit_count,
        "configured_area_per_unit_mm2": area_per_unit,
        "configured_total_area_mm2": total_area,
        "configured_throughput_ops_per_cycle": throughput_ops,
        "configured_bandwidth_bytes_per_cycle": bandwidth_bytes,
    }


def _write_detail_csv(nodes_by_instruction, hw_cfg, out_path: Path):
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        writer.writeheader()

        for instruction, nodes in nodes_by_instruction.items():
            lookup = _node_map(nodes)
            for node_index, node in enumerate(nodes):
                est = _estimate_node(node, lookup, hw_cfg)
                writer.writerow(
                    {
                        "instruction": instruction,
                        "node_index": node_index,
                        "id": node.get("id", ""),
                        "op": node.get("op", ""),
                        "hardware_abstraction_class": est["hardware_abstraction_class"],
                        "configured_resource_class": est["configured_resource_class"],
                        "implementation": est["implementation"],
                        "source_op": node.get("source_op", ""),
                        "inputs": "|".join(node.get("inputs", [])),
                        "shape": "|".join(node.get("shape", [])),
                        "dtype": node.get("dtype", ""),
                        "attrs": json.dumps(node.get("attrs", {}), sort_keys=True),
                        **est,
                    }
                )


def _collect_instruction_summaries(nodes_by_instruction, hw_cfg):
    summaries = OrderedDict()

    for instruction, nodes in nodes_by_instruction.items():
        lookup = _node_map(nodes)
        compute_terms = []
        read_terms = []
        write_terms = []
        energy_terms = []
        abstraction_classes = OrderedDict()
        resource_classes = OrderedDict()
        implementations = OrderedDict()

        for node in nodes:
            est = _estimate_node(node, lookup, hw_cfg)
            compute_terms.append(est["compute_ops_formula"])
            read_terms.append(est["read_bytes_formula"])
            write_terms.append(est["write_bytes_formula"])
            energy_terms.append(est["energy_pj_formula"])
            ac = est["hardware_abstraction_class"]
            if ac not in abstraction_classes:
                abstraction_classes[ac] = est["configured_total_area_mm2"]
            rc = est["configured_resource_class"]
            if rc not in resource_classes:
                resource_classes[rc] = True
            impl = est["implementation"]
            if impl not in implementations:
                implementations[impl] = True

        energy_formula = _add_terms(energy_terms)
        summaries[instruction] = {
            "compute_ops_formula": _add_terms(compute_terms),
            "read_bytes_formula": _add_terms(read_terms),
            "write_bytes_formula": _add_terms(write_terms),
            "energy_pj_formula": energy_formula,
            "energy_pj_if_numeric": _safe_eval_numeric(energy_formula),
            "configured_area_mm2": sum(abstraction_classes.values()),
            "abstractions_used": list(abstraction_classes.keys()),
            "resource_classes_used": list(resource_classes.keys()),
            "implementations_used": list(implementations.keys()),
        }

    return summaries


def _write_summary_csv(nodes_by_instruction, hw_cfg, out_path: Path):
    summaries = _collect_instruction_summaries(nodes_by_instruction, hw_cfg)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for instruction, summary in summaries.items():
            writer.writerow(
                {
                    "instruction": instruction,
                    "compute_ops_formula": summary["compute_ops_formula"],
                    "read_bytes_formula": summary["read_bytes_formula"],
                    "write_bytes_formula": summary["write_bytes_formula"],
                    "energy_pj_formula": summary["energy_pj_formula"],
                    "configured_area_mm2": summary["configured_area_mm2"],
                    "abstractions_used": "|".join(summary["abstractions_used"]),
                    "resource_classes_used": "|".join(summary["resource_classes_used"]),
                    "implementations_used": "|".join(summary["implementations_used"]),
                }
            )


def _write_summary_json(nodes_by_instruction, hw_cfg, out_path: Path):
    with out_path.open("w") as f:
        json.dump(_collect_instruction_summaries(nodes_by_instruction, hw_cfg), f, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to taidl_primitive_nodes.json")
    parser.add_argument("--hw_config", required=True, help="Path to primitive_hw_config.json")
    parser.add_argument("--detail_csv", required=True, help="Output path for node-level estimate CSV")
    parser.add_argument("--summary_csv", required=True, help="Output path for instruction summary estimate CSV")
    parser.add_argument("--summary_json", help="Optional output path for instruction summary JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    hw_config_path = Path(args.hw_config)
    detail_csv = Path(args.detail_csv)
    summary_csv = Path(args.summary_csv)
    summary_json = Path(args.summary_json) if args.summary_json else None

    nodes_by_instruction = _load_json(input_path)
    hw_cfg = _load_json(hw_config_path)

    detail_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)

    _write_detail_csv(nodes_by_instruction, hw_cfg, detail_csv)
    _write_summary_csv(nodes_by_instruction, hw_cfg, summary_csv)
    if summary_json:
        _write_summary_json(nodes_by_instruction, hw_cfg, summary_json)

    print(f"Wrote primitive estimate detail CSV: {detail_csv}")
    print(f"Wrote primitive estimate summary CSV: {summary_csv}")
    if summary_json:
        print(f"Wrote instruction cost summary JSON: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
