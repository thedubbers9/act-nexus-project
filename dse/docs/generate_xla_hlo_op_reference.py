#!/usr/bin/env python3
"""Generate a sourced CSV of XLA/HLO ops from OpenXLA operation semantics."""

import csv
import html
import re
from pathlib import Path

import requests


XLA_OP_SEMANTICS_URL = "https://openxla.org/xla/operation_semantics"
TAIDL_PAPER_URL = "https://act-compiler.github.io/assets/pdf/taidl-micro25.pdf"


def infer_category(op_name):
    collectives = {
        "AllGather",
        "AllReduce",
        "AllToAll",
        "CollectiveBroadcast",
        "CollectivePermute",
        "ReduceScatter",
        "PartitionID",
        "ReplicaId",
    }
    control = {
        "AddDependency",
        "AfterAll",
        "Async",
        "Call",
        "Conditional",
        "CustomCall",
        "Domain",
        "Fusion",
        "Map",
        "OptimizationBarrier",
        "While",
    }
    io_token = {"Infeed", "Outfeed", "Recv", "Send"}
    tuple_structural = {"Tuple", "GetTupleElement"}
    rng = {"RngNormal", "RngUniform", "RngBitGenerator", "RngGetAndUpdateState"}
    normalization = {"BatchNormGrad", "BatchNormInference", "BatchNormTraining"}
    linear_algebra = {
        "Cholesky",
        "Conv (Convolution)",
        "Dot",
        "Fft",
        "TriangularSolve",
    }
    reduction = {"Reduce", "ReducePrecision", "ReduceWindow", "Scan", "Sort", "TopK"}
    predication = {"Compare", "Select", "SelectAndScatter", "Clamp"}
    movement = {
        "Broadcast",
        "ConcatInDim (Concatenate)",
        "Copy",
        "DynamicSlice",
        "DynamicUpdateSlice",
        "Gather",
        "Pad",
        "Reshape",
        "Rev (reverse)",
        "Scatter",
        "Slice",
        "Transpose",
    }
    layout = {
        "Bitcast",
        "BitcastConvertType",
        "Collapse",
        "ConvertElementType",
        "DynamicReshape",
        "GetDimensionSize",
        "SetDimensionSize",
        "Iota",
    }
    unary_math = {
        "Abs",
        "Cbrt",
        "Ceil",
        "Clz",
        "Cos",
        "Cosh",
        "Erf",
        "Exp",
        "Expm1",
        "Floor",
        "Imag",
        "IsFinite",
        "Log",
        "Log1p",
        "Logistic",
        "Neg",
        "Not",
        "PopulationCount",
        "Real",
        "Round",
        "Rsqrt",
        "Sign",
        "Sin",
        "Sqrt",
        "Tan",
        "Tanh",
    }
    binary_alu = {
        "Add",
        "And",
        "Atan2",
        "Complex",
        "Div",
        "Max",
        "Min",
        "Mul",
        "Or",
        "Pow",
        "Rem",
        "ShiftLeft",
        "ShiftRightArithmetic",
        "ShiftRightLogical",
        "Sub",
        "Xor",
    }
    literals = {"Constant", "Parameter"}

    if op_name in collectives:
        return "collective_communication", "collective"
    if op_name in control:
        return "control_token_async", "control_or_token"
    if op_name in io_token:
        return "io_and_transfer", "io_or_transfer"
    if op_name in tuple_structural:
        return "tuple_structural", "tuple_structural"
    if op_name in rng:
        return "random_generation", "rng"
    if op_name in normalization:
        return "normalization", "specialized_compute"
    if op_name in linear_algebra:
        return "linear_algebra", "tensor_compute"
    if op_name in reduction:
        return "reduction_sort", "reduction"
    if op_name in predication:
        return "predication_selection", "predication_select"
    if op_name in movement:
        return "tensor_movement", "data_movement"
    if op_name in layout:
        return "layout_shape_type", "layout_metadata"
    if op_name in unary_math:
        return "elementwise_unary", "elementwise_special"
    if op_name in binary_alu:
        return "elementwise_binary", "elementwise_alu"
    if op_name in literals:
        return "literals_and_inputs", "input_literal"
    return "uncategorized", "uncategorized"


def strip_tags(text):
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(text).split())


def is_signature_like(text):
    return bool(re.fullmatch(r"[A-Za-z0-9_ ,./=:-]+(?:\([^)]*\))?", text)) and "(" in text and ")" in text


def extract_sections(page_html):
    pattern = re.compile(
        r'<h2[^>]*id="([^"]+)"[^>]*>(.*?)</h2>(.*?)(?=<h2[^>]*id="[^"]+"|$)',
        re.S,
    )
    return pattern.findall(page_html)


def first_description(section_html):
    paragraphs = [
        strip_tags(match)
        for match in re.findall(r"<p[^>]*>(.*?)</p>", section_html, re.S)
    ]
    for text in paragraphs:
        if not text or text.startswith("See also"):
            continue
        if is_signature_like(text):
            continue
        return text
    return paragraphs[0] if paragraphs else ""


def main():
    out_path = Path(__file__).with_name("xla_hlo_operation_reference.csv")
    html_text = requests.get(XLA_OP_SEMANTICS_URL, timeout=30).text
    sections = extract_sections(html_text)

    fieldnames = [
        "xla_hlo_op",
        "anchor_id",
        "category",
        "suggested_abstraction_group",
        "description",
        "source_url",
        "source_page",
        "notes",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for anchor_id, title_html, section_html in sections:
            op_name = strip_tags(title_html)
            category, abstraction = infer_category(op_name)
            writer.writerow(
                {
                    "xla_hlo_op": op_name,
                    "anchor_id": anchor_id,
                    "category": category,
                    "suggested_abstraction_group": abstraction,
                    "description": first_description(section_html),
                    "source_url": "{}#{}".format(XLA_OP_SEMANTICS_URL, anchor_id),
                    "source_page": "OpenXLA operation semantics",
                    "notes": "Current official XLA op section. TAIDL paper states XLA-HLO supports 120+ ops: {}".format(
                        TAIDL_PAPER_URL
                    ),
                }
            )

    print("Wrote {} rows to {}".format(len(sections), out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
