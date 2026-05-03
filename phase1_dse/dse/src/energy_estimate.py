"""Static energy (pJ) estimation from Phase-1 .pii and primitive_hw_config.json.

Maps each TAIDL/ISA op in the trace to abstraction classes in
``dse/config/primitive_hw_config.json``, using the same byte/op proxies as
``features._instruction_contrib``. Fused TAIDL instructions (e.g. ``softmax``)
are charged as one block, consistent with ``fused_patterns`` in
``hardware_interface/hardware_mapping_interface_package/final_mapping.json`` (softmax_fused).
"""

import json
from collections import defaultdict
from pathlib import Path

from .features import _extract_width, _instruction_contrib

_DEFAULT_MAPPING = (
    Path(__file__).resolve().parents[1]
    / "hardware_interface"
    / "hardware_mapping_interface_package"
    / "final_mapping.json"
)


def _load_abstraction_classes(hw_config_path):
    data = json.loads(Path(hw_config_path).read_text())
    return data["abstraction_classes"]


def _energy_terms(class_row, ops, bytes_):
    e_op = float(class_row.get("energy_per_op_pj") or 0)
    e_b = float(class_row.get("energy_per_byte_pj") or 0)
    return ops * e_op + float(bytes_) * e_b


def estimate_instruction_energy(ins, width, bytes_per_elem, classes):
    """Return defaultdict abstraction_class -> energy (pJ) for one PII op."""
    c = _instruction_contrib(ins, width, bytes_per_elem)
    name = ins.name
    parts = defaultdict(float)

    def add(cls_name, ops, bytes_):
        row = classes.get(cls_name) or classes["uncategorized"]
        parts[cls_name] += _energy_terms(row, ops, bytes_)

    if name.startswith("load_") or name.startswith("store_"):
        b = c["hbm_read"] + c["hbm_write"]
        add("contiguous_move", 0.0, b)
        return parts

    if name.startswith("mov"):
        add("contiguous_move", 0.0, c["local"])
        return parts

    if name in ("gemm", "gemm_acc"):
        add("tensor_compute", c["ops"], c["local"])
        return parts

    if name == "softmax":
        add("special_math", c["ops"], c["local"])
        return parts

    if name in ("eltwise_add", "add", "vadd", "relu", "scale", "maxpool"):
        add("vector_compute_add", c["ops"], c["local"])
        return parts

    if name == "layernorm":
        add("special_math", c["ops"], c["local"])
        return parts

    if c["known"] < 0.5:
        b = c["hbm_read"] + c["hbm_write"] + c["local"]
        add("uncategorized", 0.0, b)
        return parts

    return parts


def estimate_program(program, hw_config_path):
    width = _extract_width(program)
    bytes_per_elem = 2
    classes = _load_abstraction_classes(hw_config_path)
    total_by_class = defaultdict(float)
    per_line = []

    for ins in program.instructions:
        parts = estimate_instruction_energy(ins, width, bytes_per_elem, classes)
        line_e = sum(parts.values())
        per_line.append(
            {
                "op": ins.name,
                "lineno": ins.lineno,
                "energy_pj": line_e,
                "by_class_pj": dict(parts),
            }
        )
        for k, v in parts.items():
            total_by_class[k] += v

    return {
        "candidate": program.path.name,
        "kernel_name": program.kernel_name,
        "total_energy_pj": sum(total_by_class.values()),
        "by_abstraction_class_pj": dict(total_by_class),
        "per_instruction": per_line,
    }


def load_mapping_meta(mapping_json_path=None):
    """Return version and fused-pattern names from final_mapping.json (provenance)."""
    path = Path(mapping_json_path or _DEFAULT_MAPPING)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    fused = [x.get("name") for x in (data.get("fused_patterns") or [])]
    return {
        "path": str(path),
        "version": data.get("version"),
        "model": data.get("model"),
        "fused_patterns": [n for n in fused if n],
    }
