"""Static energy (pJ) estimation from Phase-1 .pii + realization-first mapping.

Energy flow:
1) Normalize each PII op to a primitive name.
2) Select a realization from final_mapping.json.
3) Use the realization's cost_tag to look up energy coefficients.
4) Compute energy with features._instruction_contrib op/byte proxies.

Preferred hw config schema is `cost_tags`; legacy `abstraction_classes` is still
accepted for backward compatibility.
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

_OP_ALIASES = {
    "gemm": "dot",
    "gemm_acc": "dot",
    "eltwise_add": "add",
    "vadd": "add",
    "relu": "maximum",
    "maxpool": "maximum",
    "scale": "multiply",
    "layernorm": "reduce",
    "exp": "exponential",
    "bitcvt": "bitcast_convert",
    "concat": "concatenate",
}

# Legacy class-name fallback when hw config has only abstraction_classes.
_LEGACY_CLASS_BY_COST_TAG = {
    "tensor_compute": "tensor_compute",
    "tensor_compute_tiled": "tensor_compute",
    "zero_cost": "logical_view",
    "onchip_compute": "vector_compute_add",
    "host_fallback": "special_math",
    "activation_tag": "vector_compute_add",
    "special_function": "special_math",
    "control_addressing": "logical_view",
    "onchip_movement": "contiguous_move",
    "offchip_movement": "contiguous_move",
}


def _load_cost_profiles(hw_config_path):
    data = json.loads(Path(hw_config_path).read_text())
    if "cost_tags" in data:
        return data["cost_tags"], "cost_tags"

    # Backward compatibility for pre-realization configs.
    classes = data.get("abstraction_classes")
    if not isinstance(classes, dict):
        raise KeyError(
            "hw_config must provide either `cost_tags` (preferred) or `abstraction_classes` (legacy)"
        )

    tags = {}
    for tag, class_name in _LEGACY_CLASS_BY_COST_TAG.items():
        if class_name in classes:
            tags[tag] = classes[class_name]

    if "uncategorized" in classes and "uncategorized" not in tags:
        tags["uncategorized"] = classes["uncategorized"]
    return tags, "abstraction_classes"


def _load_mapping(mapping_json_path=None):
    path = Path(mapping_json_path or _DEFAULT_MAPPING)
    if not path.is_file():
        return {"path": str(path), "primitives": {}, "fused_patterns": [], "model": None, "version": None}
    data = json.loads(path.read_text())
    return {
        "path": str(path),
        "primitives": data.get("primitives", {}),
        "fused_patterns": data.get("fused_patterns", []),
        "model": data.get("model"),
        "version": data.get("version"),
    }


def _energy_terms(class_row, ops, bytes_):
    e_op = float(class_row.get("energy_per_op_pj") or 0)
    e_b = float(class_row.get("energy_per_byte_pj") or 0)
    return ops * e_op + float(bytes_) * e_b


def _normalize_primitive_name(op_name):
    name = (op_name or "").strip().lower()
    if name.startswith("load_") or name.startswith("store_"):
        return "copy"
    if name.startswith("mov"):
        return "copy"
    return _OP_ALIASES.get(name, name)


def _select_realization(primitive_name, primitive_entry, ins):
    realizations = list(primitive_entry.get("realizations") or [])
    if not realizations:
        return None

    # Minimal heuristic: GEMM shapes larger than DIM likely use loop_ws tiling.
    # This keeps selection deterministic with currently available PII attrs.
    if primitive_name == "dot":
        shape = ins.attrs.get("_shape")
        if isinstance(shape, (list, tuple)) and any(isinstance(d, int) and d > 8 for d in shape):
            for r in realizations:
                if r.get("realization_id") == "loop_ws_tiled":
                    return r

    for r in realizations:
        if bool(r.get("is_default")):
            return r
    return realizations[0]


def _resolve_mapping(ins, mapping):
    primitive_name = _normalize_primitive_name(ins.name)
    prim = mapping["primitives"].get(primitive_name)
    if not prim:
        return {
            "primitive": primitive_name,
            "realization_id": None,
            "cost_tag": None,
            "source": "unmapped",
        }

    realization = _select_realization(primitive_name, prim, ins)
    if not realization:
        return {
            "primitive": primitive_name,
            "realization_id": None,
            "cost_tag": None,
            "source": "unmapped",
        }
    return {
        "primitive": primitive_name,
        "realization_id": realization.get("realization_id"),
        "cost_tag": realization.get("cost_tag"),
        "source": "realization_default_or_heuristic",
    }


def _metric_terms_for_cost_tag(cost_tag, contrib):
    total_hbm = contrib["hbm_read"] + contrib["hbm_write"]

    if cost_tag == "zero_cost":
        return 0.0, 0.0
    if cost_tag == "control_addressing":
        return 0.0, 0.0
    if cost_tag == "onchip_movement":
        return 0.0, contrib["local"] if contrib["local"] > 0 else total_hbm
    if cost_tag == "offchip_movement":
        return 0.0, total_hbm if total_hbm > 0 else contrib["local"]
    if cost_tag in {
        "tensor_compute",
        "tensor_compute_tiled",
        "onchip_compute",
        "host_fallback",
        "special_function",
        "activation_tag",
    }:
        return contrib["ops"], contrib["local"]
    return contrib["ops"], contrib["local"] + total_hbm


def estimate_instruction_energy(ins, width, bytes_per_elem, cost_profiles, mapping):
    """Return per-cost-tag energy and selected realization metadata for one op."""
    c = _instruction_contrib(ins, width, bytes_per_elem)
    parts = defaultdict(float)

    def add(cost_tag, ops, bytes_):
        row = cost_profiles.get(cost_tag) or cost_profiles.get("uncategorized", {})
        parts[cost_tag] += _energy_terms(row, ops, bytes_)

    resolved = _resolve_mapping(ins, mapping)

    if ins.name == "softmax":
        # PII may represent softmax as a single instruction.
        # Prefer the mapping's fused softmax tag when available.
        fused_softmax = None
        for fp in mapping.get("fused_patterns", []):
            if fp.get("name") == "softmax_fused":
                fused_softmax = fp
                break
        if fused_softmax:
            resolved = {
                "primitive": "softmax_fused",
                "realization_id": fused_softmax.get("realization_id"),
                "cost_tag": fused_softmax.get("cost_tag"),
                "source": "fused_pattern",
            }
        elif not resolved.get("cost_tag"):
            resolved = {
                "primitive": "softmax",
                "realization_id": "legacy_softmax",
                "cost_tag": "special_function",
                "source": "legacy_fallback",
            }

    if resolved.get("cost_tag"):
        ops, bytes_ = _metric_terms_for_cost_tag(resolved["cost_tag"], c)
        add(resolved["cost_tag"], ops, bytes_)
        return parts, resolved

    # Legacy fallback path for unmapped or unknown ops.
    name = (ins.name or "").lower()
    if name.startswith("load_") or name.startswith("store_"):
        add("offchip_movement", 0.0, c["hbm_read"] + c["hbm_write"])
    elif name.startswith("mov"):
        add("onchip_movement", 0.0, c["local"])
    elif name in ("gemm", "gemm_acc", "dot"):
        add("tensor_compute", c["ops"], c["local"])
    elif name in ("softmax", "layernorm", "exp", "exponential"):
        add("special_function", c["ops"], c["local"])
    elif name in ("eltwise_add", "add", "vadd", "relu", "scale", "maxpool"):
        add("onchip_compute", c["ops"], c["local"])
    elif c["known"] < 0.5:
        add("uncategorized", 0.0, c["hbm_read"] + c["hbm_write"] + c["local"])
    else:
        add("host_fallback", c["ops"], c["local"])

    return parts, {
        "primitive": _normalize_primitive_name(ins.name),
        "realization_id": "legacy_fallback",
        "cost_tag": next(iter(parts.keys()), "uncategorized"),
        "source": "legacy_fallback",
    }


def estimate_program(program, hw_config_path, mapping_json_path=None):
    width = _extract_width(program)
    bytes_per_elem = 2
    cost_profiles, profile_source = _load_cost_profiles(hw_config_path)
    mapping = _load_mapping(mapping_json_path)
    total_by_tag = defaultdict(float)
    total_by_realization = defaultdict(float)
    per_line = []

    for ins in program.instructions:
        parts, resolved = estimate_instruction_energy(
            ins, width, bytes_per_elem, cost_profiles, mapping
        )
        line_e = sum(parts.values())
        rid = resolved.get("realization_id") or "unresolved"
        per_line.append(
            {
                "op": ins.name,
                "lineno": ins.lineno,
                "primitive": resolved.get("primitive"),
                "realization_id": resolved.get("realization_id"),
                "cost_tag": resolved.get("cost_tag"),
                "resolution_source": resolved.get("source"),
                "energy_pj": line_e,
                "by_cost_tag_pj": dict(parts),
            }
        )
        for k, v in parts.items():
            total_by_tag[k] += v
        total_by_realization[rid] += line_e

    return {
        "candidate": program.path.name,
        "kernel_name": program.kernel_name,
        "total_energy_pj": sum(total_by_tag.values()),
        "by_cost_tag_pj": dict(total_by_tag),
        "by_realization_pj": dict(total_by_realization),
        "per_instruction": per_line,
        "cost_profile_source": profile_source,
        "mapping_interface": load_mapping_meta(mapping_json_path),
    }


def load_mapping_meta(mapping_json_path=None):
    """Return mapping metadata from final_mapping.json (provenance)."""
    path = Path(mapping_json_path or _DEFAULT_MAPPING)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    fused = [x.get("name") for x in (data.get("fused_patterns") or [])]
    return {
        "path": str(path),
        "version": data.get("version"),
        "model": data.get("model"),
        "num_primitives": len(data.get("primitives", {})),
        "fused_patterns": [n for n in fused if n],
    }
