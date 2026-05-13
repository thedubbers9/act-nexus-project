"""Energy (pJ) estimation from Phase-1 .pii + realization-first mapping.

Energy flow:
1) Normalize each PII op to a primitive name.
2) Select a realization from final_mapping.json.
3) Use the realization's cost_tag to look up energy coefficients.
4) Compute energy with features._instruction_contrib op/byte proxies.

Preferred hw config schema is `cost_tags`; legacy `abstraction_classes` is still
accepted for backward compatibility.
"""

import json
import math
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
    "matmul8": "dot",
    "matmul32": "dot",
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




def _bytes_per_elem_from_metadata(program):
    """Infer element size from kernel I/O dtypes (defaults to 2 for bf16-style)."""
    for t in program.metadata.input_tensors:
        dt = str(t.get("dtype") or "").lower()
        if "int8" in dt or "s8" in dt:
            return 1
        if "int32" in dt or "s32" in dt:
            return 4
        if "bf16" in dt or "bfloat16" in dt:
            return 2
        if "fp16" in dt or "f16" in dt:
            return 2
        if "fp32" in dt or "f32" in dt:
            return 4
    return 2

def _load_cost_profiles(hw_config_path):
    data = json.loads(Path(hw_config_path).read_text())
    if "cost_tags" in data:
        return data["cost_tags"], "cost_tags", data

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
    return tags, "abstraction_classes", data


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
    if name.startswith("mvin_") or name.startswith("mvout_"):
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

    if primitive_name == "copy":
        nm = (ins.name or "").lower()
        if (
            nm.startswith("load_rm")
            or nm.startswith("store_rm")
            or nm.startswith("mvin_")
            or nm.startswith("mvout_")
        ):
            for r in realizations:
                if r.get("realization_id") == "dma_offchip":
                    return r

    if primitive_name == "add":
        # In Gemmini calibration workloads, elementwise ADD is implemented by
        # accumulator accumulate/read-write behavior, not Rocket host fallback.
        nm = (ins.name or "").lower()
        if nm in ("eltwise_add", "vadd") or ins.attrs.get("_buffer") == "ACC":
            for r in realizations:
                if r.get("realization_id") == "acc_accumulate":
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
        "compound_action_model": realization.get("compound_action_model"),
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


def _matrix_shape_for_gemmini(ins, width):
    shape = ins.attrs.get("_shape")
    m = width
    n = width
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        if isinstance(shape[0], int) and shape[0] > 0:
            m = int(shape[0])
        if isinstance(shape[1], int) and shape[1] > 0:
            n = int(shape[1])
    k = int(ins.attrs.get("k", width) or width)
    return int(m), int(n), int(k)


def _dtype_bytes(dtype, default=2):
    dt = str(dtype or "").lower()
    if "int8" in dt or "s8" in dt or "u8" in dt:
        return 1
    if "int32" in dt or "s32" in dt or "u32" in dt:
        return 4
    if "bf16" in dt or "bfloat16" in dt or "fp16" in dt or "f16" in dt:
        return 2
    if "fp32" in dt or "f32" in dt:
        return 4
    return default


def _shape_num_elements(shape):
    if not isinstance(shape, (list, tuple)):
        return 0
    prod = 1
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            return 0
        prod *= dim
    return prod


def _instruction_payload_bytes(ins, width, bytes_per_elem):
    elems = _shape_num_elements(ins.attrs.get("_shape"))
    if elems <= 0:
        rows = int(ins.attrs.get("rows", ins.attrs.get("n", 0)) or 0)
        elems = rows * width
    dtype_bytes = _dtype_bytes(ins.attrs.get("_dtype"), bytes_per_elem)
    return elems * dtype_bytes


def _gemmini_copy_tile_count(ins, width, hw_config):
    model = (hw_config or {}).get("gemmini_schedule_event_model") or {}
    if not model.get("count_dma_commands_per_tile", False):
        return 1.0

    tile_dim = int(model.get("tile_dim", 8) or 8)
    if tile_dim <= 0:
        return 1.0

    shape = ins.attrs.get("_shape")
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        m = int(shape[0]) if isinstance(shape[0], int) and shape[0] > 0 else width
        n = int(shape[1]) if isinstance(shape[1], int) and shape[1] > 0 else width
    else:
        rows = int(ins.attrs.get("rows", ins.attrs.get("n", 0)) or 0)
        m = rows if rows > 0 else width
        n = width

    return float(math.ceil(float(m) / float(tile_dim)) * math.ceil(float(n) / float(tile_dim)))


def _merge_event_counts(dst, src):
    for k, v in (src or {}).items():
        dst[k] += v


def _schedule_event_onchip_bytes(events, exclude_prefixes=None):
    """Count physical on-chip movement bytes from expanded Gemmini events.

    External HBM/DRAM bytes are deliberately excluded. PrimeTime full-chip
    reports include on-chip DMA/SPAD/ACC activity, but not external memory
    array energy, so this term models the local bytes only.
    """
    prefixes = tuple(exclude_prefixes or ())
    total = 0.0
    for key, value in (events or {}).items():
        if not key.endswith("_bytes"):
            continue
        if key.startswith("offchip_"):
            continue
        if prefixes and key.startswith(prefixes):
            continue
        total += float(value or 0.0)
    return total


def _add_gemmini_schedule_event_memory(parts, schedule_events, hw_config):
    model = (hw_config or {}).get("gemmini_schedule_event_model") or {}
    if not model.get("enabled", False):
        return None
    if not model.get("charge_onchip_memory_movement", False):
        return None

    exclude_prefixes = list(model.get("exclude_event_prefixes") or [])
    if model.get("exclude_accumulator_event_bytes", False):
        exclude_prefixes.append("acc_")
    onchip_bytes = _schedule_event_onchip_bytes(schedule_events, exclude_prefixes)
    if onchip_bytes <= 0.0:
        return None

    memory_cfg = (hw_config or {}).get("memory_energy_pj_per_byte") or {}
    sram_pj_per_byte = float(
        model.get(
            "sram_energy_per_byte_pj",
            memory_cfg.get("sram_energy_per_byte_pj", 6.85),
        )
        or 0.0
    )
    tag = str(model.get("onchip_memory_cost_tag") or "onchip_movement")
    energy_pj = onchip_bytes * sram_pj_per_byte
    parts[tag] += energy_pj
    return {
        "model": "gemmini_schedule_event_model",
        "cost_tag": tag,
        "onchip_bytes": onchip_bytes,
        "excluded_prefixes": exclude_prefixes,
        "sram_energy_per_byte_pj": sram_pj_per_byte,
        "energy_pj": energy_pj,
    }


def _event_bytes_to_tiles(events, keys, tile_bytes):
    if tile_bytes <= 0:
        return 0.0
    return sum(float((events or {}).get(key, 0.0) or 0.0) for key in keys) / float(tile_bytes)


def _rmw_event_bytes_to_tiles(events, tile_bytes):
    if tile_bytes <= 0:
        return 0.0
    read_bytes = float((events or {}).get("acc_rmw_read_bytes", 0.0) or 0.0)
    write_bytes = float((events or {}).get("acc_rmw_write_bytes", 0.0) or 0.0)
    # The ACC microbench coefficient is per read+write lane/chunk. For an RMW
    # action, read and write bytes describe the same chunks, so use the larger
    # direction instead of summing both directions and double-counting chunks.
    return max(read_bytes, write_bytes) / float(tile_bytes)


def _add_gemmini_accumulator_events(parts, schedule_events, hw_config):
    model = (hw_config or {}).get("gemmini_accumulator_event_model") or {}
    if not model.get("enabled", False):
        return None
    if not schedule_events:
        return None

    tile_bytes = float(model.get("tile_bytes", 32) or 32)
    energy_by_tile = model.get("energy_pj_per_tile") or {}
    read_pj = float(energy_by_tile.get("read", 0.0) or 0.0)
    write_pj = float(energy_by_tile.get("write", 0.0) or 0.0)
    rmw_pj = float(energy_by_tile.get("read_modify_write", 0.0) or 0.0)

    read_tiles = _event_bytes_to_tiles(
        schedule_events,
        (
            "acc_final_read_bytes",
            "acc_preload_read_bytes",
            "acc_dma_read_bytes",
            "acc_move_read_bytes",
        ),
        tile_bytes,
    )
    write_tiles = _event_bytes_to_tiles(
        schedule_events,
        (
            "acc_init_write_bytes",
            "acc_dma_write_bytes",
            "acc_move_write_bytes",
        ),
        tile_bytes,
    )
    rmw_tiles_from_bytes = _rmw_event_bytes_to_tiles(schedule_events, tile_bytes)
    rmw_tiles = max(float(schedule_events.get("acc_rmw_tiles", 0.0) or 0.0), rmw_tiles_from_bytes)

    data_energy_pj = read_tiles * read_pj + write_tiles * write_pj + rmw_tiles * rmw_pj

    envelope_model = model.get("active_envelope") or {}
    envelope_energy_pj = 0.0
    active_cycles = 0.0
    if envelope_model.get("enabled", False):
        cycles_per_read = float(envelope_model.get("cycles_per_read_tile", 1.0) or 0.0)
        cycles_per_write = float(envelope_model.get("cycles_per_write_tile", 1.0) or 0.0)
        cycles_per_rmw = float(envelope_model.get("cycles_per_read_modify_write_tile", 2.0) or 0.0)
        active_cycles = (
            read_tiles * cycles_per_read
            + write_tiles * cycles_per_write
            + rmw_tiles * cycles_per_rmw
        )
        envelope_pj_per_cycle = float(
            envelope_model.get("energy_pj_per_active_cycle", 0.0) or 0.0
        )
        envelope_energy_pj = active_cycles * envelope_pj_per_cycle

    if data_energy_pj <= 0.0 and envelope_energy_pj <= 0.0:
        return None

    tag = str(model.get("cost_tag") or "gemmini_accumulator")
    parts[tag] += data_energy_pj
    envelope_tag = str(envelope_model.get("cost_tag") or f"{tag}_envelope")
    if envelope_energy_pj > 0.0:
        parts[envelope_tag] += envelope_energy_pj
    return {
        "model": "gemmini_accumulator_event_model",
        "cost_tag": tag,
        "envelope_cost_tag": envelope_tag if envelope_energy_pj > 0.0 else None,
        "tile_bytes": tile_bytes,
        "read_tiles": read_tiles,
        "write_tiles": write_tiles,
        "read_modify_write_tiles": rmw_tiles,
        "active_envelope_cycles": active_cycles,
        "read_modify_write_tiles_from_bytes": rmw_tiles_from_bytes,
        "logical_read_modify_write_tiles": float(schedule_events.get("acc_rmw_tiles", 0.0) or 0.0),
        "energy_pj_per_tile": {
            "read": read_pj,
            "write": write_pj,
            "read_modify_write": rmw_pj,
        },
        "active_envelope_pj_per_cycle": float(
            envelope_model.get("energy_pj_per_active_cycle", 0.0) or 0.0
        ),
        "data_energy_pj": data_energy_pj,
        "active_envelope_energy_pj": envelope_energy_pj,
        "energy_pj": data_energy_pj + envelope_energy_pj,
    }


def _gemmini_copy_schedule_events(ins, width, bytes_per_elem, hw_config, resolved=None):
    name = (ins.name or "").lower()
    if not (
        name.startswith("load_")
        or name.startswith("store_")
        or name.startswith("mvin_")
        or name.startswith("mvout_")
        or name.startswith("mov")
    ):
        return None

    events = defaultdict(float)
    payload_bytes = float(_instruction_payload_bytes(ins, width, bytes_per_elem))
    command_count = _gemmini_copy_tile_count(ins, width, hw_config)
    dst_buffer = str(ins.attrs.get("_buffer") or "").upper()
    child_meta = ins.attrs.get("_child_meta") or []
    src_buffer = ""
    if child_meta:
        src_buffer = str(child_meta[0].get("buffer") or "").upper()

    if name.startswith("load_") or name.startswith("mvin_"):
        events["dma_read_commands"] += command_count
        events["offchip_read_bytes"] += payload_bytes
        if dst_buffer == "SPAD":
            events["spad_dma_write_bytes"] += payload_bytes
        elif dst_buffer == "ACC":
            events["acc_dma_write_bytes"] += payload_bytes
        else:
            events["local_dma_write_bytes"] += payload_bytes
    elif name.startswith("store_") or name.startswith("mvout_"):
        events["dma_write_commands"] += command_count
        events["offchip_write_bytes"] += payload_bytes
        if src_buffer == "SPAD":
            events["spad_dma_read_bytes"] += payload_bytes
        elif src_buffer == "ACC":
            events["acc_dma_read_bytes"] += payload_bytes
        else:
            events["local_dma_read_bytes"] += payload_bytes
    elif name.startswith("mov"):
        events["onchip_move_commands"] += 1.0
        events["onchip_move_bytes"] += payload_bytes
        if src_buffer == "ACC":
            events["acc_move_read_bytes"] += payload_bytes
        elif src_buffer == "SPAD":
            events["spad_move_read_bytes"] += payload_bytes
        if dst_buffer == "ACC":
            events["acc_move_write_bytes"] += payload_bytes
        elif dst_buffer == "SPAD":
            events["spad_move_write_bytes"] += payload_bytes

    return dict(events)


def _gemmini_compute_schedule_events(ins, width, bytes_per_elem, hw_config, resolved=None):
    name = (ins.name or "").lower()
    model = (hw_config or {}).get("gemmini_schedule_event_model") or {}
    action_model = ((resolved or {}).get("compound_action_model") or {})
    action_model_name = str(action_model.get("model") or "")
    use_loop_ws_actions = action_model_name == "gemmini_loop_ws_tiled_v3"
    use_acc_add_actions = action_model_name == "gemmini_acc_accumulate_v3"
    tile_dim = int(model.get("tile_dim", 8) or 8)
    if tile_dim <= 0:
        tile_dim = 8

    events = defaultdict(float)
    m, n, k = _matrix_shape_for_gemmini(ins, width)
    tm = int(math.ceil(float(m) / float(tile_dim)))
    tn = int(math.ceil(float(n) / float(tile_dim)))
    tk = int(math.ceil(float(k) / float(tile_dim)))
    output_tiles = tm * tn
    tile_passes = output_tiles * tk
    tile_input_bytes = float(tile_dim * tile_dim * bytes_per_elem)
    # Gemmini accumulator data is wider than input data. Use s32 unless the PII
    # explicitly uses another accumulator dtype.
    acc_bytes_per_elem = int(model.get("acc_bytes_per_elem", 4) or 4)
    tile_acc_bytes = float(tile_dim * tile_dim * acc_bytes_per_elem)

    if name in ("gemm", "gemm_acc", "dot", "matmul8", "matmul32"):
        events["tile_dim"] = float(tile_dim)
        events["m_tiles"] += float(tm)
        events["n_tiles"] += float(tn)
        events["k_tiles"] += float(tk)
        events["output_tiles"] += float(output_tiles)
        events["mesh_tile_passes"] += float(tile_passes)
        events["spad_operand_tile_reads"] += float(tile_passes * 2)
        events["spad_operand_read_bytes"] += float(tile_passes * 2) * tile_input_bytes
        events["compute_commands"] += float(tile_passes)
        events["final_output_tiles"] += float(output_tiles)
        events["acc_final_read_bytes"] += float(output_tiles) * tile_acc_bytes

        if name == "gemm_acc":
            events["acc_preload_tiles"] += float(output_tiles)
            events["acc_preload_read_bytes"] += float(output_tiles) * tile_acc_bytes
            events["acc_rmw_tiles"] += float(tile_passes)
            events["acc_rmw_read_bytes"] += float(tile_passes) * tile_acc_bytes
            events["acc_rmw_write_bytes"] += float(tile_passes) * tile_acc_bytes
        else:
            first_pass_tiles = output_tiles
            remaining_pass_tiles = output_tiles * max(tk - 1, 0)
            events["acc_init_write_tiles"] += float(first_pass_tiles)
            events["acc_init_write_bytes"] += float(first_pass_tiles) * tile_acc_bytes
            events["acc_rmw_tiles"] += float(remaining_pass_tiles)
            events["acc_rmw_read_bytes"] += float(remaining_pass_tiles) * tile_acc_bytes
            events["acc_rmw_write_bytes"] += float(remaining_pass_tiles) * tile_acc_bytes

        if use_loop_ws_actions or model.get("match_fullchip_tiled_probe_v3", False):
            # The current full-chip SAIF driver intentionally reloads operands
            # for every compute tile pass. The symbolic PII load ops model one
            # matrix load, so add the extra tiled MVIN traffic here to make ACT
            # and PT count the same command stream. This now belongs to the
            # selected realization's compound_action_model; the hw_config flag
            # remains as a compatibility fallback for older mappings.
            events["config_commands"] += float(tile_passes * 3)
            events["flush_commands"] += float(tile_passes + 1)
            operands_per_pass = 3 if name == "gemm_acc" else 2
            symbolic_operand_tiles = (tm * tk) + (tk * tn)
            if name == "gemm_acc":
                symbolic_operand_tiles += output_tiles
                events["preload_commands"] += float(tile_passes)
                events["spad_preload_read_bytes"] += float(tile_passes) * tile_input_bytes
            target_operand_tiles = tile_passes * operands_per_pass
            extra_operand_tiles = max(float(target_operand_tiles - symbolic_operand_tiles), 0.0)
            extra_operand_bytes = extra_operand_tiles * tile_input_bytes
            events["dma_read_commands"] += extra_operand_tiles
            events["offchip_read_bytes"] += extra_operand_bytes
            events["spad_dma_write_bytes"] += extra_operand_bytes
        return dict(events)

    if name in ("add", "eltwise_add", "vadd"):
        events["tile_dim"] = float(tile_dim)
        events["m_tiles"] += float(tm)
        events["n_tiles"] += float(tn)
        events["output_tiles"] += float(output_tiles)
        events["acc_add_tiles"] += float(output_tiles)
        events["acc_rmw_tiles"] += float(output_tiles)
        events["acc_rmw_read_bytes"] += float(output_tiles) * tile_acc_bytes
        events["acc_rmw_write_bytes"] += float(output_tiles) * tile_acc_bytes
        events["spad_operand_tile_reads"] += float(output_tiles)
        events["spad_operand_read_bytes"] += float(output_tiles) * tile_input_bytes
        events["compute_commands"] += float(output_tiles)
        events["final_output_tiles"] += float(output_tiles)
        if use_acc_add_actions or model.get("match_fullchip_tiled_probe_v3", False):
            events["config_commands"] += float(output_tiles * 4)
            events["flush_commands"] += float(output_tiles + 1)
        return dict(events)

    return None


def _gemmini_schedule_events_for_instruction(ins, width, bytes_per_elem, hw_config, resolved=None):
    events = _gemmini_compute_schedule_events(ins, width, bytes_per_elem, hw_config, resolved)
    if events:
        return events
    return _gemmini_copy_schedule_events(ins, width, bytes_per_elem, hw_config, resolved)


def _add_gemmini_tiled_subsystem_residuals(parts, ins, width, hw_config):
    model = (hw_config or {}).get("gemmini_tiled_matmul_subsystem_model") or {}
    if not model.get("enabled", False):
        return None

    tile_dim = int(model.get("tile_dim", 8) or 8)
    if tile_dim <= 0:
        return None

    m, n, k = _matrix_shape_for_gemmini(ins, width)
    tm = int(math.ceil(float(m) / float(tile_dim)))
    tn = int(math.ceil(float(n) / float(tile_dim)))
    tk = int(math.ceil(float(k) / float(tile_dim)))
    tile_passes = tm * tn * tk

    residuals = model.get("residual_energy_pj_per_tile_pass") or {}
    for tag, pj_per_tile_pass in residuals.items():
        parts[tag] += tile_passes * float(pj_per_tile_pass or 0.0)

    return {
        "model": "gemmini_tiled_matmul_subsystem_model",
        "tile_dim": tile_dim,
        "m": m,
        "n": n,
        "k": k,
        "tm": tm,
        "tn": tn,
        "tk": tk,
        "tile_passes": tile_passes,
    }


def estimate_instruction_energy(ins, width, bytes_per_elem, cost_profiles, mapping, hw_config=None):
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
        if resolved["cost_tag"] == "tensor_compute_tiled":
            details = _add_gemmini_tiled_subsystem_residuals(parts, ins, width, hw_config)
            if details:
                resolved["analysis_details"] = details
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
    bytes_per_elem = _bytes_per_elem_from_metadata(program)
    cost_profiles, profile_source, hw_config = _load_cost_profiles(hw_config_path)
    mapping = _load_mapping(mapping_json_path)
    total_by_tag = defaultdict(float)
    total_by_realization = defaultdict(float)
    total_schedule_events = defaultdict(float)
    per_line = []

    for ins in program.instructions:
        parts, resolved = estimate_instruction_energy(
            ins, width, bytes_per_elem, cost_profiles, mapping, hw_config
        )
        schedule_events = _gemmini_schedule_events_for_instruction(
            ins, width, bytes_per_elem, hw_config, resolved
        )
        schedule_memory_details = _add_gemmini_schedule_event_memory(
            parts, schedule_events, hw_config
        )
        accumulator_details = _add_gemmini_accumulator_events(
            parts, schedule_events, hw_config
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
        if resolved.get("analysis_details"):
            per_line[-1]["analysis_details"] = resolved["analysis_details"]
        if resolved.get("compound_action_model"):
            per_line[-1]["compound_action_model"] = resolved["compound_action_model"]
        if schedule_events:
            per_line[-1]["gemmini_schedule_events"] = schedule_events
            _merge_event_counts(total_schedule_events, schedule_events)
        if schedule_memory_details:
            per_line[-1]["gemmini_schedule_event_memory"] = schedule_memory_details
        if accumulator_details:
            per_line[-1]["gemmini_accumulator_event_energy"] = accumulator_details
        for k, v in parts.items():
            total_by_tag[k] += v
        total_by_realization[rid] += line_e

    return {
        "candidate": program.path.name,
        "kernel_name": program.kernel_name,
        "total_energy_pj": sum(total_by_tag.values()),
        "by_cost_tag_pj": dict(total_by_tag),
        "by_realization_pj": dict(total_by_realization),
        "gemmini_schedule_events": dict(total_schedule_events),
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
