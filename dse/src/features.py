"""Feature extraction for forward-bound analysis on ACT candidate programs."""


class FeatureRow(object):
    def __init__(
        self,
        candidate,
        kernel_name,
        instruction_count,
        hbm_read_bytes,
        hbm_write_bytes,
        hbm_bytes,
        local_bytes,
        compute_ops,
        intensity_ops_per_hbm_byte,
        unknown_instruction_count,
        unknown_instructions,
    ):
        self.candidate = candidate
        self.kernel_name = kernel_name
        self.instruction_count = instruction_count
        self.hbm_read_bytes = hbm_read_bytes
        self.hbm_write_bytes = hbm_write_bytes
        self.hbm_bytes = hbm_bytes
        self.local_bytes = local_bytes
        self.compute_ops = compute_ops
        self.intensity_ops_per_hbm_byte = intensity_ops_per_hbm_byte
        self.unknown_instruction_count = unknown_instruction_count
        self.unknown_instructions = unknown_instructions

    def to_dict(self):
        return {
            "candidate": self.candidate,
            "kernel_name": self.kernel_name,
            "instruction_count": self.instruction_count,
            "hbm_read_bytes": self.hbm_read_bytes,
            "hbm_write_bytes": self.hbm_write_bytes,
            "hbm_bytes": self.hbm_bytes,
            "local_bytes": self.local_bytes,
            "compute_ops": self.compute_ops,
            "intensity_ops_per_hbm_byte": self.intensity_ops_per_hbm_byte,
            "unknown_instruction_count": self.unknown_instruction_count,
            "unknown_instructions": self.unknown_instructions,
        }


def _to_float(v, default=0.0):
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except Exception:
        return default


def _extract_width(program, default_width=64):
    inputs = program.metadata.input_tensors
    if inputs:
        shape = inputs[0].get("shape")
        if isinstance(shape, (list, tuple)) and len(shape) >= 2 and isinstance(shape[1], int):
            return int(shape[1])

    # For .pii-driven runs, infer from first instruction that carries a 2D shape.
    for ins in program.instructions:
        shape = ins.attrs.get("_shape")
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            if isinstance(shape[1], int) and shape[1] > 0:
                return int(shape[1])

    return default_width


def _shape_elements(shape):
    if not isinstance(shape, (list, tuple)):
        return 0.0
    prod = 1.0
    count = 0
    for dim in shape:
        if not isinstance(dim, int):
            return 0.0
        prod *= dim
        count += 1
    return prod if count > 0 else 0.0


def _instruction_contrib(ins, width, bytes_per_elem):
    name = ins.name
    attrs = ins.attrs
    n = _to_float(attrs.get("n", attrs.get("rows", 0)), default=0.0)
    shape = attrs.get("_shape")

    hbm_read = 0.0
    hbm_write = 0.0
    local = 0.0
    ops = 0.0
    known = True

    if name.startswith("load_"):
        w = width
        if isinstance(shape, (list, tuple)) and len(shape) >= 2 and isinstance(shape[1], int):
            w = float(shape[1])
        hbm_read = n * w * bytes_per_elem
        local = n * w * bytes_per_elem
    elif name.startswith("store_"):
        w = width
        hbm_write = n * w * bytes_per_elem
        local = n * w * bytes_per_elem
    elif name == "gemm":
        m = width
        nn = width
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            if isinstance(shape[0], int) and shape[0] > 0:
                m = float(shape[0])
            if isinstance(shape[1], int) and shape[1] > 0:
                nn = float(shape[1])
        k = width
        ops = 2.0 * m * k * nn
        local = (m * k + k * nn + m * nn) * bytes_per_elem
    elif name == "softmax":
        w = width
        if isinstance(shape, (list, tuple)) and len(shape) >= 2 and isinstance(shape[1], int):
            w = float(shape[1])
        ops = n * (3.0 * w - 1.0)
        local = n * w * bytes_per_elem * 3.0
    elif name.startswith("mov"):
        w = width
        if isinstance(shape, (list, tuple)) and len(shape) >= 2 and isinstance(shape[1], int):
            w = float(shape[1])
        local = n * w * bytes_per_elem * 2.0
    elif name in set(["add", "eltwise_add", "vadd"]):
        w = width
        if isinstance(shape, (list, tuple)) and len(shape) >= 2 and isinstance(shape[1], int):
            w = float(shape[1])
        # Two local reads + one local write and one FP add per element.
        ops = n * w
        local = n * w * bytes_per_elem * 3.0
    elif name in set(["Var", "DCC", "constant", "slice", "concat", "broadcast", "reshape", "bitcvt", "copy", "dot", "divide", "exponential", "reduce", "transpose", "convert", "add"]):
        # Allowed but ignored in QKV MVP op accounting.
        known = True
    else:
        known = False

    # Optional generic HBM traffic proxy for explicit HBM-buffer nodes in PII if op is unknown.
    if not known and attrs.get("_buffer") == "HBM":
        elems = _shape_elements(shape)
        if elems > 0:
            local_guess = elems * bytes_per_elem
            hbm_write += local_guess

    return {
        "known": 1.0 if known else 0.0,
        "hbm_read": hbm_read,
        "hbm_write": hbm_write,
        "local": local,
        "ops": ops,
    }


def extract_features(program):
    width = _extract_width(program, default_width=64)
    bytes_per_elem = 2

    hbm_read = 0.0
    hbm_write = 0.0
    local = 0.0
    ops = 0.0
    unknown = []

    for ins in program.instructions:
        c = _instruction_contrib(ins, width, bytes_per_elem)
        hbm_read += c["hbm_read"]
        hbm_write += c["hbm_write"]
        local += c["local"]
        ops += c["ops"]
        if c["known"] < 0.5:
            unknown.append(ins.name)

    hbm_total = hbm_read + hbm_write
    if hbm_total > 0:
        intensity = ops / hbm_total
    else:
        intensity = float("inf") if ops > 0 else 0.0

    return FeatureRow(
        candidate=program.path.name,
        kernel_name=program.kernel_name,
        instruction_count=len(program.instructions),
        hbm_read_bytes=hbm_read,
        hbm_write_bytes=hbm_write,
        hbm_bytes=hbm_total,
        local_bytes=local,
        compute_ops=ops,
        intensity_ops_per_hbm_byte=intensity,
        unknown_instruction_count=len(unknown),
        unknown_instructions=",".join(unknown),
    )


def extract_feature_table(programs):
    return [extract_features(p) for p in programs]
