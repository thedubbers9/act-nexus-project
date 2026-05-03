"""Parse ACT pre-schedule PII graph dumps into normalized instruction records."""

import re
from pathlib import Path


class InstructionCall(object):
    def __init__(self, name, attrs, lineno):
        self.name = name
        self.attrs = attrs
        self.lineno = lineno


class KernelMetadata(object):
    def __init__(self, hbm=None, input_tensors=None, output_tensors=None, constant_tensors=None):
        self.hbm = hbm
        self.input_tensors = input_tensors or []
        self.output_tensors = output_tensors or []
        self.constant_tensors = constant_tensors or []


class CandidateProgram(object):
    def __init__(self, path, kernel_name, metadata, instructions):
        self.path = path
        self.kernel_name = kernel_name
        self.metadata = metadata
        self.instructions = instructions


class ParseError(ValueError):
    """Raised when candidate PII file does not match expected ACT output format."""


_PII_RE = re.compile(
    r"^t(?P<id>\d+):\s+"
    r"(?P<buffer>[A-Za-z0-9_]+)\[(?P<offset>-?\d+)\]\s*=\s*"
    r"(?P<dtype>[A-Za-z0-9_]+)\[(?P<shape>[^\]]*)\]\s+"
    r"(?P<op>[^\(]+)\((?P<children>[^\)]*)\)\s*$"
)


def _parse_shape(shape_text):
    shape_text = shape_text.strip()
    if not shape_text:
        return []
    dims = []
    for tok in shape_text.split(','):
        tok = tok.strip()
        if not tok:
            continue
        dims.append(int(tok))
    return dims


def _coerce_attr_value(val):
    sval = val.strip().strip("\"'")
    try:
        return int(sval)
    except Exception:
        try:
            return float(sval)
        except Exception:
            return sval


def _parse_op(op_text):
    op_text = op_text.strip()
    attrs = {}

    if '[' in op_text and op_text.endswith(']'):
        name = op_text[: op_text.index('[')].strip()
        body = op_text[op_text.index('[') + 1 : -1].strip()
        if body:
            for part in body.split(','):
                part = part.strip()
                if not part:
                    continue
                if '=' in part:
                    k, v = part.split('=', 1)
                    attrs[k.strip()] = _coerce_attr_value(v)
                else:
                    attrs[part] = True
    else:
        name = op_text

    if 'rows' in attrs and 'n' not in attrs:
        attrs['n'] = attrs['rows']

    return name, attrs


def parse_pii(path):
    p = Path(path)
    instructions = []

    for lineno, line in enumerate(p.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        m = _PII_RE.match(line)
        if not m:
            raise ParseError("Failed to parse .pii line {} in {}: {}".format(lineno, p, line))

        dtype = m.group('dtype').strip()
        shape = _parse_shape(m.group('shape'))
        buffer_name = m.group('buffer').strip()
        offset = int(m.group('offset'))
        op_raw = m.group('op').strip()
        op_name, attrs = _parse_op(op_raw)

        attrs['_dtype'] = dtype
        attrs['_shape'] = shape
        attrs['_buffer'] = buffer_name
        attrs['_offset'] = offset

        instructions.append(InstructionCall(name=op_name, attrs=attrs, lineno=lineno))

    if not instructions:
        raise ParseError("No instruction lines found in {}".format(p))

    metadata = KernelMetadata(hbm=None, input_tensors=[], output_tensors=[], constant_tensors=[])
    kernel_name = p.stem
    return CandidateProgram(path=p, kernel_name=kernel_name, metadata=metadata, instructions=instructions)


def parse_pii_dir(path):
    p = Path(path)
    if p.is_file() and p.suffix == '.pii':
        return [parse_pii(p)]
    if not p.exists() or not p.is_dir():
        raise IOError("pii_dir does not exist or is not a directory: {}".format(p))

    def sort_key(x):
        stem = x.stem
        if stem.isdigit():
            return (0, "{:012d}".format(int(stem)))
        return (1, stem)

    programs = []
    for candidate in sorted(p.glob('*.pii'), key=sort_key):
        programs.append(parse_pii(candidate))

    if not programs:
        raise ParseError("No .pii candidate files found under {}".format(p))
    return programs
