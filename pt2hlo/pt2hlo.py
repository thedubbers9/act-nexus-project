#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, List, Sequence, Tuple
import re

import torch
import torch.nn as nn


DTYPE_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "long": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

DEFAULT_ISA_PROFILES = {
    # Conservative baseline aligned with the current ACT QKV workload examples.
    "qkv_dse": {
        "add",
        "broadcast",
        "constant",
        "divide",
        "dot",
        "entry",
        "exponential",
        "module",
        "parameter",
        "reduce",
    }
}

HLO_OP_RE = re.compile(r"=\s*[^=]*?\b([a-z][a-z0-9\-]*)\(")


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_model(module: Any, entry: str) -> nn.Module:
    if not hasattr(module, entry):
        raise AttributeError(f"Entry '{entry}' not found in module {module.__name__}")
    obj = getattr(module, entry)

    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, type) and issubclass(obj, nn.Module):
        model = obj()
    elif callable(obj):
        model = obj()
    else:
        raise TypeError(
            f"Entry '{entry}' is unsupported. Use Module instance/class or factory callable."
        )

    if not isinstance(model, nn.Module):
        raise TypeError(f"Entry '{entry}' did not resolve to torch.nn.Module")
    return model.eval()


def _parse_input_spec(spec: str) -> Tuple[Tuple[int, ...], torch.dtype]:
    try:
        shape_str, dtype_str = spec.split(":")
        shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
        if not shape:
            raise ValueError("empty shape")
        dtype_key = dtype_str.strip().lower()
        dtype = DTYPE_MAP[dtype_key]
        return shape, dtype
    except Exception as exc:
        raise ValueError(
            f"Invalid input spec '{spec}'. Expected '<d0,d1,...>:<dtype>'"
        ) from exc


def _tensor_from_spec(shape: Sequence[int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device, dtype=torch.int32).to(torch.bool)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return torch.randint(0, 8, shape, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def _flatten_tensors(obj: Any) -> List[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        out: List[torch.Tensor] = []
        for item in obj:
            out.extend(_flatten_tensors(item))
        return out
    if isinstance(obj, dict):
        out: List[torch.Tensor] = []
        for key in sorted(obj.keys()):
            out.extend(_flatten_tensors(obj[key]))
        return out
    return []


def _parse_workloads(inputs: List[str], workloads: List[str]) -> List[List[str]]:
    if workloads:
        parsed = []
        for w in workloads:
            group = [x.strip() for x in w.split(";") if x.strip()]
            if not group:
                raise ValueError(f"Invalid workload '{w}'.")
            parsed.append(group)
        return parsed
    if inputs:
        return [inputs]
    raise ValueError("Provide either --input (one workload) or --workload (many workloads).")


def _extract_ops_with_lines(hlo_text: str) -> List[Tuple[int, str]]:
    found: List[Tuple[int, str]] = []
    for lineno, line in enumerate(hlo_text.splitlines(), start=1):
        for m in HLO_OP_RE.finditer(line):
            found.append((lineno, m.group(1)))
    return found


def _load_allowed_ops(
    isa_profile: str,
    allow_ops_file: str,
    allow_ops_hlo: str,
) -> set:
    allowed = set()
    if isa_profile:
        if isa_profile not in DEFAULT_ISA_PROFILES:
            raise ValueError(
                f"Unknown --isa-profile '{isa_profile}'. Choices: {sorted(DEFAULT_ISA_PROFILES.keys())}"
            )
        allowed |= set(DEFAULT_ISA_PROFILES[isa_profile])

    if allow_ops_file:
        p = Path(allow_ops_file).resolve()
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip().lower()
            if not s or s.startswith("#"):
                continue
            allowed.add(s)

    if allow_ops_hlo:
        p = Path(allow_ops_hlo).resolve()
        ref_ops = {op for _, op in _extract_ops_with_lines(p.read_text(encoding="utf-8"))}
        allowed |= ref_ops

    return allowed


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch model invocations to HLO text.")
    parser.add_argument("--model-file", required=True, help="Python file containing your model entry.")
    parser.add_argument(
        "--model-entry",
        default="build_model",
        help="Symbol name in model-file (Module instance/class or factory callable).",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="One input tensor spec: '<d0,d1,...>:<dtype>'. Repeat for multiple model args.",
    )
    parser.add_argument(
        "--workload",
        action="append",
        default=[],
        help="One semicolon-delimited workload, e.g. '1,128:float32;1,128:float32'. Repeat for many workloads.",
    )
    parser.add_argument(
        "--isa-profile",
        default="",
        help="Optional built-in op allowlist profile (e.g. qkv_dse).",
    )
    parser.add_argument(
        "--allow-ops-file",
        default="",
        help="Optional text file with one allowed HLO op per line.",
    )
    parser.add_argument(
        "--allow-ops-from-hlo",
        default="",
        help="Optional reference HLO file; all ops seen there are considered allowed.",
    )
    parser.add_argument(
        "--strict-ops",
        action="store_true",
        help="Fail if generated HLO contains ops outside the allowlist.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for .hlo files.")
    args = parser.parse_args()

    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
    except Exception as exc:
        raise RuntimeError(
            "torch-xla import failed. Install compatible torch/torch-xla packages first."
        ) from exc

    module_path = Path(args.model_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    module = _load_module(module_path)
    model = _resolve_model(module, args.model_entry)

    device = xm.xla_device()
    model = model.to(device)

    workloads = _parse_workloads(args.input, args.workload)
    allowed_ops = _load_allowed_ops(args.isa_profile, args.allow_ops_file, args.allow_ops_from_hlo)

    for idx, workload_specs in enumerate(workloads):
        parsed = [_parse_input_spec(s) for s in workload_specs]
        model_inputs = [_tensor_from_spec(shape, dtype, device) for shape, dtype in parsed]

        with torch.no_grad():
            outputs = model(*model_inputs)
            output_tensors = _flatten_tensors(outputs)
            if not output_tensors:
                raise RuntimeError("Model produced no tensor outputs to export.")

            hlo_text = torch_xla._XLAC._get_xla_tensors_hlo(output_tensors)

        if allowed_ops:
            found = _extract_ops_with_lines(hlo_text)
            unsupported = [(ln, op) for ln, op in found if op not in allowed_ops]
            if unsupported:
                uniq = sorted(set(op for _, op in unsupported))
                print(
                    f"[warn] workload_{idx:03d} has unsupported ops ({len(uniq)}): {', '.join(uniq)}"
                )
                for ln, op in unsupported[:25]:
                    print(f"       line {ln}: {op}")
                if len(unsupported) > 25:
                    print(f"       ... and {len(unsupported) - 25} more occurrences")
                if args.strict_ops:
                    raise RuntimeError(
                        f"Unsupported HLO ops found for workload_{idx:03d}; see warnings above."
                    )
            else:
                print(f"[ok] workload_{idx:03d} ops all allowed by selected ISA constraints")

        stem = f"workload_{idx:03d}"
        hlo_path = output_dir / f"{stem}.hlo"
        meta_path = output_dir / f"{stem}.meta.json"

        hlo_path.write_text(hlo_text, encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "model_file": str(module_path),
                    "model_entry": args.model_entry,
                    "workload_index": idx,
                    "inputs": workload_specs,
                    "output_hlo": str(hlo_path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[ok] wrote {hlo_path}")


if __name__ == "__main__":
    main()
