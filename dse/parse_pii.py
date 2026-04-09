"""Shim: ``from dse.parse_pii import ...`` (implementation in ``dse.src.parse_pii``)."""

from dse.src.parse_pii import (
    CandidateProgram,
    InstructionCall,
    KernelMetadata,
    ParseError,
    parse_pii,
    parse_pii_dir,
)

__all__ = [
    "CandidateProgram",
    "InstructionCall",
    "KernelMetadata",
    "ParseError",
    "parse_pii",
    "parse_pii_dir",
]
