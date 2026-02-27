"""QKV_DSE Accelerator ISA Definition for DSE benchmarking."""

from taidl import Accelerator


# Separate backend name so existing QKV flow remains unchanged.
qkv = Accelerator("QKV_DSE")


# Data models:
# - d1: primary on-chip tensor buffer (larger row capacity than tutorial QKV).
# - d2: compute/intermediate buffer with same tile width.
qkv.add_data_model("d1", [4096], [64], "bf16")
qkv.add_data_model("d2", [4096], [64], "bf16")


# (1) load_rm: HBM (d0) -> d1, row-major
instr = qkv.add_instruction("load_rm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])
instr.set_outputs([["d1", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY load_rm {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
"""
)


# (2) load_cm: HBM (d0) -> d1, column-major view via transpose
instr = qkv.add_instruction("load_cm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])
instr.set_outputs([["d1", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY load_cm {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    %b = bf16[`@c.n`,64] bitcast_convert(%a);
    ROOT %Out0 = bf16[64,`@c.n`] transpose(%b), dimensions={1,0};
}
"""
)


# (3) store_rm: d1 -> HBM (d0), row-major
instr = qkv.add_instruction("store_rm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d1", ["@a.addr_in"], ["@c.n"]]])
instr.set_outputs([["d0", ["@a.addr_out"], ["@c.n * 128"]]])
instr.add_semantics(
    """
ENTRY store_rm {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64,2] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%a);
}
"""
)


# (4) store_cm: d1 -> HBM (d0), column-major format
instr = qkv.add_instruction("store_cm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d1", ["@a.addr_in"], ["@c.n"]]])
instr.set_outputs([["d0", ["@a.addr_out"], ["@c.n * 128"]]])
instr.add_semantics(
    """
ENTRY store_cm {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = bf16[64,`@c.n`] transpose(%In1), dimensions={1,0};
    %b = u8[64,`@c.n`,2] bitcast_convert(%a);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%b);
}
"""
)


# (5) mov: d2 -> d1 copy for writeback or chaining
instr = qkv.add_instruction("mov", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d2", ["@a.addr_in"], ["@c.n"]]])
instr.set_outputs([["d1", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY mov {
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In1);
}
"""
)


# (5b) mov_rev: d1 -> d2 copy for residual/add paths
instr = qkv.add_instruction("mov_rev", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d1", ["@a.addr_in"], ["@c.n"]]])
instr.set_outputs([["d2", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY mov_rev {
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In1);
}
"""
)


# (6) gemm: A[n,64] * B[64,64] -> C[n,64]
# This extends the tutorial ISA to support variable row count in one instruction.
instr = qkv.add_instruction("gemm", ["n"], ["addr_1", "addr_2", "addr_out"])
instr.set_inputs([["d1", ["@a.addr_1"], ["@c.n"]], ["d1", ["@a.addr_2"], ["64"]]])
instr.set_outputs([["d2", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY gemm {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %In2 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[`@c.n`,64] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
"""
)


# (7) softmax: row-wise softmax over width-64 vectors
instr = qkv.add_instruction("softmax", ["n"], ["addr"])
instr.set_inputs([["d2", ["@a.addr"], ["@c.n"]]])
instr.set_outputs([["d2", ["@a.addr"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY softmax {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = bf16[`@c.n`,64] exponential(%In1);
    %reduced = bf16[`@c.n`] reduce_add(%a), dimensions={1};
    %b = bf16[`@c.n`,64] broadcast(%reduced), dimensions={0};
    ROOT %Out0 = bf16[`@c.n`,64] divide(%a, %b);
}
"""
)


# (8) eltwise_add: elementwise residual-style add on d2 tiles
instr = qkv.add_instruction("eltwise_add", ["n"], ["addr_1", "addr_2", "addr_out"])
instr.set_inputs([["d2", ["@a.addr_1"], ["@c.n"]], ["d2", ["@a.addr_2"], ["@c.n"]]])
instr.set_outputs([["d2", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY eltwise_add {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %In2 = bf16[`@c.n`,64] parameter(1);
    ROOT %Out0 = bf16[`@c.n`,64] add(%In1, %In2);
}
"""
)


# Generate programming APIs and generated backend.
qkv.generate_oracle()
qkv.generate_backend()
