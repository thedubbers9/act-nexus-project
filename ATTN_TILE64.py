"""ATTN_TILE64 Accelerator ISA Definition.

This is a small but realistic transformer-attention tile accelerator:

- Workload family: scaled dot-product attention core on a 64x64 tile
- Accelerator style: scratchpad + matmul unit + vector softmax unit
- Scope: intentionally limited to a manageable HLO subset that ACT's current
  backend path already handles well

v1 focuses on:
  O = softmax(Q @ K^T) @ V

and deliberately leaves out:
  - LayerNorm / RMSNorm
  - GELU / SwiGLU
  - KV-cache updates
  - dynamic sequence lengths
  - explicit scaling and masking

The goal is to establish a non-dummy, current transformer workload family that
maps cleanly from HLO to a generated ACT backend.
"""

from taidl import Accelerator


attn = Accelerator("ATTN_TILE64")


# Data models
# spad: primary scratchpad used for Q/K/V/output staging.
# acc: compute scratchpad used for GEMM outputs and in-place softmax.
attn.add_data_model("spad", [256], [64], "bf16")
attn.add_data_model("acc", [128], [64], "bf16")


# HBM -> spad, row-major load
instr = attn.add_instruction("load_rm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])
instr.set_outputs([["spad", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY load_rm {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
"""
)


# HBM -> spad, logical column-major load.
# This is useful for K^T-style access patterns in attention.
instr = attn.add_instruction("load_cm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])
instr.set_outputs([["spad", ["@a.addr_out"], ["@c.n"]]])
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


# spad -> HBM, row-major store
instr = attn.add_instruction("store_rm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["spad", ["@a.addr_in"], ["@c.n"]]])
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


# acc -> spad copy for writeback or chaining
instr = attn.add_instruction("mov", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["acc", ["@a.addr_in"], ["@c.n"]]])
instr.set_outputs([["spad", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY mov {
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In1);
}
"""
)


# spad -> acc copy for residual-style or staging flows
instr = attn.add_instruction("mov_rev", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["spad", ["@a.addr_in"], ["@c.n"]]])
instr.set_outputs([["acc", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY mov_rev {
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In1);
}
"""
)


# GEMM on one attention tile row-block.
# Typical uses:
#   scores = Q @ K^T
#   output = probs @ V
instr = attn.add_instruction("gemm", ["n"], ["addr_lhs", "addr_rhs", "addr_out"])
instr.set_inputs([["spad", ["@a.addr_lhs"], ["@c.n"]], ["spad", ["@a.addr_rhs"], ["64"]]])
instr.set_outputs([["acc", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY gemm {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %In2 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[`@c.n`,64] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
"""
)


# Row-wise softmax over 64 columns.
# This captures the core normalization step in transformer attention.
instr = attn.add_instruction("softmax", ["n"], ["addr"])
instr.set_inputs([["acc", ["@a.addr"], ["@c.n"]]])
instr.set_outputs([["acc", ["@a.addr"], ["@c.n"]]])
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


# Residual-style elementwise add on acc tiles.
instr = attn.add_instruction("eltwise_add", ["n"], ["addr_1", "addr_2", "addr_out"])
instr.set_inputs([["acc", ["@a.addr_1"], ["@c.n"]], ["acc", ["@a.addr_2"], ["@c.n"]]])
instr.set_outputs([["acc", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY eltwise_add {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %In2 = bf16[`@c.n`,64] parameter(1);
    ROOT %Out0 = bf16[`@c.n`,64] add(%In1, %In2);
}
"""
)


attn.generate_oracle()
attn.generate_backend()
