"""GEMMINI_17 — 16-instruction TAIDL ISA (Gemmini-style blocks: mvin/mvout, mesh, acc, norm).

Elementwise multiply is omitted: the shared ACT backend e-graph has no multiply IR op, and adding
one would require changing generators/backend. Attention-style workloads do not need it.

For per-instruction ISA energy plots/tables (plot_isa_workload_costs.py), generate
``targets/GEMMINI_17/backend/taidl_instruction_costs.json`` after this script:

    bash scripts/bash/run_isa_primitives.sh --isa-name GEMMINI_17
"""

from taidl import Accelerator


gem = Accelerator("GEMMINI_17")

gem.add_data_model("spad", [256], [64], "bf16")
gem.add_data_model("acc", [128], [64], "bf16")


# --- Data movement ---

instr = gem.add_instruction("load_rm", ["n"], ["addr_in", "addr_out"])
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

instr = gem.add_instruction("load_cm", ["n"], ["addr_in", "addr_out"])
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

instr = gem.add_instruction("load_scaled", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])
instr.set_outputs([["spad", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY load_scaled {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    %b = bf16[`@c.n`,64] bitcast_convert(%a);
    ROOT %Out0 = bf16[`@c.n`,64] convert(%b);
}
"""
)

instr = gem.add_instruction("load_bias", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])
instr.set_outputs([["acc", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY load_bias {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
"""
)

instr = gem.add_instruction("store_rm", ["n"], ["addr_in", "addr_out"])
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

instr = gem.add_instruction("store_cm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["spad", ["@a.addr_in"], ["@c.n"]]])
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


# --- On-chip moves ---

instr = gem.add_instruction("mov", ["n"], ["addr_in", "addr_out"])
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

instr = gem.add_instruction("mov_rev", ["n"], ["addr_in", "addr_out"])
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


# --- Matrix ---

instr = gem.add_instruction("gemm", ["n"], ["addr_lhs", "addr_rhs", "addr_out"])
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

instr = gem.add_instruction("gemm_acc", ["n"], ["addr_lhs", "addr_rhs", "addr_acc", "addr_out"])
instr.set_inputs([
    ["spad", ["@a.addr_lhs"], ["@c.n"]],
    ["spad", ["@a.addr_rhs"], ["64"]],
    ["acc", ["@a.addr_acc"], ["@c.n"]],
])
instr.set_outputs([["acc", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY gemm_acc {
    %A = bf16[`@c.n`,64] parameter(0);
    %B = bf16[64,64] parameter(1);
    %C = bf16[`@c.n`,64] parameter(2);
    %M = bf16[`@c.n`,64] dot(%A, %B), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    ROOT %Out0 = bf16[`@c.n`,64] add(%M, %C);
}
"""
)


# --- Elementwise ---

instr = gem.add_instruction("eltwise_add", ["n"], ["addr_1", "addr_2", "addr_out"])
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

instr = gem.add_instruction("relu", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["acc", ["@a.addr_in"], ["@c.n"]]])
instr.set_outputs([["acc", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY relu {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %z = bf16[`@c.n`,64] subtract(%In1, %In1);
    ROOT %Out0 = bf16[`@c.n`,64] maximum(%In1, %z);
}
"""
)

instr = gem.add_instruction("scale", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["acc", ["@a.addr_in"], ["@c.n"]]])
instr.set_outputs([["spad", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY scale {
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] convert(%In1);
}
"""
)


# --- Special ---

instr = gem.add_instruction("softmax", ["n"], ["addr"])
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

instr = gem.add_instruction("layernorm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["acc", ["@a.addr_in"], ["@c.n"]]])
instr.set_outputs([["acc", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY layernorm {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %s = bf16[`@c.n`] reduce_add(%In1), dimensions={1};
    %b = bf16[`@c.n`,64] broadcast(%s), dimensions={0};
    ROOT %Out0 = bf16[`@c.n`,64] divide(%In1, %b);
}
"""
)


# --- Pooling: pairwise max of two staged tiles ---

instr = gem.add_instruction("maxpool", ["n"], ["addr_a", "addr_b", "addr_out"])
instr.set_inputs([
    ["acc", ["@a.addr_a"], ["@c.n"]],
    ["acc", ["@a.addr_b"], ["@c.n"]],
])
instr.set_outputs([["spad", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics(
    """
ENTRY maxpool {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %In2 = bf16[`@c.n`,64] parameter(1);
    ROOT %Out0 = bf16[`@c.n`,64] maximum(%In1, %In2);
}
"""
)


gem.generate_oracle()
gem.generate_backend()
