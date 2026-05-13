"""Gemmini ISA Definition."""

from taidl import Accelerator

acc = Accelerator("Gemmini")

# Define Data Models
acc.add_data_model("spad", [1024 * 16], [16], "s8")
acc.add_data_model("acc", [64 * 16], [16], "s32")

instr = acc.add_instruction("mvin_spad", ["rows"], ["hbm_addr", "sp_addr"])
instr.set_inputs([["d0", ["@a.hbm_addr"], ["@c.rows * 16"]]])
instr.set_outputs([["spad", ["@a.sp_addr"], ["@c.rows"]]])
instr.add_semantics(
    """
ENTRY mvin_spad{
    %In1 = u8[`@c.rows*16`] parameter(0);
    %data = u8[`@c.rows`,16] reshape(%In1);
    ROOT %Out0 = s8[`@c.rows`,16] bitcast_convert(%data);
}
"""
)

instr = acc.add_instruction("mvout_spad", ["rows"], ["hbm_addr", "sp_addr"])
instr.set_inputs([["spad", ["@a.sp_addr"], ["@c.rows"]]])
instr.set_outputs([["d0", ["@a.hbm_addr"], ["@c.rows * 16"]]])
instr.add_semantics(
    """
ENTRY mvout_spad{
    %In1 = s8[`@c.rows`, 16] parameter(0);
    %data = u8[`@c.rows`,16] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.rows*16`] reshape(%data);
}
"""
)

instr = acc.add_instruction("mvin_acc", ["rows"], ["hbm_addr", "acc_addr"])
instr.set_inputs([["d0", ["@a.hbm_addr"], ["@c.rows * 64"]]])
instr.set_outputs([["acc", ["@a.acc_addr"], ["@c.rows"]]])
instr.add_semantics(
    """
ENTRY mvin_acc{
    %In1 = u8[`@c.rows * 16 * 4`] parameter(0);
    %r = u8[`@c.rows`,16,4] reshape(%In1);
    ROOT %Out0 = s32[`@c.rows`, 16] bitcast_convert(%r);
}
"""
)

instr = acc.add_instruction("mvout_acc", ["rows"], ["hbm_addr", "acc_addr"])
instr.set_inputs([["acc", ["@a.acc_addr"], ["@c.rows"]]])
instr.set_outputs([["d0", ["@a.hbm_addr"], ["@c.rows * 64"]]])
instr.add_semantics(
    """
ENTRY mvout_acc{
    %In1 = s32[`@c.rows`, 16] parameter(0);
    %data = u8[`@c.rows`,16,4] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.rows*64`] reshape(%data);
}
"""
)

instr = acc.add_instruction("mvin_acc_low", ["rows"], ["hbm_addr", "acc_addr"])
instr.set_inputs([["d0", ["@a.hbm_addr"], ["@c.rows * 16"]]])
instr.set_outputs([["acc", ["@a.acc_addr"], ["@c.rows"]]])
instr.add_semantics(
    """
ENTRY mvin_acc_low{
    %In1 = u8[`@c.rows * 16`] parameter(0);
    %a = u8[`@c.rows`,16] reshape(%In1);
    %b = s8[`@c.rows`,16] bitcast_convert(%a);
    ROOT %Out0 = s32[`@c.rows`,16] convert(%b);
}
"""
)

instr = acc.add_instruction("mvin_acc_low_add", ["rows"], ["hbm_addr", "acc_addr"])
instr.set_inputs(
    [["acc", ["@a.acc_addr"], ["@c.rows"]], ["d0", ["@a.hbm_addr"], ["@c.rows * 16"]]]
)
instr.set_outputs([["acc", ["@a.acc_addr"], ["@c.rows"]]])
instr.add_semantics(
    """
ENTRY mvin_acc_low_add{
    %In1 = s32[`@c.rows`, 16] parameter(0);
    %In2 = u8[`@c.rows * 16`] parameter(1);
    %a = u8[`@c.rows`,16] reshape(%In2);
    %b = s8[`@c.rows`,16] bitcast_convert(%a);
    %data = s32[`@c.rows`,16] convert(%b);
    ROOT %Out0 = s32[`@c.rows`,16] add(%In1, %data);
}
"""
)

instr = acc.add_instruction("mvout_acc_low", ["rows"], ["hbm_addr", "acc_addr"])
instr.set_inputs([["acc", ["@a.acc_addr"], ["@c.rows"]]])
instr.set_outputs([["d0", ["@a.hbm_addr"], ["@c.rows * 16"]]])
instr.add_semantics(
    """
ENTRY mvout_acc_low{
    %In1 = s32[`@c.rows`, 16] parameter(0);
    %a = s8[`@c.rows`,16] convert(%In1);
    %b = u8[`@c.rows`,16] bitcast_convert(%a);
    ROOT %Out0 = u8[`@c.rows * 16`] reshape(%b);
}
"""
)

# Matmuls

instr = acc.add_instruction("matmul8", [], ["C_dst", "A_src", "B_src"])
instr.set_inputs([["spad", ["@a.A_src"], [16]], ["spad", ["@a.B_src"], [16]]])
instr.set_outputs([["spad", ["@a.C_dst"], [16]]])
instr.add_semantics(
    """
ENTRY matmul_8{
    %In1 = s8[16, 16] parameter(0);
    %In2 = s8[16, 16] parameter(1);
    ROOT %Out0 = s8[16,16] dot(%In1, %In2), lhs_batch_dims={}, lhs_contracting_dims={1}, rhs_batch_dims={}, rhs_contracting_dims={0};
}
"""
)

instr = acc.add_instruction("matmul32", [], ["C_dst", "A_src", "B_src"])
instr.set_inputs([["spad", ["@a.A_src"], [16]], ["spad", ["@a.B_src"], [16]]])
instr.set_outputs([["acc", ["@a.C_dst"], [16]]])
instr.add_semantics(
    """
ENTRY matmul_32{
    %In1 = s8[16, 16] parameter(0);
    %In2 = s8[16, 16] parameter(1);
    %a = s32[16, 16] convert(%In1);
    %b = s32[16, 16] convert(%In2);
    ROOT %Out0 = s32[16,16] dot(%a, %b), lhs_batch_dims={}, lhs_contracting_dims={1}, rhs_batch_dims={}, rhs_contracting_dims={0};
}
"""
)

instr = acc.add_instruction("mac8", [], ["C_dst", "A_src", "B_src", "D_src"])
instr.set_inputs(
    [
        ["spad", ["@a.A_src"], [16]],
        ["spad", ["@a.B_src"], [16]],
        ["spad", ["@a.D_src"], [16]],
    ]
)
instr.set_outputs([["spad", ["@a.C_dst"], [16]]])
instr.add_semantics(
    """
ENTRY mvin_spad{
    %In1 = s8[16, 16] parameter(0);
    %In2 = s8[16, 16] parameter(1);
    %In3 = s8[16, 16] parameter(2);
    %dot = s8[16, 16] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    ROOT %Out0 = s8[16, 16] add(%dot, %In3);
}
"""
)

instr = acc.add_instruction("mac32", [], ["C_dst", "A_src", "B_src", "D_src"])
instr.set_inputs(
    [
        ["spad", ["@a.A_src"], [16]],
        ["spad", ["@a.B_src"], [16]],
        ["acc", ["@a.D_src"], [16]],
    ]
)
instr.set_outputs([["acc", ["@a.C_dst"], [16]]])
instr.add_semantics(
    """
ENTRY mac_32{
    %In1 = s8[16, 16] parameter(0);
    %In2 = s8[16, 16] parameter(1);
    %In3 = s32[16, 16] parameter(2);
    %a = s32[16,16] convert(%In1);
    %b = s32[16,16] convert(%In2);
    %dot = s32[16,16] dot(%a, %b), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    ROOT %Out0 = s32[16,16] add(%dot, %In3);
}
"""
)

acc.generate_backend()
acc.generate_oracle()
