use super::*;

#[inline(always)]
pub(crate) fn jal<C: Counters, S: Snapshotter<C>, R: RAM>(
    state: &mut State<C>,
    ram: &mut R,
    snapshotter: &mut S,
    instr: Instruction,
    tape: &impl InstructionTape,
) {
    let _rs1_value = read_register::<C, 0>(state, instr.rs1);
    let _rs2_value = read_register::<C, 1>(state, instr.rs2); // formal
    let mut rd = state.pc.wrapping_add(core::mem::size_of::<u32>() as u32); // address of next opcode
    unsafe {
        core::hint::assert_unchecked(instr.imm % 4 == 0)
    };
    let jump_address = state.pc.wrapping_add(instr.imm);
    unsafe {
        core::hint::assert_unchecked(jump_address % 4 == 0)
    };
    state.pc = jump_address;
    // if core::hint::unlikely(jump_address & 0x3 != 0) {
    //     // unaligned PC
    //     panic!("Unaligned jump address 0x{:08x}", jump_address);
    // } else {
    //     state.pc = jump_address;
    // }
    tape.prefetch_instruction(state.pc);
    write_register::<C, 2>(state, instr.rd, &mut rd);
    increment_family_counter::<C, JUMP_BRANCH_SLT_CIRCUIT_FAMILY_IDX>(state);
}

#[inline(always)]
pub(crate) fn jalr<C: Counters, S: Snapshotter<C>, R: RAM>(
    state: &mut State<C>,
    ram: &mut R,
    snapshotter: &mut S,
    instr: Instruction,
    tape: &impl InstructionTape,
) {
    let rs1_value = read_register::<C, 0>(state, instr.rs1);
    let _rs2_value = read_register::<C, 1>(state, instr.rs2); // formal
    let mut rd = state.pc.wrapping_add(core::mem::size_of::<u32>() as u32); // address of next opcode
    let jump_address = rs1_value.wrapping_add(instr.imm) & !0x1;
    if core::hint::unlikely(jump_address & 0x3 != 0) {
        // unaligned PC
        panic!("Unaligned jump address 0x{:08x}", jump_address);
    } else {
        state.pc = jump_address;
    }
    tape.prefetch_instruction(state.pc);
    write_register::<C, 2>(state, instr.rd, &mut rd);
    increment_family_counter::<C, JUMP_BRANCH_SLT_CIRCUIT_FAMILY_IDX>(state);
}
