use std::{mem::offset_of, ptr::addr_of_mut};
use common_constants::TimestampScalar;
use dynasmrt::{dynasm, x64, DynasmApi, DynasmLabelApi};
use riscv_decode::Instruction;

use crate::vm::*;

#[cfg(test)]
mod tests;

const MAX_RAM_SIZE: usize = 1 << 30; // 1 Gb, as we want to avoid having separate pointers to RAM (that we want to have continuous to perform very simple read/writes), and timestamp bookkeping space

const RAM_SIZE: usize = 1 << 30;
const NUM_RAM_WORDS: usize = RAM_SIZE / core::mem::size_of::<u32>();

// Register use and mapping

// - x10-x15 (RV) are stored in r10-r15 (X86)
// - RDI holds a pointer to backing array for snapshot itself, with elements being Register struct (TODO: decide if we want aligned or not timestamps. Most likely yes)
// - RSI will contain a pointer to the special structure that begins with backing array for memory, followed by backing array for word timestamps
// - r8 holds a timestamp (0 mod 4 in the cycle)
// - r9 holds a number of elements in the snapshot

// For registers with no dedicated x86 register,
// register writes go via rax and reads via rdx
// rcx also doesn't contain a register because it must be used for bitshifts
//
// x10 - x15 are assiged to r10 - r15
// rbx is for x9

// Registers that are placed not in the GPR are instead placed into 128-bit vector registers, and loaded using PEXTRD and stored using PINSRD.
// In total we still use upper bound of 8 vector registers xmm0-xmm7.

// On the stack we will have a structure that will allows us to pass in a single pointer all the global machine state.

#[repr(C, align(8))]
#[derive(Debug)]
pub struct TraceChunk<const CHUNK_SIZE: usize> {
    pub values: [u32; CHUNK_SIZE],
    pub timestamps: [TimestampScalar; CHUNK_SIZE],
}

impl<const CHUNK_SIZE: usize> TraceChunk<CHUNK_SIZE> {
    const TIMESTAMPS_OFFSET: usize = offset_of!(Self, timestamps);
}

// We will measure trace chunk in a number of memory accesses and not in a fixed number of cycles that did pass between them
const TRACE_CHUNK_LEN: usize = 1 << 4;
// const TRACE_CHUNK_LEN: usize = 1 << 20;

pub type ReceiveTraceFn = extern "sysv64" fn(*mut (), &mut TraceChunk<TRACE_CHUNK_LEN>, &MachineState) -> *mut TraceChunk<TRACE_CHUNK_LEN>;
pub type ReceiveFinalStateFn = extern "sysv64" fn(*mut (), &mut TraceChunk<TRACE_CHUNK_LEN>, &MachineState, final_pc: u32);

const MAX_NUM_COUNTERS: usize = 16;

#[repr(u8)]
pub enum CounterType {
    AddSubLui = 0,
    BranchSlt,
    ShiftBinaryCsr,
    MulDiv,
    MemWord,
    MemSubword,
    BlakeDelegation,
    BigintDelegation,
    KeccakDelegation,
    FormalEnd, // must always be the last
}

const _: () = const {
    assert!(CounterType::FormalEnd as u8 as usize <= MAX_NUM_COUNTERS);
};

#[repr(C, align(16))]
#[derive(Debug)]
pub struct MachineState {
    registers: [u32; 32], // aligned at 16, so we can write XMMs directly into the stack
    register_timestamps: [TimestampScalar; 32],
    counters: [u32; MAX_NUM_COUNTERS],
    pc_or_partial_trace_chunk: u32,
    timestamp: TimestampScalar,
    context_ptr: *mut (),
}

impl MachineState {
    const SIZE: usize = core::mem::size_of::<Self>();
    const _T: () = const {
        assert!(Self::SIZE % core::mem::size_of::<u64>() == 0);
        assert!(Self::SIZE % 16 == 0); // so our stack is aligned if we just grow it by this structure size
    };

    const SIZE_IN_QWORDS: usize = Self::SIZE / core::mem::size_of::<u64>();
    const REGISTER_TIMESTAMPS_OFFSET: usize = offset_of!(Self, register_timestamps);
    const COUNTERS_OFFSET: usize = offset_of!(Self, counters);
    const PC_OR_PARTIAL_TRACE_CHUNK_OFFSET: usize = offset_of!(Self, pc_or_partial_trace_chunk);
    const TIMESTAMP_OFFSET: usize = offset_of!(Self, timestamp);
    const CONTEXT_PTR_OFFSET: usize = offset_of!(Self, context_ptr);
}

#[repr(C, align(8))]
pub struct MemoryHolder {
    memory: [u32; NUM_RAM_WORDS],
    timestamps: [TimestampScalar; NUM_RAM_WORDS],
}

impl MemoryHolder {
    const TIMESTAMPS_OFFSET: usize = offset_of!(Self, timestamps);
}

// We need to maintain extra information, that are counters of circuit families and delegations - those are also saved in 128-bit vector registers.
// We need at most 6 circuit families and 3 delegation types, and we assume u32 counters at most in realistic scenarios. So we reserve xmm8 and xmm9

// Timestamps of registers will be held on the stack, as well as a pointer to the non-determinism servant. We will later on restructure
// RAM and non-determinism traits to use separate "memory peek" trait, that only allows to view values, but not affect them or timestamps

// NOTE: stack on x86 must be 16-byte aligned, so we should carefully adjust stack when we push/pop

// The prologue saves all callee-saved registers
// This allows us to use all but rbp and rsp
// Using rbp would mess with debuggers
// Using rsp would cause signal handlers to write to some random location
// instead of the stack.
macro_rules! prologue {
    ($ops:ident) => {
        dynasm!($ops
            // this is "enter 0, 0"
            ; push rbp
            ; mov rbp, rsp

            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; push r15

            // align stack
            ; add rsp, 8
        )
    };
}

macro_rules! epilogue {
    ($ops:ident) => {
        dynasm!($ops
            ; sub rsp, 8

            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; leave // movs RBP into RSP, and pops RBP

            ; ret
        )
    };
}

macro_rules! receive_trace {
    ($ops:ident, $recv:expr) => {
        dynasm!($ops
            // handler for full trace chunk. In r9 we expect the current PC
            ; ->trace_buffer_full:
            // we only call this function after executing the opcode in full,
            // so we do not care about rax (for stores), rdx (for loads) or rcx (scratch)

            // put current RSP into RDX - it's our pointer to MachineState
            ; mov rdx, rsp
            // add 8 as we did CALL into here
            ; add rdx, 8

            // ; push rax
            // ; push rcx
            // ; push rdx
            ;; before_call!($ops, 8) // actual structure is 8 bytes above RSP
            ; mov rax, QWORD $recv as _
            ; mov rsi, rdi // second argument is our trace chunk
            ; mov rdi, [rdx + (MachineState::CONTEXT_PTR_OFFSET as i32)] // first argument is pointer to the context
            // third is our machine state - already in RDX - no need to load it
            ; add rsp, 8 // align stack
            ; call rax
            ; sub rsp, 8
            ;; after_call!($ops, 8) // actual structure is 8 bytes above RSP
            // and in RAX we expect the return value, that is a NEW pointer to the scratch space if needed
            ; mov rdi, rax
            // ; pop rdx
            // ; pop rcx
            // ; pop rax
            ; xor r9, r9 // reset counter of snapshots
            ; ret
        )
    };
}

macro_rules! quit {
    ($ops:ident, $recv:expr) => {
        dynasm!($ops
            // handler for final trace chunk. In r9 we have a counter of snapshotted data in the last chunk, and in RCX we have a final PC
            ; ->quit:
            // we only call this function after executing the opcode in full,
            // so we do not care about rax (for stores), rdx (for loads) or rcx (scratch)

            // put current RSP into RDX - it's our pointer to MachineState
            ; mov rdx, rsp

            // ; push rax
            // ; push rcx
            // ; push rdx
            ;; before_call!($ops, 0) // structure is at RSP
            ; mov rax, QWORD $recv as _
            ; mov rsi, rdi // second argument is our trace chunk
            ; mov rdi, [rdx + (MachineState::CONTEXT_PTR_OFFSET as i32)] // first argument is pointer to the context
            // third is our machine state - already in RDX - no need to load it
            // fourth is final pc - already in RCX - no need to load it
            ; call rax
            ;; after_call!($ops, 0) // structure is at RSP
            // and in RAX we expect the return value, that is a NEW pointer to the scratch space if needed
            ; mov rdi, rax
            // ; pop rdx
            // ; pop rcx
            // ; pop rax

            // we return nothing
            ;; epilogue!($ops)
        )
    };
}

// this macro saves registers, and "updates" a machine state structure on the stack to reflect the current state
macro_rules! before_call {
    ($ops:ident, $off:expr) => {
        dynasm!($ops
            ; push rsi
            ; push rdi

            // this we will save below
            // ; push r8
            // ; push r9
            // ; push r10
            // ; push r11

            // 2 pushes

            ;; save_machine_state!($ops, 2 * 8 + $off)
        )
    }
}

// this macro saves registers, and "updates" a machine state structure on the stack to reflect the current state
macro_rules! save_machine_state {
    ($ops:ident, $offset:expr) => {
        dynasm!($ops
            // offset is an offset of our MachineState from RSP

            // First write all registers that are mapped into XMMs
            ; movdqu [rsp + ($offset as i32) + 0], xmm0
            ; movdqu [rsp + ($offset as i32) + 16], xmm1
            ; movdqu [rsp + ($offset as i32) + 32], xmm2 // r8 fits here
            // ; movdqu [rsp + ($offset as i32) + 48], xmm3
            ; movdqu [rsp + ($offset as i32) + 64], xmm4
            ; movdqu [rsp + ($offset as i32) + 80], xmm5
            ; movdqu [rsp + ($offset as i32) + 96], xmm6
            ; movdqu [rsp + ($offset as i32) + 112], xmm7

            // Save RV registers that are mapped into X86 GPRs (r9-r15)
            ; mov [rsp + ($offset as i32) + (9 * 4 as i32)], ebx // R9 -> RBX
            ; mov [rsp + ($offset as i32) + (10 * 4 as i32)], r10d // R10 -> R10
            ; mov [rsp + ($offset as i32) + (11 * 4 as i32)], r11d // R11 -> R11
            ; mov [rsp + ($offset as i32) + (12 * 4 as i32)], r12d // R12 -> R12
            ; mov [rsp + ($offset as i32) + (13 * 4 as i32)], r13d // R13 -> R13
            ; mov [rsp + ($offset as i32) + (14 * 4 as i32)], r14d // R14 -> R14
            ; mov [rsp + ($offset as i32) + (15 * 4 as i32)], r15d // R15 -> R15

            // put current timestamp (without asumptions about mod 4)
            ; mov [rsp + ($offset as i32) + (MachineState::TIMESTAMP_OFFSET as i32)], r8
            // and PC (that should be in r9) if we care
            ; mov [rsp + ($offset as i32) + (MachineState::PC_OR_PARTIAL_TRACE_CHUNK_OFFSET as i32)], r9d
        )
    }
}

macro_rules! after_call {
    ($ops:ident, $off:expr) => {
        dynasm!($ops
            ;; update_machine_state_post_call!($ops, 2 * 8 + $off)

            // restored above
            // ; pop r11
            // ; pop r10
            // ; pop r9
            // ; pop r8

            ; pop rdi
            ; pop rsi
        )
    }
}

// this macro updates machine state out of 
macro_rules! update_machine_state_post_call {
    ($ops:ident, $offset:expr) => {
        dynasm!($ops
            // offset is an offset of our MachineState from RSP

            // take updated timestamp (also without assumptions)
            ; mov r8, [rsp + ($offset as i32) + (MachineState::TIMESTAMP_OFFSET as i32)]
            // and r9 is not important

            // Restore RV registers that are mapped into X86 GPRs (r9-r15) 
            ; mov ebx, [rsp + ($offset as i32) + (9 * 4 as i32)]  // R9 -> RBX
            ; mov r10d, [rsp + ($offset as i32) + (10 * 4 as i32)]  // R10 -> R10
            ; mov r11d, [rsp + ($offset as i32) + (11 * 4 as i32)]  // R11 -> R11
            ; mov r12d, [rsp + ($offset as i32) + (12 * 4 as i32)]  // R12 -> R12
            ; mov r13d, [rsp + ($offset as i32) + (13 * 4 as i32)]  // R13 -> R13
            ; mov r14d, [rsp + ($offset as i32) + (14 * 4 as i32)]  // R14 -> R14
            ; mov r15d, [rsp + ($offset as i32) + (15 * 4 as i32)]  // R15 -> R15

            ; movdqu xmm0, [rsp + ($offset as i32) + 0]
            ; movdqu xmm1, [rsp + ($offset as i32) + 16] 
            ; movdqu xmm2, [rsp + ($offset as i32) + 32] 
            // ; movdqu xmm3, [rsp + ($offset as i32) + 48]
            ; movdqu xmm4, [rsp + ($offset as i32) + 64]
            ; movdqu xmm5, [rsp + ($offset as i32) + 80] 
            ; movdqu xmm6, [rsp + ($offset as i32) + 96] 
            ; movdqu xmm7, [rsp + ($offset as i32) + 112] 
        )
    }
}

const SCRATCH_REGISTER: u8 = x64::Rq::RCX as u8;

fn rv_to_gpr(x: u32) -> Option<u8> {
    use x64::Rq::*;
    assert!(x < 32);

    Some(
        (match x {
            9 => RBX,
            10 => R10,
            11 => R11,
            12 => R12,
            13 => R13,
            14 => R14,
            15 => R15,
            _ => return None,
        }) as u8,
    )
}

fn destination_gpr(x: u32) -> u8 {
    rv_to_gpr(x).unwrap_or(x64::Rq::RAX as u8)
}

const RV_REGISTERS_NUM_XMMS: u8 = 8;

fn rv_reg_to_xmm_reg(x: u8) -> (u8, u8) {
    assert!(x < 32);
    let imm = x & 0b11;
    let xmm_register = x >> 2;
    assert!(xmm_register < RV_REGISTERS_NUM_XMMS);

    (xmm_register, imm)
}

fn store_result(ops: &mut x64::Assembler, x: u32) {
    assert!(x != 0);
    assert!(x < 32);

    if rv_to_gpr(x).is_none() {
        let x = x as u8;
        let (xmm_register, imm) = rv_reg_to_xmm_reg(x);
        dynasm!(ops
            ; pinsrd Rx(xmm_register), eax, imm as i8
        )
    }
}

/// Returns the general purpose register that now holds the value of the
/// RISC-V register `x`.
/// Do not use in quick succession; the first value will get overwritten.
fn load(ops: &mut x64::Assembler, x: u32) -> u8 {
    rv_to_gpr(x).unwrap_or_else(|| {
        if x == 0 {
            dynasm!(ops
                ; xor edx, edx
            );
        } else {
            let x = x as u8;
            let (xmm_register, imm) = rv_reg_to_xmm_reg(x);
            dynasm!(ops
                ; pextrd edx, Rx(xmm_register), imm as i8
            );
        }

        x64::Rq::RDX as u8
    })
}

/// Loads the RISC-V register `x` into the specified register.
fn load_into(ops: &mut x64::Assembler, x: u32, destination: u8) {
    if let Some(gpr) = rv_to_gpr(x) {
        if destination != gpr {
            dynasm!(ops
                ; mov Rd(destination), Rd(gpr)
            );
        }
    } else {
        if x == 0 {
            dynasm!(ops
                ; xor Rd(destination), Rd(destination)
            );
        } else {
            let x = x as u8;
            let (xmm_register, imm) = rv_reg_to_xmm_reg(x);
            dynasm!(ops
                ; pextrd Rd(destination), Rx(xmm_register), imm as i8
            );
        }
    }
}

fn load_abelian(ops: &mut x64::Assembler, x: u32, y: u32, destination: u8) -> u8 {
    let a = rv_to_gpr(x);
    let b = rv_to_gpr(y);
    if a == Some(destination) {
        load(ops, y)
    } else if b == Some(destination) {
        load(ops, x)
    } else {
        load_into(ops, x, destination);
        load(ops, y)
    }
}

macro_rules! print_registers {
    ($ops:ident) => {
        dynasm!($ops
            ; sub rsp, 32 * 4
            ; mov DWORD [rsp], 0
        );
        for i in 1..32 {
            let reg = load(&mut $ops, i);
            dynasm!($ops
                ; mov [rsp + 4 * i as i32], Rd(reg)
            );
        }

        dynasm!($ops
            ; mov rcx, rsp

            ; push rdi
            ; push rsi
            ; push r8
            ; push r9

            ; mov rax, QWORD print_registers as _
            ; mov rdi, rcx
            ; call rax

            ; pop r9
            ; pop r8
            ; pop rsi
            ; pop rdi
        );

        for i in 1..32 {
            let out = destination_gpr(i);
            dynasm!($ops
                ; mov Rd(out), [rsp + 4 * i as i32]
            );
            store_result(&mut $ops, i);
        }
        dynasm!($ops
            ; add rsp, 32 * 4
        );
    };
}

macro_rules! increment_trace {
    ($ops:ident, $pc: expr) => {
        dynasm!($ops
            ; inc r9
            ; cmp r9, TRACE_CHUNK_LEN as _
            ; jne >skip
            ; mov r9d, ($pc as i32) // abuse r9 for it
            ; call ->trace_buffer_full
            ; skip:
        );
    };
}

fn record_circuit_type(ops: &mut x64::Assembler, circuit_type: CounterType, by: u16) {
    assert!(by > 0);
    let x = circuit_type as u8;

    if by == 1 {
        dynasm!(ops
            ; inc DWORD [rsp + 4 * (x as i32) + (MachineState::COUNTERS_OFFSET as i32)]
        );
    } else {
        todo!();
        // dynasm!(ops
        //     ; pextrd ecx, Rx(xmm_register), imm as i8
        //     ; add ecx, (by as i32)
        //     ; pinsrd Rx(xmm_register), ecx, imm as i8
        // );
    }
}

macro_rules! pre_bump_timestamp_and_touch {
    ($ops:ident, $d:expr, $r:expr) => {
        dynasm!($ops
            ; add r8, $d
            ; mov [rsp + 8*($r as i32) + (MachineState::REGISTER_TIMESTAMPS_OFFSET as i32)], r8
        );
    };
}

macro_rules! touch_register_and_increment_timestamp {
    ($ops:ident, $r:expr) => {
        dynasm!($ops
            ; mov [rsp + 8*($r as i32) + (MachineState::REGISTER_TIMESTAMPS_OFFSET as i32)], r8
            ; inc r8
        );
    };
}

macro_rules! touch_register_and_bump_timestamp {
    ($ops:ident, $r:expr, $d:expr) => {
        dynasm!($ops
            ; mov [rsp + 8*($r as i32) + (MachineState::REGISTER_TIMESTAMPS_OFFSET as i32)], r8
            ; add r8, $d
        );
    };
}

macro_rules! bump_timestamp {
    ($ops:ident, $r:expr) => {
        dynasm!($ops
            ; add r8, $r
        );
    };
}

macro_rules! emit_runtime_error {
    ($ops:ident) => {
        dynasm!($ops
            ; jmp ->exit_with_error
        )
    };
}

pub fn run_alternative_simulator<'a, N: NonDeterminismCSRSource<RamWithRomRegion< {common_constants::rom::ROM_SECOND_WORD_BITS } >>>(
    program: &[u32],
    non_determinism_source: &'a mut N,
    initial_memory: &[u32],
) {
    assert!(initial_memory.len() <= common_constants::rom::ROM_WORD_SIZE);
    let mut ops = x64::Assembler::new().unwrap();
    let start = ops.offset();

    dynasm!(ops
        ; ->start:
        ;; prologue!(ops)
        ; vzeroall
        ; xor rbx, rbx
        ; xor r10, r10
        ; xor r11, r11
        ; xor r12, r12
        ; xor r13, r13
        ; xor r14, r14
        ; xor r15, r15

        // set initial timestamp and snapshot counter
        ; mov r8, INITIAL_TIMESTAMP as _
        ; xor r9, r9
    );

    // allocate stack space for Machine state
    dynasm!(ops
        ; sub rsp, (MachineState::SIZE as i32)
    );
    for i in 0..MachineState::SIZE_IN_QWORDS {
        dynasm!(ops
            ; mov QWORD [rsp + 8 * i as i32], 0
        );
    }

    // we expect trace chunk in RDI, and memory in RSI, and context pointer in RDX,
    // so we need to copy context pointer into our structure
    dynasm!(ops
        ; mov [rsp + (MachineState::CONTEXT_PTR_OFFSET as i32)], rdx
    );

    // Static jump targets for JAL and branch instructions
    let instruction_labels = (0..program.len())
        .map(|_| ops.new_dynamic_label())
        .collect::<Vec<_>>();

    // Jump target array for Jalr
    // Records the position of each RISC-V instruction relative to the start
    let mut jump_offsets = vec![];

    for (i, raw_instruction) in program.iter().enumerate() {
        let pc = i as u32 * 4;

        dynasm!(ops
            ; =>instruction_labels[i]
        );
        jump_offsets.push(ops.offset().0);

        let Ok(instruction) = riscv_decode::decode(*raw_instruction) else {
            emit_runtime_error!(ops);
            continue;
        };

        let mut instruction_emitted = false;

        // Pure instructions
        if matches!(
            instruction,
            Instruction::Addi(_)
                | Instruction::Andi(_)
                | Instruction::Ori(_)
                | Instruction::Xori(_)
                | Instruction::Slti(_)
                | Instruction::Sltiu(_)
                | Instruction::Slli(_)
                | Instruction::Srli(_)
                | Instruction::Srai(_)
                | Instruction::Lui(_)
                | Instruction::Auipc(_)
                | Instruction::Add(_)
                | Instruction::Sub(_)
                | Instruction::Slt(_)
                | Instruction::Sltu(_)
                | Instruction::And(_)
                | Instruction::Or(_)
                | Instruction::Xor(_)
                | Instruction::Sll(_)
                | Instruction::Srl(_)
                | Instruction::Sra(_)
                | Instruction::Lb(_)
                | Instruction::Lbu(_)
                | Instruction::Lh(_)
                | Instruction::Lhu(_)
                | Instruction::Lw(_)
                | Instruction::Mul(_)
                | Instruction::Mulh(_)
                | Instruction::Mulhu(_)
                | Instruction::Mulhsu(_)
        ) {
            let rd = (raw_instruction >> 7) & 0x1F;
            let out = destination_gpr(rd);
            // Instructions that just compute a result are NOPs if they write to x0, and formally touch x0 twice on read
            if rd == 0 {
                pre_bump_timestamp_and_touch!(ops, 2, 0);
                bump_timestamp!(ops, 2);
                continue;
            }

            match instruction {
                // Arithmetic
                Instruction::Addi(parts) => {
                    let source = load(&mut ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; lea Rd(out), [Rd(source) + sign_extend::<12>(parts.imm())]
                    );
                    record_circuit_type(&mut ops, CounterType::AddSubLui, 1);
                }
                Instruction::Andi(parts) => {
                    load_into(&mut ops, parts.rs1(), out);
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; and Rd(out), sign_extend::<12>(parts.imm())
                    );
                    record_circuit_type(&mut ops, CounterType::ShiftBinaryCsr, 1);
                }
                Instruction::Ori(parts) => {
                    load_into(&mut ops, parts.rs1(), out);
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; or Rd(out), sign_extend::<12>(parts.imm())
                    );
                    record_circuit_type(&mut ops, CounterType::ShiftBinaryCsr, 1);
                }
                Instruction::Xori(parts) => {
                    load_into(&mut ops, parts.rs1(), out);
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; xor Rd(out), sign_extend::<12>(parts.imm())
                    );
                    record_circuit_type(&mut ops, CounterType::ShiftBinaryCsr, 1);
                }
                Instruction::Slti(parts) => {
                    let source = load(&mut ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; cmp Rd(source), sign_extend::<12>(parts.imm())
                        ; setl Rb(out)
                        ; movzx Rd(out), Rb(out)
                    );
                    record_circuit_type(&mut ops, CounterType::BranchSlt, 1);
                }
                Instruction::Sltiu(parts) => {
                    let source = load(&mut ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; cmp Rd(source), sign_extend::<12>(parts.imm())
                        ; setb Rb(out)
                        ; movzx Rd(out), Rb(out)
                    );
                    record_circuit_type(&mut ops, CounterType::BranchSlt, 1);
                }
                Instruction::Slli(parts) => {
                    load_into(&mut ops, parts.rs1(), out);
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; shl Rd(out), parts.shamt() as i8
                    );
                    record_circuit_type(&mut ops, CounterType::ShiftBinaryCsr, 1);
                }
                Instruction::Srli(parts) => {
                    load_into(&mut ops, parts.rs1(), out);
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; shr Rd(out), parts.shamt() as i8
                    );
                    record_circuit_type(&mut ops, CounterType::ShiftBinaryCsr, 1);
                }
                Instruction::Srai(parts) => {
                    load_into(&mut ops, parts.rs1(), out);
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; sar Rd(out), parts.shamt() as i8
                    );
                    record_circuit_type(&mut ops, CounterType::ShiftBinaryCsr, 1);
                }
                Instruction::Lui(parts) => {
                    pre_bump_timestamp_and_touch!(ops, 2, 0);
                    dynasm!(ops
                        ; mov Rd(out), parts.imm() as i32
                    );
                    record_circuit_type(&mut ops, CounterType::AddSubLui, 1);
                }
                Instruction::Auipc(parts) => {
                    pre_bump_timestamp_and_touch!(ops, 2, 0);
                    dynasm!(ops
                        ; mov Rd(out), (pc + parts.imm()) as i32
                    );
                    record_circuit_type(&mut ops, CounterType::AddSubLui, 1);
                }
                Instruction::Add(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    let other = load_abelian(&mut ops, parts.rs1(), parts.rs2(), out);
                    dynasm!(ops
                        ; add Rd(out), Rd(other)
                    );
                    record_circuit_type(&mut ops, CounterType::AddSubLui, 1);
                }
                Instruction::Sub(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs2(), SCRATCH_REGISTER);
                    load_into(&mut ops, parts.rs1(), out);
                    dynasm!(ops
                        ; sub Rd(out), Rd(SCRATCH_REGISTER)
                    );
                    record_circuit_type(&mut ops, CounterType::AddSubLui, 1);
                }
                Instruction::Slt(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs2(), SCRATCH_REGISTER);
                    load_into(&mut ops, parts.rs1(), out);
                    dynasm!(ops
                        ; cmp Rd(out), Rd(SCRATCH_REGISTER)
                        ; setl Rb(out)
                        ; movzx Rd(out), Rb(out)
                    );
                    record_circuit_type(&mut ops, CounterType::BranchSlt, 1);
                }
                Instruction::Sltu(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs2(), SCRATCH_REGISTER);
                    load_into(&mut ops, parts.rs1(), out);
                    dynasm!(ops
                        ; cmp Rd(out), Rd(SCRATCH_REGISTER)
                        ; setb Rb(out)
                        ; movzx Rd(out), Rb(out)
                    );
                    record_circuit_type(&mut ops, CounterType::BranchSlt, 1);
                }
                Instruction::And(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    let other = load_abelian(&mut ops, parts.rs1(), parts.rs2(), out);
                    dynasm!(ops
                        ; and Rd(out), Rd(other)
                    );
                    record_circuit_type(&mut ops, CounterType::ShiftBinaryCsr, 1);
                }
                Instruction::Or(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    let other = load_abelian(&mut ops, parts.rs1(), parts.rs2(), out);
                    dynasm!(ops
                        ; or Rd(out), Rd(other)
                    );
                    record_circuit_type(&mut ops, CounterType::ShiftBinaryCsr, 1);
                }
                Instruction::Xor(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    let other = load_abelian(&mut ops, parts.rs1(), parts.rs2(), out);
                    dynasm!(ops
                        ; xor Rd(out), Rd(other)
                    );
                    record_circuit_type(&mut ops, CounterType::ShiftBinaryCsr, 1);
                }
                Instruction::Sll(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs2(), x64::Rq::RCX as u8);
                    load_into(&mut ops, parts.rs1(), out);
                    dynasm!(ops
                        ; and rcx, 0x1f
                        ; shl Rd(out), cl
                    );
                    record_circuit_type(&mut ops, CounterType::ShiftBinaryCsr, 1);
                }
                Instruction::Srl(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs2(), x64::Rq::RCX as u8);
                    load_into(&mut ops, parts.rs1(), out);
                    dynasm!(ops
                        ; and rcx, 0x1f
                        ; shr Rd(out), cl
                    );
                    record_circuit_type(&mut ops, CounterType::ShiftBinaryCsr, 1);
                }
                Instruction::Sra(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs2(), x64::Rq::RCX as u8);
                    load_into(&mut ops, parts.rs1(), out);
                    dynasm!(ops
                        ; and rcx, 0x1f
                        ; sar Rd(out), cl
                    );
                    record_circuit_type(&mut ops, CounterType::ShiftBinaryCsr, 1);
                }

                // Loads
                Instruction::Lb(parts) => {
                    emit_runtime_error!(ops);
                    // let address = load(&mut ops, parts.rs1());
                    // dynasm!(ops
                    //     ; movsx Rq(out), Rd(address)
                    //     ; movsx Rd(out), BYTE [rsi + Rq(out) + sign_extend::<12>(parts.imm())]
                    // );
                    record_circuit_type(&mut ops, CounterType::MemSubword, 1);
                }
                Instruction::Lbu(parts) => {
                    emit_runtime_error!(ops);
                    // let address = load(&mut ops, parts.rs1());
                    // dynasm!(ops
                    //     ; movsx Rq(out), Rd(address)
                    //     ; movzx Rd(out), BYTE [rsi + Rq(out) + sign_extend::<12>(parts.imm())]
                    // );
                    record_circuit_type(&mut ops, CounterType::MemSubword, 1);
                }
                Instruction::Lh(parts) => {
                    emit_runtime_error!(ops);
                    // // TODO: exception on misalignment
                    // let address = load(&mut ops, parts.rs1());
                    // dynasm!(ops
                    //     ; movsx Rq(out), Rd(address)
                    //     ; movsx Rd(out), WORD [rsi + Rq(out) + sign_extend::<12>(parts.imm())]
                    // );
                    record_circuit_type(&mut ops, CounterType::MemSubword, 1);
                }
                Instruction::Lhu(parts) => {
                    emit_runtime_error!(ops);
                    // // TODO: exception on misalignment
                    // let address = load(&mut ops, parts.rs1());
                    // dynasm!(ops
                    //     ; movsx Rq(out), Rd(address)
                    //     ; movzx Rd(out), WORD [rsi + Rq(out) + sign_extend::<12>(parts.imm())]
                    // );
                    record_circuit_type(&mut ops, CounterType::MemSubword, 1);
                }
                Instruction::Lw(parts) => {
                    // TODO: exception on misalignment
                    let address = load(&mut ops, parts.rs1());
                    dynasm!(ops
                        ; movsx Rq(SCRATCH_REGISTER), Rd(address)
                        ; add Rq(SCRATCH_REGISTER), sign_extend::<12>(parts.imm()) // compute address, as we will need it a lot
                    );
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    dynasm!(ops
                        ; mov Rd(out), [rsi + Rq(SCRATCH_REGISTER)] // load old value into destination
                        ; mov rdx, [rsi + (MemoryHolder::TIMESTAMPS_OFFSET as i32) + 2 * Rq(SCRATCH_REGISTER)] // reuse RDX for read timestamp
                        ; mov [rsi + (MemoryHolder::TIMESTAMPS_OFFSET as i32) + 2 * Rq(SCRATCH_REGISTER)], r8 // update timestamp
                        ; mov [rdi + r9 * 4], Rd(out) // write value into trace
                        ; mov [rdi + r9 * 8 + (TraceChunk::<TRACE_CHUNK_LEN>::TIMESTAMPS_OFFSET as i32)], rdx // write old value into trace
                    );
                    bump_timestamp!(ops, 1);
                    record_circuit_type(&mut ops, CounterType::MemWord, 1);
                    increment_trace!(ops, pc);
                }

                // Multiplication
                Instruction::Mul(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    let other = load_abelian(&mut ops, parts.rs1(), parts.rs2(), out);
                    dynasm!(ops
                        ; imul Rd(out), Rd(other)
                    );
                    record_circuit_type(&mut ops, CounterType::MulDiv, 1);
                }
                Instruction::Mulh(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs1(), x64::Rq::RAX as u8);
                    let other = load(&mut ops, parts.rs2());
                    dynasm!(ops
                        ; imul Rd(other)
                    );
                    if out != x64::Rq::RDX as u8 {
                        dynasm!(ops
                            ; mov Rd(out), edx
                        );
                    }
                    record_circuit_type(&mut ops, CounterType::MulDiv, 1);
                }
                Instruction::Mulhu(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs1(), x64::Rq::RAX as u8);
                    let other = load(&mut ops, parts.rs2());
                    dynasm!(ops
                        ; mul Rd(other)
                    );
                    if out != x64::Rq::RDX as u8 {
                        dynasm!(ops
                            ; mov Rd(out), edx
                        );
                    }
                    record_circuit_type(&mut ops, CounterType::MulDiv, 1);
                }
                Instruction::Mulhsu(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs2(), SCRATCH_REGISTER);
                    load_into(&mut ops, parts.rs1(), out);
                    dynasm!(ops
                        ; movsx Rq(out), Rd(out)
                        ; imul Rq(out), Rq(SCRATCH_REGISTER)
                        ; shr Rq(out), 32
                    );
                    record_circuit_type(&mut ops, CounterType::MulDiv, 1);
                }
                _ => unreachable!(),
            }

            touch_register_and_bump_timestamp!(ops, rd, 2);
            // trace_register!(ops, out);
            store_result(&mut ops, rd);
            continue;
        }

        match instruction {
            // Control transfer instructions
            Instruction::Jal(parts) => {
                let rd = (raw_instruction >> 7) & 0x1F;
                let out = destination_gpr(rd);
                // No reads (so read x0 twice)
                if rd != 0 {
                    pre_bump_timestamp_and_touch!(ops, 1, 0);
                    dynasm!(ops
                        ; mov Rd(out), (pc + 4) as i32
                    );
                    store_result(&mut ops, rd);
                    pre_bump_timestamp_and_touch!(ops, 1, rd);
                } else {
                    pre_bump_timestamp_and_touch!(ops, 2, 0);
                }

                bump_timestamp!(ops, 2);
                record_circuit_type(&mut ops, CounterType::BranchSlt, 1);

                let offset = sign_extend::<21>(parts.imm());
                let jump_target = pc as i32 + offset;
                if offset == 0 {
                    // An infinite loop is used to signal end of execution
                    dynasm!(ops
                        ; mov ecx, pc as i32 // final PC into RCX
                        ; jmp ->quit
                    );
                } else if jump_target % 4 != 0 {
                    panic!("Unaligned just destination");
                    // emit_runtime_error!(ops)
                } else {
                    if let Some(&label) = instruction_labels.get((jump_target / 4) as usize) {
                        dynasm!(ops
                            ; jmp =>label
                        );
                    } else {
                        panic!("Unknown just destination");
                        // emit_runtime_error!(ops)
                    }
                }
            }
            Instruction::Jalr(parts) => {
                let rd = (raw_instruction >> 7) & 0x1F;
                let out = destination_gpr(rd);
                let offset = sign_extend::<12>(parts.imm());
                touch_register_and_increment_timestamp!(ops, parts.rs1());
                load_into(&mut ops, parts.rs1(), SCRATCH_REGISTER);
                dynasm!(ops
                    ; add Rd(SCRATCH_REGISTER), offset
                    // Must be aligned to an instruction but no need to test the least significant bit,
                    // as it is set to zero according to the specification
                    ; test Rd(SCRATCH_REGISTER), 2
                    ; jnz >misaligned
                    ; shr Rd(SCRATCH_REGISTER), 2
                    ; lea rdx, [->jump_offsets]
                    ; mov rax, [rdx + Rq(SCRATCH_REGISTER) * 8]
                    ; lea rdx, [->start]
                    ; add rdx, rax
                );

                // Return address may not be written into register before jump target is computed,
                // otherwise it could affect the jump target.
                if rd != 0 {
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; mov Rd(out), (pc + 4) as i32
                    );
                    touch_register_and_bump_timestamp!(ops, rd, 2);
                    store_result(&mut ops, rd);
                } else {
                    pre_bump_timestamp_and_touch!(ops, 1, 0);
                    bump_timestamp!(ops, 2);
                }
                record_circuit_type(&mut ops, CounterType::BranchSlt, 1);

                dynasm!(ops
                    ; jmp rdx
                    ; misaligned:
                    ;; emit_runtime_error!(ops)
                );
            }
            Instruction::Beq(parts)
            | Instruction::Bne(parts)
            | Instruction::Blt(parts)
            | Instruction::Bltu(parts)
            | Instruction::Bge(parts)
            | Instruction::Bgeu(parts) => {
                let jump_target = pc as i32 + sign_extend::<13>(parts.imm());
                if jump_target % 4 != 0 {
                    panic!("Unaligned just destination");
                    // emit_runtime_error!(ops);
                } else {
                    let a = load(&mut ops, parts.rs1());
                    load_into(&mut ops, parts.rs2(), SCRATCH_REGISTER);

                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());

                    touch_register_and_bump_timestamp!(ops, 0, 2);
                    record_circuit_type(&mut ops, CounterType::BranchSlt, 1);

                    if let Some(&label) = instruction_labels.get((jump_target / 4) as usize) {
                        dynasm!(ops
                            ; cmp Rd(a), Rd(SCRATCH_REGISTER)
                        );
                        match instruction {
                            Instruction::Beq(_) => {
                                dynasm!(ops
                                    ; je =>label
                                );
                            }
                            Instruction::Bne(_) => {
                                dynasm!(ops
                                    ; jne =>label
                                );
                            }
                            Instruction::Blt(_) => {
                                dynasm!(ops
                                    ; jl =>label
                                );
                            }
                            Instruction::Bltu(_) => {
                                dynasm!(ops
                                    ; jb =>label
                                );
                            }
                            Instruction::Bge(_) => {
                                dynasm!(ops
                                    ; jge =>label
                                );
                            }
                            Instruction::Bgeu(_) => {
                                dynasm!(ops
                                    ; jae =>label
                                );
                            }
                            _ => unreachable!(),
                        }
                    } else {
                        emit_runtime_error!(ops)
                    }

                    // print_registers!(ops);
                }
            }

            // Stores
            Instruction::Sb(parts) => {
                emit_runtime_error!(ops);
                // let address = load(&mut ops, parts.rs1());
                // dynasm!(ops
                //     ; movsx Rq(SCRATCH_REGISTER), Rd(address)
                // );
                // let value = load(&mut ops, parts.rs2());
                // dynasm!(ops
                //     ; mov [rsi + Rq(SCRATCH_REGISTER) + sign_extend::<12>(parts.imm())], Rb(value)
                // );
                // trace_zero!(ops);
                record_circuit_type(&mut ops, CounterType::MemSubword, 1);
            }
            Instruction::Sh(parts) => {
                emit_runtime_error!(ops);
                // // TODO: exception on misalignment
                // let address = load(&mut ops, parts.rs1());
                // dynasm!(ops
                //     ; movsx Rq(SCRATCH_REGISTER), Rd(address)
                // );
                // let value = load(&mut ops, parts.rs2());
                // dynasm!(ops
                //     ; mov [rsi + Rq(SCRATCH_REGISTER) + sign_extend::<12>(parts.imm())], Rw(value)
                // );
                // trace_zero!(ops);
                record_circuit_type(&mut ops, CounterType::MemSubword, 1);
            }
            Instruction::Sw(parts) => {
                // TODO: exception on misalignment
                let address = load(&mut ops, parts.rs1());
                dynasm!(ops
                    ; movsx Rq(SCRATCH_REGISTER), Rd(address)
                    ; add Rq(SCRATCH_REGISTER), sign_extend::<12>(parts.imm()) // compute address, as we will need it a lot
                );
                let value = load(&mut ops, parts.rs2());
                touch_register_and_increment_timestamp!(ops, parts.rs1());
                touch_register_and_increment_timestamp!(ops, parts.rs2());
                dynasm!(ops
                    // this sequence of operations is: read old value and timestamp, save it, write new value and timestamp
                    ; mov eax, [rsi + Rq(SCRATCH_REGISTER)] // load old value into RAX
                    ; mov [rsi + Rq(SCRATCH_REGISTER)], Rd(value) // store new value
                    ; mov Rq(value), [rsi + (MemoryHolder::TIMESTAMPS_OFFSET as i32) + 2 * Rq(SCRATCH_REGISTER)] // reuse Rq(value) for read timestamp
                    ; mov [rsi + (MemoryHolder::TIMESTAMPS_OFFSET as i32) + 2 * Rq(SCRATCH_REGISTER)], r8 // update timestamp
                    ; mov [rdi + r9 * 4], eax // write old value into trace
                    ; mov [rdi + r9 * 8 + (TraceChunk::<TRACE_CHUNK_LEN>::TIMESTAMPS_OFFSET as i32)], Rq(value) // write timestamp value into trace
                );
                bump_timestamp!(ops, 2);
                record_circuit_type(&mut ops, CounterType::MemWord, 1);
                increment_trace!(ops, pc);
            }

            Instruction::Csrrw(parts) => {
                todo!();
            }

            // Instruction::Csrrw(parts) => match parts.csr() {
            //     todo!();

            //     NON_DETERMINISM_CSR => {
            //         if rd != 0 {
            //             before_call!(ops);
            //             dynasm!(ops
            //                 ; mov rax, QWORD Context::<N>::read_nondeterminism as _
            //                 ; call rax
            //             );
            //             after_call!(ops);
            //             dynasm!(ops
            //                 ; mov Rd(out), eax
            //             );
            //             trace_register!(ops, out);
            //             store_result(&mut ops, rd);
            //         } else {
            //             trace_zero!(ops);
            //         }
            //         if parts.rs1() != 0 {
            //             load_into(&mut ops, parts.rs1(), SCRATCH_REGISTER);
            //             before_call!(ops);
            //             dynasm!(ops
            //                 ; mov rax, QWORD Context::<N>::write_nondeterminism as _
            //                 ; mov esi, Rd(SCRATCH_REGISTER)
            //                 ; call rax
            //             );
            //             after_call!(ops);
            //         }
            //     }
            //     csr => {
            //         let function = match csr {
            //             BLAKE2_ROUND_FUNCTION_WITH_EXTENDED_CONTROL_ACCESS_ID => {
            //                 Context::<N>::process_csr::<
            //                     BLAKE2_ROUND_FUNCTION_WITH_EXTENDED_CONTROL_ACCESS_ID,
            //                 > as _
            //             }
            //             U256_OPS_WITH_CONTROL_ACCESS_ID => {
            //                 Context::<N>::process_csr::<U256_OPS_WITH_CONTROL_ACCESS_ID> as _
            //             }
            //             x => {
            //                 emit_runtime_error!(ops);
            //                 continue;
            //             }
            //         };

            //         dynasm!(ops
            //             ; sub rsp, 32 * 4
            //             ; mov DWORD [rsp], 0
            //         );
            //         for i in 1..32 {
            //             let reg = load(&mut ops, i);
            //             dynasm!(ops
            //                 ; mov [rsp + 4 * i as i32], Rd(reg)
            //             );
            //         }

            //         load_into(&mut ops, parts.rs1(), x64::Rq::RCX as u8);

            //         dynasm!(ops
            //             ; mov rdx, rsp

            //             ; push rdi
            //             ; push rsi
            //             ; push r8
            //             ; push r9

            //             ; mov rax, QWORD function
            //             ; mov esi, ecx
            //             ; call rax

            //             ; pop r9
            //             ; pop r8
            //             ; pop rsi
            //             ; pop rdi
            //         );

            //         for i in 1..32 {
            //             let out = destination_gpr(i);
            //             dynasm!(ops
            //                 ; mov Rd(out), [rsp + 4 * i as i32]
            //             );
            //             store_result(&mut ops, i);
            //         }
            //         dynasm!(ops
            //             ; add rsp, 32 * 4
            //         );

            //         if rd != 0 {
            //             dynasm!(ops
            //                 ; mov Rd(out), eax
            //             );
            //             trace_register!(ops, out);
            //             store_result(&mut ops, rd);
            //         } else {
            //             trace_zero!(ops);
            //         }
            //     }
            // },

            _ => {
                emit_runtime_error!(ops)
            }
        }
    }

    // if we even come here without exit condition - it's an error
    emit_runtime_error!(ops);

    // now service functions
    dynasm!(ops
        ; ->jump_offsets:
        ; .bytes jump_offsets.into_iter().flat_map(|x| x.to_le_bytes())

        ; ->exit_with_error:
        ; mov rax, QWORD print_complaint as _
        ; call rax
    );

    let receive_trace_fn = Context::<N>::receive_trace;
    receive_trace!(ops, receive_trace_fn);

    let quit_trace_fn = Context::<N>::receive_final_trace_piece;
    quit!(ops, quit_trace_fn);

    let code = ops.finalize().unwrap();

    // let assembly = unsafe {
    //     core::slice::from_raw_parts(code.ptr(start), code.len())
    // };
    // view_assembly(&assembly[..100], start.0);

    let mut context = Context {
        non_determinism_source,
        trace_len: 0,
        final_timestamp: 0,
        final_pc: 0,
    };

    let mut memory: Box<MemoryHolder> = unsafe {
        let mut memory: Box<MemoryHolder> = Box::new_uninit().assume_init();
        for i in 0..initial_memory.len() {
            core::ptr::write(addr_of_mut!(memory.memory).cast::<u32>().add(i), initial_memory[i]);
        }
        for i in initial_memory.len()..NUM_RAM_WORDS {
            core::ptr::write(addr_of_mut!(memory.memory).cast::<u32>().add(i), 0);
        }
        for i in 0..NUM_RAM_WORDS {
            core::ptr::write(addr_of_mut!(memory.timestamps).cast::<u64>().add(i), 0);
        }

        memory
    };

    let mut trace: Box<TraceChunk<TRACE_CHUNK_LEN>> = unsafe {
        let trace = Box::new_uninit().assume_init();

        trace
    };

    // unsafe {
    //     view_assembly(
    //         core::slice::from_raw_parts(code.ptr(start), code.len()),
    //         start.0,
    //     );
    // };

    let before = std::time::Instant::now();

    let context_ref_mut = &mut context;
    dbg!((context_ref_mut as *mut Context<N>).addr());

    let run_program: extern "sysv64" fn(&mut TraceChunk<TRACE_CHUNK_LEN>, &mut MemoryHolder, &mut Context<N>) =
        unsafe { std::mem::transmute(code.ptr(start)) };

    run_program(trace.as_mut(), memory.as_mut(), context_ref_mut);

    let final_timestamp = context.final_timestamp;
    assert_eq!(final_timestamp % TIMESTAMP_STEP, 0);
    use common_constants::*;
    let num_instructions = (final_timestamp - INITIAL_TIMESTAMP) / TIMESTAMP_STEP;
    println!(
        "Frequency is {} MHz over {} instructions",
        (num_instructions as f64)/(before.elapsed().as_micros() as f64),
        num_instructions
    );
}

#[repr(C)]
pub struct Context<'a, N: NonDeterminismCSRSource<RamWithRomRegion<{common_constants::rom::ROM_SECOND_WORD_BITS }>>> {
    non_determinism_source: &'a mut N,
    trace_len: usize,
    final_timestamp: TimestampScalar,
    final_pc: u32,
}

impl<'a, N: NonDeterminismCSRSource<RamWithRomRegion<{common_constants::rom::ROM_SECOND_WORD_BITS }>>> Context<'a, N> {
    extern "sysv64" fn read_nondeterminism(&mut self) -> u32 {
        self.non_determinism_source.read()
    }
    extern "sysv64" fn write_nondeterminism(&mut self, value: u32, memory: &[u32; 1 << 28]) {
        todo!();
        // self.non_determinism_source
            // .write_with_memory_access(&self.memory, value);
    }
    extern "sysv64" fn process_csr<const CSR_NUMBER: u32>(
        &mut self,
        registers: &mut [u32; 32],
    ) -> u32 {
        todo!();
    }

    extern "sysv64" fn receive_trace(&mut self, trace_piece: &mut TraceChunk<TRACE_CHUNK_LEN>, machine_state: &MachineState) -> *mut TraceChunk<TRACE_CHUNK_LEN> {
        debug_assert!((machine_state as *const MachineState).is_aligned_to(core::mem::align_of::<MachineState>()));
        dbg!(machine_state);
        dbg!(self.trace_len);
        use common_constants::*;
        println!("{} cycles passed", (machine_state.timestamp - INITIAL_TIMESTAMP) / TIMESTAMP_STEP);
        println!("PC = 0x{:08x}", machine_state.pc_or_partial_trace_chunk);
        self.trace_len += TRACE_CHUNK_LEN;

        dbg!(&*trace_piece);

        trace_piece as *mut TraceChunk<TRACE_CHUNK_LEN>
    }

    extern "sysv64" fn receive_final_trace_piece(&mut self, trace_piece: &mut TraceChunk<TRACE_CHUNK_LEN>, machine_state: &MachineState, final_pc: u32) {
        debug_assert!((machine_state as *const MachineState).is_aligned_to(core::mem::align_of::<MachineState>()));
        dbg!(machine_state);
        use common_constants::*;
        println!("In total {} cycles passed", (machine_state.timestamp - INITIAL_TIMESTAMP) / TIMESTAMP_STEP);
        println!("Final trace chunk len = {}", machine_state.pc_or_partial_trace_chunk);
        println!("Final PC = 0x{:08x}", final_pc);
        self.trace_len += machine_state.pc_or_partial_trace_chunk as usize;
        self.final_timestamp = machine_state.timestamp;
        self.final_pc = final_pc;
    }
}

extern "sysv64" fn print_registers(registers: &mut [u32; 32]) {
    println!("{registers:?}");
}

extern "sysv64" fn print_complaint() {
    println!("Runtime error!")
}

fn sign_extend<const SOURCE_BITS: u8>(x: u32) -> i32 {
    let shift = 32 - SOURCE_BITS;
    i32::from_ne_bytes((x << shift).to_ne_bytes()) >> shift
}

fn view_assembly(assembly: &[u8], start: usize) {
    /// Print register names
    fn reg_names(cs: &Capstone, regs: &[RegId]) -> String {
        let names: Vec<String> = regs.iter().map(|&x| cs.reg_name(x).unwrap()).collect();
        names.join(", ")
    }

    /// Print instruction group names
    fn group_names(cs: &Capstone, regs: &[InsnGroupId]) -> String {
        let names: Vec<String> = regs.iter().map(|&x| cs.group_name(x).unwrap()).collect();
        names.join(", ")
    }

    use capstone::*;
    use capstone::arch::*;

    let cs = Capstone::new()
        .x86()
        .mode(arch::x86::ArchMode::Mode64)
        .syntax(arch::x86::ArchSyntax::Att)
        .detail(true)
        .build()
        .expect("Failed to create Capstone object");

    let insns = cs.disasm_all(assembly, start as u64)
        .expect("Failed to disassemble");
    println!("Found {} instructions", insns.len());
    for i in insns.as_ref() {
        println!();
        println!("{}", i);

        let detail: InsnDetail = cs.insn_detail(&i).expect("Failed to get insn detail");
        let arch_detail: ArchDetail = detail.arch_detail();
        let ops = arch_detail.operands();

        let output: &[(&str, String)] = &[
            ("insn id:", format!("{:?}", i.id().0)),
            ("bytes:", format!("{:?}", i.bytes())),
            ("read regs:", reg_names(&cs, detail.regs_read())),
            ("write regs:", reg_names(&cs, detail.regs_write())),
            ("insn groups:", group_names(&cs, detail.groups())),
        ];

        for &(ref name, ref message) in output.iter() {
            println!("{:4}{:12} {}", "", name, message);
        }

        println!("{:4}operands: {}", "", ops.len());
        for op in ops {
            println!("{:8}{:?}", "", op);
        }
    }
}