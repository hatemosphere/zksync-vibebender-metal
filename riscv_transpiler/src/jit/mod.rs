use crate::vm::*;
use common_constants::*;
use std::collections::HashSet;
use std::{mem::offset_of, ptr::addr_of_mut};

mod delegations;

pub use self::delegations::*;

#[cfg(all(target_arch = "x86_64", feature = "jit"))]
mod impls;

#[cfg(all(target_arch = "x86_64", feature = "jit"))]
pub use self::impls::*;

#[cfg(all(target_arch = "x86_64", feature = "jit", test))]
mod tests;

const MAX_RAM_SIZE: usize = 1 << 30; // 1 Gb, as we want to avoid having separate pointers to RAM (that we want to have continuous to perform very simple read/writes), and timestamp bookkeping space

const RAM_SIZE: usize = 1 << 30;
const NUM_RAM_WORDS: usize = RAM_SIZE / core::mem::size_of::<u32>();

// We will measure trace chunk in a number of memory accesses and not in a almost fixed number of cycles that did pass between them.
// At most we extend a chunk by the number of accesses in delegation
pub const TRACE_CHUNK_LEN: usize = 1 << 20;
pub const MAX_TRACE_CHUNK_LEN: usize = const {
    let mut max = core::cmp::max(24 + 16, 31 * 2);
    max = core::cmp::max(max, 8 + 8 + 1);

    TRACE_CHUNK_LEN + max
};

#[repr(C, align(8))]
#[derive(Debug)]
pub struct TraceChunk {
    pub values: [u32; MAX_TRACE_CHUNK_LEN],
    pub timestamps: [TimestampScalar; MAX_TRACE_CHUNK_LEN],
    pub len: u64,
}

impl TraceChunk {
    pub fn empty() -> Self {
        Self {
            values: [0u32; MAX_TRACE_CHUNK_LEN],
            timestamps: [0; MAX_TRACE_CHUNK_LEN],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn add_element(&mut self, value: u32, ts: TimestampScalar) {
        assert!((self.len as usize) < MAX_TRACE_CHUNK_LEN);
        unsafe {
            self.values.as_mut_ptr().add(self.len as usize).write(value);
            self.timestamps
                .as_mut_ptr()
                .add(self.len as usize)
                .write(ts);

            self.len += 1;
        }
    }

    pub fn data(&'_ self) -> (&'_ [u32], &'_ [TimestampScalar]) {
        unsafe {
            let values =
                core::slice::from_raw_parts(self.values.as_ptr().cast::<u32>(), self.len as usize);
            let timestamps = core::slice::from_raw_parts(
                self.timestamps.as_ptr().cast::<TimestampScalar>(),
                self.len as usize,
            );

            (values, timestamps)
        }
    }

    const TIMESTAMPS_OFFSET: usize = offset_of!(Self, timestamps);
    const LEN_OFFSET: usize = offset_of!(Self, len);
}

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
#[derive(Debug, Clone, Copy)]
pub struct MachineState {
    registers: [u32; 32], // aligned at 16, so we can write XMMs directly into the stack
    register_timestamps: [TimestampScalar; 32],
    counters: [u32; MAX_NUM_COUNTERS],
    pc: u32,
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
    const PC_OFFSET: usize = offset_of!(Self, pc);
    const TIMESTAMP_OFFSET: usize = offset_of!(Self, timestamp);
    const CONTEXT_PTR_OFFSET: usize = offset_of!(Self, context_ptr);
}

#[repr(C, align(8))]
pub struct MemoryHolder {
    memory: [u32; NUM_RAM_WORDS],
    timestamps: [TimestampScalar; NUM_RAM_WORDS],
}

impl MemoryHolder {
    pub fn empty() -> Self {
        Self {
            memory: [0u32; NUM_RAM_WORDS],
            timestamps: [0; NUM_RAM_WORDS],
        }
    }

    const TIMESTAMPS_OFFSET: usize = offset_of!(Self, timestamps);
}
