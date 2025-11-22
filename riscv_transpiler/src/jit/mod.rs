use crate::vm::*;
use common_constants::*;
use std::alloc::Allocator;
use std::collections::HashSet;
use std::ptr::NonNull;
use std::{mem::offset_of, ptr::addr_of_mut};

mod delegations;

pub use self::delegations::*;

pub mod minimal_tracer;

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
        debug_assert!((self.len as usize) < MAX_TRACE_CHUNK_LEN);
        unsafe {
            self.values.as_mut_ptr().add(self.len as usize).write(value);
            self.timestamps
                .as_mut_ptr()
                .add(self.len as usize)
                .write(ts);

            self.len += 1;
        }
    }

    #[inline(always)]
    pub fn append_arbitrary_value(&mut self, value: u32) {
        self.add_element(value, 0);
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

    pub fn reset(&mut self) {
        self.values.fill(0);
        self.timestamps.fill(0);
        self.len = 0;
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

    pub fn initial() -> Self {
        Self {
            registers: [0; 32],
            register_timestamps: [0; 32],
            counters: [0; MAX_NUM_COUNTERS],
            pc: 0,
            timestamp: INITIAL_TIMESTAMP,
            context_ptr: core::ptr::dangling_mut(),
        }
    }
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

    pub fn reset<A: Allocator>(this: &mut Box<Self, A>) {
        this.memory.fill(0);
        this.timestamps.fill(0);
    }

    pub fn collect_inits_and_teardowns<A: Allocator + Clone + Send + Sync>(
        &self,
        worker: &worker::Worker,
        allocator: A,
    ) -> Vec<Vec<(u32, (TimestampScalar, u32)), A>> {
        // parallel collect
        // first we will walk over access_bitmask and collect subparts
        let mut chunks: Vec<Vec<(u32, (TimestampScalar, u32)), A>> =
            vec![Vec::new_in(allocator).clone(); worker.get_num_cores()];
        let mut dst = &mut chunks[..];
        worker.scope(NUM_RAM_WORDS, |scope, geometry| {
            for thread_idx in 0..geometry.len() {
                let chunk_size = geometry.get_chunk_size(thread_idx);
                let chunk_start = geometry.get_chunk_start_pos(thread_idx);
                let range = chunk_start..(chunk_start + chunk_size);
                let (el, rest) = dst.split_at_mut(1);
                dst = rest;
                let values = &self.memory[range.clone()];
                let timestamps = &self.timestamps[range];

                worker::Worker::smart_spawn(scope, thread_idx == geometry.len() - 1, move |_| {
                    let el = &mut el[0];
                    let mut address = chunk_start * core::mem::size_of::<u32>();
                    for (idx, ts) in timestamps.iter().enumerate() {
                        if *ts != 0 {
                            let mut word_value = unsafe { *values.get_unchecked(idx) };
                            // we mask ROM region to be zero-valued
                            if address < common_constants::rom::ROM_BYTE_SIZE {
                                word_value = 0;
                            }
                            let last_timestamp: TimestampScalar = *ts;
                            el.push((address as u32, (last_timestamp, word_value)));
                        }

                        address += core::mem::size_of::<u32>();
                    }
                });
            }
        });

        chunks
    }
}

#[derive(Debug)]
pub struct ReplayerMemChunks<'a> {
    pub chunks: &'a mut [(&'a [u32], &'a [TimestampScalar])],
}

impl<'a> RamPeek for ReplayerMemChunks<'a> {
    #[inline(always)]
    fn peek_word(&self, address: u32) -> u32 {
        debug_assert_eq!(address % 4, 0);
        debug_assert!(self.chunks.len() > 0);
        unsafe {
            let value = *self.chunks.get_unchecked(0).0.get_unchecked(0);

            value
        }
    }
}

impl<'a> RAM for ReplayerMemChunks<'a> {
    const REPLAY_NON_DETERMINISM_VIA_RAM_STUB: bool = true;

    #[track_caller]
    #[inline(always)]
    fn read_word(&mut self, address: u32, timestamp: TimestampScalar) -> (TimestampScalar, u32) {
        debug_assert_eq!(address % 4, 0);
        debug_assert!(self.chunks.len() > 0);
        unsafe {
            let src = self.chunks.get_unchecked_mut(0);
            let value = *src.0.get_unchecked(0);
            let read_timestamp = *src.1.get_unchecked(0);
            let next_values = src.0.get_unchecked(1..);
            let next_timestamps = src.1.get_unchecked(1..);
            if next_values.len() > 0 {
                *src = (next_values, next_timestamps);
            } else {
                self.chunks = core::mem::transmute(self.chunks.get_unchecked_mut(1..));
            }

            debug_assert!(read_timestamp < timestamp, "trying to read replay log at address 0x{:08x} with timestamp {}, but read timestamp is {}", address, timestamp, read_timestamp);

            // println!("Read at address 0x{:08x} at timestamp {} into value {} and read timestamp {}", address, timestamp, value, read_timestamp);
            (read_timestamp, value)
        }
    }

    #[inline(always)]
    fn mask_read_for_witness(&self, address: &mut u32, value: &mut u32) {
        debug_assert_eq!(*address % 4, 0);
        if (*address as usize) < common_constants::rom::ROM_BYTE_SIZE {
            // NOTE: we no longer mask an address, just a value as it's only initialized to
            // 0 via inits, and can not be writen over by circuits
            // *address = 0u32;
            *value = 0u32;
        }
    }

    #[inline(always)]
    fn write_word(
        &mut self,
        address: u32,
        _word: u32,
        timestamp: TimestampScalar,
    ) -> (TimestampScalar, u32) {
        debug_assert_eq!(address % 4, 0);
        debug_assert!(self.chunks.len() > 0);
        unsafe {
            let src = self.chunks.get_unchecked_mut(0);
            let value = *src.0.get_unchecked(0);
            let read_timestamp = *src.1.get_unchecked(0);
            let next_values = src.0.get_unchecked(1..);
            let next_timestamps = src.1.get_unchecked(1..);
            if next_values.len() > 0 {
                *src = (next_values, next_timestamps);
            } else {
                self.chunks = core::mem::transmute(self.chunks.get_unchecked_mut(1..));
            }

            debug_assert!(read_timestamp < timestamp, "trying to read replay log at address 0x{:08x} with timestamp {}, but read timestamp is {}", address, timestamp, read_timestamp);

            // println!("Read at address 0x{:08x} at timestamp {} into value {} and read timestamp {}", address, timestamp, value, read_timestamp);
            (read_timestamp, value)
        }
    }

    #[inline(always)]
    fn skip_if_replaying(&mut self, num_snapshots: usize) {
        unsafe {
            let src = self.chunks.get_unchecked_mut(0);
            debug_assert!(src.0.len() >= num_snapshots);
            debug_assert!(src.1.len() >= num_snapshots);
            let next_values = src.0.get_unchecked(num_snapshots..);
            let next_timestamps = src.1.get_unchecked(num_snapshots..);
            if next_values.len() > 0 {
                *src = (next_values, next_timestamps);
            } else {
                self.chunks = core::mem::transmute(self.chunks.get_unchecked_mut(1..));
            }
        }
    }
}

pub trait ContextImpl {
    fn read_nondeterminism(&mut self) -> u32;

    fn write_nondeterminism(&mut self, value: u32, memory: &[u32; RAM_SIZE]);

    fn receive_trace(
        &mut self,
        trace_piece: NonNull<TraceChunk>,
        machine_state: &MachineState,
    ) -> NonNull<TraceChunk>;

    fn receive_final_trace_piece(
        &mut self,
        trace_piece: NonNull<TraceChunk>,
        machine_state: &MachineState,
    );

    fn take_final_state(&mut self) -> Option<MachineState>;
    fn final_state_ref(&'_ self) -> Option<&'_ MachineState>;
}

pub struct DefaultContextImpl<'a, N: NonDeterminismCSRSource> {
    non_determinism_source: &'a mut N,
    trace_len: usize,
    final_state: Option<MachineState>,
}

impl<'a, N: NonDeterminismCSRSource> ContextImpl for DefaultContextImpl<'a, N> {
    fn read_nondeterminism(&mut self) -> u32 {
        self.non_determinism_source.read()
    }

    fn write_nondeterminism(&mut self, value: u32, memory: &[u32; RAM_SIZE]) {
        self.non_determinism_source
            .write_with_memory_access(memory, value);
    }

    fn receive_trace(
        &mut self,
        mut trace_chunk: NonNull<TraceChunk>,
        machine_state: &MachineState,
    ) -> NonNull<TraceChunk> {
        debug_assert!((machine_state as *const MachineState)
            .is_aligned_to(core::mem::align_of::<MachineState>()));
        debug_assert!((trace_chunk.as_ptr() as *const TraceChunk)
            .is_aligned_to(core::mem::align_of::<TraceChunk>()));

        let trace_piece = unsafe { trace_chunk.as_mut() };
        assert!((trace_piece.len as usize) >= TRACE_CHUNK_LEN);
        assert!((trace_piece.len as usize) <= MAX_TRACE_CHUNK_LEN);
        // println!(
        //     "Received snapshot of length {} after {} cycles",
        //     trace_piece.len,
        //     (machine_state.timestamp - INITIAL_TIMESTAMP) / TIMESTAMP_STEP
        // );
        self.trace_len += trace_piece.len as usize;

        #[cfg(debug_assertions)]
        {
            for i in (trace_piece.len as usize)..MAX_TRACE_CHUNK_LEN {
                assert_eq!(
                    trace_piece.values[i], 0,
                    "invalid canary value at slot {}",
                    i
                );
                assert_eq!(
                    trace_piece.timestamps[i], 0,
                    "invalid canary timestamp at slot {}",
                    i
                );
            }

            trace_piece.values.fill(0);
            trace_piece.timestamps.fill(0);
        }
        trace_piece.len = 0;

        trace_chunk
    }

    fn receive_final_trace_piece(
        &mut self,
        mut trace_chunk: NonNull<TraceChunk>,
        machine_state: &MachineState,
    ) {
        println!("Execution completed");
        debug_assert!((machine_state as *const MachineState)
            .is_aligned_to(core::mem::align_of::<MachineState>()));
        debug_assert!((trace_chunk.as_ptr() as *const TraceChunk)
            .is_aligned_to(core::mem::align_of::<TraceChunk>()));
        let trace_piece = unsafe { trace_chunk.as_mut() };
        // println!(
        //     "In total {} cycles passed",
        //     (machine_state.timestamp - INITIAL_TIMESTAMP) / TIMESTAMP_STEP
        // );
        // println!("Final trace chunk len = {}", trace_piece.len);
        // println!("Final PC = 0x{:08x}", machine_state.pc);
        self.trace_len += trace_piece.len as usize;
        self.final_state = Some(*machine_state);
    }

    fn take_final_state(&mut self) -> Option<MachineState> {
        self.final_state.take()
    }
    fn final_state_ref(&'_ self) -> Option<&'_ MachineState> {
        self.final_state.as_ref()
    }
}
