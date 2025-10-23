use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DelegationsCounters {
    pub non_determinism_reads: usize,
    pub blake_calls: usize,
    pub bigint_calls: usize,
    pub keccak_calls: usize,
}

impl Counters for DelegationsCounters {
    #[inline(always)]
    fn bump_bigint(&mut self) {
        self.bigint_calls += 1;
    }
    #[inline(always)]
    fn bump_blake2_round_function(&mut self) {
        self.blake_calls += 1;
    }
    #[inline(always)]
    fn bump_keccak_special5(&mut self) {
        self.keccak_calls += 1;
    }
    #[inline(always)]
    fn bump_non_determinism(&mut self) {
        self.non_determinism_reads += 1;
    }
    #[inline(always)]
    fn log_circuit_family<const FAMILY: u8>(&mut self) {}
    #[inline(always)]
    fn get_calls_to_circuit_family<const FAMILY: u8>(&self) -> usize {
        0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DelegationsAndFamiliesCounters {
    pub non_determinism_reads: usize,
    pub blake_calls: usize,
    pub bigint_calls: usize,
    pub keccak_calls: usize,
    pub add_sub_family: usize,
    pub binary_shift_csr_family: usize,
    pub slt_branch_family: usize,
    pub mul_div_family: usize,
    pub word_size_mem_family: usize,
    pub subword_size_mem_family: usize,
}

impl Counters for DelegationsAndFamiliesCounters {
    #[inline(always)]
    fn bump_bigint(&mut self) {
        self.bigint_calls += 1;
    }
    #[inline(always)]
    fn bump_blake2_round_function(&mut self) {
        self.blake_calls += 1;
    }
    #[inline(always)]
    fn bump_keccak_special5(&mut self) {
        self.keccak_calls += 1;
    }
    #[inline(always)]
    fn bump_non_determinism(&mut self) {
        self.non_determinism_reads += 1;
    }
    #[inline(always)]
    fn log_circuit_family<const FAMILY: u8>(&mut self) {
        if const { FAMILY == ADD_SUB_LUI_AUIPC_MOP_CIRCUIT_FAMILY_IDX } {
            self.add_sub_family += 1;
        } else if const { FAMILY == JUMP_BRANCH_SLT_CIRCUIT_FAMILY_IDX } {
            self.slt_branch_family += 1;
        } else if const { FAMILY == SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX } {
            self.binary_shift_csr_family += 1;
        } else if const { FAMILY == MUL_DIV_CIRCUIT_FAMILY_IDX } {
            self.mul_div_family += 1;
        } else if const { FAMILY == LOAD_STORE_WORD_ONLY_CIRCUIT_FAMILY_IDX } {
            self.word_size_mem_family += 1;
        } else if const { FAMILY == LOAD_STORE_SUBWORD_ONLY_CIRCUIT_FAMILY_IDX } {
            self.subword_size_mem_family += 1;
        } else {
            unsafe { core::hint::unreachable_unchecked() }
        }
    }
    #[inline(always)]
    fn get_calls_to_circuit_family<const FAMILY: u8>(&self) -> usize {
        if const { FAMILY == ADD_SUB_LUI_AUIPC_MOP_CIRCUIT_FAMILY_IDX } {
            self.add_sub_family
        } else if const { FAMILY == JUMP_BRANCH_SLT_CIRCUIT_FAMILY_IDX } {
            self.slt_branch_family
        } else if const { FAMILY == SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX } {
            self.binary_shift_csr_family
        } else if const { FAMILY == MUL_DIV_CIRCUIT_FAMILY_IDX } {
            self.mul_div_family
        } else if const { FAMILY == LOAD_STORE_WORD_ONLY_CIRCUIT_FAMILY_IDX } {
            self.word_size_mem_family
        } else if const { FAMILY == LOAD_STORE_SUBWORD_ONLY_CIRCUIT_FAMILY_IDX } {
            self.subword_size_mem_family
        } else {
            unsafe { core::hint::unreachable_unchecked() }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimpleSnapshot<C: Counters> {
    pub state: State<C>,
    pub last_zero_address_read_timestamp: TimestampScalar,
    pub non_determinism_reads_start: usize,
    pub non_determinism_reads_end: usize,
    pub memory_reads_start: usize,
    pub memory_reads_end: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartialSnapshot {
    pub last_zero_address_read_timestamp: TimestampScalar,
    pub non_determinism_reads_offset: usize,
    pub memory_reads_offset: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BiVec<T: Sized> {
    buffers: Vec<Vec<T>>,
    inner_capacity: usize,
    total_len: usize,
}

impl<T: Sized> BiVec<T> {
    pub fn new_with_capacities(inner_capacity: usize, outer_capacity: usize) -> Self {
        assert!(outer_capacity > 0);
        let mut buffers = Vec::with_capacity(outer_capacity);
        buffers.push(Vec::with_capacity(inner_capacity));

        Self {
            buffers,
            inner_capacity,
            total_len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, value: T) {
        unsafe {
            let dst = self.buffers.last_mut().unwrap_unchecked();
            if dst.len() == self.inner_capacity {
                let mut next = Vec::with_capacity(self.inner_capacity);
                next.push_within_capacity(value).unwrap_unchecked();
                self.buffers.push(next);
            } else {
                dst.push_within_capacity(value).unwrap_unchecked();
            }
            self.total_len += 1;
        }
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.total_len
    }

    pub fn make_range<'a>(&'a self, range: core::ops::Range<usize>) -> Vec<&'a [T]>
    where
        T: 'a,
    {
        assert!(range.end <= self.total_len);
        let (s_o, s_i) = (
            range.start / self.inner_capacity,
            range.start % self.inner_capacity,
        );
        let (e_o, e_i) = (
            range.end / self.inner_capacity,
            range.end % self.inner_capacity,
        );

        let mut result = Vec::with_capacity(e_o + 1 - s_o);
        let mut inner_offset = s_i;
        for i in s_o..=e_o {
            if i == e_o {
                if e_i != 0 {
                    result.push(&self.buffers[i][inner_offset..e_i]);
                }
            } else {
                result.push(&self.buffers[i][inner_offset..]);
                inner_offset = 0;
            }
        }

        result
    }
}

pub struct SimpleSnapshotter<C: Counters, const ROM_BOUND_SECOND_WORD_BITS: usize> {
    pub current_partial_snapshot: PartialSnapshot,
    pub snapshots: Vec<SimpleSnapshot<C>>,
    pub last_zero_address_read_timestamp: TimestampScalar,
    pub reads_buffer: BiVec<(u32, (u32, u32))>,
    pub non_determinism_reads_buffer: BiVec<u32>,
    pub initial_snapshot: SimpleSnapshot<C>,
}

impl<C: Counters, const ROM_BOUND_SECOND_WORD_BITS: usize>
    SimpleSnapshotter<C, ROM_BOUND_SECOND_WORD_BITS>
{
    const DEFAULT_INNER_CAPACITY: usize = 1 << 24;

    pub fn new_with_cycle_limit(limit: usize, period: usize, initial_state: State<C>) -> Self {
        let initial_snapshot = SimpleSnapshot {
            state: initial_state,
            last_zero_address_read_timestamp: 0,
            non_determinism_reads_start: 0,
            non_determinism_reads_end: 0,
            memory_reads_start: 0,
            memory_reads_end: 0,
        };

        let outer_capacity = limit.div_ceil(Self::DEFAULT_INNER_CAPACITY);

        Self {
            current_partial_snapshot: PartialSnapshot {
                last_zero_address_read_timestamp: 0,
                non_determinism_reads_offset: 0,
                memory_reads_offset: 0,
            },
            snapshots: Vec::with_capacity(limit.div_ceil(period)),
            last_zero_address_read_timestamp: 0,
            reads_buffer: BiVec::new_with_capacities(Self::DEFAULT_INNER_CAPACITY, outer_capacity),
            non_determinism_reads_buffer: BiVec::new_with_capacities(
                Self::DEFAULT_INNER_CAPACITY,
                outer_capacity,
            ),
            initial_snapshot,
        }
    }
}

impl<C: Counters, const ROM_BOUND_SECOND_WORD_BITS: usize> Snapshotter<C>
    for SimpleSnapshotter<C, ROM_BOUND_SECOND_WORD_BITS>
{
    #[inline(always)]
    fn take_snapshot(&mut self, state: &State<C>) {
        let new_snapshot = SimpleSnapshot {
            state: *state,
            non_determinism_reads_start: self.current_partial_snapshot.non_determinism_reads_offset,
            non_determinism_reads_end: self.non_determinism_reads_buffer.len(),
            last_zero_address_read_timestamp: self
                .current_partial_snapshot
                .last_zero_address_read_timestamp,
            memory_reads_start: self.current_partial_snapshot.memory_reads_offset,
            memory_reads_end: self.reads_buffer.len(),
        };
        self.current_partial_snapshot
            .last_zero_address_read_timestamp = self.last_zero_address_read_timestamp;
        self.current_partial_snapshot.non_determinism_reads_offset =
            self.non_determinism_reads_buffer.len();
        self.current_partial_snapshot.memory_reads_offset = self.reads_buffer.len();
        self.snapshots.push(new_snapshot);
    }

    #[inline(always)]
    fn append_non_determinism_read(&mut self, value: u32) {
        self.non_determinism_reads_buffer.push(value);
    }

    #[inline(always)]
    fn append_memory_read(
        &mut self,
        address: u32,
        read_value: u32,
        read_timestamp: TimestampScalar,
        write_timestamp: TimestampScalar,
    ) {
        if address < (1 << (16 + ROM_BOUND_SECOND_WORD_BITS)) {
            self.last_zero_address_read_timestamp = write_timestamp;
        }
        self.reads_buffer.push((
            read_value,
            (read_timestamp as u32, (read_timestamp >> 32) as u32),
        ));
    }
}
