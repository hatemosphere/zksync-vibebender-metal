use cs::definitions::{TimestampData, TimestampScalar};
use prover::common_constants;
use prover::definitions::LazyInitAndTeardown;
use riscv_transpiler::vm::{RamPeek, RAM};
use std::mem::replace;

pub(crate) const ROM_ADDRESS_SPACE_SECOND_WORD_BITS: usize =
    common_constants::rom::ROM_SECOND_WORD_BITS;
const ROM_LOG_SIZE: u32 = 16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS as u32;
const ROM_BOUND: u32 = 1 << ROM_LOG_SIZE;
const ROM_BOUND_MASK: u32 = ROM_BOUND - 1;
const ROM_WORDS_LOG_SIZE: u32 = ROM_LOG_SIZE - 2;
const ROM_WORDS_SIZE: usize = 1 << ROM_WORDS_LOG_SIZE;

pub struct RamWithRomRegion<const RAM_LOG_SIZE: u32, const TRACK_TOUCHED_RAM: bool = true> {
    pub(crate) word_values: Box<[u32]>,
    pub(crate) word_timestamps: Box<[TimestampScalar]>,
    pub(crate) touched_words_in_page_counts: Box<[u32]>,
}

impl<const RAM_LOG_SIZE: u32, const TRACK_TOUCHED_RAM: bool>
    RamWithRomRegion<RAM_LOG_SIZE, TRACK_TOUCHED_RAM>
{
    pub fn new(rom: &[u32]) -> Self {
        assert!(RAM_LOG_SIZE >= ROM_LOG_SIZE);
        let words_log_count = RAM_LOG_SIZE - 2;
        let words_count = 1 << words_log_count;
        let mut word_values = unsafe { Box::new_zeroed_slice(words_count).assume_init() };
        let rom_len = rom.len();
        assert_eq!(rom_len, ROM_WORDS_SIZE);
        word_values[..rom_len].copy_from_slice(rom);
        let word_timestamps = unsafe { Box::new_zeroed_slice(words_count).assume_init() };
        let pages_count = if TRACK_TOUCHED_RAM {
            1 << (RAM_LOG_SIZE - ROM_LOG_SIZE)
        } else {
            0
        };
        let touched_words_in_page_counts =
            unsafe { Box::new_zeroed_slice(pages_count).assume_init() };
        Self {
            word_values,
            word_timestamps,
            touched_words_in_page_counts,
        }
    }

    #[inline(always)]
    fn touch_word(&mut self, word_idx: usize, timestamp: TimestampScalar) {
        if TRACK_TOUCHED_RAM {
            if core::hint::unlikely(timestamp == 0) {
                let page_idx = word_idx >> ROM_WORDS_LOG_SIZE;
                unsafe {
                    *self
                        .touched_words_in_page_counts
                        .get_unchecked_mut(page_idx) += 1;
                };
            }
        }
    }

    pub fn get_touched_words_count(&self) -> u32 {
        self.touched_words_in_page_counts.iter().sum::<u32>()
    }

    unsafe fn get_init_and_teardown_value<const IS_RAM: bool>(
        &self,
        index: usize,
    ) -> Option<LazyInitAndTeardown> {
        let timestamp = *self.word_timestamps.get_unchecked(index);
        if timestamp == 0 {
            None
        } else {
            let teardown_value = if IS_RAM {
                *self.word_values.get_unchecked(index)
            } else {
                0
            };
            let result = LazyInitAndTeardown {
                address: (index as u32) << 2,
                teardown_value,
                teardown_timestamp: TimestampData::from_scalar(timestamp),
            };
            Some(result)
        }
    }

    pub fn get_inits_and_teardowns_iterator(
        &self,
    ) -> impl Iterator<Item = LazyInitAndTeardown> + '_ {
        self.touched_words_in_page_counts
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(page_index, count)| if count == 0 { None } else { Some(page_index) })
            .flat_map(move |page_index| {
                let value_index = page_index << ROM_WORDS_LOG_SIZE;
                let range = value_index..value_index + ROM_WORDS_SIZE;
                let f = if page_index == 0 {
                    Self::get_init_and_teardown_value::<false>
                } else {
                    Self::get_init_and_teardown_value::<true>
                };
                range.filter_map(move |i| unsafe { f(self, i) })
            })
    }
}

impl<const RAM_LOG_SIZE: u32, const TRACK_TOUCHED_RAM: bool> RamPeek
    for RamWithRomRegion<RAM_LOG_SIZE, TRACK_TOUCHED_RAM>
{
    #[inline(always)]
    fn peek_word(&self, address: u32) -> u32 {
        debug_assert_eq!(address % 4, 0);
        debug_assert_eq!(address >> RAM_LOG_SIZE, 0);
        let word_idx = (address >> 2) as usize;
        unsafe { *self.word_values.get_unchecked(word_idx) }
    }
}

impl<const RAM_LOG_SIZE: u32, const TRACK_TOUCHED_RAM: bool> RAM
    for RamWithRomRegion<RAM_LOG_SIZE, TRACK_TOUCHED_RAM>
{
    #[inline(always)]
    fn read_word(&mut self, address: u32, timestamp: TimestampScalar) -> (TimestampScalar, u32) {
        debug_assert_eq!(address & 3, 0);
        debug_assert_eq!(address >> RAM_LOG_SIZE, 0);
        let word_idx = (address >> 2) as usize;
        let timestamp_ref = unsafe { self.word_timestamps.get_unchecked_mut(word_idx) };
        let read_timestamp = replace(timestamp_ref, timestamp | 1);
        debug_assert!(read_timestamp < timestamp | 1);
        self.touch_word(word_idx, read_timestamp);
        let word = unsafe { *self.word_values.get_unchecked(word_idx) };
        (read_timestamp, word)
    }

    #[inline(always)]
    fn mask_read_for_witness(&self, _address: &mut u32, _value: &mut u32) {}

    #[inline(always)]
    fn write_word(
        &mut self,
        address: u32,
        word: u32,
        timestamp: TimestampScalar,
    ) -> (TimestampScalar, u32) {
        debug_assert_eq!(address & 3, 0);
        debug_assert_eq!(address >> RAM_LOG_SIZE, 0);
        debug_assert!(address & !ROM_BOUND_MASK != 0);
        let word_idx = (address >> 2) as usize;
        let timestamp_ref = unsafe { self.word_timestamps.get_unchecked_mut(word_idx) };
        let read_timestamp = replace(timestamp_ref, timestamp | 2);
        debug_assert!(read_timestamp < timestamp | 2);
        self.touch_word(word_idx, read_timestamp);
        let value_ref = unsafe { self.word_values.get_unchecked_mut(word_idx) };
        let read_word = replace(value_ref, word);
        (read_timestamp, read_word)
    }

    #[inline(always)]
    fn skip_if_replaying(&mut self, _num_snapshots: usize) {
        panic!("RamWithRomRegion must not be used in replayer");
    }
}
