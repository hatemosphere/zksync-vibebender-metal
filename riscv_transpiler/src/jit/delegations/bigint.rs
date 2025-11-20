use super::*;
use crate::vm::delegations::bigint::*;
use common_constants::*;
use ruint::aliases::U256;
use std::mem::MaybeUninit;

pub fn bigint_implementation(
    trace_piece: &mut TraceChunk,
    memory_holder: &mut MemoryHolder,
    machine_state: &mut MachineState,
) -> u64 {
    // Implementer here is responsible for ALL the bookkeeping, and eventually MUST update trace piece chunk via context, and and update machine state to reflect filled part of trace chunk
    assert!((trace_piece.len as usize) < TRACE_CHUNK_LEN);
    debug_assert_eq!(machine_state.timestamp % 4, 3);
    let a_ptr = machine_state.registers[10];
    let b_ptr = machine_state.registers[11];
    let x12 = machine_state.registers[12];
    assert!(a_ptr as usize >= common_constants::rom::ROM_BYTE_SIZE);
    assert!(b_ptr as usize >= common_constants::rom::ROM_BYTE_SIZE);
    assert_eq!(a_ptr % 32, 0, "`a` pointer is unaligned");
    assert_eq!(b_ptr % 32, 0, "`b` pointer is unaligned");

    let write_ts = machine_state.timestamp;

    machine_state.register_timestamps[10] = write_ts;
    machine_state.register_timestamps[11] = write_ts;
    machine_state.register_timestamps[12] = write_ts;

    // read and save into snapshots

    let a = unsafe {
        // we are fine to NOT keep track on the initial timestamps, as we only need final write ones
        let mut a = U256::ZERO;
        let mut mem_ptr = memory_holder
            .memory
            .as_ptr()
            .add((a_ptr as usize) / core::mem::size_of::<u32>());
        let mut ts_ptr = memory_holder
            .timestamps
            .as_mut_ptr()
            .add((a_ptr as usize) / core::mem::size_of::<u32>());

        for dst in a.as_limbs_mut().iter_mut() {
            // low and high

            let low_value = mem_ptr.read();
            mem_ptr = mem_ptr.add(1);
            let high_value = mem_ptr.read();
            mem_ptr = mem_ptr.add(1);

            let low_ts = ts_ptr.read();
            ts_ptr.write(write_ts);
            ts_ptr = ts_ptr.add(1);
            let high_ts = ts_ptr.read();
            ts_ptr.write(write_ts);
            ts_ptr = ts_ptr.add(1);

            trace_piece.add_element(low_value, low_ts);
            trace_piece.add_element(high_value, high_ts);

            *dst = (low_value as u64) | ((high_value as u64) << 32);
        }

        a
    };

    let b = unsafe {
        // we are fine to NOT keep track on the initial timestamps, as we only need final write ones
        let mut b = U256::ZERO;
        let mut mem_ptr = memory_holder
            .memory
            .as_ptr()
            .add((b_ptr as usize) / core::mem::size_of::<u32>());
        let mut ts_ptr = memory_holder
            .timestamps
            .as_mut_ptr()
            .add((b_ptr as usize) / core::mem::size_of::<u32>());

        for dst in b.as_limbs_mut().iter_mut() {
            // low and high

            let low_value = mem_ptr.read();
            mem_ptr = mem_ptr.add(1);
            let high_value = mem_ptr.read();
            mem_ptr = mem_ptr.add(1);

            let low_ts = ts_ptr.read();
            ts_ptr.write(write_ts);
            ts_ptr = ts_ptr.add(1);
            let high_ts = ts_ptr.read();
            ts_ptr.write(write_ts);
            ts_ptr = ts_ptr.add(1);

            trace_piece.add_element(low_value, low_ts);
            trace_piece.add_element(high_value, high_ts);

            *dst = (low_value as u64) | ((high_value as u64) << 32);
        }

        b
    };

    let (result, of) = bigint_impl(a, b, x12);
    trace_piece.append_arbitrary_value(of as u32);

    // write back - values only
    unsafe {
        let mut mem_ptr = memory_holder
            .memory
            .as_mut_ptr()
            .add((a_ptr as usize) / core::mem::size_of::<u32>());

        for src in result.as_limbs().iter() {
            // low and high

            let low = *src as u32;
            let high = (*src >> 32) as u32;

            mem_ptr.write(low);
            mem_ptr = mem_ptr.add(1);
            mem_ptr.write(high);
            mem_ptr = mem_ptr.add(1);
        }
    };

    machine_state.registers[12] = of as u32;

    assert!((trace_piece.len as usize) < MAX_TRACE_CHUNK_LEN);
    let should_flush = ((trace_piece.len as usize) >= TRACE_CHUNK_LEN) as u64;

    // println!("Bigint, should flush = {}", should_flush);

    should_flush
}
