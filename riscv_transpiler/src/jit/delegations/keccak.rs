use super::*;
use crate::vm::delegations::keccak_special5::*;
use common_constants::*;
use std::mem::MaybeUninit;

pub fn keccak_unrolled_implementation(
    trace_piece: &mut TraceChunk,
    memory_holder: &mut MemoryHolder,
    machine_state: &mut MachineState,
) -> u64 {
    // Implementer here is responsible for ALL the bookkeeping, and eventually MUST update trace piece chunk via context, and and update machine state to reflect filled part of trace chunk
    assert!((trace_piece.len as usize) < TRACE_CHUNK_LEN);
    debug_assert_eq!(machine_state.timestamp % 4, 3);
    assert_eq!(
        machine_state.registers[10],
        INITIAL_KECCAK_F1600_CONTROL_VALUE
    ); // initial control flow is expected to be zero
    let state_ptr = machine_state.registers[11];
    assert!(state_ptr as usize >= common_constants::rom::ROM_BYTE_SIZE);
    assert_eq!(state_ptr % 256, 0, "state pointer is unaligned");

    // now we will effectively "unroll" all the invocation

    // Register accesses are easy - we just need to write final control flow value, and update timestamps

    machine_state.registers[10] = FINAL_KECCAK_F1600_CONTROL_VALUE;
    machine_state.register_timestamps[10] +=
        ((NUM_DELEGATION_CALLS_FOR_KECCAK_F1600 - 1) as TimestampScalar) * TIMESTAMP_STEP;

    machine_state.register_timestamps[11] +=
        ((NUM_DELEGATION_CALLS_FOR_KECCAK_F1600 - 1) as TimestampScalar) * TIMESTAMP_STEP;

    // save for accesses in individual cycles
    let mut write_ts = machine_state.timestamp;

    // and full machine state also moves!
    machine_state.timestamp +=
        ((NUM_DELEGATION_CALLS_FOR_KECCAK_F1600 - 1) as TimestampScalar) * TIMESTAMP_STEP;

    // just stamp keccak_f1600 on top of it...

    // now we need to be careful with accessed state elements. We always access u64s only, and for replaying purposes we will need
    // to read 31 state elements (for snapshot), and then we will work over the
    unsafe {
        // we are fine to NOT keep track on the initial timestamps, as we only need final write ones
        let mut keccak_state: [MaybeUninit<u64>; 31] = [const { MaybeUninit::uninit() }; 31];
        let mut tses: [TimestampScalar; 31] = [write_ts; 31];

        let mut mem_ptr = memory_holder
            .memory
            .as_ptr()
            .add((state_ptr as usize) / core::mem::size_of::<u32>());
        let mut ts_ptr = memory_holder
            .timestamps
            .as_ptr()
            .add((state_ptr as usize) / core::mem::size_of::<u32>());

        for i in 0..31 {
            // low and high

            let low_value = mem_ptr.read();
            mem_ptr = mem_ptr.add(1);
            let high_value = mem_ptr.read();
            mem_ptr = mem_ptr.add(1);

            let low_ts = ts_ptr.read();
            ts_ptr = ts_ptr.add(1);
            let high_ts = ts_ptr.read();
            ts_ptr = ts_ptr.add(1);

            trace_piece.add_element(low_value, low_ts);
            trace_piece.add_element(high_value, high_ts);

            keccak_state[i].write(((low_value as u64) | ((high_value as u64) << 32)));
        }

        // just run keccak_f1600 over state in sumulation
        let keccak_state = unsafe {
            let mut keccak_state = keccak_state.map(|el| el.assume_init());
            ::keccak::f1600(core::mem::transmute::<_, &mut [u64; 25]>(&mut keccak_state));

            keccak_state
        };

        // TODO: precompute timestamp differences updates

        // and write everything back

        let mut mem_ptr = memory_holder
            .memory
            .as_mut_ptr()
            .add((state_ptr as usize) / core::mem::size_of::<u32>());
        let mut ts_ptr = memory_holder
            .timestamps
            .as_mut_ptr()
            .add((state_ptr as usize) / core::mem::size_of::<u32>());

        for value in keccak_state.into_iter() {
            let low = value as u32;
            let high = (value >> 32) as u32;

            mem_ptr.write(low);
            mem_ptr = mem_ptr.add(1);
            mem_ptr.write(high);
            mem_ptr = mem_ptr.add(1);
        }
    }

    // // now we need to be careful with accessed state elements. We always access u64s only, and for replaying purposes we will need
    // // to read 31 state elements (for snapshot), and then we will work over the
    // unsafe {
    //     // we are fine to NOT keep track on the initial timestamps, as we only need final write ones
    //     let mut local_state: [MaybeUninit<(u64, TimestampScalar)>; 31] =
    //         [const { MaybeUninit::uninit() }; 31];
    //     let mut mem_ptr = memory_holder
    //         .memory
    //         .as_ptr()
    //         .add((state_ptr as usize) / core::mem::size_of::<u32>());
    //     let mut ts_ptr = memory_holder
    //         .timestamps
    //         .as_ptr()
    //         .add((state_ptr as usize) / core::mem::size_of::<u32>());

    //     for i in 0..31 {
    //         // low and high

    //         let low_value = mem_ptr.read();
    //         mem_ptr = mem_ptr.add(1);
    //         let high_value = mem_ptr.read();
    //         mem_ptr = mem_ptr.add(1);

    //         let low_ts = ts_ptr.read();
    //         ts_ptr = ts_ptr.add(1);
    //         let high_ts = ts_ptr.read();
    //         ts_ptr = ts_ptr.add(1);

    //         trace_piece.add_element(low_value, low_ts);
    //         trace_piece.add_element(high_value, high_ts);

    //         local_state[i].write((
    //             ((low_value as u64) | ((high_value as u64) << 32)),
    //             0 as TimestampScalar,
    //         ));
    //     }

    //     let mut local_state = local_state.map(|el| el.assume_init());

    //     let mut control_reg = INITIAL_CONTROL_VALUE;
    //     for round in 0..NUM_DELEGATION_CALLS_FOR_KECCAK_F1600 {
    //         let (precompile, iteration, round) = keccak_special5_impl_decode_control(control_reg);
    //         control_reg = keccak_special5_impl_bump_control(precompile, iteration, round);

    //         let state_indexes = keccak_special5_impl_extract_indices(precompile, iteration, round);
    //         // get inputs
    //         let state_inputs = state_indexes.map(|i| local_state[i].0);
    //         // get outputs
    //         let state_outputs =
    //             keccak_special5_impl_compute_outputs(precompile, iteration, round, state_inputs);
    //         // write back
    //         for i in 0..6 {
    //             let state_index = state_indexes[i];
    //             local_state[state_index] = (state_outputs[i], write_ts);
    //         }

    //         write_ts += TIMESTAMP_STEP;
    //     }

    //     for (idx, (_, ts)) in local_state.iter().enumerate() {
    //         debug_assert_ne!(*ts, 0, "element number {} was not touched", idx);
    //     }

    //     assert_eq!(control_reg, FINAL_CONTROL_VALUE);

    //     // and write everything back

    //     let mut mem_ptr = memory_holder
    //         .memory
    //         .as_mut_ptr()
    //         .add((state_ptr as usize) / core::mem::size_of::<u32>());
    //     let mut ts_ptr = memory_holder
    //         .timestamps
    //         .as_mut_ptr()
    //         .add((state_ptr as usize) / core::mem::size_of::<u32>());

    //     for (value, ts) in local_state.into_iter() {
    //         let low = value as u32;
    //         let high = (value >> 32) as u32;

    //         mem_ptr.write(low);
    //         mem_ptr = mem_ptr.add(1);
    //         mem_ptr.write(high);
    //         mem_ptr = mem_ptr.add(1);

    //         ts_ptr.write(ts);
    //         ts_ptr = ts_ptr.add(1);
    //         ts_ptr.write(ts);
    //         ts_ptr = ts_ptr.add(1);
    //     }
    // }
    // assert_eq!(machine_state.timestamp + TIMESTAMP_STEP, write_ts);

    assert!((trace_piece.len as usize) < MAX_TRACE_CHUNK_LEN);
    let should_flush = ((trace_piece.len as usize) >= TRACE_CHUNK_LEN) as u64;

    should_flush
}
