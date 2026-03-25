#[cfg(test)]
mod runtime_tests {
    use crate::metal_runtime::*;

    #[test]
    fn test_device_creation() {
        let device = system_default_device().expect("Failed to get Metal device");
        let props = DeviceProperties::query(device);
        assert!(!props.name.is_empty(), "Device should have a name");
        assert!(
            props.max_threads_per_threadgroup > 0,
            "Max threads per threadgroup should be > 0"
        );
        assert!(
            props.max_buffer_length > 0,
            "Max buffer length should be > 0"
        );
        println!("Metal device: {}", props.name);
        println!(
            "  max threads/threadgroup: {}",
            props.max_threads_per_threadgroup
        );
        println!(
            "  max threadgroup memory: {} bytes",
            props.max_threadgroup_memory_length
        );
        println!("  max buffer length: {} bytes", props.max_buffer_length);
        println!("  unified memory: {}", props.has_unified_memory);
    }

    #[test]
    fn test_buffer_alloc_and_roundtrip() {
        let device = system_default_device().expect("Failed to get Metal device");

        let data: Vec<u32> = (0..1024).collect();
        let mut buffer =
            MetalBuffer::<u32>::from_slice(device, &data).expect("Failed to create buffer");

        assert_eq!(buffer.len(), 1024);
        assert_eq!(buffer.byte_len(), 1024 * 4);

        // Read back
        let mut readback = vec![0u32; 1024];
        unsafe {
            buffer.copy_to_slice(&mut readback);
        }
        assert_eq!(data, readback);

        // Modify and verify
        unsafe {
            let slice = buffer.as_mut_slice();
            slice[0] = 999;
            slice[1023] = 888;
        }
        unsafe {
            buffer.copy_to_slice(&mut readback);
        }
        assert_eq!(readback[0], 999);
        assert_eq!(readback[1023], 888);
    }

    #[test]
    fn test_buffer_split() {
        let device = system_default_device().expect("Failed to get Metal device");
        let data: Vec<u32> = (0..100).collect();
        let buffer =
            MetalBuffer::<u32>::from_slice(device, &data).expect("Failed to create buffer");

        let (left, right) = buffer.split_at(40);
        assert_eq!(left.len(), 40);
        assert_eq!(right.len(), 60);
        assert_eq!(left.byte_offset(), 0);
        assert_eq!(right.byte_offset(), 40 * 4);
    }

    #[test]
    fn test_command_queue_and_buffer() {
        let device = system_default_device().expect("Failed to get Metal device");
        let queue = MetalCommandQueue::new(device).expect("Failed to create command queue");
        let cmd_buf = queue
            .new_command_buffer()
            .expect("Failed to create command buffer");

        // Just commit an empty command buffer to verify the pipeline works
        cmd_buf.commit_and_wait();
        cmd_buf.status().expect("Command buffer should succeed");
    }

    #[test]
    fn test_memory_info() {
        let device = system_default_device().expect("Failed to get Metal device");
        let (recommended, max_alloc) = crate::metal_runtime::memory::memory_get_info(device);
        assert!(recommended > 0, "Recommended working set should be > 0");
        assert!(max_alloc > 0, "Max alloc should be > 0");
        println!("Memory: recommended={recommended}, max_alloc={max_alloc}");
    }

    #[test]
    fn test_event_creation() {
        let device = system_default_device().expect("Failed to get Metal device");
        let event = MetalEvent::create(device).expect("Failed to create event");
        assert_eq!(event.current_value(), 0);
    }

    #[test]
    fn test_identity_kernel_dispatch() {
        let device = system_default_device().expect("Failed to get Metal device");

        // Initialize the shader library (loads embedded .metallib)
        init_shader_library(device).expect("Failed to init shader library");

        let count: u32 = 4096;
        let input_data: Vec<u32> = (0..count).collect();
        let input = MetalBuffer::<u32>::from_slice(device, &input_data)
            .expect("Failed to create input buffer");
        let output = MetalBuffer::<u32>::alloc(device, count as usize)
            .expect("Failed to create output buffer");

        let queue = MetalCommandQueue::new(device).expect("Failed to create queue");
        let cmd_buf = queue.new_command_buffer().expect("Failed to create cmd buf");

        let threads_per_group: u32 = 256;
        let threadgroups = (count + threads_per_group - 1) / threads_per_group;
        let config = MetalLaunchConfig::basic_1d(threadgroups, threads_per_group);

        dispatch::dispatch_kernel(device, &cmd_buf, "identity_copy", &config, |encoder| {
            dispatch::set_buffer(encoder, 0, input.raw(), 0);
            dispatch::set_buffer(encoder, 1, output.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 2, &count);
            }
        })
        .expect("Failed to dispatch kernel");

        cmd_buf.commit_and_wait();
        cmd_buf.status().expect("Kernel execution failed");

        // Verify output matches input
        let mut result = vec![0u32; count as usize];
        unsafe {
            output.copy_to_slice(&mut result);
        }
        assert_eq!(input_data, result, "Identity kernel output mismatch");
    }

    #[test]
    fn test_event_synchronization() {
        let device = system_default_device().expect("Failed to get Metal device");
        init_shader_library(device).expect("Failed to init shader library");

        let queue = MetalCommandQueue::new(device).expect("Failed to create queue");
        let event = MetalEvent::create(device).expect("Failed to create event");

        // First command buffer: write data via identity kernel
        let count: u32 = 1024;
        let input_data: Vec<u32> = (100..100 + count).collect();
        let input = MetalBuffer::<u32>::from_slice(device, &input_data).unwrap();
        let shared_buf = MetalBuffer::<u32>::alloc(device, count as usize).unwrap();

        let cb1 = queue.new_command_buffer().unwrap();
        let config = MetalLaunchConfig::basic_1d(
            (count + 255) / 256,
            256,
        );
        dispatch::dispatch_kernel(device, &cb1, "identity_copy", &config, |encoder| {
            dispatch::set_buffer(encoder, 0, input.raw(), 0);
            dispatch::set_buffer(encoder, 1, shared_buf.raw(), 0);
            unsafe { dispatch::set_bytes(encoder, 2, &count); }
        }).unwrap();
        let signal_val = event.signal(&cb1);
        cb1.commit();

        // Second command buffer: copy from shared_buf to output, waits on event
        let output = MetalBuffer::<u32>::alloc(device, count as usize).unwrap();
        let cb2 = queue.new_command_buffer().unwrap();
        event.wait(&cb2, signal_val);
        dispatch::dispatch_kernel(device, &cb2, "identity_copy", &config, |encoder| {
            dispatch::set_buffer(encoder, 0, shared_buf.raw(), 0);
            dispatch::set_buffer(encoder, 1, output.raw(), 0);
            unsafe { dispatch::set_bytes(encoder, 2, &count); }
        }).unwrap();
        cb2.commit_and_wait();

        let mut result = vec![0u32; count as usize];
        unsafe { output.copy_to_slice(&mut result); }
        assert_eq!(input_data, result, "Event-synchronized pipeline failed");
    }

    #[test]
    fn test_field_arithmetic_on_gpu() {
        use field::{Mersenne31Field, Field};

        let device = system_default_device().expect("Failed to get Metal device");
        init_shader_library(device).expect("Failed to init shader library");

        let p = (1u64 << 31) - 1;
        // Test pairs: (a, b) as canonical u32 values
        let test_pairs: Vec<(u32, u32)> = vec![
            (0, 0),
            (1, 1),
            (0, 1),
            (1, 0),
            (100, 200),
            (p as u32 - 1, 1),  // MINUS_ONE + 1 = 0
            (p as u32 - 1, p as u32 - 1),
            (12345, 67890),
            (1000000, 2000000),
            (7, 3),
        ];

        let pair_count = test_pairs.len() as u32;
        let mut input_data = Vec::new();
        for (a, b) in &test_pairs {
            input_data.push(*a);
            input_data.push(*b);
        }

        let input = MetalBuffer::<u32>::from_slice(device, &input_data).unwrap();
        let output = MetalBuffer::<u32>::alloc(device, (pair_count as usize) * 6).unwrap();

        let queue = MetalCommandQueue::new(device).unwrap();
        let cmd_buf = queue.new_command_buffer().unwrap();

        let config = MetalLaunchConfig::basic_1d((pair_count + 255) / 256, 256);
        dispatch::dispatch_kernel(device, &cmd_buf, "field_test", &config, |encoder| {
            dispatch::set_buffer(encoder, 0, input.raw(), 0);
            dispatch::set_buffer(encoder, 1, output.raw(), 0);
            unsafe { dispatch::set_bytes(encoder, 2, &pair_count); }
        }).unwrap();

        cmd_buf.commit_and_wait();

        let mut results = vec![0u32; pair_count as usize * 6];
        unsafe { output.copy_to_slice(&mut results); }

        // Verify against CPU field arithmetic
        for (i, (a_val, b_val)) in test_pairs.iter().enumerate() {
            let a = Mersenne31Field::from_nonreduced_u32(*a_val);
            let b = Mersenne31Field::from_nonreduced_u32(*b_val);

            let mut cpu_add = a; Field::add_assign(&mut cpu_add, &b);
            let mut cpu_sub = a; Field::sub_assign(&mut cpu_sub, &b);
            let mut cpu_mul = a; Field::mul_assign(&mut cpu_mul, &b);
            let mut cpu_sqr = a; Field::mul_assign(&mut cpu_sqr, &a);
            let mut cpu_neg = a; Field::negate(&mut cpu_neg);

            let gpu_add = results[6 * i + 0];
            let gpu_sub = results[6 * i + 1];
            let gpu_mul = results[6 * i + 2];
            let gpu_sqr = results[6 * i + 3];
            let gpu_neg = results[6 * i + 5];

            assert_eq!(gpu_add, cpu_add.to_reduced_u32(), "add mismatch at pair {i}: {a_val} + {b_val}");
            assert_eq!(gpu_sub, cpu_sub.to_reduced_u32(), "sub mismatch at pair {i}: {a_val} - {b_val}");
            assert_eq!(gpu_mul, cpu_mul.to_reduced_u32(), "mul mismatch at pair {i}: {a_val} * {b_val}");
            assert_eq!(gpu_sqr, cpu_sqr.to_reduced_u32(), "sqr mismatch at pair {i}: {a_val}^2");
            assert_eq!(gpu_neg, cpu_neg.to_reduced_u32(), "neg mismatch at pair {i}: -{a_val}");

            // Test inv: a * inv(a) should equal 1 (skip zero)
            if *a_val != 0 {
                let gpu_inv = results[6 * i + 4];
                let inv_field = Mersenne31Field::from_nonreduced_u32(gpu_inv);
                let mut product = a;
                Field::mul_assign(&mut product, &inv_field);
                assert_eq!(product.to_reduced_u32(), 1, "inv mismatch at pair {i}: {a_val} * inv({a_val})");
            }
        }
    }
}

#[cfg(test)]
mod e2e_tests {
    use blake2s_u32::Blake2sState;
    use field::Field;
    use itertools::Itertools;
    use rand::Rng;
    use serial_test::serial;

    use crate::blake2s::{self, Digest, BLOCK_SIZE};
    use crate::device_context::DeviceContext;
    use crate::field::BaseField;
    use crate::metal_runtime::buffer::MetalBuffer;
    use crate::metal_runtime::command_queue::MetalCommandQueue;
    use crate::metal_runtime::device::system_default_device;
    use crate::metal_runtime::pipeline::init_shader_library;
    use crate::ntt::{
        bitrev_Z_to_natural_composition_main_evals, bitrev_Z_to_natural_trace_coset_evals,
        natural_trace_main_evals_to_bitrev_Z, NttTwiddleData,
    };

    type BF = BaseField;
    const USE_REDUCED_BLAKE2_ROUNDS: bool = true;

    /// End-to-end proving pipeline test: NTT + Blake2s Merkle tree.
    ///
    /// Exercises the most performance-critical GPU kernels in the exact
    /// sequence used during real proof generation:
    ///   1. Generate random base field trace data (simulating witness evaluations)
    ///   2. Forward NTT: natural evals -> bitrev Z (monomial representation)
    ///   3. Inverse NTT with coset shift: bitrev Z -> coset evaluations (LDE)
    ///   4. Build Merkle tree over coset evals: blake2s leaves + nodes
    ///   5. Verify tree structure: each node = hash of its two children
    ///   6. Verify NTT roundtrip: coset evals -> Z -> main evals recovers original data
    #[test]
    #[serial]
    fn test_e2e_ntt_blake2s_pipeline() {
        let log_n: usize = 12;
        let n: usize = 1 << log_n;
        let num_bf_cols: usize = 16; // 8 Z pairs, realistic column count
        let stride = n;
        let memory_size = stride * num_bf_cols;

        // --- GPU setup ---
        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let dc = DeviceContext::create(device, 12).unwrap();
        let twiddle_data = NttTwiddleData::from_device_context(&dc, device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        // --- Step 1: Generate random trace data (simulating witness evaluations) ---
        let mut rng = rand::rng();
        let mut src: Vec<BF> = (0..memory_size)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect();

        // Enforce sum-to-zero constraint per column (required for trace polynomials
        // that vanish at the trace domain boundary).
        for col in 0..num_bf_cols {
            let start = col * stride;
            let sum: BF = src[start..start + n]
                .iter()
                .fold(BF::ZERO, |acc, val| *acc.clone().add_assign(val));
            src[start + n - 1].sub_assign(&sum);
        }
        let original_src = src.clone();

        println!(
            "Step 1: Generated random trace data: {} rows x {} columns ({} total elements)",
            n, num_bf_cols, memory_size
        );

        // --- Step 2: Forward NTT (natural evals -> bitrev Z) ---
        let src_buf = MetalBuffer::<BF>::from_slice(device, &src).unwrap();
        let mut z_buf = MetalBuffer::<BF>::alloc(device, memory_size).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        natural_trace_main_evals_to_bitrev_Z(
            device,
            &cmd_buf,
            &src_buf,
            &mut z_buf,
            stride as u32,
            log_n as u32,
            num_bf_cols as u32,
            &twiddle_data,
        )
        .unwrap();
        cmd_buf.commit_and_wait();

        let mut z_host = vec![BF::ZERO; memory_size];
        unsafe { z_buf.copy_to_slice(&mut z_host); }

        // Verify Z representation is non-trivial (not all zeros)
        let nonzero_z = z_host.iter().any(|v| v.to_reduced_u32() != 0);
        assert!(nonzero_z, "Forward NTT produced all-zero output");
        println!("Step 2: Forward NTT complete (bitrev Z representation)");

        // --- Step 3: Inverse NTT with coset (bitrev Z -> coset evals, the LDE step) ---
        let mut coset_buf = MetalBuffer::<BF>::alloc(device, memory_size).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        bitrev_Z_to_natural_trace_coset_evals(
            device,
            &cmd_buf,
            &z_buf,
            &mut coset_buf,
            stride as u32,
            log_n as u32,
            num_bf_cols as u32,
            &twiddle_data,
        )
        .unwrap();
        cmd_buf.commit_and_wait();

        let mut coset_host = vec![BF::ZERO; memory_size];
        unsafe { coset_buf.copy_to_slice(&mut coset_host); }

        // Coset evals should differ from main evals (they're on a shifted domain)
        assert_ne!(
            &original_src[..n],
            &coset_host[..n],
            "Coset evals should differ from main domain evals"
        );
        let nonzero_coset = coset_host.iter().any(|v| v.to_reduced_u32() != 0);
        assert!(nonzero_coset, "Coset evals are all zero");
        println!("Step 3: Coset evaluation complete (LDE on shifted domain)");

        // --- Step 4: Build Merkle tree over coset evaluations ---
        // In real proving, we hash rows of the coset evaluation matrix.
        // log_rows_per_hash=0 means each row hashes individually (n leaves).
        let log_rows_per_hash: u32 = 0;
        let num_leaves = n;
        let leaves_buf = MetalBuffer::<Digest>::alloc(device, num_leaves).unwrap();
        let nodes_buf = MetalBuffer::<Digest>::alloc(device, num_leaves).unwrap();

        // Zero-init nodes buffer (unified memory — direct pointer access)
        unsafe {
            std::ptr::write_bytes(nodes_buf.as_ptr() as *mut u8, 0, nodes_buf.byte_len());
        }

        let node_layers_count = log_n as u32;

        let cmd_buf = queue.new_command_buffer().unwrap();
        blake2s::build_merkle_tree_leaves(
            device,
            &cmd_buf,
            &coset_buf,
            &leaves_buf,
            log_rows_per_hash,
        )
        .unwrap();
        blake2s::build_merkle_tree_nodes(
            device,
            &cmd_buf,
            &leaves_buf,
            &nodes_buf,
            node_layers_count,
        )
        .unwrap();
        cmd_buf.commit_and_wait();

        let mut leaves_host = vec![Digest::default(); num_leaves];
        let mut nodes_host = vec![Digest::default(); num_leaves];
        unsafe {
            leaves_buf.copy_to_slice(&mut leaves_host);
            nodes_buf.copy_to_slice(&mut nodes_host);
        }

        println!(
            "Step 4: Merkle tree built ({} leaves, {} node layers)",
            num_leaves, node_layers_count
        );

        // --- Step 5: Verify Merkle tree structure ---

        // 5a: Verify leaves against CPU blake2s
        let cols_count = memory_size / num_leaves;
        for i in 0..num_leaves {
            let mut input = vec![];
            for col in 0..cols_count {
                let offset = i + col * num_leaves;
                input.push(coset_host[offset]);
            }
            let blocks_count = (input.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
            let mut state = Blake2sState::new();
            let mut expected = Digest::default();
            for (bi, chunk) in input.iter().chunks(BLOCK_SIZE).into_iter().enumerate() {
                let chunk: Vec<_> = chunk.cloned().collect();
                let block_len = chunk.len();
                let mut block = [0u32; BLOCK_SIZE];
                for (j, v) in chunk.iter().enumerate() {
                    block[j] = v.0;
                }
                if bi == blocks_count - 1 {
                    state.absorb_final_block::<USE_REDUCED_BLAKE2_ROUNDS>(
                        &block,
                        block_len,
                        &mut expected,
                    );
                } else {
                    state.absorb::<USE_REDUCED_BLAKE2_ROUNDS>(&block);
                }
            }
            assert_eq!(
                leaves_host[i], expected,
                "Leaf hash mismatch at index {}",
                i
            );
        }
        println!("Step 5a: All {} leaf hashes verified against CPU reference", num_leaves);

        // 5b: Verify node layers (each node = hash of its two children)
        fn verify_tree_layer(
            input_digests: &[Digest],
            nodes: &[Digest],
            remaining_layers: u32,
        ) {
            if remaining_layers == 0 {
                return;
            }
            let output_count = input_digests.len() / 2;
            let (this_layer, rest) = nodes.split_at(output_count);

            for i in 0..output_count {
                let left = input_digests[2 * i];
                let right = input_digests[2 * i + 1];
                let state: [u32; 16] = {
                    let mut s = [0u32; 16];
                    s[..8].copy_from_slice(&left);
                    s[8..].copy_from_slice(&right);
                    s
                };
                let mut expected = Digest::default();
                Blake2sState::compress_two_to_one::<USE_REDUCED_BLAKE2_ROUNDS>(
                    &state,
                    &mut expected,
                );
                assert_eq!(
                    this_layer[i], expected,
                    "Node hash mismatch at layer depth, index {}",
                    i
                );
            }

            verify_tree_layer(this_layer, rest, remaining_layers - 1);
        }
        verify_tree_layer(&leaves_host, &nodes_host, node_layers_count);
        println!(
            "Step 5b: All {} Merkle tree node layers verified",
            node_layers_count
        );

        // Verify tree root is non-zero
        let root_offset = num_leaves - 2;
        let root = nodes_host[root_offset];
        assert!(
            root.iter().any(|&x| x != 0),
            "Merkle tree root is all zeros"
        );
        println!("Step 5c: Merkle root is non-zero: {:08x?}", &root[..4]);

        // --- Step 6: Verify NTT roundtrip (bitrev Z -> main evals recovers original) ---
        let mut roundtrip_buf = MetalBuffer::<BF>::alloc(device, memory_size).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        bitrev_Z_to_natural_composition_main_evals(
            device,
            &cmd_buf,
            &z_buf,
            &mut roundtrip_buf,
            stride as u32,
            log_n as u32,
            num_bf_cols as u32,
            &twiddle_data,
        )
        .unwrap();
        cmd_buf.commit_and_wait();

        let mut roundtrip_host = vec![BF::ZERO; memory_size];
        unsafe { roundtrip_buf.copy_to_slice(&mut roundtrip_host); }

        assert_eq!(
            &original_src,
            &roundtrip_host,
            "NTT roundtrip failed: main evals -> Z -> main evals did not recover original data"
        );
        println!("Step 6: NTT roundtrip verified (main evals -> Z -> main evals = identity)");

        println!("E2E pipeline test PASSED: NTT + Blake2s Merkle tree pipeline is correct");
    }
}

#[cfg(test)]
mod pow_tests {
    use crate::blake2s::{blake2s_pow, STATE_SIZE};
    use crate::metal_runtime::*;
    use crate::prover::context::{ProverContext, ProverContextConfig};
    use blake2s_u32::{
        round_function_reduced_rounds, BLAKE2S_BLOCK_SIZE_U32_WORDS,
        BLAKE2S_DIGEST_SIZE_U32_WORDS, CONFIGURED_IV, IV,
    };

    /// Compute Blake2s PoW hash on CPU for a given seed + nonce, return state[0].
    fn cpu_pow_hash(seed: &[u32; 8], nonce: u64) -> u32 {
        let initial_state = [
            CONFIGURED_IV[0], CONFIGURED_IV[1], CONFIGURED_IV[2], CONFIGURED_IV[3],
            CONFIGURED_IV[4], CONFIGURED_IV[5], CONFIGURED_IV[6], CONFIGURED_IV[7],
            IV[0], IV[1], IV[2], IV[3],
            IV[4] ^ (((BLAKE2S_DIGEST_SIZE_U32_WORDS + 2) * std::mem::size_of::<u32>()) as u32),
            IV[5],
            IV[6] ^ 0xffffffff,
            IV[7],
        ];
        let mut input = [0u32; BLAKE2S_BLOCK_SIZE_U32_WORDS];
        input[..8].copy_from_slice(seed);
        input[8] = nonce as u32;
        input[9] = (nonce >> 32) as u32;

        let mut state = initial_state;
        round_function_reduced_rounds(&mut state, &input);
        // Final hash word 0 = CONFIGURED_IV[0] ^ state[0] ^ state[8]
        CONFIGURED_IV[0] ^ state[0] ^ state[8]
    }

    #[test]
    fn test_delegation_trace_raw_layout() {
        use crate::witness::trace_delegation::DelegationTraceRaw;
        let size = std::mem::size_of::<DelegationTraceRaw>();
        let align = std::mem::align_of::<DelegationTraceRaw>();
        let dummy = unsafe { std::mem::zeroed::<DelegationTraceRaw>() };
        let base = &dummy as *const _ as usize;
        let dt_off = &dummy.delegation_type as *const _ as usize - base;
        let iap_off = &dummy.indirect_accesses_properties as *const _ as usize - base;
        let wt_off = &dummy.write_timestamp as *const _ as usize - base;
        let ra_off = &dummy.register_accesses as *const _ as usize - base;
        println!("DelegationTraceRaw: size={size}, align={align}");
        println!("  delegation_type offset: {dt_off}");
        println!("  indirect_accesses_properties offset: {iap_off}");
        println!("  write_timestamp ptr offset: {wt_off}");
        println!("  register_accesses ptr offset: {ra_off}");

        // Expected: matches Metal DelegationTrace struct
        // 5 u32s = 20 bytes, u16 at 20, pad 2, u32[2][24] at 24 (192 bytes),
        // 4 ptrs at 216 (32 bytes) = total 248
        assert_eq!(dt_off, 20, "delegation_type should be at offset 20");
        assert_eq!(iap_off, 24, "indirect_accesses_properties should be at offset 24");
        assert_eq!(wt_off, 216, "write_timestamp should be at offset 216");
        assert_eq!(size, 248, "total size should be 248");
    }

    #[test]
    fn test_pow_debug_kernel_writes() {
        use crate::metal_runtime::dispatch;
        let context = ProverContext::new(&ProverContextConfig::default()).unwrap();
        let device = context.device();

        let seed = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let d_seed = context.alloc_from_slice(&seed).unwrap();
        let result: crate::metal_runtime::MetalBuffer<u32> = context.alloc_from_slice(&[0u32; 16]).unwrap();
        let bits_count = 8u32;
        let start_nonce = 488u64;
        let max_nonce = 489u64;

        let cmd_buf = context.new_command_buffer().unwrap();
        let config = dispatch::MetalLaunchConfig::basic_1d(1, 32);
        dispatch::dispatch_kernel(
            device, &cmd_buf, "ab_blake2s_pow_debug_kernel", &config,
            |encoder| {
                dispatch::set_buffer(encoder, 0, d_seed.raw(), 0);
                unsafe {
                    dispatch::set_bytes(encoder, 1, &bits_count);
                    dispatch::set_bytes(encoder, 2, &start_nonce);
                    dispatch::set_bytes(encoder, 3, &max_nonce);
                }
                dispatch::set_buffer(encoder, 4, result.raw(), 0);
            },
        ).unwrap();
        cmd_buf.commit_and_wait();

        let r: Vec<u32> = (0..9).map(|i| unsafe { *result.as_ptr().add(i) }).collect();
        println!("bits_count  = {}", r[0]);
        println!("start_nonce = 0x{:08x}_{:08x}", r[2], r[1]);
        println!("max_nonce   = 0x{:08x}_{:08x}", r[4], r[3]);
        println!("digest_mask = 0x{:08x}", r[5]);
        println!("state[0]    = 0x{:08x}", r[6]);
        println!("masked      = 0x{:08x}", r[7]);
        println!("passes      = {}", r[8]);

        let cpu_hash = cpu_pow_hash(&seed, 488);
        println!("CPU hash    = 0x{:08x}", cpu_hash);

        assert_eq!(r[0], 8, "bits_count should be 8");
        assert_eq!(r[5], 0xFF000000, "digest_mask should be 0xFF000000");
        assert_eq!(r[6], cpu_hash, "GPU and CPU hash must match");
        assert_eq!(r[8], 1, "hash should pass the PoW check");
    }

    #[test]
    fn test_pow_gpu_vs_cpu_hash() {
        let context = ProverContext::new(&ProverContextConfig::default()).unwrap();
        let device = context.device();

        // Known seed (all zeros for simplicity)
        let seed = [0u32; 8];
        // Test with a few known nonces
        for nonce in [0u64, 1, 42, 12345, 0xDEADBEEF] {
            let cpu_hash = cpu_pow_hash(&seed, nonce);

            let d_seed = context.alloc_from_slice(&seed).unwrap();
            let d_nonce_lo = context.alloc_from_slice(&[0xFFFFFFFFu32]).unwrap();
            let d_nonce_hi = context.alloc_from_slice(&[0xFFFFFFFFu32]).unwrap();

            // Search with pow_bits=1 (accept ~50% of hashes) to avoid UB with bits=0
            let cmd_buf = context.new_command_buffer().unwrap();
            blake2s_pow(device, &cmd_buf, &d_seed, 1, nonce, nonce + 1, &d_nonce_lo, &d_nonce_hi, 1024).unwrap();
            cmd_buf.commit_and_wait();

            let found_lo = unsafe { *d_nonce_lo.as_ptr() };
            let found_hi = unsafe { *d_nonce_hi.as_ptr() };
            let found_nonce = found_lo as u64 | ((found_hi as u64) << 32);

            println!("nonce={nonce}: cpu_hash=0x{cpu_hash:08x}, gpu_found_nonce={found_nonce} (lo=0x{found_lo:08x}, hi=0x{found_hi:08x})");

            // With pow_bits=1, ~50% of hashes pass. Check if GPU found this specific nonce
            // (top bit of hash is 0 for this nonce = pass)
            if cpu_hash & 0x80000000 == 0 {
                assert_eq!(found_nonce, nonce, "GPU should have found nonce {nonce} (hash passes)");
            } else {
                println!("nonce={nonce}: hash 0x{cpu_hash:08x} has top bit set, won't pass pow_bits=1");
            }
        }
    }

    #[test]
    fn test_pow_finds_known_cpu_solution() {
        let context = ProverContext::new(&ProverContextConfig::default()).unwrap();
        let device = context.device();

        // Find a nonce that satisfies 8-bit PoW on CPU
        let seed = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let pow_bits = 8u32;
        let mask = 0xffffffffu32 >> pow_bits;

        let mut cpu_nonce = None;
        for n in 0u64..100_000 {
            let h = cpu_pow_hash(&seed, n);
            if h <= mask {
                cpu_nonce = Some(n);
                println!("CPU found 8-bit PoW: nonce={n}, hash=0x{h:08x}");
                break;
            }
        }
        let cpu_nonce = cpu_nonce.expect("CPU should find 8-bit PoW in 100k nonces");

        // Now search on GPU — it should find a solution at or before cpu_nonce
        let d_seed = context.alloc_from_slice(&seed).unwrap();
        let d_nonce_lo = context.alloc_from_slice(&[0xFFFFFFFFu32]).unwrap();
        let d_nonce_hi = context.alloc_from_slice(&[0xFFFFFFFFu32]).unwrap();

        let cmd_buf = context.new_command_buffer().unwrap();
        blake2s_pow(device, &cmd_buf, &d_seed, pow_bits, 0, cpu_nonce + 1, &d_nonce_lo, &d_nonce_hi, 1024).unwrap();
        cmd_buf.commit_and_wait();

        let found_lo = unsafe { *d_nonce_lo.as_ptr() };
        let found_hi = unsafe { *d_nonce_hi.as_ptr() };
        let found_nonce = found_lo as u64 | ((found_hi as u64) << 32);

        println!("GPU found: nonce={found_nonce} (lo=0x{found_lo:08x}, hi=0x{found_hi:08x})");

        if found_lo == 0xFFFFFFFF {
            panic!("GPU did NOT find a PoW solution, but CPU found one at nonce={cpu_nonce}. Hash mismatch!");
        }

        // Verify the GPU's nonce on CPU
        let gpu_hash = cpu_pow_hash(&seed, found_nonce);
        println!("Verifying GPU nonce on CPU: hash=0x{gpu_hash:08x}, mask=0x{mask:08x}");
        assert!(gpu_hash <= mask, "GPU nonce {found_nonce} doesn't satisfy CPU PoW check! hash=0x{gpu_hash:08x}");
        println!("GPU PoW matches CPU PoW!");
    }
}

#[cfg(test)]
mod hash_verify_tests {
    use crate::blake2s;
    use crate::prover::context::{ProverContext, ProverContextConfig};
    use field::Mersenne31Field as BF;

    #[test]
    fn test_leaf_hash_matches_cpu() {
        let context = ProverContext::new(&ProverContextConfig::default()).unwrap();
        let device = context.device();

        // 12 values: [4,0,0,0,0,0,0,0,0,0,0,0]
        let mut values = vec![BF(0); 12]; // 12 columns, 1 row
        values[0] = BF(4);

        let d_values = context.alloc_from_slice(&values).unwrap();
        let d_tree: crate::metal_runtime::MetalBuffer<blake2s::Digest> =
            context.alloc_from_slice(&[[0u32; 8]]).unwrap(); // 1 leaf

        let cmd_buf = context.new_command_buffer().unwrap();
        blake2s::build_merkle_tree_leaves_raw(
            device, &cmd_buf, d_values.raw(), values.len(), &d_tree, 1, 0,
        ).unwrap();
        cmd_buf.commit_and_wait();

        let gpu_hash = unsafe { *d_tree.as_ptr() };
        println!("GPU leaf hash: {:08x?}", gpu_hash);

        // Now compute the same hash on CPU using blake2s_u32
        let mut hasher = blake2s_u32::DelegatedBlake2sState::new();
        unsafe { hasher.reset() };
        // Input: 12 u32 words (canonicalized field elements)
        for (i, v) in values.iter().enumerate() {
            hasher.input_buffer[i] = v.0;
        }
        // Zero remaining input buffer
        for i in values.len()..16 {
            hasher.input_buffer[i] = 0;
        }
        unsafe {
            hasher.run_round_function::<true>(12, true);
        }
        let cpu_hash = hasher.read_state_for_output();
        println!("CPU leaf hash: {:08x?}", cpu_hash);

        assert_eq!(gpu_hash, cpu_hash, "GPU and CPU leaf hashes must match!");
    }
}

#[cfg(test)]
mod merkle_scale_tests {
    use crate::blake2s::{self, Digest};
    use crate::prover::context::{ProverContext, ProverContextConfig};
    use crate::prover::trace_holder::commit_trace;
    use crate::metal_runtime::MetalBuffer;
    use field::Mersenne31Field as BF;

    #[test]
    fn test_merkle_tree_large_scale() {
        // Test at production scale: 2^20 leaves, 6 columns
        let log_domain_size = 20u32;
        let domain_size = 1usize << log_domain_size;
        let cols = 6;
        let log_lde_factor = 1u32;
        let log_tree_cap_size = 7u32;
        let log_rows_per_leaf = 0u32;
        let log_coset_cap_size = log_tree_cap_size - log_lde_factor;

        let context = ProverContext::new(&ProverContextConfig::default()).unwrap();

        // Create deterministic test data
        let mut data = vec![BF(0); cols * domain_size];
        for col in 0..cols {
            for row in 0..domain_size {
                data[col * domain_size + row] = BF((col * domain_size + row) as u32 % 2147483647);
            }
        }

        let d_data: MetalBuffer<BF> = context.alloc_from_slice(&data).unwrap();
        let tree_len = 1usize << (log_domain_size + 1 - log_rows_per_leaf);
        let d_tree: MetalBuffer<Digest> = context.alloc_from_slice(&vec![[0u32; 8]; tree_len]).unwrap();
        let layers_count = log_domain_size + 1 - log_rows_per_leaf - log_coset_cap_size;

        commit_trace(
            &d_data, &mut { d_tree }, log_domain_size, log_lde_factor,
            log_rows_per_leaf, log_tree_cap_size, cols, &context,
        ).unwrap();

        println!("Test complete at 2^{} scale with {} columns", log_domain_size, cols);
    }
}
