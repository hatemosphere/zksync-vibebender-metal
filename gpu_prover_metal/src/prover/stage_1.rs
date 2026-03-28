//! Stage 1: witness generation and trace commitment.
//! Ported from gpu_prover/src/prover/stage_1.rs.

use super::arg_utils::{
    FlattenedLookupExpressionsForShuffleRamLayout, FlattenedLookupExpressionsLayout,
    RangeCheck16ArgsLayout,
};
use super::callbacks::Callbacks;
use super::context::{HostAllocation, ProverContext};
use super::setup::SetupPrecomputations;
use super::trace_holder::{TraceHolder, TreesCacheMode};
use super::tracing_data::{TracingDataDevice, TracingDataTransfer};
use super::BF;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes, MetalLaunchConfig};
use crate::metal_runtime::{MetalBuffer, MetalResult};
use cs::definitions::{
    split_timestamp, timestamp_high_contribution_from_circuit_sequence,
    BoundaryConstraintLocation, COMMON_TABLE_WIDTH, NUM_COLUMNS_FOR_COMMON_TABLE_WIDTH_SETUP,
    TIMESTAMP_COLUMNS_NUM_BITS,
};
use cs::one_row_compiler::{ColumnAddress, CompiledCircuitArtifact, LookupExpression};
use fft::GoodAllocator;
use field::Field;
use prover::prover_stages::cached_data::{
    get_range_check_16_lookup_accesses, get_timestamp_range_check_lookup_accesses,
};
use std::sync::Arc;

pub(crate) struct StageOneOutput {
    pub witness_holder: TraceHolder<BF>,
    pub memory_holder: TraceHolder<BF>,
    pub generic_lookup_mapping: Option<MetalBuffer<u32>>,
    pub public_inputs: Option<HostAllocation<[BF]>>,
}

impl StageOneOutput {
    pub fn allocate_trace_holders(
        circuit: &CompiledCircuitArtifact<BF>,
        log_lde_factor: u32,
        log_tree_cap_size: u32,
        recompute_cosets: bool,
        trees_cache_mode: TreesCacheMode,
        context: &ProverContext,
    ) -> MetalResult<Self> {
        let trace_len = circuit.trace_len;
        assert!(trace_len.is_power_of_two());
        let log_domain_size = trace_len.trailing_zeros();
        let witness_columns_count = circuit.witness_layout.total_width;
        let witness_holder = TraceHolder::new(
            log_domain_size,
            log_lde_factor,
            0,
            log_tree_cap_size,
            witness_columns_count,
            true,
            true,
            recompute_cosets,
            trees_cache_mode,
            context,
        )?;
        let memory_columns_count = circuit.memory_layout.total_width;
        let memory_holder = TraceHolder::new(
            log_domain_size,
            log_lde_factor,
            0,
            log_tree_cap_size,
            memory_columns_count,
            true,
            true,
            recompute_cosets,
            trees_cache_mode,
            context,
        )?;
        Ok(Self {
            witness_holder,
            memory_holder,
            generic_lookup_mapping: None,
            public_inputs: None,
        })
    }

    pub fn generate_witness(
        &mut self,
        circuit: &CompiledCircuitArtifact<BF>,
        setup: &mut SetupPrecomputations,
        tracing_data_transfer: TracingDataTransfer<'_, impl GoodAllocator>,
        circuit_sequence: usize,
        context: &ProverContext,
    ) -> MetalResult<()> {
        let _g = crate::cpu_scoped!("s1_witness_prep");
        let trace_len = circuit.trace_len;
        assert!(trace_len.is_power_of_two());
        let log_domain_size = trace_len.trailing_zeros();
        let witness_subtree = &circuit.witness_layout;
        let memory_subtree = &circuit.memory_layout;
        let generic_lookup_mapping_size = witness_subtree.width_3_lookups.len() << log_domain_size;
        let generic_lookup_mapping: MetalBuffer<u32> =
            context.alloc(generic_lookup_mapping_size)?;

        let TracingDataTransfer {
            circuit_type,
            data_host: _,
            data_device,
            _lifetime: _,
        } = tracing_data_transfer;

        let device = context.device();
        let cmd_buf = context.new_command_buffer()?;

        assert_eq!(COMMON_TABLE_WIDTH, 3);
        assert_eq!(NUM_COLUMNS_FOR_COMMON_TABLE_WIDTH_SETUP, 4);
        let lookup_start = circuit.setup_layout.generic_lookup_setup_columns.start * trace_len;
        let lookup_tables_byte_offset = lookup_start * std::mem::size_of::<BF>();
        let setup_evaluations = setup.trace_holder.get_evaluations(context)?;

        let timestamp_high_from_circuit_sequence =
            timestamp_high_contribution_from_circuit_sequence(circuit_sequence, trace_len);

        let memory_evaluations = self.memory_holder.get_uninit_evaluations_mut();
        let witness_evaluations = self.witness_holder.get_uninit_evaluations_mut();

        let stride = trace_len as u32;

        match data_device {
            TracingDataDevice::Main {
                setup_and_teardown,
                trace,
            } => {
                let count = trace.cycle_data.len() as u32;
                // Zero witness evaluations on GPU
                crate::ops_simple::memset_zero(
                    device, &cmd_buf,
                    witness_evaluations.raw(),
                    witness_evaluations.byte_len(),
                )?;

                crate::witness::memory_main::generate_memory_and_witness_values_main(
                    device,
                    &cmd_buf,
                    memory_subtree,
                    &circuit.memory_queries_timestamp_comparison_aux_vars,
                    &setup_and_teardown,
                    circuit.lazy_init_address_aux_vars.as_ref().unwrap(),
                    &trace,
                    timestamp_high_from_circuit_sequence,
                    memory_evaluations,
                    witness_evaluations,
                    stride,
                    count,
                )?;

                crate::witness::witness_main::generate_witness_values_main(
                    device,
                    &cmd_buf,
                    circuit_type.as_main().unwrap(),
                    &trace,
                    setup_evaluations,
                    lookup_tables_byte_offset,
                    memory_evaluations,
                    witness_evaluations,
                    &generic_lookup_mapping,
                    stride,
                    count,
                )?;
            }
            TracingDataDevice::Delegation(trace) => {
                let range_check_16_multiplicities_columns =
                    witness_subtree.multiplicities_columns_for_range_check_16;
                let timestamp_range_check_multiplicities_columns =
                    witness_subtree.multiplicities_columns_for_timestamp_range_check;
                let generic_multiplicities_columns =
                    witness_subtree.multiplicities_columns_for_generic_lookup;

                assert_eq!(
                    range_check_16_multiplicities_columns.start
                        + range_check_16_multiplicities_columns.num_elements,
                    timestamp_range_check_multiplicities_columns.start
                );
                assert_eq!(
                    timestamp_range_check_multiplicities_columns.start
                        + timestamp_range_check_multiplicities_columns.num_elements,
                    generic_multiplicities_columns.start
                );

                // Zero entire witness and memory buffers on GPU.
                // Delegation kernels only fill rows 0..count; the rest must be zero.
                crate::ops_simple::memset_zero(
                    device, &cmd_buf,
                    witness_evaluations.raw(),
                    witness_evaluations.byte_len(),
                )?;
                crate::ops_simple::memset_zero(
                    device, &cmd_buf,
                    memory_evaluations.raw(),
                    memory_evaluations.byte_len(),
                )?;

                // count = padded num_requests (process ALL rows including padding).
                // The kernel's DelegationTrace.num_requests uses write_timestamp.len()
                // for bounds-checking — rows beyond actual data return 0.
                let count = trace.num_requests as u32;
                crate::witness::memory_delegation::generate_memory_and_witness_values_delegation(
                    device,
                    &cmd_buf,
                    memory_subtree,
                    &circuit.register_and_indirect_access_timestamp_comparison_aux_vars,
                    &trace,
                    memory_evaluations,
                    witness_evaluations,
                    stride,
                    count,
                    context,
                )?;

                crate::witness::witness_delegation::generate_witness_values_delegation(
                    device,
                    &cmd_buf,
                    circuit_type.as_delegation().unwrap(),
                    &trace,
                    setup_evaluations,
                    lookup_tables_byte_offset,
                    memory_evaluations,
                    witness_evaluations,
                    &generic_lookup_mapping,
                    stride,
                    count,
                    context,
                )?;
            }
        };

        // Generate generic lookup multiplicities via GPU atomic scatter-add
        // Dispatched on SAME cmd_buf as witness gen — Metal sequential encoding ensures
        // witness data is available before scatter-add reads it.
        {
            let generic_multiplicities_columns =
                witness_subtree.multiplicities_columns_for_generic_lookup;
            let num_lookup_cols = witness_subtree.width_3_lookups.len() as u32;
            let mult_cols_start = generic_multiplicities_columns.start as u32;
            let stride = trace_len as u32;
            let total_threads = num_lookup_cols * (stride - 1);
            let block_dim = 256u32;
            let grid_dim = (total_threads + block_dim - 1) / block_dim;
            let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);
            let device = context.device();
            dispatch_kernel(device, &cmd_buf, "ab_scatter_add_multiplicities_kernel", &config, |encoder| {
                set_buffer(encoder, 0, generic_lookup_mapping.raw(), 0);
                set_buffer(encoder, 1, witness_evaluations.raw(), 0);
                unsafe {
                    set_bytes(encoder, 2, &stride);
                    set_bytes(encoder, 3, &mult_cols_start);
                    set_bytes(encoder, 4, &num_lookup_cols);
                }
            })?;
        }
        cmd_buf.commit_and_wait();
        generate_range_check_multiplicities_gpu(
            circuit,
            setup_evaluations,
            witness_evaluations,
            memory_evaluations,
            timestamp_high_from_circuit_sequence,
            trace_len,
            context,
        )?;

        self.generic_lookup_mapping = Some(generic_lookup_mapping);
        Ok(())
    }

    pub fn commit_witness(
        &mut self,
        circuit: &Arc<CompiledCircuitArtifact<BF>>,
        callbacks: &mut Callbacks,
        context: &ProverContext,
    ) -> MetalResult<()> {
        {
            let _g = crate::cpu_scoped!("s1_memory_extend_commit");
            self.memory_holder
                .make_evaluations_sum_to_zero_extend_and_commit(context)?;
        }
        {
            let _g = crate::cpu_scoped!("s1_witness_extend_commit");
            self.witness_holder
                .make_evaluations_sum_to_zero_extend_and_commit(context)?;
        }
        {
            let _g = crate::cpu_scoped!("s1_public_inputs");
            self.produce_public_inputs(circuit, callbacks, context)?;
        }
        Ok(())
    }

    pub fn produce_public_inputs(
        &mut self,
        circuit: &Arc<CompiledCircuitArtifact<BF>>,
        _callbacks: &mut Callbacks,
        context: &ProverContext,
    ) -> MetalResult<()> {
        if self.public_inputs.is_some() {
            return Ok(());
        }
        if circuit.public_inputs.is_empty() {
            self.public_inputs = Some(unsafe { HostAllocation::new_uninit_slice(0) });
            return Ok(());
        }
        let holder = &mut self.witness_holder;
        let trace_len = 1 << holder.log_domain_size;
        let evaluations = holder.get_evaluations(context)?;
        let eval_slice = unsafe { evaluations.as_slice() };
        let total_len = circuit.public_inputs.len();
        let mut public_inputs = unsafe { HostAllocation::new_uninit_slice(total_len) };
        {
            let accessor = public_inputs.get_mut_accessor();
            let pi_slice = unsafe { accessor.get_mut() };
            for (idx, (location, column_address)) in circuit.public_inputs.iter().enumerate() {
                let row_idx = match location {
                    BoundaryConstraintLocation::FirstRow => 0,
                    BoundaryConstraintLocation::OneBeforeLastRow => trace_len - 2,
                    BoundaryConstraintLocation::LastRow => {
                        panic!("public inputs on the last row are not supported");
                    }
                };
                pi_slice[idx] =
                    read_public_input_value_at_row(*column_address, eval_slice, trace_len, row_idx);
            }
        }
        self.public_inputs = Some(public_inputs);
        Ok(())
    }
}

#[inline(always)]
fn read_public_input_value_at_row(
    column_address: ColumnAddress,
    eval_slice: &[BF],
    trace_len: usize,
    row_idx: usize,
) -> BF {
    match column_address {
        ColumnAddress::WitnessSubtree(offset) => eval_slice[offset * trace_len + row_idx],
        ColumnAddress::MemorySubtree(_) => {
            panic!("memory public inputs are not supported in Metal stage_1")
        }
        ColumnAddress::SetupSubtree(_) | ColumnAddress::OptimizedOut(_) => {
            panic!("unsupported public input column address in Metal stage_1")
        }
    }
}

/// Generate range check multiplicities using GPU atomic histogram kernel.
fn generate_range_check_multiplicities_gpu(
    circuit: &CompiledCircuitArtifact<BF>,
    setup_evals: &MetalBuffer<BF>,
    witness_evals: &mut MetalBuffer<BF>,
    memory_evals: &MetalBuffer<BF>,
    timestamp_high_from_circuit_sequence: cs::definitions::TimestampScalar,
    trace_len: usize,
    context: &ProverContext,
) -> MetalResult<()> {
    let _g = crate::cpu_scoped!("s1_range_check_mults");
    assert!(trace_len.is_power_of_two());

    // Build layout structs (same as stage_3_kernels Metadata::new)
    let num_stage_2_bf_cols = circuit.stage_2_layout.num_base_field_polys();
    let num_stage_2_e4_cols = circuit.stage_2_layout.num_ext4_field_polys();
    let e4_cols_offset = circuit.stage_2_layout.ext4_polys_offset;
    let translate_e4_offset = |raw_col: usize| -> usize {
        assert_eq!(raw_col % 4, 0);
        assert!(raw_col >= e4_cols_offset);
        (raw_col - e4_cols_offset) / 4
    };

    let (rc16_access, rc16_access_via_expressions) = get_range_check_16_lookup_accesses(circuit);
    let (ts_access_via_expressions, ts_access_for_shuffle_ram) =
        get_timestamp_range_check_lookup_accesses(circuit);

    let rc16_layout = RangeCheck16ArgsLayout::new(
        circuit,
        &rc16_access,
        &rc16_access_via_expressions,
        &translate_e4_offset,
    );

    let expressions_layout = if !rc16_access_via_expressions.is_empty()
        || !ts_access_via_expressions.is_empty()
    {
        let (process_shuffle_ram_init, _) =
            if circuit.memory_layout.shuffle_ram_inits_and_teardowns.is_some() {
                (true, 0)
            } else {
                (false, 0)
            };
        FlattenedLookupExpressionsLayout::new(
            &rc16_access_via_expressions,
            &ts_access_via_expressions,
            num_stage_2_bf_cols,
            num_stage_2_e4_cols,
            process_shuffle_ram_init,
            &translate_e4_offset,
        )
    } else {
        FlattenedLookupExpressionsLayout::default()
    };

    let expressions_for_shuffle_ram_layout = if !ts_access_for_shuffle_ram.is_empty() {
        FlattenedLookupExpressionsForShuffleRamLayout::new(
            &ts_access_for_shuffle_ram,
            num_stage_2_bf_cols,
            num_stage_2_e4_cols,
            &translate_e4_offset,
        )
    } else {
        FlattenedLookupExpressionsForShuffleRamLayout::default()
    };

    let (process_shuffle_ram_init, lazy_init_address_start) =
        if let Some(sat) = circuit.memory_layout.shuffle_ram_inits_and_teardowns {
            (true, sat.lazy_init_addresses_columns.start())
        } else {
            (false, 0)
        };

    let (_, high) = split_timestamp(timestamp_high_from_circuit_sequence);
    let memory_timestamp_high = BF::from_nonreduced_u32(high);

    // Allocate histograms (zero-initialized)
    let rc16_size = 1usize << 16; // 65536 bins
    let ts_size = 1usize << TIMESTAMP_COLUMNS_NUM_BITS;
    let d_rc16_hist: MetalBuffer<u32> = context.alloc(rc16_size)?;
    let d_ts_hist: MetalBuffer<u32> = context.alloc(ts_size)?;
    // Upload layout structs
    let d_rc16_layout = context.alloc_from_slice(std::slice::from_ref(&rc16_layout))?;
    let d_expressions = context.alloc_from_slice(std::slice::from_ref(&expressions_layout))?;
    let d_expressions_for_shuffle_ram =
        context.alloc_from_slice(std::slice::from_ref(&expressions_for_shuffle_ram_layout))?;

    // Zero histograms + range_check kernel in ONE command buffer
    let stride = trace_len as u32;
    let process_flag = if process_shuffle_ram_init { 1u32 } else { 0u32 };
    let lazy_start = lazy_init_address_start as u32;
    let n_threads = trace_len as u32 - 1; // skip last row
    let block_dim = 256u32;
    let grid_dim = (n_threads + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);
    let device = context.device();
    let cmd_buf = context.new_command_buffer()?;
    crate::ops_simple::memset_zero(device, &cmd_buf, d_rc16_hist.raw(), d_rc16_hist.byte_len())?;
    crate::ops_simple::memset_zero(device, &cmd_buf, d_ts_hist.raw(), d_ts_hist.byte_len())?;
    dispatch_kernel(
        device,
        &cmd_buf,
        "ab_range_check_multiplicities_kernel",
        &config,
        |encoder| {
            set_buffer(encoder, 0, setup_evals.raw(), 0);
            set_buffer(encoder, 1, witness_evals.raw(), 0);
            set_buffer(encoder, 2, memory_evals.raw(), 0);
            set_buffer(encoder, 3, d_rc16_layout.raw(), 0);
            set_buffer(encoder, 4, d_expressions.raw(), 0);
            set_buffer(encoder, 5, d_expressions_for_shuffle_ram.raw(), 0);
            set_buffer(encoder, 6, d_rc16_hist.raw(), 0);
            set_buffer(encoder, 7, d_ts_hist.raw(), 0);
            unsafe {
                set_bytes(encoder, 8, &stride);
                set_bytes(encoder, 9, &stride);
                set_bytes(encoder, 10, &stride);
                set_bytes(encoder, 11, &memory_timestamp_high);
                set_bytes(encoder, 12, &process_flag);
                set_bytes(encoder, 13, &lazy_start);
                set_bytes(encoder, 14, &stride);
            }
        },
    )?;
    cmd_buf.commit_and_wait();

    // Copy histograms to witness multiplicities columns
    let _g2 = crate::cpu_scoped!("s1_rc_hist_copy");
    let witness_slice = unsafe { witness_evals.as_mut_slice() };

    // RC16 multiplicities
    let rc16_mult_col = circuit
        .witness_layout
        .multiplicities_columns_for_range_check_16
        .start;
    let rc16_col_start = rc16_mult_col * trace_len;
    let rc16_hist = unsafe { d_rc16_hist.as_slice() };
    for i in 0..rc16_size.min(trace_len - 1) {
        witness_slice[rc16_col_start + i] = BF::from_nonreduced_u32(rc16_hist[i]);
    }
    for i in rc16_size..trace_len {
        witness_slice[rc16_col_start + i] = BF::ZERO;
    }

    // Timestamp multiplicities
    let ts_mult_cols = circuit
        .witness_layout
        .multiplicities_columns_for_timestamp_range_check;
    if ts_mult_cols.num_elements() > 0 {
        let ts_mult_col = ts_mult_cols.start;
        let ts_col_start = ts_mult_col * trace_len;
        let ts_hist = unsafe { d_ts_hist.as_slice() };
        for i in 0..ts_size.min(trace_len - 1) {
            witness_slice[ts_col_start + i] = BF::from_nonreduced_u32(ts_hist[i]);
        }
        for i in ts_size..trace_len {
            witness_slice[ts_col_start + i] = BF::ZERO;
        }
    }

    Ok(())
}

/// Generate range check multiplicities on CPU using unified memory (unused, kept as reference).
#[allow(dead_code)]
fn generate_range_check_multiplicities(
    circuit: &CompiledCircuitArtifact<BF>,
    setup_evals: &MetalBuffer<BF>,
    witness_evals: &mut MetalBuffer<BF>,
    memory_evals: &MetalBuffer<BF>,
    timestamp_high_from_circuit_sequence: cs::definitions::TimestampScalar,
    trace_len: usize,
) -> MetalResult<()> {
    assert!(trace_len.is_power_of_two());

    let setup_slice = unsafe { setup_evals.as_slice() };
    let witness_slice = unsafe { witness_evals.as_mut_slice() };
    let memory_slice = unsafe { memory_evals.as_slice() };

    // --- Range check 16 multiplicities ---
    let mut rc16_counts = vec![0u32; 1 << 16];

    // Count trivial range check 16 columns (direct witness columns)
    let num_trivial = circuit.witness_layout.range_check_16_columns.num_elements();
    for (expr_idx, expr) in circuit
        .witness_layout
        .range_check_16_lookup_expressions[..num_trivial]
        .iter()
        .enumerate()
    {
        let LookupExpression::Variable(place) = expr else {
            panic!("expected Variable in trivial range check 16 expression {}", expr_idx);
        };
        let ColumnAddress::WitnessSubtree(offset) = place else {
            panic!("expected WitnessSubtree in range check 16 expression");
        };
        let col_start = *offset * trace_len;
        for row in 0..trace_len - 1 {
            let val = witness_slice[col_start + row].to_reduced_u32();
            if val > u16::MAX as u32 {
                panic!("range check 16 value out of range: val={} at witness col={} row={} (expr_idx={})", val, offset, row, expr_idx);
            }
            rc16_counts[val as usize] += 1;
        }
    }

    // Count non-trivial range check 16 expressions
    let nontrivial_rc16 =
        &circuit.witness_layout.range_check_16_lookup_expressions[num_trivial..];
    for (nt_idx, expr) in nontrivial_rc16.iter().enumerate() {
        for row in 0..trace_len - 1 {
            let val = evaluate_lookup_expression_column_major(
                expr, witness_slice, memory_slice, setup_slice, row, trace_len,
            );
            if val > u16::MAX as u32 {
                // Debug: show which columns the expression reads
                let expr_debug = format!("{:?}", expr);
                panic!("nontrivial range check 16 out of range: val={val} (0x{val:08x}) row={row} nt_expr={nt_idx} trace_len={trace_len}\nexpr: {expr_debug}");
            }
            rc16_counts[val as usize] += 1;
        }
    }

    // Count lazy init address values (split as two 16-bit halves)
    if let Some(shuffle_ram_inits) = circuit.memory_layout.shuffle_ram_inits_and_teardowns {
        let start = shuffle_ram_inits.lazy_init_addresses_columns.start();
        for offset in start..start + 2 {
            let col_start = offset * trace_len;
            for row in 0..trace_len - 1 {
                let val = memory_slice[col_start + row].to_reduced_u32();
                debug_assert!(val <= u16::MAX as u32);
                rc16_counts[val as usize] += 1;
            }
        }
    }

    // Write range check 16 multiplicities
    let rc16_mult_col = circuit
        .witness_layout
        .multiplicities_columns_for_range_check_16
        .start;
    let mult_col_start = rc16_mult_col * trace_len;
    for i in 0..rc16_counts.len().min(trace_len - 1) {
        witness_slice[mult_col_start + i] = BF::from_nonreduced_u32(rc16_counts[i]);
    }
    for i in rc16_counts.len()..trace_len {
        witness_slice[mult_col_start + i] = BF::ZERO;
    }

    // --- Timestamp range check multiplicities ---
    let num_timestamp_rows = 1usize << TIMESTAMP_COLUMNS_NUM_BITS;
    let mut ts_counts = vec![0u32; num_timestamp_rows];

    let offset_for_special = circuit
        .witness_layout
        .offset_for_special_shuffle_ram_timestamps_range_check_expressions;

    // Non-shuffle-RAM timestamp range check expressions
    let ts_exprs_without_ram =
        &circuit.witness_layout.timestamp_range_check_lookup_expressions[..offset_for_special];
    for expr in ts_exprs_without_ram.iter() {
        for row in 0..trace_len - 1 {
            let val = evaluate_lookup_expression_column_major(
                expr, witness_slice, memory_slice, setup_slice, row, trace_len,
            );
            debug_assert!(val < (1 << TIMESTAMP_COLUMNS_NUM_BITS) as u32);
            if (val as usize) < ts_counts.len() {
                ts_counts[val as usize] += 1;
            }
        }
    }

    // Shuffle-RAM-specific timestamp range check expressions
    let ts_exprs_for_ram =
        &circuit.witness_layout.timestamp_range_check_lookup_expressions[offset_for_special..];
    if !ts_exprs_for_ram.is_empty() {
        let (_, high) = cs::definitions::split_timestamp(timestamp_high_from_circuit_sequence);
        let ts_high_bf = BF::from_nonreduced_u32(high);
        for expr in ts_exprs_for_ram.iter() {
            for row in 0..trace_len - 1 {
                let val = evaluate_lookup_expression_column_major_with_ts_high(
                    expr,
                    witness_slice,
                    memory_slice,
                    setup_slice,
                    row,
                    trace_len,
                    ts_high_bf,
                );
                if (val as usize) < ts_counts.len() {
                    ts_counts[val as usize] += 1;
                }
            }
        }
    }

    // Write timestamp range check multiplicities
    let ts_mult_col = circuit
        .witness_layout
        .multiplicities_columns_for_timestamp_range_check
        .start;
    let ts_mult_col_start = ts_mult_col * trace_len;
    for i in 0..ts_counts.len().min(trace_len - 1) {
        witness_slice[ts_mult_col_start + i] = BF::from_nonreduced_u32(ts_counts[i]);
    }
    for i in ts_counts.len()..trace_len {
        witness_slice[ts_mult_col_start + i] = BF::ZERO;
    }

    Ok(())
}

/// Read a single value from column-major layout given a ColumnAddress.
fn read_column_major_value(
    place: &ColumnAddress,
    witness: &[BF],
    memory: &[BF],
    setup: &[BF],
    row: usize,
    trace_len: usize,
) -> BF {
    match place {
        ColumnAddress::WitnessSubtree(offset) => witness[*offset * trace_len + row],
        ColumnAddress::MemorySubtree(offset) => memory[*offset * trace_len + row],
        ColumnAddress::SetupSubtree(offset) => setup[*offset * trace_len + row],
        _ => panic!("unsupported column address in lookup expression"),
    }
}

/// Evaluate a lookup expression at a given row in column-major layout.
fn evaluate_lookup_expression_column_major(
    expr: &LookupExpression<BF>,
    witness: &[BF],
    memory: &[BF],
    setup: &[BF],
    row: usize,
    trace_len: usize,
) -> u32 {
    match expr {
        LookupExpression::Variable(place) => {
            read_column_major_value(place, witness, memory, setup, row, trace_len).to_reduced_u32()
        }
        LookupExpression::Expression(constraint) => {
            let mut result = constraint.constant_term;
            for (coeff, place) in constraint.linear_terms.iter() {
                let mut value = read_column_major_value(place, witness, memory, setup, row, trace_len);
                value.mul_assign(coeff);
                result.add_assign(&value);
            }
            // Degree1 constraints have no quadratic terms
            result.to_reduced_u32()
        }
    }
}

/// Same as above but with timestamp high contribution for shuffle RAM expressions.
fn evaluate_lookup_expression_column_major_with_ts_high(
    expr: &LookupExpression<BF>,
    witness: &[BF],
    memory: &[BF],
    setup: &[BF],
    row: usize,
    trace_len: usize,
    _ts_high_bf: BF,
) -> u32 {
    evaluate_lookup_expression_column_major(expr, witness, memory, setup, row, trace_len)
}
