//! Stage 2: argument polynomial computation.
//! Ported from gpu_prover/src/prover/stage_2.rs.

use super::arg_utils::*;
use super::callbacks::Callbacks;
use super::context::{HostAllocation, ProverContext};
use super::setup::SetupPrecomputations;
use super::stage_1::StageOneOutput;
use super::stage_2_kernels::*;
use super::trace_holder::{flatten_tree_caps, TraceHolder, TreesCacheMode};
use super::{BF, E4};
use crate::metal_runtime::{MetalBuffer, MetalResult};
use crate::ops_cub::device_reduce::{segmented_reduce_bf, ReduceOperation};
use crate::ops_cub::device_scan::{get_scan_temp_storage_elems, scan_e4, ScanOperation};
use blake2s_u32::BLAKE2S_DIGEST_SIZE_U32_WORDS;
use cs::definitions::{
    NUM_LOOKUP_ARGUMENT_LINEARIZATION_CHALLENGES, NUM_TIMESTAMP_COLUMNS_FOR_RAM, REGISTER_SIZE,
    TIMESTAMP_COLUMNS_NUM_BITS,
};
use cs::one_row_compiler::CompiledCircuitArtifact;
use field::{Field, FieldExtension};
use prover::definitions::Transcript;
use prover::prover_stages::cached_data::ProverCachedData;
use prover::transcript::Seed;

pub(crate) struct StageTwoOutput {
    pub(crate) trace_holder: TraceHolder<BF>,
    pub(crate) lookup_challenges: Option<HostAllocation<LookupChallenges>>,
    pub(crate) last_row: Option<HostAllocation<[BF]>>,
    pub(crate) offset_for_grand_product_poly: usize,
    pub(crate) offset_for_sum_over_delegation_poly: Option<usize>,
}

impl StageTwoOutput {
    pub fn allocate_trace_evaluations(
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
        let layout = circuit.stage_2_layout;
        let num_stage_2_cols = layout.total_width;
        let trace_holder = TraceHolder::allocate_only_evaluation(
            log_domain_size,
            log_lde_factor,
            0,
            log_tree_cap_size,
            num_stage_2_cols,
            true,
            true,
            recompute_cosets,
            trees_cache_mode,
            context,
        )?;
        Ok(Self {
            trace_holder,
            lookup_challenges: None,
            last_row: None,
            offset_for_grand_product_poly: 0,
            offset_for_sum_over_delegation_poly: None,
        })
    }

    pub fn generate(
        &mut self,
        seed: &mut Seed,
        circuit: &CompiledCircuitArtifact<BF>,
        cached_data: &ProverCachedData,
        setup: &mut SetupPrecomputations,
        stage_1_output: &mut StageOneOutput,
        _callbacks: &mut Callbacks,
        context: &ProverContext,
    ) -> MetalResult<()> {
        let trace_len = circuit.trace_len;
        assert!(trace_len.is_power_of_two());
        let log_domain_size = trace_len.trailing_zeros();
        let layout = circuit.stage_2_layout;
        let num_stage_2_cols = layout.total_width;

        // Derive lookup challenges from transcript
        let mut lookup_challenges = LookupChallenges::default();
        {
            let mut transcript_challenges = [0u32;
                ((NUM_LOOKUP_ARGUMENT_LINEARIZATION_CHALLENGES + 1) * 4)
                    .next_multiple_of(BLAKE2S_DIGEST_SIZE_U32_WORDS)];
            Transcript::draw_randomness(seed, &mut transcript_challenges);
            let mut it = transcript_challenges.as_chunks::<4>().0.iter();
            let mut get_challenge =
                || E4::from_array_of_base(it.next().unwrap().map(BF::from_nonreduced_u32));
            lookup_challenges.linearization_challenges =
                std::array::from_fn(|_| get_challenge());
            lookup_challenges.gamma = get_challenge();
        }

        let num_stage_2_bf_cols = layout.num_base_field_polys();
        let num_stage_2_e4_cols = layout.num_ext4_field_polys();
        assert_eq!(
            num_stage_2_cols,
            4 * (((num_stage_2_bf_cols + 3) / 4) + num_stage_2_e4_cols)
        );

        let setup_evaluations = setup.trace_holder.get_evaluations(context)?;
        let generic_lookup_mappings = stage_1_output.generic_lookup_mapping.take().unwrap();
        let witness_evaluations = stage_1_output.witness_holder.get_evaluations(context)?;
        let memory_evaluations = stage_1_output.memory_holder.get_evaluations(context)?;

        let trace_holder = &mut self.trace_holder;
        let evaluations = trace_holder.get_uninit_evaluations_mut();

        // Compute stage 2 arguments on main domain
        compute_stage_2_args_on_main_domain(
            &setup_evaluations,
            &witness_evaluations,
            &memory_evaluations,
            &generic_lookup_mappings,
            evaluations,
            &lookup_challenges,
            cached_data,
            circuit,
            circuit.total_tables_size,
            log_domain_size,
            context,
        )?;

        // Extend to full LDE and commit
        trace_holder.allocate_to_full(context)?;
        trace_holder.extend_and_commit(0, context)?;

        // Copy last row to host for accumulator values
        let evaluations = trace_holder.get_evaluations(context)?;
        let eval_slice = unsafe { evaluations.as_slice() };
        let mut last_row = unsafe { HostAllocation::new_uninit_slice(num_stage_2_cols) };
        {
            let accessor = last_row.get_mut_accessor();
            let lr = unsafe { accessor.get_mut() };
            for col in 0..num_stage_2_cols {
                lr[col] = eval_slice[col * trace_len + trace_len - 1];
            }
        }
        self.last_row = Some(last_row);

        // Store offsets for grand product and delegation polys
        let offset_for_grand_product_poly = layout
            .intermediate_polys_for_memory_argument
            .get_range(cached_data.offset_for_grand_product_accumulation_poly)
            .start;
        self.offset_for_grand_product_poly = offset_for_grand_product_poly;
        let offset_for_sum_over_delegation_poly =
            if cached_data.handle_delegation_requests || cached_data.process_delegations {
                Some(cached_data.delegation_processing_aux_poly.start())
            } else {
                None
            };
        self.offset_for_sum_over_delegation_poly = offset_for_sum_over_delegation_poly;

        // Store lookup challenges
        let mut lc_host: HostAllocation<LookupChallenges> =
            unsafe { HostAllocation::new_uninit() };
        {
            let accessor = lc_host.get_mut_accessor();
            let lc = unsafe { accessor.get_mut() };
            *lc = lookup_challenges;
        }
        self.lookup_challenges = Some(lc_host);

        // Update seed with stage 2 tree caps and accumulator values
        let has_delegation_processing_aux_poly = circuit
            .stage_2_layout
            .delegation_processing_aux_poly
            .is_some();
        let tree_caps_accessors = trace_holder.get_tree_caps_accessors();
        let last_row_accessor = self.last_row.as_ref().unwrap().get_accessor();
        let last_row_slice = unsafe { last_row_accessor.get() };
        let mut transcript_input = vec![];
        transcript_input.extend(flatten_tree_caps(&tree_caps_accessors));
        transcript_input.extend(
            Self::get_grand_product_accumulator(offset_for_grand_product_poly, last_row_slice)
                .into_coeffs_in_base()
                .iter()
                .map(BF::to_reduced_u32),
        );
        if has_delegation_processing_aux_poly {
            transcript_input.extend(
                Self::get_sum_over_delegation_poly(
                    offset_for_sum_over_delegation_poly,
                    last_row_slice,
                )
                .unwrap_or_default()
                .into_coeffs_in_base()
                .iter()
                .map(BF::to_reduced_u32),
            );
        }
        Transcript::commit_with_seed(seed, &transcript_input);

        Ok(())
    }

    pub fn get_grand_product_accumulator(
        offset_for_grand_product_poly: usize,
        last_row: &[BF],
    ) -> E4 {
        E4::from_array_of_base(std::array::from_fn(|i| {
            last_row[offset_for_grand_product_poly + i]
        }))
    }

    pub fn get_sum_over_delegation_poly(
        offset: Option<usize>,
        last_row: &[BF],
    ) -> Option<E4> {
        offset.map(|offset| {
            let mut value = E4::from_array_of_base(std::array::from_fn(|i| last_row[offset + i]));
            value.negate();
            value
        })
    }
}

/// Compute stage 2 arguments on the main domain.
/// This orchestrates the kernel dispatches for lookup, memory, and delegation arguments.
#[allow(clippy::too_many_arguments)]
fn compute_stage_2_args_on_main_domain(
    setup_evals: &MetalBuffer<BF>,
    witness_evals: &MetalBuffer<BF>,
    memory_evals: &MetalBuffer<BF>,
    generic_lookup_map: &MetalBuffer<u32>,
    stage_2_evals: &mut MetalBuffer<BF>,
    lookup_challenges: &LookupChallenges,
    cached_data: &ProverCachedData,
    circuit: &CompiledCircuitArtifact<BF>,
    num_generic_table_rows: usize,
    log_n: u32,
    context: &ProverContext,
) -> MetalResult<()> {
    assert_eq!(REGISTER_SIZE, 2);
    assert_eq!(NUM_TIMESTAMP_COLUMNS_FOR_RAM, 2);
    let n = 1usize << log_n;
    let layout = circuit.stage_2_layout;
    let num_generic_args = layout
        .intermediate_polys_for_generic_lookup
        .num_elements();
    let num_memory_args = layout
        .intermediate_polys_for_memory_argument
        .num_elements();
    let num_stage_2_bf_cols = layout.num_base_field_polys();
    let num_stage_2_e4_cols = layout.num_ext4_field_polys();
    let e4_cols_offset = layout.ext4_polys_offset;
    assert_eq!(e4_cols_offset % 4, 0);
    assert!(num_stage_2_bf_cols <= e4_cols_offset);
    assert!(e4_cols_offset - num_stage_2_bf_cols < 4);

    let e4_byte_offset = e4_cols_offset * n * std::mem::size_of::<BF>();
    let translate_e4_offset = |raw_col: usize| -> usize {
        assert_eq!(raw_col % 4, 0);
        assert!(raw_col >= e4_cols_offset);
        (raw_col - e4_cols_offset) / 4
    };

    let device = context.device();
    let stride = n as u32;

    // Upload lookup challenges to device buffer
    let d_lookup_challenges: MetalBuffer<LookupChallenges> =
        MetalBuffer::from_slice(device, std::slice::from_ref(lookup_challenges))?;

    // Allocate separate buffers for aggregated entry inverses
    let aggregated_entry_invs_for_rc16: MetalBuffer<E4> =
        MetalBuffer::alloc(device, 1 << 16)?;
    let aggregated_entry_invs_for_ts: MetalBuffer<E4> =
        MetalBuffer::alloc(device, 1 << TIMESTAMP_COLUMNS_NUM_BITS)?;
    let aggregated_entry_invs_for_generic: MetalBuffer<E4> = if circuit.total_tables_size > 0 {
        MetalBuffer::alloc(device, circuit.total_tables_size)?
    } else {
        MetalBuffer::alloc(device, 1)?
    };

    // Split stage_2 evaluations into bf and e4 sections
    // bf section: columns [0, e4_cols_offset) at stride n
    // e4 section: columns [e4_cols_offset, total_width) at stride n
    // The kernel wrappers take the full buffer; offsets are encoded via col indices.
    // For Metal, we pass the same buffer with different byte offsets.
    // Actually, the Metal kernel wrappers take separate bf and e4 buffers.
    // We need to create sub-buffers or use the full buffer with offset.

    // Zero padding columns between num_stage_2_bf_cols and e4_cols_offset
    {
        let stage_2_slice = unsafe { stage_2_evals.as_mut_slice() };
        for padding_col in num_stage_2_bf_cols..e4_cols_offset {
            let col_start = padding_col * n;
            for row in 0..n {
                stage_2_slice[col_start + row] = BF::ZERO;
            }
        }
    }

    // Clone cached data fields
    let ProverCachedData {
        trace_len,
        memory_timestamp_high_from_circuit_idx,
        delegation_type: _,
        memory_argument_challenges,
        execute_delegation_argument,
        delegation_challenges,
        process_shuffle_ram_init,
        shuffle_ram_inits_and_teardowns,
        lazy_init_address_range_check_16,
        handle_delegation_requests,
        delegation_request_layout: _,
        process_batch_ram_access,
        process_registers_and_indirect_access,
        delegation_processor_layout,
        process_delegations,
        delegation_processing_aux_poly,
        num_set_polys_for_memory_shuffle,
        offset_for_grand_product_accumulation_poly: _,
        range_check_16_multiplicities_src,
        range_check_16_multiplicities_dst,
        timestamp_range_check_multiplicities_src,
        timestamp_range_check_multiplicities_dst,
        generic_lookup_multiplicities_src_start,
        generic_lookup_multiplicities_dst_start,
        generic_lookup_setup_columns_start,
        range_check_16_width_1_lookups_access,
        range_check_16_width_1_lookups_access_via_expressions,
        timestamp_range_check_width_1_lookups_access_via_expressions,
        timestamp_range_check_width_1_lookups_access_via_expressions_for_shuffle_ram,
        memory_accumulator_dst_start,
        ..
    } = cached_data.clone();
    assert_eq!(trace_len, n);

    // Range check 16 aggregated entry invs + multiplicities
    let dummy_setup_column = 0u32;
    let num_range_check_16_rows = 1u32 << 16;
    let range_check_16_multiplicities_dst_col =
        translate_e4_offset(range_check_16_multiplicities_dst) as u32;
    // ALL stage 2 dispatches share ONE command buffer.
    // Metal guarantees sequential execution of compute encoders within a cmd buffer,
    // so data dependencies (entry invs → lookup → negate → scan) are naturally satisfied.
    let cmd_buf = context.new_command_buffer()?;

    let num_timestamp_range_check_rows = 1u32 << TIMESTAMP_COLUMNS_NUM_BITS;
    let timestamp_range_check_multiplicities_dst_col =
        translate_e4_offset(timestamp_range_check_multiplicities_dst) as u32;
    launch_range_check_aggregated_entry_invs(
        device,
        &cmd_buf,
        &d_lookup_challenges,
        witness_evals,
        stride,
        setup_evals,
        stride,
        stage_2_evals,
        e4_byte_offset,
        stride,
        &aggregated_entry_invs_for_rc16,
        dummy_setup_column,
        range_check_16_multiplicities_src as u32,
        range_check_16_multiplicities_dst_col,
        1,
        num_range_check_16_rows,
        log_n,
    )?;
    launch_range_check_aggregated_entry_invs(
        device,
        &cmd_buf,
        &d_lookup_challenges,
        witness_evals,
        stride,
        setup_evals,
        stride,
        stage_2_evals,
        e4_byte_offset,
        stride,
        &aggregated_entry_invs_for_ts,
        dummy_setup_column,
        timestamp_range_check_multiplicities_src as u32,
        timestamp_range_check_multiplicities_dst_col,
        1,
        num_timestamp_range_check_rows,
        log_n,
    )?;

    // Generic aggregated entry invs + multiplicities
    if num_generic_table_rows > 0 {
        assert!(num_generic_args > 0);
        let num_generic_multiplicities_cols = circuit
            .setup_layout
            .generic_lookup_setup_columns
            .num_elements();
        let generic_lookup_multiplicities_dst_cols_start =
            translate_e4_offset(generic_lookup_multiplicities_dst_start) as u32;
        let lookup_encoding_capacity = n - 1;
        let num_generic_table_rows_tail =
            (num_generic_table_rows % lookup_encoding_capacity) as u32;

        launch_generic_aggregated_entry_invs(
            device,
            &cmd_buf,
            &d_lookup_challenges,
            witness_evals,
            stride,
            setup_evals,
            stride,
            stage_2_evals,
            e4_byte_offset,
            stride,
            &aggregated_entry_invs_for_generic,
            generic_lookup_setup_columns_start as u32,
            generic_lookup_multiplicities_src_start as u32,
            generic_lookup_multiplicities_dst_cols_start,
            num_generic_multiplicities_cols as u32,
            num_generic_table_rows_tail,
            log_n,
        )?;
    } else {
        assert_eq!(num_generic_args, 0);
    }

    // Delegation aux poly
    if circuit.memory_layout.delegation_processor_layout.is_none()
        && circuit.memory_layout.delegation_request_layout.is_none()
    {
        assert_eq!(
            circuit
                .stage_2_layout
                .intermediate_polys_for_generic_multiplicities
                .full_range()
                .end,
            circuit
                .stage_2_layout
                .intermediate_polys_for_memory_argument
                .start()
        );
    } else {
        assert!(!delegation_challenges.delegation_argument_gamma.is_zero());
    }

    if handle_delegation_requests || process_delegations {
        assert!(execute_delegation_argument);
        let del_challenges = DelegationChallenges::new(&delegation_challenges);
        let (request_metadata, processing_metadata) =
            get_delegation_metadata(cached_data, circuit);
        let delegation_aux_poly_col =
            translate_e4_offset(delegation_processing_aux_poly.start()) as u32;

        let d_del_challenges: MetalBuffer<DelegationChallenges> =
            MetalBuffer::from_slice(device, std::slice::from_ref(&del_challenges))?;
        let d_request_metadata: MetalBuffer<DelegationRequestMetadata> =
            MetalBuffer::from_slice(device, std::slice::from_ref(&request_metadata))?;
        let d_processing_metadata: MetalBuffer<DelegationProcessingMetadata> =
            MetalBuffer::from_slice(device, std::slice::from_ref(&processing_metadata))?;

        launch_delegation_aux_poly(
            device,
            &cmd_buf,
            &d_del_challenges,
            &d_request_metadata,
            &d_processing_metadata,
            memory_evals,
            stride,
            setup_evals,
            stride,
            stage_2_evals,
            e4_byte_offset,
            stride,
            delegation_aux_poly_col,
            handle_delegation_requests,
            log_n,
        )?;
    }

    // Lookup args kernel
    let range_check_16_layout = RangeCheck16ArgsLayout::new(
        circuit,
        &range_check_16_width_1_lookups_access,
        &range_check_16_width_1_lookups_access_via_expressions,
        &translate_e4_offset,
    );
    let expressions_layout = if !range_check_16_width_1_lookups_access_via_expressions.is_empty()
        || !timestamp_range_check_width_1_lookups_access_via_expressions.is_empty()
    {
        let expect_constant_terms_are_zero = process_shuffle_ram_init;
        FlattenedLookupExpressionsLayout::new(
            &range_check_16_width_1_lookups_access_via_expressions,
            &timestamp_range_check_width_1_lookups_access_via_expressions,
            num_stage_2_bf_cols,
            num_stage_2_e4_cols,
            expect_constant_terms_are_zero,
            &translate_e4_offset,
        )
    } else {
        FlattenedLookupExpressionsLayout::default()
    };
    let expressions_for_shuffle_ram_layout =
        if !timestamp_range_check_width_1_lookups_access_via_expressions_for_shuffle_ram.is_empty()
        {
            FlattenedLookupExpressionsForShuffleRamLayout::new(
                &timestamp_range_check_width_1_lookups_access_via_expressions_for_shuffle_ram,
                num_stage_2_bf_cols,
                num_stage_2_e4_cols,
                &translate_e4_offset,
            )
        } else {
            FlattenedLookupExpressionsForShuffleRamLayout::default()
        };
    let lazy_init_teardown_layout = if process_shuffle_ram_init {
        LazyInitTeardownLayout::new(
            circuit,
            &lazy_init_address_range_check_16,
            &shuffle_ram_inits_and_teardowns,
            &translate_e4_offset,
        )
    } else {
        LazyInitTeardownLayout::default()
    };
    let generic_args_start = if num_generic_args > 0 {
        translate_e4_offset(
            circuit
                .stage_2_layout
                .intermediate_polys_for_generic_lookup
                .start(),
        ) as u32
    } else {
        0
    };

    // Upload struct layouts to device
    let d_rc16_layout: MetalBuffer<RangeCheck16ArgsLayout> =
        MetalBuffer::from_slice(device, std::slice::from_ref(&range_check_16_layout))?;
    let d_expressions: MetalBuffer<FlattenedLookupExpressionsLayout> =
        MetalBuffer::from_slice(device, std::slice::from_ref(&expressions_layout))?;
    let d_expressions_shuffle_ram: MetalBuffer<FlattenedLookupExpressionsForShuffleRamLayout> =
        MetalBuffer::from_slice(
            device,
            std::slice::from_ref(&expressions_for_shuffle_ram_layout),
        )?;
    let d_lazy_init: MetalBuffer<LazyInitTeardownLayout> =
        MetalBuffer::from_slice(device, std::slice::from_ref(&lazy_init_teardown_layout))?;

    launch_lookup_args(
        device,
        &cmd_buf,
        &d_rc16_layout,
        &d_expressions,
        &d_expressions_shuffle_ram,
        &d_lazy_init,
        setup_evals,
        stride,
        witness_evals,
        stride,
        memory_evals,
        stride,
        &aggregated_entry_invs_for_rc16,
        &aggregated_entry_invs_for_ts,
        &aggregated_entry_invs_for_generic,
        generic_args_start,
        num_generic_args as u32,
        generic_lookup_map,
        stride,
        stage_2_evals,
        stride,
        stage_2_evals,
        e4_byte_offset,
        stride,
        memory_timestamp_high_from_circuit_idx,
        num_stage_2_bf_cols as u32,
        num_stage_2_e4_cols as u32,
        log_n,
    )?;

    // Memory args
    let memory_challenges = MemoryChallenges::new(&memory_argument_challenges);
    let raw_memory_args_start = circuit
        .stage_2_layout
        .intermediate_polys_for_memory_argument
        .start();
    assert_eq!(raw_memory_args_start, memory_accumulator_dst_start);
    let memory_args_start = translate_e4_offset(raw_memory_args_start) as u32;

    if process_shuffle_ram_init {
        assert!(!process_batch_ram_access);
        assert!(!process_registers_and_indirect_access);
        let write_timestamp_in_setup_start = circuit.setup_layout.timestamp_setup_columns.start();
        let shuffle_ram_access_sets = &circuit.memory_layout.shuffle_ram_access_sets;
        assert_eq!(
            num_memory_args,
            1 + shuffle_ram_access_sets.len() + 1,
        );
        assert_eq!(num_memory_args, num_set_polys_for_memory_shuffle);
        let shuffle_ram_accesses =
            ShuffleRamAccesses::new(shuffle_ram_access_sets, write_timestamp_in_setup_start);

        let d_memory_challenges: MetalBuffer<MemoryChallenges> =
            MetalBuffer::from_slice(device, std::slice::from_ref(&memory_challenges))?;
        let d_shuffle_ram_accesses: MetalBuffer<ShuffleRamAccesses> =
            MetalBuffer::from_slice(device, std::slice::from_ref(&shuffle_ram_accesses))?;

        launch_shuffle_ram_memory_args(
            device,
            &cmd_buf,
            &d_memory_challenges,
            &d_shuffle_ram_accesses,
            setup_evals,
            stride,
            memory_evals,
            stride,
            stage_2_evals,
            e4_byte_offset,
            stride,
            &d_lazy_init,
            memory_timestamp_high_from_circuit_idx,
            memory_args_start,
            log_n,
        )?;
    } else {
        assert!(process_batch_ram_access || process_registers_and_indirect_access);
        assert!(process_batch_ram_access != process_registers_and_indirect_access);
        assert_eq!(circuit.memory_layout.shuffle_ram_access_sets.len(), 0);
    }

    if process_batch_ram_access {
        let batched_ram_accesses = &circuit.memory_layout.batched_ram_accesses;
        assert!(!batched_ram_accesses.is_empty());
        let write_timestamp_col = delegation_processor_layout.write_timestamp.start();
        let abi_mem_offset_high_col = delegation_processor_layout.abi_mem_offset_high.start();
        let batched_ram_accesses = BatchedRamAccesses::new(
            &memory_challenges,
            batched_ram_accesses,
            write_timestamp_col,
            abi_mem_offset_high_col,
        );

        let d_memory_challenges: MetalBuffer<MemoryChallenges> =
            MetalBuffer::from_slice(device, std::slice::from_ref(&memory_challenges))?;
        let d_batched_ram: MetalBuffer<BatchedRamAccesses> =
            MetalBuffer::from_slice(device, std::slice::from_ref(&batched_ram_accesses))?;

        launch_batched_ram_memory_args(
            device,
            &cmd_buf,
            &d_memory_challenges,
            &d_batched_ram,
            memory_evals,
            stride,
            stage_2_evals,
            e4_byte_offset,
            stride,
            memory_args_start,
            log_n,
        )?;
    }

    if process_registers_and_indirect_access {
        let register_and_indirect_accesses = &circuit.memory_layout.register_and_indirect_accesses;
        assert!(!register_and_indirect_accesses.is_empty());
        let write_timestamp_col = delegation_processor_layout.write_timestamp.start();
        let register_and_indirect_accesses = RegisterAndIndirectAccesses::new(
            &memory_challenges,
            register_and_indirect_accesses,
            write_timestamp_col,
        );

        let d_memory_challenges: MetalBuffer<MemoryChallenges> =
            MetalBuffer::from_slice(device, std::slice::from_ref(&memory_challenges))?;
        let d_reg_indirect: MetalBuffer<RegisterAndIndirectAccesses> =
            MetalBuffer::from_slice(
                device,
                std::slice::from_ref(&register_and_indirect_accesses),
            )?;

        launch_register_and_indirect_memory_args(
            device,
            &cmd_buf,
            &d_memory_challenges,
            &d_reg_indirect,
            memory_evals,
            stride,
            stage_2_evals,
            e4_byte_offset,
            stride,
            memory_args_start,
            log_n,
        )?;
    }

    // c0 = 0 adjustment for bf cols: GPU segmented reduce + negate last row
    {
        let num_cols_to_reduce = if handle_delegation_requests || process_delegations {
            // Also reduce 4 E4 columns (stored as 4 BF columns) for delegation aux poly
            let start_col = delegation_processing_aux_poly.start();
            assert!(start_col + 4 <= stage_2_evals.len() / n);
            start_col + 4
        } else {
            num_stage_2_bf_cols
        };
        let d_col_sums: MetalBuffer<BF> = context.alloc(num_cols_to_reduce)?;
        segmented_reduce_bf(
            context.device(),
            &cmd_buf,
            ReduceOperation::Sum,
            stage_2_evals,
            &d_col_sums,
            n as u32,              // stride between segments
            num_cols_to_reduce as u32,
            (n - 1) as u32,        // sum first n-1 rows (exclude last)
        )?;
        // Negate sums and scatter to last row — all on GPU (no CPU readback)
        let num_cols_u32 = num_cols_to_reduce as u32;
        let stride_u32 = n as u32;
        let config = crate::metal_runtime::dispatch::MetalLaunchConfig::basic_1d(
            (num_cols_u32 + 255) / 256, 256);
        crate::metal_runtime::dispatch::dispatch_kernel(
            device, &cmd_buf,
            "ab_negate_and_scatter_to_last_row_kernel",
            &config,
            |encoder| {
                crate::metal_runtime::dispatch::set_buffer(encoder, 0, d_col_sums.raw(), 0);
                crate::metal_runtime::dispatch::set_buffer(encoder, 1, stage_2_evals.raw(), 0);
                unsafe {
                    crate::metal_runtime::dispatch::set_bytes(encoder, 2, &num_cols_u32);
                    crate::metal_runtime::dispatch::set_bytes(encoder, 3, &stride_u32);
                }
            },
        )?;
    }

    // Grand product: prefix product scan of second-to-last memory arg column,
    // result goes into last memory arg column.
    assert!(num_memory_args >= 2);
    {
        // The second-to-last memory arg column (vectorized E4) is at:
        //   e4_cols_offset + 4*(memory_args_start_local + num_memory_args - 2)
        // where memory_args_start_local = translate_e4_offset(raw_memory_args_start)
        let memory_args_start_local = translate_e4_offset(raw_memory_args_start);
        let second_to_last_e4_col = memory_args_start_local + num_memory_args - 2;
        let last_e4_col = memory_args_start_local + num_memory_args - 1;

        // Transpose column-major E4 data into contiguous E4 buffer for GPU scan
        let d_scan_in: MetalBuffer<E4> = context.alloc(n)?;
        let d_scan_out: MetalBuffer<E4> = context.alloc(n)?;
        let num_blocks = get_scan_temp_storage_elems(n as u32);
        let d_scan_temp: MetalBuffer<E4> = context.alloc(num_blocks as usize)?;
        {
            let base_col = (e4_cols_offset + 4 * second_to_last_e4_col) as u32;
            crate::ops_simple::transpose_bf4_to_e4_strided(
                context.device(), &cmd_buf,
                stage_2_evals.raw(), d_scan_in.raw(),
                n as u32, base_col, n as u32,
            )?;
        }

        // GPU exclusive prefix product scan
        scan_e4(
            context.device(),
            &cmd_buf,
            ScanOperation::Product,
            false, // exclusive
            &d_scan_in,
            &d_scan_out,
            &d_scan_temp,
            n as u32,
        )?;

        // Transpose scan result back into column-major layout (GPU)
        {
            let base_col = (e4_cols_offset + 4 * last_e4_col) as u32;
            crate::ops_simple::transpose_e4_to_bf4(
                context.device(), &cmd_buf,
                d_scan_out.raw(), stage_2_evals.raw(),
                n as u32, base_col, n as u32,
            )?;
        }
    }

    // ONE sync point for entire stage 2 argument computation
    cmd_buf.commit_and_wait();

    Ok(())
}
