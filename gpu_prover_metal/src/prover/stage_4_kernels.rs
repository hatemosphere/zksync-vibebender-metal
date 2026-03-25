//! Metal dispatch wrappers for stage 4 kernels.
//! Ports gpu_prover/src/prover/stage_4_kernels.rs from CUDA to Metal.

use crate::device_context::DeviceContext;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes};
use crate::metal_runtime::{MetalBuffer, MetalCommandBuffer, MetalLaunchConfig, MetalResult};
use cs::one_row_compiler::{ColumnAddress, CompiledCircuitArtifact};
use fft::materialize_powers_serial_starting_with_one;
use field::{Field, FieldExtension, Mersenne31Complex, Mersenne31Field, Mersenne31Quartic};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use prover::prover_stages::cached_data::ProverCachedData;
use std::alloc::Global;

type BF = Mersenne31Field;
type E2 = Mersenne31Complex;
type E4 = Mersenne31Quartic;

const WARP_SIZE: u32 = 32;
const MAX_WITNESS_COLS: usize = 672;
const DOES_NOT_NEED_Z_OMEGA: u32 = u32::MAX;
const MAX_NON_WITNESS_TERMS_AT_Z: usize = 704;
const MAX_NON_WITNESS_TERMS_AT_Z_OMEGA: usize = 3;

#[derive(Clone)]
#[repr(C)]
pub struct ColIdxsToChallengeIdxsMap {
    pub map: [u32; MAX_WITNESS_COLS],
}

#[derive(Clone, Default)]
#[repr(C)]
pub struct NonWitnessChallengesAtZOmega {
    pub challenges: [E4; MAX_NON_WITNESS_TERMS_AT_Z_OMEGA],
}

#[derive(Clone, Default)]
#[repr(C)]
pub(super) struct ChallengesTimesEvals {
    pub at_z_sum_neg: E4,
    pub at_z_omega_sum_neg: E4,
}

pub fn compute_deep_denom_at_z_on_main_domain(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    denom_at_z: &MetalBuffer<E4>,
    z: &MetalBuffer<E4>,
    log_n: u32,
    bit_reversed: bool,
    device_ctx: &DeviceContext,
) -> MetalResult<()> {
    let inv_batch: u32 = 4;
    let n = 1u32 << log_n;
    let block_dim = WARP_SIZE * 4;
    let grid_dim = (n + inv_batch * block_dim - 1) / (inv_batch * block_dim);
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);
    let bit_reversed_flag: u32 = if bit_reversed { 1 } else { 0 };
    let denom_stride = n;

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_deep_denom_at_z_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            set_buffer(encoder, idx, denom_at_z.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &denom_stride); } idx += 1;
            set_buffer(encoder, idx, z.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &log_n); } idx += 1;
            unsafe { set_bytes(encoder, idx, &bit_reversed_flag); } idx += 1;
            // Powers data
            set_buffer(encoder, idx, device_ctx.powers_of_w_fine.raw(), 0); idx += 1;
            let fine_mask = (1u32 << device_ctx.fine_log_count) - 1;
            unsafe { set_bytes(encoder, idx, &fine_mask); } idx += 1;
            unsafe { set_bytes(encoder, idx, &device_ctx.fine_log_count); } idx += 1;
            set_buffer(encoder, idx, device_ctx.powers_of_w_coarser.raw(), 0); idx += 1;
            let coarser_mask = (1u32 << device_ctx.coarser_log_count) - 1;
            unsafe { set_bytes(encoder, idx, &coarser_mask); } idx += 1;
            unsafe { set_bytes(encoder, idx, &device_ctx.coarser_log_count); } idx += 1;
            set_buffer(encoder, idx, device_ctx.powers_of_w_coarsest.raw(), 0); idx += 1;
            let coarsest_mask = (1u32 << device_ctx.coarsest_log_count) - 1;
            unsafe { set_bytes(encoder, idx, &coarsest_mask); }
        },
    )
}

pub fn get_e4_scratch_count_for_deep_quotiening() -> usize {
    2 * MAX_WITNESS_COLS + MAX_NON_WITNESS_TERMS_AT_Z
}

#[derive(Clone)]
pub(super) struct Metadata {
    pub witness_cols_to_challenges_at_z_omega_map: ColIdxsToChallengeIdxsMap,
    pub memory_lazy_init_addresses_cols_start: usize,
    pub num_non_witness_terms_at_z: usize,
}

pub(super) fn get_metadata(
    evals: &[E4],
    alpha: E4,
    omega_inv: E2,
    cached_data: &ProverCachedData,
    circuit: &CompiledCircuitArtifact<BF>,
    scratch_e4: &mut [E4],
    challenges_times_evals: &mut ChallengesTimesEvals,
    non_witness_challenges_at_z_omega: &mut NonWitnessChallengesAtZOmega,
) -> Metadata {
    let num_setup_cols = circuit.setup_layout.total_width;
    let num_witness_cols = circuit.witness_layout.total_width;
    let num_memory_cols = circuit.memory_layout.total_width;
    let num_stage_2_bf_cols = circuit.stage_2_layout.num_base_field_polys();
    let num_stage_2_e4_cols = circuit.stage_2_layout.num_ext4_field_polys();
    let e4_cols_offset = circuit.stage_2_layout.ext4_polys_offset;
    assert_eq!(e4_cols_offset % 4, 0);
    assert!(num_stage_2_bf_cols <= e4_cols_offset);
    assert!(e4_cols_offset - num_stage_2_bf_cols < 4);
    let num_terms_at_z = circuit.num_openings_at_z();
    let mut num_terms_at_z_doublecheck = num_setup_cols;
    num_terms_at_z_doublecheck += num_witness_cols;
    num_terms_at_z_doublecheck += num_memory_cols;
    num_terms_at_z_doublecheck += num_stage_2_bf_cols;
    num_terms_at_z_doublecheck += num_stage_2_e4_cols;
    num_terms_at_z_doublecheck += 1; // composition quotient
    assert_eq!(num_terms_at_z, num_terms_at_z_doublecheck);
    let num_terms_at_z_omega = circuit.num_openings_at_z_omega();
    let num_terms_total = num_terms_at_z + num_terms_at_z_omega;
    let mut challenges =
        materialize_powers_serial_starting_with_one::<_, Global>(alpha, num_terms_total);
    // Fold omega adjustment into challenges at z * omega
    for challenge in (&mut challenges[num_terms_at_z..]).iter_mut() {
        challenge.mul_assign_by_base(&omega_inv);
    }
    assert_eq!(evals.len(), num_terms_total);
    let challenges_at_z = &challenges[0..num_terms_at_z];
    let evals_at_z = &evals[0..num_terms_at_z];
    let challenges_times_evals_at_z_sum_neg = *challenges_at_z
        .iter()
        .zip(evals_at_z)
        .fold(E4::ZERO, |acc, (challenge, eval)| {
            *acc.clone().add_assign(challenge.clone().mul_assign(&eval))
        })
        .negate();
    let challenges_at_z_omega = &challenges[num_terms_at_z..];
    let evals_at_z_omega = &evals[num_terms_at_z..];
    let challenges_times_evals_at_z_omega_sum_neg = *challenges_at_z_omega
        .iter()
        .zip(evals_at_z_omega)
        .fold(E4::ZERO, |acc, (challenge, eval)| {
            *acc.clone().add_assign(challenge.clone().mul_assign(&eval))
        })
        .negate();
    // Organize challenges so the kernel can associate them with cols
    let mut flat_offset = 0;
    // Organize challenges at z
    let setup_challenges = &challenges[0..num_setup_cols];
    flat_offset += num_setup_cols;
    (&mut scratch_e4[0..num_witness_cols])
        .copy_from_slice(&challenges[flat_offset..flat_offset + num_witness_cols]);
    flat_offset += num_witness_cols;
    let memory_challenges_at_z = &challenges[flat_offset..flat_offset + num_memory_cols];
    flat_offset += num_memory_cols;
    let stage_2_bf_challenges = &challenges[flat_offset..flat_offset + num_stage_2_bf_cols];
    flat_offset += num_stage_2_bf_cols;
    let stage_2_e4_challenges = &challenges[flat_offset..flat_offset + num_stage_2_e4_cols];
    flat_offset += num_stage_2_e4_cols;
    let composition_challenge = &challenges[flat_offset..flat_offset + 1];
    flat_offset += 1;
    assert_eq!(flat_offset, num_terms_at_z);
    // Organize challenges at z * omega
    assert!(num_witness_cols <= MAX_WITNESS_COLS);
    let mut witness_cols_to_challenges_at_z_omega_map = ColIdxsToChallengeIdxsMap {
        map: [DOES_NOT_NEED_Z_OMEGA; MAX_WITNESS_COLS],
    };
    for (i, (_src, dst)) in circuit.state_linkage_constraints.iter().enumerate() {
        let ColumnAddress::WitnessSubtree(col_idx) = *dst else {
            panic!()
        };
        assert_eq!(
            witness_cols_to_challenges_at_z_omega_map.map[col_idx],
            DOES_NOT_NEED_Z_OMEGA
        );
        assert!(i < (MAX_WITNESS_COLS as usize));
        witness_cols_to_challenges_at_z_omega_map.map[col_idx] = i as u32;
        scratch_e4[num_witness_cols + i] = challenges[flat_offset];
        flat_offset += 1;
    }
    let num_witness_terms_at_z_omega = circuit.state_linkage_constraints.len();
    let (memory_challenges_at_z_omega, memory_lazy_init_addresses_cols_start) =
        if let Some(shuffle_ram_inits_and_teardowns) =
            circuit.memory_layout.shuffle_ram_inits_and_teardowns
        {
            assert!(cached_data.process_shuffle_ram_init);
            let challenges = (&challenges[flat_offset..flat_offset + 2]).to_vec();
            let start = shuffle_ram_inits_and_teardowns
                .lazy_init_addresses_columns
                .start();
            flat_offset += 2;
            (challenges, start)
        } else {
            assert!(!cached_data.process_shuffle_ram_init);
            (vec![], 0)
        };
    let stage_2_memory_grand_product_challenge = &challenges[flat_offset..flat_offset + 1];
    flat_offset += 1;
    assert_eq!(flat_offset, num_terms_total);
    // Now marshal arguments for GPU transfer
    let flat_non_witness_challenges_at_z: Vec<E4> = setup_challenges
        .iter()
        .chain(memory_challenges_at_z.iter())
        .chain(stage_2_bf_challenges.iter())
        .chain(stage_2_e4_challenges.iter())
        .chain(composition_challenge.iter())
        .map(|x| *x)
        .collect();
    let num_non_witness_terms_at_z = flat_non_witness_challenges_at_z.len();
    assert!(num_non_witness_terms_at_z < MAX_NON_WITNESS_TERMS_AT_Z);
    let flat_non_witness_challenges_at_z_omega: Vec<E4> = memory_challenges_at_z_omega
        .iter()
        .chain(stage_2_memory_grand_product_challenge.iter())
        .map(|x| *x)
        .collect();
    assert!(flat_non_witness_challenges_at_z_omega.len() <= MAX_NON_WITNESS_TERMS_AT_Z_OMEGA);
    assert_eq!(
        flat_non_witness_challenges_at_z_omega.len() + num_witness_terms_at_z_omega,
        num_terms_at_z_omega
    );
    let num_witness_terms = num_witness_cols + num_witness_terms_at_z_omega;
    assert_eq!(
        flat_non_witness_challenges_at_z.len()
            + flat_non_witness_challenges_at_z_omega.len()
            + num_witness_terms,
        num_terms_total,
    );
    assert!(num_witness_terms + num_non_witness_terms_at_z <= scratch_e4.len());
    (&mut scratch_e4[num_witness_terms..num_witness_terms + num_non_witness_terms_at_z])
        .copy_from_slice(&flat_non_witness_challenges_at_z);
    *challenges_times_evals = ChallengesTimesEvals {
        at_z_sum_neg: challenges_times_evals_at_z_sum_neg,
        at_z_omega_sum_neg: challenges_times_evals_at_z_omega_sum_neg,
    };
    for (i, challenge) in flat_non_witness_challenges_at_z_omega.iter().enumerate() {
        non_witness_challenges_at_z_omega.challenges[i] = *challenge;
    }
    Metadata {
        witness_cols_to_challenges_at_z_omega_map,
        memory_lazy_init_addresses_cols_start,
        num_non_witness_terms_at_z,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn compute_deep_quotient_on_main_domain(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    metadata: Metadata,
    setup_cols: &MetalBuffer<BF>,
    setup_stride: u32,
    witness_cols: &MetalBuffer<BF>,
    witness_stride: u32,
    memory_cols: &MetalBuffer<BF>,
    memory_stride: u32,
    stage_2_cols: &MetalBuffer<BF>,
    stage_2_stride: u32,
    composition_col: &MetalBuffer<BF>,
    composition_stride: u32,
    denom_at_z: &MetalBuffer<E4>,
    scratch_e4: &MetalBuffer<E4>,
    challenges_times_evals: &MetalBuffer<ChallengesTimesEvals>,
    non_witness_challenges_at_z_omega: &MetalBuffer<NonWitnessChallengesAtZOmega>,
    witness_cols_to_challenges_at_z_omega_map: &MetalBuffer<ColIdxsToChallengeIdxsMap>,
    quotient: &MetalBuffer<BF>,
    quotient_stride: u32,
    cached_data: &ProverCachedData,
    circuit: &CompiledCircuitArtifact<BF>,
    log_n: u32,
    bit_reversed: bool,
) -> MetalResult<()> {
    let n = 1u32 << log_n;
    let num_setup_cols = circuit.setup_layout.total_width as u32;
    let num_witness_cols = circuit.witness_layout.total_width as u32;
    let num_memory_cols = circuit.memory_layout.total_width as u32;
    let num_stage_2_bf_cols = circuit.stage_2_layout.num_base_field_polys() as u32;
    let num_stage_2_e4_cols = circuit.stage_2_layout.num_ext4_field_polys() as u32;
    let e4_cols_offset = circuit.stage_2_layout.ext4_polys_offset;

    // Split stage_2_cols into bf and e4 sections
    let stage_2_bf_offset = 0usize;
    let stage_2_e4_offset_bytes = e4_cols_offset * (stage_2_stride as usize) * std::mem::size_of::<BF>();

    let num_witness_terms_at_z_omega = circuit.state_linkage_constraints.len();
    let num_witness_terms = num_witness_cols as usize + num_witness_terms_at_z_omega;

    let Metadata {
        witness_cols_to_challenges_at_z_omega_map: _,
        memory_lazy_init_addresses_cols_start,
        num_non_witness_terms_at_z: _,
    } = metadata;

    let block_dim = 512u32;
    let grid_dim = (n + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);
    let process_shuffle_ram_init_flag: u32 = if cached_data.process_shuffle_ram_init { 1 } else { 0 };
    let bit_reversed_flag: u32 = if bit_reversed { 1 } else { 0 };
    let memory_lazy_init_addresses_cols_start = memory_lazy_init_addresses_cols_start as u32;

    // witness_challenges_at_z is scratch_e4[0..num_witness_cols]
    // witness_challenges_at_z_omega is scratch_e4[num_witness_cols..num_witness_cols + num_witness_terms_at_z_omega]
    // non_witness_challenges_at_z is scratch_e4[num_witness_terms..num_witness_terms + num_non_witness_terms_at_z]
    let witness_challenges_at_z_offset = 0usize;
    let witness_challenges_at_z_omega_offset = num_witness_cols as usize * std::mem::size_of::<E4>();
    let non_witness_challenges_at_z_offset = num_witness_terms * std::mem::size_of::<E4>();

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_deep_quotient_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            set_buffer(encoder, idx, setup_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &setup_stride); } idx += 1;
            set_buffer(encoder, idx, witness_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &witness_stride); } idx += 1;
            set_buffer(encoder, idx, memory_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &memory_stride); } idx += 1;
            // stage_2_bf_cols
            set_buffer(encoder, idx, stage_2_cols.raw(), stage_2_bf_offset); idx += 1;
            unsafe { set_bytes(encoder, idx, &stage_2_stride); } idx += 1;
            // stage_2_e4_cols
            set_buffer(encoder, idx, stage_2_cols.raw(), stage_2_e4_offset_bytes); idx += 1;
            unsafe { set_bytes(encoder, idx, &stage_2_stride); } idx += 1;
            set_buffer(encoder, idx, composition_col.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &composition_stride); } idx += 1;
            set_buffer(encoder, idx, denom_at_z.raw(), 0); idx += 1;
            // witness_challenges_at_z
            set_buffer(encoder, idx, scratch_e4.raw(), witness_challenges_at_z_offset); idx += 1;
            // witness_challenges_at_z_omega
            set_buffer(encoder, idx, scratch_e4.raw(), witness_challenges_at_z_omega_offset); idx += 1;
            set_buffer(encoder, idx, witness_cols_to_challenges_at_z_omega_map.raw(), 0); idx += 1;
            // non_witness_challenges_at_z
            set_buffer(encoder, idx, scratch_e4.raw(), non_witness_challenges_at_z_offset); idx += 1;
            set_buffer(encoder, idx, non_witness_challenges_at_z_omega.raw(), 0); idx += 1;
            set_buffer(encoder, idx, challenges_times_evals.raw(), 0); idx += 1;
            set_buffer(encoder, idx, quotient.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &quotient_stride); } idx += 1;
            unsafe { set_bytes(encoder, idx, &num_setup_cols); } idx += 1;
            unsafe { set_bytes(encoder, idx, &num_witness_cols); } idx += 1;
            unsafe { set_bytes(encoder, idx, &num_memory_cols); } idx += 1;
            unsafe { set_bytes(encoder, idx, &num_stage_2_bf_cols); } idx += 1;
            unsafe { set_bytes(encoder, idx, &num_stage_2_e4_cols); } idx += 1;
            unsafe { set_bytes(encoder, idx, &process_shuffle_ram_init_flag); } idx += 1;
            unsafe { set_bytes(encoder, idx, &memory_lazy_init_addresses_cols_start); } idx += 1;
            unsafe { set_bytes(encoder, idx, &log_n); } idx += 1;
            unsafe { set_bytes(encoder, idx, &bit_reversed_flag); }
        },
    )
}
