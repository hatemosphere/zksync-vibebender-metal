//! Metal dispatch wrappers for stage 3 kernels.
//! Ports gpu_prover/src/prover/stage_3_kernels.rs from CUDA to Metal.

use super::arg_utils::*;
use super::context::ProverContext;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes};
use crate::metal_runtime::{MetalBuffer, MetalCommandBuffer, MetalLaunchConfig, MetalResult};
use cs::definitions::{
    BoundaryConstraintLocation, LookupExpression, TableIndex, COMMON_TABLE_WIDTH,
    DELEGATION_ARGUMENT_CHALLENGED_IDX_FOR_TIMESTAMP_HIGH,
    DELEGATION_ARGUMENT_CHALLENGED_IDX_FOR_TIMESTAMP_LOW, NUM_LOOKUP_ARGUMENT_KEY_PARTS,
};
use cs::one_row_compiler::{ColumnAddress, CompiledCircuitArtifact};
use field::{Field, FieldExtension, Mersenne31Complex, Mersenne31Field, Mersenne31Quartic, PrimeField};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use prover::definitions::ExternalValues;
use prover::prover_stages::cached_data::ProverCachedData;
use prover::prover_stages::stage3::AlphaPowersLayout;
use std::alloc::Allocator;
use std::mem::MaybeUninit;

type BF = Mersenne31Field;
type E2 = Mersenne31Complex;
type E4 = Mersenne31Quartic;

pub const BETA_POWERS_COUNT: usize = 6;

const WARP_SIZE: u32 = 32;

fn stash_coeff(
    coeff: BF,
    coeffs_info: &mut [u8],
    explicit_coeffs: &mut [BF],
    flat_term_idx: &mut usize,
    explicit_coeff_idx: &mut usize,
) {
    if coeff == BF::ONE {
        coeffs_info[*flat_term_idx] = COEFF_IS_ONE;
    } else if coeff == BF::MINUS_ONE {
        coeffs_info[*flat_term_idx] = COEFF_IS_MINUS_ONE;
    } else {
        coeffs_info[*flat_term_idx] = COEFF_IS_EXPLICIT;
        explicit_coeffs[*explicit_coeff_idx] = coeff;
        *explicit_coeff_idx += 1;
    }
    *flat_term_idx += 1;
}

/// These values must match stage3.metal
const MAX_NON_BOOLEAN_CONSTRAINTS: usize = 192;
const MAX_TERMS: usize = 1824;
const MAX_EXPLICIT_COEFFS: usize = 632;
const MAX_FLAT_COL_IDXS: usize = 3520;
const MAX_QUADRATIC_TERMS_PER_CONSTRAINT: usize = 256;
const MAX_LINEAR_TERMS_PER_CONSTRAINT: usize = 256;
const COEFF_IS_ONE: u8 = 0x00;
const COEFF_IS_MINUS_ONE: u8 = 0x01;
const COEFF_IS_EXPLICIT: u8 = 0x02;

pub(super) const MAX_HELPER_VALUES: usize = 1536;

const LOOKUP_VAL_IS_COL_FLAG: u8 = u8::MAX;

#[derive(Clone)]
#[repr(C)]
pub struct FlattenedGenericConstraintsMetadata {
    pub coeffs_info: [u8; MAX_TERMS],
    pub explicit_coeffs: [BF; MAX_EXPLICIT_COEFFS],
    pub col_idxs: [u16; MAX_FLAT_COL_IDXS],
    pub num_linear_and_quadratic_terms_per_constraint: [[u8; 2]; MAX_NON_BOOLEAN_CONSTRAINTS],
    pub decompression_factor: E2,
    pub decompression_factor_squared: E2,
    pub every_row_zerofier: E2,
    pub omega_inv: E2,
    pub current_flat_col_idx: u32,
    pub current_flat_term_idx: u32,
    pub num_boolean_constraints: u32,
    pub num_non_boolean_quadratic_constraints: u32,
    pub num_non_boolean_constraints: u32,
}

impl FlattenedGenericConstraintsMetadata {
    fn stash_column_address(address: &ColumnAddress) -> u16 {
        match address {
            ColumnAddress::WitnessSubtree(col) => *col as u16,
            ColumnAddress::MemorySubtree(col) => (*col as u16) | ColTypeFlags::MEMORY,
            _ => panic!("unexpected ColumnAddress variant"),
        }
    }

    fn compute_every_row_zerofier(decompression_factor_squared: E2) -> E2 {
        let mut zerofier = decompression_factor_squared.clone();
        assert_eq!(zerofier, E2::from_base(BF::MINUS_ONE));
        zerofier.sub_assign_base(&BF::ONE);
        zerofier.inverse().expect("must exist")
    }

    pub fn new(
        circuit: &CompiledCircuitArtifact<BF>,
        alpha_powers: &[E4],
        tau: E2,
        omega_inv: E2,
        domain_size: usize,
        constants_times_challenges: &mut ConstantsTimesChallenges,
    ) -> Self {
        let d1cs = &circuit.degree_1_constraints;
        let d2cs = &circuit.degree_2_constraints;
        let num_degree_2_constraints = d2cs.len();
        let num_degree_1_constraints = d1cs.len();
        let num_quadratic_terms: usize = d2cs.iter().map(|x| x.quadratic_terms.len()).sum();
        let num_boolean_constraints = circuit
            .witness_layout
            .boolean_vars_columns_range
            .num_elements();
        let boolean_constraints_start = circuit.witness_layout.boolean_vars_columns_range.start();
        let num_linear_terms_in_quadratic_constraints: usize =
            d2cs.iter().map(|x| x.linear_terms.len()).sum();
        let num_linear_terms_in_linear_constraints: usize =
            d1cs.iter().map(|x| x.linear_terms.len()).sum();

        let mut coeffs_info = [0u8; MAX_TERMS];
        let mut explicit_coeffs = [BF::ZERO; MAX_EXPLICIT_COEFFS];
        let mut col_idxs = [0u16; MAX_FLAT_COL_IDXS];
        let mut num_linear_and_quadratic_terms_per_constraint =
            [[0u8; 2]; MAX_NON_BOOLEAN_CONSTRAINTS];
        let mut flat_col_idx = 0;
        let mut d2cs_iter = d2cs.iter();
        // Special economized treatment of boolean quadratic constraints
        for i in 0..num_boolean_constraints {
            let constraint = d2cs_iter.next().unwrap();
            assert_eq!(constraint.quadratic_terms.len(), 1);
            assert_eq!(constraint.linear_terms.len(), 1);
            let (coeff, a, b) = constraint.quadratic_terms[0];
            assert_eq!(coeff, BF::ONE);
            assert_eq!(a, b);
            let (coeff, a) = constraint.linear_terms[0];
            assert_eq!(coeff, BF::MINUS_ONE);
            assert_eq!(a, b);
            if let ColumnAddress::WitnessSubtree(col) = a {
                assert_eq!(col, i + boolean_constraints_start);
                col_idxs[flat_col_idx] = col as u16;
            } else {
                panic!("Boolean vars columns should be in witness trace");
            };
            flat_col_idx += 1;
        }
        let mut constraint_idx = 0;
        let mut flat_term_idx = 0;
        let mut explicit_coeff_idx = 0;
        // Non-boolean quadratic constraints
        for _ in num_boolean_constraints..num_degree_2_constraints {
            let constraint = d2cs_iter.next().unwrap();
            let num_quadratic_terms = constraint.quadratic_terms.len();
            assert!(num_quadratic_terms < MAX_QUADRATIC_TERMS_PER_CONSTRAINT);
            for (coeff, a, b) in constraint.quadratic_terms.iter() {
                stash_coeff(
                    *coeff,
                    &mut coeffs_info,
                    &mut explicit_coeffs,
                    &mut flat_term_idx,
                    &mut explicit_coeff_idx,
                );
                col_idxs[flat_col_idx] = Self::stash_column_address(a);
                flat_col_idx += 1;
                col_idxs[flat_col_idx] = Self::stash_column_address(b);
                flat_col_idx += 1;
            }
            let num_quadratic_terms = u8::try_from(num_quadratic_terms).unwrap();
            let num_linear_terms = constraint.linear_terms.len();
            assert!(num_linear_terms < MAX_LINEAR_TERMS_PER_CONSTRAINT);
            for (coeff, a) in constraint.linear_terms.iter() {
                stash_coeff(
                    *coeff,
                    &mut coeffs_info,
                    &mut explicit_coeffs,
                    &mut flat_term_idx,
                    &mut explicit_coeff_idx,
                );
                col_idxs[flat_col_idx] = Self::stash_column_address(a);
                flat_col_idx += 1;
            }
            let num_linear_terms = u8::try_from(num_linear_terms).unwrap();
            num_linear_and_quadratic_terms_per_constraint[constraint_idx] =
                [num_quadratic_terms, num_linear_terms];
            let mut constant_times_challenge =
                alpha_powers[constraint_idx + num_boolean_constraints];
            constant_times_challenge.mul_assign_by_base(&constraint.constant_term);
            constants_times_challenges
                .sum
                .add_assign(&constant_times_challenge);
            constraint_idx += 1;
        }
        assert_eq!(d2cs_iter.next(), None);
        for constraint in d1cs.iter() {
            let num_linear_terms = constraint.linear_terms.len();
            assert!(num_linear_terms < MAX_LINEAR_TERMS_PER_CONSTRAINT);
            for (coeff, a) in constraint.linear_terms.iter() {
                stash_coeff(
                    *coeff,
                    &mut coeffs_info,
                    &mut explicit_coeffs,
                    &mut flat_term_idx,
                    &mut explicit_coeff_idx,
                );
                col_idxs[flat_col_idx] = Self::stash_column_address(a);
                flat_col_idx += 1;
            }
            let num_linear_terms = u8::try_from(num_linear_terms).unwrap();
            num_linear_and_quadratic_terms_per_constraint[constraint_idx] =
                [0u8, num_linear_terms];
            let mut constant_times_challenge =
                alpha_powers[constraint_idx + num_boolean_constraints];
            constant_times_challenge.mul_assign_by_base(&constraint.constant_term);
            constants_times_challenges
                .sum
                .add_assign(&constant_times_challenge);
            constraint_idx += 1;
        }

        assert_eq!(
            constraint_idx,
            num_degree_2_constraints + num_degree_1_constraints - num_boolean_constraints,
        );
        assert_eq!(
            flat_term_idx + 2 * num_boolean_constraints,
            num_quadratic_terms
                + num_linear_terms_in_quadratic_constraints
                + num_linear_terms_in_linear_constraints,
        );
        assert_eq!(
            flat_col_idx + 2 * num_boolean_constraints,
            2 * num_quadratic_terms
                + num_linear_terms_in_quadratic_constraints
                + num_linear_terms_in_linear_constraints,
        );
        let decompression_factor = tau.pow((domain_size / 2) as u32);
        let decompression_factor_squared = *decompression_factor.clone().square();
        let every_row_zerofier = Self::compute_every_row_zerofier(decompression_factor_squared);
        Self {
            coeffs_info,
            explicit_coeffs,
            col_idxs,
            num_linear_and_quadratic_terms_per_constraint,
            decompression_factor,
            decompression_factor_squared,
            every_row_zerofier,
            omega_inv,
            current_flat_col_idx: flat_col_idx as u32,
            current_flat_term_idx: flat_term_idx as u32,
            num_boolean_constraints: num_boolean_constraints as u32,
            num_non_boolean_quadratic_constraints: (num_degree_2_constraints
                - num_boolean_constraints)
                as u32,
            num_non_boolean_constraints: (num_degree_2_constraints + num_degree_1_constraints
                - num_boolean_constraints) as u32,
        }
    }
}

// Width 3 lookups — using concrete types matching the Metal arg_utils.rs structs
// (the CUDA version uses generic const params, but Metal arg_utils already defines
// concrete DelegatedWidth3LookupsLayout and NonDelegatedWidth3LookupsLayout structs)

const DELEGATED_MAX_WIDTH_3_LOOKUPS: usize = 224;
const DELEGATED_MAX_WIDTH_3_LOOKUP_VALS: usize = 640;
const DELEGATED_MAX_WIDTH_3_LOOKUP_COEFFS: usize = 1408;
const DELEGATED_MAX_WIDTH_3_LOOKUP_COLS: usize = 1888;

const NON_DELEGATED_MAX_WIDTH_3_LOOKUPS: usize = 24;
const NON_DELEGATED_MAX_WIDTH_3_LOOKUP_VALS: usize = 72;
const NON_DELEGATED_MAX_WIDTH_3_LOOKUP_COEFFS: usize = 32;
const NON_DELEGATED_MAX_WIDTH_3_LOOKUP_COLS: usize = 96;

const MAX_TERMS_PER_WIDTH3_EXPRESSION: usize = 32;

fn width_3_lookups_new<
    const MAX_W3L: usize,
    const MAX_W3LV: usize,
    const MAX_W3LC: usize,
    const MAX_W3LCOLS: usize,
>(
    circuit: &CompiledCircuitArtifact<BF>,
    lookup_challenges: &[E4],
    lookup_gamma: E4,
    alphas: &[E4],
    alphas_offset: &mut usize,
    helpers: &mut Vec<E4, impl Allocator>,
    decompression_factor_inv: E2,
    constants_times_challenges: &mut ConstantsTimesChallenges,
    translate_e4_offset: &impl Fn(usize) -> usize,
) -> (
    [u32; MAX_W3LC],
    [u16; MAX_W3LCOLS],
    [u8; MAX_W3LV],
    [bool; MAX_W3L],
    [u16; MAX_W3L],
    u32,
    u32,
    u32,
    u32,
) {
    assert_eq!(COMMON_TABLE_WIDTH, 3);
    let mut coeffs = [0u32; MAX_W3LC];
    let mut col_idxs = [0u16; MAX_W3LCOLS];
    let mut num_terms_per_expression = [0u8; MAX_W3LV];
    let mut table_id_is_col = [false; MAX_W3L];
    let mut e4_arg_cols = [0u16; MAX_W3L];
    let mut val_idx: usize = 0;
    let mut col_idx: usize = 0;
    let mut coeff_idx: usize = 0;
    let table_id_challenge = lookup_challenges[NUM_LOOKUP_ARGUMENT_KEY_PARTS - 2];
    let mut val_challenges = Vec::with_capacity(NUM_LOOKUP_ARGUMENT_KEY_PARTS - 1);
    val_challenges.push(E4::ONE);
    val_challenges.append(&mut (&lookup_challenges[0..(NUM_LOOKUP_ARGUMENT_KEY_PARTS - 2)]).to_vec());
    let num_lookups = circuit.witness_layout.width_3_lookups.len();
    assert!(num_lookups > 0);
    assert_eq!(
        num_lookups,
        circuit
            .stage_2_layout
            .intermediate_polys_for_generic_lookup
            .num_elements()
    );
    let helpers_offset = helpers.len();
    for (term_idx, lookup_set) in circuit.witness_layout.width_3_lookups.iter().enumerate() {
        let e4_arg_col = translate_e4_offset(
            circuit
                .stage_2_layout
                .intermediate_polys_for_generic_lookup
                .get_range(term_idx)
                .start,
        );
        e4_arg_cols[term_idx] = u16::try_from(e4_arg_col).unwrap();
        let alpha = alphas[*alphas_offset];
        *alphas_offset += 1;
        match lookup_set.table_index {
            TableIndex::Constant(table_type) => {
                let id = BF::from_u64_unchecked(table_type.to_table_id() as u64);
                helpers.push(
                    *table_id_challenge
                        .clone()
                        .mul_assign_by_base(&id)
                        .add_assign(&lookup_gamma)
                        .mul_assign_by_base(&decompression_factor_inv)
                        .mul_assign(&alpha),
                );
            }
            TableIndex::Variable(place) => {
                table_id_is_col[term_idx] = true;
                col_idxs[col_idx] = match place {
                    ColumnAddress::WitnessSubtree(col) => col as u16,
                    _ => panic!("unexpected ColumnAddress variant"),
                };
                col_idx += 1;
                helpers.push(
                    *alpha
                        .clone()
                        .mul_assign(&lookup_gamma)
                        .mul_assign_by_base(&decompression_factor_inv),
                );
                helpers.push(*alpha.clone().mul_assign(&table_id_challenge));
            }
        }
        let mut lookup_is_empty = true;
        for (val, val_challenge) in lookup_set.input_columns.iter().zip(val_challenges.iter()) {
            match val {
                LookupExpression::Variable(place) => {
                    lookup_is_empty = false;
                    helpers.push(*alpha.clone().mul_assign(val_challenge));
                    col_idxs[col_idx] = match place {
                        ColumnAddress::WitnessSubtree(col) => *col as u16,
                        ColumnAddress::MemorySubtree(col) => {
                            (*col as u16) | ColTypeFlags::MEMORY
                        }
                        _ => panic!("unexpected ColumnAddress variant"),
                    };
                    col_idx += 1;
                    num_terms_per_expression[val_idx] = LOOKUP_VAL_IS_COL_FLAG;
                    val_idx += 1;
                }
                LookupExpression::Expression(a) => {
                    let num_terms = a.linear_terms.len();
                    if num_terms > 0 {
                        lookup_is_empty = false;
                        helpers.push(*alpha.clone().mul_assign(val_challenge));
                    }
                    assert_eq!(a.constant_term, BF::ZERO);
                    assert!(num_terms <= MAX_TERMS_PER_WIDTH3_EXPRESSION);
                    num_terms_per_expression[val_idx] = u8::try_from(num_terms).unwrap();
                    for (coeff, column_address) in a.linear_terms.iter() {
                        coeffs[coeff_idx] = coeff.0;
                        col_idxs[col_idx] = match column_address {
                            ColumnAddress::WitnessSubtree(col) => *col as u16,
                            ColumnAddress::MemorySubtree(col) => {
                                (*col as u16) | ColTypeFlags::MEMORY
                            }
                            _ => panic!("unexpected ColumnAddress variant"),
                        };
                        coeff_idx += 1;
                        col_idx += 1;
                    }
                    val_idx += 1;
                }
            };
        }
        assert!(!lookup_is_empty);
        constants_times_challenges.sum.sub_assign(&alpha);
    }
    let e4_arg_cols_start = translate_e4_offset(
        circuit
            .stage_2_layout
            .intermediate_polys_for_generic_lookup
            .start(),
    );
    assert_eq!(e4_arg_cols_start, e4_arg_cols[0] as usize);
    let num_helpers_used = helpers.len() - helpers_offset;
    (
        coeffs,
        col_idxs,
        num_terms_per_expression,
        table_id_is_col,
        e4_arg_cols,
        helpers_offset as u32,
        num_helpers_used as u32,
        num_lookups as u32,
        e4_arg_cols_start as u32,
    )
}

fn build_delegated_width_3_lookups(
    circuit: &CompiledCircuitArtifact<BF>,
    lookup_challenges: &[E4],
    lookup_gamma: E4,
    alphas: &[E4],
    alphas_offset: &mut usize,
    helpers: &mut Vec<E4, impl Allocator>,
    decompression_factor_inv: E2,
    constants_times_challenges: &mut ConstantsTimesChallenges,
    translate_e4_offset: &impl Fn(usize) -> usize,
) -> DelegatedWidth3LookupsLayout {
    let (coeffs, col_idxs, num_terms_per_expression, table_id_is_col, e4_arg_cols,
         helpers_offset, num_helpers_used, num_lookups, e4_arg_cols_start) =
        width_3_lookups_new::<
            DELEGATED_MAX_WIDTH_3_LOOKUPS,
            DELEGATED_MAX_WIDTH_3_LOOKUP_VALS,
            DELEGATED_MAX_WIDTH_3_LOOKUP_COEFFS,
            DELEGATED_MAX_WIDTH_3_LOOKUP_COLS,
        >(
            circuit, lookup_challenges, lookup_gamma, alphas, alphas_offset,
            helpers, decompression_factor_inv, constants_times_challenges, translate_e4_offset,
        );
    DelegatedWidth3LookupsLayout {
        coeffs, col_idxs, num_terms_per_expression, table_id_is_col, e4_arg_cols,
        helpers_offset, num_helpers_used, num_lookups, e4_arg_cols_start,
    }
}

fn build_non_delegated_width_3_lookups(
    circuit: &CompiledCircuitArtifact<BF>,
    lookup_challenges: &[E4],
    lookup_gamma: E4,
    alphas: &[E4],
    alphas_offset: &mut usize,
    helpers: &mut Vec<E4, impl Allocator>,
    decompression_factor_inv: E2,
    constants_times_challenges: &mut ConstantsTimesChallenges,
    translate_e4_offset: &impl Fn(usize) -> usize,
) -> NonDelegatedWidth3LookupsLayout {
    let (coeffs, col_idxs, num_terms_per_expression, table_id_is_col, e4_arg_cols,
         helpers_offset, num_helpers_used, num_lookups, e4_arg_cols_start) =
        width_3_lookups_new::<
            NON_DELEGATED_MAX_WIDTH_3_LOOKUPS,
            NON_DELEGATED_MAX_WIDTH_3_LOOKUP_VALS,
            NON_DELEGATED_MAX_WIDTH_3_LOOKUP_COEFFS,
            NON_DELEGATED_MAX_WIDTH_3_LOOKUP_COLS,
        >(
            circuit, lookup_challenges, lookup_gamma, alphas, alphas_offset,
            helpers, decompression_factor_inv, constants_times_challenges, translate_e4_offset,
        );
    NonDelegatedWidth3LookupsLayout {
        coeffs, col_idxs, num_terms_per_expression, table_id_is_col, e4_arg_cols,
        helpers_offset, num_helpers_used, num_lookups, e4_arg_cols_start,
    }
}

fn build_multiplicities_layout(
    src_cols_start: usize,
    dst_cols_start: usize,
    setup_cols_start: usize,
    num_dst_cols: usize,
    entry_width: usize,
    lookup_challenges: &LookupChallenges,
    alphas: &[E4],
    alpha_offset: &mut usize,
    helpers: &mut Vec<E4, impl Allocator>,
    decompression_factor_inv: E2,
    translate_e4_offset: &impl Fn(usize) -> usize,
) -> MultiplicitiesLayout {
    for _ in 0..num_dst_cols {
        let alpha = alphas[*alpha_offset];
        *alpha_offset += 1;
        helpers.push(
            *alpha
                .clone()
                .mul_assign(&lookup_challenges.gamma)
                .mul_assign_by_base(&decompression_factor_inv),
        );
        for j in 0..entry_width - 1 {
            helpers.push(
                *alpha
                    .clone()
                    .mul_assign(&lookup_challenges.linearization_challenges[j]),
            );
        }
    }
    MultiplicitiesLayout {
        src_cols_start: src_cols_start as u32,
        dst_cols_start: translate_e4_offset(dst_cols_start) as u32,
        setup_cols_start: setup_cols_start as u32,
        num_dst_cols: num_dst_cols as u32,
    }
}

fn build_state_linkage_constraints(circuit: &CompiledCircuitArtifact<BF>) -> StateLinkageConstraints {
    let num_constraints = circuit.state_linkage_constraints.len();
    assert!(num_constraints <= 2);
    let mut srcs = [0u32; 2];
    let mut dsts = [0u32; 2];
    for (i, (src, dst)) in circuit.state_linkage_constraints.iter().enumerate() {
        let ColumnAddress::WitnessSubtree(col) = *src else {
            panic!()
        };
        srcs[i] = col as u32;
        let ColumnAddress::WitnessSubtree(col) = *dst else {
            panic!()
        };
        dsts[i] = col as u32;
    }
    StateLinkageConstraints {
        srcs,
        dsts,
        num_constraints: num_constraints as u32,
    }
}

fn build_boundary_constraints(
    circuit: &CompiledCircuitArtifact<BF>,
    external_values: &ExternalValues,
    public_inputs: &[BF],
    process_shuffle_ram_init: bool,
    lazy_init_address_start: usize,
    lazy_teardown_value_start: usize,
    lazy_teardown_timestamp_start: usize,
    alphas_first_row: &[E4],
    alphas_one_before_last_row: &[E4],
    helpers: &mut Vec<E4, impl Allocator>,
    beta_powers: &[E4],
    decompression_factor: E2,
    constants_times_challenges: &mut ConstantsTimesChallenges,
) -> BoundaryConstraints {
    let mut first_row_cols = [0u32; 8];
    let mut one_before_last_row_cols = [0u32; 8];
    constants_times_challenges.first_row = E4::ZERO;
    constants_times_challenges.one_before_last_row = E4::ZERO;
    let mut num_first_row = 0;
    let mut num_one_before_last_row = 0;
    let mut helpers_first_row = Vec::with_capacity(8);
    let mut helpers_one_before_last_row = Vec::with_capacity(8);
    if process_shuffle_ram_init {
        let beta_power = beta_powers[3];
        let mut stash_limb_pair_first = |start_col: usize, vals: &[BF]| {
            for i in 0..=1 {
                let mut alpha = alphas_first_row[num_first_row];
                alpha.mul_assign(&beta_power);
                helpers_first_row
                    .push(*alpha.clone().mul_assign_by_base(&decompression_factor));
                first_row_cols[num_first_row] = (start_col + i) as u32;
                constants_times_challenges
                    .first_row
                    .sub_assign(alpha.mul_assign_by_base(&vals[i]));
                num_first_row += 1;
            }
        };
        stash_limb_pair_first(
            lazy_init_address_start,
            &external_values.aux_boundary_values.lazy_init_first_row[..],
        );
        stash_limb_pair_first(
            lazy_teardown_value_start,
            &external_values.aux_boundary_values.teardown_value_first_row[..],
        );
        stash_limb_pair_first(
            lazy_teardown_timestamp_start,
            &external_values
                .aux_boundary_values
                .teardown_timestamp_first_row[..],
        );
        let beta_power = beta_powers[2];
        let mut stash_limb_pair_obl = |start_col: usize, vals: &[BF]| {
            for i in 0..=1 {
                let mut alpha = alphas_one_before_last_row[num_one_before_last_row];
                alpha.mul_assign(&beta_power);
                helpers_one_before_last_row
                    .push(*alpha.clone().mul_assign_by_base(&decompression_factor));
                one_before_last_row_cols[num_one_before_last_row] = (start_col + i) as u32;
                constants_times_challenges
                    .one_before_last_row
                    .sub_assign(alpha.mul_assign_by_base(&vals[i]));
                num_one_before_last_row += 1;
            }
        };
        stash_limb_pair_obl(
            lazy_init_address_start,
            &external_values
                .aux_boundary_values
                .lazy_init_one_before_last_row[..],
        );
        stash_limb_pair_obl(
            lazy_teardown_value_start,
            &external_values
                .aux_boundary_values
                .teardown_value_one_before_last_row[..],
        );
        stash_limb_pair_obl(
            lazy_teardown_timestamp_start,
            &external_values
                .aux_boundary_values
                .teardown_timestamp_one_before_last_row[..],
        );
    }
    for ((location, column_address), val) in
        circuit.public_inputs.iter().zip(public_inputs.iter())
    {
        match location {
            BoundaryConstraintLocation::FirstRow => {
                first_row_cols[num_first_row] = match column_address {
                    ColumnAddress::WitnessSubtree(col) => *col as u32,
                    _ => panic!("public inputs should be in witness"),
                };
                let beta_power = beta_powers[3];
                let mut alpha = alphas_first_row[num_first_row];
                alpha.mul_assign(&beta_power);
                helpers_first_row
                    .push(*alpha.clone().mul_assign_by_base(&decompression_factor));
                constants_times_challenges
                    .first_row
                    .sub_assign(alpha.clone().mul_assign_by_base(val));
                num_first_row += 1;
            }
            BoundaryConstraintLocation::OneBeforeLastRow => {
                one_before_last_row_cols[num_one_before_last_row] = match column_address {
                    ColumnAddress::WitnessSubtree(col) => *col as u32,
                    _ => panic!("public inputs should be in witness"),
                };
                let beta_power = beta_powers[2];
                let mut alpha = alphas_one_before_last_row[num_one_before_last_row];
                alpha.mul_assign(&beta_power);
                helpers_one_before_last_row
                    .push(*alpha.clone().mul_assign_by_base(&decompression_factor));
                constants_times_challenges
                    .one_before_last_row
                    .sub_assign(alpha.mul_assign_by_base(val));
                num_one_before_last_row += 1;
            }
            BoundaryConstraintLocation::LastRow => {
                panic!("public inputs on the last row are not supported");
            }
        }
    }
    // account for memory accumulator, which requires a first row constraint
    let mut alpha = alphas_first_row[num_first_row];
    alpha.mul_assign(&beta_powers[3]);
    let grand_product_helper = *alpha.clone().mul_assign_by_base(&decompression_factor);
    constants_times_challenges.first_row.sub_assign(&alpha);
    // pushing grand product helper first is a bit more convenient for the kernel
    helpers.push(grand_product_helper);
    helpers.extend_from_slice(&helpers_first_row);
    helpers.extend_from_slice(&helpers_one_before_last_row);
    assert!(num_first_row <= 8);
    assert!(num_one_before_last_row <= 8);
    BoundaryConstraints {
        first_row_cols,
        one_before_last_row_cols,
        num_first_row: num_first_row as u32,
        num_one_before_last_row: num_one_before_last_row as u32,
    }
}

pub struct Metadata {
    pub(super) alpha_powers_layout: AlphaPowersLayout,
    pub(super) flat_generic_constraints_metadata: FlattenedGenericConstraintsMetadata,
    pub(super) delegated_width_3_lookups_layout: DelegatedWidth3LookupsLayout,
    pub(super) non_delegated_width_3_lookups_layout: NonDelegatedWidth3LookupsLayout,
    pub(super) range_check_16_layout: RangeCheck16ArgsLayout,
    pub(super) expressions_layout: FlattenedLookupExpressionsLayout,
    pub(super) expressions_for_shuffle_ram_layout: FlattenedLookupExpressionsForShuffleRamLayout,
    pub(super) generic_lookup_multiplicities_layout: MultiplicitiesLayout,
    pub(super) state_linkage_constraints: StateLinkageConstraints,
    pub(super) boundary_constraints: BoundaryConstraints,
    pub(super) memory_args_start: usize,
    pub(super) memory_grand_product_col: usize,
    pub(super) lazy_init_teardown_layout: LazyInitTeardownLayout,
    pub(super) shuffle_ram_accesses: ShuffleRamAccesses,
    pub(super) batched_ram_accesses: BatchedRamAccesses,
    pub(super) range_check_16_multiplicities_layout: MultiplicitiesLayout,
    pub(super) timestamp_range_check_multiplicities_layout: MultiplicitiesLayout,
    pub(super) delegation_aux_poly_col: usize,
    pub(super) num_generic_constraints: usize,
    pub(super) delegation_challenges: DelegationChallenges,
    pub(super) delegation_processing_metadata: DelegationProcessingMetadata,
    pub(super) delegation_request_metadata: DelegationRequestMetadata,
    pub(super) register_and_indirect_accesses: RegisterAndIndirectAccesses,
}

impl Metadata {
    pub(crate) fn new(
        h_alpha_powers: &[E4],
        h_beta_powers: &[E4],
        tau: E2,
        omega: E2,
        omega_inv: E2,
        lookup_challenges: &LookupChallenges,
        cached_data: &ProverCachedData,
        circuit: &CompiledCircuitArtifact<BF>,
        external_values: &ExternalValues,
        public_inputs: &[BF],
        grand_product_accumulator: E4,
        sum_over_delegation_poly: E4,
        log_n: u32,
        helpers: &mut Vec<E4, impl Allocator>,
        constants_times_challenges: &mut ConstantsTimesChallenges,
    ) -> Metadata {
        let n = 1 << log_n;
        let num_stage_2_bf_cols = circuit.stage_2_layout.num_base_field_polys();
        let num_stage_2_e4_cols = circuit.stage_2_layout.num_ext4_field_polys();
        let e4_cols_offset = circuit.stage_2_layout.ext4_polys_offset;
        assert_eq!(e4_cols_offset % 4, 0);
        assert!(num_stage_2_bf_cols <= e4_cols_offset);
        assert!(e4_cols_offset - num_stage_2_bf_cols < 4);
        let alpha_powers_layout =
            AlphaPowersLayout::new(&circuit, cached_data.num_stage_3_quotient_terms);
        let cached_data = cached_data.clone();
        let ProverCachedData {
            trace_len,
            memory_timestamp_high_from_circuit_idx,
            delegation_type: _,
            memory_argument_challenges,
            execute_delegation_argument: _,
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
            offset_for_grand_product_accumulation_poly,
            range_check_16_multiplicities_src,
            range_check_16_multiplicities_dst,
            range_check_16_setup_column,
            timestamp_range_check_multiplicities_src,
            timestamp_range_check_multiplicities_dst,
            timestamp_range_check_setup_column,
            generic_lookup_multiplicities_src_start,
            generic_lookup_multiplicities_dst_start,
            generic_lookup_setup_columns_start,
            ref range_check_16_width_1_lookups_access,
            ref range_check_16_width_1_lookups_access_via_expressions,
            ref timestamp_range_check_width_1_lookups_access_via_expressions,
            ref timestamp_range_check_width_1_lookups_access_via_expressions_for_shuffle_ram,
            memory_accumulator_dst_start,
            ..
        } = cached_data;

        assert_eq!(trace_len, n);

        let AlphaPowersLayout {
            num_quotient_terms_every_row_except_last,
            num_quotient_terms_every_row_except_last_two,
            num_quotient_terms_first_row,
            num_quotient_terms_one_before_last_row,
            num_quotient_terms_last_row,
            num_quotient_terms_last_row_and_at_zero,
            precomputation_size,
        } = alpha_powers_layout;
        assert_eq!(h_alpha_powers.len(), precomputation_size);
        let h_alphas_for_every_row_except_last =
            &h_alpha_powers[(precomputation_size - num_quotient_terms_every_row_except_last)..];
        let h_alphas_for_every_row_except_last_two =
            &h_alpha_powers[(precomputation_size - num_quotient_terms_every_row_except_last_two)..];
        let h_alphas_for_first_row =
            &h_alpha_powers[(precomputation_size - num_quotient_terms_first_row)..];
        let h_alphas_for_one_before_last_row =
            &h_alpha_powers[(precomputation_size - num_quotient_terms_one_before_last_row)..];
        let h_alphas_for_last_row =
            &h_alpha_powers[(precomputation_size - num_quotient_terms_last_row)..];
        let h_alphas_for_last_row_and_at_zero =
            &h_alpha_powers[(precomputation_size - num_quotient_terms_last_row_and_at_zero)..];
        // Generic constraints
        let num_generic_constraints =
            circuit.degree_2_constraints.len() + circuit.degree_1_constraints.len();
        let (h_alphas_for_generic_constraints, h_alphas_for_hardcoded_every_row_except_last) =
            h_alphas_for_every_row_except_last.split_at(num_generic_constraints);
        constants_times_challenges.sum = E4::ZERO;
        let flat_generic_constraints_metadata = FlattenedGenericConstraintsMetadata::new(
            circuit,
            h_alphas_for_generic_constraints,
            tau,
            omega_inv,
            n,
            constants_times_challenges,
        );
        // Hardcoded constraints
        let translate_e4_offset = |raw_col: usize| -> usize {
            assert_eq!(raw_col % 4, 0);
            assert!(raw_col >= e4_cols_offset);
            (raw_col - e4_cols_offset) / 4
        };
        let num_range_check_16_multiplicities_cols = circuit
            .witness_layout
            .multiplicities_columns_for_range_check_16
            .num_elements();
        assert_eq!(num_range_check_16_multiplicities_cols, 1);
        let num_timestamp_range_check_multiplicities_cols = circuit
            .witness_layout
            .multiplicities_columns_for_timestamp_range_check
            .num_elements();
        assert!(
            (num_timestamp_range_check_multiplicities_cols == 0)
                || (num_timestamp_range_check_multiplicities_cols == 1)
        );
        let num_generic_multiplicities_cols = circuit
            .setup_layout
            .generic_lookup_setup_columns
            .num_elements();
        assert_eq!(circuit.setup_layout.generic_lookup_setup_columns.width(), 4);

        let (delegation_aux_poly_col, delegation_challenges) =
            if handle_delegation_requests || process_delegations {
                (
                    translate_e4_offset(delegation_processing_aux_poly.start()),
                    DelegationChallenges::new(&delegation_challenges),
                )
            } else {
                (0, DelegationChallenges::default())
            };
        let (delegation_request_metadata, delegation_processing_metadata) =
            get_delegation_metadata(&cached_data, circuit);
        let memory_challenges = MemoryChallenges::new(&memory_argument_challenges);
        let num_memory_args = circuit
            .stage_2_layout
            .intermediate_polys_for_memory_argument
            .num_elements();
        let batched_ram_accesses = if process_batch_ram_access {
            assert!(!process_shuffle_ram_init);
            assert_eq!(circuit.memory_layout.shuffle_ram_access_sets.len(), 0);
            assert!(!process_registers_and_indirect_access);
            let batched_ram_accesses = &circuit.memory_layout.batched_ram_accesses;
            assert!(batched_ram_accesses.len() > 0);
            let write_timestamp_col = delegation_processor_layout.write_timestamp.start();
            let abi_mem_offset_high_col = delegation_processor_layout.abi_mem_offset_high.start();
            assert_eq!(
                num_memory_args,
                batched_ram_accesses.len() + 1,
            );
            assert_eq!(num_memory_args, num_set_polys_for_memory_shuffle);
            BatchedRamAccesses::new(
                &memory_challenges,
                batched_ram_accesses,
                write_timestamp_col,
                abi_mem_offset_high_col,
            )
        } else {
            BatchedRamAccesses::default()
        };
        let register_and_indirect_accesses = if process_registers_and_indirect_access {
            assert!(!process_shuffle_ram_init);
            assert_eq!(circuit.memory_layout.shuffle_ram_access_sets.len(), 0);
            assert!(!process_batch_ram_access);
            let register_and_indirect_accesses =
                &circuit.memory_layout.register_and_indirect_accesses;
            assert!(register_and_indirect_accesses.len() > 0);
            let write_timestamp_col = delegation_processor_layout.write_timestamp.start();
            RegisterAndIndirectAccesses::new(
                &memory_challenges,
                register_and_indirect_accesses,
                write_timestamp_col,
            )
        } else {
            RegisterAndIndirectAccesses::default()
        };
        let range_check_16_layout = RangeCheck16ArgsLayout::new(
            circuit,
            &range_check_16_width_1_lookups_access,
            &range_check_16_width_1_lookups_access_via_expressions,
            &translate_e4_offset,
        );
        let expressions_layout = if range_check_16_width_1_lookups_access_via_expressions.len() > 0
            || timestamp_range_check_width_1_lookups_access_via_expressions.len() > 0
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
            if timestamp_range_check_width_1_lookups_access_via_expressions_for_shuffle_ram.len()
                > 0
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
            assert!(circuit.lazy_init_address_aux_vars.is_some());
            LazyInitTeardownLayout::new(
                circuit,
                &lazy_init_address_range_check_16,
                &shuffle_ram_inits_and_teardowns,
                &translate_e4_offset,
            )
        } else {
            assert!(circuit.lazy_init_address_aux_vars.is_none());
            LazyInitTeardownLayout::default()
        };
        // Host work to precompute constants_times_challenges_sum and helpers
        assert_eq!(helpers.len(), 0);
        assert_eq!(helpers.capacity(), MAX_HELPER_VALUES);
        let decompression_factor = flat_generic_constraints_metadata.decompression_factor;
        let decompression_factor_inv = decompression_factor.inverse().expect("must exist");
        let two = BF::from_u64_unchecked(2);
        let lookup_linearization_challenges = &lookup_challenges.linearization_challenges;
        let lookup_gamma = lookup_challenges.gamma;
        let lookup_gamma_squared = *lookup_gamma.clone().square();
        let lookup_two_gamma = *lookup_gamma.clone().mul_assign_by_base(&two);
        let mut alpha_offset = 0;
        if process_delegations {
            alpha_offset += 4;
        }
        if process_batch_ram_access {
            for i in 0..batched_ram_accesses.num_accesses as usize {
                if batched_ram_accesses.accesses[i].is_write {
                    alpha_offset += 6;
                } else {
                    alpha_offset += 4;
                }
            }
        }
        if process_registers_and_indirect_access {
            let mut flat_indirect_idx = 0;
            for i in 0..register_and_indirect_accesses.num_register_accesses as usize {
                let register_access = &register_and_indirect_accesses.register_accesses[i];
                if register_access.is_write {
                    alpha_offset += 6;
                } else {
                    alpha_offset += 4;
                }
                for j in 0..register_and_indirect_accesses.indirect_accesses_per_register_access[i]
                {
                    let indirect_access =
                        &register_and_indirect_accesses.indirect_accesses[flat_indirect_idx];
                    if indirect_access.is_write {
                        alpha_offset += 6;
                    } else {
                        alpha_offset += 4;
                    }
                    if j > 0 && indirect_access.address_derivation_carry_bit_num_elements > 0 {
                        alpha_offset += 1;
                    }
                    flat_indirect_idx += 1;
                }
            }
        }
        let mut bound = range_check_16_width_1_lookups_access.len();
        if expressions_layout.range_check_16_constant_terms_are_zero {
            bound += range_check_16_width_1_lookups_access_via_expressions.len();
        }
        for _ in 0..bound {
            alpha_offset += 1;
            let alpha = h_alphas_for_hardcoded_every_row_except_last[alpha_offset];
            helpers.push(*alpha.clone().mul_assign(&lookup_gamma));
            helpers.push(
                *alpha
                    .clone()
                    .mul_assign(&lookup_gamma_squared)
                    .mul_assign_by_base(&decompression_factor_inv),
            );
            constants_times_challenges
                .sum
                .sub_assign(alpha.clone().mul_assign(&lookup_two_gamma));
            alpha_offset += 1;
        }
        let stash_helpers_for_expressions_with_constant_terms =
            |num_expression_pairs: usize,
             constant_terms: &[BF],
             alpha_offset: &mut usize,
             helpers: &mut Vec<E4, _>,
             constants_times_challenges: &mut ConstantsTimesChallenges| {
                for i in 0..num_expression_pairs {
                    let mut alpha = h_alphas_for_hardcoded_every_row_except_last[*alpha_offset];
                    let a_constant_term = constant_terms[2 * i];
                    let b_constant_term = constant_terms[2 * i + 1];
                    let constants_prod = *a_constant_term.clone().mul_assign(&b_constant_term);
                    constants_times_challenges
                        .sum
                        .add_assign(alpha.mul_assign_by_base(&constants_prod));
                    *alpha_offset += 1;
                    let alpha = h_alphas_for_hardcoded_every_row_except_last[*alpha_offset];
                    helpers.push(*alpha.clone().mul_assign(&lookup_gamma));
                    let constants_sum = *a_constant_term.clone().add_assign(&b_constant_term);
                    let mut gamma_corrections =
                        *lookup_gamma.clone().mul_assign_by_base(&constants_sum);
                    gamma_corrections.add_assign(&lookup_gamma_squared);
                    helpers.push(
                        *alpha
                            .clone()
                            .mul_assign(&gamma_corrections)
                            .mul_assign_by_base(&decompression_factor_inv),
                    );
                    constants_times_challenges
                        .sum
                        .sub_assign(alpha.clone().mul_assign_by_base(&constants_sum));
                    constants_times_challenges
                        .sum
                        .sub_assign(alpha.clone().mul_assign(&lookup_two_gamma));
                    *alpha_offset += 1;
                }
            };
        if !expressions_layout.range_check_16_constant_terms_are_zero {
            let num_pairs = expressions_layout.num_range_check_16_expression_pairs as usize;
            stash_helpers_for_expressions_with_constant_terms(
                num_pairs,
                &expressions_layout.constant_terms[0..2 * num_pairs],
                &mut alpha_offset,
                helpers,
                constants_times_challenges,
            );
        }
        if process_shuffle_ram_init {
            alpha_offset += 1;
            let alpha = h_alphas_for_hardcoded_every_row_except_last[alpha_offset];
            helpers.push(*alpha.clone().mul_assign(&lookup_gamma));
            helpers.push(
                *alpha
                    .clone()
                    .mul_assign(&lookup_gamma_squared)
                    .mul_assign_by_base(&decompression_factor_inv),
            );
            constants_times_challenges
                .sum
                .sub_assign(alpha.clone().mul_assign(&lookup_two_gamma));
            alpha_offset += 1;
        }
        if expressions_layout.timestamp_constant_terms_are_zero {
            for _ in 0..expressions_layout.num_timestamp_expression_pairs as usize {
                alpha_offset += 1;
                let alpha = h_alphas_for_hardcoded_every_row_except_last[alpha_offset];
                helpers.push(*alpha.clone().mul_assign(&lookup_gamma));
                helpers.push(
                    *alpha
                        .clone()
                        .mul_assign(&lookup_gamma_squared)
                        .mul_assign_by_base(&decompression_factor_inv),
                );
                constants_times_challenges
                    .sum
                    .sub_assign(alpha.clone().mul_assign(&lookup_two_gamma));
                alpha_offset += 1;
            }
        } else {
            let num_pairs = expressions_layout.num_timestamp_expression_pairs as usize;
            let start = 2 * expressions_layout.num_range_check_16_expression_pairs as usize;
            let end = start + 2 * num_pairs;
            stash_helpers_for_expressions_with_constant_terms(
                num_pairs,
                &expressions_layout.constant_terms[start..end],
                &mut alpha_offset,
                helpers,
                constants_times_challenges,
            );
        }
        let num_pairs = expressions_for_shuffle_ram_layout.num_expression_pairs as usize;
        let constant_terms_with_timestamp_high_circuit_idx_adjustment: Vec<BF> =
            expressions_for_shuffle_ram_layout
                .constant_terms
                .iter()
                .enumerate()
                .map(|(i, val)| {
                    if i % 2 == 0 {
                        *val
                    } else {
                        *val.clone()
                            .sub_assign(&memory_timestamp_high_from_circuit_idx)
                    }
                })
                .collect();
        stash_helpers_for_expressions_with_constant_terms(
            num_pairs,
            &constant_terms_with_timestamp_high_circuit_idx_adjustment[0..2 * num_pairs],
            &mut alpha_offset,
            helpers,
            constants_times_challenges,
        );
        let (delegated_width_3_lookups_layout, non_delegated_width_3_lookups_layout) =
            if process_delegations {
                let delegated_layout = build_delegated_width_3_lookups(
                    circuit,
                    lookup_linearization_challenges,
                    lookup_gamma,
                    h_alphas_for_hardcoded_every_row_except_last,
                    &mut alpha_offset,
                    helpers,
                    decompression_factor_inv,
                    constants_times_challenges,
                    &translate_e4_offset,
                );
                let non_delegated_placeholder = NonDelegatedWidth3LookupsLayout {
                    coeffs: [0u32; NON_DELEGATED_MAX_WIDTH_3_LOOKUP_COEFFS],
                    col_idxs: [0u16; NON_DELEGATED_MAX_WIDTH_3_LOOKUP_COLS],
                    num_terms_per_expression: [0u8; NON_DELEGATED_MAX_WIDTH_3_LOOKUP_VALS],
                    table_id_is_col: [false; NON_DELEGATED_MAX_WIDTH_3_LOOKUPS],
                    e4_arg_cols: [0u16; NON_DELEGATED_MAX_WIDTH_3_LOOKUPS],
                    helpers_offset: 0,
                    num_helpers_used: delegated_layout.num_helpers_used,
                    num_lookups: delegated_layout.num_lookups,
                    e4_arg_cols_start: delegated_layout.e4_arg_cols_start,
                };
                (delegated_layout, non_delegated_placeholder)
            } else {
                (
                    DelegatedWidth3LookupsLayout::default(),
                    build_non_delegated_width_3_lookups(
                        circuit,
                        lookup_linearization_challenges,
                        lookup_gamma,
                        h_alphas_for_hardcoded_every_row_except_last,
                        &mut alpha_offset,
                        helpers,
                        decompression_factor_inv,
                        constants_times_challenges,
                        &translate_e4_offset,
                    ),
                )
            };
        let range_check_16_multiplicities_layout = build_multiplicities_layout(
            range_check_16_multiplicities_src,
            range_check_16_multiplicities_dst,
            range_check_16_setup_column,
            num_range_check_16_multiplicities_cols,
            1,
            lookup_challenges,
            h_alphas_for_hardcoded_every_row_except_last,
            &mut alpha_offset,
            helpers,
            decompression_factor_inv,
            &translate_e4_offset,
        );
        let timestamp_range_check_multiplicities_layout = build_multiplicities_layout(
            timestamp_range_check_multiplicities_src,
            timestamp_range_check_multiplicities_dst,
            timestamp_range_check_setup_column,
            num_timestamp_range_check_multiplicities_cols,
            1,
            lookup_challenges,
            h_alphas_for_hardcoded_every_row_except_last,
            &mut alpha_offset,
            helpers,
            decompression_factor_inv,
            &translate_e4_offset,
        );
        let generic_lookup_multiplicities_layout = build_multiplicities_layout(
            generic_lookup_multiplicities_src_start,
            generic_lookup_multiplicities_dst_start,
            generic_lookup_setup_columns_start,
            num_generic_multiplicities_cols,
            NUM_LOOKUP_ARGUMENT_KEY_PARTS,
            lookup_challenges,
            h_alphas_for_hardcoded_every_row_except_last,
            &mut alpha_offset,
            helpers,
            decompression_factor_inv,
            &translate_e4_offset,
        );
        if handle_delegation_requests {
            let alpha = h_alphas_for_hardcoded_every_row_except_last[alpha_offset];
            alpha_offset += 1;
            let mut timestamp_low_constant = delegation_challenges.linearization_challenges
                [DELEGATION_ARGUMENT_CHALLENGED_IDX_FOR_TIMESTAMP_LOW];
            timestamp_low_constant
                .mul_assign_by_base(&delegation_request_metadata.in_cycle_write_idx);
            let mut timestamp_high_constant = delegation_challenges.linearization_challenges
                [DELEGATION_ARGUMENT_CHALLENGED_IDX_FOR_TIMESTAMP_HIGH];
            timestamp_high_constant.mul_assign_by_base(&memory_timestamp_high_from_circuit_idx);
            helpers.push(
                *delegation_challenges
                    .gamma
                    .clone()
                    .add_assign(&timestamp_low_constant)
                    .add_assign(&timestamp_high_constant)
                    .mul_assign(&alpha)
                    .mul_assign_by_base(&decompression_factor_inv),
            );
            for challenge in delegation_challenges.linearization_challenges.iter() {
                helpers.push(*alpha.clone().mul_assign(challenge));
            }
        }
        if process_delegations {
            let alpha = h_alphas_for_hardcoded_every_row_except_last[alpha_offset];
            alpha_offset += 1;
            helpers.push(
                *delegation_challenges
                    .gamma
                    .clone()
                    .add_assign_base(&delegation_processing_metadata.delegation_type)
                    .mul_assign(&alpha)
                    .mul_assign_by_base(&decompression_factor_inv),
            );
            for challenge in delegation_challenges.linearization_challenges.iter() {
                helpers.push(*alpha.clone().mul_assign(challenge));
            }
        }
        let memory_args_start = translate_e4_offset(memory_accumulator_dst_start);
        if process_shuffle_ram_init {
            alpha_offset += 6;
            let alpha = h_alphas_for_hardcoded_every_row_except_last[alpha_offset];
            alpha_offset += 1;
            let alpha_times_gamma = *alpha.clone().mul_assign(&memory_challenges.gamma);
            constants_times_challenges
                .sum
                .sub_assign(&alpha_times_gamma);
            let mc = &memory_challenges;
            helpers.push(*alpha.clone().mul_assign(&mc.address_low_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.address_high_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.value_low_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.value_high_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.timestamp_low_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.timestamp_high_challenge));
            helpers.push(
                *alpha_times_gamma
                    .clone()
                    .mul_assign_by_base(&decompression_factor_inv),
            );
        }

        let shuffle_ram_accesses = if process_shuffle_ram_init {
            let shuffle_ram_access_sets = &circuit.memory_layout.shuffle_ram_access_sets;
            let write_timestamp_in_setup_start =
                circuit.setup_layout.timestamp_setup_columns.start();
            ShuffleRamAccesses::new(shuffle_ram_access_sets, write_timestamp_in_setup_start)
        } else {
            ShuffleRamAccesses::default()
        };
        for i in 0..shuffle_ram_accesses.num_accesses as usize {
            let access = &shuffle_ram_accesses.accesses[i];
            let alpha = h_alphas_for_hardcoded_every_row_except_last[alpha_offset];
            alpha_offset += 1;
            let mc = &memory_challenges;
            let mut numerator_constant = mc.gamma;
            if access.is_register_only {
                numerator_constant.add_assign_base(&BF::ONE);
            }
            let mut denom_constant = numerator_constant;
            let write_timestamp_low_constant = *mc
                .timestamp_low_challenge
                .clone()
                .mul_assign_by_base(&BF::from_u64_unchecked(i as u64));
            let write_timestamp_high_constant = *mc
                .timestamp_high_challenge
                .clone()
                .mul_assign_by_base(&memory_timestamp_high_from_circuit_idx);
            numerator_constant
                .add_assign(&write_timestamp_low_constant)
                .add_assign(&write_timestamp_high_constant);
            helpers.push(*alpha.clone().mul_assign(&mc.address_low_challenge));
            if !access.is_register_only {
                helpers.push(*alpha.clone().mul_assign(&mc.address_high_challenge));
            }
            helpers.push(*alpha.clone().mul_assign(&mc.value_low_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.value_high_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.timestamp_low_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.timestamp_high_challenge));
            helpers.push(
                *denom_constant
                    .mul_assign(&alpha)
                    .mul_assign_by_base(&decompression_factor_inv),
            );
            helpers.push(
                *numerator_constant
                    .mul_assign(&alpha)
                    .mul_assign_by_base(&decompression_factor_inv),
            );
        }
        for i in 0..batched_ram_accesses.num_accesses as usize {
            let access = &batched_ram_accesses.accesses[i];
            let alpha = h_alphas_for_hardcoded_every_row_except_last[alpha_offset];
            alpha_offset += 1;
            let mc = &memory_challenges;
            let mut constant = access.gamma_plus_address_low_contribution;
            constant.mul_assign(&alpha);
            if i == 0 {
                constants_times_challenges.sum.sub_assign(&constant);
            }
            helpers.push(*alpha.clone().mul_assign(&mc.address_high_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.value_low_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.value_high_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.timestamp_low_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.timestamp_high_challenge));
            helpers.push(*constant.mul_assign_by_base(&decompression_factor_inv));
        }
        let mut flat_indirect_idx = 0;
        for i in 0..register_and_indirect_accesses.num_register_accesses as usize {
            let register_access = &register_and_indirect_accesses.register_accesses[i];
            let alpha = h_alphas_for_hardcoded_every_row_except_last[alpha_offset];
            alpha_offset += 1;
            let mc = &memory_challenges;
            let mut constant = register_access.gamma_plus_one_plus_address_low_contribution;
            constant.mul_assign(&alpha);
            if i == 0 {
                constants_times_challenges.sum.sub_assign(&constant);
            }
            helpers.push(*alpha.clone().mul_assign(&mc.value_low_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.value_high_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.timestamp_low_challenge));
            helpers.push(*alpha.clone().mul_assign(&mc.timestamp_high_challenge));
            helpers.push(*constant.mul_assign_by_base(&decompression_factor_inv));
            for j in 0..register_and_indirect_accesses.indirect_accesses_per_register_access[i] {
                let indirect_access =
                    &register_and_indirect_accesses.indirect_accesses[flat_indirect_idx];
                flat_indirect_idx += 1;
                let alpha = h_alphas_for_hardcoded_every_row_except_last[alpha_offset];
                alpha_offset += 1;
                assert_eq!(j == 0, indirect_access.offset == 0);
                let offset = BF::from_u64_unchecked(indirect_access.offset as u64);
                let mut constant = *mc
                    .address_low_challenge
                    .clone()
                    .mul_assign_by_base(&offset)
                    .add_assign(&mc.gamma)
                    .mul_assign(&alpha);
                helpers.push(*alpha.clone().mul_assign(&mc.address_low_challenge));
                helpers.push(*alpha.clone().mul_assign(&mc.address_high_challenge));
                helpers.push(*alpha.clone().mul_assign(&mc.value_low_challenge));
                helpers.push(*alpha.clone().mul_assign(&mc.value_high_challenge));
                helpers.push(*alpha.clone().mul_assign(&mc.timestamp_low_challenge));
                helpers.push(*alpha.clone().mul_assign(&mc.timestamp_high_challenge));
                helpers.push(*constant.mul_assign_by_base(&decompression_factor_inv));
            }
        }
        assert_eq!(
            offset_for_grand_product_accumulation_poly,
            num_memory_args - 1
        );
        let memory_grand_product_col = memory_args_start + num_memory_args - 1;
        alpha_offset += 1;
        assert_eq!(
            alpha_offset,
            h_alphas_for_hardcoded_every_row_except_last.len()
        );
        // Prepare args and helpers for constraints on all rows except the last two
        alpha_offset = 0;
        let state_linkage_constraints = build_state_linkage_constraints(circuit);
        alpha_offset += state_linkage_constraints.num_constraints as usize;
        if process_shuffle_ram_init {
            alpha_offset += 2;
        }
        assert_eq!(alpha_offset, h_alphas_for_every_row_except_last_two.len());
        // Args and helpers for boundary constraints
        let boundary_constraints = build_boundary_constraints(
            circuit,
            external_values,
            public_inputs,
            process_shuffle_ram_init,
            lazy_init_teardown_layout.init_address_start as usize,
            lazy_init_teardown_layout.teardown_value_start as usize,
            lazy_init_teardown_layout.teardown_timestamp_start as usize,
            h_alphas_for_first_row,
            h_alphas_for_one_before_last_row,
            helpers,
            h_beta_powers,
            decompression_factor,
            constants_times_challenges,
        );
        assert_eq!(
            boundary_constraints.num_first_row as usize + 1,
            h_alphas_for_first_row.len()
        );
        assert_eq!(
            boundary_constraints.num_one_before_last_row as usize,
            h_alphas_for_one_before_last_row.len()
        );
        // Just one constraint at last row (grand product accumulator)
        let mut alpha = h_alphas_for_last_row[0];
        alpha.mul_assign(&h_beta_powers[1]);
        helpers.push(*alpha.clone().mul_assign_by_base(&decompression_factor));
        helpers.push(*alpha.negate().mul_assign(&grand_product_accumulator));
        assert_eq!(1, h_alphas_for_last_row.len());
        // Constraints at last row and zero
        let args_metadata = &circuit.stage_2_layout.intermediate_polys_for_range_check_16;
        let num_range_check_16_e4_args = args_metadata.ext_4_field_oracles.num_elements();
        assert_eq!(num_range_check_16_e4_args, args_metadata.num_pairs);
        assert_eq!(
            num_range_check_16_e4_args,
            (range_check_16_layout.num_dst_cols
                + expressions_layout.num_range_check_16_expression_pairs) as usize,
        );
        assert_eq!(
            translate_e4_offset(args_metadata.ext_4_field_oracles.start()),
            range_check_16_layout.e4_args_start as usize
        );
        let mut alpha_offset = 0;
        let mut alpha = h_alphas_for_last_row_and_at_zero[alpha_offset];
        alpha_offset += 1;
        helpers.push(*alpha.negate().mul_assign_by_base(&decompression_factor));
        // timestamp range check e4 arg sums
        if timestamp_range_check_multiplicities_layout.num_dst_cols > 0 {
            let args_metadata = &circuit
                .stage_2_layout
                .intermediate_polys_for_timestamp_range_checks;
            let num_timestamp_range_check_e4_args =
                args_metadata.ext_4_field_oracles.num_elements();
            let num_non_shuffle_ram_args =
                expressions_layout.num_timestamp_expression_pairs as usize;
            let num_shuffle_ram_args =
                expressions_for_shuffle_ram_layout.num_expression_pairs as usize;
            assert_eq!(num_timestamp_range_check_e4_args, args_metadata.num_pairs);
            assert_eq!(
                num_timestamp_range_check_e4_args,
                num_non_shuffle_ram_args + num_shuffle_ram_args,
            );
            let offset = expressions_layout.num_range_check_16_expression_pairs as usize;
            for (i, dst) in args_metadata.ext_4_field_oracles.iter().enumerate() {
                if i < num_non_shuffle_ram_args {
                    assert_eq!(
                        expressions_layout.e4_dst_cols[i + offset] as usize,
                        translate_e4_offset(dst.start),
                    );
                } else {
                    assert_eq!(
                        expressions_for_shuffle_ram_layout.e4_dst_cols[i - num_non_shuffle_ram_args]
                            as usize,
                        translate_e4_offset(dst.start),
                    );
                }
            }
            let mut alpha = h_alphas_for_last_row_and_at_zero[alpha_offset];
            alpha_offset += 1;
            helpers.push(*alpha.negate().mul_assign_by_base(&decompression_factor));
        }
        // generic lookup e4 arg sums
        assert!(num_generic_multiplicities_cols > 0);
        let mut alpha = h_alphas_for_last_row_and_at_zero[alpha_offset];
        alpha_offset += 1;
        helpers.push(*alpha.negate().mul_assign_by_base(&decompression_factor));
        if handle_delegation_requests || process_delegations {
            let mut alpha = h_alphas_for_last_row_and_at_zero[alpha_offset];
            alpha_offset += 1;
            let mut delegation_accumulator_interpolant_prefactor = sum_over_delegation_poly;
            delegation_accumulator_interpolant_prefactor
                .negate()
                .mul_assign_by_base(&omega)
                .mul_assign_by_base(&decompression_factor_inv);
            helpers.push(delegation_accumulator_interpolant_prefactor);
            helpers.push(*alpha.mul_assign_by_base(&decompression_factor));
        }
        assert_eq!(alpha_offset, h_alphas_for_last_row_and_at_zero.len());
        assert!(helpers.len() <= MAX_HELPER_VALUES);
        helpers
            .spare_capacity_mut()
            .fill(MaybeUninit::new(E4::ZERO));
        unsafe {
            helpers.set_len(MAX_HELPER_VALUES);
        }
        Metadata {
            alpha_powers_layout,
            flat_generic_constraints_metadata,
            delegated_width_3_lookups_layout,
            non_delegated_width_3_lookups_layout,
            range_check_16_layout,
            expressions_layout,
            expressions_for_shuffle_ram_layout,
            generic_lookup_multiplicities_layout,
            state_linkage_constraints,
            boundary_constraints,
            memory_args_start,
            memory_grand_product_col,
            lazy_init_teardown_layout,
            shuffle_ram_accesses,
            batched_ram_accesses,
            range_check_16_multiplicities_layout,
            timestamp_range_check_multiplicities_layout,
            delegation_aux_poly_col,
            num_generic_constraints,
            delegation_challenges,
            delegation_processing_metadata,
            delegation_request_metadata,
            register_and_indirect_accesses,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn compute_stage_3_composition_quotient_on_coset(
    cached_data: &ProverCachedData,
    circuit: &CompiledCircuitArtifact<BF>,
    metadata: Metadata,
    setup_cols: &MetalBuffer<BF>,
    witness_cols: &MetalBuffer<BF>,
    memory_cols: &MetalBuffer<BF>,
    stage_2_cols: &MetalBuffer<BF>,
    d_alpha_powers: &MetalBuffer<E4>,
    d_beta_powers: &MetalBuffer<E4>,
    d_helpers: &MetalBuffer<E4>,
    d_constants_times_challenges: &MetalBuffer<ConstantsTimesChallenges>,
    quotient: &mut MetalBuffer<BF>,
    log_n: u32,
    context: &ProverContext,
) -> MetalResult<()> {
    let cmd_buf = context.new_command_buffer()?;
    compute_stage_3_composition_quotient_on_coset_into(
        &cmd_buf,
        cached_data,
        circuit,
        metadata,
        setup_cols,
        witness_cols,
        memory_cols,
        stage_2_cols,
        d_alpha_powers,
        d_beta_powers,
        d_helpers,
        d_constants_times_challenges,
        quotient,
        log_n,
        context,
    )?;
    cmd_buf.commit_and_wait();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn compute_stage_3_composition_quotient_on_coset_into(
    cmd_buf: &MetalCommandBuffer,
    cached_data: &ProverCachedData,
    circuit: &CompiledCircuitArtifact<BF>,
    metadata: Metadata,
    setup_cols: &MetalBuffer<BF>,
    witness_cols: &MetalBuffer<BF>,
    memory_cols: &MetalBuffer<BF>,
    stage_2_cols: &MetalBuffer<BF>,
    d_alpha_powers: &MetalBuffer<E4>,
    d_beta_powers: &MetalBuffer<E4>,
    d_helpers: &MetalBuffer<E4>,
    d_constants_times_challenges: &MetalBuffer<ConstantsTimesChallenges>,
    quotient: &mut MetalBuffer<BF>,
    log_n: u32,
    context: &ProverContext,
) -> MetalResult<()> {
    let n = 1usize << log_n;
    let _num_setup_cols = circuit.setup_layout.total_width;
    let _num_witness_cols = circuit.witness_layout.total_width;
    let _num_memory_cols = circuit.memory_layout.total_width;
    let num_stage_2_bf_cols = circuit.stage_2_layout.num_base_field_polys();
    let _num_stage_2_e4_cols = circuit.stage_2_layout.num_ext4_field_polys();
    let e4_cols_offset = circuit.stage_2_layout.ext4_polys_offset;
    assert_eq!(e4_cols_offset % 4, 0);
    assert!(num_stage_2_bf_cols <= e4_cols_offset);
    assert!(e4_cols_offset - num_stage_2_bf_cols < 4);
    let e4_byte_offset = e4_cols_offset * n * std::mem::size_of::<BF>();

    let ProverCachedData {
        trace_len,
        memory_timestamp_high_from_circuit_idx,
        handle_delegation_requests,
        process_batch_ram_access,
        process_registers_and_indirect_access,
        process_delegations,
        ..
    } = cached_data.clone();
    assert_eq!(trace_len, n);

    let Metadata {
        alpha_powers_layout,
        flat_generic_constraints_metadata,
        delegated_width_3_lookups_layout,
        non_delegated_width_3_lookups_layout,
        range_check_16_layout,
        expressions_layout,
        expressions_for_shuffle_ram_layout,
        generic_lookup_multiplicities_layout,
        state_linkage_constraints,
        boundary_constraints,
        memory_args_start,
        memory_grand_product_col,
        lazy_init_teardown_layout,
        shuffle_ram_accesses,
        batched_ram_accesses,
        range_check_16_multiplicities_layout,
        timestamp_range_check_multiplicities_layout,
        delegation_aux_poly_col,
        num_generic_constraints,
        delegation_challenges,
        delegation_processing_metadata,
        delegation_request_metadata,
        register_and_indirect_accesses,
    } = metadata;

    let AlphaPowersLayout {
        num_quotient_terms_every_row_except_last,
        num_quotient_terms_every_row_except_last_two,
        precomputation_size,
        ..
    } = alpha_powers_layout;

    let device = context.device();

    // Compute strides: each column has n rows
    let stride = n as u32;

    // Upload metadata structs to GPU
    let d_generic_metadata =
        context.alloc_from_slice(std::slice::from_ref(&flat_generic_constraints_metadata))?;

    // Split alpha powers for generic constraints
    let generic_alpha_start = precomputation_size - num_quotient_terms_every_row_except_last;
    let hardcoded_alpha_start = generic_alpha_start + num_generic_constraints;
    let every_row_except_last_two_start =
        precomputation_size - num_quotient_terms_every_row_except_last_two;

    // Launch generic constraints kernel
    // Zero the quotient buffer first
    crate::ops_simple::memset_zero(device, cmd_buf, quotient.raw(), quotient.byte_len())?;
    let generic_alphas_byte_offset = generic_alpha_start * std::mem::size_of::<E4>();
    launch_generic_constraints(
        device,
        cmd_buf,
        &d_generic_metadata,
        witness_cols,
        stride,
        memory_cols,
        stride,
        d_alpha_powers,
        generic_alphas_byte_offset,
        quotient,
        stride,
        log_n,
        e4_byte_offset,
    )?;

    // Launch delegated width 3 lookups kernel if needed
    if process_delegations {
        let d_delegated_layout =
            context.alloc_from_slice(std::slice::from_ref(&delegated_width_3_lookups_layout))?;
        // stage_2_e4_cols is a sub-region of stage_2_cols
        launch_delegated_width_3_lookups(
            device,
            cmd_buf,
            &d_delegated_layout,
            witness_cols,
            stride,
            memory_cols,
            stride,
            stage_2_cols, // e4 portion
            stride,
            d_helpers,
            quotient,
            stride,
            flat_generic_constraints_metadata.decompression_factor_squared,
            log_n,
            e4_byte_offset,
        )?;
    }

    // Upload all hardcoded constraint metadata to GPU
    let d_delegation_challenges =
        context.alloc_from_slice(std::slice::from_ref(&delegation_challenges))?;
    let d_delegation_processing_metadata =
        context.alloc_from_slice(std::slice::from_ref(&delegation_processing_metadata))?;
    let d_delegation_request_metadata =
        context.alloc_from_slice(std::slice::from_ref(&delegation_request_metadata))?;
    let d_lazy_init_teardown =
        context.alloc_from_slice(std::slice::from_ref(&lazy_init_teardown_layout))?;
    let d_shuffle_ram_accesses =
        context.alloc_from_slice(std::slice::from_ref(&shuffle_ram_accesses))?;
    let d_batched_ram_accesses =
        context.alloc_from_slice(std::slice::from_ref(&batched_ram_accesses))?;
    let d_register_and_indirect =
        context.alloc_from_slice(std::slice::from_ref(&register_and_indirect_accesses))?;
    let d_range_check_16_layout =
        context.alloc_from_slice(std::slice::from_ref(&range_check_16_layout))?;
    let d_expressions_layout =
        context.alloc_from_slice(std::slice::from_ref(&expressions_layout))?;
    let d_expressions_for_shuffle_ram =
        context.alloc_from_slice(std::slice::from_ref(&expressions_for_shuffle_ram_layout))?;
    let d_width_3_lookups =
        context.alloc_from_slice(std::slice::from_ref(&non_delegated_width_3_lookups_layout))?;
    let d_rc16_mult_layout =
        context.alloc_from_slice(std::slice::from_ref(&range_check_16_multiplicities_layout))?;
    let d_ts_mult_layout = context
        .alloc_from_slice(std::slice::from_ref(&timestamp_range_check_multiplicities_layout))?;
    let d_generic_mult_layout =
        context.alloc_from_slice(std::slice::from_ref(&generic_lookup_multiplicities_layout))?;
    let d_state_linkage =
        context.alloc_from_slice(std::slice::from_ref(&state_linkage_constraints))?;
    let d_boundary_constraints =
        context.alloc_from_slice(std::slice::from_ref(&boundary_constraints))?;

    let omega_inv = flat_generic_constraints_metadata.omega_inv;
    let omega_inv_squared = *omega_inv.clone().square();

    let hardcoded_alphas_byte_offset = hardcoded_alpha_start * std::mem::size_of::<E4>();
    let erly2_alphas_byte_offset = every_row_except_last_two_start * std::mem::size_of::<E4>();

    // Get twiddle data for get_power_of_w
    let device_ctx = context.device_context();

    let hardcoded_params = HardcodedConstraintsParams {
        setup_stride: stride,
        witness_stride: stride,
        memory_stride: stride,
        stage_2_bf_stride: stride,
        stage_2_e4_stride: stride,
        quotient_stride: stride,
        process_delegations_flag: if process_delegations { 1 } else { 0 },
        handle_delegation_requests_flag: if handle_delegation_requests { 1 } else { 0 },
        process_batch_ram_access_flag: if process_batch_ram_access { 1 } else { 0 },
        process_registers_flag: if process_registers_and_indirect_access { 1 } else { 0 },
        delegation_aux_poly_col: delegation_aux_poly_col as u32,
        memory_args_start: memory_args_start as u32,
        memory_grand_product_col: memory_grand_product_col as u32,
        log_n,
        memory_timestamp_high_from_circuit_idx,
        _pad0: 0,
        decompression_factor: flat_generic_constraints_metadata.decompression_factor,
        decompression_factor_squared: flat_generic_constraints_metadata.decompression_factor_squared,
        every_row_zerofier: flat_generic_constraints_metadata.every_row_zerofier,
        omega_inv,
        omega_inv_squared,
        twiddle_fine_mask: (1u32 << device_ctx.fine_log_count) - 1,
        twiddle_fine_log_count: device_ctx.fine_log_count,
        twiddle_coarser_mask: (1u32 << device_ctx.coarser_log_count) - 1,
        twiddle_coarser_log_count: device_ctx.coarser_log_count,
        twiddle_coarsest_mask: (1u32 << device_ctx.coarsest_log_count) - 1,
        twiddle_coarsest_log_count: device_ctx.coarsest_log_count,
    };

    launch_hardcoded_constraints(
        device,
        cmd_buf,
        setup_cols,
        witness_cols,
        memory_cols,
        stage_2_cols,
        stage_2_cols,
        e4_byte_offset,
        &d_delegation_challenges,
        &d_delegation_processing_metadata,
        &d_delegation_request_metadata,
        &d_lazy_init_teardown,
        &d_shuffle_ram_accesses,
        &d_batched_ram_accesses,
        &d_register_and_indirect,
        &d_range_check_16_layout,
        &d_expressions_layout,
        &d_expressions_for_shuffle_ram,
        &d_width_3_lookups,
        &d_rc16_mult_layout,
        &d_ts_mult_layout,
        &d_generic_mult_layout,
        &d_state_linkage,
        &d_boundary_constraints,
        d_alpha_powers,
        hardcoded_alphas_byte_offset,
        d_alpha_powers,
        erly2_alphas_byte_offset,
        d_beta_powers,
        d_helpers,
        d_constants_times_challenges,
        quotient,
        &device_ctx.powers_of_w_fine,
        &device_ctx.powers_of_w_coarser,
        &device_ctx.powers_of_w_coarsest,
        &hardcoded_params,
    )?;
    Ok(())
}

pub fn launch_generic_constraints(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    metadata: &MetalBuffer<FlattenedGenericConstraintsMetadata>,
    witness_cols: &MetalBuffer<BF>,
    witness_stride: u32,
    memory_cols: &MetalBuffer<BF>,
    memory_stride: u32,
    alphas: &MetalBuffer<E4>,
    alphas_byte_offset: usize,
    quotient: &MetalBuffer<BF>,
    quotient_stride: u32,
    log_n: u32,
    _e4_byte_offset: usize,
) -> MetalResult<()> {
    let n = 1u32 << log_n;
    let block_dim = WARP_SIZE * 4;
    let grid_dim = (n + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_generic_constraints_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            set_buffer(encoder, idx, metadata.raw(), 0); idx += 1;
            set_buffer(encoder, idx, witness_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &witness_stride); } idx += 1;
            set_buffer(encoder, idx, memory_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &memory_stride); } idx += 1;
            set_buffer(encoder, idx, alphas.raw(), alphas_byte_offset); idx += 1;
            set_buffer(encoder, idx, quotient.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &quotient_stride); } idx += 1;
            unsafe { set_bytes(encoder, idx, &log_n); }
        },
    )
}

pub fn launch_delegated_width_3_lookups(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    layout: &MetalBuffer<DelegatedWidth3LookupsLayout>,
    witness_cols: &MetalBuffer<BF>,
    witness_stride: u32,
    memory_cols: &MetalBuffer<BF>,
    memory_stride: u32,
    stage_2_e4_cols: &MetalBuffer<BF>,
    stage_2_e4_stride: u32,
    e4_helpers: &MetalBuffer<E4>,
    quotient: &MetalBuffer<BF>,
    quotient_stride: u32,
    decompression_factor_squared: E2,
    log_n: u32,
    e4_byte_offset: usize,
) -> MetalResult<()> {
    let n = 1u32 << log_n;
    let block_dim = WARP_SIZE * 4;
    let grid_dim = (n + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_delegated_width_3_lookups_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            set_buffer(encoder, idx, layout.raw(), 0); idx += 1;
            set_buffer(encoder, idx, witness_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &witness_stride); } idx += 1;
            set_buffer(encoder, idx, memory_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &memory_stride); } idx += 1;
            set_buffer(encoder, idx, stage_2_e4_cols.raw(), e4_byte_offset); idx += 1;
            unsafe { set_bytes(encoder, idx, &stage_2_e4_stride); } idx += 1;
            set_buffer(encoder, idx, e4_helpers.raw(), 0); idx += 1;
            set_buffer(encoder, idx, quotient.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &quotient_stride); } idx += 1;
            unsafe { set_bytes(encoder, idx, &decompression_factor_squared); } idx += 1;
            unsafe { set_bytes(encoder, idx, &log_n); }
        },
    )
}

#[allow(clippy::too_many_arguments)]
/// Packed scalar parameters for the hardcoded constraints kernel.
/// Must match `HardcodedConstraintsParams` in stage3.metal exactly.
#[derive(Clone)]
#[repr(C)]
pub struct HardcodedConstraintsParams {
    pub setup_stride: u32,
    pub witness_stride: u32,
    pub memory_stride: u32,
    pub stage_2_bf_stride: u32,
    pub stage_2_e4_stride: u32,
    pub quotient_stride: u32,
    pub process_delegations_flag: u32,
    pub handle_delegation_requests_flag: u32,
    pub process_batch_ram_access_flag: u32,
    pub process_registers_flag: u32,
    pub delegation_aux_poly_col: u32,
    pub memory_args_start: u32,
    pub memory_grand_product_col: u32,
    pub log_n: u32,
    pub memory_timestamp_high_from_circuit_idx: BF,
    pub _pad0: u32, // align E2 fields to 8-byte boundary (Rust E2 has align(8), Metal e2 has align(4))
    pub decompression_factor: E2,
    pub decompression_factor_squared: E2,
    pub every_row_zerofier: E2,
    pub omega_inv: E2,
    pub omega_inv_squared: E2,
    pub twiddle_fine_mask: u32,
    pub twiddle_fine_log_count: u32,
    pub twiddle_coarser_mask: u32,
    pub twiddle_coarser_log_count: u32,
    pub twiddle_coarsest_mask: u32,
    pub twiddle_coarsest_log_count: u32,
}

pub fn launch_hardcoded_constraints(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    // Column data
    setup_cols: &MetalBuffer<BF>,
    witness_cols: &MetalBuffer<BF>,
    memory_cols: &MetalBuffer<BF>,
    stage_2_bf_cols: &MetalBuffer<BF>,
    stage_2_e4_cols: &MetalBuffer<BF>,
    e4_byte_offset: usize,
    // Constraint metadata
    delegation_challenges: &MetalBuffer<DelegationChallenges>,
    delegation_processing_metadata: &MetalBuffer<DelegationProcessingMetadata>,
    delegation_request_metadata: &MetalBuffer<DelegationRequestMetadata>,
    lazy_init_teardown_layout: &MetalBuffer<LazyInitTeardownLayout>,
    shuffle_ram_accesses: &MetalBuffer<ShuffleRamAccesses>,
    batched_ram_accesses: &MetalBuffer<BatchedRamAccesses>,
    register_and_indirect_accesses: &MetalBuffer<RegisterAndIndirectAccesses>,
    range_check_16_layout: &MetalBuffer<RangeCheck16ArgsLayout>,
    expressions_layout: &MetalBuffer<FlattenedLookupExpressionsLayout>,
    expressions_for_shuffle_ram_layout: &MetalBuffer<FlattenedLookupExpressionsForShuffleRamLayout>,
    width_3_lookups_layout: &MetalBuffer<NonDelegatedWidth3LookupsLayout>,
    range_check_16_multiplicities_layout: &MetalBuffer<MultiplicitiesLayout>,
    timestamp_range_check_multiplicities_layout: &MetalBuffer<MultiplicitiesLayout>,
    generic_lookup_multiplicities_layout: &MetalBuffer<MultiplicitiesLayout>,
    state_linkage_constraints: &MetalBuffer<StateLinkageConstraints>,
    boundary_constraints: &MetalBuffer<BoundaryConstraints>,
    // Alpha/beta/helper vectors
    alpha_powers: &MetalBuffer<E4>,
    alphas_byte_offset: usize,
    alpha_powers_every_row_except_last_two: &MetalBuffer<E4>,
    alphas_erly2_byte_offset: usize,
    beta_powers: &MetalBuffer<E4>,
    e4_helpers: &MetalBuffer<E4>,
    constants_times_challenges: &MetalBuffer<ConstantsTimesChallenges>,
    // Quotient output
    quotient: &MetalBuffer<BF>,
    // Twiddle data
    twiddle_fine: &MetalBuffer<E2>,
    twiddle_coarser: &MetalBuffer<E2>,
    twiddle_coarsest: &MetalBuffer<E2>,
    // Packed params
    params: &HardcodedConstraintsParams,
) -> MetalResult<()> {
    let n = 1u32 << params.log_n;
    let block_dim = WARP_SIZE * 4;
    let grid_dim = (n + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_hardcoded_constraints_kernel",
        &config,
        |encoder| {
            // Buffer layout must match stage3.metal buffer indices 0-30
            set_buffer(encoder, 0, setup_cols.raw(), 0);
            set_buffer(encoder, 1, witness_cols.raw(), 0);
            set_buffer(encoder, 2, memory_cols.raw(), 0);
            set_buffer(encoder, 3, stage_2_bf_cols.raw(), 0);
            set_buffer(encoder, 4, stage_2_e4_cols.raw(), e4_byte_offset);
            set_buffer(encoder, 5, delegation_challenges.raw(), 0);
            set_buffer(encoder, 6, delegation_processing_metadata.raw(), 0);
            set_buffer(encoder, 7, delegation_request_metadata.raw(), 0);
            set_buffer(encoder, 8, lazy_init_teardown_layout.raw(), 0);
            set_buffer(encoder, 9, shuffle_ram_accesses.raw(), 0);
            set_buffer(encoder, 10, batched_ram_accesses.raw(), 0);
            set_buffer(encoder, 11, register_and_indirect_accesses.raw(), 0);
            set_buffer(encoder, 12, range_check_16_layout.raw(), 0);
            set_buffer(encoder, 13, expressions_layout.raw(), 0);
            set_buffer(encoder, 14, expressions_for_shuffle_ram_layout.raw(), 0);
            set_buffer(encoder, 15, width_3_lookups_layout.raw(), 0);
            set_buffer(encoder, 16, range_check_16_multiplicities_layout.raw(), 0);
            set_buffer(encoder, 17, timestamp_range_check_multiplicities_layout.raw(), 0);
            set_buffer(encoder, 18, generic_lookup_multiplicities_layout.raw(), 0);
            set_buffer(encoder, 19, state_linkage_constraints.raw(), 0);
            set_buffer(encoder, 20, boundary_constraints.raw(), 0);
            set_buffer(encoder, 21, alpha_powers.raw(), alphas_byte_offset);
            set_buffer(encoder, 22, alpha_powers_every_row_except_last_two.raw(), alphas_erly2_byte_offset);
            set_buffer(encoder, 23, beta_powers.raw(), 0);
            set_buffer(encoder, 24, e4_helpers.raw(), 0);
            set_buffer(encoder, 25, constants_times_challenges.raw(), 0);
            set_buffer(encoder, 26, quotient.raw(), 0);
            set_buffer(encoder, 27, twiddle_fine.raw(), 0);
            set_buffer(encoder, 28, twiddle_coarser.raw(), 0);
            set_buffer(encoder, 29, twiddle_coarsest.raw(), 0);
            unsafe { set_bytes(encoder, 30, params); }
        },
    )
}
