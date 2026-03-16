extern crate alloc;
use alloc::vec::Vec;

use cs::gkr_compiler::GKRCircuitArtifact;
use field::{Field, FieldExtension, PrimeField};
use prover::gkr::prover::GKRProof;
use prover::merkle_trees::ColumnMajorMerkleTreeConstructor;

fn flatten_field_els<F: PrimeField, E: FieldExtension<F>>(src: &[E], dst: &mut Vec<u32>)
where
    [(); E::DEGREE]: Sized,
{
    use field::FixedArrayConvertible;
    for el in src.iter() {
        let coeffs = E::into_coeffs(*el)
            .into_array::<{ E::DEGREE }>()
            .map(|e: F| e.as_u32_raw_repr_reduced());
        dst.extend(coeffs);
    }
}

/// Flatten a GKR proof into NDS reading order.
///
/// The output stream contains (in order):
/// 1. Initial transcript preamble (reconstructed from proof fields):
///    - `inits_and_teardowns_top_bits` (if present)
///    - external challenges
///    - setup, memory, witness Merkle caps
/// 2. `final_explicit_evaluations` (in BTreeMap order by OutputType, poly0 then poly1)
/// 3. Per-layer sumcheck data (dim-reducing first top-to-bottom, then standard top-to-bottom):
///    - For each regular round: 4 field elements `[E; 4]`
///    - For each final step: evaluations per address (2 for standard, 4 for dim-reducing)
/// 4. `grand_product_accumulator_computed` (1 field element)
pub fn flatten_gkr_proof_for_nds<F: PrimeField, E: FieldExtension<F> + Field, T>(
    proof: &GKRProof<F, E, T>,
    compiled_circuit: &GKRCircuitArtifact<F>,
) -> Vec<u32>
where
    T: ColumnMajorMerkleTreeConstructor<F>,
    [(); E::DEGREE]: Sized,
{
    let mut result = Vec::new();

    if let Some(top_bits) = proof.inits_and_teardowns_top_bits {
        result.push(top_bits);
    }
    proof.external_challenges.flatten_into_buffer(&mut result);
    proof
        .whir_proof
        .setup_commitment
        .commitment
        .cap
        .add_into_buffer(&mut result);
    proof
        .whir_proof
        .memory_commitment
        .commitment
        .cap
        .add_into_buffer(&mut result);
    proof
        .whir_proof
        .witness_commitment
        .commitment
        .cap
        .add_into_buffer(&mut result);

    for (_output_type, pair) in proof.final_explicit_evaluations.iter() {
        flatten_field_els::<F, E>(&pair[0], &mut result);
        flatten_field_els::<F, E>(&pair[1], &mut result);
    }

    let num_standard_layers = compiled_circuit.layers.len();
    let initial_layer_for_sumcheck = *proof
        .sumcheck_intermediate_values
        .keys()
        .max()
        .expect("proof must have sumcheck values");

    let dim_reducing_indices: Vec<usize> = (num_standard_layers..=initial_layer_for_sumcheck)
        .rev()
        .collect();

    let standard_indices: Vec<usize> = (0..num_standard_layers).rev().collect();

    for &layer_idx in dim_reducing_indices.iter().chain(standard_indices.iter()) {
        let proof_values = proof
            .sumcheck_intermediate_values
            .get(&layer_idx)
            .expect("missing sumcheck values for layer");

        for coeffs in proof_values.internal_round_coefficients.iter() {
            flatten_field_els::<F, E>(coeffs, &mut result);
        }

        for (_addr, evals) in proof_values.final_step_evaluations.iter() {
            flatten_field_els::<F, E>(evals, &mut result);
        }
    }

    flatten_field_els::<F, E>(&[proof.grand_product_accumulator_computed], &mut result);

    result
}
