use super::*;
use crate::gkr::sumcheck::access_and_fold::BaseFieldPoly;
use crate::{cs::definitions::*, gkr::sumcheck::access_and_fold::ExtensionFieldPoly};
use cs::gkr_compiler::{
    CompiledAddressSpaceRelationStrict, CompiledAddressStrict, NoFieldGKRRelation,
};
use cs::{
    definitions::{gkr::DECODER_LOOKUP_FORMAL_SET_INDEX, GKRAddress},
    gkr_compiler::{GKRLayerDescription, NoFieldGKRCacheRelation},
};

pub fn evaluate_sumcheck_for_layer<F: PrimeField, E: FieldExtension<F> + Field>(
    layer_idx: usize,
    layer: &GKRLayerDescription,
    claim_points: &mut BTreeMap<usize, Vec<E>>, // tells that for claims at layer X we used random point Y
    claims_storage: &mut BTreeMap<usize, BTreeMap<GKRAddress, E>>, // claim values that we produced at layer_idx + 1, so those are inputs to sumchecks
    gkr_storage: &mut GKRStorage<F, E>,
    compiled_circuit: &GKRCircuitArtifact<F>,
    external_challenges: &GKRExternalChallenges<F, E>,
    trace_len: usize,
    lookup_challenges_additive_part: E,
    constraints_batch_challenge: E,
    worker: &Worker,
) {
    println!("Evaluating layer {} in sumcheck direction", layer_idx);

    todo!();

    // // first we compute caches
    // for (addr, cache_relation) in layer.cached_relations.iter() {
    //     // println!(
    //     //     "Computing cache relation {:?} for output {:?}",
    //     //     cache_relation, addr
    //     // );

    //     addr.assert_as_layer(layer_idx);
    //     evaluate_cache_relation(
    //         layer_idx,
    //         *addr,
    //         cache_relation,
    //         gkr_storage,
    //         external_challenges,
    //         witness_trace,
    //         trace_len,
    //         preprocessed_range_check_16,
    //         preprocessed_timestamp_range_checks,
    //         preprocessed_generic_lookup,
    //         lookup_challenges_additive_part,
    //         worker,
    //     );
    // }

    // let expected_output_layer = layer_idx + 1;
    // assert!(layer.gates.is_empty() ^ layer.gates_with_external_connections.is_empty());
    // if layer_idx != compiled_circuit.layers.len() - 1 {
    //     assert!(layer.gates_with_external_connections.is_empty());
    // } else {
    //     assert!(layer.gates.is_empty());
    // }

    // for gate in layer
    //     .gates
    //     .iter()
    //     .chain(layer.gates_with_external_connections.iter())
    // {
    //     assert_eq!(gate.output_layer, expected_output_layer);

    //     // println!("Should evaluate {:?}", &gate.enforced_relation);
    //     match &gate.enforced_relation {
    //         NoFieldGKRRelation::Copy { input, output } => {
    //             // println!("Should evaluate {:?}", &gate.enforced_relation);
    //             copy::forward_evaluate_copy(
    //                 *input,
    //                 *output,
    //                 gkr_storage,
    //                 expected_output_layer,
    //                 trace_len,
    //                 worker,
    //             );
    //         }
    //         NoFieldGKRRelation::InitialGrandProductFromCaches { input, output } => {
    //             // println!("Should evaluate {:?}", &gate.enforced_relation);
    //             pairwise_product::forward_evaluate_pairwise_product(
    //                 *input,
    //                 *output,
    //                 gkr_storage,
    //                 expected_output_layer,
    //                 trace_len,
    //                 worker,
    //             );
    //         }
    //         NoFieldGKRRelation::MaskIntoIdentityProduct {
    //             input,
    //             mask,
    //             output,
    //         } => {
    //             // println!("Should evaluate {:?}", &gate.enforced_relation);
    //             mask_product::forward_evaluate_mask_into_identity(
    //                 *input,
    //                 *mask,
    //                 *output,
    //                 gkr_storage,
    //                 expected_output_layer,
    //                 trace_len,
    //                 worker,
    //             );
    //         }
    //         NoFieldGKRRelation::TrivialProduct { input, output } => {
    //             // println!("Should evaluate {:?}", &gate.enforced_relation);
    //             pairwise_product::forward_evaluate_pairwise_product(
    //                 *input,
    //                 *output,
    //                 gkr_storage,
    //                 expected_output_layer,
    //                 trace_len,
    //                 worker,
    //             );
    //         }
    //         NoFieldGKRRelation::EnforceConstraintsMaxQuadratic { .. } => {
    //             // we do nothing as it should result in all zeroes in case if constraints are satisfied
    //         }
    //         NoFieldGKRRelation::LookupFromBaseInputsWithSetup { .. } => {
    //             unimplemented!("not used");
    //         }
    //         NoFieldGKRRelation::LookupFromMaterializedBaseInputWithSetup {
    //             input,
    //             setup,
    //             output,
    //         } => {
    //             // println!("Should evaluate {:?}", &gate.enforced_relation);
    //             lookup_from_base_inputs::forward_evaluate_lookup_from_base_inputs_with_setup(
    //                 *input,
    //                 *setup,
    //                 *output,
    //                 gkr_storage,
    //                 expected_output_layer,
    //                 trace_len,
    //                 lookup_challenges_additive_part,
    //                 worker,
    //             );
    //         }
    //         NoFieldGKRRelation::LookupPairFromMaterializedBaseInputs { input, output } => {
    //             // println!("Should evaluate {:?}", &gate.enforced_relation);
    //             lookup_from_base_inputs::forward_evaluate_lookup_base_inputs_pair(
    //                 *input,
    //                 *output,
    //                 gkr_storage,
    //                 expected_output_layer,
    //                 trace_len,
    //                 lookup_challenges_additive_part,
    //                 worker,
    //             );
    //         }
    //         NoFieldGKRRelation::LookupWithCachedDensAndSetup {
    //             input,
    //             setup,
    //             output,
    //         } => {
    //             // println!("Should evaluate {:?}", &gate.enforced_relation);
    //             lookup_from_vector_inputs::forward_evaluate_masked_lookup_from_vector_inputs_with_setup(*input, *setup, *output, gkr_storage, expected_output_layer, trace_len, worker);
    //         }
    //         NoFieldGKRRelation::LookupPair { input, output } => {
    //             // println!("Should evaluate {:?}", &gate.enforced_relation);
    //             lookup_pair::forward_evaluate_lookup_pair(
    //                 *input,
    //                 *output,
    //                 gkr_storage,
    //                 expected_output_layer,
    //                 trace_len,
    //                 worker,
    //             );
    //         }
    //         NoFieldGKRRelation::LookupUnbalancedPairWithMaterializedBaseInputs {
    //             input,
    //             remainder,
    //             output,
    //         } => {
    //             // println!("Should evaluate {:?}", &gate.enforced_relation);
    //             lookup_from_base_inputs::forward_evaluate_lookup_rational_with_base_remainder_input(
    //                 *input,
    //                 *remainder,
    //                 *output,
    //                 gkr_storage,
    //                 expected_output_layer,
    //                 trace_len,
    //                 lookup_challenges_additive_part,
    //                 worker,
    //             );
    //         }
    //         rel @ _ => {
    //             println!("Should evaluate {:?}", &gate.enforced_relation);
    //         }
    //     }
    // }

    // at the end for any cache relation we should compute claims of the corresponding inputs, if those were not computed already
}
