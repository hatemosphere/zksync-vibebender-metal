use crate::gkr::sumcheck::evaluation_kernels::{
    lookup_base_minus_multiplicity_base, lookup_base_pair, BatchedGKRKernel,
};
use std::sync::Arc;

use super::*;

pub fn forward_evaluate_copy<F: PrimeField, E: FieldExtension<F> + Field>(
    input: GKRAddress,
    output: GKRAddress,
    gkr_storage: &mut GKRStorage<F, E>,
    expected_output_layer: usize,
    _trace_len: usize,
    _worker: &Worker,
) {
    output.assert_as_layer(expected_output_layer);
    assert!(output.is_cache() == false);
    input.assert_as_layer(expected_output_layer - 1);

    if let Some(source) = gkr_storage.layers[expected_output_layer - 1]
        .base_field_inputs
        .get(&input)
        .map(|el| el.arc_clone())
    {
        println!("Copying base field {:?} -> {:?}", input, output);
        gkr_storage.insert_base_field_at_layer(expected_output_layer, output, source);
    } else {
        if let Some(source) = gkr_storage.layers[expected_output_layer - 1]
            .extension_field_inputs
            .get(&input)
            .map(|el| el.arc_clone())
        {
            println!("Copying extension field {:?} -> {:?}", input, output);
            gkr_storage.insert_extension_at_layer(expected_output_layer, output, source);
        } else {
            panic!(
                "Trying to copy {:?} -> {:?}, but the input is not present in storage",
                input, output
            );
        }
    };
}
