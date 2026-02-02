use crate::gkr::sumcheck::evaluation_kernels::{
    mask_into_identity, BatchedGKRKernel,
};

use super::*;

pub fn forward_evaluate_mask_into_identity<
    F: PrimeField,
    E: FieldExtension<F> + Field,
>(
    input: GKRAddress,
    mask: GKRAddress,
    output: GKRAddress,
    gkr_storage: &mut GKRStorage<F, E>,
    expected_output_layer: usize,
    trace_len: usize,
    worker: &Worker,
) {
    let kernel =
        mask_into_identity::MaskIntoIdentityProductGKRRelation {
            input,
            mask,
            output,
        };
    kernel.evaluate_forward_over_storage(gkr_storage, expected_output_layer, trace_len, worker);
}