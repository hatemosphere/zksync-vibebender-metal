use crate::gkr::sumcheck::evaluation_kernels::{
    lookup_pair, BatchedGKRKernel,
};

use super::*;

pub fn forward_evaluate_lookup_pair<
    F: PrimeField,
    E: FieldExtension<F> + Field,
>(
    inputs: [[GKRAddress; 2]; 2],
    outputs: [GKRAddress; 2],
    gkr_storage: &mut GKRStorage<F, E>,
    expected_output_layer: usize,
    trace_len: usize,
    worker: &Worker,
) {
    let kernel = lookup_pair::LookupPairGKRRelation {
        inputs,
        outputs,
    };
    kernel.evaluate_forward_over_storage(gkr_storage, expected_output_layer, trace_len, worker);
}