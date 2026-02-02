use std::collections::BTreeMap;

use cs::definitions::GKRAddress;
use field::{Field, FieldExtension, PrimeField};
use worker::Worker;

use super::{
    evaluate_single_input_type_fixed_in_out_kernel_with_extension_inputs,
    forward_evaluate_single_input_type_fixed_in_out_kernel_with_extension_inputs, BatchedGKRKernel,
    EvaluationRepresentation, ExtensionFieldInOutFixedSizesEvaluationKernel,
    ExtensionFieldRepresentation, GKRInputs,
};
use crate::gkr::sumcheck::access_and_fold::GKRStorage;

// pub struct UnbalancedGrandProductWithCacheGKRRelation {
//     pub inputs: [GKRAddress; 2],
//     pub output: GKRAddress,
// }

// impl UnbalancedGrandProductWithCacheGKRRelation {
//     /// Validates that exactly one input is from cache, output is not cached
//     #[inline]
//     fn validate(&self) -> bool {
//         let cached_count = self.inputs.iter().filter(|a| a.is_cache()).count();
//         cached_count == 1 && !self.output.is_cache()
//     }
// }

// impl<F: PrimeField, E: FieldExtension<F> + Field> BatchedGKRKernel<F, E>
//     for UnbalancedGrandProductWithCacheGKRRelation
// {
//     fn num_challenges(&self) -> usize {
//         1
//     }

//     fn get_inputs(&self) -> GKRInputs {
//         debug_assert!(self.validate());
//         GKRInputs {
//             inputs_in_base: Vec::new(),
//             inputs_in_extension: self.inputs.to_vec(),
//             outputs_in_base: Vec::new(),
//             outputs_in_extension: vec![self.output],
//         }
//     }

//     fn evaluate_forward_over_storage(
//         &self,
//         storage: &mut GKRStorage<F, E>,
//         expected_output_layer: usize,
//         trace_len: usize,
//         worker: &Worker,
//     ) {
//         let kernel = ProductGKRRelationKernel::default();
//         let inputs = <Self as BatchedGKRKernel<F, E>>::get_inputs(self);
//         forward_evaluate_single_input_type_fixed_in_out_kernel_with_extension_inputs(
//             &kernel,
//             &inputs,
//             storage,
//             expected_output_layer,
//             trace_len,
//             worker,
//         );
//     }

//     fn evaluate_over_storage(
//         &self,
//         storage: &mut GKRStorage<F, E>,
//         step: usize,
//         batch_challenges: &[E],
//         folding_challenges: &[E],
//         accumulator: &mut [[E; 2]],
//         total_sumcheck_rounds: usize,
//         last_evaluations: &mut BTreeMap<GKRAddress, [E; 2]>,
//         worker: &Worker,
//     ) {
//         assert_eq!(
//             batch_challenges.len(),
//             <Self as BatchedGKRKernel<F, E>>::num_challenges(self)
//         );
//         let kernel = ProductGKRRelationKernel::default();
//         let inputs = <Self as BatchedGKRKernel<F, E>>::get_inputs(self);

//         evaluate_single_input_type_fixed_in_out_kernel_with_extension_inputs(
//             &kernel,
//             &inputs,
//             storage,
//             step,
//             batch_challenges,
//             folding_challenges,
//             accumulator,
//             total_sumcheck_rounds,
//             last_evaluations,
//             worker,
//         );
//     }
// }

// pub struct InitialGrandProductFromCachesGKRRelation {
//     pub inputs: [GKRAddress; 2],
//     pub output: GKRAddress,
// }

// impl InitialGrandProductFromCachesGKRRelation {
//     /// Validates that both inputs are from caches
//     #[inline]
//     fn validate(&self) -> bool {
//         self.inputs[0].is_cache() && self.inputs[1].is_cache() && !self.output.is_cache()
//     }
// }

// impl<F: PrimeField, E: FieldExtension<F> + Field> BatchedGKRKernel<F, E>
//     for InitialGrandProductFromCachesGKRRelation
// {
//     fn num_challenges(&self) -> usize {
//         1
//     }

//     fn get_inputs(&self) -> GKRInputs {
//         debug_assert!(self.validate());
//         GKRInputs {
//             inputs_in_base: Vec::new(),
//             inputs_in_extension: self.inputs.to_vec(),
//             outputs_in_base: Vec::new(),
//             outputs_in_extension: vec![self.output],
//         }
//     }

//     fn evaluate_forward_over_storage(
//         &self,
//         storage: &mut GKRStorage<F, E>,
//         expected_output_layer: usize,
//         trace_len: usize,
//         worker: &Worker,
//     ) {
//         let kernel = ProductGKRRelationKernel::default();
//         let inputs = <Self as BatchedGKRKernel<F, E>>::get_inputs(self);
//         forward_evaluate_single_input_type_fixed_in_out_kernel_with_extension_inputs(
//             &kernel,
//             &inputs,
//             storage,
//             expected_output_layer,
//             trace_len,
//             worker,
//         );
//     }

//     fn evaluate_over_storage(
//         &self,
//         storage: &mut GKRStorage<F, E>,
//         step: usize,
//         batch_challenges: &[E],
//         folding_challenges: &[E],
//         accumulator: &mut [[E; 2]],
//         total_sumcheck_rounds: usize,
//         last_evaluations: &mut BTreeMap<GKRAddress, [E; 2]>,
//         worker: &Worker,
//     ) {
//         assert_eq!(
//             batch_challenges.len(),
//             <Self as BatchedGKRKernel<F, E>>::num_challenges(self)
//         );
//         let kernel = ProductGKRRelationKernel::default();
//         let inputs = <Self as BatchedGKRKernel<F, E>>::get_inputs(self);

//         evaluate_single_input_type_fixed_in_out_kernel_with_extension_inputs(
//             &kernel,
//             &inputs,
//             storage,
//             step,
//             batch_challenges,
//             folding_challenges,
//             accumulator,
//             total_sumcheck_rounds,
//             last_evaluations,
//             worker,
//         );
//     }
// }

// pub struct LookupWithCachedDensAndSetupGKRRelation {
//     pub input: [GKRAddress; 2], // [mask, denominator]
//     pub setup: [GKRAddress; 2], // [multiplicity, setup_denominator]
//     pub output: [GKRAddress; 2],
// }

// impl LookupWithCachedDensAndSetupGKRRelation {
//     /// Validates: input[0] (mask) NOT cached, input[1] (denominator) IS cached
//     #[inline]
//     fn validate(&self) -> bool {
//         !self.input[0].is_cache() && self.input[1].is_cache()
//     }
// }

// impl<F: PrimeField, E: FieldExtension<F> + Field> BatchedGKRKernel<F, E>
//     for LookupWithCachedDensAndSetupGKRRelation
// {
//     fn num_challenges(&self) -> usize {
//         2
//     }

//     fn get_inputs(&self) -> GKRInputs {
//         debug_assert!(self.validate());
//         GKRInputs {
//             inputs_in_base: Vec::new(),
//             inputs_in_extension: [self.input[0], self.input[1], self.setup[0], self.setup[1]]
//                 .to_vec(),
//             outputs_in_base: Vec::new(),
//             outputs_in_extension: self.output.to_vec(),
//         }
//     }

//     fn evaluate_forward_over_storage(
//         &self,
//         storage: &mut GKRStorage<F, E>,
//         expected_output_layer: usize,
//         trace_len: usize,
//         worker: &Worker,
//     ) {
//         let kernel = LookupSubGKRRelationKernel::default();
//         let inputs = <Self as BatchedGKRKernel<F, E>>::get_inputs(self);
//         forward_evaluate_single_input_type_fixed_in_out_kernel_with_extension_inputs(
//             &kernel,
//             &inputs,
//             storage,
//             expected_output_layer,
//             trace_len,
//             worker,
//         );
//     }

//     fn evaluate_over_storage(
//         &self,
//         storage: &mut GKRStorage<F, E>,
//         step: usize,
//         batch_challenges: &[E],
//         folding_challenges: &[E],
//         accumulator: &mut [[E; 2]],
//         total_sumcheck_rounds: usize,
//         last_evaluations: &mut BTreeMap<GKRAddress, [E; 2]>,
//         worker: &Worker,
//     ) {
//         assert_eq!(
//             batch_challenges.len(),
//             <Self as BatchedGKRKernel<F, E>>::num_challenges(self)
//         );
//         let kernel = LookupSubGKRRelationKernel::default();
//         let inputs = <Self as BatchedGKRKernel<F, E>>::get_inputs(self);

//         evaluate_single_input_type_fixed_in_out_kernel_with_extension_inputs(
//             &kernel,
//             &inputs,
//             storage,
//             step,
//             batch_challenges,
//             folding_challenges,
//             accumulator,
//             total_sumcheck_rounds,
//             last_evaluations,
//             worker,
//         );
//     }
// }

// #[derive(Default)]
// pub struct LookupSubGKRRelationKernel<F: PrimeField, E: FieldExtension<F> + Field> {
//     _marker: core::marker::PhantomData<(F, E)>,
// }

// impl<F: PrimeField, E: FieldExtension<F> + Field>
//     ExtensionFieldInOutFixedSizesEvaluationKernel<F, E, 4, 2> for LookupSubGKRRelationKernel<F, E>
// {
//     #[inline(always)]
//     fn pointwise_eval(&self, input: &[ExtensionFieldRepresentation<F, E>; 4]) -> [E; 2] {
//         let [a, b, c, d] = input.each_ref().map(|x| x.into_value());
//         // a/b - c/d = (a*d - c*b) / (b*d)
//         let mut num = a;
//         num.mul_assign(&d);
//         let mut cb = c;
//         cb.mul_assign(&b);
//         num.sub_assign(&cb);

//         let mut den = b;
//         den.mul_assign(&d);
//         [num, den]
//     }
// }
