use cs::definitions::GKRAddress;
use worker::Worker;

use super::*;

pub struct SameSizeProductGKRRelation {
    pub inputs: [GKRAddress; 2],
    pub output: GKRAddress,
}

impl<F: PrimeField, E: FieldExtension<F> + Field> BatchedGKRKernel<F, E>
    for SameSizeProductGKRRelation
{
    fn num_challenges(&self) -> usize {
        1
    }

    fn get_inputs(&self) -> GKRInputs {
        GKRInputs {
            inputs_in_base: Vec::new(),
            inputs_in_extension: self.inputs.to_vec(),
            outputs_in_base: Vec::new(),
            outputs_in_extension: vec![self.output],
        }
    }

    fn evaluate_forward_over_storage(
        &self,
        storage: &mut GKRStorage<F, E>,
        expected_output_layer: usize,
        trace_len: usize,
        worker: &Worker,
    ) {
        let kernel = SameSizeProductGKRRelationKernel {
            _marker: core::marker::PhantomData,
        };
        let inputs = <Self as BatchedGKRKernel<F, E>>::get_inputs(self);
        forward_evaluate_single_input_type_fixed_in_out_kernel_with_extension_inputs(
            &kernel,
            &inputs,
            storage,
            expected_output_layer,
            trace_len,
            worker,
        );
    }

    fn evaluate_over_storage(
        &self,
        storage: &mut GKRStorage<F, E>,
        step: usize,
        batch_challenges: &[E],
        folding_challenges: &[E],
        accumulator: &mut [[E; 2]],
        total_sumcheck_rounds: usize,
        last_evaluations: &mut BTreeMap<GKRAddress, [E; 2]>,
        worker: &Worker,
    ) {
        assert_eq!(
            batch_challenges.len(),
            <Self as BatchedGKRKernel<F, E>>::num_challenges(self)
        );
        let kernel = SameSizeProductGKRRelationKernel {
            _marker: core::marker::PhantomData,
        };
        let inputs = <Self as BatchedGKRKernel<F, E>>::get_inputs(self);

        evaluate_single_input_type_fixed_in_out_kernel_with_extension_inputs(
            &kernel,
            &inputs,
            storage,
            step,
            batch_challenges,
            folding_challenges,
            accumulator,
            total_sumcheck_rounds,
            last_evaluations,
            worker,
        );

        // evaluate_single_input_kernel_with_extension_inputs(
        //     &kernel,
        //     &inputs,
        //     storage,
        //     step,
        //     batch_challenges,
        //     folding_challenges,
        //     accumulator,
        //     total_sumcheck_rounds,
        //     last_evaluations,
        // );
    }
}

// Assumes reordering of access implementors, to have lhs at 0 and rhs at 1
pub struct SameSizeProductGKRRelationKernel<F: PrimeField, E: FieldExtension<F> + Field> {
    _marker: core::marker::PhantomData<(F, E)>,
}

impl<F: PrimeField, E: FieldExtension<F> + Field>
    ExtensionFieldInOutFixedSizesEvaluationKernel<F, E, 2, 1>
    for SameSizeProductGKRRelationKernel<F, E>
{
    #[inline(always)]
    fn pointwise_eval(&self, input: &[ExtensionFieldRepresentation<F, E>; 2]) -> [E; 1] {
        let [a, b] = input;
        let mut a = *a;
        a.repr_mul_assign::<false>(b);
        [a.into_value()]
    }
}

// impl<F: PrimeField, E: FieldExtension<F> + Field> SingleInputTypeBatchSumcheckEvaluationKernel<F, E>
//     for SameSizeProductGKRRelationKernel<F, E>
// {
//     fn num_challenges(&self) -> usize {
//         1
//     }
//     fn evaluate_first_round<
//         R0: EvaluationRepresentation<F, E>,
//         S0: EvaluationFormStorage<F, E, R0>,
//         ROUT: EvaluationRepresentation<F, E>,
//         SOUT: EvaluationFormStorage<F, E, ROUT>,
//     >(
//         &self,
//         index: usize,
//         r0_sources: &[S0],
//         output_sources: &[SOUT],
//         batch_challenges: &[E],
//     ) -> [E; 2] {
//         debug_assert_eq!(batch_challenges.len(), self.num_challenges());

//         unsafe {
//             let [lhs, rhs] = r0_sources
//                 .as_chunks::<2>()
//                 .0
//                 .iter()
//                 .next()
//                 .unwrap_unchecked();
//             let output_source = &output_sources[0];
//             let ctx = lhs.get_collapse_context();
//             let lhs = lhs.get_f1_minus_f0_only(index);
//             let rhs = rhs.get_f1_minus_f0_only(index);
//             let out_ctx = output_source.get_collapse_context();
//             let output = output_source.get_f0_only(index);
//             let mut result = [const { MaybeUninit::uninit() }; 2];
//             // we have access to output
//             {
//                 result[0].write(output.collapse_for_batch_eval(out_ctx, &batch_challenges[0]));
//             }
//             {
//                 let mut product = lhs;
//                 product.repr_mul_assign::<true>(&rhs);
//                 result[1].write(product.collapse_for_batch_eval(ctx, &batch_challenges[0]));
//             }

//             result.map(|el| el.assume_init())
//         }
//     }

//     fn evaluate<
//         R0: EvaluationRepresentation<F, E>,
//         S0: EvaluationFormStorage<F, E, R0>,
//         const EXPLICIT_FORM: bool,
//     >(
//         &self,
//         index: usize,
//         r0_sources: &[S0],
//         batch_challenges: &[E],
//     ) -> [E; 2] {
//         debug_assert_eq!(batch_challenges.len(), self.num_challenges());
//         unsafe {
//             let [lhs, rhs] = r0_sources
//                 .as_chunks::<2>()
//                 .0
//                 .iter()
//                 .next()
//                 .unwrap_unchecked();
//             let ctx = lhs.get_collapse_context();
//             let lhs = lhs.get_two_points::<EXPLICIT_FORM>(index);
//             let rhs = rhs.get_two_points::<EXPLICIT_FORM>(index);
//             let mut result = [const { MaybeUninit::uninit() }; 2];
//             for i in 0..2 {
//                 let mut product = lhs[i];
//                 product.repr_mul_assign::<true>(&rhs[i]);
//                 result[i].write(product.collapse_for_batch_eval(ctx, &batch_challenges[0]));
//             }

//             result.map(|el| el.assume_init())
//         }
//     }
// }
