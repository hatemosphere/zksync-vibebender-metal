use std::mem::MaybeUninit;

use cs::definitions::GKRAddress;

use super::*;

pub struct SameSizeProductGKRRelation {
    pub inputs: [GKRAddress; 2],
    pub output: GKRAddress,
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> BatchedGKRKernel<F, E>
    for SameSizeProductGKRRelation
{
    fn get_inputs(&self) -> GKRInputs {
        GKRInputs {
            inputs_in_base: self.inputs.to_vec(),
            inputs_in_extension: Vec::new(),
            outputs_in_base: Vec::new(),
            outputs_in_extension: vec![self.output],
        }
    }

    fn evaluate_over_storage(
        &self,
        storage: &mut GKRStorage<F, E>,
        step: usize,
        batch_challenge: &E,
        folding_challenges: &[E],
        accumulator: &mut [[E; 2]],
    ) {
        let kernel = SameSizeProductGKRRelationKernel {
            _marker: core::marker::PhantomData,
        };
        let inputs = <Self as BatchedGKRKernel<F, E>>::get_inputs(self);
        evaluate_single_input_kernel_with_extension_inputs(
            &kernel,
            &inputs,
            storage,
            step,
            batch_challenge,
            folding_challenges,
            accumulator,
        );
    }
}

// Assumes reordering of access implementors, to have lhs at 0 and rhs at 1
pub struct SameSizeProductGKRRelationKernel<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    _marker: core::marker::PhantomData<(F, E)>,
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField>
    SingleInputTypeBatchSumcheckEvaluationKernel<F, E> for SameSizeProductGKRRelationKernel<F, E>
{
    fn evaluate_first_round<
        R0: EvaluationRepresentation<F, E>,
        S0: EvaluationFormStorage<F, E, R0>,
        ROUT: EvaluationRepresentation<F, E>,
        SOUT: EvaluationFormStorage<F, E, ROUT>,
    >(
        &self,
        index: usize,
        r0_sources: &[S0],
        output_sources: &[SOUT],
        batch_challenge: &E,
    ) -> [E; 2] {
        unsafe {
            let [lhs, rhs] = r0_sources
                .as_chunks::<2>()
                .0
                .iter()
                .next()
                .unwrap_unchecked();
            let output_source = &output_sources[0];
            let ctx = lhs.get_collapse_context();
            let lhs = lhs.get_f1_minus_f0_only(index);
            let rhs = rhs.get_f1_minus_f0_only(index);
            let out_ctx = output_source.get_collapse_context();
            let output = output_source.get_f0_only(index);
            let mut result = [const { MaybeUninit::uninit() }; 2];
            // we have access to output
            {
                result[0].write(output.collapse_for_batch_eval(out_ctx, batch_challenge));
            }
            {
                let mut product = lhs;
                product.repr_mul_assign::<true>(&rhs);
                result[1].write(product.collapse_for_batch_eval(ctx, batch_challenge));
            }

            result.map(|el| el.assume_init())
        }
    }

    fn evaluate<R0: EvaluationRepresentation<F, E>, S0: EvaluationFormStorage<F, E, R0>>(
        &self,
        index: usize,
        r0_sources: &[S0],
        batch_challenge: &E,
    ) -> [E; 2] {
        unsafe {
            let [lhs, rhs] = r0_sources
                .as_chunks::<2>()
                .0
                .iter()
                .next()
                .unwrap_unchecked();
            let ctx = lhs.get_collapse_context();
            let lhs = lhs.get_f0_and_f1_minus_f0(index);
            let rhs = rhs.get_f0_and_f1_minus_f0(index);
            let mut result = [const { MaybeUninit::uninit() }; 2];
            for i in 0..2 {
                let mut product = lhs[i];
                product.repr_mul_assign::<true>(&rhs[i]);
                result[i].write(product.collapse_for_batch_eval(ctx, batch_challenge));
            }

            result.map(|el| el.assume_init())
        }
    }
}
