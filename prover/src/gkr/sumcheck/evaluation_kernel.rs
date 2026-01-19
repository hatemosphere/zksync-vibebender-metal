use crate::gkr::sumcheck::storage::PolysStorage;
use cs::definitions::GKRAddress;
use field::FieldExtension;

use super::*;

pub trait BatchGKRRelation<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    type FirstStep<'a>: BatchGKREvaluationKernel<F, E>
    where
        Self: 'a;
    fn prepare_first_step<'a>(&'a self, storage: &'a mut PolysStorage<F, E>)
        -> Self::FirstStep<'a>;
    fn finish_first_step(&self, storage: &mut PolysStorage<F, E>);

    type NextSteps<'a>: BatchGKREvaluationKernel<F, E>
    where
        Self: 'a;
    fn prepare_next_steps<'a>(
        &'a self,
        storage: &'a mut PolysStorage<F, E>,
        step: usize,
    ) -> Self::NextSteps<'a>;
}

pub trait BatchGKREvaluationKernel<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    fn evaluate_in_batched_mode(
        &mut self,
        batch_gkr_challenge: &E,
        accumulation_buffer: &mut [[E; 3]], // max quadratic
        num_steps: usize,
        index_offset: usize,
    );
}

pub struct SameSizeProductGKRRelation {
    pub cached_sources: [GKRAddress; 2],
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> BatchGKRRelation<F, E>
    for SameSizeProductGKRRelation
{
    type FirstStep<'a> = SameSizeProductEvaluationKernelFirstStep<'a, F, E>;
    fn prepare_first_step<'a>(
        &'a self,
        storage: &'a mut PolysStorage<F, E>,
    ) -> Self::FirstStep<'a> {
        todo!()
    }
    fn finish_first_step(&self, storage: &mut PolysStorage<F, E>) {
        todo!()
    }

    type NextSteps<'a> = SameSizeProductEvaluationKernelFirstStep<'a, F, E>;
    fn prepare_next_steps<'a>(
        &'a self,
        storage: &'a mut PolysStorage<F, E>,
        step: usize,
    ) -> Self::NextSteps<'a> {
        todo!()
    }
}

pub struct SameSizeProductEvaluationKernelFirstStep<
    'a,
    F: PrimeField,
    E: FieldExtension<F> + PrimeField,
> {
    // TODO: if we will also add an access to inputs themselves, then we can avoid recomputing
    // f0_l * f0_r and f1_l * f1_r below
    lhs: FirstStepInExtensionLazyFolder<'a, E>,
    rhs: FirstStepInExtensionLazyFolder<'a, E>,
    _marker: core::marker::PhantomData<F>,
}

impl<'a, F: PrimeField, E: FieldExtension<F> + PrimeField> BatchGKREvaluationKernel<F, E>
    for SameSizeProductEvaluationKernelFirstStep<'a, F, E>
{
    fn evaluate_in_batched_mode(
        &mut self,
        batch_gkr_challenge: &E,
        accumulation_buffer: &mut [[E; 3]], // max quadratic
        num_steps: usize,
        index_offset: usize,
    ) {
        let mut index = index_offset;
        for i in 0..num_steps {
            // TODO: use access to original polys to avoid re-computation of multiplication,
            // and only access via self.lhs.get_intermediate_only(index);
            let (f0_l, f1_l, t_l) = self.lhs.get_triple(index);
            let (f0_r, f1_r, t_r) = self.rhs.get_triple(index);

            // TODO: consider buffering them too

            let mut f_at_minus_one_l = f0_l;
            f_at_minus_one_l.sub_assign(&t_l);

            let mut f_at_minus_one_r = f0_r;
            f_at_minus_one_r.sub_assign(&t_r);

            let dst = &mut accumulation_buffer[i];

            for (idx, (a, b)) in [
                (f0_l, f0_r),
                (f1_l, f1_r),
                (f_at_minus_one_l, f_at_minus_one_r),
            ]
            .into_iter()
            .enumerate()
            {
                let mut eval = a;
                eval.mul_assign(&b);
                eval.mul_assign(batch_gkr_challenge);
                dst[idx].add_assign(&eval);
            }

            index += 1;
        }

        // and outer caller can multiply the sumcheck buffer by evaluations of equality poly and sum them up
    }
}
