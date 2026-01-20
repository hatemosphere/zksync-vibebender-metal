use std::marker::PhantomData;

use field::{Field, FieldExtension};
use poly::UniPoly;

use crate::SumcheckInstanceProver;

/// Sumcheck for proving sum_x eq(r, x) * p(x, 0) * p(x, 1)
pub struct DimensionReductionSumcheck<F, E>(PhantomData<(F, E)>);

impl<F: Field, E: FieldExtension<F> + Field> DimensionReductionSumcheck<F, E> {
}

impl<F: Field, E: FieldExtension<F> + Field> SumcheckInstanceProver<F, E> for DimensionReductionSumcheck<F, E> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        todo!()
    }

    fn input_claim(&self) -> E {
        todo!()
    }

    fn compute_message(&mut self, _round: usize, previous_claim: &E) -> UniPoly<E> {
        todo!()
    }

    fn ingest_challenge(&mut self, r: &E, _round: usize) {
        todo!()
    }
}
