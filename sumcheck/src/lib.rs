pub mod batched;
pub mod dimension_reduction;
pub mod mle_eval;
pub mod mle_product;

use field::{Field, FieldExtension};
use poly::UniPoly;

pub trait SumcheckInstanceProver<F: Field, E: FieldExtension<F>> {
    fn degree(&self) -> usize;

    fn num_rounds(&self) -> usize;

    fn input_claim(&self) -> E;

    fn compute_message(&mut self, round: usize, previous_claim: &E) -> UniPoly<E>;

    fn ingest_challenge(&mut self, r: &E, round: usize);
}

pub trait SumcheckInstanceVerifier<F: Field, E: FieldExtension<F>> {
    fn degree(&self) -> usize;

    fn num_rounds(&self) -> usize;

    fn input_claim(&self) -> E;

    fn expected_output_claim(&self, sumcheck_challenges: &[E]) -> E;
}

