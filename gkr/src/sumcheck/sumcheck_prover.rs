use crate::poly::UniPoly;
use field::{Field, FieldExtension};

pub trait SumcheckInstanceProver<F: Field> {
    type E: FieldExtension<F> + Field;

    fn degree(&self) -> usize;

    fn num_rounds(&self) -> usize;

    fn input_claim(&self) -> Self::E;

    fn compute_message(&mut self, round: usize, previous_claim: Self::E) -> UniPoly<Self::E>;

    fn ingest_challenge(&mut self, r_j: Self::E, round: usize);
}
