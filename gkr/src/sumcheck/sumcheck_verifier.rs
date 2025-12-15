use field::{Field, FieldExtension};

pub trait SumcheckInstanceVerifier<F: Field> {
    type E: FieldExtension<F> + Field;

    fn degree(&self) -> usize;

    fn num_rounds(&self) -> usize;

    fn input_claim(&self) -> F;

    fn expected_output_claim(&self, sumcheck_challenges: &[Self::E]) -> Self::E;
}
