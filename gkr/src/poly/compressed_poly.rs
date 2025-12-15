use field::{Field, FieldExtension};
use crate::{poly::AppendToTranscript, transcript::Transcript};

use super::univariate::UniPoly;

/// Univariate polynomial without the linear term
#[derive(Clone, Debug)]
pub struct CommpressedPoly<T> {
    coeffs_except_linear_term: Vec<T>,
}

impl<F: Field> CommpressedPoly<F> {
    pub fn new(coeffs_except_linear_term: Vec<F>) -> Self {
        Self { coeffs_except_linear_term }
    }

    /// `hint = p(0) + p(1) = c0 + c0 + c1 + c2 + ...` so we can solve for `c1`
    pub fn decompress(self, hint: &F) -> UniPoly<F> {
        let mut linear_term = *hint;
        linear_term.sub_assign(&self.coeffs_except_linear_term[0]);
        linear_term.sub_assign(&self.coeffs_except_linear_term[0]);

        for i in 1..self.coeffs_except_linear_term.len() {
            linear_term.sub_assign(&self.coeffs_except_linear_term[i]);
        }

        let mut coeffs = vec![self.coeffs_except_linear_term[0], linear_term];
        coeffs.extend(&self.coeffs_except_linear_term[1..]);
        assert_eq!(self.coeffs_except_linear_term.len() + 1, coeffs.len());
        UniPoly::new(coeffs)
    }

    pub fn eval_from_hint(&self, hint: &F, x: &F) -> F {
        let mut linear_term = *hint;
        linear_term.sub_assign(&self.coeffs_except_linear_term[0]);
        linear_term.sub_assign(&self.coeffs_except_linear_term[0]);

        for i in 1..self.coeffs_except_linear_term.len() {
            linear_term.sub_assign(&self.coeffs_except_linear_term[i]);
        }

        let mut running_point = *x;
        let mut running_sum = *x;
        running_sum.mul_assign(&linear_term);
        running_sum.add_assign(&self.coeffs_except_linear_term[0]);
        for i in 1..self.coeffs_except_linear_term.len() {
            running_point.mul_assign(x);
            let mut tmp = self.coeffs_except_linear_term[i];
            tmp.mul_assign(&running_point);
            running_sum.add_assign(&tmp);
        }

        running_sum
    }

    pub fn degree(&self) -> usize {
        self.coeffs_except_linear_term.len()
    }
}

impl<F, E, T> AppendToTranscript<F, E, T> for CommpressedPoly<E> 
where
    F: Field,
    E: FieldExtension<F> + Field,
    T: Transcript<F, Extension = E>
{
    fn append_to_transcript(&self, transcript: &mut T) {
        transcript.absorb_extensions(&self.coeffs_except_linear_term);
    }
}