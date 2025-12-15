use std::marker::PhantomData;

use field::{BaseField, Field, FieldExtension};

use crate::{
    poly::{MultilinearPoly, Polynomial, SumcheckPoly, UniPoly},
    sumcheck::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
};

pub struct MultiLinearSumcheckProver<F, E> {
    poly: MultilinearPoly<E>,
    challenges: Vec<E>,
    claimed_sum: E,
    num_rounds: usize,
    _marker: PhantomData<F>,
}

impl<F: BaseField, E: FieldExtension<F> + Field> MultiLinearSumcheckProver<F, E> {
    pub fn new(poly: MultilinearPoly<E>) -> Self {
        let claimed_sum = poly.sum_over_hypercube();
        let num_rounds = poly.num_vars();

        Self {
            poly: poly,
            challenges: Vec::new(),
            claimed_sum,
            num_rounds,
            _marker: PhantomData,
        }
    }
}

impl<F: BaseField, E: FieldExtension<F> + Field> SumcheckInstanceProver<F>
    for MultiLinearSumcheckProver<F, E>
{
    type E = E;

    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self) -> Self::E {
        self.claimed_sum
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let g0 = self.poly.partial_eval(&[E::ZERO]).sum_over_hypercube();
        let mut g1_minus_g0 = self.poly.partial_eval(&[E::ONE]).sum_over_hypercube();
        g1_minus_g0.sub_assign(&g0);

        UniPoly::new(vec![g0, g1_minus_g0])
    }

    fn ingest_challenge(&mut self, r_j: E, _round: usize) {
        self.poly = self.poly.partial_eval(&[r_j]);
        self.challenges.push(r_j);
    }
}

pub struct MultiLinearSumcheckVerifier<F, E> {
    claimed_sum: F,
    num_vars: usize,
    poly: MultilinearPoly<F>,
    _marker: PhantomData<E>
}

impl<F, E> MultiLinearSumcheckVerifier<F, E> {
    pub fn new(claimed_sum: F, num_vars: usize, poly: MultilinearPoly<F>) -> Self {
        Self {
            claimed_sum,
            num_vars,
            poly,
            _marker: PhantomData
        }
    }
}

impl<F: Field, E: FieldExtension<F> + Field> SumcheckInstanceVerifier<F>
    for MultiLinearSumcheckVerifier<F, E>
{
    type E = E;

    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn input_claim(&self) -> F {
        self.claimed_sum
    }

    fn expected_output_claim(&self, sumcheck_challenges: &[E]) -> E {
        // TODO: for now we just have the verifier evaluate the polynomial
        self.poly.clone().lift().eval_at(sumcheck_challenges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::{Mersenne31Complex, Mersenne31Field};
    use proptest::{prop_assert_eq, proptest};

    type F = Mersenne31Field;
    type E = Mersenne31Complex;

    #[test]
    fn test_full_sumcheck() {
        proptest!(|(evals: Vec<u32>)| {
            let evals = evals.into_iter().map(F::from_nonreduced_u32).collect();
            let poly = MultilinearPoly::new(evals).lift();

            let mut prover = MultiLinearSumcheckProver::<F, E>::new(poly.clone());
            let claim = poly.sum_over_hypercube();
            let num_rounds = prover.num_rounds();

            prop_assert_eq!(claim, prover.input_claim());

            let mut claim = E::from_base(claim);
            let mut challenges = Vec::new();

            for round in 0..num_rounds {
                let msg = prover.compute_message(round, claim);

                let g0 = msg.eval_at(&[E::ZERO]);
                let g1 = msg.eval_at(&[E::ONE]);
                let mut sum = g0;
                sum.add_assign(&g1);

                prop_assert_eq!(sum, claim);

                let r = E::from_base(F::from_nonreduced_u32(round as u32 + 1));
                claim = msg.eval_at(&[r]);
                prover.ingest_challenge(r, round);
                challenges.push(r);
            }

            let expected = poly.lift::<E>().eval_at(&challenges);
            prop_assert_eq!(expected, claim);
        });
    }
}
