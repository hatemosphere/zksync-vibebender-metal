use std::marker::PhantomData;

use field::{Field, FieldExtension};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    poly::{MultilinearPoly, Polynomial, SumcheckPoly, UniPoly},
    sumcheck::SumcheckInstanceProver,
    circuit::CircuitLayer,
};

/// Proves the sum of selector(z, b, c) * W(b) * W(c)
pub struct LayerSumcheckProver<F, E> {
    _marker: PhantomData<F>,
    selector: MultilinearPoly<E>,
    wiring: MultilinearPoly<E>,
    challenges: Vec<E>,
    claim: E,
    num_rounds: usize,
}

impl<F: Field, E: FieldExtension<F> + Field> LayerSumcheckProver<F, E> {
    pub fn new(
        z: Vec<E>,
        layer: CircuitLayer<F>,
        prev_layer_values: MultilinearPoly<F>,
        claim: E,
    ) -> Self {
        let selector = layer
            .selector(prev_layer_values.num_vars())
            .lift()
            .partial_eval(&z);
        let num_rounds = 2 * prev_layer_values.num_vars();

        // w(b) * w(c)
        let wiring = prev_layer_values.tensor(&prev_layer_values).lift();

        debug_assert_eq!(selector.num_vars(), wiring.num_vars());

        Self {
            _marker: PhantomData,
            selector,
            wiring,
            challenges: Vec::new(),
            claim,
            num_rounds,
        }
    }
}

impl<F: Field, E: FieldExtension<F> + Field> SumcheckInstanceProver<F>
    for LayerSumcheckProver<F, E>
{
    type E = E;

    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self) -> Self::E {
        self.claim
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: Self::E) -> UniPoly<Self::E> {
        let mut evals = Vec::with_capacity(3);
        let remaining_vars = self.wiring.num_vars() - 1;

        let mut neg_one = E::ONE;
        neg_one.negate();
        // sum selector(z, x, a) * wiring(x, a) over the hypercube
        for x in [E::ZERO, E::ONE, neg_one] {
            evals.push(
                (0..1 << remaining_vars)
                    .into_par_iter()
                    .map(|i| {
                        (0..remaining_vars)
                            .rev()
                            .map(|j| if (i >> j) & 1 == 1 { E::ONE } else { E::ZERO })
                            .collect::<Vec<E>>()
                    })
                    .map(|mut a| {
                        a.insert(0, x);
                        let mut res = self.selector.eval_at(&a);
                        res.mul_assign(&self.wiring.eval_at(&a));
                        res
                    })
                    .reduce(
                        || E::ZERO,
                        |mut acc, val| {
                            acc.add_assign(&val);
                            acc
                        },
                    ),
            );
        }

        // and now we interpolate to get the coefficients
        // f(x)= ((c1+c2)/2 -c0)* x^2 + (c1-c2)/2 * x + c0
        let c = evals[0];

        let mut half = E::ONE;
        half.double();
        half = half.inverse().expect("two should be invertible");

        let mut a = evals[1];
        a.add_assign(&evals[2]);
        a.mul_assign(&half);
        a.sub_assign(&c);

        let mut b = evals[1];
        b.sub_assign(&evals[2]);
        b.mul_assign(&half);

        let round_poly = UniPoly::new(vec![c, b, a]);

        debug_assert_eq!(round_poly.eval_at(&[E::ZERO]), evals[0]);
        debug_assert_eq!(round_poly.eval_at(&[E::ONE]), evals[1]);
        debug_assert_eq!(round_poly.eval_at(&[neg_one]), evals[2]);

        round_poly
    }

    fn ingest_challenge(&mut self, r_j: E, _round: usize) {
        self.selector = self.selector.partial_eval(&[r_j]);
        self.wiring = self.wiring.partial_eval(&[r_j]);

        self.challenges.push(r_j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{QuadGate, QuadTerm};
    use field::{Mersenne31Complex as E, Mersenne31Field as F};

    #[test]
    fn test_layer_sumcheck_basic() {
        // Simple circuit: single gate computing wire[0] * wire[1]
        let gates = vec![QuadGate::new(vec![QuadTerm::new(F::ONE, 0, 1)])];
        let layer = CircuitLayer::new(gates).unwrap();

        // Wire values: [2, 3]
        let wire_vals =
            MultilinearPoly::new(vec![F::from_nonreduced_u32(2), F::from_nonreduced_u32(3)]);

        // Evaluate at gate 0 (single gate => no gate variables => empty z)
        let z: Vec<E> = vec![];

        // Expected: 2 * 3 = 6
        let claim = E::from_base(F::from_nonreduced_u32(6));

        let mut prover = LayerSumcheckProver::new(z, layer, wire_vals, claim);

        assert_eq!(prover.num_rounds(), 2); // 2 wire vars (1 for b, 1 for c)
        assert_eq!(prover.degree(), 2);
        assert_eq!(prover.input_claim(), claim);

        // Test first round consistency
        let uni_poly = prover.compute_message(0, claim);
        let g_0 = uni_poly.eval_at(&[E::ZERO]);
        let g_1 = uni_poly.eval_at(&[E::ONE]);
        let mut sum = g_0;
        sum.add_assign(&g_1);
        assert_eq!(sum, claim, "Round 0: g(0) + g(1) should equal claim");
    }

    #[test]
    fn test_layer_sumcheck_with_coefficients() {
        // Gate with coefficient: 5 * wire[0] * wire[1]
        let gates = vec![QuadGate::new(vec![QuadTerm::new(
            F::from_nonreduced_u32(5),
            0,
            1,
        )])];
        let layer = CircuitLayer::new(gates).unwrap();

        let wire_vals =
            MultilinearPoly::new(vec![F::from_nonreduced_u32(3), F::from_nonreduced_u32(7)]);

        let z: Vec<E> = vec![];
        let claim = E::from_base(F::from_nonreduced_u32(5 * 3 * 7)); // 105

        let mut prover = LayerSumcheckProver::new(z, layer, wire_vals, claim);

        // Verify first round
        let uni_poly = prover.compute_message(0, claim);
        let g_0 = uni_poly.eval_at(&[E::ZERO]);
        let g_1 = uni_poly.eval_at(&[E::ONE]);
        let mut sum = g_0;
        sum.add_assign(&g_1);

        assert_eq!(sum, claim);
    }

    #[test]
    fn test_layer_sumcheck_multiple_terms() {
        // Gate with multiple terms: 2*w[0]*w[1] + 3*w[0]*w[2]
        let gates = vec![QuadGate::new(vec![
            QuadTerm::new(F::from_nonreduced_u32(2), 0, 1),
            QuadTerm::new(F::from_nonreduced_u32(3), 0, 2),
        ])];
        let layer = CircuitLayer::new(gates).unwrap();

        // Wire values: [2, 3, 5, 0] (padding to power of 2)
        let wire_vals = MultilinearPoly::new(vec![
            F::from_nonreduced_u32(2),
            F::from_nonreduced_u32(3),
            F::from_nonreduced_u32(5),
            F::ZERO,
        ]);

        let z: Vec<E> = vec![];
        // Expected: 2*2*3 + 3*2*5 = 12 + 30 = 42
        let claim = E::from_base(F::from_nonreduced_u32(42));

        let mut prover = LayerSumcheckProver::new(z, layer, wire_vals, claim);

        // Run one round to verify
        let uni_poly = prover.compute_message(0, claim);
        let g_0 = uni_poly.eval_at(&[E::ZERO]);
        let g_1 = uni_poly.eval_at(&[E::ONE]);
        let mut sum = g_0;
        sum.add_assign(&g_1);

        assert_eq!(sum, claim);
    }

    #[test]
    fn test_layer_sumcheck_gate_selection() {
        // 4 gates, verify we can select specific gates with z
        let gates = vec![
            QuadGate::new(vec![QuadTerm::new(F::from_nonreduced_u32(1), 0, 1)]),
            QuadGate::new(vec![QuadTerm::new(F::from_nonreduced_u32(2), 0, 1)]),
            QuadGate::new(vec![QuadTerm::new(F::from_nonreduced_u32(3), 0, 1)]),
            QuadGate::new(vec![QuadTerm::new(F::from_nonreduced_u32(4), 0, 1)]),
        ];
        let layer = CircuitLayer::new(gates).unwrap();

        let wire_vals =
            MultilinearPoly::new(vec![F::from_nonreduced_u32(5), F::from_nonreduced_u32(7)]);

        // Select gate 2 (index 2 = binary [1, 0] with our bit ordering)
        let z = vec![E::ONE, E::ZERO];

        // Gate 2 has coefficient 3, so: 3 * 5 * 7 = 105
        let claim = E::from_base(F::from_nonreduced_u32(105));

        let mut prover = LayerSumcheckProver::new(z, layer, wire_vals, claim);

        let uni_poly = prover.compute_message(0, claim);
        let g_0 = uni_poly.eval_at(&[E::ZERO]);
        let g_1 = uni_poly.eval_at(&[E::ONE]);
        let mut sum = g_0;
        sum.add_assign(&g_1);

        assert_eq!(sum, claim);
    }

    #[test]
    fn test_layer_sumcheck_extension_field_challenges() {
        // Test with complex extension field challenges
        let gates = vec![QuadGate::new(vec![QuadTerm::new(F::ONE, 0, 1)])];
        let layer = CircuitLayer::new(gates).unwrap();

        let wire_vals =
            MultilinearPoly::new(vec![F::from_nonreduced_u32(3), F::from_nonreduced_u32(5)]);

        let z: Vec<E> = vec![];
        let claim = E::from_base(F::from_nonreduced_u32(15));

        let mut prover = LayerSumcheckProver::new(z, layer.clone(), wire_vals.clone(), claim);

        // Round 0 with complex challenge
        let complex_challenge = E::new(F::from_nonreduced_u32(13), F::from_nonreduced_u32(17));

        let uni_poly = prover.compute_message(0, claim);

        // Check round 0 consistency
        let g0_0 = uni_poly.eval_at(&[E::ZERO]);
        let g0_1 = uni_poly.eval_at(&[E::ONE]);
        let mut sum_0 = g0_0;
        sum_0.add_assign(&g0_1);
        assert_eq!(sum_0, claim, "Round 0 should be consistent");

        let next_claim = uni_poly.eval_at(&[complex_challenge]);

        prover.ingest_challenge(complex_challenge, 0);

        // Round 1
        let uni_poly_2 = prover.compute_message(1, next_claim);

        let g_0 = uni_poly_2.eval_at(&[E::ZERO]);
        let g_1 = uni_poly_2.eval_at(&[E::ONE]);

        let mut sum = g_0;
        sum.add_assign(&g_1);

        assert_eq!(sum, next_claim, "Round 1: sum check failed");
    }

    #[test]
    fn test_layer_sumcheck_zero_gate() {
        // Gate that evaluates to zero
        let gates = vec![QuadGate::new(vec![QuadTerm::new(F::ZERO, 0, 1)])];
        let layer = CircuitLayer::new(gates).unwrap();

        let wire_vals =
            MultilinearPoly::new(vec![F::from_nonreduced_u32(3), F::from_nonreduced_u32(5)]);

        let z: Vec<E> = vec![];
        let claim = E::ZERO;

        let mut prover = LayerSumcheckProver::new(z, layer, wire_vals, claim);

        let uni_poly = prover.compute_message(0, claim);

        // All evaluations should be zero
        assert_eq!(uni_poly.eval_at(&[E::ZERO]), E::ZERO);
        assert_eq!(uni_poly.eval_at(&[E::ONE]), E::ZERO);
    }
}
