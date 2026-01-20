use field::{Field, FieldExtension};
use poly::UniPoly;

use crate::SumcheckInstanceProver;

pub fn prove_batched<F: Field, E: FieldExtension<F> + Field>(
    sumcheck_instances: &mut [&mut dyn SumcheckInstanceProver<F, E>],
    batching_coeffs: &[E],
) -> (Vec<UniPoly<E>>, Vec<E>) {
    assert_eq!(
        sumcheck_instances.len(),
        batching_coeffs.len(),
        "Number of batching coefficients must match number of instances"
    );

    let max_num_rounds = sumcheck_instances
        .iter()
        .map(|sumcheck| sumcheck.num_rounds())
        .max()
        .expect("Cannot prove an empty list of instances");

    let mut individual_claims: Vec<E> = sumcheck_instances
        .iter()
        .map(|sumcheck| {
            let num_rounds = sumcheck.num_rounds();
            let input_claim = sumcheck.input_claim();

            mul_pow_two(input_claim, max_num_rounds - num_rounds)
        })
        .collect();

    let mut r_sumcheck = Vec::with_capacity(max_num_rounds);
    let mut round_polys = Vec::with_capacity(max_num_rounds);

    for round in 0..max_num_rounds {
        let remaining_rounds = max_num_rounds - round;

        // Compute round polynomial for each instance
        let uni_polys: Vec<UniPoly<E>> = sumcheck_instances
            .iter_mut()
            .zip(individual_claims.iter())
            .map(|(sumcheck, previous_claim)| {
                let num_rounds = sumcheck.num_rounds();

                if remaining_rounds > num_rounds {
                    let scaled_claim =
                        mul_pow_two(sumcheck.input_claim(), remaining_rounds - num_rounds - 1);
                    UniPoly::<E>::new(vec![scaled_claim])
                } else {
                    let offset = max_num_rounds - num_rounds;
                    sumcheck.compute_message(round - offset, previous_claim)
                }
            })
            .collect();

        // Batch the polynomials using random linear combination
        let batched_uni_poly: UniPoly<E> = uni_polys
            .iter()
            .zip(batching_coeffs)
            .fold(UniPoly::<E>::zero(), |batched, (poly, coeff)| {
                batched + poly.clone() * coeff
            });

        // TODO: add batched_uni_poly to transcript and draw challenge
        let r_j = E::ONE;

        r_sumcheck.push(r_j);

        // Update individual claims by evaluating each polynomial at r_j
        individual_claims
            .iter_mut()
            .zip(uni_polys.into_iter())
            .for_each(|(claim, poly)| *claim = poly.eval(&r_j));

        // Ingest challenge into each active instance
        for sumcheck in sumcheck_instances.iter_mut() {
            if remaining_rounds <= sumcheck.num_rounds() {
                let offset = max_num_rounds - sumcheck.num_rounds();
                sumcheck.ingest_challenge(&r_j, round - offset);
            }
        }

        round_polys.push(batched_uni_poly);
    }

    (round_polys, r_sumcheck)
}

fn mul_pow_two<F: Field>(mut a: F, pow: usize) -> F {
    for _ in 0..pow {
        a.double();
    }
    a
}
