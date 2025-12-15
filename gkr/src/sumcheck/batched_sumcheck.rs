use std::marker::PhantomData;

use crate::{
    poly::{AppendToTranscript, CommpressedPoly, Group, Polynomial, UniPoly},
    sumcheck::{sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier},
    transcript::Transcript,
};
use field::{Field, FieldExtension};

#[derive(Clone)]
pub struct SumcheckInstanceProof<F, E> {
    compressed_polys: Vec<CommpressedPoly<E>>,
    _marker: PhantomData<F>,
}

impl<F, E> SumcheckInstanceProof<F, E> {
    pub fn new(compressed_polys: Vec<CommpressedPoly<E>>) -> Self {
        Self {
            compressed_polys,
            _marker: PhantomData,
        }
    }
}

impl<F: Field, E: FieldExtension<F> + Field> SumcheckInstanceProof<F, E> {
    pub fn verify(
        &self,
        claim: E,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut impl Transcript<F, Extension = E>,
    ) -> Result<(E, Vec<E>), ()> {
        let mut e = claim;
        let mut r = Vec::with_capacity(num_rounds);

        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..num_rounds {
            if self.compressed_polys[i].degree() > degree_bound {
                return Err(());
            }

            self.compressed_polys[i].append_to_transcript(transcript);
            let r_i = transcript.draw_challenge_extension();
            r.push(r_i);

            e = self.compressed_polys[i].eval_from_hint(&e, &r_i);
        }

        Ok((e, r))
    }
}

/// Front loaded batched sumcheck (https://hackmd.io/s/HyxaupAAA)
pub fn prove<F: Field, E: FieldExtension<F> + Field, T: Transcript<F, Extension = E>>(
    mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, E = E>>,
    transcript: &mut T,
) -> (SumcheckInstanceProof<F, E>, Vec<E>) {
    let max_num_rounds = sumcheck_instances
        .iter()
        .map(|sumcheck| sumcheck.num_rounds())
        .max()
        .expect("Cannot prove an empty list of instances");
    let batching_coeffs = transcript.draw_challenges_extension(sumcheck_instances.len());

    let mut individual_claims: Vec<E> = sumcheck_instances
        .iter()
        .map(|sumcheck| {
            let num_rounds = sumcheck.num_rounds();
            let input_claim = sumcheck.input_claim();
            transcript.absorb_extension(&input_claim);

            mul_pow_two(input_claim, max_num_rounds - num_rounds)
        })
        .collect();

    let mut r_sumcheck = Vec::with_capacity(max_num_rounds);
    let mut compressed_polys = Vec::with_capacity(max_num_rounds);

    for round in 0..max_num_rounds {
        let remaining_rounds = max_num_rounds - round;

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
                    sumcheck.compute_message(round - offset, *previous_claim)
                }
            })
            .collect();

        let batched_uni_poly: UniPoly<E> = uni_polys
            .iter()
            .zip(&batching_coeffs)
            .fold(UniPoly::<E>::zero(), |batched, (poly, coeff)| {
                batched + poly.clone() * coeff
            });

        let compressed_poly = batched_uni_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        let r_j= transcript.draw_challenge_extension();
        r_sumcheck.push(r_j);

        individual_claims
            .iter_mut()
            .zip(uni_polys.into_iter())
            .for_each(|(claim, poly)| *claim = poly.eval_at(&[r_j]));

        for sumcheck in sumcheck_instances.iter_mut() {
            if remaining_rounds <= sumcheck.num_rounds() {
                let offset = max_num_rounds - sumcheck.num_rounds();
                sumcheck.ingest_challenge(r_j, round - offset);
            }
        }

        compressed_polys.push(compressed_poly);
    }

    (SumcheckInstanceProof::new(compressed_polys), r_sumcheck)
}

pub fn verify<F: Field, E: FieldExtension<F> + Field, T: Transcript<F, Extension = E>>(
    proof: SumcheckInstanceProof<F, E>,
    sumcheck_instaces: Vec<&dyn SumcheckInstanceVerifier<F, E = E>>,
    transcript: &mut T,
) -> Result<Vec<E>, ()> {
    let max_degree = sumcheck_instaces
        .iter()
        .map(|sumcheck| sumcheck.degree())
        .max()
        .unwrap();
    let max_num_rounds = sumcheck_instaces
        .iter()
        .map(|sumcheck| sumcheck.num_rounds())
        .max()
        .unwrap();

    let batching_coeffs = transcript.draw_challenges_extension(sumcheck_instaces.len());

    let claim = sumcheck_instaces
        .iter()
        .zip(batching_coeffs.iter())
        .map(|(sumcheck, coeff)| {
            let num_rounds = sumcheck.num_rounds();
            let input_claim = sumcheck.input_claim();
            transcript.absorb_base(&input_claim);

            let mut claim = E::from_base(mul_pow_two(input_claim, max_num_rounds - num_rounds));
            claim.mul_assign(coeff);
            claim
        })
        .fold(E::ZERO, |mut acc, val| {
            acc.add_assign(&val);
            acc
        });

    let (output_claim, r_sumcheck) = proof.verify(claim, max_num_rounds, max_degree, transcript)?;

    let expected_output_claim = sumcheck_instaces
        .iter()
        .zip(batching_coeffs.iter())
        .map(|(sumcheck, coeff)| {
            let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];
            let mut claim = sumcheck.expected_output_claim(r_slice);

            claim.mul_assign(coeff);
            claim
        })
        .fold(E::ZERO, |mut acc, val| {
            acc.add_assign(&val);
            acc
        });

    if output_claim != expected_output_claim {
        return Err(());
    }

    Ok(r_sumcheck)
}

fn mul_pow_two<F: Field>(mut a: F, pow: usize) -> F {
    for _ in 0..pow {
        a.double();
    }

    a
}
