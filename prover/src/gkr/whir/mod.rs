// The original paper is overly complicated in it's notations, so here is a description.
// We will use capital letter for univariate polys, and small one for multivatiate, and same letter
// of different capitalization is just reinterpretation of one for another
// - Prover starts with oracle of evaluations F0 of the original poly F(X) at some smooth domain L0
// - also we assume that we have an original claim that F(Y) = Z, that can also we rewritten as sumcheck claim
// F(Y) = Z = f(y^0, y^1, y^2, ...) = \sum_{x} eq(x, y^0, y^1, y^2, ...) f(x) - our original sumcheck claim.
// If we sum over all the {x} in the right-hand side, but one, we can view it as a univariate f0(Y), and f0(0) + f0(1) == Z -
// all the standard sumcheck staff
// - Note that in the same manner we can express in-domain value F(omega^k) = \sum_{x} eq(x, omega^k decomposition over powers) f(x)
// - Prover and verifier can engage in more than 1 sumcheck steps (here the tradeoff is less steps later, but more accesses to F0 oracle)
// ---- Steps below are recursive, but we only use indexes 0/1 for clarity. Each step NUM_QUERIES also differs
// - At this moment we would have something like
// claim_0 = \sum_{x/folded coordiantes} eq(r1, r2, r3, x4, x5, ... y^0, y^1, y^2, y^4, ...) f(r1, r2, r3, x4, x5, ...)
// - Now prover sends an oracle F1 to f1(x4, x5, ...) = f(r1, r2, r3, x4, x5, ...) at domain L1. Note that "degree" of f1(x4, x5, ...)
// is smaller that of original f(x), but prover can decrease the rate for further iterations of the protocol
// - As in STIR, we want to perform out of domain sampling. So, we draw OOD point y1 and prover sends evaluation of f1(y1^0, y1^1, ...) = z1
// - Now prover also samples NUM_QUERIES indexes in the 3 (in our example) times folded image of L0. Those indexes trivially map
// into the |L0|/2^3 roots of unity. We will use notations Q_i for such indexes and corresponding roots of unity interchangeably
// - As in FRI, verifier has oracle access to f1(Q_i) by accessing 2^3 corresponding elements in F0 (at L0) and folding them.
// - We denote those values as G_i and in the original paper we do not need those values from prover YET, and instead they update our sumcheck claim formally at first,
// but it doesn't affect the protocol, and we will show that verification can be performed right away
// - start with the old one (prefactors aside)
// claim_0 = \sum_{x} eq(x, y^4, y^8, ...) f1(x)
// - add a contribution about f1(y1) = z1
// claim_0 + gamma^1 * z1 = \sum_{x} eq(x, y^4, y^8, ...) f1(x) + gamma^1 * \sum_{x} eq(x, y1^0, y1^1, ...) f1(x)
// - add NUM_QUERIES contribution about Q_i
// claim_0 + gamma^1 * z1 + \sum_{i = 0..NUM_QUERIES} gamma^{i + 1} G_i =
// = \sum_{x} eq(x, y^4, y^8, ...) f1(x) +
// + gamma^1 * \sum_{x} eq(x, y1^0, y1^1, ...) f1(x) +
// + \sum_{i = 0..NUM_QUERIES} gamma^{1+i} * \sum_{x} eq(x, Q_i) f1(x)
// - Those terms re-arrange nicely over f1(x)
// - To continue the sumcheck prover would send some univariate poly f2(Y), but as usual
// f2(0) + f2(1) == claim_0 + gamma^1 * z1 + \sum_{i = 0..NUM_QUERIES} gamma^{i + 1} G_i
// and verifier already has all the values to perform this check and forget about anything that happened before:
// - claim_0 comes from the previous work
// - z1 was sent by the prover
// - G_i are available via oracle access to F0 at L0 (in our example verifier needs 8 elements to fold 3 times and get those values)
// ---- Steps above are recursive until f_i(x) becomes "small-ish"
// - prover and verifier can engate in folding f1(x) few times until it becomes "small"
// - prover sends explicit form of the corresponding f_final(x)
// - same as above, we choose NUM_QUERIES_FINAL indexes, access previous step's oracle to get NUM_QUERIES_FINAL f1(x) values
// - Those values are checked to correspond to the explicit f_final(x) form
// - evaluate the last sumcheck explicitly
// - Due to complexity of such sumcheck (that drags various prefactors from previous rounds), most likely size of the final polynomial
// should be very small (much smaller compared to FRI case)

use crate::gkr::prover::stages::stage1::{
    compute_column_major_lde_from_main_domain_and_output_monomial_form,
    ColumnMajorCosetBoundTracePart,
};
use crate::gkr::sumcheck::*;
use crate::gkr::sumcheck::eq_poly::{make_eq_poly_in_full, make_eq_poly_reduced};
use crate::{gkr::prover::apply_row_wise, merkle_trees::ColumnMajorMerkleTreeConstructor};
use fft::{materialize_powers_serial_starting_with_one, Twiddles};
use field::{Field, FieldExtension, PrimeField, TwoAdicField};
use std::alloc::Global;
use worker::Worker;

pub struct ColumnMajorBaseOracleForCoset<
    F: PrimeField + TwoAdicField,
    T: ColumnMajorMerkleTreeConstructor<F>,
> {
    pub original_values_normal_order: Vec<ColumnMajorCosetBoundTracePart<F, F>>, // num_columns
    pub tree: T,
    pub values_per_leaf: usize,
    pub trace_len_log2: usize,
}

impl<F: PrimeField + TwoAdicField, T: ColumnMajorMerkleTreeConstructor<F>>
    ColumnMajorBaseOracleForCoset<F, T>
{
    pub fn query_for_folded_index(&self, index: usize) -> Vec<Vec<F>> {
        assert!(self.values_per_leaf.is_power_of_two());
        assert!(index < (1 << self.trace_len_log2) / self.values_per_leaf);

        todo!()
    }
}

pub struct ColumnMajorBaseOracleForLDE<
    F: PrimeField + TwoAdicField,
    T: ColumnMajorMerkleTreeConstructor<F>,
> {
    pub cosets: Vec<ColumnMajorBaseOracleForCoset<F, T>>,
}

impl<F: PrimeField + TwoAdicField, T: ColumnMajorMerkleTreeConstructor<F>>
    ColumnMajorBaseOracleForLDE<F, T>
{
    pub fn query_for_folded_index(&self, index: usize) -> Vec<Vec<F>> {
        let coset_index = index >> self.cosets[0].trace_len_log2;
        let internal_index = index & ((1 << self.cosets[0].trace_len_log2) - 1);
        self.cosets[coset_index].query_for_folded_index(index)
    }
}

pub fn whir_fold<
    F: PrimeField + TwoAdicField,
    E: FieldExtension<F> + Field,
    T: ColumnMajorMerkleTreeConstructor<F>,
>(
    base_layer_oracles: [ColumnMajorBaseOracleForLDE<F, T>; 3], // memory, witness, setup
    original_claims: [Vec<E>; 3], // memory, witness, setup
    original_evaluation_point: Vec<E>,
    original_lde_factor: usize,
    batching_challenge: &E,
    whir_steps_schedule: Vec<usize>,
    whir_queries_schedule: Vec<usize>,
    whir_steps_lde_factors: Vec<usize>,
    twiddles: &Twiddles<F, Global>,
    // TODO: LDE precomputations if needed

    // TODO: transcript
    trace_len_log2: usize,
    worker: &Worker,
) {
    assert!(original_lde_factor.is_power_of_two());
    assert_eq!(whir_steps_schedule.len(), whir_steps_lde_factors.len());

    // first compute batched poly. We do compute it on main domain only, and then FFT,
    // especially if we are going to offload cosets from the original commitment to disk instead of keeping in RAM

    let total_base_oracles = base_layer_oracles
        .iter()
        .map(|el| el.cosets[0].original_values_normal_order.len())
        .sum();
    assert_eq!(
        total_base_oracles,
        original_claims.iter().map(|el| el.len()).sum::<usize>()
    );
    for (a, b) in base_layer_oracles.iter().zip(original_claims.iter()) {
        assert_eq!(a.cosets[0].original_values_normal_order.len(), b.len());
    }

    let challenge_powers = materialize_powers_serial_starting_with_one::<E, Global>(
        *batching_challenge,
        total_base_oracles,
    );
    let challenge_powers = vec![
        E::ONE,
        E::ZERO,
        E::ZERO
    ];
    let (base_mem_powers, rest) = challenge_powers.split_at(original_claims[0].len());
    let (base_witness_powers, base_setup_powers) = rest.split_at(original_claims[1].len());
    assert_eq!(base_setup_powers.len(), original_claims[2].len());

    let batch_challenges = [
        base_mem_powers.to_vec(),
        base_witness_powers.to_vec(),
        base_setup_powers.to_vec(),
    ];

    let mut batched_poly_on_main_domain = vec![E::ZERO; 1 << trace_len_log2].into_boxed_slice();

    apply_row_wise::<F, E>(
        vec![],
        vec![&mut batched_poly_on_main_domain],
        1 << trace_len_log2,
        worker,
        |_, dest, chunk_start, chunk_size| {
            assert_eq!(dest.len(), 1);
            let mut dest = dest;
            let dest = dest.pop().unwrap();
            for (challenges_set, values_set) in [
                (
                    base_mem_powers,
                    &base_layer_oracles[0].cosets[0].original_values_normal_order,
                ),
                (
                    base_witness_powers,
                    &base_layer_oracles[1].cosets[0].original_values_normal_order,
                ),
                (
                    base_setup_powers,
                    &base_layer_oracles[2].cosets[0].original_values_normal_order,
                ),
            ] {
                for (batch_challenge, base_value) in challenges_set.iter().zip(values_set.iter()) {
                    let src = &base_value.column[..]; // main domain only
                    for i in 0..chunk_size {
                        let mut result = *batch_challenge;
                        result.mul_assign_by_base(&src[chunk_start + i]);
                        dest[i].add_assign(&result);
                    }
                }
            }
        },
    );

    // TODO: LDE and make oracles
    let (other_domains, mut monomial_form) =
        compute_column_major_lde_from_main_domain_and_output_monomial_form(
            &batched_poly_on_main_domain[..],
            twiddles,
            original_lde_factor,
        );

    let mut batched_claim = E::ZERO;
    for (challenges_set, values_set) in [base_mem_powers, base_witness_powers, base_setup_powers]
        .into_iter()
        .zip(original_claims.into_iter())
    {
        assert_eq!(challenges_set.len(), values_set.len());
        for (a, b) in challenges_set.iter().zip(values_set.into_iter()) {
            let mut result = b;
            result.mul_assign(&a);
            batched_claim.add_assign(&result);
        }
    }

    // our initial sumcheck claim is `batched_claim` = \sum_{hypercube} eq(x, `original_evaluation_point`) batched_poly(x)

    let num_rounds = whir_steps_schedule.len();
    // assert!(num_rounds > 2);
    let mut whir_steps_schedule = whir_steps_schedule.into_iter();
    let mut whir_queries_schedule = whir_queries_schedule.into_iter();
    let mut whir_steps_lde_factors = whir_steps_lde_factors.into_iter();
    let mut evaluation_coordinates = &original_evaluation_point[..];

    // as we will eventually continue to mix-in additional equality polys into sumcheck kernel,
    // we compute the largest one and update it each round
    let mut eq_polys = make_eq_poly_in_full::<F, E>(&original_evaluation_point[..]); // TODO: parallelize
    let mut eq_poly = eq_polys.pop().unwrap();
    drop(eq_polys);

    // initial folding
    let num_initial_folding_rounds = whir_steps_schedule.next().unwrap();
    let mut last_eq_poly_prefactor_contribution = E::ONE;
    let mut sumchecked_poly = batched_poly_on_main_domain.to_vec();
    dbg!(&sumchecked_poly);
    let mut claim = batched_claim;

    let mut folding_challenges = vec![];
    let mut ood_samples_per_round = vec![];
    let mut delinearization_challenges_per_round = vec![];

    {
        // compute univariate poly of degree 2, such that F(original_evaluation_point) == batched_claim.
        // As degree is here is always low, we only need to walk over half of the domain
        // F(X) = (a X + b) (c X + d), and F(0) + F(1) = batched_claim = bd + (a + b)(c + d)
        // We also know `a` and `b` from apriori decomposition of eq poly
        // eq = ((1 - X)*(1 - original_evaluation_point[0]) + X * original_evaluation_point[0])
        // so
        // `a` = (2 * original_evaluation_point[0] - 1)
        // `b` = (1 - original_evaluation_point[0])
        // other two are also easy
        // `c` = \sum_{x[1..]} eq(x[1..], original_evaluation_point[1..]) * (batched_poly(1, x[1..]) - batched_poly(0, x[1..]))
        // `d` = \sum_{x[1..]} eq(x[1..], original_evaluation_point[1..]) * batched_poly(0, x[1..])
        // last one is half-size in terms of memory reads

        // so we can compute `c` as (batched_claim - bd) / (a + b) - d, and completely avoid access to batched_poly(1, x[1..])

        // Same logic applies to all other sumcheck steps that we will have

        for el in base_layer_oracles.iter() {
            for el in el.cosets.iter() {
                assert_eq!(el.values_per_leaf, 1 << num_initial_folding_rounds);
            }
        }
        let mut folding_challenges_in_round = vec![];

        for _ in 0..num_initial_folding_rounds {
            let evaluation_point = evaluation_coordinates[0];
            evaluation_coordinates = &evaluation_coordinates[1..];
            let mut a = E::ONE;
            a.sub_assign(&evaluation_point);

            let mut b = evaluation_point;
            b.double();
            b.sub_assign_base(&F::ONE);

            let mut d = E::ZERO;

            // There is also an important aspect of mapping univariate and multivariate enumeration.
            // When we fold for proximity check, we always pair omega^k and -omega^k,
            // and so our coordinate should reflect this in enumeration. If we enumerate normally (1, omega, omega^2, ...),
            // and we treat our coordinate as MSB of the integer index, then we automatically get the behavior we want

            let half_cube = &sumchecked_poly[..sumchecked_poly.len() / 2];
            dbg!(half_cube);
            dbg!(&eq_poly);
            assert_eq!(eq_poly.len(), half_cube.len());
            for (a, b) in eq_poly.iter().zip(half_cube.iter()) {
                let mut t = *a;
                t.mul_assign(&b);
                d.add_assign(&t);
            }

            let mut a_plus_b = a;
            a_plus_b.add_assign(&b);
            let a_plus_b_inv = a_plus_b.inverse().expect("must exist");

            let mut bd = b;
            bd.mul_assign(&d);

            let mut normalized_claim = claim;
            normalized_claim.mul_assign(
                &last_eq_poly_prefactor_contribution
                    .inverse()
                    .expect("not zero"),
            );

            let mut c = normalized_claim;
            c.sub_assign(&bd);
            c.mul_assign(&a_plus_b_inv);
            c.sub_assign(&d);

            let c0 = bd;
            let mut ad = a;
            ad.mul_assign(&d);
            let mut bc = b;
            bc.mul_assign(&c);
            let mut c1 = ad;
            c1.add_assign(&bc);
            let mut c2 = a;
            c2.mul_assign(&c);

            let coeffs = [c0, c1, c2];

            {
                // self-check
                let s0 = evaluate_small_univariate_poly(&coeffs, &E::ZERO);
                let s1 = evaluate_small_univariate_poly(&coeffs, &E::ONE);
                let mut v = s0;
                v.add_assign(&s1);
                v.mul_assign(&last_eq_poly_prefactor_contribution);
                assert_eq!(v, claim);
            }

            // draw folding challenge
            let folding_challenge = E::from_base(F::from_u32_unchecked(42));
            folding_challenges_in_round.push(folding_challenge);
            {
                let t =
                    evaluate_eq_poly::<F, E>(&folding_challenge, &evaluation_point);
                last_eq_poly_prefactor_contribution = t;
            }
            let next_claim = evaluate_small_univariate_poly(&coeffs, &folding_challenge);
            claim = next_claim;
            // and fold the poly itself - both multivariate evals mapping, and monomial form

            todo!()

        }

        folding_challenges.push(folding_challenges_in_round);

        // draw OOD sample
        let ood_point = E::from_base(F::from_u32_unchecked(42));
        // compute OOD value
        let ood_value = E::ZERO;

        ood_samples_per_round.push((ood_point, ood_value));

        // now can draw challenges

        // and we can immediatelly query all the original oracles, and drop them. For that we need to draw indexes
        let num_bits = trace_len_log2 + (original_lde_factor.trailing_zeros() as usize) - num_initial_folding_rounds;
        let num_queries = whir_queries_schedule.next().unwrap();
        let mut query_indexes = vec![];
        for _ in 0..num_queries {
            query_indexes.push(0usize);
        }
        // and delinearization challenge
        let delinearization_challenge = E::from_base(F::from_u32_unchecked(7));
        delinearization_challenges_per_round.push(delinearization_challenge);
        let mut claim_correction = E::ZERO;
        {
            let mut t = ood_value;
            t.mul_assign(&delinearization_challenge);
            claim_correction.add_assign(&t);
        }
        let mut current_delinearization_challenge = delinearization_challenge;
        current_delinearization_challenge.square();
        for query_index in query_indexes.into_iter() {
            // get original leaf, compute batched, and then folded value
            let mut batched_evals = vec![E::ZERO; base_layer_oracles[0].cosets[0].values_per_leaf];
            for (oracle, batching_challenges) in base_layer_oracles.iter().zip(batch_challenges.iter()) {
                let leaf = oracle.query_for_folded_index(query_index);
                assert_eq!(batched_evals.len(), leaf.len());
                for (dst, src) in batched_evals.iter_mut().zip(leaf.iter()) {
                    assert_eq!(src.len(), batching_challenges.len());
                    for (a, b) in src.iter().zip(batching_challenges.iter()) {
                        let mut t = *b;
                        t.mul_assign_by_base(a);
                        dst.add_assign(&t);
                    }
                }
            }
            // and then fold

            // and add into sumcheck claim
        }

        drop(base_layer_oracles);
    }

    // now we step into recursive procesure,

    todo!()
}


#[cfg(test)]
mod test {
    use std::sync::Arc;

    use field::baby_bear::{base::BabyBearField, ext4::BabyBearExt4};
    use crate::{gkr::sumcheck::eq_poly::make_eq_poly_in_full, merkle_trees::blake2s_for_everything_tree::Blake2sU32MerkleTreeWithCap};
    use super::*;
    use crate::gkr::prover::stages::stage1::*;
    type F = BabyBearField;
    type E = BabyBearExt4;

    fn make_base_oracle(worker: &Worker) -> ColumnMajorBaseOracleForLDE<F, Blake2sU32MerkleTreeWithCap> {
        let main_domain = vec![
            F::from_u32_unchecked(1),
            F::from_u32_unchecked(2),
            F::from_u32_unchecked(3),
            F::from_u32_unchecked(4),
        ];
        let twiddles = Twiddles::<F, Global>::new(4, worker);
        let main_domain = Arc::new(main_domain.into_boxed_slice());

        let other_domains = compute_column_major_lde_from_main_domain(main_domain.clone(), &twiddles, 2);
        let original_values_normal_order = ColumnMajorCosetBoundTracePart {
            column: main_domain,
            offset: F::ONE,
        };
        let source = Some(original_values_normal_order).into_iter().chain(other_domains.into_iter());

        let mut result = ColumnMajorBaseOracleForLDE {
            cosets: vec![],
        };
        for coset in source {
            let el = ColumnMajorBaseOracleForCoset {
                original_values_normal_order: vec![coset],
                tree: <Blake2sU32MerkleTreeWithCap as ColumnMajorMerkleTreeConstructor<F>>::dummy(),
                values_per_leaf: 2,
                trace_len_log2: 2,
            };
            result.cosets.push(el);
        }

        result
    }

    fn evaluate_base_multivarite(evals: &[F], point: &[E]) -> E {
        let mut eqs = make_eq_poly_in_full::<F, E>(point);
        let eq = eqs.pop().unwrap();
        assert_eq!(eq.len(), evals.len());
        let mut result = E::ZERO;
        for (a, b) in eq.iter().zip(evals.iter()) {
            let mut t = *a;
            t.mul_assign_by_base(b);
            result.add_assign(&t);
        }
        result
    }

    fn evaluate_multivarite(evals: &[E], point: &[E]) -> E {
        let mut eqs = make_eq_poly_in_full::<F, E>(point);
        let eq = eqs.pop().unwrap();
        assert_eq!(eq.len(), evals.len());
        let mut result = E::ZERO;
        for (a, b) in eq.iter().zip(evals.iter()) {
            let mut t = *a;
            t.mul_assign(b);
            result.add_assign(&t);
        }
        result
    }

    #[test]
    fn test_whir() {
        let worker = Worker::new_with_num_threads(1);

        let inputs: [_; 3] = std::array::from_fn(|_| {
            make_base_oracle(&worker)
        });
        let original_evaluation_point = vec![
            E::from_base(F::from_u32_unchecked(1)),
            E::from_base(F::from_u32_unchecked(2)),
        ];
        let original_claims = inputs.each_ref().map(|el| {
            el.cosets[0].original_values_normal_order.iter().map(|el| {
                let evals = &el.column[..];
                evaluate_base_multivarite(evals, &original_evaluation_point)
            }).collect::<Vec<_>>()
        });

        let twiddles = Twiddles::<F, Global>::new(4, &worker);

        dbg!(&original_claims);

        whir_fold(
            inputs,
            original_claims,
            original_evaluation_point,
            2,
            &E::ONE,
            vec![1],
            vec![1],
            vec![8],
            &twiddles,
            2,
            &worker
        );
    }
}