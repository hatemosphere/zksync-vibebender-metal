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

// NOTE: as we can choose rates somewhat independently from folding parameters, then for 100 bits of conjectured security we can do like
// let's say initially poly is 2^24
// - initial rate 1/2, and fold once - 100 queries, size 2^23
// - we discard all other polys from memory - RAM is no longer a problem, and next computations are cheap
// - rate 1/8, we do 33 queries and fold 4 times. Next step is 2^20
// - rate 1/64, we do 18 queries and fold 4 times. Next step is 2^16
// - rate 1/256, 13 queries, fold 4 times. Next step 2^12
// - rate 1/1024, 10 queries, fold 4 times. Next step 2^8
// - rate 1/4096, 8 queries, fold 4 times, output final 2^4 values

// NOTE on hypercube construction
// Original WHIR paper assumes knowledge of the monomial form, and implementations perform sumcheck over it. For efficiency we want to use
// "main domain" as the multilinear poly representation, and so our definition of the multivariate poly should be different:
// - usually f(x1, x2, ...) = \sum_{hypercube of 0/1} eq(x1, x2, ..., 0/1, 0/1, ...) evals(0/1, 0/1, 0/1)
// - so evaluation at random point is just \sum_{hypercube of 0/1} eq(r1, r2, ..., 0/1, 0/1, ...) evals(0/1, 0/1, 0/1) for concerete values of r1, r2, etc
// - however, we want to use evals at (1/omega, 1/omega^2, ...) in the RHS of the definition instead
// - so our equality poly is a little more complex, and we should change summation - instead of summing over b \in {0, 1}^N, we instead sum over b \in {1, omega} * {1, omega^2}
// for the corresponding root of unity, and equality poly's definition becomes "bivariate multilinear poly with 1s at 1 and omega", and it's
// explicit form is eq(X, Y) = (1 - X')(1 - Y') + X'Y' with linear mapping of (X' - 1)/(X - 0) = (omega - 1), so X = (X' - 1)/(omega - 1) and so for Y', and so
// eq(X, Y) = 1 / (omega - 1) ^ 2 * (X - 1)(Y - 1) + (1 - (X - 1)/(omega - 1))(1 - (Y - 1)/(omega - 1))
// Cheking: eq(1, 1) = 0 + 1
// eq(omega, omega) = 1 + 0

// NOTE: for GKR part we can use assymmetric definition of eq poly, such that it's equality table is defined as
//  X      Y       Value
//  0      1         1
//  1     omega      1
// so after first GKR layer we can use "standard" 0/1 hypercube

use crate::gkr::prover::stages::stage1::{
    compute_column_major_monomial_form_from_main_domain,
    ColumnMajorCosetBoundTracePart,
};
use crate::gkr::sumcheck::eq_poly::make_eq_poly_in_full;
use crate::gkr::sumcheck::*;
use crate::{gkr::prover::apply_row_wise, merkle_trees::ColumnMajorMerkleTreeConstructor};
use fft::{Twiddles, batch_inverse_inplace, bitreverse_enumeration_inplace, domain_generator_for_size, materialize_powers_serial_starting_with_one};
use field::{Field, FieldExtension, PrimeField, TwoAdicField};
use std::alloc::Global;
use worker::Worker;

#[derive(Debug)]
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
        let trace_len = 1 << self.trace_len_log2;

        let mut result: Vec<Vec<F>> = (0..self.values_per_leaf).into_iter().map(|_| {
            Vec::with_capacity(self.original_values_normal_order.len())
        }).collect();

        match self.values_per_leaf {
            2 => {
                let offsets = [0, trace_len/2];
                for src_poly in self.original_values_normal_order.iter() {
                    for (j, offset) in offsets.iter().enumerate() {
                        let i = *offset + index;
                        let value = src_poly.column[i];
                        result[j].push(value);
                    }
                }
            },
            a @ _ => {
                panic!("unsupported: {} values per leaf", a);
            }
        }

        result
    }
}

#[derive(Debug)]
pub struct ColumnMajorBaseOracleForLDE<
    F: PrimeField + TwoAdicField,
    T: ColumnMajorMerkleTreeConstructor<F>,
> {
    pub cosets: Vec<ColumnMajorBaseOracleForCoset<F, T>>,
}

impl<F: PrimeField + TwoAdicField, T: ColumnMajorMerkleTreeConstructor<F>>
    ColumnMajorBaseOracleForLDE<F, T>
{
    pub fn query_for_folded_index(&self, index: usize) -> (usize, Vec<Vec<F>>) {
        // let coset_index = index >> self.cosets[0].trace_len_log2;
        // let internal_index = index & ((1 << self.cosets[0].trace_len_log2) - 1);
        let coset_index = index & (self.cosets.len() - 1);
        let internal_index = index / self.cosets.len();
        (coset_index, self.cosets[coset_index].query_for_folded_index(internal_index))
    }
}

pub fn whir_fold<
    F: PrimeField + TwoAdicField,
    E: FieldExtension<F> + Field,
    T: ColumnMajorMerkleTreeConstructor<F>,
>(
    base_layer_oracles: [ColumnMajorBaseOracleForLDE<F, T>; 3], // memory, witness, setup
    original_claims: [Vec<E>; 3],                               // memory, witness, setup
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
    dbg!(&base_layer_oracles);

    let two_inv = F::from_u32_unchecked(2).inverse().unwrap();

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
    let challenge_powers = vec![E::ONE, E::ZERO, E::ZERO];
    let (base_mem_powers, rest) = challenge_powers.split_at(original_claims[0].len());
    let (base_witness_powers, base_setup_powers) = rest.split_at(original_claims[1].len());
    assert_eq!(base_setup_powers.len(), original_claims[2].len());

    let batch_challenges = [
        base_mem_powers.to_vec(),
        base_witness_powers.to_vec(),
        base_setup_powers.to_vec(),
    ];

    let mut batched_poly_on_main_domain = vec![E::ZERO; 1 << trace_len_log2];

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
    let monomial_form =
        compute_column_major_monomial_form_from_main_domain(
            &batched_poly_on_main_domain[..],
            twiddles,
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
    // so we can NOT easily use the same trick with splitting out eq poly highest coordinate in sumcheck.
    // So we make EQ poly explicitly, and then we will update it after every step, and use naively

    dbg!(&original_evaluation_point);

    let mut eq_polys = make_eq_poly_in_full::<E>(&original_evaluation_point[..]); // TODO: parallelize
    let mut eq_poly_box = eq_polys.pop().unwrap();
    let mut eq_poly = &mut eq_poly_box[..];
    drop(eq_polys);

    // initial folding
    let num_initial_folding_rounds = whir_steps_schedule.next().unwrap();
    let mut last_eq_poly_prefactor_contribution = E::ONE;
    let mut total_eq_poly_prefactor = E::ONE;
    let mut sumchecked_poly_evaluation_form_vec = batched_poly_on_main_domain;
    let mut sumchecked_poly_evaluation_form = &mut sumchecked_poly_evaluation_form_vec[..];
    let mut sumchecked_poly_monomial_form = monomial_form;
    let mut monomial_form_buffer = Vec::with_capacity(sumchecked_poly_monomial_form.len());

    dbg!(&sumchecked_poly_evaluation_form);
    let mut claim = batched_claim;

    assert_eq!(eq_poly.len(), sumchecked_poly_evaluation_form.len());
    assert_eq!(eq_poly.len(), sumchecked_poly_monomial_form.len());

    let mut folding_challenges = vec![];
    let mut ood_samples_per_round = vec![];
    let mut delinearization_challenges_per_round = vec![];

    {
        // Even though we can do all the same trick as in our GKR kernels and only evaluate sum of half-size,
        // instead we naively evaluate at 0 and 1, and use input claim to get the monomial form via Lagrange interpolation

        for el in base_layer_oracles.iter() {
            for el in el.cosets.iter() {
                assert_eq!(el.values_per_leaf, 1 << num_initial_folding_rounds);
            }
        }
        let mut folding_challenges_in_round = vec![];

        for _ in 0..num_initial_folding_rounds {
            let half_len = sumchecked_poly_evaluation_form.len() / 2;
            let eval_at_0 = dot_product(
                &sumchecked_poly_evaluation_form[..half_len],
                &eq_poly[..half_len],
                worker,
            );
            dbg!(eval_at_0);

            let eval_at_1 = dot_product(
                &sumchecked_poly_evaluation_form[half_len..],
                &eq_poly[half_len..],
                worker,
            );
            dbg!(eval_at_1);

            // Lagrange intepolation
            let evaluation_point = evaluation_coordinates[0];
            evaluation_coordinates = &evaluation_coordinates[1..];

            let univariate_coeffs =
                special_lagrange_interpolate(eval_at_0, eval_at_1, claim, evaluation_point);

            {
                // self-check
                let s0 = evaluate_small_univariate_poly(&univariate_coeffs, &E::ZERO);
                assert_eq!(s0, eval_at_0);
                let s1 = evaluate_small_univariate_poly(&univariate_coeffs, &E::ONE);
                assert_eq!(s1, eval_at_1);
                let mut v = s0;
                v.add_assign(&s1);
                v.mul_assign(&last_eq_poly_prefactor_contribution);
                assert_eq!(v, claim);
            }

            // draw folding challenge
            let folding_challenge = E::from_base(F::from_u32_unchecked(42));
            let folding_challenge = E::ZERO;
            folding_challenges_in_round.push(folding_challenge);
            {
                let t = evaluate_eq_poly::<F, E>(&folding_challenge, &evaluation_point);
                last_eq_poly_prefactor_contribution = t;
                total_eq_poly_prefactor.mul_assign(&t);
            }
            let next_claim = evaluate_small_univariate_poly(&univariate_coeffs, &folding_challenge);
            claim = next_claim;
            // and fold the poly itself - both multivariate evals mapping, and monomial form

            dbg!(&sumchecked_poly_monomial_form);
            fold_monomial_form(
                &mut sumchecked_poly_monomial_form,
                &mut monomial_form_buffer,
                &folding_challenge,
                worker,
            );
            dbg!(&sumchecked_poly_monomial_form);

            let domain_size = sumchecked_poly_evaluation_form.len();
            let omegas_inv = materialize_powers_serial_starting_with_one::<F, Global>(F::TWO_ADICITY_GENERATORS_INVERSED[domain_size.trailing_zeros() as usize], domain_size / 2);
            sumchecked_poly_evaluation_form = fold_evaluation_form(
                &mut sumchecked_poly_evaluation_form[..],
                &omegas_inv,
                // &twiddles.inverse_twiddles,
                &folding_challenge,
                &two_inv,
                worker,
            );

            assert_eq!(sumchecked_poly_monomial_form.len(), sumchecked_poly_evaluation_form.len());

            // and so we fold equality poly too

            eq_poly = fold_eq_poly(eq_poly, &folding_challenge, worker);
        }

        folding_challenges.push(folding_challenges_in_round.clone());

        let mut contributions_to_eq_poly = vec![];
        let mut contributions_to_eq_poly_with_base_points = vec![];

        // draw OOD sample
        let ood_point = E::from_base(F::from_u32_unchecked(42));
        // compute OOD value
        let ood_value = evaluate_monomial_form(&sumchecked_poly_monomial_form[..], &ood_point, worker);
        // {
        //     dbg!(ood_value);
        //     let mut pows = make_pows(ood_point, sumchecked_poly_evaluation_form.len().trailing_zeros() as usize);
        //     let value = evaluate_multivarite(&sumchecked_poly_evaluation_form, &pows);
        //     dbg!(value);
        // }

        ood_samples_per_round.push((ood_point, ood_value));

        // now can draw challenges

        // and we can immediatelly query all the original oracles, and drop them. For that we need to draw indexes
        let original_domain_log2 = trace_len_log2 + (original_lde_factor.trailing_zeros() as usize);
        let query_domain_log2 = original_domain_log2 - num_initial_folding_rounds;
        let query_domain_size = 1u64 << query_domain_log2;
        let generator = domain_generator_for_size::<F>(query_domain_size);
        let input_domain_size = 1u64 << original_domain_log2;
        let extended_generator = domain_generator_for_size::<F>(input_domain_size);

        let set_generator = domain_generator_for_size::<F>(1u64 << num_initial_folding_rounds);
        let mut folding_set_offsets: Vec<F> = (0..(1u64 << (num_initial_folding_rounds - 1))).map(|el| set_generator.pow(el as u32)).collect();
        bitreverse_enumeration_inplace(&mut folding_set_offsets);

        let cosets_invs: Vec<_> = base_layer_oracles[0].cosets.iter().map(|el| el.original_values_normal_order[0].offset.inverse().unwrap()).collect();
        let coset_invs_powers: Vec<Vec<F>> = cosets_invs.iter().map(|el| {
            (0..num_initial_folding_rounds).map(|pow| {
                el.pow(pow as u32)
            }).collect::<Vec<_>>()
        }).collect();

        let num_queries = whir_queries_schedule.next().unwrap();
        let mut query_indexes = vec![];
        for _ in 0..num_queries {
            // query index is power for omega^k expression
            query_indexes.push(0usize);
        }
        // and delinearization challenge
        let delinearization_challenge = E::from_base(F::from_u32_unchecked(7));
        delinearization_challenges_per_round.push(delinearization_challenge);

        // we will have OOD sample contribution
        contributions_to_eq_poly.push((ood_point, delinearization_challenge));

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
            let base_root = extended_generator.pow(query_index as u32);
            let base_root_inv = base_root.inverse().unwrap();
            let mut coset_idx = usize::MAX;
            let mut batched_evals = vec![E::ZERO; base_layer_oracles[0].cosets[0].values_per_leaf];
            for (oracle, batching_challenges) in
                base_layer_oracles.iter().zip(batch_challenges.iter())
            {
                let (idx, leaf) = oracle.query_for_folded_index(query_index);
                coset_idx = idx;
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
            dbg!(&batched_evals);

            // Now we can fold
            let mut buffer = Vec::with_capacity(batched_evals.len());
            for folding_step in 0..num_initial_folding_rounds {
                let (src, dst) = if folding_step % 2 == 0 {
                    (&batched_evals[..], &mut buffer)
                } else {
                    (&buffer[..], &mut batched_evals)
                };
                assert!(dst.is_empty());
                assert!(src.is_empty() == false);
                assert!(src.len().is_power_of_two());
                for (set_idx, [a, b]) in src.as_chunks::<2>().0.iter().enumerate() {
                    let mut t = *a;
                    t.sub_assign(b);
                    t.mul_assign(&folding_challenges_in_round[folding_step]);
                    t.mul_assign_by_base(&coset_invs_powers[coset_idx][folding_step]);
                    t.mul_assign_by_base(&base_root_inv);
                    t.mul_assign_by_base(&folding_set_offsets[set_idx]);
                    
                    t.add_assign(a);
                    t.add_assign(b);
                    t.mul_assign_by_base(&two_inv);
                    dst.push(t);
                }
                if folding_step % 2 == 0 {
                    batched_evals.clear();
                } else {
                    buffer.clear();
                };
            }
            
            let folded = if num_initial_folding_rounds % 2 == 1 {
                &buffer[..]
            } else {
                &batched_evals[..]
            };

            dbg!(folded);
            dbg!(&sumchecked_poly_evaluation_form);
            dbg!(&sumchecked_poly_monomial_form);
            {
                let mut t = E::ZERO;
                for el in sumchecked_poly_monomial_form.iter() {
                    t.add_assign(el);
                }
                dbg!(t);
            }

            {
                {
                    let mut source = sumchecked_poly_monomial_form.clone();
                    // TODO: very stupid and slow...
                    bitreverse_enumeration_inplace(&mut source[..]);
                    fft::naive::serial_ct_ntt_bitreversed_to_natural(
                        &mut source[..],
                        sumchecked_poly_monomial_form.len().trailing_zeros(),
                        &twiddles.forward_twiddles,
                    );
                    assert_eq!(source, sumchecked_poly_evaluation_form);
                }
            }

            {
                let omega = domain_generator_for_size::<F>((sumchecked_poly_evaluation_form.len() * 2) as u64);
                for i in 0..sumchecked_poly_evaluation_form.len() {
                    let root = omega.pow(i as u32);
                    // let t = evaluate_monomial_form(&sumchecked_poly_monomial_form, &E::from_base(root), worker);
                    // assert_eq!(t, sumchecked_poly_evaluation_form[i]);
                    let mut t = sumchecked_poly_evaluation_form.to_vec();
                    bitreverse_enumeration_inplace(&mut t);
                    let pows = make_pows(root, sumchecked_poly_evaluation_form.len().trailing_zeros() as usize);
                    // pows.reverse();
                    let value = evaluate_multivariate_at_base(&t, &pows);
                    dbg!(value);
                }
            }

            // and add into sumcheck claim
            contributions_to_eq_poly_with_base_points.push((base_root, delinearization_challenge));
            current_delinearization_challenge.mul_assign(&delinearization_challenge);
        }

        drop(base_layer_oracles);

        // we now update the equality poly - initially we had eq(X, original_evalution_point), from which we folded few coordinates.
        // Now we should add more terms there to reflect OOD and in-domain samples
        update_eq_poly(
            eq_poly,
            &contributions_to_eq_poly,
            &contributions_to_eq_poly_with_base_points,
        );


    }

    // now we step into recursive procesure over one batched polynomial
    {

    }

    // and final step is almost the same as the first one - we can fold few times, output evaluation form, and draw final query indexes,
    // check consistency between them, and perform final explicit sumcheck
    {

    }


    todo!()
}

fn fold_monomial_form<E: Field>(input: &mut Vec<E>, buffer: &mut Vec<E>, challenge: &E, worker: &Worker) {
    // TODO: parallelize
    assert!(input.len().is_power_of_two());
    assert!(buffer.capacity() >= input.len() / 2);
    assert!(buffer.is_empty());

    for ([c0, c1], dst) in input
        .as_chunks::<2>()
        .0
        .iter()
        .zip(buffer.spare_capacity_mut()[..input.len() / 2].iter_mut())
    {
        let mut result = *c1;
        result.mul_assign(challenge);
        result.add_assign(c0);
        dst.write(result);
    }
    unsafe {
        buffer.set_len(input.len() / 2);
    }

    core::mem::swap(input, buffer);
    buffer.clear();
}

fn fold_evaluation_form<'a, F: PrimeField, E: FieldExtension<F> + Field>(
    input: &'a mut [E],
    omegas_inv: &[F],
    challenge: &E,
    two_inv: &F,
    worker: &Worker,
) -> &'a mut [E] {
    // TODO: parallelize
    assert!(input.len().is_power_of_two());
    assert_eq!(input.len() / 2, omegas_inv.len());
    let half_len = input.len() / 2;
    let stride = input.len() / 2;

    for i in 0..input.len() / 2 {
        let f_x = input[i];
        let f_minus_x = input[i + stride];
        let x_inv = omegas_inv[i];

        let mut sum = f_x;
        sum.add_assign(&f_minus_x);

        let mut diff = f_x;
        diff.sub_assign(&f_minus_x);
        diff.mul_assign(challenge);
        diff.mul_assign_by_base(&x_inv);

        sum.add_assign(&diff);
        sum.mul_assign_by_base(two_inv);

        input[i] = sum;
    }

    &mut input[..half_len]
}

fn fold_eq_poly<'a, F: PrimeField, E: FieldExtension<F> + Field>(
    eq_poly: &'a mut [E],
    challenge: &E,
    worker: &Worker,
) -> &'a mut [E] {
    // TODO: parallelize
    assert!(eq_poly.len().is_power_of_two());
    let stride = eq_poly.len() / 2;
    let mut f0_coeff = E::ONE;
    f0_coeff.sub_assign(challenge);
    let f1_coeff = *challenge;

    for i in 0..eq_poly.len() / 2 {
        let mut a = f0_coeff;
        a.mul_assign(&eq_poly[i]);
        let mut b = f1_coeff;
        b.mul_assign(&eq_poly[i + stride]);
        a.add_assign(&b);
        eq_poly[i] = a;
    }

    let next_len = eq_poly.len() / 2;
    &mut eq_poly[..next_len]
}

fn dot_product<F: PrimeField, E: FieldExtension<F> + Field>(
    a: &[E],
    b: &[E],
    worker: &Worker,
) -> E {
    // TODO: parallelize
    assert!(a.len() > 0);
    assert_eq!(a.len(), b.len());

    dbg!(&a);
    dbg!(&b);

    let mut result = E::ZERO;

    for (a, b) in a.iter().zip(b.iter()) {
        let mut t = *a;
        t.mul_assign(b);
        result.add_assign(&t);
    }

    result
}

fn evaluate_monomial_form<E: Field>(
    coeffs: &[E],
    point: &E,
    worker: &Worker,
) -> E {
    // TODO: parallelize

    let mut result = E::ZERO;
    let mut c = E::ONE;

    for a in coeffs.iter() {
        let mut t = *a;
        t.mul_assign(&c);
        c.mul_assign(point);
        result.add_assign(&t);
    }

    result
}

fn special_lagrange_interpolate<E: Field>(
    eval_at_0: E,
    eval_at_1: E,
    eval_at_random: E,
    random_point: E,
) -> [E; 3] {
    // easier to compute special case than generic
    let mut coeffs_for_0 = [E::ZERO, E::ZERO, E::ONE];
    coeffs_for_0[1] = E::ONE;
    coeffs_for_0[1].add_assign(&random_point);
    coeffs_for_0[1].negate();

    coeffs_for_0[0] = E::ONE;
    coeffs_for_0[0].mul_assign(&random_point);

    let mut coeffs_for_1 = [E::ZERO, E::ZERO, E::ONE];
    coeffs_for_1[1] = E::ZERO;
    coeffs_for_1[1].add_assign(&random_point);
    coeffs_for_1[1].negate();

    coeffs_for_1[0] = E::ZERO;
    coeffs_for_1[0].mul_assign(&random_point);

    let mut coeffs_for_random_point = [E::ZERO, E::ZERO, E::ONE];
    coeffs_for_random_point[1] = E::ZERO;
    coeffs_for_random_point[1].add_assign(&E::ONE);
    coeffs_for_random_point[1].negate();

    coeffs_for_random_point[0] = E::ZERO;
    coeffs_for_random_point[0].mul_assign(&E::ONE);

    let mut dens = [E::ONE, E::ONE, E::ONE];

    let mut t = E::ZERO;
    t.sub_assign(&E::ONE);
    dens[0].mul_assign(&t);
    let mut t = E::ZERO;
    t.sub_assign(&random_point);
    dens[0].mul_assign(&t);

    let mut t = E::ONE;
    t.sub_assign(&E::ZERO);
    dens[1].mul_assign(&t);
    let mut t = E::ONE;
    t.sub_assign(&random_point);
    dens[1].mul_assign(&t);

    let mut t = random_point;
    t.sub_assign(&E::ZERO);
    dens[2].mul_assign(&t);
    let mut t = random_point;
    t.sub_assign(&E::ONE);
    dens[2].mul_assign(&t);

    let mut buffer = [E::ZERO; 3];
    batch_inverse_inplace(&mut dens, &mut buffer);

    let mut result = [E::ZERO; 3];
    for (eval, den, coeffs) in [
        (eval_at_0, dens[0], coeffs_for_0),
        (eval_at_1, dens[1], coeffs_for_1),
        (eval_at_random, dens[2], coeffs_for_random_point),
    ]
    .into_iter()
    {
        for (i, c) in coeffs.into_iter().enumerate() {
            let mut t = c;
            t.mul_assign(&den);
            t.mul_assign(&eval);
            result[i].add_assign(&t);
        }
    }

    result
}

fn make_pows<E: Field>(
    el: E,
    num_powers: usize
) -> Vec<E> {
    let mut result = Vec::with_capacity(num_powers);
    let mut current = el;
    for _ in 0..num_powers {
        result.push(current);
        current.square();
    }

    result
}

fn update_eq_poly<F: PrimeField, E: FieldExtension<F> + Field>(
    eq_poly: &mut [E],
    ood_samples: &[(E, E)],
    in_domain_samples: &[(F, E)],
) {
    assert!(eq_poly.len().is_power_of_two());
    assert_eq!(ood_samples.len(), 1);
    for (point, challenge) in ood_samples.iter() {
        let pows = make_pows(*point, eq_poly.len().trailing_zeros() as usize);
        let eq_polys = make_eq_poly_in_full::<E>(&pows);
        for (dst, src) in eq_poly.iter_mut().zip(eq_polys.last().unwrap().iter()) {
            let mut t = *challenge;
            t.mul_assign(src);
            dst.add_assign(&t);
        }
    }
    for (point, challenge) in in_domain_samples.iter() {
        let pows = make_pows(*point, eq_poly.len().trailing_zeros() as usize);
        let eq_polys = make_eq_poly_in_full::<F>(&pows);
        for (dst, src) in eq_poly.iter_mut().zip(eq_polys.last().unwrap().iter()) {
            let mut t = *challenge;
            t.mul_assign_by_base(src);
            dst.add_assign(&t);
        }
    }
}

fn evaluate_base_multivariate<F: PrimeField, E: FieldExtension<F> + Field>(evals: &[F], point: &[E]) -> E {
    let mut eqs = make_eq_poly_in_full::<E>(point);
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

fn evaluate_multivariate<E: Field>(evals: &[E], point: &[E]) -> E {
    let mut eqs = make_eq_poly_in_full::<E>(point);
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

fn evaluate_multivariate_at_base<F: PrimeField, E: FieldExtension<F> + Field>(evals: &[E], point: &[F]) -> E {
    let mut eqs = make_eq_poly_in_full::<F>(point);
    let eq = eqs.pop().unwrap();
    assert_eq!(eq.len(), evals.len());
    let mut result = E::ZERO;
    for (a, b) in eq.iter().zip(evals.iter()) {
        let mut t = *b;
        t.mul_assign_by_base(a);
        result.add_assign(&t);
    }
    result
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use super::*;
    use crate::gkr::prover::stages::stage1::*;
    use crate::{
        gkr::sumcheck::eq_poly::make_eq_poly_in_full,
        merkle_trees::blake2s_for_everything_tree::Blake2sU32MerkleTreeWithCap,
    };
    use field::baby_bear::{base::BabyBearField, ext4::BabyBearExt4};
    type F = BabyBearField;
    type E = BabyBearExt4;

    fn make_base_oracle(
        size: usize,
        worker: &Worker,
    ) -> ColumnMajorBaseOracleForLDE<F, Blake2sU32MerkleTreeWithCap> {
        let main_domain: Vec<F> = (1..=size).map(|el| {
            F::from_u32_unchecked(el as u32)
        }).collect();
        let twiddles = Twiddles::<F, Global>::new(size, worker);
        let main_domain = Arc::new(main_domain.into_boxed_slice());

        let other_domains =
            compute_column_major_lde_from_main_domain(main_domain.clone(), &twiddles, 2);
        let original_values_normal_order = ColumnMajorCosetBoundTracePart {
            column: main_domain,
            offset: F::ONE,
        };
        let source = Some(original_values_normal_order)
            .into_iter()
            .chain(other_domains.into_iter());

        let mut result = ColumnMajorBaseOracleForLDE { cosets: vec![] };
        for coset in source {
            let el = ColumnMajorBaseOracleForCoset {
                original_values_normal_order: vec![coset],
                tree: <Blake2sU32MerkleTreeWithCap as ColumnMajorMerkleTreeConstructor<F>>::dummy(),
                values_per_leaf: 2,
                trace_len_log2: size.trailing_zeros() as usize,
            };
            result.cosets.push(el);
        }

        result
    }

    #[test]
    fn test_whir() {
        let worker = Worker::new_with_num_threads(1);
        let size = 8;

        let inputs: [_; 3] = std::array::from_fn(|_| make_base_oracle(size, &worker));
        let original_evaluation_point = vec![
            E::from_base(F::from_u32_unchecked(4)),
            E::from_base(F::from_u32_unchecked(8)),
            E::from_base(F::from_u32_unchecked(16)),
        ];
        let original_claims = inputs.each_ref().map(|el| {
            el.cosets[0]
                .original_values_normal_order
                .iter()
                .map(|el| {
                    let evals = &el.column[..];
                    evaluate_base_multivariate(evals, &original_evaluation_point)
                })
                .collect::<Vec<_>>()
        });

        let twiddles = Twiddles::<F, Global>::new(size, &worker);

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
            size.trailing_zeros() as usize,
            &worker,
        );
    }
}

// (c0 + c1), (c0 - c1)
// evaluation point is omega = 1
// eq = (1 - 1) * (c0 + c1) + 1 * (c0 - c1)