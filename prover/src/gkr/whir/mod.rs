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

use crate::gkr::prover::stages::stage1::{
    compute_column_major_lde_from_monomial_form,
    compute_column_major_monomial_form_from_main_domain_owned, ColumnMajorCosetBoundTracePart,
};
use crate::gkr::sumcheck::eq_poly::{make_domain_eq_poly_in_full, make_eq_poly_in_full};
use crate::gkr::sumcheck::*;
use crate::gkr::whir::hypercube_to_monomial::multivariate_coeffs_into_hypercube_evals;
use crate::{gkr::prover::apply_row_wise, merkle_trees::ColumnMajorMerkleTreeConstructor};
use fft::{
    batch_inverse_inplace, bitreverse_enumeration_inplace, domain_generator_for_size,
    materialize_powers_serial_starting_with_one, Twiddles,
};
use field::{Field, FieldExtension, PrimeField, TwoAdicField};
use std::alloc::Global;
use std::sync::Arc;
use worker::Worker;

pub mod hypercube_to_monomial;
pub mod queries;
pub mod whir_proof;

pub use self::queries::*;
pub use self::whir_proof::*;

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
    pub fn query_for_folded_index(&self, index: usize) -> (Vec<Vec<F>>, BaseFieldQuery<F, T>) {
        assert!(self.values_per_leaf.is_power_of_two());
        assert!(index < (1 << self.trace_len_log2) / self.values_per_leaf);
        let trace_len = 1 << self.trace_len_log2;

        let mut result: Vec<Vec<F>> = (0..self.values_per_leaf)
            .into_iter()
            .map(|_| Vec::with_capacity(self.original_values_normal_order.len()))
            .collect();

        match self.values_per_leaf {
            2 => {
                let offsets = [0, trace_len / 2];
                for src_poly in self.original_values_normal_order.iter() {
                    for (j, offset) in offsets.iter().enumerate() {
                        let i = *offset + index;
                        let value = src_poly.column[i];
                        result[j].push(value);
                    }
                }
            }
            a @ _ => {
                panic!("unsupported: {} values per leaf", a);
            }
        }
        let (_leaf_hash, path) = self.tree.get_proof(index);
        let query = BaseFieldQuery {
            index,
            leaf_values_concatenated: result.iter().flatten().copied().collect(),
            path,
        };

        (result, query)
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
    pub fn query_for_folded_index(
        &self,
        index: usize,
    ) -> (usize, Vec<Vec<F>>, BaseFieldQuery<F, T>) {
        // let coset_index = index >> self.cosets[0].trace_len_log2;
        // let internal_index = index & ((1 << self.cosets[0].trace_len_log2) - 1);
        let coset_index = index & (self.cosets.len() - 1);
        let internal_index = index / self.cosets.len();
        let (values, mut query) = self.cosets[coset_index].query_for_folded_index(internal_index);
        query.index = index; // remap index to outer
        (coset_index, values, query)
    }
}

#[derive(Debug)]
pub struct ColumnMajorExtensionOracleForCoset<
    F: PrimeField + TwoAdicField,
    E: FieldExtension<F> + Field,
    T: ColumnMajorMerkleTreeConstructor<F>,
> {
    pub values_normal_order: ColumnMajorCosetBoundTracePart<F, E>, // single columns
    pub tree: T,
    pub values_per_leaf: usize,
    pub trace_len_log2: usize,
}

fn offsets_for_leaf_construction<const N: usize>(trace_len: usize) -> [usize; N] {
    assert!(trace_len.is_power_of_two());
    assert!(N.is_power_of_two());
    let mut result = [0; N];
    let stride = trace_len / N;
    for i in 0..N {
        result[i] = stride * i;
    }
    bitreverse_enumeration_inplace(&mut result);

    result
}

pub(crate) fn offsets_vec_for_leaf_construction(trace_len: usize, combine_by: usize) -> Vec<usize> {
    assert!(trace_len.is_power_of_two());
    assert!(combine_by.is_power_of_two());
    let mut result = Vec::with_capacity(combine_by);
    let stride = trace_len / combine_by;
    for i in 0..combine_by {
        result.push(stride * i);
    }
    bitreverse_enumeration_inplace(&mut result);

    result
}

impl<
        F: PrimeField + TwoAdicField,
        E: FieldExtension<F> + Field,
        T: ColumnMajorMerkleTreeConstructor<F>,
    > ColumnMajorExtensionOracleForCoset<F, E, T>
{
    pub fn query_for_folded_index(&self, index: usize) -> (Vec<E>, ExtensionFieldQuery<F, E, T>) {
        assert!(self.values_per_leaf.is_power_of_two());
        assert!(index < (1 << self.trace_len_log2) / self.values_per_leaf);
        let trace_len = 1 << self.trace_len_log2;

        let mut result: Vec<E> = Vec::with_capacity(self.values_per_leaf);

        match self.values_per_leaf {
            2 => {
                let offsets = offsets_for_leaf_construction::<2>(trace_len);
                for offset in offsets.iter() {
                    let i = *offset + index;
                    let value = self.values_normal_order.column[i];
                    result.push(value);
                }
            }
            4 => {
                let offsets = offsets_for_leaf_construction::<4>(trace_len);
                for offset in offsets.iter() {
                    let i = *offset + index;
                    let value = self.values_normal_order.column[i];
                    result.push(value);
                }
            }
            8 => {
                let offsets = offsets_for_leaf_construction::<8>(trace_len);
                for offset in offsets.iter() {
                    let i = *offset + index;
                    let value = self.values_normal_order.column[i];
                    result.push(value);
                }
            }
            16 => {
                let offsets = offsets_for_leaf_construction::<16>(trace_len);
                for offset in offsets.iter() {
                    let i = *offset + index;
                    let value = self.values_normal_order.column[i];
                    result.push(value);
                }
            }
            a @ _ => {
                panic!("unsupported: {} values per leaf", a);
            }
        }

        let (_leaf_hash, path) = self.tree.get_proof(index);
        let query = ExtensionFieldQuery {
            index,
            leaf_values_concatenated: result.clone(),
            path,
        };

        (result, query)
    }
}

#[derive(Debug)]
pub struct ColumnMajorExtensionOracleForLDE<
    F: PrimeField + TwoAdicField,
    E: FieldExtension<F> + Field,
    T: ColumnMajorMerkleTreeConstructor<F>,
> {
    pub cosets: Vec<ColumnMajorExtensionOracleForCoset<F, E, T>>,
}

impl<
        F: PrimeField + TwoAdicField,
        E: FieldExtension<F> + Field,
        T: ColumnMajorMerkleTreeConstructor<F>,
    > ColumnMajorExtensionOracleForLDE<F, E, T>
{
    pub fn query_for_folded_index(
        &self,
        index: usize,
    ) -> (usize, Vec<E>, ExtensionFieldQuery<F, E, T>) {
        // let coset_index = index >> self.cosets[0].trace_len_log2;
        // let internal_index = index & ((1 << self.cosets[0].trace_len_log2) - 1);
        let coset_index = index & (self.cosets.len() - 1);
        let internal_index = index / self.cosets.len();
        let (values, mut query) = self.cosets[coset_index].query_for_folded_index(internal_index);
        query.index = index; // remap index to outer
        (coset_index, values, query)
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

    // TODO: transcript
    tree_cap_size: usize,
    trace_len_log2: usize,
    worker: &Worker,
) -> WhirPolyCommitProof<F, E, T>
where
    [(); E::DEGREE]: Sized,
{
    let two_inv = F::from_u32_unchecked(2).inverse().unwrap();

    let mut commitments = Vec::with_capacity(3);
    for i in 0..3 {
        let t = WhirBaseLayerCommitmentAndQueries {
            commitment: WhirCommitment {
                coset_caps: base_layer_oracles[i]
                    .cosets
                    .iter()
                    .map(|el| el.tree.get_cap())
                    .collect(),
                _marker: core::marker::PhantomData,
            },
            num_columns: base_layer_oracles[i].cosets[0]
                .original_values_normal_order
                .len(),
            evals: original_claims[i].clone(),
            queries: vec![],
        };
        commitments.push(t);
    }

    let [memory_commitment, witness_commitment, setup_commitment] = commitments.try_into().unwrap();

    let mut proof = WhirPolyCommitProof {
        witness_commitment,
        memory_commitment,
        setup_commitment,
        sumcheck_polys: vec![],
        intermediate_whir_oracles: Vec::with_capacity(whir_steps_lde_factors.len()),
        final_monomials: vec![],
    };

    let mut final_poly_log2 = trace_len_log2;
    for el in whir_steps_schedule.iter() {
        assert!(*el <= final_poly_log2);
        final_poly_log2 -= *el;
    }

    // proof elements

    // for logical simplicity, we will always treat first/last rounds as separate
    let mut initial_queries: Vec<[Vec<()>; 3]> = vec![];
    let initial_ood_sample;

    let mut commits_in_rounds: Vec<()> = vec![];
    let mut queries_in_rounds: Vec<Vec<()>> = vec![];
    let mut ood_samples_in_rounds: Vec<(E, E)> = vec![];

    assert!(original_lde_factor.is_power_of_two());
    let num_whir_steps = whir_steps_lde_factors.len();
    assert_eq!(whir_steps_schedule.len(), whir_steps_lde_factors.len() + 1);
    assert_eq!(whir_steps_schedule.len(), whir_queries_schedule.len());

    let mut rs_oracle;

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
                    assert_eq!(src.len(), 1 << trace_len_log2);
                    for i in 0..chunk_size {
                        let mut result = *batch_challenge;
                        result.mul_assign_by_base(&src[chunk_start + i]);
                        dest[i].add_assign(&result);
                    }
                }
            }
        },
    );

    let monomial_form = compute_column_major_monomial_form_from_main_domain_owned(
        batched_poly_on_main_domain,
        twiddles,
    );

    assert_eq!(monomial_form.len(), 1 << trace_len_log2);

    let mut sumcheck_evals = monomial_form.clone();
    multivariate_coeffs_into_hypercube_evals(
        &mut sumcheck_evals,
        monomial_form.len().trailing_zeros(),
    );
    bitreverse_enumeration_inplace(&mut sumcheck_evals);

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
    assert!(num_rounds >= 2);

    let mut whir_steps_schedule = whir_steps_schedule.into_iter().peekable();
    let mut whir_queries_schedule = whir_queries_schedule.into_iter();
    let mut whir_steps_lde_factors = whir_steps_lde_factors.into_iter();

    // as we will eventually continue to mix-in additional equality polys into sumcheck kernel,
    // so we can NOT easily use the same trick with splitting out eq poly highest coordinate in sumcheck.
    // So we make EQ poly explicitly, and then we will update it after every step, and use naively

    dbg!(&original_evaluation_point);

    let mut eq_polys = make_eq_poly_in_full::<E>(&original_evaluation_point[..]); // TODO: parallelize
    let mut eq_poly_box = eq_polys.pop().unwrap();
    let mut eq_poly = &mut eq_poly_box[..];
    drop(eq_polys);

    let mut last_eq_poly_prefactor_contribution = E::ONE;
    let mut total_eq_poly_prefactor = E::ONE;
    let mut sumchecked_poly_evaluation_form_vec = sumcheck_evals;
    let mut sumchecked_poly_evaluation_form = &mut sumchecked_poly_evaluation_form_vec[..];
    let mut sumchecked_poly_monomial_form = monomial_form;
    let mut monomial_form_buffer = Vec::with_capacity(sumchecked_poly_monomial_form.len());

    dbg!(&sumchecked_poly_evaluation_form);
    let mut claim = batched_claim;

    assert_eq!(eq_poly.len(), sumchecked_poly_evaluation_form.len());
    assert_eq!(eq_poly.len(), sumchecked_poly_monomial_form.len());

    let mut folding_challenges = vec![];
    let mut delinearization_challenges_per_round = vec![];

    let mut poly_size_log2 = trace_len_log2;

    // initial round where we fold and query existing oracles
    {
        let num_initial_folding_rounds = whir_steps_schedule.next().unwrap();
        println!("Initialy fold by {}", 1 << num_initial_folding_rounds);

        assert!(num_initial_folding_rounds <= poly_size_log2);
        let rs_domain_log2 = trace_len_log2 + (original_lde_factor.trailing_zeros() as usize);
        let query_domain_log2 = rs_domain_log2 - num_initial_folding_rounds;

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

            let eval_at_1 = dot_product(
                &sumchecked_poly_evaluation_form[half_len..],
                &eq_poly[half_len..],
                worker,
            );

            // Lagrange intepolation
            let evaluation_point = original_evaluation_point[0];
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

            sumchecked_poly_evaluation_form = fold_evaluation_form(
                &mut sumchecked_poly_evaluation_form[..],
                &folding_challenge,
                worker,
            );

            assert_eq!(
                sumchecked_poly_monomial_form.len(),
                sumchecked_poly_evaluation_form.len()
            );

            // and so we fold equality poly too
            eq_poly = fold_eq_poly(eq_poly, &folding_challenge, worker);
            assert_eq!(sumchecked_poly_evaluation_form.len(), eq_poly.len());
        }
        poly_size_log2 -= num_initial_folding_rounds;

        assert_eq!(sumchecked_poly_evaluation_form.len(), 1 << poly_size_log2);
        assert_eq!(sumchecked_poly_monomial_form.len(), 1 << poly_size_log2);
        assert_eq!(eq_poly.len(), 1 << poly_size_log2);

        folding_challenges.push(folding_challenges_in_round.clone());

        // compute RS for folded one (we will NOT query it this round)
        {
            let lde_factor = whir_steps_lde_factors.next().unwrap();
            let rs = compute_column_major_lde_from_monomial_form(
                &sumchecked_poly_monomial_form,
                twiddles,
                lde_factor,
            );
            let next_folding_steps = *whir_steps_schedule.peek().unwrap();
            let next_oracle = commit_single_ext_poly::<F, E, T>(
                rs,
                1 << next_folding_steps,
                tree_cap_size,
                worker,
            );
            let c = WhirIntermediateCommitmentAndQueries {
                commitment: WhirCommitment {
                    coset_caps: next_oracle
                        .cosets
                        .iter()
                        .map(|el| el.tree.get_cap())
                        .collect(),
                    _marker: core::marker::PhantomData,
                },
                queries: vec![],
            };
            proof.intermediate_whir_oracles.push(c);
            rs_oracle = next_oracle;
        }

        let mut contributions_to_eq_poly = vec![];
        let mut contributions_to_eq_poly_with_base_points = vec![];

        // draw OOD sample
        let ood_point = E::from_base(F::from_u32_unchecked(42));
        // compute OOD value
        let ood_value =
            evaluate_monomial_form(&sumchecked_poly_monomial_form[..], &ood_point, worker);
        {
            let pows = make_pows(
                ood_point,
                sumchecked_poly_evaluation_form.len().trailing_zeros() as usize,
            );
            let value = evaluate_multivariate(&sumchecked_poly_evaluation_form, &pows);
            assert_eq!(value, ood_value);
        }

        initial_ood_sample = (ood_point, ood_value);

        // now can draw challenges

        // and we can immediatelly query all the original oracles, and drop them. For that we need to draw indexes
        let query_domain_size = 1u64 << query_domain_log2;

        dbg!(query_domain_size);
        let query_domain_generator = domain_generator_for_size::<F>(query_domain_size);
        let input_domain_size = 1u64 << rs_domain_log2;
        let extended_generator = domain_generator_for_size::<F>(input_domain_size);

        let set_generator = domain_generator_for_size::<F>(1u64 << num_initial_folding_rounds);
        let mut folding_set_offsets: Vec<F> = (0..(1u64 << (num_initial_folding_rounds - 1)))
            .map(|el| set_generator.pow(el as u32))
            .collect();
        bitreverse_enumeration_inplace(&mut folding_set_offsets);

        let cosets_invs: Vec<_> = base_layer_oracles[0]
            .cosets
            .iter()
            .map(|el| el.original_values_normal_order[0].offset.inverse().unwrap())
            .collect();
        let coset_invs_powers: Vec<Vec<F>> = cosets_invs
            .iter()
            .map(|el| {
                (0..num_initial_folding_rounds)
                    .map(|pow| el.pow((1 << pow) as u32))
                    .collect::<Vec<_>>()
            })
            .collect();

        let num_queries = whir_queries_schedule.next().unwrap();
        let mut query_indexes = vec![];
        for _ in 0..num_queries {
            // query index is power for omega^k expression
        }
        {
            query_indexes.push(0usize);
            query_indexes.push(1usize);
            query_indexes.push(2usize);
            query_indexes.push(3usize);
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
            assert!(query_index < query_domain_size as usize);
            let query_point = query_domain_generator.pow(query_index as u32);

            // we have a query point, and now we need to understand "preimages" for it
            // assume that we have a query point omega_q ^ i, and we fold K times.
            // Then what we need is a set of points omega^{0 << (log(query_domain)) || i }, omega^{1 << (log(query_domain)) || i}, etc
            // where omega = root(omega_q, 2^K)

            // So for "base root" we take omega^i from the notations above, and when we fold we multiply it by root(1, 2^K)

            // get original leaf, compute batched, and then folded value
            let base_root = extended_generator.pow(query_index as u32);
            let base_root_inv = base_root.inverse().unwrap();
            let mut coset_idx = usize::MAX;
            let mut batched_evals = vec![E::ZERO; base_layer_oracles[0].cosets[0].values_per_leaf];
            for (set_idx, (oracle, batching_challenges)) in base_layer_oracles
                .iter()
                .zip(batch_challenges.iter())
                .enumerate()
            {
                let (idx, leaf, query) = oracle.query_for_folded_index(query_index);
                match set_idx {
                    0 => {
                        proof.memory_commitment.queries.push(query);
                    }
                    1 => {
                        proof.witness_commitment.queries.push(query);
                    }
                    2 => {
                        proof.setup_commitment.queries.push(query);
                    }
                    _ => {
                        unreachable!()
                    }
                }
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

            // Now we can fold queries values, in a normal FRI style
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
            assert_eq!(folded.len(), 1);

            // dbg!((query_index, query_point, folded[0]));

            // and add into sumcheck claim
            contributions_to_eq_poly_with_base_points
                .push((query_point, delinearization_challenge));
            {
                let mut t = folded[0];
                t.mul_assign(&delinearization_challenge);
                claim_correction.add_assign(&t);
            }
            current_delinearization_challenge.mul_assign(&delinearization_challenge);
        }

        dbg!(&sumchecked_poly_evaluation_form);
        dbg!(&sumchecked_poly_monomial_form);

        {
            // self-check that our multivariate evals match monomial representation
            {
                let mut source = sumchecked_poly_monomial_form.clone();
                multivariate_coeffs_into_hypercube_evals(
                    &mut source,
                    sumchecked_poly_monomial_form.len().trailing_zeros(),
                );
                bitreverse_enumeration_inplace(&mut source);
                assert_eq!(source, sumchecked_poly_evaluation_form);
            }

            // self-check that our domain evaluations from monomial form match pows (so, RS code) definition
            {
                let omega = domain_generator_for_size::<F>(query_domain_size);
                for i in 0..query_domain_size {
                    let root = omega.pow(i as u32);
                    let eval_from_monomial = evaluate_monomial_form(
                        &sumchecked_poly_monomial_form,
                        &E::from_base(root),
                        worker,
                    );
                    // dbg!((i, root, eval_from_monomial));
                    let t = sumchecked_poly_evaluation_form.to_vec();
                    let pows = make_pows(
                        root,
                        sumchecked_poly_evaluation_form.len().trailing_zeros() as usize,
                    );
                    // pows.reverse();
                    let eval_from_multivariate = evaluate_multivariate_at_base(&t, &pows);
                    assert_eq!(eval_from_monomial, eval_from_multivariate);
                }
            }
        }

        drop(base_layer_oracles);

        // we now update the equality poly - initially we had eq(X, original_evalution_point), from which we folded few coordinates.
        // Now we should add more terms there to reflect OOD and in-domain samples
        update_eq_poly(
            eq_poly,
            &contributions_to_eq_poly,
            &contributions_to_eq_poly_with_base_points,
        );

        // and remember new sumcheck claim
        dbg!(&claim);
        dbg!(&claim_correction);
        claim.add_assign(&claim_correction);
    }

    let num_internal_whir_steps = num_whir_steps - 1;
    println!(
        "Initial queries and folding are complete, now can proceed into {} internal rounds",
        num_internal_whir_steps
    );

    // now we step into recursive procesure over one batched polynomial and it's evals. Our sequence is
    // - fold
    // - RS code word computation and commit
    // - query previous(!) RS oracle
    // - update claim and eq poly
    for _ in 0..num_internal_whir_steps {
        // commit
        let num_folding_steps = whir_steps_schedule.next().unwrap();
        let num_queries = whir_queries_schedule.next().unwrap();
        assert!(num_folding_steps <= poly_size_log2);

        let rs_domain_log2 = poly_size_log2 + (rs_oracle.cosets.len().trailing_zeros() as usize);
        let query_domain_log2 = rs_domain_log2 - num_folding_steps;

        // fold

        // NOTE: we can no longer use the fact that sumcheck kernel is simple as \sum_X eq(r, X) p(X),
        // and so to send degree 2 poly to the verifier we need to compute such kernel at 3 points and interpolate.
        // As our hypercube is 0/1, then we choose to compute at 0, 1 and 1/2

        // eval(1/2) = \sum_{X} multilinear(1/2, X) * p(1/2, X) = 1/4 \sum_{X} (multilinear(1, X) - multilinear(0, X)) * (P(1, X) - P(0, X))

        let mut folding_challenges_in_round = Vec::with_capacity(num_folding_steps);

        for _ in 0..num_folding_steps {
            let (f0, f1, f_half) = special_three_point_eval(
                &sumchecked_poly_evaluation_form[..],
                &eq_poly[..],
                worker,
            );
            let univariate_coeffs =
                special_lagrange_interpolate(f0, f1, f_half, E::from_base(two_inv));
            // send the monomial form to the verifier

            {
                // self-check
                let s0 = evaluate_small_univariate_poly(&univariate_coeffs, &E::ZERO);
                assert_eq!(s0, f0);
                let s1 = evaluate_small_univariate_poly(&univariate_coeffs, &E::ONE);
                assert_eq!(s1, f1);
                let s_half =
                    evaluate_small_univariate_poly(&univariate_coeffs, &E::from_base(two_inv));
                assert_eq!(s_half, f_half);
                let mut v = s0;
                v.add_assign(&s1);
                assert_eq!(v, claim);
            }

            let folding_challenge = E::from_base(F::from_u32_unchecked(42));
            let folding_challenge = E::ZERO;
            folding_challenges_in_round.push(folding_challenge);

            let next_claim = evaluate_small_univariate_poly(&univariate_coeffs, &folding_challenge);
            claim = next_claim;
            // and fold the poly itself - both multivariate evals mapping, and monomial form

            fold_monomial_form(
                &mut sumchecked_poly_monomial_form,
                &mut monomial_form_buffer,
                &folding_challenge,
                worker,
            );

            sumchecked_poly_evaluation_form = fold_evaluation_form(
                &mut sumchecked_poly_evaluation_form[..],
                &folding_challenge,
                worker,
            );

            assert_eq!(
                sumchecked_poly_monomial_form.len(),
                sumchecked_poly_evaluation_form.len()
            );

            eq_poly = fold_eq_poly(eq_poly, &folding_challenge, worker);
            assert_eq!(sumchecked_poly_evaluation_form.len(), eq_poly.len());
        }

        poly_size_log2 -= num_folding_steps;

        assert_eq!(sumchecked_poly_evaluation_form.len(), 1 << poly_size_log2);
        assert_eq!(sumchecked_poly_monomial_form.len(), 1 << poly_size_log2);
        assert_eq!(eq_poly.len(), 1 << poly_size_log2);

        // query
        folding_challenges.push(folding_challenges_in_round.clone());

        let rs_oracle_to_query = {
            let lde_factor = whir_steps_lde_factors.next().unwrap();
            let rs = compute_column_major_lde_from_monomial_form(
                &sumchecked_poly_monomial_form,
                twiddles,
                lde_factor,
            );
            let next_folding_steps = *whir_steps_schedule.peek().unwrap();
            let next_oracle = commit_single_ext_poly::<F, E, T>(
                rs,
                1 << next_folding_steps,
                tree_cap_size,
                worker,
            );
            proof
                .intermediate_whir_oracles
                .push(WhirIntermediateCommitmentAndQueries {
                    commitment: WhirCommitment {
                        coset_caps: next_oracle
                            .cosets
                            .iter()
                            .map(|el| el.tree.get_cap())
                            .collect(),
                        _marker: core::marker::PhantomData,
                    },
                    queries: vec![],
                });
            core::mem::replace(&mut rs_oracle, next_oracle)
        };

        // draw OOD sample
        let ood_point = E::from_base(F::from_u32_unchecked(42));
        // compute OOD value
        let ood_value =
            evaluate_monomial_form(&sumchecked_poly_monomial_form[..], &ood_point, worker);
        {
            let pows = make_pows(
                ood_point,
                sumchecked_poly_evaluation_form.len().trailing_zeros() as usize,
            );
            let value = evaluate_multivariate(&sumchecked_poly_evaluation_form, &pows);
            assert_eq!(value, ood_value);
        }

        ood_samples_in_rounds.push((ood_point, ood_value));

        let mut contributions_to_eq_poly = vec![];
        let mut contributions_to_eq_poly_with_base_points = vec![];

        let query_domain_size = 1u64 << query_domain_log2;

        dbg!(query_domain_size);
        let query_domain_generator = domain_generator_for_size::<F>(query_domain_size);
        let input_domain_size = 1u64 << rs_domain_log2;
        let extended_generator = domain_generator_for_size::<F>(input_domain_size);

        let set_generator = domain_generator_for_size::<F>(1u64 << num_folding_steps);
        let mut folding_set_offsets: Vec<F> = (0..(1u64 << (num_folding_steps - 1)))
            .map(|el| set_generator.pow(el as u32))
            .collect();
        bitreverse_enumeration_inplace(&mut folding_set_offsets);

        let cosets_invs: Vec<_> = rs_oracle_to_query
            .cosets
            .iter()
            .map(|el| el.values_normal_order.offset.inverse().unwrap())
            .collect();
        let coset_invs_powers: Vec<Vec<F>> = cosets_invs
            .iter()
            .map(|el| {
                (0..num_folding_steps)
                    .map(|pow| el.pow((1 << pow) as u32))
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut query_indexes = vec![];
        for _ in 0..num_queries {
            // query index is power for omega^k expression
        }
        {
            query_indexes.push(0usize);
            query_indexes.push(1usize);
            query_indexes.push(2usize);
            query_indexes.push(3usize);
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
            assert!(query_index < query_domain_size as usize);
            let query_point = query_domain_generator.pow(query_index as u32);

            // we have a query point, and now we need to understand "preimages" for it
            // assume that we have a query point omega_q ^ i, and we fold K times.
            // Then what we need is a set of points omega^{0 << (log(query_domain)) || i }, omega^{1 << (log(query_domain)) || i}, etc
            // where omega = root(omega_q, 2^K)

            // So for "base root" we take omega^i from the notations above, and when we fold we multiply it by root(1, 2^K)

            // get original leaf, compute batched, and then folded value
            let base_root = extended_generator.pow(query_index as u32);
            let base_root_inv = base_root.inverse().unwrap();

            let (coset_idx, mut evals, query) =
                rs_oracle_to_query.query_for_folded_index(query_index);
            let num_intermediate_oracles = proof.intermediate_whir_oracles.len();
            assert!(num_intermediate_oracles >= 2);
            let intermediate_oracle =
                &mut proof.intermediate_whir_oracles[num_intermediate_oracles - 2];
            intermediate_oracle.queries.push(query);

            // Now we can fold queries values, in a normal FRI style
            let mut buffer = Vec::with_capacity(evals.len());
            for folding_step in 0..num_folding_steps {
                let (src, dst) = if folding_step % 2 == 0 {
                    (&evals[..], &mut buffer)
                } else {
                    (&buffer[..], &mut evals)
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
                    evals.clear();
                } else {
                    buffer.clear();
                };
            }

            let folded = if num_folding_steps % 2 == 1 {
                &buffer[..]
            } else {
                &evals[..]
            };
            assert_eq!(folded.len(), 1);

            dbg!((query_index, query_point, folded[0]));

            // and add into sumcheck claim
            contributions_to_eq_poly_with_base_points
                .push((query_point, delinearization_challenge));
            {
                let mut t = folded[0];
                t.mul_assign(&delinearization_challenge);
                claim_correction.add_assign(&t);
            }
            current_delinearization_challenge.mul_assign(&delinearization_challenge);
        }

        dbg!(&sumchecked_poly_evaluation_form);
        dbg!(&sumchecked_poly_monomial_form);

        {
            // self-check that our multivariate evals match monomial representation
            {
                let mut source = sumchecked_poly_monomial_form.clone();
                multivariate_coeffs_into_hypercube_evals(
                    &mut source,
                    sumchecked_poly_monomial_form.len().trailing_zeros(),
                );
                bitreverse_enumeration_inplace(&mut source);
                assert_eq!(source, sumchecked_poly_evaluation_form);
            }

            // self-check that our domain evaluations from monomial form match pows (so, RS code) definition
            {
                let omega = domain_generator_for_size::<F>(query_domain_size);
                for i in 0..query_domain_size {
                    let root = omega.pow(i as u32);
                    let eval_from_monomial = evaluate_monomial_form(
                        &sumchecked_poly_monomial_form,
                        &E::from_base(root),
                        worker,
                    );
                    dbg!((i, root, eval_from_monomial));
                    let t = sumchecked_poly_evaluation_form.to_vec();
                    let pows = make_pows(
                        root,
                        sumchecked_poly_evaluation_form.len().trailing_zeros() as usize,
                    );
                    // pows.reverse();
                    let eval_from_multivariate = evaluate_multivariate_at_base(&t, &pows);
                    assert_eq!(eval_from_monomial, eval_from_multivariate);
                }
            }
        }

        // we now update the equality poly - initially we had eq(X, original_evalution_point), from which we folded few coordinates.
        // Now we should add more terms there to reflect OOD and in-domain samples
        update_eq_poly(
            eq_poly,
            &contributions_to_eq_poly,
            &contributions_to_eq_poly_with_base_points,
        );

        // and remember new sumcheck claim
        dbg!(&claim);
        dbg!(&claim_correction);
        claim.add_assign(&claim_correction);
    }

    // and final step is almost the same as the first one - we can fold few times, output evaluation form, and draw final query indexes,
    // check consistency between them, and perform final explicit sumcheck
    {
        let num_folding_steps = whir_steps_schedule.next().unwrap();
        let num_queries = whir_queries_schedule.next().unwrap();
        assert!(num_folding_steps <= poly_size_log2);

        let rs_domain_log2 = poly_size_log2 + (rs_oracle.cosets.len().trailing_zeros() as usize);
        let query_domain_log2 = rs_domain_log2 - num_folding_steps;

        // fold and send explicit form

        let mut folding_challenges_in_round = Vec::with_capacity(num_folding_steps);

        for _ in 0..num_folding_steps {
            let (f0, f1, f_half) = special_three_point_eval(
                &sumchecked_poly_evaluation_form[..],
                &eq_poly[..],
                worker,
            );
            let univariate_coeffs =
                special_lagrange_interpolate(f0, f1, f_half, E::from_base(two_inv));
            // send the monomial form to the verifier

            {
                // self-check
                let s0 = evaluate_small_univariate_poly(&univariate_coeffs, &E::ZERO);
                assert_eq!(s0, f0);
                let s1 = evaluate_small_univariate_poly(&univariate_coeffs, &E::ONE);
                assert_eq!(s1, f1);
                let s_half =
                    evaluate_small_univariate_poly(&univariate_coeffs, &E::from_base(two_inv));
                assert_eq!(s_half, f_half);
                let mut v = s0;
                v.add_assign(&s1);
                assert_eq!(v, claim);
            }

            let folding_challenge = E::from_base(F::from_u32_unchecked(42));
            let folding_challenge = E::ZERO;
            folding_challenges_in_round.push(folding_challenge);

            let next_claim = evaluate_small_univariate_poly(&univariate_coeffs, &folding_challenge);
            claim = next_claim;
            // and fold the poly itself - both multivariate evals mapping, and monomial form

            fold_monomial_form(
                &mut sumchecked_poly_monomial_form,
                &mut monomial_form_buffer,
                &folding_challenge,
                worker,
            );

            sumchecked_poly_evaluation_form = fold_evaluation_form(
                &mut sumchecked_poly_evaluation_form[..],
                &folding_challenge,
                worker,
            );

            assert_eq!(
                sumchecked_poly_monomial_form.len(),
                sumchecked_poly_evaluation_form.len()
            );

            eq_poly = fold_eq_poly(eq_poly, &folding_challenge, worker);
            assert_eq!(sumchecked_poly_evaluation_form.len(), eq_poly.len());
        }

        poly_size_log2 -= num_folding_steps;

        assert_eq!(sumchecked_poly_evaluation_form.len(), 1 << poly_size_log2);
        assert_eq!(sumchecked_poly_monomial_form.len(), 1 << poly_size_log2);
        assert_eq!(eq_poly.len(), 1 << poly_size_log2);

        dbg!(&sumchecked_poly_evaluation_form);
        dbg!(&sumchecked_poly_monomial_form);
        dbg!(&eq_poly);

        // query

        let rs_oracle_to_query = rs_oracle;
        let query_domain_size = 1u64 << query_domain_log2;

        dbg!(query_domain_size);
        let query_domain_generator = domain_generator_for_size::<F>(query_domain_size);
        let input_domain_size = 1u64 << rs_domain_log2;
        let extended_generator = domain_generator_for_size::<F>(input_domain_size);

        let set_generator = domain_generator_for_size::<F>(1u64 << num_folding_steps);
        let mut folding_set_offsets: Vec<F> = (0..(1u64 << (num_folding_steps - 1)))
            .map(|el| set_generator.pow(el as u32))
            .collect();
        bitreverse_enumeration_inplace(&mut folding_set_offsets);

        let cosets_invs: Vec<_> = rs_oracle_to_query
            .cosets
            .iter()
            .map(|el| el.values_normal_order.offset.inverse().unwrap())
            .collect();
        let coset_invs_powers: Vec<Vec<F>> = cosets_invs
            .iter()
            .map(|el| {
                (0..num_folding_steps)
                    .map(|pow| el.pow((1 << pow) as u32))
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut query_indexes = vec![];
        for _ in 0..num_queries {
            // query index is power for omega^k expression
        }
        {
            query_indexes.push(0usize);
            query_indexes.push(1usize);
            query_indexes.push(2usize);
            query_indexes.push(3usize);
        }

        for query_index in query_indexes.into_iter() {
            assert!(query_index < query_domain_size as usize);
            let query_point = query_domain_generator.pow(query_index as u32);

            // we have a query point, and now we need to understand "preimages" for it
            // assume that we have a query point omega_q ^ i, and we fold K times.
            // Then what we need is a set of points omega^{0 << (log(query_domain)) || i }, omega^{1 << (log(query_domain)) || i}, etc
            // where omega = root(omega_q, 2^K)

            // So for "base root" we take omega^i from the notations above, and when we fold we multiply it by root(1, 2^K)

            // get original leaf, compute batched, and then folded value
            let base_root = extended_generator.pow(query_index as u32);
            let base_root_inv = base_root.inverse().unwrap();

            let (coset_idx, mut evals, query) =
                rs_oracle_to_query.query_for_folded_index(query_index);
            let intermediate_oracle = proof.intermediate_whir_oracles.last_mut().unwrap();
            intermediate_oracle.queries.push(query);

            // Now we can fold queries values, in a normal FRI style
            let mut buffer = Vec::with_capacity(evals.len());
            for folding_step in 0..num_folding_steps {
                let (src, dst) = if folding_step % 2 == 0 {
                    (&evals[..], &mut buffer)
                } else {
                    (&buffer[..], &mut evals)
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
                    evals.clear();
                } else {
                    buffer.clear();
                };
            }

            let folded = if num_folding_steps % 2 == 1 {
                &buffer[..]
            } else {
                &evals[..]
            };
            assert_eq!(folded.len(), 1);

            dbg!((query_index, query_point, folded[0]));

            // check against explicit form
            let eval_from_monomial = evaluate_monomial_form(
                &sumchecked_poly_monomial_form,
                &E::from_base(query_point),
                worker,
            );
            assert_eq!(eval_from_monomial, folded[0]);
        }

        assert_eq!(sumchecked_poly_monomial_form.len(), 1 << final_poly_log2);

        // conclude the sumcheck - we self-check as would the verifier do

        // It is quite tricky part, as we have to take the values, and
        // form the following sumcheck kernel

        // \sum_{X} [ eq(original_eval_point_remaining, X) + delin_challenge_1^{1} eq(pows(ood_sample_1, pows_size_1).remaining, X) +
        // + \sum delin_challenge_1^{i+1} eq(pows(query_roots_1_{i}, pows_size_1).remaining, X) +
        // + delin_challenge_2^{1} eq(pows(ood_sample_2, pows_size_2).remaining, X) + ...]

        // but at the prover we have eq poly, so we just sum it
        {
            let value = dot_product(&sumchecked_poly_evaluation_form[..], &eq_poly[..], worker);
            assert_eq!(value, claim);
        }
    }

    assert!(whir_steps_lde_factors.next().is_none());
    assert!(whir_steps_schedule.next().is_none());
    assert!(whir_queries_schedule.next().is_none());

    proof
}

fn commit_single_ext_poly<
    F: PrimeField + TwoAdicField,
    E: FieldExtension<F> + Field,
    T: ColumnMajorMerkleTreeConstructor<F>,
>(
    cosets: Vec<(Box<[E]>, F)>,
    values_per_leaf: usize,
    cap_size: usize,
    worker: &Worker,
) -> ColumnMajorExtensionOracleForLDE<F, E, T>
where
    [(); E::DEGREE]: Sized,
{
    let mut result = ColumnMajorExtensionOracleForLDE { cosets: vec![] };
    for (column, offset) in cosets.into_iter() {
        let trace_len_log2 = column.len().trailing_zeros() as usize;
        let tree = T::construct_for_column_major_coset::<E, Global>(
            &[&column[..]],
            values_per_leaf,
            cap_size,
            true,
            false,
            worker,
        );
        let el = ColumnMajorExtensionOracleForCoset {
            values_normal_order: ColumnMajorCosetBoundTracePart {
                column: Arc::new(column),
                offset,
            },
            tree,
            values_per_leaf,
            trace_len_log2,
        };
        result.cosets.push(el);
    }

    result
}

fn fold_monomial_form<E: Field>(
    input: &mut Vec<E>,
    buffer: &mut Vec<E>,
    challenge: &E,
    worker: &Worker,
) {
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
    challenge: &E,
    worker: &Worker,
) -> &'a mut [E] {
    // TODO: parallelize
    assert!(input.len().is_power_of_two());
    let half_len = input.len() / 2;
    let stride = input.len() / 2;
    let mut f0_coeff = E::ONE;
    f0_coeff.sub_assign(challenge);

    for i in 0..input.len() / 2 {
        let mut f0 = input[i];
        f0.mul_assign(&f0_coeff);
        let mut f1 = input[i + stride];
        f1.mul_assign(&challenge);

        f0.add_assign(&f1);

        input[i] = f0;
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
    assert!(eq_poly.len() >= 2);
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

fn special_three_point_eval<F: PrimeField, E: FieldExtension<F> + Field>(
    a: &[E],
    b: &[E],
    worker: &Worker,
) -> (E, E, E) {
    // TODO: parallelize
    assert!(a.len() > 0);
    assert_eq!(a.len(), b.len());

    let quart = F::from_u32_unchecked(4).inverse().unwrap();

    let mut f0 = E::ZERO;
    let mut f1 = E::ZERO;
    let mut f_half = E::ZERO;
    let stride = a.len() / 2;

    for i in 0..a.len() / 2 {
        let a0 = a[i];
        let a1 = a[i + stride];
        let b0 = b[i];
        let b1 = b[i + stride];

        let mut t0 = a0;
        t0.mul_assign(&b0);
        f0.add_assign(&t0);

        let mut t1 = a1;
        t1.mul_assign(&b1);
        f1.add_assign(&t1);

        let mut tt = a1;
        tt.sub_assign(&a0);
        let mut t_half = b1;
        t_half.sub_assign(&b0);
        t_half.mul_assign(&tt);
        f_half.add_assign(&t_half);
    }

    f_half.mul_assign_by_base(&quart);

    (f0, f1, f_half)
}

fn evaluate_monomial_form<E: Field>(coeffs: &[E], point: &E, worker: &Worker) -> E {
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

fn make_pows<E: Field>(el: E, num_powers: usize) -> Vec<E> {
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

fn evaluate_base_multivariate<F: PrimeField, E: FieldExtension<F> + Field>(
    evals: &[F],
    point: &[E],
) -> E {
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

fn evaluate_multivariate_at_base<F: PrimeField, E: FieldExtension<F> + Field>(
    evals: &[E],
    point: &[F],
) -> E {
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

fn evaluate_multivariate_at_base_for_domain_hypercube<
    F: PrimeField + TwoAdicField,
    E: FieldExtension<F> + Field,
>(
    evals: &[E],
    point: &[F],
) -> E {
    let mut eqs = make_domain_eq_poly_in_full::<F, F>(point);
    let eq = eqs.pop().unwrap();
    dbg!(&eq);
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
    use fft::materialize_powers_parallel_starting_with_one;
    use field::baby_bear::{base::BabyBearField, ext4::BabyBearExt4};
    type F = BabyBearField;
    type E = BabyBearExt4;

    // fn make_base_oracle(
    //     size: usize,
    //     worker: &Worker,
    // ) -> ColumnMajorBaseOracleForLDE<F, Blake2sU32MerkleTreeWithCap> {
    //     let main_domain: Vec<F> = (1..=size).map(|el| {
    //         F::from_u32_unchecked(el as u32)
    //     }).collect();
    //     let twiddles = Twiddles::<F, Global>::new(size, worker);
    //     let main_domain = Arc::new(main_domain.into_boxed_slice());

    //     let other_domains =
    //         compute_column_major_lde_from_main_domain(main_domain.clone(), &twiddles, 2);
    //     let original_values_normal_order = ColumnMajorCosetBoundTracePart {
    //         column: main_domain,
    //         offset: F::ONE,
    //     };
    //     let source = Some(original_values_normal_order)
    //         .into_iter()
    //         .chain(other_domains.into_iter());

    //     let mut result = ColumnMajorBaseOracleForLDE { cosets: vec![] };
    //     for coset in source {
    //         let el = ColumnMajorBaseOracleForCoset {
    //             original_values_normal_order: vec![coset],
    //             tree: <Blake2sU32MerkleTreeWithCap as ColumnMajorMerkleTreeConstructor<F>>::dummy(),
    //             values_per_leaf: 2,
    //             trace_len_log2: size.trailing_zeros() as usize,
    //         };
    //         result.cosets.push(el);
    //     }

    //     result
    // }

    fn make_base_oracle(
        size: usize,
        worker: &Worker,
    ) -> (
        ColumnMajorBaseOracleForLDE<F, Blake2sU32MerkleTreeWithCap>,
        Vec<F>,
    ) {
        let coeffs: Vec<F> = (1..=size)
            .map(|el| F::from_u32_unchecked(el as u32))
            .collect();
        let twiddles = Twiddles::<F, Global>::new(size, worker);

        let cosets = compute_column_major_lde_from_monomial_form(&coeffs, &twiddles, 2);

        let mut result = ColumnMajorBaseOracleForLDE { cosets: vec![] };
        for (column, offset) in cosets.into_iter() {
            let el = ColumnMajorBaseOracleForCoset {
                original_values_normal_order: vec![ColumnMajorCosetBoundTracePart {
                    column: Arc::new(column),
                    offset,
                }],
                tree: <Blake2sU32MerkleTreeWithCap as ColumnMajorMerkleTreeConstructor<F>>::dummy(),
                values_per_leaf: 2,
                trace_len_log2: size.trailing_zeros() as usize,
            };
            result.cosets.push(el);
        }

        (result, coeffs)
    }

    #[test]
    fn test_domain_hypercube_evals() {
        let worker = Worker::new_with_num_threads(1);
        let size: usize = 4;

        let main_domain: Vec<F> = (1..=size)
            .map(|el| F::from_u32_unchecked(el as u32))
            .collect();
        dbg!(&main_domain);

        let root = domain_generator_for_size::<F>(size as u64);
        let domain = materialize_powers_serial_starting_with_one::<F, Global>(root, size);
        dbg!(&domain);
        for i in 0..size {
            dbg!(i);
            let domain_point = root.pow(i as u32);
            let pows = make_pows(domain_point, size.trailing_zeros() as usize);
            dbg!(&pows);
            let value = evaluate_multivariate_at_base_for_domain_hypercube(&main_domain, &pows);
            dbg!(value);
        }
    }

    #[test]
    fn test_whir() {
        let worker = Worker::new_with_num_threads(1);
        let size = 8;

        let mut inputs = vec![];
        let mut monomial_forms = vec![];
        for _ in 0..3 {
            let (input, monomial) = make_base_oracle(size, &worker);
            inputs.push(input);
            monomial_forms.push(monomial);
        }

        let inputs: [_; 3] = inputs.try_into().unwrap();

        let original_evaluation_point = vec![
            E::from_base(F::from_u32_unchecked(4)),
            E::from_base(F::from_u32_unchecked(8)),
            E::from_base(F::from_u32_unchecked(16)),
        ];
        let twiddles = Twiddles::<F, Global>::new(size, &worker);

        let original_claims: Vec<_> = monomial_forms
            .iter()
            .map(|el| {
                // compute on hypercube
                let mut t = el.to_vec();
                bitreverse_enumeration_inplace(&mut t);
                multivariate_coeffs_into_hypercube_evals(&mut t, size.trailing_zeros());
                let eval = evaluate_base_multivariate(&t, &original_evaluation_point);

                vec![eval]
            })
            .collect::<Vec<_>>();

        let original_claims: [_; 3] = original_claims.try_into().unwrap();

        dbg!(&original_claims);

        whir_fold(
            inputs,
            original_claims,
            original_evaluation_point,
            2,
            &E::ONE,
            vec![1, 1, 1],
            vec![1, 1, 1],
            vec![8, 8],
            &twiddles,
            1,
            size.trailing_zeros() as usize,
            &worker,
        );
    }
}
