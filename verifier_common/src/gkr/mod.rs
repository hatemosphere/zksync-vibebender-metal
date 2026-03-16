use core::mem::MaybeUninit;

use blake2s_u32::{AlignedArray64, DelegatedBlake2sState, BLAKE2S_DIGEST_SIZE_U32_WORDS};
use cs::definitions::gkr_static_types::{
    OutputType, StaticGKRLayerDescription, StaticNoFieldGKRRelation,
    StaticNoFieldMaxQuadraticConstraintsGKRRelation,
};
use cs::definitions::GKRAddress;
use field::{Field, FieldExtension, PrimeField};
use non_determinism_source::NonDeterminismSource;
use transcript::{Blake2sTranscript, Seed};

#[cfg(any(test, feature = "proof_utils"))]
pub mod flatten;

const DRAW_BUF_CAPACITY: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct GKROutputGroup {
    pub output_type: OutputType,
    pub num_addresses: usize,
}

#[derive(Clone, Debug)]
pub struct GKRLayerMeta<'a> {
    pub is_dim_reducing: usize, // 0 = standard (N=2), nonzero = dim-reducing (N=4)
    pub num_sumcheck_rounds: usize,
    pub output_groups: &'a [GKROutputGroup],
    pub layer_desc: Option<&'a StaticGKRLayerDescription<'a>>,
    pub sorted_dedup_input_addrs: &'a [GKRAddress],
    /// For dim-reducing layers: mapping from iteration order (output_groups walk)
    /// to sorted position in sorted_dedup_input_addrs.
    /// Empty for standard layers.
    pub input_sorted_indices: &'a [usize],
}

#[derive(Clone, Debug)]
pub struct GKRVerifierConfig<'a> {
    pub layers: &'a [GKRLayerMeta<'a>],
    pub has_inits_teardowns: usize, // 0 = false, nonzero = true
    pub initial_transcript_num_u32_words: usize,
    pub final_trace_size_log_2: usize,
    pub num_standard_layers: usize,
    pub global_input_addrs: &'a [GKRAddress],
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct LazyVec<V: Copy, const N: usize> {
    data: [MaybeUninit<V>; N],
    len: usize,
}

impl<V: Copy, const N: usize> LazyVec<V, N> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            data: [MaybeUninit::uninit(); N],
            len: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, val: V) {
        debug_assert!(self.len < N);
        unsafe {
            self.data.get_unchecked_mut(self.len).write(val);
        }
        self.len += 1;
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> &V {
        debug_assert!(idx < self.len);
        unsafe { self.data.get_unchecked(idx).assume_init_ref() }
    }

    #[inline(always)]
    fn as_slice(&self) -> &[V] {
        unsafe { core::slice::from_raw_parts(self.data.as_ptr() as *const V, self.len) }
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.len = 0;
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub unsafe fn as_array(self) -> [V; N] {
        MaybeUninit::array_assume_init(self.data)
    }
}

#[inline(always)]
fn read_field_el<F: PrimeField, E: FieldExtension<F>, I: NonDeterminismSource>() -> E
where
    [(); E::DEGREE]: Sized,
{
    use field::FixedArrayConvertible;
    let mut array = [F::ZERO; E::DEGREE];
    for a in array.iter_mut() {
        let value = I::read_word();
        *a = F::from_reduced_raw_repr(value);
    }
    let coeffs = E::Coeffs::from_array(array);
    E::from_coeffs(coeffs)
}

#[inline(always)]
fn read_field_els<F: PrimeField, E: FieldExtension<F>, I: NonDeterminismSource>(dst: &mut [E])
where
    [(); E::DEGREE]: Sized,
{
    for el in dst.iter_mut() {
        *el = read_field_el::<F, E, I>();
    }
}

#[inline(always)]
fn commit_field_els<F: PrimeField, E: FieldExtension<F>>(seed: &mut Seed, els: &[E])
where
    [(); E::DEGREE]: Sized,
{
    use core::mem::{align_of, size_of};
    assert!(size_of::<F>() == size_of::<u32>());
    assert!(align_of::<F>() == align_of::<u32>());

    let total = els.len() * E::DEGREE;
    let as_u32 = unsafe { core::slice::from_raw_parts(els.as_ptr() as *const u32, total) };
    Blake2sTranscript::commit_with_seed(seed, as_u32);
}

#[inline(always)]
fn draw_field_els_into<F: PrimeField, E: FieldExtension<F>>(seed: &mut Seed, dst: &mut [E])
where
    [(); E::DEGREE]: Sized,
{
    use field::FixedArrayConvertible;
    let n = dst.len();
    let padded = (n * E::DEGREE).next_multiple_of(BLAKE2S_DIGEST_SIZE_U32_WORDS);
    assert!(padded <= DRAW_BUF_CAPACITY, "draw buffer too small");

    let mut words = [MaybeUninit::<u32>::uninit(); DRAW_BUF_CAPACITY];
    let mut arr = [F::ZERO; E::DEGREE];

    unsafe {
        let dst = core::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u32, padded);
        Blake2sTranscript::draw_randomness(seed, dst);
    }

    for (i, chunk) in words[..n * E::DEGREE]
        .chunks_exact(E::DEGREE)
        .map(|chunk| unsafe { chunk.assume_init_ref() })
        .enumerate()
    {
        for (j, &w) in chunk.iter().enumerate() {
            arr[j] = F::from_u32_with_reduction(w);
        }
        let fixed = E::Coeffs::from_array(arr);
        dst[i] = E::from_coeffs(fixed);
    }
}

#[inline(always)]
fn draw_one_field_el<F: PrimeField, E: FieldExtension<F> + Field>(seed: &mut Seed) -> E
where
    [(); E::DEGREE]: Sized,
{
    let mut dst = [E::ZERO; 1];
    draw_field_els_into::<F, E>(seed, &mut dst);
    dst[0]
}

#[derive(Clone, Debug)]
pub enum GKRVerificationError {
    SumcheckRoundFailed { layer: usize, round: usize },
    FinalStepCheckFailed { layer: usize },
}

#[inline(always)]
fn dot_eq<E: Field>(values: &[E], eq: &[E]) -> E {
    assert_eq!(values.len(), eq.len());
    let mut result = E::ZERO;
    for (v, e) in values.iter().zip(eq.iter()) {
        let mut t = *v;
        t.mul_assign(e);
        result.add_assign(&t);
    }
    result
}

#[inline(always)]
fn eval_constraint_kernel<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const ADDRS: usize,
    const MAX_POW: usize,
>(
    rel: &StaticNoFieldMaxQuadraticConstraintsGKRRelation,
    challenge_powers: &[E; MAX_POW],
    evals: &LazyVec<[E; 2], ADDRS>,
    j: usize,
) -> E {
    let get_value_at = |idx: usize| &evals.get(idx)[j];

    let mut result = E::ZERO;

    let get_pow = |pow: usize| unsafe { *challenge_powers.get_unchecked(pow) };

    // Coefficients are pre-converted to Montgomery form by the code generator,
    // so we use from_reduced_raw_repr
    for &(coeff, pow) in rel.constants.iter() {
        let mut t = get_pow(pow);
        let c = F::from_reduced_raw_repr(coeff);
        t.mul_assign_by_base(&c);
        result.add_assign(&t);
    }

    for (idx, terms) in rel.linear_terms.iter() {
        let val = get_value_at(*idx);
        for &(coeff, pow) in terms.iter() {
            let mut t = get_pow(pow);
            let c = F::from_reduced_raw_repr(coeff);
            t.mul_assign_by_base(&c);
            t.mul_assign(val);
            result.add_assign(&t);
        }
    }

    for ((idx_a, idx_b), terms) in rel.quadratic_terms.iter() {
        let va = get_value_at(*idx_a);
        let vb = get_value_at(*idx_b);
        let mut prod = *va;
        prod.mul_assign(vb);
        for &(coeff, pow) in terms.iter() {
            let mut t = get_pow(pow);
            let c = F::from_reduced_raw_repr(coeff);
            t.mul_assign_by_base(&c);
            t.mul_assign(&prod);
            result.add_assign(&t);
        }
    }

    result
}

#[inline(always)]
fn compute_standard_final_step_accumulator<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const ADDRS: usize,
    const MAX_POW: usize,
>(
    layer: &StaticGKRLayerDescription,
    final_step_evaluations: &LazyVec<[E; 2], ADDRS>,
    batching_challenge: E,
    lookup_additive_challenge: E,
    constraints_batch_challenge: E,
) -> [E; 2]
where
    [(); E::DEGREE]: Sized,
{
    use StaticNoFieldGKRRelation as Rel;

    let mut acc = [E::ZERO; 2];
    let mut current_batch = E::ONE;
    let batch_base = batching_challenge;

    let mut challenge_powers = [E::ZERO; MAX_POW];
    challenge_powers[0] = E::ONE;
    challenge_powers.iter_mut().reduce(|prev, cur| {
        *cur = *prev;
        cur.mul_assign(&constraints_batch_challenge);
        cur
    });

    let relations = layer
        .gates
        .iter()
        .chain(layer.gates_with_external_connections.iter())
        .map(|g| &g.enforced_relation);

    for relation in relations {
        let mut get_challenge = || {
            let c = current_batch;
            current_batch.mul_assign(&batch_base);
            c
        };

        match relation {
            Rel::EnforceConstraintsMaxQuadratic { input } => {
                let bc = get_challenge();
                for (j, acc_j) in acc.iter_mut().enumerate() {
                    let val = eval_constraint_kernel::<F, E, ADDRS, MAX_POW>(
                        input,
                        &challenge_powers,
                        final_step_evaluations,
                        j,
                    );
                    let mut contrib = bc;
                    contrib.mul_assign(&val);
                    acc_j.add_assign(&contrib);
                }
            }
            Rel::Copy { .. }
            | Rel::InitialGrandProductFromCaches { .. }
            | Rel::TrivialProduct { .. }
            | Rel::MaskIntoIdentityProduct { .. } => {
                let bc = get_challenge();
                for (j, acc_j) in acc.iter_mut().enumerate() {
                    let val = eval_single_output_relation::<F, E, ADDRS>(
                        relation,
                        final_step_evaluations,
                        lookup_additive_challenge,
                        j,
                    );
                    let mut contrib = bc;
                    contrib.mul_assign(&val);
                    acc_j.add_assign(&contrib);
                }
            }
            Rel::LookupPair { .. }
            | Rel::LookupPairFromMaterializedBaseInputs { .. }
            | Rel::LookupFromMaterializedBaseInputWithSetup { .. }
            | Rel::LookupUnbalancedPairWithMaterializedBaseInputs { .. }
            | Rel::LookupWithCachedDensAndSetup { .. } => {
                let bc0 = get_challenge();
                let bc1 = get_challenge();
                for (j, acc_j) in acc.iter_mut().enumerate() {
                    let [out0, out1] = eval_two_output_relation::<F, E, ADDRS>(
                        relation,
                        final_step_evaluations,
                        lookup_additive_challenge,
                        j,
                    );
                    let mut c0 = bc0;
                    c0.mul_assign(&out0);
                    let mut c1 = bc1;
                    c1.mul_assign(&out1);
                    acc_j.add_assign(&c0);
                    acc_j.add_assign(&c1);
                }
            }
            Rel::UnbalancedGrandProductWithCache { .. }
            | Rel::MaterializeSingleLookupInput { .. }
            | Rel::MaterializedVectorLookupInput { .. }
            | Rel::LookupPairFromBaseInputs { .. }
            | Rel::LookupUnbalancedPairWithBaseInputs { .. }
            | Rel::LookupFromBaseInputsWithSetup { .. }
            | Rel::LookupPairFromVectorInputs { .. } => {
                panic!("unimplemented relation variant in final step check");
            }
        }
    }

    acc
}

#[inline(always)]
fn eval_single_output_relation<F: PrimeField, E: FieldExtension<F> + Field, const ADDRS: usize>(
    relation: &StaticNoFieldGKRRelation,
    evals: &LazyVec<[E; 2], ADDRS>,
    _lookup_additive_challenge: E,
    j: usize,
) -> E {
    use StaticNoFieldGKRRelation as Rel;
    let get = |idx: usize| &evals.get(idx)[j];

    match relation {
        Rel::Copy { input, .. } => *get(*input),
        Rel::InitialGrandProductFromCaches { input, .. } | Rel::TrivialProduct { input, .. } => {
            let mut v = *get(input[0]);
            v.mul_assign(get(input[1]));
            v
        }
        Rel::MaskIntoIdentityProduct { input, mask, .. } => {
            let input_val = get(*input);
            let mask_val = get(*mask);
            let mut out = *input_val;
            out.sub_assign_base(&F::ONE);
            out.mul_assign(mask_val);
            out.add_assign_base(&F::ONE);
            out
        }
        _ => unreachable!("eval_single_output_relation called with non-single-output relation"),
    }
}

#[inline(always)]
fn eval_two_output_relation<F: PrimeField, E: FieldExtension<F> + Field, const ADDRS: usize>(
    relation: &StaticNoFieldGKRRelation,
    evals: &LazyVec<[E; 2], ADDRS>,
    lookup_additive_challenge: E,
    j: usize,
) -> [E; 2] {
    use StaticNoFieldGKRRelation as Rel;
    let get = |idx: usize| &evals.get(idx)[j];
    let gamma = lookup_additive_challenge;

    match relation {
        Rel::LookupPair { input, .. } => {
            let a = get(input[0][0]);
            let b = get(input[0][1]);
            let c = get(input[1][0]);
            let d = get(input[1][1]);
            let mut num = *a;
            num.mul_assign(d);
            let mut cb = *c;
            cb.mul_assign(b);
            num.add_assign(&cb);
            let mut den = *b;
            den.mul_assign(d);
            [num, den]
        }
        Rel::LookupPairFromMaterializedBaseInputs { input, .. } => {
            let mut b_g = *get(input[0]);
            b_g.add_assign(&gamma);
            let mut d_g = *get(input[1]);
            d_g.add_assign(&gamma);
            let mut num = b_g;
            num.add_assign(&d_g);
            let mut den = b_g;
            den.mul_assign(&d_g);
            [num, den]
        }
        Rel::LookupFromMaterializedBaseInputWithSetup { input, setup, .. } => {
            let mut b_g = *get(*input);
            b_g.add_assign(&gamma);
            let mut d_g = *get(setup[1]);
            d_g.add_assign(&gamma);
            let mut cb_g = *get(setup[0]);
            cb_g.mul_assign(&b_g);
            let mut num = d_g;
            num.sub_assign(&cb_g);
            let mut den = b_g;
            den.mul_assign(&d_g);
            [num, den]
        }
        Rel::LookupUnbalancedPairWithMaterializedBaseInputs {
            input, remainder, ..
        } => {
            let a = get(input[0]);
            let b = get(input[1]);
            let mut d_g = *get(*remainder);
            d_g.add_assign(&gamma);
            let mut num = *a;
            num.mul_assign(&d_g);
            num.add_assign(b);
            let mut den = *b;
            den.mul_assign(&d_g);
            [num, den]
        }
        Rel::LookupWithCachedDensAndSetup { input, setup, .. } => {
            let a = get(input[0]);
            let b = get(input[1]);
            let c = get(setup[0]);
            let d = get(setup[1]);
            let mut ad = *a;
            ad.mul_assign(d);
            let mut cb = *c;
            cb.mul_assign(b);
            ad.sub_assign(&cb);
            let mut den = *b;
            den.mul_assign(d);
            [ad, den]
        }
        _ => unreachable!("eval_two_output_relation called with non-two-output relation"),
    }
}

#[inline(always)]
fn compute_dim_reducing_final_step_accumulator<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const ADDRS: usize,
>(
    output_groups: &[GKROutputGroup],
    input_sorted_indices: &[usize],
    final_step_evaluations: &LazyVec<[E; 4], ADDRS>,
    batching_challenge: E,
) -> [E; 2]
where
    [(); E::DEGREE]: Sized,
{
    let mut acc = [E::ZERO; 2];
    let mut current_batch = E::ONE;
    let batch_base = batching_challenge;

    let mut get_challenge = || {
        let c = current_batch;
        current_batch.mul_assign(&batch_base);
        c
    };

    let mut iter_idx = 0;
    for group in output_groups.iter() {
        match group.output_type {
            OutputType::PermutationProduct => {
                for _ in 0..group.num_addresses {
                    debug_assert!(iter_idx < input_sorted_indices.len());
                    let sorted_idx = unsafe { *input_sorted_indices.get_unchecked(iter_idx) };
                    iter_idx += 1;
                    let bc = get_challenge();
                    let evals = final_step_evaluations.get(sorted_idx);
                    unsafe {
                        let mut v01 = *evals.get_unchecked(0);
                        v01.mul_assign(evals.get_unchecked(1));
                        let mut c0 = bc;
                        c0.mul_assign(&v01);
                        acc[0].add_assign(&c0);
                        let mut v23 = *evals.get_unchecked(2);
                        v23.mul_assign(evals.get_unchecked(3));
                        let mut c1 = bc;
                        c1.mul_assign(&v23);
                        acc[1].add_assign(&c1);
                    }
                }
            }
            OutputType::Lookup16Bits | OutputType::LookupTimestamps | OutputType::GenericLookup => {
                let bc0 = get_challenge();
                let bc1 = get_challenge();
                debug_assert!(iter_idx + 1 < input_sorted_indices.len());
                let si0 = unsafe { *input_sorted_indices.get_unchecked(iter_idx) };
                let si1 = unsafe { *input_sorted_indices.get_unchecked(iter_idx + 1) };
                iter_idx += 2;
                let v0 = final_step_evaluations.get(si0);
                let v1 = final_step_evaluations.get(si1);
                unsafe {
                    // j=0
                    let mut num0 = *v0.get_unchecked(0);
                    num0.mul_assign(v1.get_unchecked(1));
                    let mut cb0 = *v0.get_unchecked(1);
                    cb0.mul_assign(v1.get_unchecked(0));
                    num0.add_assign(&cb0);
                    let mut den0 = *v1.get_unchecked(0);
                    den0.mul_assign(v1.get_unchecked(1));
                    let mut c00 = bc0;
                    c00.mul_assign(&num0);
                    let mut c01 = bc1;
                    c01.mul_assign(&den0);
                    acc[0].add_assign(&c00);
                    acc[0].add_assign(&c01);
                    // j=1
                    let mut num1 = *v0.get_unchecked(2);
                    num1.mul_assign(v1.get_unchecked(3));
                    let mut cb1 = *v0.get_unchecked(3);
                    cb1.mul_assign(v1.get_unchecked(2));
                    num1.add_assign(&cb1);
                    let mut den1 = *v1.get_unchecked(2);
                    den1.mul_assign(v1.get_unchecked(3));
                    let mut c10 = bc0;
                    c10.mul_assign(&num1);
                    let mut c11 = bc1;
                    c11.mul_assign(&den1);
                    acc[1].add_assign(&c10);
                    acc[1].add_assign(&c11);
                }
            }
        }
    }

    acc
}

#[inline(always)]
fn make_eq_poly_last<E: Field>(challenges: &[E], buf: &mut [E]) {
    assert!(!challenges.is_empty());
    let n = challenges.len();
    assert!(buf.len() >= 1 << n);

    buf[0] = E::ONE;
    let mut size = 1usize;
    let mut idx = n;
    for _ in 0..n {
        idx -= 1;
        debug_assert!(idx < challenges.len());
        let c = unsafe { *challenges.get_unchecked(idx) };
        let f1 = c;
        let mut f0 = E::ONE;
        f0.sub_assign(&c);
        let half = size;

        for i in (0..half).rev() {
            debug_assert!(i < buf.len());
            debug_assert!(i + half < buf.len());
            let prev = unsafe { *buf.get_unchecked(i) };
            let mut left = prev;
            let mut right = prev;
            left.mul_assign(&f0);
            right.mul_assign(&f1);
            unsafe {
                *buf.get_unchecked_mut(i) = left;
                *buf.get_unchecked_mut(i + half) = right;
            }
        }
        size *= 2;
    }
}

#[inline(always)]
fn compute_standard_layer_claim<F: PrimeField, E: FieldExtension<F> + Field, const ADDRS: usize>(
    layer: &StaticGKRLayerDescription,
    output_claims: &LazyVec<E, ADDRS>,
    batching_challenge: E,
) -> E {
    let gates = layer
        .gates
        .iter()
        .chain(layer.gates_with_external_connections.iter())
        .map(|g| &g.enforced_relation);

    let mut combined = E::ZERO;
    let mut current_batch = E::ONE;
    let batch_base = batching_challenge;

    for relation in gates {
        let mut get_challenge = || {
            let c = current_batch;
            current_batch.mul_assign(&batch_base);
            c
        };
        add_relation_claim(relation, output_claims, &mut get_challenge, &mut combined);
    }
    combined
}

#[inline(always)]
fn add_relation_claim<F: PrimeField, E: FieldExtension<F> + Field, const ADDRS: usize>(
    relation: &StaticNoFieldGKRRelation,
    output_claims: &LazyVec<E, ADDRS>,
    get_challenge: &mut impl FnMut() -> E,
    combined: &mut E,
) {
    use StaticNoFieldGKRRelation as Rel;
    match relation {
        Rel::EnforceConstraintsMaxQuadratic { .. } => {
            let _ = get_challenge();
        }
        Rel::Copy { output, .. }
        | Rel::InitialGrandProductFromCaches { output, .. }
        | Rel::TrivialProduct { output, .. }
        | Rel::MaskIntoIdentityProduct { output, .. }
        | Rel::UnbalancedGrandProductWithCache { output, .. }
        | Rel::MaterializeSingleLookupInput { output, .. }
        | Rel::MaterializedVectorLookupInput { output, .. } => {
            let bc = get_challenge();
            let claim = output_claims.get(*output);
            let mut t = bc;
            t.mul_assign(claim);
            combined.add_assign(&t);
        }
        Rel::LookupPair { output, .. }
        | Rel::LookupPairFromBaseInputs { output, .. }
        | Rel::LookupPairFromMaterializedBaseInputs { output, .. }
        | Rel::LookupUnbalancedPairWithBaseInputs { output, .. }
        | Rel::LookupUnbalancedPairWithMaterializedBaseInputs { output, .. }
        | Rel::LookupFromBaseInputsWithSetup { output, .. }
        | Rel::LookupFromMaterializedBaseInputWithSetup { output, .. }
        | Rel::LookupPairFromVectorInputs { output, .. }
        | Rel::LookupWithCachedDensAndSetup { output, .. } => {
            let bc0 = get_challenge();
            let bc1 = get_challenge();
            for (bc, idx) in [(bc0, output[0]), (bc1, output[1])] {
                let claim = output_claims.get(idx);
                let mut t = bc;
                t.mul_assign(claim);
                combined.add_assign(&t);
            }
        }
    }
}

#[inline(always)]
fn compute_dim_reducing_layer_claim<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const ADDRS: usize,
>(
    output_groups: &[GKROutputGroup],
    output_claims: &LazyVec<E, ADDRS>,
    batching_challenge: E,
) -> E {
    let mut combined = E::ZERO;
    let mut current_batch = E::ONE;
    let batch_base = batching_challenge;

    let mut get_challenge = || {
        let c = current_batch;
        current_batch.mul_assign(&batch_base);
        c
    };

    // For dim-reducing layers, output addresses are sequential InnerLayer addresses.
    // They are already in sorted order matching output_groups iteration,
    // so output index = iteration index.
    let mut out_idx = 0;
    for group in output_groups.iter() {
        match group.output_type {
            OutputType::PermutationProduct => {
                for _ in 0..group.num_addresses {
                    let bc = get_challenge();
                    let claim = output_claims.get(out_idx);
                    out_idx += 1;
                    let mut t = bc;
                    t.mul_assign(claim);
                    combined.add_assign(&t);
                }
            }
            OutputType::Lookup16Bits | OutputType::LookupTimestamps | OutputType::GenericLookup => {
                let bc0 = get_challenge();
                let bc1 = get_challenge();
                for bc in [bc0, bc1] {
                    let claim = output_claims.get(out_idx);
                    out_idx += 1;
                    let mut t = bc;
                    t.mul_assign(claim);
                    combined.add_assign(&t);
                }
            }
        }
    }
    combined
}

pub struct GKRVerifierOutput<'a, E: Field, const ROUNDS: usize, const ADDRS: usize> {
    pub base_layer_claims: LazyVec<E, ADDRS>,
    /// The sorted_dedup_input_addrs of the base (layer 0) config entry.
    /// Use this to map indices back to GKRAddress values.
    pub base_layer_addrs: &'a [GKRAddress],
    pub evaluation_point: [E; ROUNDS],
    pub evaluation_point_len: usize,
    pub grand_product_accumulator: E,
    pub additional_base_layer_openings: &'a [GKRAddress],
    pub whir_batching_challenge: E,
    pub whir_transcript_seed: Seed,
}

/// Run the regular sumcheck rounds (all rounds except the final step).
/// Returns (final_claim, final_eq_prefactor, folding_challenges, fc_len).
#[inline(always)]
fn verify_sumcheck_rounds<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    I: NonDeterminismSource,
    const ROUNDS: usize,
>(
    seed: &mut Seed,
    initial_claim: E,
    prev_point: &[E],
    layer_idx: usize,
) -> Result<(E, E, [E; ROUNDS], usize), GKRVerificationError>
where
    [(); E::DEGREE]: Sized,
{
    let num_regular_rounds = prev_point.len();
    let mut claim = initial_claim;
    let mut eq_prefactor = E::ONE;
    let mut folding_challenges: LazyVec<E, ROUNDS> = LazyVec::new();

    // Sumcheck coefficients: 4 ext field elements = 4 * E::DEGREE u32 words.
    // With seed (8 words), total = 8 + 4*DEGREE words.
    // For BabyBearExt4 (DEGREE=4): 24 words = 2 blocks of 16.
    let coeff_data_words = 4 * E::DEGREE;
    let total_commit_words = BLAKE2S_DIGEST_SIZE_U32_WORDS + coeff_data_words;

    // Aligned buffer: zero-initialized (padding stays zero), reused across rounds.
    let mut commit_buf: AlignedArray64<u32, 32> = AlignedArray64::from_value(0u32);
    let mut hasher = DelegatedBlake2sState::new();

    for round in 0..num_regular_rounds {
        // Copy seed into aligned buffer
        commit_buf[0..BLAKE2S_DIGEST_SIZE_U32_WORDS].copy_from_slice(&seed.0);

        // Read NDS words directly into the data region of the aligned buffer.
        // The words are Montgomery-form field elements, so the buffer can be
        // reinterpreted as &[E; 4] without conversion.
        for i in 0..coeff_data_words {
            commit_buf[BLAKE2S_DIGEST_SIZE_U32_WORDS + i] = I::read_word();
        }

        let coeffs =
            unsafe { &*(commit_buf.as_ptr().add(BLAKE2S_DIGEST_SIZE_U32_WORDS) as *const [E; 4]) };

        let p0 = coeffs[0];
        let mut p1 = coeffs[0];
        p1.add_assign(&coeffs[1]);
        p1.add_assign(&coeffs[2]);
        p1.add_assign(&coeffs[3]);

        let mut sum = p0;
        sum.add_assign(&p1);
        sum.mul_assign(&eq_prefactor);

        if sum != claim {
            return Err(GKRVerificationError::SumcheckRoundFailed {
                layer: layer_idx,
                round,
            });
        }

        Blake2sTranscript::commit_with_seed_using_hasher_and_aligned_buffer(
            &mut hasher,
            seed,
            &commit_buf,
            total_commit_words,
        );
        let r_k = draw_one_field_el::<F, E>(seed);

        // Fused eval_cubic_poly + eval_eq to share r_k in registers
        {
            let mut result = coeffs[3];
            result.mul_assign(&r_k);
            result.add_assign(&coeffs[2]);
            result.mul_assign(&r_k);
            result.add_assign(&coeffs[1]);
            result.mul_assign(&r_k);
            result.add_assign(&coeffs[0]);
            claim = result;
        }
        {
            debug_assert!(round < prev_point.len());
            let p = unsafe { *prev_point.get_unchecked(round) };
            let mut one_minus_r = E::ONE;
            one_minus_r.sub_assign(&r_k);
            let mut one_minus_p = E::ONE;
            one_minus_p.sub_assign(&p);
            let mut t = one_minus_r;
            t.mul_assign(&one_minus_p);
            let mut rp = r_k;
            rp.mul_assign(&p);
            t.add_assign(&rp);
            eq_prefactor = t;
        }

        folding_challenges.push(r_k);
    }

    let fc_len = folding_challenges.len();

    for _ in num_regular_rounds..ROUNDS {
        folding_challenges.push(E::ZERO);
    }
    let folding_challenges = unsafe { folding_challenges.as_array() };

    Ok((claim, eq_prefactor, folding_challenges, fc_len))
}

/// Verify the final step consistency: ((1 - last_prev) * f[0] + last_prev * f[1]) * eq_prefactor == claim.
#[inline(always)]
fn verify_final_step_check<F: PrimeField, E: FieldExtension<F> + Field>(
    f: [E; 2],
    last_prev_point: E,
    final_eq_prefactor: E,
    final_claim: E,
    layer_idx: usize,
) -> Result<(), GKRVerificationError> {
    let mut eq0 = E::ONE;
    eq0.sub_assign(&last_prev_point);
    let mut rhs = eq0;
    rhs.mul_assign(&f[0]);
    let mut t = last_prev_point;
    t.mul_assign(&f[1]);
    rhs.add_assign(&t);
    rhs.mul_assign(&final_eq_prefactor);
    if rhs != final_claim {
        return Err(GKRVerificationError::FinalStepCheckFailed { layer: layer_idx });
    }
    Ok(())
}

pub fn verify_gkr_sumcheck<
    'cfg,
    F: PrimeField,
    E: FieldExtension<F> + Field,
    I: NonDeterminismSource,
    const ROUNDS: usize,
    const ADDRS: usize,
    const EVALS: usize,
    const TRANSCRIPT_U32: usize,
    const MAX_POW: usize,
>(
    config: &'cfg GKRVerifierConfig,
) -> Result<GKRVerifierOutput<'cfg, E, ROUNDS, ADDRS>, GKRVerificationError>
where
    [(); E::DEGREE]: Sized,
    [(); ROUNDS + 1]: Sized,
{
    let mut transcript_buf = LazyVec::<u32, TRANSCRIPT_U32>::new();
    let transcript_len = config.initial_transcript_num_u32_words;
    assert!(transcript_len <= TRANSCRIPT_U32);
    for _ in 0..transcript_len {
        transcript_buf.push(I::read_word());
    }
    let mut seed = Blake2sTranscript::commit_initial(transcript_buf.as_slice());

    // init_challenges[0] is lookup_alpha for lookups preprocessing - not needed by the verifier
    let mut init_challenges = [E::ZERO; 3];
    draw_field_els_into::<F, E>(&mut seed, &mut init_challenges);
    let lookup_additive_challenge = init_challenges[1];
    let constraints_batch_challenge = init_challenges[2];

    let total_layers = config.layers.len();
    let num_standard = config.num_standard_layers;
    let num_dim_reducing = total_layers - num_standard;

    let top_layer_meta = &config.layers[total_layers - 1];

    let mut total_output_polys = 0usize;
    for group in top_layer_meta.output_groups.iter() {
        total_output_polys += group.num_addresses;
    }

    let evals_per_poly = 1usize << config.final_trace_size_log_2;
    let total_evals = total_output_polys * evals_per_poly;

    assert!(total_evals <= EVALS);

    let mut evals_flat = [MaybeUninit::<E>::uninit(); EVALS];

    read_field_els::<F, E, I>(unsafe {
        core::slice::from_raw_parts_mut(evals_flat.as_mut_ptr().cast(), total_evals)
    });

    let evals_slice =
        unsafe { core::slice::from_raw_parts(evals_flat.as_ptr().cast(), total_evals) };
    commit_field_els::<F, E>(&mut seed, evals_slice);

    let num_challenges = config.final_trace_size_log_2 + 1;
    let mut all_challenges = [E::ZERO; ROUNDS + 1];
    draw_field_els_into::<F, E>(&mut seed, &mut all_challenges[..num_challenges]);
    let mut batching_challenge = all_challenges[num_challenges - 1];
    let evaluation_point_len = config.final_trace_size_log_2;

    let mut eq_buf = [E::ZERO; 64]; // max 2^6 = 64 entries
    assert!(evals_per_poly <= 64);
    make_eq_poly_last(&all_challenges[..evaluation_point_len], &mut eq_buf);

    let initial_layer_for_sumcheck = if num_dim_reducing > 0 {
        num_standard + num_dim_reducing - 1
    } else {
        num_standard - 1
    };
    let top_layer_idx = initial_layer_for_sumcheck + 1;

    // Build initial claims. For the topmost dim-reducing layer, output addresses are
    // InnerLayer{layer: top_layer_idx, offset: ...} — sequential, so output index = iteration index.
    let mut current_claims: LazyVec<E, ADDRS> = LazyVec::new();
    let mut eval_offset = 0usize;
    for group in top_layer_meta.output_groups.iter() {
        match group.output_type {
            OutputType::PermutationProduct => {
                for _ in 0..group.num_addresses {
                    let claim = dot_eq(
                        &evals_slice[eval_offset..eval_offset + evals_per_poly],
                        &eq_buf[..evals_per_poly],
                    );
                    current_claims.push(claim);
                    eval_offset += evals_per_poly;
                }
            }
            OutputType::Lookup16Bits | OutputType::LookupTimestamps | OutputType::GenericLookup => {
                for _ in 0..2 {
                    let claim = dot_eq(
                        &evals_slice[eval_offset..eval_offset + evals_per_poly],
                        &eq_buf[..evals_per_poly],
                    );
                    current_claims.push(claim);
                    eval_offset += evals_per_poly;
                }
            }
        }
    }

    let mut prev_point = [E::ZERO; ROUNDS];
    prev_point[..evaluation_point_len].copy_from_slice(&all_challenges[..evaluation_point_len]);
    let mut prev_point_len = evaluation_point_len;
    let mut prev_claims = current_claims;

    for config_idx in (num_standard..total_layers).rev() {
        let layer_meta = &config.layers[config_idx];
        assert!(layer_meta.is_dim_reducing != 0);

        let circuit_layer_idx = config_idx;

        let initial_claim = compute_dim_reducing_layer_claim::<F, E, ADDRS>(
            layer_meta.output_groups,
            &prev_claims,
            batching_challenge,
        );

        let num_regular_rounds = layer_meta.num_sumcheck_rounds - 1;
        let (final_claim, final_eq_prefactor, mut folding_challenges, mut fc_len) =
            verify_sumcheck_rounds::<F, E, I, ROUNDS>(
                &mut seed,
                initial_claim,
                &prev_point[..num_regular_rounds],
                circuit_layer_idx,
            )?;

        let mut final_step_evals: LazyVec<[E; 4], ADDRS> = LazyVec::new();

        let num_input_addrs = layer_meta.sorted_dedup_input_addrs.len();

        for _ in 0..num_input_addrs {
            let mut vals = [E::ZERO; 4];
            read_field_els::<F, E, I>(&mut vals);
            final_step_evals.push(vals);
        }

        // final step consistency check
        {
            let f = compute_dim_reducing_final_step_accumulator::<F, E, ADDRS>(
                layer_meta.output_groups,
                layer_meta.input_sorted_indices,
                &final_step_evals,
                batching_challenge,
            );
            debug_assert!(1 <= prev_point_len);
            debug_assert!(prev_point_len <= prev_point.len());
            verify_final_step_check::<F, E>(
                f,
                unsafe { *prev_point.get_unchecked(prev_point_len - 1) },
                final_eq_prefactor,
                final_claim,
                circuit_layer_idx,
            )?;
        }

        {
            // all dim-reducing rows have exactly 4 evals, stored contiguously
            let flat = final_step_evals.as_slice();
            let flat_e =
                unsafe { core::slice::from_raw_parts(flat.as_ptr() as *const E, flat.len() * 4) };
            commit_field_els::<F, E>(&mut seed, flat_e);
        }

        let mut draw_buf = [E::ZERO; 3];
        draw_field_els_into::<F, E>(&mut seed, &mut draw_buf);
        let r_before_last = draw_buf[0];
        let r_last = draw_buf[1];
        let next_batching = draw_buf[2];

        debug_assert!(fc_len + 1 < folding_challenges.len());
        unsafe {
            *folding_challenges.get_unchecked_mut(fc_len) = r_before_last;
            fc_len += 1;
            *folding_challenges.get_unchecked_mut(fc_len) = r_last;
            fc_len += 1;
        }

        let mut eq4 = [E::ZERO; 4];
        make_eq_poly_last(&[r_before_last, r_last], &mut eq4);

        prev_claims.clear();
        for i in 0..num_input_addrs {
            let evals = final_step_evals.get(i);
            let claim = dot_eq(evals, &eq4);
            prev_claims.push(claim);
        }

        batching_challenge = next_batching;
        prev_point = folding_challenges;
        prev_point_len = fc_len;
    }

    for config_idx in (0..num_standard).rev() {
        let layer_meta = &config.layers[config_idx];
        assert!(layer_meta.is_dim_reducing == 0);
        let layer_desc = layer_meta
            .layer_desc
            .expect("standard layer must have layer_desc");

        let initial_claim = compute_standard_layer_claim::<F, E, ADDRS>(
            layer_desc,
            &prev_claims,
            batching_challenge,
        );

        let num_regular_rounds = layer_meta.num_sumcheck_rounds - 1;
        let (final_claim, final_eq_prefactor, mut folding_challenges, mut fc_len) =
            verify_sumcheck_rounds::<F, E, I, ROUNDS>(
                &mut seed,
                initial_claim,
                &prev_point[..num_regular_rounds],
                config_idx,
            )?;

        let num_dedup_addrs = layer_meta.sorted_dedup_input_addrs.len();

        let mut final_step_evals: LazyVec<[E; 2], ADDRS> = LazyVec::new();
        for _ in 0..num_dedup_addrs {
            let mut vals = [E::ZERO; 2];
            read_field_els::<F, E, I>(&mut vals);
            final_step_evals.push(vals);
        }

        {
            let f = compute_standard_final_step_accumulator::<F, E, ADDRS, MAX_POW>(
                layer_desc,
                &final_step_evals,
                batching_challenge,
                lookup_additive_challenge,
                constraints_batch_challenge,
            );

            debug_assert!(1 <= prev_point_len);
            debug_assert!(prev_point_len <= prev_point.len());
            verify_final_step_check::<F, E>(
                f,
                unsafe { *prev_point.get_unchecked(prev_point_len - 1) },
                final_eq_prefactor,
                final_claim,
                config_idx,
            )?;
        }

        {
            // Standard layers: 2 evals per addr, stored contiguously
            let flat = final_step_evals.as_slice();
            let flat_e =
                unsafe { core::slice::from_raw_parts(flat.as_ptr() as *const E, flat.len() * 2) };
            commit_field_els::<F, E>(&mut seed, flat_e);
        }

        let mut draw_buf = [E::ZERO; 2];
        draw_field_els_into::<F, E>(&mut seed, &mut draw_buf);
        let last_r = draw_buf[0];
        let next_batching = draw_buf[1];

        debug_assert!(fc_len < folding_challenges.len());
        unsafe { *folding_challenges.get_unchecked_mut(fc_len) = last_r };
        fc_len += 1;

        prev_claims.clear();
        for i in 0..num_dedup_addrs {
            let evals = final_step_evals.get(i);
            unsafe {
                let f0 = *evals.get_unchecked(0);
                let f1 = *evals.get_unchecked(1);
                let mut diff = f1;
                diff.sub_assign(&f0);
                diff.mul_assign(&last_r);
                diff.add_assign(&f0);
                prev_claims.push(diff);
            }
        }

        batching_challenge = next_batching;
        prev_point = folding_challenges;
        prev_point_len = fc_len;
    }

    let grand_product_accumulator: E = read_field_el::<F, E, I>();
    commit_field_els::<F, E>(&mut seed, &[grand_product_accumulator]);

    let whir_batching_challenge = draw_one_field_el::<F, E>(&mut seed);

    let additional_base_layer_openings = config
        .layers
        .first()
        .and_then(|l| l.layer_desc)
        .map(|desc| desc.additional_base_layer_openings)
        .unwrap_or(&[]);

    let base_layer_addrs = config
        .layers
        .first()
        .map(|l| l.sorted_dedup_input_addrs)
        .unwrap_or(&[]);

    Ok(GKRVerifierOutput {
        base_layer_claims: prev_claims,
        base_layer_addrs,
        evaluation_point: prev_point,
        evaluation_point_len: prev_point_len,
        grand_product_accumulator,
        additional_base_layer_openings,
        whir_batching_challenge,
        whir_transcript_seed: seed,
    })
}
