use blake2s_u32::BLAKE2S_DIGEST_SIZE_U32_WORDS;
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
pub struct SmallMap<K, V, const N: usize> {
    data: [(K, V); N],
    len: usize,
}

impl<K: PartialEq + Copy + Ord, V: Copy, const N: usize> SmallMap<K, V, N> {
    fn new(default_key: K, default_val: V) -> Self {
        Self {
            data: [(default_key, default_val); N],
            len: 0,
        }
    }

    fn insert(&mut self, key: K, val: V) {
        assert!(self.len < N);
        debug_assert!(self.len == 0 || self.data[self.len - 1].0 < key);

        unsafe { *self.data.get_unchecked_mut(self.len) = (key, val) };
        self.len += 1;
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn get(&self, key: &K) -> Option<&V> {
        match self.data[..self.len].binary_search_by_key(&key, |(k, _)| k) {
            Ok(idx) => Some(unsafe { &self.data.get_unchecked(idx).1 }),
            Err(_) => None,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
struct SmallVecMap<K, V, const N: usize, const M: usize> {
    keys: [K; N],
    vals: [[V; M]; N],
    val_lens: [usize; N],
    len: usize,
}

impl<K: PartialEq + Copy + Ord, V: Copy, const N: usize, const M: usize> SmallVecMap<K, V, N, M> {
    fn new(default_key: K, default_val: V) -> Self {
        Self {
            keys: [default_key; N],
            vals: [[default_val; M]; N],
            val_lens: [0; N],
            len: 0,
        }
    }

    fn insert_vec(&mut self, key: K, vals: &[V]) {
        assert!(vals.len() <= M);
        assert!(self.len < N);
        debug_assert!(self.len == 0 || self.keys[self.len - 1] < key);

        unsafe {
            *self.keys.get_unchecked_mut(self.len) = key;
            self.vals.get_unchecked_mut(self.len)[..vals.len()].copy_from_slice(vals);
            *self.val_lens.get_unchecked_mut(self.len) = vals.len();
        }
        self.len += 1;
    }

    fn get(&self, key: &K) -> Option<&[V]> {
        match self.keys[..self.len].binary_search(key) {
            Ok(idx) => unsafe {
                let len = *self.val_lens.get_unchecked(idx);
                Some(&self.vals.get_unchecked(idx)[..len])
            },
            Err(_) => None,
        }
    }

    /// Reinterpret the first `self.len` rows of `vals` as a flat `&[V]` of length `self.len * M`.
    /// Caller must ensure every inserted row uses exactly M elements.
    unsafe fn vals_as_flat_slice(&self) -> &[V] {
        debug_assert!(self.val_lens[..self.len].iter().all(|&len| len == M));
        core::slice::from_raw_parts(self.vals.as_ptr() as *const V, self.len * M)
    }
}

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

fn read_field_els<F: PrimeField, E: FieldExtension<F>, I: NonDeterminismSource>(dst: &mut [E])
where
    [(); E::DEGREE]: Sized,
{
    for el in dst.iter_mut() {
        *el = read_field_el::<F, E, I>();
    }
}

fn commit_field_els<F: PrimeField, E: FieldExtension<F>>(seed: &mut Seed, els: &[E])
where
    [(); E::DEGREE]: Sized,
{
    use core::mem::{size_of, align_of};
    assert!(size_of::<F>() == size_of::<u32>());
    assert!(align_of::<F>() == align_of::<u32>());

    let total = els.len() * E::DEGREE;
    let as_u32 = unsafe { core::slice::from_raw_parts(els.as_ptr() as *const u32, total) };
    Blake2sTranscript::commit_with_seed(seed, as_u32);
}

fn draw_field_els_into<F: PrimeField, E: FieldExtension<F>>(seed: &mut Seed, dst: &mut [E])
where
    [(); E::DEGREE]: Sized,
{
    use field::FixedArrayConvertible;
    let n = dst.len();
    let padded = (n * E::DEGREE).next_multiple_of(BLAKE2S_DIGEST_SIZE_U32_WORDS);
    debug_assert!(padded <= DRAW_BUF_CAPACITY, "draw buffer too small");
    let mut words = [0u32; DRAW_BUF_CAPACITY];
    Blake2sTranscript::draw_randomness(seed, &mut words[..padded]);
    for (i, chunk) in words[..n * E::DEGREE].chunks_exact(E::DEGREE).enumerate() {
        let mut arr = [F::ZERO; E::DEGREE];
        for (j, &w) in chunk.iter().enumerate() {
            arr[j] = F::from_u32_with_reduction(w);
        }
        let fixed = E::Coeffs::from_array(arr);
        dst[i] = E::from_coeffs(fixed);
    }
}

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

fn eval_cubic_poly<F: PrimeField, E: FieldExtension<F> + Field>(coeffs: &[E; 4], r: E) -> E {
    let mut result = coeffs[3];
    result.mul_assign(&r);
    result.add_assign(&coeffs[2]);
    result.mul_assign(&r);
    result.add_assign(&coeffs[1]);
    result.mul_assign(&r);
    result.add_assign(&coeffs[0]);
    result
}

fn eval_eq<E: Field>(x: E, y: E) -> E {
    let mut one_minus_x = E::ONE;
    one_minus_x.sub_assign(&x);
    let mut one_minus_y = E::ONE;
    one_minus_y.sub_assign(&y);
    let mut t = one_minus_x;
    t.mul_assign(&one_minus_y);
    let mut xy = x;
    xy.mul_assign(&y);
    t.add_assign(&xy);
    t
}

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

fn get_eval_at_j<E: Field, const N: usize, const M: usize>(
    evals: &SmallVecMap<GKRAddress, E, N, M>,
    addr: GKRAddress,
    j: usize,
) -> E {
    evals
        .get(&addr)
        .and_then(|v| v.get(j).copied())
        .expect("address should exist")
}

fn eval_constraint_kernel<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const N: usize,
    const M: usize,
    const MAX_POW: usize,
>(
    rel: &StaticNoFieldMaxQuadraticConstraintsGKRRelation,
    constraints_batch_challenge: E,
    evals: &SmallVecMap<GKRAddress, E, N, M>,
    j: usize,
) -> E {
    let get_value_at = |addr: GKRAddress| -> E {
        debug_assert!(addr != GKRAddress::placeholder());
        get_eval_at_j(evals, addr, j)
    };

    let mut result = E::ZERO;
    let mut challenge_powers = [E::ZERO; MAX_POW];
    challenge_powers[0] = E::ONE;
    challenge_powers.iter_mut().reduce(|prev, cur| {
        *cur = *prev;
        cur.mul_assign(&constraints_batch_challenge);
        cur
    });

    // pow < MAX_POW guaranteed by generator
    let get_pow = |pow: usize| unsafe { *challenge_powers.get_unchecked(pow) };

    for &(coeff, pow) in rel.constants.iter() {
        let mut t = get_pow(pow);
        let c = F::from_u32_with_reduction(coeff);
        t.mul_assign_by_base(&c);
        result.add_assign(&t);
    }

    for (addr, terms) in rel.linear_terms.iter() {
        let val = get_value_at(*addr);
        for &(coeff, pow) in terms.iter() {
            let mut t = get_pow(pow);
            let c = F::from_u32_with_reduction(coeff);
            t.mul_assign_by_base(&c);
            t.mul_assign(&val);
            result.add_assign(&t);
        }
    }

    for ((addr_a, addr_b), terms) in rel.quadratic_terms.iter() {
        let va = get_value_at(*addr_a);
        let vb = get_value_at(*addr_b);
        let mut prod = va;
        prod.mul_assign(&vb);
        for &(coeff, pow) in terms.iter() {
            let mut t = get_pow(pow);
            let c = F::from_u32_with_reduction(coeff);
            t.mul_assign_by_base(&c);
            t.mul_assign(&prod);
            result.add_assign(&t);
        }
    }

    result
}

fn compute_standard_final_step_accumulator<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const N: usize,
    const M: usize,
    const MAX_POW: usize,
>(
    layer: &StaticGKRLayerDescription,
    final_step_evaluations: &SmallVecMap<GKRAddress, E, N, M>,
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
                    let val = eval_constraint_kernel::<F, E, N, M, MAX_POW>(
                        input,
                        constraints_batch_challenge,
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
                    let val = eval_single_output_relation::<F, E, N, M>(
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
                    let [out0, out1] = eval_two_output_relation::<F, E, N, M>(
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

fn eval_single_output_relation<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const N: usize,
    const M: usize,
>(
    relation: &StaticNoFieldGKRRelation,
    evals: &SmallVecMap<GKRAddress, E, N, M>,
    _lookup_additive_challenge: E,
    j: usize,
) -> E {
    use StaticNoFieldGKRRelation as Rel;
    let get = |addr: GKRAddress| get_eval_at_j(evals, addr, j);

    match relation {
        Rel::Copy { input, .. } => get(*input),
        Rel::InitialGrandProductFromCaches { input, .. } | Rel::TrivialProduct { input, .. } => {
            let mut v = get(input[0]);
            v.mul_assign(&get(input[1]));
            v
        }
        Rel::MaskIntoIdentityProduct { input, mask, .. } => {
            let input_val = get(*input);
            let mask_val = get(*mask);
            let mut out = input_val;
            out.sub_assign_base(&F::ONE);
            out.mul_assign(&mask_val);
            out.add_assign_base(&F::ONE);
            out
        }
        _ => unreachable!("eval_single_output_relation called with non-single-output relation"),
    }
}

fn eval_two_output_relation<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const N: usize,
    const M: usize,
>(
    relation: &StaticNoFieldGKRRelation,
    evals: &SmallVecMap<GKRAddress, E, N, M>,
    lookup_additive_challenge: E,
    j: usize,
) -> [E; 2] {
    use StaticNoFieldGKRRelation as Rel;
    let get = |addr: GKRAddress| get_eval_at_j(evals, addr, j);
    let gamma = lookup_additive_challenge;

    match relation {
        Rel::LookupPair { input, .. } => {
            let a = get(input[0][0]);
            let b = get(input[0][1]);
            let c = get(input[1][0]);
            let d = get(input[1][1]);
            let mut num = a;
            num.mul_assign(&d);
            let mut cb = c;
            cb.mul_assign(&b);
            num.add_assign(&cb);
            let mut den = b;
            den.mul_assign(&d);
            [num, den]
        }
        Rel::LookupPairFromMaterializedBaseInputs { input, .. } => {
            let mut b_g = get(input[0]);
            b_g.add_assign(&gamma);
            let mut d_g = get(input[1]);
            d_g.add_assign(&gamma);
            let mut num = b_g;
            num.add_assign(&d_g);
            let mut den = b_g;
            den.mul_assign(&d_g);
            [num, den]
        }
        Rel::LookupFromMaterializedBaseInputWithSetup { input, setup, .. } => {
            let mut b_g = get(*input);
            b_g.add_assign(&gamma);
            let mut d_g = get(setup[1]);
            d_g.add_assign(&gamma);
            let c = get(setup[0]);
            let mut cb_g = c;
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
            let mut d_g = get(*remainder);
            d_g.add_assign(&gamma);
            let mut num = a;
            num.mul_assign(&d_g);
            num.add_assign(&b);
            let mut den = b;
            den.mul_assign(&d_g);
            [num, den]
        }
        Rel::LookupWithCachedDensAndSetup { input, setup, .. } => {
            let a = get(input[0]);
            let b = get(input[1]);
            let c = get(setup[0]);
            let d = get(setup[1]);
            let mut ad = a;
            ad.mul_assign(&d);
            let mut cb = c;
            cb.mul_assign(&b);
            ad.sub_assign(&cb);
            let mut den = b;
            den.mul_assign(&d);
            [ad, den]
        }
        _ => unreachable!("eval_two_output_relation called with non-two-output relation"),
    }
}

fn compute_dim_reducing_final_step_accumulator<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const ADDRS: usize,
>(
    output_groups: &[GKROutputGroup],
    input_addrs: &[GKRAddress],
    final_step_evaluations: &SmallVecMap<GKRAddress, E, ADDRS, 4>,
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

    let mut addr_idx = 0;
    for group in output_groups.iter() {
        match group.output_type {
            OutputType::PermutationProduct => {
                for _ in 0..group.num_addresses {
                    debug_assert!(addr_idx < input_addrs.len());
                    let addr = unsafe { *input_addrs.get_unchecked(addr_idx) };
                    addr_idx += 1;
                    let bc = get_challenge();
                    if let Some(evals) = final_step_evaluations.get(&addr) {
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
            }
            OutputType::Lookup16Bits | OutputType::LookupTimestamps | OutputType::GenericLookup => {
                let bc0 = get_challenge();
                let bc1 = get_challenge();
                debug_assert!(addr_idx + 1 < input_addrs.len());
                let v0 = final_step_evaluations.get(unsafe { input_addrs.get_unchecked(addr_idx) });
                let v1 = final_step_evaluations.get(unsafe { input_addrs.get_unchecked(addr_idx + 1) });
                addr_idx += 2;
                if let (Some(v0), Some(v1)) = (v0, v1) {
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
    }

    acc
}

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

fn compute_standard_layer_claim<F: PrimeField, E: FieldExtension<F> + Field, const ADDRS: usize>(
    layer: &StaticGKRLayerDescription,
    output_claims: &SmallMap<GKRAddress, E, ADDRS>,
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

fn add_relation_claim<F: PrimeField, E: FieldExtension<F> + Field, const ADDRS: usize>(
    relation: &StaticNoFieldGKRRelation,
    output_claims: &SmallMap<GKRAddress, E, ADDRS>,
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
            if let Some(claim) = output_claims.get(output) {
                let mut t = bc;
                t.mul_assign(claim);
                combined.add_assign(&t);
            }
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
            for (bc, addr) in [(bc0, &output[0]), (bc1, &output[1])] {
                if let Some(claim) = output_claims.get(addr) {
                    let mut t = bc;
                    t.mul_assign(claim);
                    combined.add_assign(&t);
                }
            }
        }
    }
}

fn compute_dim_reducing_layer_claim<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const ADDRS: usize,
>(
    output_groups: &[GKROutputGroup],
    output_addrs: &[GKRAddress],
    output_claims: &SmallMap<GKRAddress, E, ADDRS>,
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

    let mut addr_idx = 0;
    for group in output_groups.iter() {
        match group.output_type {
            OutputType::PermutationProduct => {
                for _ in 0..group.num_addresses {
                    let bc = get_challenge();
                    debug_assert!(addr_idx < output_addrs.len());
                    let addr = unsafe { *output_addrs.get_unchecked(addr_idx) };
                    addr_idx += 1;
                    if let Some(claim) = output_claims.get(&addr) {
                        let mut t = bc;
                        t.mul_assign(claim);
                        combined.add_assign(&t);
                    }
                }
            }
            OutputType::Lookup16Bits | OutputType::LookupTimestamps | OutputType::GenericLookup => {
                let bc0 = get_challenge();
                let bc1 = get_challenge();
                debug_assert!(addr_idx + 1 < output_addrs.len());
                for (bc, i) in [(bc0, 0), (bc1, 1)] {
                    let addr = unsafe { *output_addrs.get_unchecked(addr_idx + i) };
                    if let Some(claim) = output_claims.get(&addr) {
                        let mut t = bc;
                        t.mul_assign(claim);
                        combined.add_assign(&t);
                    }
                }
                addr_idx += 2;
            }
        }
    }
    combined
}

pub struct GKRVerifierOutput<'a, E: Field, const ROUNDS: usize, const ADDRS: usize> {
    pub base_layer_claims: SmallMap<GKRAddress, E, ADDRS>,
    pub evaluation_point: [E; ROUNDS],
    pub evaluation_point_len: usize,
    pub grand_product_accumulator: E,
    pub additional_base_layer_openings: &'a [GKRAddress],
    pub whir_batching_challenge: E,
    pub whir_transcript_seed: Seed,
}

/// Run the regular sumcheck rounds (all rounds except the final step).
/// Returns (final_claim, final_eq_prefactor, folding_challenges, fc_len).
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
    let mut folding_challenges = [E::ZERO; ROUNDS];
    let mut fc_len = 0;

    for round in 0..num_regular_rounds {
        let mut coeffs = [E::ZERO; 4];
        read_field_els::<F, E, I>(&mut coeffs);

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

        commit_field_els::<F, E>(seed, &coeffs);
        let r_k = draw_one_field_el::<F, E>(seed);

        claim = eval_cubic_poly(&coeffs, r_k);
        debug_assert!(round < prev_point.len());
        eq_prefactor = eval_eq(r_k, unsafe { *prev_point.get_unchecked(round) });

        debug_assert!(fc_len < folding_challenges.len());
        unsafe { *folding_challenges.get_unchecked_mut(fc_len) = r_k };
        fc_len += 1;
    }

    Ok((claim, eq_prefactor, folding_challenges, fc_len))
}

/// Verify the final step consistency: ((1 - last_prev) * f[0] + last_prev * f[1]) * eq_prefactor == claim.
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
    const RAW_ADDRS: usize,
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
    let mut transcript_buf = [0u32; TRANSCRIPT_U32];
    let transcript_len = config.initial_transcript_num_u32_words;
    assert!(transcript_len <= TRANSCRIPT_U32);
    for word in transcript_buf[..transcript_len].iter_mut() {
        *word = I::read_word();
    }
    let mut seed = Blake2sTranscript::commit_initial(&transcript_buf[..transcript_len]);

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

    let mut evals_flat = [E::ZERO; EVALS];
    assert!(total_evals <= EVALS);
    read_field_els::<F, E, I>(&mut evals_flat[..total_evals]);
    commit_field_els::<F, E>(&mut seed, &evals_flat[..total_evals]);

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

    let mut current_claims: SmallMap<GKRAddress, E, ADDRS> =
        SmallMap::new(GKRAddress::placeholder(), E::ZERO);
    let mut eval_offset = 0usize;
    let mut addr_offset = 0usize;
    for group in top_layer_meta.output_groups.iter() {
        match group.output_type {
            OutputType::PermutationProduct => {
                for i in 0..group.num_addresses {
                    let addr = GKRAddress::InnerLayer {
                        layer: top_layer_idx,
                        offset: addr_offset + i,
                    };
                    let claim = dot_eq(
                        &evals_flat[eval_offset..eval_offset + evals_per_poly],
                        &eq_buf[..evals_per_poly],
                    );
                    current_claims.insert(addr, claim);
                    eval_offset += evals_per_poly;
                }
                addr_offset += group.num_addresses;
            }
            OutputType::Lookup16Bits | OutputType::LookupTimestamps | OutputType::GenericLookup => {
                for i in 0..2 {
                    let addr = GKRAddress::InnerLayer {
                        layer: top_layer_idx,
                        offset: addr_offset + i,
                    };
                    let claim = dot_eq(
                        &evals_flat[eval_offset..eval_offset + evals_per_poly],
                        &eq_buf[..evals_per_poly],
                    );
                    current_claims.insert(addr, claim);
                    eval_offset += evals_per_poly;
                }
                addr_offset += 2;
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
        let output_layer_idx = circuit_layer_idx + 1;

        let mut output_addrs = [GKRAddress::placeholder(); ADDRS];
        let mut output_addr_count = 0;
        {
            let mut off = 0;
            for group in layer_meta.output_groups.iter() {
                match group.output_type {
                    OutputType::PermutationProduct => {
                        for i in 0..group.num_addresses {
                            debug_assert!(output_addr_count < output_addrs.len());
                            unsafe {
                                *output_addrs.get_unchecked_mut(output_addr_count) =
                                    GKRAddress::InnerLayer { layer: output_layer_idx, offset: off + i };
                            }
                            output_addr_count += 1;
                        }
                        off += group.num_addresses;
                    }
                    OutputType::Lookup16Bits
                    | OutputType::LookupTimestamps
                    | OutputType::GenericLookup => {
                        debug_assert!(output_addr_count + 1 < output_addrs.len());
                        unsafe {
                            *output_addrs.get_unchecked_mut(output_addr_count) =
                                GKRAddress::InnerLayer { layer: output_layer_idx, offset: off };
                            *output_addrs.get_unchecked_mut(output_addr_count + 1) =
                                GKRAddress::InnerLayer { layer: output_layer_idx, offset: off + 1 };
                        }
                        output_addr_count += 2;
                        off += 2;
                    }
                }
            }
        }

        let mut input_addrs = [GKRAddress::placeholder(); ADDRS];
        let mut input_addr_count = 0;
        if config_idx == num_standard {
            let n = config.global_input_addrs.len();
            input_addrs[..n].copy_from_slice(config.global_input_addrs);
            input_addr_count = n;
        } else {
            let below_meta = &config.layers[config_idx - 1];
            let mut off = 0;
            for group in below_meta.output_groups.iter() {
                match group.output_type {
                    OutputType::PermutationProduct => {
                        for i in 0..group.num_addresses {
                            debug_assert!(input_addr_count < input_addrs.len());
                            unsafe {
                                *input_addrs.get_unchecked_mut(input_addr_count) =
                                    GKRAddress::InnerLayer { layer: circuit_layer_idx, offset: off + i };
                            }
                            input_addr_count += 1;
                        }
                        off += group.num_addresses;
                    }
                    OutputType::Lookup16Bits
                    | OutputType::LookupTimestamps
                    | OutputType::GenericLookup => {
                        debug_assert!(input_addr_count + 1 < input_addrs.len());
                        unsafe {
                            *input_addrs.get_unchecked_mut(input_addr_count) =
                                GKRAddress::InnerLayer { layer: circuit_layer_idx, offset: off };
                            *input_addrs.get_unchecked_mut(input_addr_count + 1) =
                                GKRAddress::InnerLayer { layer: circuit_layer_idx, offset: off + 1 };
                        }
                        input_addr_count += 2;
                        off += 2;
                    }
                }
            }
        }

        let initial_claim = compute_dim_reducing_layer_claim::<F, E, ADDRS>(
            layer_meta.output_groups,
            &output_addrs[..output_addr_count],
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

        let mut final_step_evals: SmallVecMap<GKRAddress, E, ADDRS, 4> =
            SmallVecMap::new(GKRAddress::placeholder(), E::ZERO);

        let mut sorted_input_addrs = [GKRAddress::placeholder(); ADDRS];
        sorted_input_addrs[..input_addr_count].copy_from_slice(&input_addrs[..input_addr_count]);
        sorted_input_addrs[..input_addr_count].sort();

        for &addr in sorted_input_addrs[..input_addr_count].iter() {
            let mut vals = [E::ZERO; 4];
            read_field_els::<F, E, I>(&mut vals);
            final_step_evals.insert_vec(addr, &vals);
        }

        // final step consistency check
        {
            let f = compute_dim_reducing_final_step_accumulator::<F, E, ADDRS>(
                layer_meta.output_groups,
                &input_addrs[..input_addr_count],
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
            // all dim-reducing rows have exactly 4 evals, so vals[..len] is contiguous.
            // keys were inserted in sorted order, matching the commit order.
            let flat = unsafe { final_step_evals.vals_as_flat_slice() };
            commit_field_els::<F, E>(&mut seed, flat);
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
        for &addr in sorted_input_addrs[..input_addr_count].iter() {
            if let Some(evals) = final_step_evals.get(&addr) {
                let claim = dot_eq(evals, &eq4);
                prev_claims.insert(addr, claim);
            }
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

        let mut final_step_addrs = [GKRAddress::placeholder(); RAW_ADDRS];
        let mut fsa_len = 0;
        collect_standard_layer_input_addrs(layer_desc, &mut final_step_addrs, &mut fsa_len);
        final_step_addrs[..fsa_len].sort();

        let mut dedup_addrs = [GKRAddress::placeholder(); ADDRS];
        let mut dedup_len = 0;
        for &addr in final_step_addrs[..fsa_len].iter() {
            debug_assert!(dedup_len < dedup_addrs.len());
            unsafe {
                if dedup_len == 0 || *dedup_addrs.get_unchecked(dedup_len - 1) != addr {
                    *dedup_addrs.get_unchecked_mut(dedup_len) = addr;
                    dedup_len += 1;
                }
            }
        }

        let mut final_step_evals: SmallVecMap<GKRAddress, E, ADDRS, 2> =
            SmallVecMap::new(GKRAddress::placeholder(), E::ZERO);
        for &addr in dedup_addrs[..dedup_len].iter() {
            let mut vals = [E::ZERO; 2];
            read_field_els::<F, E, I>(&mut vals);
            final_step_evals.insert_vec(addr, &vals);
        }

        {
            let f = compute_standard_final_step_accumulator::<F, E, ADDRS, 2, MAX_POW>(
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
            // Standard layers: 2 evals per addr, all rows fully used.
            // Keys inserted in sorted order, matching commit order.
            let flat = unsafe { final_step_evals.vals_as_flat_slice() };
            commit_field_els::<F, E>(&mut seed, flat);
        }

        let mut draw_buf = [E::ZERO; 2];
        draw_field_els_into::<F, E>(&mut seed, &mut draw_buf);
        let last_r = draw_buf[0];
        let next_batching = draw_buf[1];

        debug_assert!(fc_len < folding_challenges.len());
        unsafe { *folding_challenges.get_unchecked_mut(fc_len) = last_r };
        fc_len += 1;

        prev_claims.clear();
        for &addr in dedup_addrs[..dedup_len].iter() {
            if let Some(evals) = final_step_evals.get(&addr) {
                unsafe {
                    let f0 = *evals.get_unchecked(0);
                    let f1 = *evals.get_unchecked(1);
                    let mut diff = f1;
                    diff.sub_assign(&f0);
                    diff.mul_assign(&last_r);
                    diff.add_assign(&f0);
                    prev_claims.insert(addr, diff);
                }
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

    Ok(GKRVerifierOutput {
        base_layer_claims: prev_claims,
        evaluation_point: prev_point,
        evaluation_point_len: prev_point_len,
        grand_product_accumulator,
        additional_base_layer_openings,
        whir_batching_challenge,
        whir_transcript_seed: seed,
    })
}

fn collect_standard_layer_input_addrs(
    layer: &StaticGKRLayerDescription,
    dst: &mut [GKRAddress],
    len: &mut usize,
) {
    use StaticNoFieldGKRRelation as Rel;

    let relations = layer
        .gates
        .iter()
        .chain(layer.gates_with_external_connections.iter())
        .map(|g| &g.enforced_relation);

    for relation in relations {
        match relation {
            Rel::EnforceConstraintsMaxQuadratic { input } => {
                for (addr, _) in input.linear_terms.iter() {
                    unsafe { *dst.get_unchecked_mut(*len) = *addr };
                    *len += 1;
                }
                for ((addr_a, addr_b), _) in input.quadratic_terms.iter() {
                    unsafe { *dst.get_unchecked_mut(*len) = *addr_a };
                    *len += 1;
                    unsafe { *dst.get_unchecked_mut(*len) = *addr_b };
                    *len += 1;
                }
            }
            Rel::Copy { input, .. } => {
                unsafe { *dst.get_unchecked_mut(*len) = *input };
                *len += 1;
            }
            Rel::InitialGrandProductFromCaches { input, .. }
            | Rel::TrivialProduct { input, .. } => {
                unsafe { *dst.get_unchecked_mut(*len) = input[0] };
                *len += 1;
                unsafe { *dst.get_unchecked_mut(*len) = input[1] };
                *len += 1;
            }
            Rel::MaskIntoIdentityProduct { input, mask, .. } => {
                unsafe { *dst.get_unchecked_mut(*len) = *input };
                *len += 1;
                unsafe { *dst.get_unchecked_mut(*len) = *mask };
                *len += 1;
            }
            Rel::LookupPair { input, .. } => {
                unsafe { *dst.get_unchecked_mut(*len) = input[0][0] };
                *len += 1;
                unsafe { *dst.get_unchecked_mut(*len) = input[0][1] };
                *len += 1;
                unsafe { *dst.get_unchecked_mut(*len) = input[1][0] };
                *len += 1;
                unsafe { *dst.get_unchecked_mut(*len) = input[1][1] };
                *len += 1;
            }
            Rel::LookupPairFromMaterializedBaseInputs { input, .. } => {
                unsafe { *dst.get_unchecked_mut(*len) = input[0] };
                *len += 1;
                unsafe { *dst.get_unchecked_mut(*len) = input[1] };
                *len += 1;
            }
            Rel::LookupFromMaterializedBaseInputWithSetup { input, setup, .. } => {
                unsafe { *dst.get_unchecked_mut(*len) = *input };
                *len += 1;
                unsafe { *dst.get_unchecked_mut(*len) = setup[0] };
                *len += 1;
                unsafe { *dst.get_unchecked_mut(*len) = setup[1] };
                *len += 1;
            }
            Rel::LookupUnbalancedPairWithMaterializedBaseInputs {
                input, remainder, ..
            } => {
                unsafe { *dst.get_unchecked_mut(*len) = input[0] };
                *len += 1;
                unsafe { *dst.get_unchecked_mut(*len) = input[1] };
                *len += 1;
                unsafe { *dst.get_unchecked_mut(*len) = *remainder };
                *len += 1;
            }
            Rel::LookupWithCachedDensAndSetup { input, setup, .. } => {
                unsafe { *dst.get_unchecked_mut(*len) = input[0] };
                *len += 1;
                unsafe { *dst.get_unchecked_mut(*len) = input[1] };
                *len += 1;
                unsafe { *dst.get_unchecked_mut(*len) = setup[0] };
                *len += 1;
                unsafe { *dst.get_unchecked_mut(*len) = setup[1] };
                *len += 1;
            }
            _ => {
                panic!("unimplemented relation variant in collect_input_addrs");
            }
        }
    }
    
}
