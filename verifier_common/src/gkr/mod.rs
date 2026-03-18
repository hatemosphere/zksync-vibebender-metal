use core::mem::MaybeUninit;

use blake2s_u32::{
    AlignedArray64, DelegatedBlake2sState, BLAKE2S_BLOCK_SIZE_U32_WORDS,
    BLAKE2S_DIGEST_SIZE_U32_WORDS,
};
use cs::definitions::GKRAddress;
use field::{Field, FieldExtension, PrimeField};
use non_determinism_source::NonDeterminismSource;
use transcript::{Blake2sTranscript, Seed};

#[cfg(any(test, feature = "proof_utils"))]
pub mod flatten;

const DRAW_BUF_CAPACITY: usize = 64;

/// Mutable state threaded through each layer of the GKR sumcheck verification.
#[derive(Clone, Debug)]
pub struct LayerState<E: Field, const ROUNDS: usize, const ADDRS: usize> {
    pub prev_point: [E; ROUNDS],
    pub prev_point_len: usize,
    pub prev_claims: LazyVec<E, ADDRS>,
    pub batching_challenge: E,
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct LazyVec<V: Copy, const N: usize> {
    data: [MaybeUninit<V>; N],
    len: usize,
}

impl<V: Copy, const N: usize> LazyVec<V, N> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            data: [MaybeUninit::uninit(); N],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, val: V) {
        debug_assert!(self.len < N);
        unsafe {
            self.data.get_unchecked_mut(self.len).write(val);
        }
        self.len += 1;
    }

    #[inline(always)]
    pub fn get(&self, idx: usize) -> &V {
        debug_assert!(idx < self.len);
        unsafe { self.data.get_unchecked(idx).assume_init_ref() }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[V] {
        unsafe { core::slice::from_raw_parts(self.data.as_ptr().cast::<V>(), self.len) }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [V] {
        unsafe { core::slice::from_raw_parts_mut(self.data.as_mut_ptr().cast::<V>(), self.len)}
    }

    #[inline(always)]
    pub const fn clear(&mut self) {
        self.len = 0;
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub unsafe fn get_unchecked(&self, idx: usize) -> &V {
        self.data.get_unchecked(idx).assume_init_ref()
    }

    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, idx: usize, val: V) {
        self.data.get_unchecked_mut(idx).write(val);
    }

    #[inline(always)]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= N);
        self.len = new_len;
    }

    #[inline(always)]
    pub unsafe fn into_array(self) -> [V; N] {
        MaybeUninit::array_assume_init(self.data)
    }
}

#[inline(always)]
pub fn read_field_el<F: PrimeField, E: FieldExtension<F>, I: NonDeterminismSource>() -> E
where
    [(); E::DEGREE]: Sized,
{
    use field::FixedArrayConvertible;
    let mut words = LazyVec::<F, {E::DEGREE}>::new();
    for _ in 0..E::DEGREE {
        words.push(F::from_reduced_raw_repr(I::read_word()));
    }
    let coeffs = E::Coeffs::from_array(unsafe { words.into_array() });
    E::from_coeffs(coeffs)
}

#[inline(always)]
pub fn read_field_els<F: PrimeField, E: FieldExtension<F>, I: NonDeterminismSource>(dst: &mut [E])
where
    [(); E::DEGREE]: Sized,
{
    for el in dst.iter_mut() {
        *el = read_field_el::<F, E, I>();
    }
}

#[inline(always)]
pub fn commit_field_els<F: PrimeField, E: FieldExtension<F>>(seed: &mut Seed, els: &[E])
where
    [(); E::DEGREE]: Sized,
{
    use core::mem::{align_of, size_of};
    assert!(size_of::<F>() == size_of::<u32>());
    assert!(align_of::<F>() == align_of::<u32>());

    let total = els.len() * E::DEGREE;
    let as_u32 = unsafe { core::slice::from_raw_parts(els.as_ptr().cast::<u32>(), total) };
    Blake2sTranscript::commit_with_seed(seed, as_u32);
}

#[inline(always)]
pub fn draw_field_els_into<F: PrimeField, E: FieldExtension<F>>(
    hasher: &mut DelegatedBlake2sState,
    seed: &mut Seed,
    dst: &mut [E],
) where
    [(); E::DEGREE]: Sized,
{
    let n = dst.len();
    let padded = (n * E::DEGREE).next_multiple_of(BLAKE2S_DIGEST_SIZE_U32_WORDS);
    assert!(padded <= DRAW_BUF_CAPACITY, "draw buffer too small");

    let mut words = LazyVec::<u32, DRAW_BUF_CAPACITY>::new();
    let mut arr = LazyVec::<F, { E::DEGREE }>::new();

    unsafe {
        words.set_len(padded);
        Blake2sTranscript::draw_randomness_using_hasher(hasher, seed, words.as_mut_slice());
    }

    for (i, chunk) in words.as_slice()[..n * E::DEGREE]
        .chunks_exact(E::DEGREE)
        .enumerate()
    {
        for &w in chunk {
            arr.push(F::from_u32_with_reduction(w));
        }
        
        dst[i] = E::from_coeffs_ref(unsafe { arr.as_slice().as_ptr().cast::<E::Coeffs>().as_ref_unchecked() });
        arr.clear();
    }
}

#[inline(always)]
pub fn read_eval_data_from_nds<I: NonDeterminismSource, const BUF: usize>(
    buf: &mut AlignedArray64<MaybeUninit<u32>, BUF>,
    data_words: usize,
) {
    let total_commit_words = BLAKE2S_DIGEST_SIZE_U32_WORDS + data_words;
    for i in 0..data_words {
        buf.write(BLAKE2S_DIGEST_SIZE_U32_WORDS + i, I::read_word());
    }
    let padded = total_commit_words.next_multiple_of(BLAKE2S_BLOCK_SIZE_U32_WORDS);
    unsafe { buf.zero_range(BLAKE2S_DIGEST_SIZE_U32_WORDS + data_words, padded) };
}

#[inline(always)]
pub fn commit_eval_buffer<const BUF: usize>(
    buf: &mut AlignedArray64<MaybeUninit<u32>, BUF>,
    hasher: &mut DelegatedBlake2sState,
    seed: &mut Seed,
    data_words: usize,
) {
    let total_commit_words = BLAKE2S_DIGEST_SIZE_U32_WORDS + data_words;
    buf.copy_from_slice(0, &seed.0);
    let buf_ref = unsafe { buf.assume_init_ref() };
    Blake2sTranscript::commit_with_seed_using_hasher_and_aligned_buffer(
        hasher,
        seed,
        buf_ref,
        total_commit_words,
    );
}

#[derive(Clone, Debug)]
pub enum GKRVerificationError {
    SumcheckRoundFailed { layer: usize, round: usize },
    FinalStepCheckFailed { layer: usize },
}

#[inline(always)]
pub fn dot_eq<E: Field, const N: usize>(values: &[E; N], eq: &[E; N]) -> E {
    let mut result = E::ZERO;
    for i in 0..N {
        let mut t = unsafe { *values.get_unchecked(i) };
        t.mul_assign(unsafe { eq.get_unchecked(i) });
        result.add_assign(&t);
    }
    result
}

#[inline(always)]
pub fn make_eq_poly<E: Field + Copy, const N: usize>(
    challenges: &[E; N],
    buf: &mut LazyVec<E, {1 << N}>,
) {
    unsafe { buf.set_unchecked(0, E::ONE) };
    let mut size = 1usize;
    let mut idx = N;
    for _ in 0..N {
        idx -= 1;
        let c = unsafe { *challenges.get_unchecked(idx) };
        let f1 = c;
        let mut f0 = E::ONE;
        f0.sub_assign(&c);
        let half = size;

        for i in (0..half).rev() {
            let prev = unsafe { *buf.get_unchecked(i) };
            let mut left = prev;
            let mut right = prev;
            left.mul_assign(&f0);
            right.mul_assign(&f1);
            unsafe {
                buf.set_unchecked(i, left);
                buf.set_unchecked(i + half, right);
            }
        }
        size *= 2;
    }
    unsafe { buf.set_len(1 << N) };
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

/// Run the regular sumcheck rounds (all rounds except the final step)
#[inline(always)]
pub fn verify_sumcheck_rounds<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    I: NonDeterminismSource,
    const NUM_ROUNDS: usize,
    const COMMIT_BUF: usize,
>(
    seed: &mut Seed,
    initial_claim: E,
    challenges: &mut [E],
    layer_idx: usize,
) -> Result<(E, E), GKRVerificationError>
where
    [(); E::DEGREE]: Sized,
{
    let mut claim = initial_claim;
    let mut eq_prefactor = E::ONE;

    let coeff_data_words = 4 * E::DEGREE;
    let total_commit_words = BLAKE2S_DIGEST_SIZE_U32_WORDS + coeff_data_words;

    let mut commit_buf: AlignedArray64<u32, COMMIT_BUF> = AlignedArray64::from_value(0u32);
    let mut hasher = DelegatedBlake2sState::new();
    let mut draw_buf = [0u32; BLAKE2S_DIGEST_SIZE_U32_WORDS];

    for round in 0..NUM_ROUNDS {
        commit_buf[0..BLAKE2S_DIGEST_SIZE_U32_WORDS].copy_from_slice(&seed.0);

        for i in 0..coeff_data_words {
            commit_buf[BLAKE2S_DIGEST_SIZE_U32_WORDS + i] = I::read_word();
        }

        let coeffs =
            unsafe { &*commit_buf.as_ptr().add(BLAKE2S_DIGEST_SIZE_U32_WORDS).cast::<[E; 4]>() };

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

        Blake2sTranscript::draw_randomness_using_hasher(&mut hasher, seed, &mut draw_buf);
        let r_k = {
            let mut arr = LazyVec::<F, {E::DEGREE}>::new();
            for i in 0..E::DEGREE {
                let w = draw_buf[i];
                arr.push(F::from_u32_with_reduction(w));
            }
            E::from_coeffs_ref(unsafe { arr.as_slice().as_ptr().cast::<E::Coeffs>().as_ref_unchecked() })
        };

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
            let p = unsafe { *challenges.get_unchecked(round) };
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

        unsafe { *challenges.get_unchecked_mut(round) = r_k };
    }

    Ok((claim, eq_prefactor))
}

/// Verify the final step consistency: ((1 - last_prev) * f[0] + last_prev * f[1]) * eq_prefactor == claim.
#[inline(always)]
pub fn verify_final_step_check<F: PrimeField, E: FieldExtension<F> + Field>(
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

/// Compute new claims for a standard layer via linear interpolation.
#[inline(always)]
pub fn fold_standard_claims<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const NUM_ADDRS: usize,
    const ADDRS: usize,
    const BUF: usize,
>(
    eval_buf: &AlignedArray64<MaybeUninit<u32>, BUF>,
    last_r: E,
    claims: &mut LazyVec<E, ADDRS>,
) {
    let final_step_evals: &[[E; 2]] =
        unsafe { eval_buf.transmute_subslice(BLAKE2S_DIGEST_SIZE_U32_WORDS, NUM_ADDRS) };
    claims.clear();
    for i in 0..NUM_ADDRS {
        let evals = unsafe { final_step_evals.get_unchecked(i) };
        let f0 = evals[0];
        let f1 = evals[1];
        let mut diff = f1;
        diff.sub_assign(&f0);
        diff.mul_assign(&last_r);
        diff.add_assign(&f0);
        claims.push(diff);
    }
}
