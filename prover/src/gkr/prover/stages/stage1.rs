use super::*;
use fft::Twiddles;
use fft::{
    bitreverse_enumeration_inplace, distribute_powers_serial, domain_generator_for_size,
    materialize_powers_serial_starting_with_one, GoodAllocator,
};
use field::{Field, FieldExtension, PrimeField, TwoAdicField};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ColumnMajorCosetBoundTracePart<
    F: PrimeField + TwoAdicField,
    E: FieldExtension<F> + Field,
> {
    pub column: Arc<Box<[E]>>,
    pub offset: F,
}

pub fn compute_column_major_lde_from_main_domain<
    F: PrimeField + TwoAdicField,
    E: FieldExtension<F> + Field,
    A: GoodAllocator,
>(
    source_domain: Arc<Box<[E]>>,
    twiddles: &Twiddles<F, A>,
    // lde_precomputations: &LdePrecomputations<A>,
    lde_factor: usize,
) -> Vec<ColumnMajorCosetBoundTracePart<F, E>> {
    let mut result = Vec::with_capacity(lde_factor);
    result.push(ColumnMajorCosetBoundTracePart {
        column: Arc::clone(&source_domain),
        offset: F::ONE,
    });
    let other_domains =
        compute_column_major_lde_from_main_domain_inner(&source_domain[..], twiddles, lde_factor);
    result.extend(other_domains.into_iter().map(|(column, offset)| {
        ColumnMajorCosetBoundTracePart {
            column: Arc::new(column),
            offset,
        }
    }));

    result
}

pub(crate) fn compute_column_major_lde_from_main_domain_inner<
    F: PrimeField + TwoAdicField,
    E: FieldExtension<F> + Field,
    A: GoodAllocator,
>(
    source_domain: &[E],
    twiddles: &Twiddles<F, A>,
    // lde_precomputations: &LdePrecomputations<A>,
    lde_factor: usize,
) -> Vec<(Box<[E]>, F)> {
    assert!(lde_factor.is_power_of_two());

    assert!(lde_factor > 1, "No reason to call this function");

    let trace_len_log2 = source_domain.len().trailing_zeros();

    let mut ifft: Box<[E]> = source_domain.to_vec().into_boxed_slice();
    fft::naive::cache_friendly_ntt_natural_to_bitreversed(
        &mut ifft[..],
        trace_len_log2,
        &twiddles.inverse_twiddles[..],
    );

    let mut ifft = Some(ifft);

    let next_root = domain_generator_for_size::<F>(((1 << trace_len_log2) * lde_factor) as u64);
    let root_powers =
        materialize_powers_serial_starting_with_one::<F, Global>(next_root, lde_factor);
    assert_eq!(root_powers[0], F::ONE);

    let mut result = Vec::with_capacity(lde_factor - 1);

    let roots = &root_powers[1..];
    let size_inv = F::from_u32_unchecked(1 << trace_len_log2)
        .inverse()
        .unwrap();

    #[cfg(feature = "timing_logs")]
    let now = std::time::Instant::now();
    for i in 0..(lde_factor - 1) {
        let mut source = if i == (lde_factor - 2) {
            ifft.take().unwrap()
        } else {
            ifft.as_ref().unwrap().clone()
        };
        // TODO: very stupid and slow...
        bitreverse_enumeration_inplace(&mut source[..]);
        // normalize by 1/N
        let offset = roots[i];
        distribute_powers_serial(&mut source[..], size_inv, offset);
        bitreverse_enumeration_inplace(&mut source[..]);
        fft::naive::serial_ct_ntt_bitreversed_to_natural(
            &mut source[..],
            trace_len_log2,
            &twiddles.forward_twiddles,
        );
        result.push((source, offset));
    }
    #[cfg(feature = "timing_logs")]
    dbg!(now.elapsed());

    assert!(ifft.is_none());
    assert_eq!(result.len(), lde_factor - 1);

    result
}

pub(crate) fn compute_column_major_lde_from_main_domain_and_output_monomial_form<
    F: PrimeField + TwoAdicField,
    E: FieldExtension<F> + Field,
    A: GoodAllocator,
>(
    source_domain: &[E],
    twiddles: &Twiddles<F, A>,
    // lde_precomputations: &LdePrecomputations<A>,
    lde_factor: usize,
) -> (Vec<(Box<[E]>, F)>, Vec<E>) {
    assert!(lde_factor.is_power_of_two());

    assert!(lde_factor > 1, "No reason to call this function");

    let trace_len_log2 = source_domain.len().trailing_zeros();

    let mut ifft: Vec<E> = source_domain.to_vec();
    let size_inv = F::from_u32_unchecked(1 << trace_len_log2)
        .inverse()
        .unwrap();
    fft::naive::cache_friendly_ntt_natural_to_bitreversed(
        &mut ifft[..],
        trace_len_log2,
        &twiddles.inverse_twiddles[..],
    );
    for el in ifft.iter_mut() {
        el.mul_assign_by_base(&size_inv);
    }
    bitreverse_enumeration_inplace(&mut ifft[..]);

    let next_root = domain_generator_for_size::<F>(((1 << trace_len_log2) * lde_factor) as u64);
    let root_powers =
        materialize_powers_serial_starting_with_one::<F, Global>(next_root, lde_factor);
    assert_eq!(root_powers[0], F::ONE);

    let mut result = Vec::with_capacity(lde_factor - 1);

    {
        let offset = root_powers[0];
        let mut source = ifft.clone();
        // TODO: very stupid and slow...
        distribute_powers_serial(&mut source[..], F::ONE, offset);
        bitreverse_enumeration_inplace(&mut source[..]);
        fft::naive::serial_ct_ntt_bitreversed_to_natural(
            &mut source[..],
            trace_len_log2,
            &twiddles.forward_twiddles,
        );
        assert_eq!(source, source_domain);
    }

    let roots = &root_powers[1..];

    #[cfg(feature = "timing_logs")]
    let now = std::time::Instant::now();
    for i in 0..(lde_factor - 1) {
        let mut source = ifft.clone();
        // TODO: very stupid and slow...
        let offset = roots[i];
        distribute_powers_serial(&mut source[..], F::ONE, offset);
        bitreverse_enumeration_inplace(&mut source[..]);
        fft::naive::serial_ct_ntt_bitreversed_to_natural(
            &mut source[..],
            trace_len_log2,
            &twiddles.forward_twiddles,
        );
        result.push((source.into_boxed_slice(), offset));
    }
    #[cfg(feature = "timing_logs")]
    dbg!(now.elapsed());

    assert_eq!(result.len(), lde_factor - 1);

    (result, ifft)
}


pub(crate) fn compute_column_major_monomial_form_from_main_domain<
    F: PrimeField + TwoAdicField,
    E: FieldExtension<F> + Field,
    A: GoodAllocator,
>(
    source_domain: &[E],
    twiddles: &Twiddles<F, A>,
) -> Vec<E> {
    let trace_len_log2 = source_domain.len().trailing_zeros();

    let mut ifft: Vec<E> = source_domain.to_vec();
    let size_inv = F::from_u32_unchecked(1 << trace_len_log2)
        .inverse()
        .unwrap();
    fft::naive::cache_friendly_ntt_natural_to_bitreversed(
        &mut ifft[..],
        trace_len_log2,
        &twiddles.inverse_twiddles[..],
    );
    for el in ifft.iter_mut() {
        el.mul_assign_by_base(&size_inv);
    }
    bitreverse_enumeration_inplace(&mut ifft[..]);

    ifft
}