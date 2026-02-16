use std::mem::size_of;
use std::os::raw::c_void;

use era_cudart::memory::{memory_copy, DeviceAllocation};
use era_cudart::result::{CudaResult, CudaResultWrap};
use era_cudart::slice::DeviceSlice;
use era_cudart_sys::{cudaMemcpyToSymbol, cuda_struct_and_stub, CudaMemoryCopyKind};
use fft::bitreverse_enumeration_inplace;
use fft::field_utils::{distribute_powers_serial, domain_generator_for_size};
use field::Field;

use crate::field::BaseField;

type BF = BaseField;

pub const TWO_ADICITY: u32 = 27;
pub const LOG_MAX_NTT_SIZE: u32 = 24;
pub const CMEM_TWO_ADICITY: u32 = 19;
pub const CMEM_COARSE_LOG_COUNT: u32 = 10;
pub const LENGTH_CMEM_COARSE: u32 = 1 << CMEM_COARSE_LOG_COUNT;
// "- 1" accounts for NTT twiddle arrays only covering half the range
pub const CMEM_FINE_LOG_COUNT: u32 = CMEM_TWO_ADICITY - CMEM_COARSE_LOG_COUNT - 1;
pub const GMEM_COARSE_LOG_COUNT: u32 = 13;
pub const LENGTH_GMEM_COARSE: u32 = 1 << GMEM_COARSE_LOG_COUNT;
pub const CMEM_FINEST_LOG_COUNT: u32 = LOG_MAX_NTT_SIZE - GMEM_COARSE_LOG_COUNT - 1;
pub const LENGTH_CMEM_FINEST: u32 = 1 << CMEM_FINEST_LOG_COUNT;

#[repr(C)]
struct PowersLayerData {
    values: *const BF,
    mask: u32,
    log_count: u32,
}

impl PowersLayerData {
    fn new(values: *const BF, log_count: u32) -> Self {
        let mask = (1 << log_count) - 1;
        Self {
            values,
            mask,
            log_count,
        }
    }
}

#[cfg(no_cuda)]
unsafe impl Sync for PowersLayerData {}

#[repr(C)]
struct PowersData2Layer {
    fine: PowersLayerData,
    coarse: PowersLayerData,
}

impl PowersData2Layer {
    fn new(
        fine_values: *const BF,
        fine_log_count: u32,
        coarse_values: *const BF,
        coarse_log_count: u32,
    ) -> Self {
        let fine = PowersLayerData::new(fine_values, fine_log_count);
        let coarse = PowersLayerData::new(coarse_values, coarse_log_count);
        Self { fine, coarse }
    }
}

#[cfg(no_cuda)]
unsafe impl Sync for PowersData2Layer {}

cuda_struct_and_stub! { static ab_inv_sizes: [BF; TWO_ADICITY as usize + 1]; }
cuda_struct_and_stub! { static ab_fwd_cmem_twiddles_coarse: [BF; 1 << CMEM_COARSE_LOG_COUNT as usize]; }
cuda_struct_and_stub! { static ab_inv_cmem_twiddles_coarse: [BF; 1 << CMEM_COARSE_LOG_COUNT as usize]; }
cuda_struct_and_stub! { static ab_fwd_cmem_twiddles_fine: [BF; 1 << CMEM_FINE_LOG_COUNT as usize]; }
cuda_struct_and_stub! { static ab_inv_cmem_twiddles_fine: [BF; 1 << CMEM_FINE_LOG_COUNT as usize]; }
cuda_struct_and_stub! { static ab_fwd_cmem_twiddles_finest: [BF; 1 << CMEM_FINEST_LOG_COUNT as usize]; }
cuda_struct_and_stub! { static ab_inv_cmem_twiddles_finest: [BF; 1 << CMEM_FINEST_LOG_COUNT as usize]; }
cuda_struct_and_stub! { static ab_fwd_gmem_twiddles_coarse: *const BF; }
cuda_struct_and_stub! { static ab_inv_gmem_twiddles_coarse: *const BF; }

unsafe fn copy_to_symbol<T>(symbol: &T, src: &T) -> CudaResult<()> {
    cudaMemcpyToSymbol(
        symbol as *const T as *const c_void,
        src as *const T as *const c_void,
        size_of::<T>(),
        0,
        CudaMemoryCopyKind::HostToDevice,
    )
    .wrap()
}

#[allow(clippy::too_many_arguments)]
unsafe fn copy_to_symbols(
    fwd_gmem_twiddles_coarse: *const BF,
    inv_gmem_twiddles_coarse: *const BF,
    inv_sizes_host: [BF; TWO_ADICITY as usize + 1],
    fwd_cmem_twiddles_coarse: [BF; 1 << CMEM_COARSE_LOG_COUNT as usize],
    inv_cmem_twiddles_coarse: [BF; 1 << CMEM_COARSE_LOG_COUNT as usize],
    fwd_cmem_twiddles_fine: [BF; 1 << CMEM_FINE_LOG_COUNT as usize],
    inv_cmem_twiddles_fine: [BF; 1 << CMEM_FINE_LOG_COUNT as usize],
    fwd_cmem_twiddles_finest: [BF; 1 << CMEM_FINEST_LOG_COUNT as usize],
    inv_cmem_twiddles_finest: [BF; 1 << CMEM_FINEST_LOG_COUNT as usize],
) -> CudaResult<()> {
    copy_to_symbol(&ab_inv_sizes, &inv_sizes_host)?;
    copy_to_symbol(&ab_fwd_gmem_twiddles_coarse, &fwd_gmem_twiddles_coarse)?;
    copy_to_symbol(&ab_inv_gmem_twiddles_coarse, &inv_gmem_twiddles_coarse)?;
    copy_to_symbol(&ab_fwd_cmem_twiddles_coarse, &fwd_cmem_twiddles_coarse)?;
    copy_to_symbol(&ab_inv_cmem_twiddles_coarse, &inv_cmem_twiddles_coarse)?;
    copy_to_symbol(&ab_fwd_cmem_twiddles_fine, &fwd_cmem_twiddles_fine)?;
    copy_to_symbol(&ab_inv_cmem_twiddles_fine, &inv_cmem_twiddles_fine)?;
    copy_to_symbol(&ab_fwd_cmem_twiddles_finest, &fwd_cmem_twiddles_finest)?;
    copy_to_symbol(&ab_inv_cmem_twiddles_finest, &inv_cmem_twiddles_finest)?;
    Ok(())
}

fn generate_powers_dev<F: Field>(
    base: F,
    powers_dev: &mut DeviceSlice<F>,
    bit_reverse: bool,
    swizzle: bool,
) -> CudaResult<()> {
    let mut powers_host = vec![F::ONE; powers_dev.len()];
    distribute_powers_serial::<F, F>(&mut powers_host, F::ONE, base);
    if bit_reverse {
        bitreverse_enumeration_inplace(&mut powers_host);
    }
    let linear_to_swizzled = |i: usize| -> usize {
        const LOG_BANKS: usize = 5;
        const BANKS: usize = 1 << LOG_BANKS;
        const BANK_MASK: usize = BANKS - 1;
        let x = i & BANK_MASK;
        let y = i >> LOG_BANKS;
        y * BANKS + ((y & BANK_MASK) ^ x)
    };
    if swizzle {
        let mut powers_swizzled_host = vec![F::ZERO; powers_dev.len()];
        for i in 0..powers_dev.len() {
            powers_swizzled_host[i] = powers_host[linear_to_swizzled(i)];
        }
        memory_copy(powers_dev, &powers_swizzled_host)?;
        return Ok(());
    }
    memory_copy(powers_dev, &powers_host)
}

pub struct DeviceContext {
    pub fwd_gmem_twiddles_coarse: DeviceAllocation<BF>,
    pub inv_gmem_twiddles_coarse: DeviceAllocation<BF>,
}

impl DeviceContext {
    pub fn create(powers_of_w_coarse_log_count: u32) -> CudaResult<Self> {
        let two_inv = BF::new(2).inverse().expect("must exist");
        let mut inv_sizes_host = [BF::ONE; TWO_ADICITY as usize + 1];
        distribute_powers_serial(&mut inv_sizes_host, BF::ONE, two_inv);

        let mut fwd_gmem_twiddles_coarse =
            DeviceAllocation::<BF>::alloc(LENGTH_GMEM_COARSE as usize)?;
        let generator_fwd_gmem_coarse =
            domain_generator_for_size::<BF>((LENGTH_GMEM_COARSE * 2) as u64);
        generate_powers_dev(generator_fwd_gmem_coarse, &mut fwd_gmem_twiddles_coarse, true, true)?;

        let mut inv_gmem_twiddles_coarse =
            DeviceAllocation::<BF>::alloc(LENGTH_GMEM_COARSE as usize)?;
        let generator_inv_gmem_coarse = generator_fwd_gmem_coarse.inverse().expect("must exist");
        generate_powers_dev(generator_inv_gmem_coarse, &mut inv_gmem_twiddles_coarse, true, true)?;

        // trust me
        let generator_fwd_cmem_coarse =
            domain_generator_for_size::<BF>((LENGTH_CMEM_COARSE * 2) as u64);
        let mut fwd_cmem_twiddles_coarse = [BF::ONE; 1 << CMEM_COARSE_LOG_COUNT as usize];
        distribute_powers_serial(
            &mut fwd_cmem_twiddles_coarse,
            BF::ONE,
            generator_fwd_cmem_coarse,
        );
        bitreverse_enumeration_inplace(&mut fwd_cmem_twiddles_coarse);

        let generator_inv_cmem_coarse = generator_fwd_cmem_coarse.inverse().expect("must exist");
        let mut inv_cmem_twiddles_coarse = [BF::ONE; 1 << CMEM_COARSE_LOG_COUNT as usize];
        distribute_powers_serial(
            &mut inv_cmem_twiddles_coarse,
            BF::ONE,
            generator_inv_cmem_coarse,
        );
        bitreverse_enumeration_inplace(&mut inv_cmem_twiddles_coarse);

        let generator_fwd_cmem_fine = domain_generator_for_size::<BF>(1u64 << CMEM_TWO_ADICITY);
        let mut fwd_cmem_twiddles_fine = [BF::ONE; 1 << CMEM_FINE_LOG_COUNT as usize];
        distribute_powers_serial(
            &mut fwd_cmem_twiddles_fine,
            BF::ONE,
            generator_fwd_cmem_fine,
        );
        bitreverse_enumeration_inplace(&mut fwd_cmem_twiddles_fine);

        let generator_inv_cmem_fine = generator_fwd_cmem_fine.inverse().expect("must exist");
        let mut inv_cmem_twiddles_fine = [BF::ONE; 1 << CMEM_FINE_LOG_COUNT as usize];
        distribute_powers_serial(
            &mut inv_cmem_twiddles_fine,
            BF::ONE,
            generator_inv_cmem_fine,
        );
        bitreverse_enumeration_inplace(&mut inv_cmem_twiddles_fine);

        let generator_fwd_cmem_finest = domain_generator_for_size::<BF>(1u64 << LOG_MAX_NTT_SIZE);
        let mut fwd_cmem_twiddles_finest = [BF::ONE; 1 << CMEM_FINEST_LOG_COUNT as usize];
        distribute_powers_serial(
            &mut fwd_cmem_twiddles_finest,
            BF::ONE,
            generator_fwd_cmem_finest,
        );
        bitreverse_enumeration_inplace(&mut fwd_cmem_twiddles_finest);

        let generator_inv_cmem_finest = generator_fwd_cmem_finest.inverse().expect("must exist");
        let mut inv_cmem_twiddles_finest = [BF::ONE; 1 << CMEM_FINEST_LOG_COUNT as usize];
        distribute_powers_serial(
            &mut inv_cmem_twiddles_finest,
            BF::ONE,
            generator_inv_cmem_finest,
        );
        bitreverse_enumeration_inplace(&mut inv_cmem_twiddles_finest);

        unsafe {
            copy_to_symbols(
                fwd_gmem_twiddles_coarse.as_ptr(),
                inv_gmem_twiddles_coarse.as_ptr(),
                inv_sizes_host,
                fwd_cmem_twiddles_coarse,
                inv_cmem_twiddles_coarse,
                fwd_cmem_twiddles_fine,
                inv_cmem_twiddles_fine,
                fwd_cmem_twiddles_finest,
                inv_cmem_twiddles_finest,
            )?;
        }
        Ok(Self {
            fwd_gmem_twiddles_coarse,
            inv_gmem_twiddles_coarse,
        })
    }

    pub fn destroy(self) -> CudaResult<()> {
        self.fwd_gmem_twiddles_coarse.free()?;
        self.inv_gmem_twiddles_coarse.free()?;
        Ok(())
    }
}
