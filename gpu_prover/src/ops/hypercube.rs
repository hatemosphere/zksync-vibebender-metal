use era_cudart::execution::{CudaLaunchConfig, Dim3, KernelFunction};
use era_cudart::result::{CudaResult, CudaResultWrap};
use era_cudart::slice::DeviceSlice;
use era_cudart::stream::CudaStream;
use era_cudart::{cuda_kernel_declaration, cuda_kernel_signature_arguments_and_function};
use era_cudart_sys::{cudaFuncSetAttribute, CudaFuncAttribute};

use crate::field::BF;

const BLOCK_THREADS: u32 = 256;
const LOG21_INITIAL_BLOCK_THREADS: u32 = 1024;
const LOG21_NONINITIAL_BLOCK_THREADS: u32 = 256;
const LOG21_INITIAL_DYNAMIC_SMEM_BYTES: usize = 0;
const LOG21_NONINITIAL_GRID_MULTIPLIER: u32 = 4;
const LOG24_INITIAL_BLOCK_THREADS: u32 = 1024;
const LOG24_INITIAL_DYNAMIC_SMEM_BYTES: usize = 0;
const LOG24_NONINITIAL_BLOCK_THREADS: u32 = 1024;
const LOG24_NONINITIAL_DYNAMIC_SMEM_BYTES: usize = 1usize << 16;
const LOG24_NONINITIAL_GRID_MULTIPLIER: u32 = 4;
const MIN_SUPPORTED_LOG_ROWS: u32 = 21;
const MAX_SUPPORTED_LOG_ROWS: u32 = 24;
const NONINITIAL_PARTITION_LOG_ROWS: u32 = 5;

const LOG21_INITIAL_ROUNDS: u32 = 13;
const LOG21_NONINITIAL_STAGE2_ROUNDS: u32 = 8;
const LOG21_NONINITIAL_STAGE2_START: u32 = LOG21_INITIAL_ROUNDS;

const LOG22_INITIAL_ROUNDS: u32 = 11;
const LOG22_NONINITIAL_STAGE2_ROUNDS: u32 = 5;
const LOG22_NONINITIAL_STAGE3_ROUNDS: u32 = 6;
const LOG22_NONINITIAL_STAGE2_START: u32 = LOG22_INITIAL_ROUNDS;
const LOG22_NONINITIAL_STAGE3_START: u32 =
    LOG22_NONINITIAL_STAGE2_START + LOG22_NONINITIAL_STAGE2_ROUNDS;

const LOG23_INITIAL_ROUNDS: u32 = 11;
const LOG23_NONINITIAL_STAGE2_ROUNDS: u32 = 6;
const LOG23_NONINITIAL_STAGE3_ROUNDS: u32 = 6;
const LOG23_NONINITIAL_STAGE2_START: u32 = LOG23_INITIAL_ROUNDS;
const LOG23_NONINITIAL_STAGE3_START: u32 =
    LOG23_NONINITIAL_STAGE2_START + LOG23_NONINITIAL_STAGE2_ROUNDS;

const LOG24_INITIAL_ROUNDS: u32 = 13;
const LOG24_NONINITIAL_STAGE2_ROUNDS: u32 = 11;
const LOG24_NONINITIAL_STAGE2_START: u32 = LOG24_INITIAL_ROUNDS;

const LOG24_3PASS_INITIAL_ROUNDS: u32 = 12;
const LOG24_3PASS_NONINITIAL_STAGE2_ROUNDS: u32 = 6;
const LOG24_3PASS_NONINITIAL_STAGE3_ROUNDS: u32 = 6;
const LOG24_3PASS_NONINITIAL_STAGE2_START: u32 = LOG24_3PASS_INITIAL_ROUNDS;
const LOG24_3PASS_NONINITIAL_STAGE3_START: u32 =
    LOG24_3PASS_NONINITIAL_STAGE2_START + LOG24_3PASS_NONINITIAL_STAGE2_ROUNDS;

cuda_kernel_signature_arguments_and_function!(
    HypercubeBitrevInitial,
    src: *const BF,
    dst: *mut BF,
);

cuda_kernel_signature_arguments_and_function!(
    HypercubeBitrevNonInitial,
    src: *const BF,
    dst: *mut BF,
    start_stage: u32,
);

macro_rules! declare_h2m_initial_kernel {
    ($name:ident) => {
        cuda_kernel_declaration!(
            $name(
                src: *const BF,
                dst: *mut BF,
            )
        );
    };
}

macro_rules! declare_h2m_noninitial_kernel {
    ($name:ident) => {
        cuda_kernel_declaration!(
            $name(
                src: *const BF,
                dst: *mut BF,
                start_stage: u32,
            )
        );
    };
}

declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial12_out_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial12_in_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial14_out_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial14_in_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial14_out_512_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial14_in_512_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial15_out_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial15_in_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial15_out_512_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial15_in_512_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial13_out_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial13_in_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial11_out_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial11_in_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial5_stage2_out_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial5_stage2_in_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial5_stage3_in_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial5_stage3_out_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage2_out_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage2_in_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage3_in_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage3_out_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial5_stage2_out_start11_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial5_stage2_in_start11_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial5_stage2_out_start11_x2_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial5_stage2_in_start11_x2_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial5_stage3_out_start16_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial5_stage3_in_start16_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial7_stage3_out_start14_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial7_stage3_in_start14_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial8_stage3_out_start13_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial8_stage3_in_start13_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial9_stage3_out_start15_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial9_stage3_in_start15_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial9_stage3_out_start15_x2_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial9_stage3_in_start15_x2_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial11_stage3_out_start13_x4_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial11_stage3_in_start13_x4_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial10_stage3_out_start14_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial10_stage3_in_start14_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial10_stage3_out_start14_x2_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial10_stage3_in_start14_x2_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage2_out_start11_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage2_out_start12_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage2_in_start11_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage2_in_start12_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage3_out_start16_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage3_in_start16_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage3_out_start17_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage3_out_start18_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage3_in_start17_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage3_in_start18_kernel);

fn validate_len(rows: usize) -> u32 {
    assert!(rows.is_power_of_two());
    let log_rows = rows.trailing_zeros();
    assert!(
        (MIN_SUPPORTED_LOG_ROWS..=MAX_SUPPORTED_LOG_ROWS).contains(&log_rows),
        "only log21/log22/log23/log24 (2^21..2^24 rows) are supported",
    );
    log_rows
}

#[derive(Clone, Copy)]
struct LaunchPlan {
    initial_kernel: HypercubeBitrevInitialSignature,
    noninitial_stage2_kernel: HypercubeBitrevNonInitialSignature,
    noninitial_stage3_kernel: HypercubeBitrevNonInitialSignature,
    initial_rounds: u32,
    noninitial_stage2_rounds: u32,
    noninitial_stage3_rounds: u32,
    noninitial_stage2_tiles_per_cta: u32,
    noninitial_stage3_tiles_per_cta: u32,
    noninitial_stage2_start: u32,
    noninitial_stage3_start: u32,
}

fn select_out_of_place_plan(log_rows: u32) -> LaunchPlan {
    match log_rows {
        24 => LaunchPlan {
            initial_kernel: ab_h2m_bitrev_bf_initial12_out_kernel,
            noninitial_stage2_kernel: ab_h2m_bitrev_bf_noninitial6_stage2_out_start12_kernel,
            noninitial_stage3_kernel: ab_h2m_bitrev_bf_noninitial6_stage3_out_start18_kernel,
            initial_rounds: LOG24_3PASS_INITIAL_ROUNDS,
            noninitial_stage2_rounds: LOG24_3PASS_NONINITIAL_STAGE2_ROUNDS,
            noninitial_stage3_rounds: LOG24_3PASS_NONINITIAL_STAGE3_ROUNDS,
            noninitial_stage2_tiles_per_cta: 1,
            noninitial_stage3_tiles_per_cta: 1,
            noninitial_stage2_start: LOG24_3PASS_NONINITIAL_STAGE2_START,
            noninitial_stage3_start: LOG24_3PASS_NONINITIAL_STAGE3_START,
        },
        23 => LaunchPlan {
            initial_kernel: ab_h2m_bitrev_bf_initial11_out_kernel,
            noninitial_stage2_kernel: ab_h2m_bitrev_bf_noninitial6_stage2_out_start11_kernel,
            noninitial_stage3_kernel: ab_h2m_bitrev_bf_noninitial6_stage3_out_start17_kernel,
            initial_rounds: LOG23_INITIAL_ROUNDS,
            noninitial_stage2_rounds: LOG23_NONINITIAL_STAGE2_ROUNDS,
            noninitial_stage3_rounds: LOG23_NONINITIAL_STAGE3_ROUNDS,
            noninitial_stage2_tiles_per_cta: 1,
            noninitial_stage3_tiles_per_cta: 1,
            noninitial_stage2_start: LOG23_NONINITIAL_STAGE2_START,
            noninitial_stage3_start: LOG23_NONINITIAL_STAGE3_START,
        },
        22 => LaunchPlan {
            initial_kernel: ab_h2m_bitrev_bf_initial11_out_kernel,
            noninitial_stage2_kernel: ab_h2m_bitrev_bf_noninitial5_stage2_out_start11_x2_kernel,
            noninitial_stage3_kernel: ab_h2m_bitrev_bf_noninitial6_stage3_out_start16_kernel,
            initial_rounds: LOG22_INITIAL_ROUNDS,
            noninitial_stage2_rounds: LOG22_NONINITIAL_STAGE2_ROUNDS,
            noninitial_stage3_rounds: LOG22_NONINITIAL_STAGE3_ROUNDS,
            noninitial_stage2_tiles_per_cta: 2,
            noninitial_stage3_tiles_per_cta: 1,
            noninitial_stage2_start: LOG22_NONINITIAL_STAGE2_START,
            noninitial_stage3_start: LOG22_NONINITIAL_STAGE3_START,
        },
        _ => unreachable!("validate_len enforces supported log rows"),
    }
}

fn select_in_place_plan(log_rows: u32) -> LaunchPlan {
    match log_rows {
        24 => LaunchPlan {
            initial_kernel: ab_h2m_bitrev_bf_initial12_in_kernel,
            noninitial_stage2_kernel: ab_h2m_bitrev_bf_noninitial6_stage2_in_start12_kernel,
            noninitial_stage3_kernel: ab_h2m_bitrev_bf_noninitial6_stage3_in_start18_kernel,
            initial_rounds: LOG24_3PASS_INITIAL_ROUNDS,
            noninitial_stage2_rounds: LOG24_3PASS_NONINITIAL_STAGE2_ROUNDS,
            noninitial_stage3_rounds: LOG24_3PASS_NONINITIAL_STAGE3_ROUNDS,
            noninitial_stage2_tiles_per_cta: 1,
            noninitial_stage3_tiles_per_cta: 1,
            noninitial_stage2_start: LOG24_3PASS_NONINITIAL_STAGE2_START,
            noninitial_stage3_start: LOG24_3PASS_NONINITIAL_STAGE3_START,
        },
        23 => LaunchPlan {
            initial_kernel: ab_h2m_bitrev_bf_initial11_in_kernel,
            noninitial_stage2_kernel: ab_h2m_bitrev_bf_noninitial6_stage2_in_start11_kernel,
            noninitial_stage3_kernel: ab_h2m_bitrev_bf_noninitial6_stage3_in_start17_kernel,
            initial_rounds: LOG23_INITIAL_ROUNDS,
            noninitial_stage2_rounds: LOG23_NONINITIAL_STAGE2_ROUNDS,
            noninitial_stage3_rounds: LOG23_NONINITIAL_STAGE3_ROUNDS,
            noninitial_stage2_tiles_per_cta: 1,
            noninitial_stage3_tiles_per_cta: 1,
            noninitial_stage2_start: LOG23_NONINITIAL_STAGE2_START,
            noninitial_stage3_start: LOG23_NONINITIAL_STAGE3_START,
        },
        22 => LaunchPlan {
            initial_kernel: ab_h2m_bitrev_bf_initial11_in_kernel,
            noninitial_stage2_kernel: ab_h2m_bitrev_bf_noninitial5_stage2_in_start11_x2_kernel,
            noninitial_stage3_kernel: ab_h2m_bitrev_bf_noninitial6_stage3_in_start16_kernel,
            initial_rounds: LOG22_INITIAL_ROUNDS,
            noninitial_stage2_rounds: LOG22_NONINITIAL_STAGE2_ROUNDS,
            noninitial_stage3_rounds: LOG22_NONINITIAL_STAGE3_ROUNDS,
            noninitial_stage2_tiles_per_cta: 2,
            noninitial_stage3_tiles_per_cta: 1,
            noninitial_stage2_start: LOG22_NONINITIAL_STAGE2_START,
            noninitial_stage3_start: LOG22_NONINITIAL_STAGE3_START,
        },
        _ => unreachable!("validate_len enforces supported log rows"),
    }
}

fn noninitial_grid(rows: usize, rounds: u32, tiles_per_cta: u32) -> u32 {
    let tiles = rows >> (NONINITIAL_PARTITION_LOG_ROWS + rounds);
    let tiles_per_cta = tiles_per_cta as usize;
    debug_assert!(tiles_per_cta > 0);
    debug_assert_eq!(tiles % tiles_per_cta, 0);
    (tiles / tiles_per_cta) as u32
}

fn log21_noninitial_grid(rows: usize) -> u32 {
    noninitial_grid(rows, LOG21_NONINITIAL_STAGE2_ROUNDS, 1) * LOG21_NONINITIAL_GRID_MULTIPLIER
}

fn log24_noninitial_grid(rows: usize) -> u32 {
    noninitial_grid(rows, LOG24_NONINITIAL_STAGE2_ROUNDS, 1) * LOG24_NONINITIAL_GRID_MULTIPLIER
}

fn configure_log21_initial_dynamic_smem(
    initial_kernel: HypercubeBitrevInitialSignature,
) -> CudaResult<()> {
    let kernel_function = HypercubeBitrevInitialFunction(initial_kernel);
    unsafe {
        cudaFuncSetAttribute(
            kernel_function.as_ptr(),
            CudaFuncAttribute::MaxDynamicSharedMemorySize,
            LOG21_INITIAL_DYNAMIC_SMEM_BYTES as i32,
        )
    }
    .wrap()
}

fn configure_log24_noninitial_dynamic_smem(
    noninitial_kernel: HypercubeBitrevNonInitialSignature,
) -> CudaResult<()> {
    let kernel_function = HypercubeBitrevNonInitialFunction(noninitial_kernel);
    unsafe {
        cudaFuncSetAttribute(
            kernel_function.as_ptr(),
            CudaFuncAttribute::MaxDynamicSharedMemorySize,
            LOG24_NONINITIAL_DYNAMIC_SMEM_BYTES as i32,
        )
    }
    .wrap()
}

fn launch_chain(
    launch0_kernel: HypercubeBitrevInitialSignature,
    launch1_kernel: HypercubeBitrevNonInitialSignature,
    launch2_kernel: HypercubeBitrevNonInitialSignature,
    initial_rounds: u32,
    noninitial_stage2_rounds: u32,
    noninitial_stage3_rounds: u32,
    noninitial_stage2_tiles_per_cta: u32,
    noninitial_stage3_tiles_per_cta: u32,
    noninitial_stage2_start: u32,
    noninitial_stage3_start: u32,
    launch0_src: *const BF,
    launch_dst: *mut BF,
    rows: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    // Locked cache policy by stage role for the 3-launch path:
    // - log23 schedule: [11, 6, 6]
    // - log22 schedule: [11, 5, 6]
    // - log24 schedule: [12, 6, 6]
    // - log21 is handled by a dedicated 2-launch path.
    // Noninitial stages use fixed-start kernel entrypoints selected on host.
    // out-of-place:  #1 ld.cs/st.wt, #2 ld.cs/st.wt, #3 ld.cs/st.cs
    // in-place:      #1 ld.cg/st.wt, #2 ld.ca/st.wt, #3 ld.ca/st.cs
    let grid_initial = (rows >> initial_rounds) as u32;
    let grid_stage2 = noninitial_grid(rows, noninitial_stage2_rounds, noninitial_stage2_tiles_per_cta);
    let grid_stage3 = noninitial_grid(rows, noninitial_stage3_rounds, noninitial_stage3_tiles_per_cta);

    let config0 = CudaLaunchConfig::basic(
        Dim3 {
            x: grid_initial,
            y: 1,
            z: 1,
        },
        BLOCK_THREADS,
        stream,
    );
    let args0 = HypercubeBitrevInitialArguments::new(launch0_src, launch_dst);
    HypercubeBitrevInitialFunction(launch0_kernel).launch(&config0, &args0)?;

    let launch1_src = launch_dst as *const BF;

    let config1 = CudaLaunchConfig::basic(
        Dim3 {
            x: grid_stage2,
            y: 1,
            z: 1,
        },
        BLOCK_THREADS,
        stream,
    );
    let args1 =
        HypercubeBitrevNonInitialArguments::new(launch1_src, launch_dst, noninitial_stage2_start);
    HypercubeBitrevNonInitialFunction(launch1_kernel).launch(&config1, &args1)?;

    let config2 = CudaLaunchConfig::basic(
        Dim3 {
            x: grid_stage3,
            y: 1,
            z: 1,
        },
        BLOCK_THREADS,
        stream,
    );
    let args2 =
        HypercubeBitrevNonInitialArguments::new(launch1_src, launch_dst, noninitial_stage3_start);
    HypercubeBitrevNonInitialFunction(launch2_kernel).launch(&config2, &args2)?;

    Ok(())
}

fn launch_log21_chain_2launch(
    initial_kernel: HypercubeBitrevInitialSignature,
    noninitial_kernel: HypercubeBitrevNonInitialSignature,
    launch0_src: *const BF,
    launch_dst: *mut BF,
    rows: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    let grid_initial = (rows >> LOG21_INITIAL_ROUNDS) as u32;
    let grid_noninitial = log21_noninitial_grid(rows);
    if LOG21_INITIAL_DYNAMIC_SMEM_BYTES > 0 {
        configure_log21_initial_dynamic_smem(initial_kernel)?;
    }

    let config0 = CudaLaunchConfig::builder()
        .grid_dim(Dim3 {
            x: grid_initial,
            y: 1,
            z: 1,
        })
        .block_dim(LOG21_INITIAL_BLOCK_THREADS)
        .dynamic_smem_bytes(LOG21_INITIAL_DYNAMIC_SMEM_BYTES)
        .stream(stream)
        .build();
    let args0 = HypercubeBitrevInitialArguments::new(launch0_src, launch_dst);
    HypercubeBitrevInitialFunction(initial_kernel).launch(&config0, &args0)?;

    // The second launch is the final noninitial8 stage at fixed start=13.
    let launch1_src = launch_dst as *const BF;
    let config1 = CudaLaunchConfig::basic(
        Dim3 {
            x: grid_noninitial,
            y: 1,
            z: 1,
        },
        LOG21_NONINITIAL_BLOCK_THREADS,
        stream,
    );
    let args1 =
        HypercubeBitrevNonInitialArguments::new(launch1_src, launch_dst, LOG21_NONINITIAL_STAGE2_START);
    HypercubeBitrevNonInitialFunction(noninitial_kernel).launch(&config1, &args1)?;

    Ok(())
}

fn launch_log24_chain_2launch(
    initial_kernel: HypercubeBitrevInitialSignature,
    noninitial_kernel: HypercubeBitrevNonInitialSignature,
    launch0_src: *const BF,
    launch_dst: *mut BF,
    rows: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    let grid_initial = (rows >> LOG24_INITIAL_ROUNDS) as u32;
    let grid_noninitial = log24_noninitial_grid(rows);
    if LOG24_INITIAL_DYNAMIC_SMEM_BYTES > 0 {
        configure_log21_initial_dynamic_smem(initial_kernel)?;
    }
    if LOG24_NONINITIAL_DYNAMIC_SMEM_BYTES > 0 {
        configure_log24_noninitial_dynamic_smem(noninitial_kernel)?;
    }

    let config0 = CudaLaunchConfig::builder()
        .grid_dim(Dim3 {
            x: grid_initial,
            y: 1,
            z: 1,
        })
        .block_dim(LOG24_INITIAL_BLOCK_THREADS)
        .dynamic_smem_bytes(LOG24_INITIAL_DYNAMIC_SMEM_BYTES)
        .stream(stream)
        .build();
    let args0 = HypercubeBitrevInitialArguments::new(launch0_src, launch_dst);
    HypercubeBitrevInitialFunction(initial_kernel).launch(&config0, &args0)?;

    // The second launch is the final noninitial11 stage at fixed start=13.
    let launch1_src = launch_dst as *const BF;
    let config1 = CudaLaunchConfig::builder()
        .grid_dim(Dim3 {
            x: grid_noninitial,
            y: 1,
            z: 1,
        })
        .block_dim(LOG24_NONINITIAL_BLOCK_THREADS)
        .dynamic_smem_bytes(LOG24_NONINITIAL_DYNAMIC_SMEM_BYTES)
        .stream(stream)
        .build();
    let args1 =
        HypercubeBitrevNonInitialArguments::new(launch1_src, launch_dst, LOG24_NONINITIAL_STAGE2_START);
    HypercubeBitrevNonInitialFunction(noninitial_kernel).launch(&config1, &args1)?;

    Ok(())
}

pub fn hypercube_evals_into_coeffs_bitrev_bf(
    src: &DeviceSlice<BF>,
    dst: &mut DeviceSlice<BF>,
    stream: &CudaStream,
) -> CudaResult<()> {
    let rows = src.len();
    assert_eq!(dst.len(), rows);
    let log_rows = validate_len(rows);
    if log_rows == 21 {
        return launch_log21_chain_2launch(
            ab_h2m_bitrev_bf_initial13_out_kernel,
            ab_h2m_bitrev_bf_noninitial8_stage3_out_start13_kernel,
            src.as_ptr(),
            dst.as_mut_ptr(),
            rows,
            stream,
        );
    }
    let plan = select_out_of_place_plan(log_rows);

    launch_chain(
        plan.initial_kernel,
        plan.noninitial_stage2_kernel,
        plan.noninitial_stage3_kernel,
        plan.initial_rounds,
        plan.noninitial_stage2_rounds,
        plan.noninitial_stage3_rounds,
        plan.noninitial_stage2_tiles_per_cta,
        plan.noninitial_stage3_tiles_per_cta,
        plan.noninitial_stage2_start,
        plan.noninitial_stage3_start,
        src.as_ptr(),
        dst.as_mut_ptr(),
        rows,
        stream,
    )
}

pub fn hypercube_evals_into_coeffs_bitrev_bf_in_place(
    values: &mut DeviceSlice<BF>,
    stream: &CudaStream,
) -> CudaResult<()> {
    let log_rows = validate_len(values.len());
    if log_rows == 21 {
        let dst = values.as_mut_ptr();
        return launch_log21_chain_2launch(
            ab_h2m_bitrev_bf_initial13_in_kernel,
            ab_h2m_bitrev_bf_noninitial8_stage3_in_start13_kernel,
            dst as *const BF,
            dst,
            values.len(),
            stream,
        );
    }
    let plan = select_in_place_plan(log_rows);
    let dst = values.as_mut_ptr();

    launch_chain(
        plan.initial_kernel,
        plan.noninitial_stage2_kernel,
        plan.noninitial_stage3_kernel,
        plan.initial_rounds,
        plan.noninitial_stage2_rounds,
        plan.noninitial_stage3_rounds,
        plan.noninitial_stage2_tiles_per_cta,
        plan.noninitial_stage3_tiles_per_cta,
        plan.noninitial_stage2_start,
        plan.noninitial_stage3_start,
        dst as *const BF,
        dst,
        values.len(),
        stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use era_cudart::device::{device_get_attribute, get_device};
    use era_cudart::memory::{memory_copy_async, DeviceAllocation};
    use era_cudart_sys::CudaDeviceAttr;
    use field::Field;
    use prover::gkr::whir::hypercube_to_monomial::multivariate_hypercube_evals_into_coeffs;
    use rand::{rng, rngs::StdRng, Rng, SeedableRng};
    use std::time::Instant;

    const PROFILE_MULTI_WARMUP_ITERS: usize = 20;
    const PROFILE_MULTI_MEASURE_ITERS: usize = 100;
    const PROFILE_STAGE_WARMUP_ITERS: usize = 20;
    const PROFILE_STAGE_MEASURE_ITERS: usize = 100;
    const L2_RING_TARGET_BYTES_MULTIPLIER: usize = 2;
    const L2_RING_TARGET_MIN_BYTES: usize = 8 * 1024 * 1024;
    const LOG21_ROWS: usize = 1usize << 21;
    const LOG22_ROWS: usize = 1usize << 22;
    const LOG23_ROWS: usize = 1usize << 23;
    const LOG24_ROWS: usize = 1usize << 24;

    #[derive(Clone, Copy)]
    struct ProfileStats {
        mean: f64,
        median: f64,
        p90: f64,
        p95: f64,
        min: f64,
        max: f64,
        stddev: f64,
        cv_pct: f64,
    }

    fn stage1_ring_len(rows: usize) -> usize {
        let row_bytes = rows * std::mem::size_of::<BF>();
        let device_id = get_device().unwrap();
        let l2_bytes = device_get_attribute(CudaDeviceAttr::L2CacheSize, device_id).unwrap() as usize;
        let target_bytes =
            (l2_bytes * L2_RING_TARGET_BYTES_MULTIPLIER).max(L2_RING_TARGET_MIN_BYTES);
        ((target_bytes + row_bytes - 1) / row_bytes).max(2)
    }

    fn alloc_filled_ring(
        rows: usize,
        ring_len: usize,
        h_input: &[BF],
        stream: &CudaStream,
    ) -> Vec<DeviceAllocation<BF>> {
        let mut ring = Vec::with_capacity(ring_len);
        for _ in 0..ring_len {
            let mut d_values = DeviceAllocation::alloc(rows).unwrap();
            memory_copy_async(&mut d_values, h_input, stream).unwrap();
            ring.push(d_values);
        }
        stream.synchronize().unwrap();
        ring
    }

    fn alloc_empty_ring(rows: usize, ring_len: usize) -> Vec<DeviceAllocation<BF>> {
        (0..ring_len)
            .map(|_| DeviceAllocation::alloc(rows).unwrap())
            .collect()
    }

    fn bitreverse_permute(values: &[BF]) -> Vec<BF> {
        assert!(values.len().is_power_of_two());
        let log_rows = values.len().trailing_zeros();
        let mut out = vec![BF::ZERO; values.len()];
        for (i, value) in values.iter().copied().enumerate() {
            let j = (i as u32).reverse_bits() >> (u32::BITS - log_rows);
            out[j as usize] = value;
        }
        out
    }

    fn reference_h2m_bitrev(input_bitrev: &[BF]) -> Vec<BF> {
        let mut canonical = bitreverse_permute(input_bitrev);
        let size_log2 = canonical.len().trailing_zeros();
        multivariate_hypercube_evals_into_coeffs(&mut canonical, size_log2);
        bitreverse_permute(&canonical)
    }

    fn run_out_of_place_case(rows: usize) {
        let mut rng = rng();
        let h_input = (0..rows)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect::<Vec<_>>();
        let h_expected = reference_h2m_bitrev(&h_input);

        let stream = CudaStream::default();
        let mut d_src = DeviceAllocation::alloc(rows).unwrap();
        let mut d_dst = DeviceAllocation::alloc(rows).unwrap();
        memory_copy_async(&mut d_src, &h_input, &stream).unwrap();

        hypercube_evals_into_coeffs_bitrev_bf(&d_src, &mut d_dst, &stream).unwrap();
        let mut h_actual = vec![BF::ZERO; rows];
        memory_copy_async(&mut h_actual, &d_dst, &stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(h_actual, h_expected);
    }

    fn run_in_place_case(rows: usize) {
        let mut rng = rng();
        let h_input = (0..rows)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect::<Vec<_>>();
        let h_expected = reference_h2m_bitrev(&h_input);

        let stream = CudaStream::default();
        let mut d_values = DeviceAllocation::alloc(rows).unwrap();
        memory_copy_async(&mut d_values, &h_input, &stream).unwrap();

        hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut d_values, &stream).unwrap();
        let mut h_actual = vec![BF::ZERO; rows];
        memory_copy_async(&mut h_actual, &d_values, &stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(h_actual, h_expected);
    }

    fn percentile_from_sorted(sorted_samples_us: &[f64], percentile: f64) -> f64 {
        assert!(!sorted_samples_us.is_empty());
        let clamped = percentile.clamp(0.0, 1.0);
        let idx = ((sorted_samples_us.len() - 1) as f64 * clamped).round() as usize;
        sorted_samples_us[idx]
    }

    fn run_profile_invocations<F>(
        warmup_iters: usize,
        measure_iters: usize,
        stream: &CudaStream,
        mut launch: F,
    ) -> Vec<f64>
    where
        F: FnMut() -> CudaResult<()>,
    {
        assert!(measure_iters > 0);

        for _ in 0..warmup_iters {
            launch().unwrap();
        }
        stream.synchronize().unwrap();

        let mut samples_us = Vec::with_capacity(measure_iters);
        for _ in 0..measure_iters {
            let start = Instant::now();
            launch().unwrap();
            stream.synchronize().unwrap();
            samples_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }
        samples_us
    }

    fn run_profile_compare_invocations<F, G>(
        warmup_iters: usize,
        measure_iters: usize,
        stream: &CudaStream,
        mut launch_a: F,
        mut launch_b: G,
    ) -> (Vec<f64>, Vec<f64>)
    where
        F: FnMut() -> CudaResult<()>,
        G: FnMut() -> CudaResult<()>,
    {
        assert!(measure_iters > 0);

        for _ in 0..warmup_iters {
            launch_a().unwrap();
            launch_b().unwrap();
        }
        stream.synchronize().unwrap();

        let mut samples_a_us = Vec::with_capacity(measure_iters);
        let mut samples_b_us = Vec::with_capacity(measure_iters);
        for _ in 0..measure_iters {
            let start_a = Instant::now();
            launch_a().unwrap();
            stream.synchronize().unwrap();
            samples_a_us.push(start_a.elapsed().as_secs_f64() * 1_000_000.0);

            let start_b = Instant::now();
            launch_b().unwrap();
            stream.synchronize().unwrap();
            samples_b_us.push(start_b.elapsed().as_secs_f64() * 1_000_000.0);
        }
        (samples_a_us, samples_b_us)
    }

    fn compute_profile_stats(samples_us: &[f64]) -> ProfileStats {
        assert!(!samples_us.is_empty());
        let mut sorted = samples_us.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = samples_us.iter().sum::<f64>() / samples_us.len() as f64;
        let variance = samples_us
            .iter()
            .map(|sample| {
                let delta = sample - mean;
                delta * delta
            })
            .sum::<f64>()
            / samples_us.len() as f64;
        let stddev = variance.sqrt();
        let cv_pct = if mean > 0.0 {
            (stddev / mean) * 100.0
        } else {
            0.0
        };

        ProfileStats {
            mean,
            median: percentile_from_sorted(&sorted, 0.50),
            p90: percentile_from_sorted(&sorted, 0.90),
            p95: percentile_from_sorted(&sorted, 0.95),
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            stddev,
            cv_pct,
        }
    }

    fn print_chain_profile(
        rows: usize,
        warmup_iters: usize,
        measure_iters: usize,
        stats: ProfileStats,
    ) {
        println!(
            "profile_h2m_chain rows={} log_rows={} warmup={} iters={} mean_us={:.3} median_us={:.3} p90_us={:.3} p95_us={:.3} min_us={:.3} max_us={:.3} stddev_us={:.3} cv_pct={:.2}",
            rows,
            rows.trailing_zeros(),
            warmup_iters,
            measure_iters,
            stats.mean,
            stats.median,
            stats.p90,
            stats.p95,
            stats.min,
            stats.max,
            stats.stddev,
            stats.cv_pct,
        );
    }

    fn print_stage_profile(
        rows: usize,
        stage: &str,
        warmup_iters: usize,
        measure_iters: usize,
        stats: ProfileStats,
    ) {
        println!(
            "profile_h2m_stage rows={} log_rows={} stage={} warmup={} iters={} mean_us={:.3} median_us={:.3} p90_us={:.3} p95_us={:.3} min_us={:.3} max_us={:.3} stddev_us={:.3} cv_pct={:.2}",
            rows,
            rows.trailing_zeros(),
            stage,
            warmup_iters,
            measure_iters,
            stats.mean,
            stats.median,
            stats.p90,
            stats.p95,
            stats.min,
            stats.max,
            stats.stddev,
            stats.cv_pct,
        );
    }

    fn run_profile_out_of_place_multi_invocation(
        rows: usize,
        warmup_iters: usize,
        measure_iters: usize,
    ) {
        let mut rng = StdRng::seed_from_u64(0xA1B2_C3D4_55AA_77EEu64 ^ rows as u64);
        let h_input = (0..rows)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect::<Vec<_>>();

        let stream = CudaStream::default();
        let ring_len = stage1_ring_len(rows);
        let d_src_ring = alloc_filled_ring(rows, ring_len, &h_input, &stream);
        let mut d_dst_ring = alloc_empty_ring(rows, ring_len);
        let mut ring_idx = 0usize;

        let samples_us = run_profile_invocations(warmup_iters, measure_iters, &stream, || {
            let idx = ring_idx;
            ring_idx += 1;
            if ring_idx == ring_len {
                ring_idx = 0;
            }
            hypercube_evals_into_coeffs_bitrev_bf(&d_src_ring[idx], &mut d_dst_ring[idx], &stream)
        });
        let stats = compute_profile_stats(&samples_us);
        print_chain_profile(rows, warmup_iters, measure_iters, stats);
    }

    fn run_profile_in_place_multi_invocation(
        rows: usize,
        warmup_iters: usize,
        measure_iters: usize,
    ) {
        let mut rng = StdRng::seed_from_u64(0x6EAF_2D8C_A5B1_1173u64 ^ rows as u64);
        let h_input = (0..rows)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect::<Vec<_>>();

        let stream = CudaStream::default();
        let ring_len = stage1_ring_len(rows);
        let mut d_values_ring = alloc_filled_ring(rows, ring_len, &h_input, &stream);
        let mut ring_idx = 0usize;

        let samples_us = run_profile_invocations(warmup_iters, measure_iters, &stream, || {
            let idx = ring_idx;
            ring_idx += 1;
            if ring_idx == ring_len {
                ring_idx = 0;
            }
            hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut d_values_ring[idx], &stream)
        });
        let stats = compute_profile_stats(&samples_us);
        print_chain_profile(rows, warmup_iters, measure_iters, stats);
    }

    fn run_profile_out_of_place_stage_breakdown_multi_invocation(
        rows: usize,
        warmup_iters: usize,
        measure_iters: usize,
    ) {
        let log_rows = validate_len(rows);
        if log_rows == 24 {
            let mut rng = StdRng::seed_from_u64(0x52A9_64B7_14CF_73D8u64 ^ rows as u64);
            let h_input = (0..rows)
                .map(|_| BF::from_nonreduced_u32(rng.random()))
                .collect::<Vec<_>>();

            let stream = CudaStream::default();
            let ring_len = stage1_ring_len(rows);
            let d_src_ring = alloc_filled_ring(rows, ring_len, &h_input, &stream);
            let mut d_dst_ring = alloc_empty_ring(rows, ring_len);
            let mut d_stage_ring = alloc_empty_ring(rows, ring_len);
            let mut d_stage = DeviceAllocation::alloc(rows).unwrap();
            memory_copy_async(&mut d_stage, &h_input, &stream).unwrap();
            stream.synchronize().unwrap();

            let initial_grid = (rows >> LOG24_INITIAL_ROUNDS) as u32;
            let noninitial_grid = super::log24_noninitial_grid(rows);
            if LOG24_INITIAL_DYNAMIC_SMEM_BYTES > 0 {
                super::configure_log21_initial_dynamic_smem(ab_h2m_bitrev_bf_initial13_out_kernel)
                    .unwrap();
            }
            if LOG24_NONINITIAL_DYNAMIC_SMEM_BYTES > 0 {
                super::configure_log24_noninitial_dynamic_smem(
                    ab_h2m_bitrev_bf_noninitial11_stage3_out_start13_x4_kernel,
                )
                .unwrap();
            }

            let config_initial = CudaLaunchConfig::builder()
                .grid_dim(Dim3 {
                    x: initial_grid,
                    y: 1,
                    z: 1,
                })
                .block_dim(LOG24_INITIAL_BLOCK_THREADS)
                .dynamic_smem_bytes(LOG24_INITIAL_DYNAMIC_SMEM_BYTES)
                .stream(&stream)
                .build();
            let config_noninitial = CudaLaunchConfig::builder()
                .grid_dim(Dim3 {
                    x: noninitial_grid,
                    y: 1,
                    z: 1,
                })
                .block_dim(LOG24_NONINITIAL_BLOCK_THREADS)
                .dynamic_smem_bytes(LOG24_NONINITIAL_DYNAMIC_SMEM_BYTES)
                .stream(&stream)
                .build();

            let initial_fn = HypercubeBitrevInitialFunction(ab_h2m_bitrev_bf_initial13_out_kernel);
            let noninitial_fn =
                HypercubeBitrevNonInitialFunction(ab_h2m_bitrev_bf_noninitial11_stage3_out_start13_x4_kernel);

            let stage_ptr = d_stage.as_mut_ptr();
            let initial_args_for_noninitial =
                HypercubeBitrevInitialArguments::new(d_src_ring[0].as_ptr(), stage_ptr);
            let noninitial_args = HypercubeBitrevNonInitialArguments::new(
                stage_ptr as *const BF,
                stage_ptr,
                LOG24_NONINITIAL_STAGE2_START,
            );

            let mut chain_ring_idx = 0usize;
            let chain_samples = run_profile_invocations(warmup_iters, measure_iters, &stream, || {
                let idx = chain_ring_idx;
                chain_ring_idx += 1;
                if chain_ring_idx == ring_len {
                    chain_ring_idx = 0;
                }
                hypercube_evals_into_coeffs_bitrev_bf(
                    &d_src_ring[idx],
                    &mut d_dst_ring[idx],
                    &stream,
                )
            });
            let chain_stats = compute_profile_stats(&chain_samples);
            print_chain_profile(rows, warmup_iters, measure_iters, chain_stats);

            let mut initial_ring_idx = 0usize;
            let initial_samples = run_profile_invocations(warmup_iters, measure_iters, &stream, || {
                let idx = initial_ring_idx;
                initial_ring_idx += 1;
                if initial_ring_idx == ring_len {
                    initial_ring_idx = 0;
                }
                let stage_ptr = d_stage_ring[idx].as_mut_ptr();
                let initial_args =
                    HypercubeBitrevInitialArguments::new(d_src_ring[idx].as_ptr(), stage_ptr);
                initial_fn.launch(&config_initial, &initial_args)
            });
            let initial_stats = compute_profile_stats(&initial_samples);
            print_stage_profile(rows, "initial", warmup_iters, measure_iters, initial_stats);

            memory_copy_async(&mut d_stage, &h_input, &stream).unwrap();
            initial_fn
                .launch(&config_initial, &initial_args_for_noninitial)
                .unwrap();
            stream.synchronize().unwrap();
            let noninitial_samples =
                run_profile_invocations(warmup_iters, measure_iters, &stream, || {
                    noninitial_fn.launch(&config_noninitial, &noninitial_args)
                });
            let noninitial_stats = compute_profile_stats(&noninitial_samples);
            print_stage_profile(
                rows,
                "stage2final",
                warmup_iters,
                measure_iters,
                noninitial_stats,
            );

            let stage_sum_median = initial_stats.median + noninitial_stats.median;
            let median_gap = chain_stats.median - stage_sum_median;
            let chain_median = chain_stats.median;
            let initial_share_pct = (initial_stats.median / chain_median) * 100.0;
            let noninitial_share_pct = (noninitial_stats.median / chain_median) * 100.0;
            println!(
                "profile_h2m_stage_breakdown rows={} log_rows={} warmup={} iters={} chain_median_us={:.3} initial_median_us={:.3} stage2final_median_us={:.3} initial_pct={:.2} stage2final_pct={:.2} stage_sum_median_us={:.3} median_gap_us={:.3}",
                rows,
                log_rows,
                warmup_iters,
                measure_iters,
                chain_median,
                initial_stats.median,
                noninitial_stats.median,
                initial_share_pct,
                noninitial_share_pct,
                stage_sum_median,
                median_gap,
            );
            return;
        }

        if log_rows == 21 {
            let mut rng = StdRng::seed_from_u64(0x9C7F_D142_1B35_EAAAu64 ^ rows as u64);
            let h_input = (0..rows)
                .map(|_| BF::from_nonreduced_u32(rng.random()))
                .collect::<Vec<_>>();

            let stream = CudaStream::default();
            let ring_len = stage1_ring_len(rows);
            let d_src_ring = alloc_filled_ring(rows, ring_len, &h_input, &stream);
            let mut d_dst_ring = alloc_empty_ring(rows, ring_len);
            let mut d_stage_ring = alloc_empty_ring(rows, ring_len);
            let mut d_stage = DeviceAllocation::alloc(rows).unwrap();
            memory_copy_async(&mut d_stage, &h_input, &stream).unwrap();
            stream.synchronize().unwrap();

            let initial_grid = (rows >> LOG21_INITIAL_ROUNDS) as u32;
            let noninitial_grid = super::log21_noninitial_grid(rows);
            if LOG21_INITIAL_DYNAMIC_SMEM_BYTES > 0 {
                super::configure_log21_initial_dynamic_smem(ab_h2m_bitrev_bf_initial13_out_kernel)
                    .unwrap();
            }

            let config_initial = CudaLaunchConfig::builder()
                .grid_dim(Dim3 {
                    x: initial_grid,
                    y: 1,
                    z: 1,
                })
                .block_dim(LOG21_INITIAL_BLOCK_THREADS)
                .dynamic_smem_bytes(LOG21_INITIAL_DYNAMIC_SMEM_BYTES)
                .stream(&stream)
                .build();
            let config_noninitial = CudaLaunchConfig::basic(
                Dim3 {
                    x: noninitial_grid,
                    y: 1,
                    z: 1,
                },
                LOG21_NONINITIAL_BLOCK_THREADS,
                &stream,
            );

            let initial_fn = HypercubeBitrevInitialFunction(ab_h2m_bitrev_bf_initial13_out_kernel);
            let noninitial_fn =
                HypercubeBitrevNonInitialFunction(ab_h2m_bitrev_bf_noninitial8_stage3_out_start13_kernel);

            let stage_ptr = d_stage.as_mut_ptr();
            let initial_args_for_noninitial =
                HypercubeBitrevInitialArguments::new(d_src_ring[0].as_ptr(), stage_ptr);
            let noninitial_args = HypercubeBitrevNonInitialArguments::new(
                stage_ptr as *const BF,
                stage_ptr,
                LOG21_NONINITIAL_STAGE2_START,
            );

            let mut chain_ring_idx = 0usize;
            let chain_samples = run_profile_invocations(warmup_iters, measure_iters, &stream, || {
                let idx = chain_ring_idx;
                chain_ring_idx += 1;
                if chain_ring_idx == ring_len {
                    chain_ring_idx = 0;
                }
                hypercube_evals_into_coeffs_bitrev_bf(
                    &d_src_ring[idx],
                    &mut d_dst_ring[idx],
                    &stream,
                )
            });
            let chain_stats = compute_profile_stats(&chain_samples);
            print_chain_profile(rows, warmup_iters, measure_iters, chain_stats);

            let mut initial_ring_idx = 0usize;
            let initial_samples = run_profile_invocations(warmup_iters, measure_iters, &stream, || {
                let idx = initial_ring_idx;
                initial_ring_idx += 1;
                if initial_ring_idx == ring_len {
                    initial_ring_idx = 0;
                }
                let stage_ptr = d_stage_ring[idx].as_mut_ptr();
                let initial_args =
                    HypercubeBitrevInitialArguments::new(d_src_ring[idx].as_ptr(), stage_ptr);
                initial_fn.launch(&config_initial, &initial_args)
            });
            let initial_stats = compute_profile_stats(&initial_samples);
            print_stage_profile(rows, "initial", warmup_iters, measure_iters, initial_stats);

            memory_copy_async(&mut d_stage, &h_input, &stream).unwrap();
            initial_fn
                .launch(&config_initial, &initial_args_for_noninitial)
                .unwrap();
            stream.synchronize().unwrap();
            let noninitial_samples =
                run_profile_invocations(warmup_iters, measure_iters, &stream, || {
                    noninitial_fn.launch(&config_noninitial, &noninitial_args)
                });
            let noninitial_stats = compute_profile_stats(&noninitial_samples);
            print_stage_profile(
                rows,
                "stage2final",
                warmup_iters,
                measure_iters,
                noninitial_stats,
            );

            let stage_sum_median = initial_stats.median + noninitial_stats.median;
            let median_gap = chain_stats.median - stage_sum_median;
            let chain_median = chain_stats.median;
            let initial_share_pct = (initial_stats.median / chain_median) * 100.0;
            let noninitial_share_pct = (noninitial_stats.median / chain_median) * 100.0;
            println!(
                "profile_h2m_stage_breakdown rows={} log_rows={} warmup={} iters={} chain_median_us={:.3} initial_median_us={:.3} stage2final_median_us={:.3} initial_pct={:.2} stage2final_pct={:.2} stage_sum_median_us={:.3} median_gap_us={:.3}",
                rows,
                log_rows,
                warmup_iters,
                measure_iters,
                chain_median,
                initial_stats.median,
                noninitial_stats.median,
                initial_share_pct,
                noninitial_share_pct,
                stage_sum_median,
                median_gap,
            );
            return;
        }

        let plan = super::select_out_of_place_plan(log_rows);

        let mut rng = StdRng::seed_from_u64(0x9C7F_D142_1B35_EAAAu64 ^ rows as u64);
        let h_input = (0..rows)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect::<Vec<_>>();

        let stream = CudaStream::default();
        let ring_len = stage1_ring_len(rows);
        let d_src_ring = alloc_filled_ring(rows, ring_len, &h_input, &stream);
        let mut d_dst_ring = alloc_empty_ring(rows, ring_len);
        let mut d_stage_ring = alloc_empty_ring(rows, ring_len);
        let mut d_stage = DeviceAllocation::alloc(rows).unwrap();
        memory_copy_async(&mut d_stage, &h_input, &stream).unwrap();
        stream.synchronize().unwrap();

        let initial_grid = (rows >> plan.initial_rounds) as u32;
        let noninitial_stage2_grid = super::noninitial_grid(
            rows,
            plan.noninitial_stage2_rounds,
            plan.noninitial_stage2_tiles_per_cta,
        );
        let noninitial_stage3_grid = super::noninitial_grid(
            rows,
            plan.noninitial_stage3_rounds,
            plan.noninitial_stage3_tiles_per_cta,
        );
        let config_initial = CudaLaunchConfig::basic(
            Dim3 {
                x: initial_grid,
                y: 1,
                z: 1,
            },
            BLOCK_THREADS,
            &stream,
        );
        let config_stage2 = CudaLaunchConfig::basic(
            Dim3 {
                x: noninitial_stage2_grid,
                y: 1,
                z: 1,
            },
            BLOCK_THREADS,
            &stream,
        );
        let config_stage3 = CudaLaunchConfig::basic(
            Dim3 {
                x: noninitial_stage3_grid,
                y: 1,
                z: 1,
            },
            BLOCK_THREADS,
            &stream,
        );

        let initial_fn = HypercubeBitrevInitialFunction(plan.initial_kernel);
        let stage2_fn = HypercubeBitrevNonInitialFunction(plan.noninitial_stage2_kernel);
        let stage3_fn = HypercubeBitrevNonInitialFunction(plan.noninitial_stage3_kernel);

        let stage_ptr = d_stage.as_mut_ptr();
        let initial_args_for_noninitial =
            HypercubeBitrevInitialArguments::new(d_src_ring[0].as_ptr(), stage_ptr);
        let stage2_args = HypercubeBitrevNonInitialArguments::new(
            stage_ptr as *const BF,
            stage_ptr,
            plan.noninitial_stage2_start,
        );
        let stage3_args = HypercubeBitrevNonInitialArguments::new(
            stage_ptr as *const BF,
            stage_ptr,
            plan.noninitial_stage3_start,
        );

        let mut chain_ring_idx = 0usize;
        let chain_samples = run_profile_invocations(warmup_iters, measure_iters, &stream, || {
            let idx = chain_ring_idx;
            chain_ring_idx += 1;
            if chain_ring_idx == ring_len {
                chain_ring_idx = 0;
            }
            hypercube_evals_into_coeffs_bitrev_bf(&d_src_ring[idx], &mut d_dst_ring[idx], &stream)
        });
        let chain_stats = compute_profile_stats(&chain_samples);
        print_chain_profile(rows, warmup_iters, measure_iters, chain_stats);

        let mut initial_ring_idx = 0usize;
        let initial_samples = run_profile_invocations(warmup_iters, measure_iters, &stream, || {
            let idx = initial_ring_idx;
            initial_ring_idx += 1;
            if initial_ring_idx == ring_len {
                initial_ring_idx = 0;
            }
            let stage_ptr = d_stage_ring[idx].as_mut_ptr();
            let initial_args = HypercubeBitrevInitialArguments::new(d_src_ring[idx].as_ptr(), stage_ptr);
            initial_fn.launch(&config_initial, &initial_args)
        });
        let initial_stats = compute_profile_stats(&initial_samples);
        print_stage_profile(rows, "initial", warmup_iters, measure_iters, initial_stats);

        memory_copy_async(&mut d_stage, &h_input, &stream).unwrap();
        initial_fn
            .launch(&config_initial, &initial_args_for_noninitial)
            .unwrap();
        stream.synchronize().unwrap();
        let stage2_samples = run_profile_invocations(warmup_iters, measure_iters, &stream, || {
            stage2_fn.launch(&config_stage2, &stage2_args)
        });
        let stage2_stats = compute_profile_stats(&stage2_samples);
        print_stage_profile(rows, "stage2", warmup_iters, measure_iters, stage2_stats);

        memory_copy_async(&mut d_stage, &h_input, &stream).unwrap();
        initial_fn
            .launch(&config_initial, &initial_args_for_noninitial)
            .unwrap();
        stage2_fn.launch(&config_stage2, &stage2_args).unwrap();
        stream.synchronize().unwrap();
        let stage3_samples = run_profile_invocations(warmup_iters, measure_iters, &stream, || {
            stage3_fn.launch(&config_stage3, &stage3_args)
        });
        let stage3_stats = compute_profile_stats(&stage3_samples);
        print_stage_profile(rows, "stage3", warmup_iters, measure_iters, stage3_stats);

        let stage_sum_median = initial_stats.median + stage2_stats.median + stage3_stats.median;
        let median_gap = chain_stats.median - stage_sum_median;
        let chain_median = chain_stats.median;
        let initial_share_pct = (initial_stats.median / chain_median) * 100.0;
        let stage2_share_pct = (stage2_stats.median / chain_median) * 100.0;
        let stage3_share_pct = (stage3_stats.median / chain_median) * 100.0;
        println!(
            "profile_h2m_stage_breakdown rows={} log_rows={} warmup={} iters={} chain_median_us={:.3} initial_median_us={:.3} stage2_median_us={:.3} stage3_median_us={:.3} initial_pct={:.2} stage2_pct={:.2} stage3_pct={:.2} stage_sum_median_us={:.3} median_gap_us={:.3}",
            rows,
            log_rows,
            warmup_iters,
            measure_iters,
            chain_median,
            initial_stats.median,
            stage2_stats.median,
            stage3_stats.median,
            initial_share_pct,
            stage2_share_pct,
            stage3_share_pct,
            stage_sum_median,
            median_gap,
        );
    }

    #[test]
    fn hypercube_bitrev_bf_out_of_place_log24() {
        run_out_of_place_case(LOG24_ROWS);
    }

    #[test]
    fn hypercube_bitrev_bf_in_place_log24() {
        run_in_place_case(LOG24_ROWS);
    }

    #[test]
    fn hypercube_bitrev_bf_out_of_place_log23() {
        run_out_of_place_case(LOG23_ROWS);
    }

    #[test]
    fn hypercube_bitrev_bf_in_place_log23() {
        run_in_place_case(LOG23_ROWS);
    }

    #[test]
    fn hypercube_bitrev_bf_out_of_place_log22() {
        run_out_of_place_case(LOG22_ROWS);
    }

    #[test]
    fn hypercube_bitrev_bf_in_place_log22() {
        run_in_place_case(LOG22_ROWS);
    }

    #[test]
    fn hypercube_bitrev_bf_out_of_place_log21() {
        run_out_of_place_case(LOG21_ROWS);
    }

    #[test]
    fn hypercube_bitrev_bf_in_place_log21() {
        run_in_place_case(LOG21_ROWS);
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_single_invocation_log24() {
        // Profiling-only entrypoint: launches one kernel chain without correctness checks.
        let rows = LOG24_ROWS;
        let mut rng = rng();
        let h_input = (0..rows)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect::<Vec<_>>();

        let stream = CudaStream::default();
        let mut d_src = DeviceAllocation::alloc(rows).unwrap();
        let mut d_dst = DeviceAllocation::alloc(rows).unwrap();
        memory_copy_async(&mut d_src, &h_input, &stream).unwrap();

        hypercube_evals_into_coeffs_bitrev_bf(&d_src, &mut d_dst, &stream).unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_single_invocation_log23() {
        // Profiling-only entrypoint: launches one kernel chain without correctness checks.
        let rows = LOG23_ROWS;
        let mut rng = rng();
        let h_input = (0..rows)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect::<Vec<_>>();

        let stream = CudaStream::default();
        let mut d_src = DeviceAllocation::alloc(rows).unwrap();
        let mut d_dst = DeviceAllocation::alloc(rows).unwrap();
        memory_copy_async(&mut d_src, &h_input, &stream).unwrap();

        hypercube_evals_into_coeffs_bitrev_bf(&d_src, &mut d_dst, &stream).unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_single_invocation_log22() {
        // Profiling-only entrypoint: launches one kernel chain without correctness checks.
        let rows = LOG22_ROWS;
        let mut rng = rng();
        let h_input = (0..rows)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect::<Vec<_>>();

        let stream = CudaStream::default();
        let mut d_src = DeviceAllocation::alloc(rows).unwrap();
        let mut d_dst = DeviceAllocation::alloc(rows).unwrap();
        memory_copy_async(&mut d_src, &h_input, &stream).unwrap();

        hypercube_evals_into_coeffs_bitrev_bf(&d_src, &mut d_dst, &stream).unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_single_invocation_log21() {
        // Profiling-only entrypoint: launches one kernel chain without correctness checks.
        let rows = LOG21_ROWS;
        let mut rng = rng();
        let h_input = (0..rows)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect::<Vec<_>>();

        let stream = CudaStream::default();
        let mut d_src = DeviceAllocation::alloc(rows).unwrap();
        let mut d_dst = DeviceAllocation::alloc(rows).unwrap();
        memory_copy_async(&mut d_src, &h_input, &stream).unwrap();

        hypercube_evals_into_coeffs_bitrev_bf(&d_src, &mut d_dst, &stream).unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_multi_invocation_log24() {
        run_profile_out_of_place_multi_invocation(
            LOG24_ROWS,
            PROFILE_MULTI_WARMUP_ITERS,
            PROFILE_MULTI_MEASURE_ITERS,
        );
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_multi_invocation_log23() {
        run_profile_out_of_place_multi_invocation(
            LOG23_ROWS,
            PROFILE_MULTI_WARMUP_ITERS,
            PROFILE_MULTI_MEASURE_ITERS,
        );
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_multi_invocation_log22() {
        run_profile_out_of_place_multi_invocation(
            LOG22_ROWS,
            PROFILE_MULTI_WARMUP_ITERS,
            PROFILE_MULTI_MEASURE_ITERS,
        );
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_multi_invocation_log21() {
        run_profile_out_of_place_multi_invocation(
            LOG21_ROWS,
            PROFILE_MULTI_WARMUP_ITERS,
            PROFILE_MULTI_MEASURE_ITERS,
        );
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_multi_invocation_in_place_log24() {
        run_profile_in_place_multi_invocation(
            LOG24_ROWS,
            PROFILE_MULTI_WARMUP_ITERS,
            PROFILE_MULTI_MEASURE_ITERS,
        );
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_multi_invocation_in_place_log23() {
        run_profile_in_place_multi_invocation(
            LOG23_ROWS,
            PROFILE_MULTI_WARMUP_ITERS,
            PROFILE_MULTI_MEASURE_ITERS,
        );
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_multi_invocation_in_place_log22() {
        run_profile_in_place_multi_invocation(
            LOG22_ROWS,
            PROFILE_MULTI_WARMUP_ITERS,
            PROFILE_MULTI_MEASURE_ITERS,
        );
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_multi_invocation_in_place_log21() {
        run_profile_in_place_multi_invocation(
            LOG21_ROWS,
            PROFILE_MULTI_WARMUP_ITERS,
            PROFILE_MULTI_MEASURE_ITERS,
        );
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_stage_breakdown_multi_invocation_log24() {
        run_profile_out_of_place_stage_breakdown_multi_invocation(
            LOG24_ROWS,
            PROFILE_STAGE_WARMUP_ITERS,
            PROFILE_STAGE_MEASURE_ITERS,
        );
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_stage_breakdown_multi_invocation_log23() {
        run_profile_out_of_place_stage_breakdown_multi_invocation(
            LOG23_ROWS,
            PROFILE_STAGE_WARMUP_ITERS,
            PROFILE_STAGE_MEASURE_ITERS,
        );
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_stage_breakdown_multi_invocation_log22() {
        run_profile_out_of_place_stage_breakdown_multi_invocation(
            LOG22_ROWS,
            PROFILE_STAGE_WARMUP_ITERS,
            PROFILE_STAGE_MEASURE_ITERS,
        );
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_stage_breakdown_multi_invocation_log21() {
        run_profile_out_of_place_stage_breakdown_multi_invocation(
            LOG21_ROWS,
            PROFILE_STAGE_WARMUP_ITERS,
            PROFILE_STAGE_MEASURE_ITERS,
        );
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_compare_2pass_vs_3pass_multi_invocation_log24() {
        const LEGACY_LOG24_INITIAL_ROUNDS: u32 = 12;
        const LEGACY_LOG24_STAGE2_ROUNDS: u32 = 6;
        const LEGACY_LOG24_STAGE3_ROUNDS: u32 = 6;
        const LEGACY_LOG24_STAGE2_START: u32 = LEGACY_LOG24_INITIAL_ROUNDS;
        const LEGACY_LOG24_STAGE3_START: u32 =
            LEGACY_LOG24_STAGE2_START + LEGACY_LOG24_STAGE2_ROUNDS;

        let rows = LOG24_ROWS;
        let warmup_iters = PROFILE_MULTI_WARMUP_ITERS;
        let measure_iters = PROFILE_MULTI_MEASURE_ITERS;

        let mut rng = StdRng::seed_from_u64(0x6A31_92C4_77B0_5E28u64 ^ rows as u64);
        let h_input = (0..rows)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect::<Vec<_>>();

        let stream = CudaStream::default();
        let ring_len = stage1_ring_len(rows);
        let d_src_ring = alloc_filled_ring(rows, ring_len, &h_input, &stream);
        let mut d_dst_2pass_ring = alloc_empty_ring(rows, ring_len);
        let mut d_dst_3pass_ring = alloc_empty_ring(rows, ring_len);

        if LOG24_INITIAL_DYNAMIC_SMEM_BYTES > 0 {
            super::configure_log21_initial_dynamic_smem(ab_h2m_bitrev_bf_initial13_out_kernel)
                .unwrap();
        }
        if LOG24_NONINITIAL_DYNAMIC_SMEM_BYTES > 0 {
            super::configure_log24_noninitial_dynamic_smem(
                ab_h2m_bitrev_bf_noninitial11_stage3_out_start13_x4_kernel,
            )
            .unwrap();
        }

        let grid_2pass_initial = (rows >> LOG24_INITIAL_ROUNDS) as u32;
        let grid_2pass_noninitial = super::log24_noninitial_grid(rows);
        let config_2pass_initial = CudaLaunchConfig::builder()
            .grid_dim(Dim3 {
                x: grid_2pass_initial,
                y: 1,
                z: 1,
            })
            .block_dim(LOG24_INITIAL_BLOCK_THREADS)
            .dynamic_smem_bytes(LOG24_INITIAL_DYNAMIC_SMEM_BYTES)
            .stream(&stream)
            .build();
        let config_2pass_noninitial = CudaLaunchConfig::builder()
            .grid_dim(Dim3 {
                x: grid_2pass_noninitial,
                y: 1,
                z: 1,
            })
            .block_dim(LOG24_NONINITIAL_BLOCK_THREADS)
            .dynamic_smem_bytes(LOG24_NONINITIAL_DYNAMIC_SMEM_BYTES)
            .stream(&stream)
            .build();
        let initial_2pass_fn = HypercubeBitrevInitialFunction(ab_h2m_bitrev_bf_initial13_out_kernel);
        let noninitial_2pass_fn =
            HypercubeBitrevNonInitialFunction(ab_h2m_bitrev_bf_noninitial11_stage3_out_start13_x4_kernel);

        let grid_3pass_initial = (rows >> LEGACY_LOG24_INITIAL_ROUNDS) as u32;
        let grid_3pass_stage2 = super::noninitial_grid(rows, LEGACY_LOG24_STAGE2_ROUNDS, 1);
        let grid_3pass_stage3 = super::noninitial_grid(rows, LEGACY_LOG24_STAGE3_ROUNDS, 1);
        let config_3pass_initial = CudaLaunchConfig::basic(
            Dim3 {
                x: grid_3pass_initial,
                y: 1,
                z: 1,
            },
            BLOCK_THREADS,
            &stream,
        );
        let config_3pass_stage2 = CudaLaunchConfig::basic(
            Dim3 {
                x: grid_3pass_stage2,
                y: 1,
                z: 1,
            },
            BLOCK_THREADS,
            &stream,
        );
        let config_3pass_stage3 = CudaLaunchConfig::basic(
            Dim3 {
                x: grid_3pass_stage3,
                y: 1,
                z: 1,
            },
            BLOCK_THREADS,
            &stream,
        );
        let initial_3pass_fn = HypercubeBitrevInitialFunction(ab_h2m_bitrev_bf_initial12_out_kernel);
        let stage2_3pass_fn =
            HypercubeBitrevNonInitialFunction(ab_h2m_bitrev_bf_noninitial6_stage2_out_start12_kernel);
        let stage3_3pass_fn =
            HypercubeBitrevNonInitialFunction(ab_h2m_bitrev_bf_noninitial6_stage3_out_start18_kernel);

        let mut ring_idx_2pass = 0usize;
        let mut ring_idx_3pass = 0usize;
        let (samples_2pass_us, samples_3pass_us) = run_profile_compare_invocations(
            warmup_iters,
            measure_iters,
            &stream,
            || {
                let idx = ring_idx_2pass;
                ring_idx_2pass += 1;
                if ring_idx_2pass == ring_len {
                    ring_idx_2pass = 0;
                }

                let dst_ptr = d_dst_2pass_ring[idx].as_mut_ptr();
                let args0 = HypercubeBitrevInitialArguments::new(d_src_ring[idx].as_ptr(), dst_ptr);
                initial_2pass_fn.launch(&config_2pass_initial, &args0)?;

                let args1 = HypercubeBitrevNonInitialArguments::new(
                    dst_ptr as *const BF,
                    dst_ptr,
                    LOG24_NONINITIAL_STAGE2_START,
                );
                noninitial_2pass_fn.launch(&config_2pass_noninitial, &args1)
            },
            || {
                let idx = ring_idx_3pass;
                ring_idx_3pass += 1;
                if ring_idx_3pass == ring_len {
                    ring_idx_3pass = 0;
                }

                let dst_ptr = d_dst_3pass_ring[idx].as_mut_ptr();
                let args0 = HypercubeBitrevInitialArguments::new(d_src_ring[idx].as_ptr(), dst_ptr);
                initial_3pass_fn.launch(&config_3pass_initial, &args0)?;

                let args1 = HypercubeBitrevNonInitialArguments::new(
                    dst_ptr as *const BF,
                    dst_ptr,
                    LEGACY_LOG24_STAGE2_START,
                );
                stage2_3pass_fn.launch(&config_3pass_stage2, &args1)?;

                let args2 = HypercubeBitrevNonInitialArguments::new(
                    dst_ptr as *const BF,
                    dst_ptr,
                    LEGACY_LOG24_STAGE3_START,
                );
                stage3_3pass_fn.launch(&config_3pass_stage3, &args2)
            },
        );

        let stats_2pass = compute_profile_stats(&samples_2pass_us);
        let stats_3pass = compute_profile_stats(&samples_3pass_us);
        let delta_us = stats_3pass.median - stats_2pass.median;
        let delta_pct = if stats_3pass.median > 0.0 {
            (delta_us / stats_3pass.median) * 100.0
        } else {
            0.0
        };
        let winner = if stats_2pass.median <= stats_3pass.median {
            "2pass_13_11"
        } else {
            "3pass_12_6_6"
        };

        println!(
            "profile_h2m_compare rows={} log_rows={} variant=2pass_13_11 warmup={} iters={} mean_us={:.3} median_us={:.3} p90_us={:.3} p95_us={:.3} min_us={:.3} max_us={:.3} stddev_us={:.3} cv_pct={:.2}",
            rows,
            rows.trailing_zeros(),
            warmup_iters,
            measure_iters,
            stats_2pass.mean,
            stats_2pass.median,
            stats_2pass.p90,
            stats_2pass.p95,
            stats_2pass.min,
            stats_2pass.max,
            stats_2pass.stddev,
            stats_2pass.cv_pct,
        );
        println!(
            "profile_h2m_compare rows={} log_rows={} variant=3pass_12_6_6 warmup={} iters={} mean_us={:.3} median_us={:.3} p90_us={:.3} p95_us={:.3} min_us={:.3} max_us={:.3} stddev_us={:.3} cv_pct={:.2}",
            rows,
            rows.trailing_zeros(),
            warmup_iters,
            measure_iters,
            stats_3pass.mean,
            stats_3pass.median,
            stats_3pass.p90,
            stats_3pass.p95,
            stats_3pass.min,
            stats_3pass.max,
            stats_3pass.stddev,
            stats_3pass.cv_pct,
        );
        println!(
            "profile_h2m_compare_summary rows={} log_rows={} baseline=3pass_12_6_6 candidate=2pass_13_11 baseline_median_us={:.3} candidate_median_us={:.3} delta_us={:.3} delta_pct={:.3} winner={}",
            rows,
            rows.trailing_zeros(),
            stats_3pass.median,
            stats_2pass.median,
            delta_us,
            delta_pct,
            winner,
        );
    }

    #[test]
    #[should_panic]
    fn unsupported_log_rows_panics() {
        let len = 1usize << 20;
        let stream = CudaStream::default();
        let mut d_values = DeviceAllocation::alloc(len).unwrap();
        hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut d_values, &stream).unwrap();
    }

    #[test]
    #[should_panic]
    fn non_power_of_two_panics() {
        let len = (1usize << MAX_SUPPORTED_LOG_ROWS) - 1;
        let stream = CudaStream::default();
        let mut d_values = DeviceAllocation::alloc(len).unwrap();
        hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut d_values, &stream).unwrap();
    }
}
