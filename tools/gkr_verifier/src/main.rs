#![no_std]
#![allow(incomplete_features)]
#![feature(allocator_api)]
#![feature(generic_const_exprs)]
#![no_main]

use field::baby_bear::base::BabyBearField;
use field::baby_bear::ext4::BabyBearExt4;
use non_determinism_source::CSRBasedSource;
use riscv_common::zksync_os_finish_success;
use verifier_common::gkr::{verify_gkr_sumcheck, GKRVerificationError};

#[path = "generated/gkr_layout.rs"]
mod generated_gkr;

#[no_mangle]
extern "C" fn eh_personality() {}

#[link_section = ".init.rust"]
#[export_name = "_start_rust"]
unsafe extern "C" fn start_rust() -> ! {
    main()
}

unsafe fn workload() -> ! {
    let config = &generated_gkr::GKR_VERIFIER_CONFIG;
    use generated_gkr::*;
    let result = verify_gkr_sumcheck::<
        BabyBearField,
        BabyBearExt4,
        CSRBasedSource,
        GKR_ROUNDS,
        GKR_ADDRS,
        GKR_EVALS,
        GKR_TRANSCRIPT_U32,
        GKR_MAX_POW,
        GKR_EVAL_BUF,
    >(config);

    match result {
        Ok(_output) => {
            zksync_os_finish_success(&[1, 0, 0, 0, 0, 0, 0, 0]);
        }
        Err(GKRVerificationError::SumcheckRoundFailed { layer, round }) => {
            zksync_os_finish_success(&[0xDEAD, 1, layer as u32, round as u32, 0, 0, 0, 0]);
        }
        Err(GKRVerificationError::FinalStepCheckFailed { layer }) => {
            zksync_os_finish_success(&[0xDEAD, 2, layer as u32, 0, 0, 0, 0, 0]);
        }
    }
}

#[inline(never)]
fn main() -> ! {
    riscv_common::boot_sequence::init();
    unsafe { workload() }
}
