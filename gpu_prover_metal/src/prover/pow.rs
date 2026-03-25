//! Proof-of-work computation for Metal.

use super::context::ProverContext;
use crate::blake2s::blake2s_pow;
use crate::metal_runtime::MetalResult;
use prover::transcript::{Blake2sTranscript, Seed};

pub(crate) struct PowOutput {
    pub nonce: u64,
}

impl PowOutput {
    pub fn new(
        seed: &mut Seed,
        pow_bits: u32,
        external_nonce: Option<u64>,
        context: &ProverContext,
    ) -> MetalResult<Self> {
        if let Some(nonce) = external_nonce {
            Blake2sTranscript::verify_pow(seed, nonce, pow_bits);
            return Ok(Self { nonce });
        }

        // Dispatch PoW kernel on GPU in bounded chunks.
        let device = context.device();
        let d_seed = context.alloc_from_slice(&seed.0)?;
        let d_nonce_lo = context.alloc_from_slice(&[0xFFFFFFFFu32])?;
        let d_nonce_hi = context.alloc_from_slice(&[0xFFFFFFFFu32])?;

        let grid_size: u64 = 256 * 128; // 32768 threads
        // Each thread checks ~512 nonces per dispatch via inner loop.
        // blake2s.metal compiled at -O0 to avoid Metal compiler optimization bug.
        let nonces_per_chunk: u64 = grid_size * 512;
        let mut chunk_start: u64 = 0;
        let mut chunks_done: u64 = 0;
        let t = std::time::Instant::now();

        log::info!("PoW: searching for {pow_bits}-bit solution (GPU, chunk={:.0}M)",
            nonces_per_chunk as f64 / 1e6);

        loop {
            let chunk_end = chunk_start.saturating_add(nonces_per_chunk).min(u64::MAX);
            let cmd_buf = context.new_command_buffer()?;
            blake2s_pow(device, &cmd_buf, &d_seed, pow_bits, chunk_start, chunk_end, &d_nonce_lo, &d_nonce_hi)?;
            cmd_buf.commit_and_wait();
            chunks_done += 1;

            let nonce_lo_val = unsafe { *d_nonce_lo.as_ptr() };
            if nonce_lo_val != 0xFFFFFFFF {
                break;
            }

            if chunks_done % 100 == 0 {
                let searched = chunk_end as f64 / 1e6;
                let elapsed = t.elapsed().as_secs_f64();
                log::info!("PoW: {searched:.0}M nonces in {elapsed:.1}s, {chunks_done} chunks");
            }

            chunk_start = chunk_end;
            if chunk_start >= u64::MAX {
                panic!("PoW: exhausted nonce space");
            }
        }

        let nonce_lo = unsafe { *d_nonce_lo.as_ptr() } as u64;
        let nonce_hi = unsafe { *d_nonce_hi.as_ptr() } as u64;
        let nonce = nonce_lo | (nonce_hi << 32);
        log::info!("PoW: found nonce {} in {:.1}s ({chunks_done} chunks)", nonce, t.elapsed().as_secs_f64());

        Blake2sTranscript::verify_pow(seed, nonce, pow_bits);

        Ok(Self { nonce })
    }
}
