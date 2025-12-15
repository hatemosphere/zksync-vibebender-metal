// Possible optimisations
//  - survey by Thaler about sumcheck techniques https://eprint.iacr.org/2025/2041.pdf
//  - sumcheck for small fields https://people.cs.georgetown.edu/jthaler/small-sumcheck.pdf
//  - https://eprint.iacr.org/2024/1210.pdf
//  - batched sumcheck https://hackmd.io/s/HyxaupAAA
//  - eq-poly and small value sumchecks https://eprint.iacr.org/2025/1117.pdf
//  - prefix-suffix sumcheck https://eprint.iacr.org/2025/611.pdf

mod batched_sumcheck;
mod multilinear_sumcheck;
mod sumcheck_prover;
mod sumcheck_verifier;

pub use sumcheck_prover::SumcheckInstanceProver;
pub use sumcheck_verifier::SumcheckInstanceVerifier;
pub use batched_sumcheck::{prove as batch_prove, verify as batch_verify, SumcheckInstanceProof};
