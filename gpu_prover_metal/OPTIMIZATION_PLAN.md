# Metal GPU Prover Optimization Plan

## Current State
- Full E2E proof generation and verification working
- ~61s for hashed_fibonacci (vs ~250s CPU = **4x speedup**)
- All CPU compute fallbacks eliminated except minor ones

## Phase 1: Correctness — Byte-identical proofs (Priority: HIGH)

### 1.1 Mersenne31 Canonicalization
**Problem:** GPU proofs are mathematically correct but not byte-identical to CPU proofs. Root cause: Metal field arithmetic can produce `p` (= 2^31-1) instead of `0` for zero values. This propagates through blake2s hashing into the transcript, causing different random challenges and divergent (but valid) proofs.

**Fix:** Add canonicalization (`val >= p ? 0 : val`) at buffer boundaries:
- Before Merkle tree leaf hashing (in `blake2s_leaves_kernel`)
- Before transcript draws
- In proof serialization (`evaluations_at_random_points`)

**Effort:** 2-4 hours. **Impact:** Enables deterministic testing, CI comparison with CPU proofs.

### 1.2 Debug Logging Cleanup
Strip remaining `log::info!` hash/debug output from hot paths (stage_2.rs, stage_3.rs, queries.rs). These add CPU sync points that hurt performance.

**Effort:** 1 hour. **Impact:** ~2-5% proving time reduction from fewer GPU sync points.

## Phase 2: NTT Optimization (Priority: HIGH — NTT is ~40% of proving time)

### 2.1 Fused Multi-Stage NTT (from Lambdaworks)
**Reference:** `reference/lambdaworks/crates/math/src/gpu/metal/shaders/fft/cfft.h.metal`

Lambdaworks fuses 4 consecutive butterfly stages into a single kernel dispatch using threadgroup memory. Our NTT already fuses 7 stages for small sizes but uses separate kernels for higher stages.

**Approach:**
- Implement a `FUSED_STAGES=4` variant for the multi-column NTT extension steps
- Each threadgroup processes a 16-element block entirely in shared memory
- Eliminates 3 global memory round-trips per fused batch

**Effort:** 4-8 hours. **Impact:** 15-25% NTT speedup.

### 2.2 Threadgroup-Cached Twiddles
**Reference:** Lambdaworks `cfft_butterfly_tg` kernel

For butterfly stages where twiddle count fits in threadgroup memory (~32KB on Apple Silicon), cooperatively load twiddles into shared memory before the butterfly pass. Our current NTT loads twiddles from device memory per-thread.

**Effort:** 2-4 hours. **Impact:** 5-10% NTT speedup.

### 2.3 COL_PAIRS_PER_BLOCK Recovery
**History:** COL_PAIRS_PER_BLOCK was reduced from 4→1 to fix a shared memory corruption bug. This 4x'd the number of kernel dispatches for multi-column NTT.

**Approach:** Root-cause the shared memory conflict (likely twiddle cache vs data transpose aliasing), partition threadgroup memory properly, and restore COL_PAIRS_PER_BLOCK=4.

**Effort:** 4-8 hours. **Impact:** Up to 2-4x fewer NTT dispatches.

## Phase 3: Reduce/Scan with Simdgroup Operations (Priority: MEDIUM)

### 3.1 Three-Level Reduction Hierarchy
**Current:** Tree reduction in threadgroup memory with barriers at every step.
**Proposed:**
1. `simd_sum()` / `simd_product()` within each SIMD group (32 lanes → 1 value, no barrier needed)
2. Threadgroup reduce across SIMD groups (8 values → 1, minimal barriers)
3. Cross-threadgroup reduce (already handled by pass2)

This leverages Apple Silicon's native SIMD group hardware — `simd_sum` is ~10x faster than manual shuffle+barrier reduction.

**Reference:** Metal Shading Language spec §6.9 "SIMD-group Functions"

**Effort:** 4-6 hours. **Impact:** 20-30% speedup for reduce and scan operations.

### 3.2 Fully-GPU Prefix Scan for Large Inputs
**Current:** GPU scan with CPU fallback for block totals >256.
**Proposed:** Recursive GPU scan (scan block totals with another GPU scan dispatch).

**Effort:** 2 hours. **Impact:** Eliminates CPU sync point in prefix scan path.

## Phase 4: Kernel Micro-Optimizations (Priority: MEDIUM)

### 4.1 Branchless Coefficient Application (Stage 3)
**Current:** `maybe_apply_coeff()` uses a 3-way switch on coeff_info byte.
**Proposed:** Branchless: `val = select(val, -val, coeff == MINUS_ONE); val = select(val, val * coeff, coeff > 1);`

**Effort:** 1 hour. **Impact:** 5-15% stage 3 kernel speedup (reduces warp divergence).

### 4.2 Increase Batch Inversion Size (Stage 4)
**Current:** `INV_BATCH=3` in deep quotient kernel.
**Proposed:** Test INV_BATCH=4,5,6. Apple Silicon has large register files.

**Effort:** 30 min (just change constant + test). **Impact:** 5-10% stage 4 speedup.

### 4.3 Scatter-Add Sentinel Elimination
**Current:** `ab_scatter_add_multiplicities_kernel` checks `if (abs_idx == 0xFFFFFFFF) return;`
**Proposed:** Pre-filter sentinels, or set last-row mapping to a dummy bin.

**Effort:** 1 hour. **Impact:** 5-10% multiplicities kernel speedup.

## Phase 5: Memory & Dispatch Optimization (Priority: LOW)

### 5.1 GPU Buffer Memset
**Current:** CPU `std::ptr::write_bytes()` to zero 100-500MB GPU buffers.
**Proposed:** Simple Metal `memset_zero_kernel`.

**Effort:** 1 hour. **Impact:** 2-5x faster buffer initialization (~10-50ms per proof).

### 5.2 Reduce Command Buffer Commits
**Current:** Multiple `cmd_buf.commit_and_wait()` sync points between stages.
**Proposed:** Pipeline stages where possible (stage N+1 starts while stage N tree commits).

**Effort:** 8-16 hours (requires careful dependency analysis). **Impact:** 10-20% wall-clock reduction.

### 5.3 Query Leaf Gathering Kernel
**Current:** CPU loop over unified memory to gather query leaf data (~100-200MB).
**Proposed:** GPU kernel to reorganize column-major → query-major layout.

**Effort:** 4 hours. **Impact:** 1.5-2x faster query phase.

## Phase 6: Apple Silicon-Specific (Priority: LOW)

### 6.1 Tile Memory / Imageblock for Blake2s
Apple Silicon A15+ supports tile memory (fast on-chip storage). Blake2s leaf hashing could benefit if leaves fit in tile memory.

**Effort:** 8 hours (experimental). **Impact:** Unknown, likely 10-20% for tree building.

### 6.2 Async Copy (Metal 3)
Metal 3 supports `metal::async_copy` for overlapping compute with data movement. Could overlap twiddle loading with butterfly computation in NTT.

**Effort:** 4-8 hours. **Impact:** 5-10% NTT improvement.

## Prioritized Roadmap

| Priority | Task | Effort | Expected Impact |
|----------|------|--------|-----------------|
| 1 | Canonicalization (byte-identical proofs) | 2-4h | Correctness |
| 2 | Debug logging cleanup | 1h | 2-5% |
| 3 | Fused multi-stage NTT | 4-8h | 15-25% NTT |
| 4 | Simdgroup reduce/scan | 4-6h | 20-30% reduce |
| 5 | COL_PAIRS_PER_BLOCK=4 recovery | 4-8h | 2-4x fewer NTT dispatches |
| 6 | Branchless coefficients | 1h | 5-15% stage 3 |
| 7 | Batch inversion size | 30min | 5-10% stage 4 |
| 8 | Reduce cmd buffer commits | 8-16h | 10-20% wall-clock |
| 9 | Threadgroup-cached twiddles | 2-4h | 5-10% NTT |
| 10 | GPU buffer memset | 1h | Minor |
