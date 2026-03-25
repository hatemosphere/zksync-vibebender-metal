# zksync-vibebender-metal — Apple Metal GPU Prover for ZKsync Airbender

> **vibecoded proof-of-concept** Metal GPU prover for Apple Silicon.
> Port of the CUDA `gpu_prover` to Apple's Metal compute shader API.

Upstream repository: [matter-labs/zksync-airbender](https://github.com/matter-labs/zksync-airbender)

### Upstream GPU prover CUDA libs replaced in-line with Metal alternatives

| CUDA dependency | Purpose | Metal replacement |
|---|---|---|
| `era_cudart` (0.154) | CUDA runtime API (memory alloc, streams, kernel launch) | `objc2-metal` — native Metal API (command buffers, compute pipelines) |
| `era_cudart_sys` (0.154) | Low-level CUDA driver FFI bindings | `objc2` + `objc2-foundation` — Objective-C runtime interop |
| `nvtx` (1.x) | NVIDIA GPU profiling / trace markers | Custom Metal profiler (`gpu_trace.json` via `MTLCommandBuffer` timestamps) |
| `cmake` (build dep) | CUDA kernel compilation (nvcc/ptx) | `glob` — Metal shader discovery, compiled via `xcrun metal` → `.metallib` |
| `era_criterion_cuda` (0.2) | CUDA benchmarking harness | Removed (not needed for Metal) |
| CUDA C++ kernels (`*.cu`) | GPU compute kernels | Metal Shading Language (`*.metal`) — fully reimplemented |

All GPU compute kernels — NTT (forward/inverse), Blake2s Merkle tree hashing, Mersenne31 field arithmetic, parallel reduce/scan/sort, barycentric evaluation, witness generation — are reimplemented in Metal Shading Language. No CUDA code or NVIDIA toolchain required.

## What's here

Metal GPU implementation of the ZKsync Airbender prover pipeline:

- **5-stage proving**: Witness generation, argument polynomials, constraint quotient, DEEP/FRI, proof assembly
- **Metal compute shaders**: NTT (forward/inverse), Blake2s Merkle tree hashing, field arithmetic (Mersenne31), parallel reduce/scan/sort, barycentric evaluation
- **Unified memory**: No explicit CPU-GPU copies — Apple Silicon's shared memory model
- **GPU profiler**: Chrome trace JSON output (viewable in [Perfetto](https://ui.perfetto.dev))

## Build & Run

### Prove with Metal GPU

```bash
# Build
cargo build -p cli --features gpu --release

# Prove (basic fibonacci)
mkdir -p output_gpu
cargo run -p cli --features gpu --release -- prove \
  --bin examples/basic_fibonacci/app.bin \
  --gpu --cycles 100 \
  --output-dir output_gpu

# Prove (hashed fibonacci with delegation)
mkdir -p output_gpu_hashed
cargo run -p cli --features gpu --release -- prove \
  --bin examples/hashed_fibonacci/app.bin \
  --input-file examples/hashed_fibonacci/input.txt \
  --gpu --cycles 1000 \
  --output-dir output_gpu_hashed

# Verify
cargo run -p cli --features include_verifiers --release -- verify-all \
  --metadata output_gpu/metadata.json
```

### Prove with GPU profiling

```bash
cargo run -p cli --features gpu_profile --release -- prove \
  --bin examples/basic_fibonacci/app.bin \
  --gpu --cycles 100 \
  --output-dir output_gpu
```

This prints a GPU + CPU timing summary to stderr and writes `gpu_trace.json` to the output directory.

**View the trace:** Open `gpu_trace.json` in [ui.perfetto.dev](https://ui.perfetto.dev) or `chrome://tracing`. GPU events are on thread 1, CPU spans on thread 2.

### Prove with CPU (no GPU)

```bash
cargo run -p cli --features gpu --release -- prove \
  --bin examples/basic_fibonacci/app.bin \
  --cycles 100 \
  --output-dir output_cpu
```

(Omit the `--gpu` flag to use the CPU prover.)

## Performance

Measured on **MacBook Pro M4 Max** (16 CPU cores, 40 GPU cores, 48GB unified memory):

| Example | Circuit size | CPU (16 cores) | Metal GPU | Speedup |
|---------|-------------|----------------|-----------|---------|
| basic_fibonacci | 2^22 (1 proof) | 6.9s | **3.9s** | **1.8x** |
| hashed_fibonacci | 2^22 (2 proofs) | 20.3s | **7.8s** | **2.6x** |

> **Note on CPU baseline:** The CPU prover uses all available cores (16 on M4 Max) with SIMD-optimized field arithmetic. The speedup over a single-threaded CPU prover would be much larger (~10x+). The moderate GPU speedup reflects the fact that Apple Silicon's unified memory architecture means the GPU and CPU share the same memory bandwidth, and the prover's Fiat-Shamir transcript requires ~20 sequential GPU-CPU sync points per proof.

### Per-stage wall time breakdown

```
                    basic_fibonacci (1 proof)    hashed_fibonacci (2 proofs)
Stage 1 (witness)          254ms                       255ms + 421ms
Stage 1 (commit)          1471ms                      1394ms + 546ms
Stage 2 (args)             829ms                       872ms + 1334ms
Stage 3 (constraints)      161ms                       186ms + 136ms
Stage 4 (DEEP/FRI poly)    140ms                       134ms + 164ms
Stage 5 (FRI folding)       12ms                        10ms + 5ms
PoW                        245ms                       148ms + 909ms
Queries                    284ms                       321ms + 518ms
Proof assembly              11ms                        16ms + 7ms
─────────────────────────────────────────────────────────────────────
Total proving             3407ms                      3337ms + 4039ms
```

### GPU kernel time breakdown (basic_fibonacci)

```
Blake2s Merkle tree     650ms  35%    memory-bandwidth-bound (column-major layout)
NTT forward+inverse     352ms  19%    compute-bound
PoW search              245ms  13%    compute-bound (nonce brute-force)
Witness gen + memset    135ms   7%
Barycentric eval         93ms   5%
Queries (gather)        150ms   8%
Constraints (stage 3)    59ms   3%
Stage 2 args             33ms   2%
Other                   122ms   7%
────────────────────────────────────
Total GPU time         1839ms
```

### Resource usage (hashed_fibonacci, GPU)

```
Peak memory:     ~24.5 GB (unified, shared between CPU and GPU)
GPU utilization: ~35% (limited by ~20 commit_and_wait sync points per proof)
CPU utilization: ~100% of 1 core (transcript hashing, proof assembly)
Wall time:       ~7.8s (2 proofs)
```

## CUDA vs Metal: TL;DR

| | CUDA (`gpu_prover`) | Metal (`gpu_prover_metal`) |
|---|---|---|
| Memory model | Explicit H2D/D2H copies | Unified memory (zero-copy) |
| SIMD width | 32 (warp) | 32 (simdgroup) |
| Shared memory | 48-96KB configurable | 32KB fixed |
| Shuffle intrinsic | `__shfl_xor_sync` | `simd_shuffle_xor` |
| Atomic ops | `atomicAdd` | `atomic_fetch_add_explicit` |
| Command submission | Stream-based, async | Command buffer, explicit commit |
| Kernel compilation | PTX/SASS at build time | AIR/metallib, JIT on first dispatch |
| GPU watchdog | None (dedicated GPU) | ~5s timeout (shared with display) |

The Metal port maps CUDA concepts 1:1 where possible: warps become simdgroups, `__shared__` becomes `threadgroup`, streams become command buffers. The main structural difference is the GPU watchdog — Metal command buffers must be bounded to avoid freezing macOS.

## Proof Byte-Identity with CPU

GPU and CPU proofs in theory should be **mathematically equivalent but NOT byte-identical**. GPU Metal proofs should in theory verify correctly. The differences (CORRECTNESS IS NOT VERIFIED BY SPECIALISTS):

1. **Stage 2 divergence**: The GPU's parallel reduce/scan operations accumulate values in a different order than the CPU's sequential implementation. Both produce mathematically correct results, but the intermediate Mersenne31 field representations differ (e.g., `0` vs `2^31-1` for the zero element). This causes different Merkle tree leaf hashes starting from stage 2, which cascades into different transcript challenges and a divergent (but valid) proof.

2. **Canonicalization**: We canonicalize field values (`ORDER -> 0`) at GPU-to-proof serialization boundaries. The CPU prover doesn't always canonicalize, so the raw serialized values may differ even for identical mathematical results.

3. **PoW nonce**: The proof-of-work search finds different valid nonces due to different GPU thread scheduling.

## Disclaimer

**This is a proof-of-concept.** Only two example circuits have been tested:

- `basic_fibonacci` — simple Fibonacci computation (1 proof)
- `hashed_fibonacci` — Fibonacci + Blake2s delegation (1 main + 1 delegation proof)

Both generate valid proofs that pass the verifier. No other circuits, input sizes, or edge cases have been tested. The Metal prover should NOT be used in production without thorough testing across the full circuit suite.

Known limitations:
- Blake2s leaf hashing uses column-major layout (required by NTT) causing ~16MB stride between columns. Shared-memory tiling and multi-pass chunking were tried but add overhead on Apple Silicon's unified memory — the GPU's thread-level parallelism already hides most latency from strided reads
- NTT block-level kernels limited to COL_PAIRS=1 due to 32KB threadgroup memory shared between twiddle cache and cross-warp data exchange (warp-level kernels use COL_PAIRS=4)
- ~20 GPU-CPU sync points per proof (protocol-mandated for Fiat-Shamir transcript)
- GPU memory usage is high (~24GB for hashed_fibonacci) due to unified memory allocation model
