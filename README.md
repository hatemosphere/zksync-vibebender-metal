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
- **GPU profiler**: Chrome trace JSON output (viewable in [Perfetto](https://ui.perfetto.dev)) + Apple Instruments integration via `os_signpost` and Metal encoder labels

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
# Built-in profiler (Chrome trace + summary table)
cargo run -p cli --features gpu_profile --release -- prove \
  --bin examples/basic_fibonacci/app.bin \
  --gpu --cycles 100 \
  --output-dir output_gpu
```

This prints a GPU + CPU timing summary to stderr and writes `gpu_trace.json` to the output directory.

**View the trace:** Open `gpu_trace.json` in [ui.perfetto.dev](https://ui.perfetto.dev) or `chrome://tracing`. GPU events are on thread 1, CPU stage spans on thread 2, CPU detail on thread 3+.

```bash
# Apple Instruments (zero-overhead, per-kernel GPU timing)
cargo build -p cli --features gpu --release
xctrace record --template 'Metal System Trace' --launch -- \
  target/release/cli prove \
  --bin examples/basic_fibonacci/app.bin \
  --gpu --output-dir /tmp/output
```

Open the `.trace` file in Instruments.app. GPU compute encoders are labeled with kernel function names. CPU spans emit `os_signpost` intervals under subsystem `com.matterlabs.gpu-prover`.

### Prove with CPU (no GPU)

```bash
cargo run -p cli --features gpu --release -- prove \
  --bin examples/basic_fibonacci/app.bin \
  --cycles 100 \
  --output-dir output_cpu
```

(Omit the `--gpu` flag to use the CPU prover.)

## Performance

Recent verified measurements on **MacBook Pro M4 Max** (16 CPU cores, 40 GPU cores, 48GB unified memory):

| Example | Circuit size | CPU (16 cores) | Metal GPU | Speedup |
|---------|-------------|----------------|-----------|---------|
| basic_fibonacci | 2^22 (1 proof) | 6.9s | **2.05s** | **3.4x** |
| hashed_fibonacci | 2^22 (2 proofs) | 20.3s | **3.65s** | **5.6x** |

> **Note on CPU baseline:** The CPU prover uses all available cores (16 on M4 Max) with SIMD-optimized field arithmetic. The speedup over a single-threaded CPU prover would be much larger (~10x+). GPU utilization is ~74% with the remaining time spent on CPU-side Metal API overhead and Fiat-Shamir transcript sync points between stages.

### Per-stage wall time breakdown

```
                    basic_fibonacci (1 proof)
Stage 1 (witness)          225ms
Stage 1 (commit)           910ms
Stage 2 (args+commit)      572ms
Stage 3 (constraints)       74ms
Stage 4 (DEEP/FRI poly)    96ms
Stage 5 (FRI folding)       10ms
PoW                         12ms
Queries                      9ms
Proof assembly              32ms
──────────────────────────────────
Total proving            1940ms
```

### GPU kernel time breakdown (basic_fibonacci)

```
Stage 1 commit (trees)    712ms  50%   NTT + blake2s Merkle tree hashing
Stage 2 args + commit     358ms  25%   lookup/memory arguments + tree
Stage 1 witness gen       139ms  10%   witness + memory value generation
Stage 3 constraints        68ms   5%   constraint quotient + tree
Barycentric eval           47ms   3%   batched multi-column evaluation
Deep quotient + trees      20ms   1%
PoW search                 12ms   1%   blake2s nonce search (-O2)
Other                      71ms   5%
────────────────────────────────────
Total GPU time          1427ms
Total dispatches          455
```

### Resource usage

```
Peak process memory footprint:
  basic_fibonacci  ~42 GB
  hashed_fibonacci ~46 GB

Host buffer pool:
  12 x 648 MB = ~7.8 GB reserved up front
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
- GPU utilization is ~74% — the remaining 26% is CPU-side Metal API overhead (command buffer encoding, scheduling latency, Fiat-Shamir transcript sync points between stages)
- Blake2s leaf hashing uses column-major layout (required by NTT) causing ~16MB stride between columns. The kernel is compute-bound on blake2s compression, not memory-bound
- NTT block-level kernels limited to COL_PAIRS=1 due to 32KB threadgroup memory
- Peak process memory footprint is high (~42-46 GB)
