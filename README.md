# FSR4 RDNA3.5 Benchmark Summary

## Scope

[AMD FidelityFX Super Resolution 4 (FSR4)](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) is AMD's ML-based upscaler that ships as quantized HLSL compute shaders generated from ONNX models via AMD's ML2Code toolchain. This repository benchmarks HIP microkernels that emulate FSR4-style INT8/FP8 compute behavior on Strix Halo (RDNA3.5, `gfx1151`), targeting the core dot-product and FMA operations that dominate the model's runtime.

This is not a full DX12 frame-time benchmark of shipping FSR4 -- it isolates the low-level compute primitives to understand how RDNA3.5 handles the quantized arithmetic that FSR4 depends on.

## FSR4 Model Overview

FSR4 uses a **quantized CNN** with an encoder-bottleneck-decoder (U-Net style) architecture, compiled from ONNX to HLSL shaders via ML2Code. Key details:

- **Model version**: v07 (from `fsr4_model_v07_*.onnx`)
- **Architecture**: Encoder-decoder CNN with skip connections
  - `encoder1`: Strided 2x2 convolution (DownscaleStridedConv2x2) for spatial downsampling
  - `encoder2`: 2x residual blocks using depthwise separable convolutions (conv_dw + conv_pw_expand + conv_pw_contract)
  - `encoder3`: 2x residual blocks with spatial mixing partial convolutions
  - `bottleneck`: 2x residual blocks with spatial mixing + transposed 2x2 convolution (UpscaleConvTranspose2x2) for upsampling
  - `decoder3`/`decoder2`/`decoder1`: Mirror the encoder stages, each with residual blocks and upscale convolutions
  - Postprocessing: RCAS (Robust Contrast Adaptive Sharpening) and SPD (Single Pass Downsampler) with auto exposure
- **Compute passes**: 14 sequential dispatch passes per frame (at 1080p), each with a pre/post quantization stage
- **Scratch memory**: ~20 MB
- **Weights**: ~88 KB per quality tier (quantized, stored as embedded dwords or `initializers.bin`)
- **Input**: NHWC layout, 7 input channels (current + history frames), resolution-specific shaders for 1080p/2160p/4320p
- **Quantization**: INT8 (fixed-point with learned scale/bias) or FP8 (e4m3 format); accumulation in FP32, quantized at store

### Precision and Quality Variants

FSR4 ships multiple precision/quality tradeoffs as separate shader sets:

| Precision | Quality Modes |
|---|---|
| INT8 | balanced, quality, drs, native, performance, ultraperf |
| FP8 (e4m3) | balanced, quality, drs, native, performance, ultraperf |

Each variant also has WMMA-accelerated shaders (`pre_wmma.hlsl`, `post_wmma.hlsl`) for GPUs with Wave Matrix Multiply Accumulate support.

## RDNA3.5 Support Snapshot

RDNA3.5 (gfx1151, Strix Halo) provides native hardware acceleration for the quantized operations FSR4 relies on. This repo exercises two key instruction paths:

| Capability | Status In This Repo | Notes |
|---|---|---|
| INT8 dot compute | Benchmarked | Uses `amd_mixed_dot` (packed 4-element INT8 dot product) and scalar INT8 paths. Scalar path outperformed packed by ~32%. |
| FP8 (`e4m3`) compute | Benchmarked | Uses `hip_fp8` library for native FP8 conversion + FMA. Accumulates in FP32, quantizes once at store. |
| WMMA path | Present in FSR4 source, not benchmarked | FSR4 source contains WMMA shader files for wave-level matrix ops. Outside scope of HIP microkernel harness. |
| Wave size | Verified | `warpSize=32` confirmed on gfx1151. All kernels dispatch 256 threads (8 waves per threadgroup). |

RDNA3.5 ISA reference (online):
- https://github.com/woct0rdho/rdna35-isa-markdown

### What We Benchmark

The HIP kernels emulate the core compute loop of FSR4's quantized convolution passes:

- **INT8 path**: Dot-product accumulation of quantized activations and weights, with learned scale/bias dequantization. Tests both `amd_mixed_dot` (packed 4-element dot in one instruction) and scalar element-wise multiply-accumulate.
- **FP8 path**: FMA (fused multiply-accumulate) using `hip_fp8` e4m3 format with FP32 accumulation. Tests conversion overhead and arithmetic throughput.

Both paths use compile-time unrolled inner loops with configurable iteration depth, operating on 262,144 logical vectors per dispatch.

### HIP Harness vs Real FSR4 Shaders

Our HIP microkernels exercise the same *class* of quantized arithmetic that FSR4 uses, but they are **not a direct port** of the actual HLSL shader logic. The differences matter for interpreting results:

| Aspect | HIP Harness | Real FSR4 HLSL |
|---|---|---|
| **INT8 dot product** | `amd_mixed_dot` (packed 4×INT8→INT32) or scalar INT8 multiply-accumulate | `dot2add(half2, half2, float)` -- unpacks INT8 to FP16 via `Unpack4h()`, accumulates in FP32 |
| **FP8 compute** | Element-wise FMA via `hip_fp8` library | **Requires WMMA** -- `AmdWaveMatrixMultiply()` for 16×16 matrix ops with LDS staging |
| **Data layout** | Flat 1D arrays, simple element-per-thread mapping | 3D/4D NHWC tensors with spatial tiling, specialized fast paths for common shapes (e.g., 8-channel input, 16-feature output) |
| **Loop structure** | Single inner loop with configurable unroll depth | Nested `[unroll]` loops over kernel spatial dims (kx, ky), input channels, and output features in groups of 4 |
| **Quantization** | Scale/bias dequantization with store-time requant | Same policy (FP32 accumulation, quantize once at store) but with structured per-layer learned scales |

The harness is most useful as a **directional signal** about RDNA3.5 arithmetic throughput and memory behavior at the instruction level, not as a cycle-accurate proxy for full FSR4 frame time.

### Real-World Implications

Applying our microkernel findings to the actual FSR4 implementation would require changes at different levels depending on the optimization:

**Directly applicable** (same policy, confirmed by both harness and HLSL source):
- **Store-time quantization**: Accumulate in full precision, quantize once at output. The real HLSL already does this (`round(vs * rcpScale)` at the end of each convolution). Our harness confirmed that per-iteration requantization causes catastrophic regression (INT8 +194%, FP8 +476%). Any code path that requantizes mid-accumulation should be treated as a bug.
- **Compile-time loop unrolling**: The real HLSL uses `[unroll]` on all kernel loops. Our harness confirmed ~12% gains from unrolling vs runtime loop control. ML2Code should ensure all generated inner loops remain statically unrollable.

**Directionally relevant** (same hardware, different instruction mix):
- **Scalar element-wise > packed `amd_mixed_dot`**: Our biggest INT8 win (~32%), but the real INT8 HLSL path uses FP16 `dot2add` rather than INT8 `amd_mixed_dot`. This suggests the INT8 dot product unit on gfx1151 may not be the fastest path. ML2Code should benchmark `dot2add` vs `amd_mixed_dot` vs scalar INT8 for the actual Conv2D operators on RDNA3.5 to determine whether the current `dot2add` approach is already optimal or whether an INT8-native path could be faster.
- **LDS staging is slow on Strix Halo iGPU**: All four LDS variants we tested regressed (O10-O13), likely due to shared memory subsystem contention on iGPU. This is concerning for the FP8/WMMA path, which requires LDS staging (`groupshared uint inputLDS[]`) for wave matrix input loading. The WMMA path may underperform on iGPU vs dGPU for this reason.

**Requires further investigation**:
- **Thread block sizing**: Our harness found 256 threads (8 waves) optimal, but FSR4 shaders dispatch with `numthreads(32, 1, 1)` (single wave). The real shaders tile differently (one spatial position per thread, all features computed), so the occupancy tradeoffs are different. Worth profiling whether multi-wave dispatch could help for the larger residual block passes.
- **14-pass dispatch overhead**: At ~5µs per INT8 kernel invocation, the 14-pass sequential pipeline has significant per-dispatch overhead relative to compute time on iGPU. Fusing adjacent passes or reducing pass count could improve end-to-end latency, but this requires ML2Code changes.

**Recommendations for RDNA3.5 optimization** (actionable for ML2Code / FSR4 shader development):
- Benchmark the actual `int8_NHWC/Conv2D_k3p1b` operator with `dot2add` vs an `amd_mixed_dot` variant on gfx1151 -- our results suggest native INT8 dot products may not be optimal on this specific ISA
- Profile the FP8 WMMA path (`float8_NHWC/Conv2D_k2s2b`) on Strix Halo specifically -- LDS contention on iGPU may make WMMA less effective than on discrete RDNA3 GPUs like Navi 48
- Validate that all ML2Code-generated code paths use store-time (not per-iteration) quantization
- Consider `-O3` as the default compiler optimization level -- our testing found no benefit from `-O2` or `-Ofast/-ffast-math`, and `-O3` matched or beat all alternatives
- Investigate pass fusion opportunities for the 14-pass pipeline, especially for the smaller residual block passes where dispatch overhead may dominate compute time on low-power iGPUs

## Before/After Performance (Main Summary)
Comparison of the stable direct-TTY baseline vs the final selected defaults after the optimization loop.

- Before file: `results/baseline-benchmark-20260227-040756.json`
- After file: `results/baseline-benchmark-20260227-052146.json`

| Mode | Before Mean (ms) | After Mean (ms) | Improvement |
|---|---:|---:|---:|
| INT8 | 0.007743 | 0.005376 | 30.57% faster |
| FP8 | 0.117392 | 0.019868 | 83.08% faster |

The biggest wins came from switching INT8 I/O from packed to scalar element-wise access (~32% gain) and from loop unrolling + requantization policy fixes that dramatically improved FP8 (~83% gain). Variance also dropped significantly (INT8 cv: 1.24% -> 0.70%, FP8 cv: 1.28% -> 0.58%), meaning the optimized kernels are both faster and more predictable.

## INT8 vs FP8 Relative Performance
Lower time is better. `FP8/INT8` > 1.0 means INT8 is faster.

| Snapshot | INT8 Mean (ms) | FP8 Mean (ms) | FP8/INT8 Ratio | INT8 Speed Advantage |
|---|---:|---:|---:|---:|
| Before | 0.007743 | 0.117392 | 15.16x | 1416.10% |
| After | 0.005376 | 0.019868 | 3.70x | 269.57% |

Summary:
- INT8 remains faster than FP8 in this harness, consistent with INT8 using simpler integer ALU vs FP8's conversion overhead.
- The INT8-vs-FP8 gap narrowed significantly (about `4.10x` narrower ratio vs before), mostly because FP8 improved a lot in the updated harness path. The initial FP8 implementation had a catastrophic per-iteration requantization policy that was 5x slower than necessary.

## Benchmark Methodology
### Protocol
- Primary command:
  - `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1`
- `60s` target is per mode (`int8` and `fp8`), so `mode=both` runs about 2 minutes total.
- Stats collected: `mean`, `stddev`, `median`, `p95`, `cv_pct`, run count.
- Keep/drop gate uses classification from `--reference-stats` with uncertainty fallback (`min_uncertainty_pct=3`, `cv_scale=0.5`).

### Jitter and Stability

Benchmark jitter is a first-class concern for microkernel timing at the microsecond scale. GPU scheduling noise, power management, and system load can easily dominate the signal.

- **Direct TTY control runs** (no desktop compositor/session) were very stable with cv < 1% for the final defaults. This is the authoritative measurement environment.
- **Under interactive/system load**, several FP8 variants showed high variance (cv often 25-60%). These were marked `Unsure` unless clearly outside uncertainty bounds. This reflects real iGPU contention -- the Strix Halo shares its memory subsystem with the CPU, so background activity directly impacts kernel latency.
- **Decision gate**: Only optimizations showing improvement outside a `min_uncertainty_pct=3%` / `cv_scale=0.5` envelope were accepted. This conservative gate means we likely left some marginal gains on the table, but avoids false positives from noise.

### Environment Snapshot
| Item | Value |
|---|---|
| GPU | Radeon 8060S Graphics |
| Target arch | `gfx1151` |
| Mamba env | `therock` |
| HIP/ROCm | `HIP version: 7.12.60490-128c4eea36` |
| ROCm SDK | `7.12.0a20260226` |
| Kernel | `6.19.0-rc6-1-mainline` |

## Optimization Attempts (Single-Glance)

20 optimization attempts were evaluated systematically: 5 kept, 11 rejected, 4 uncertain (conservative defaults retained). Key takeaways: scalar INT8 I/O and compile-time loop unrolling were the biggest wins; LDS staging strategies all regressed despite theoretical benefits (likely due to memory subsystem contention on iGPU); compiler flag tuning showed negligible impact within the noise floor.

Notes:
- `O02-O06` were measured in the earlier direct-TTY phase.
- `O07-O19` were measured against phase-2 control `results/baseline-benchmark-20260227-052146.json`.

| ID | Change | Outcome | Headline Effect |
|---|---|---|---|
| O01 | Protocol lock | Keep | Stable 60s baseline captured in direct TTY. |
| O02 | Threadgroup sweep (`64/128/256`) | Keep (`256`) | `64/128` slower and/or high variance. |
| O03 | Items-per-thread sweep (`1/2/4`) | Keep (`1`) | `2` worse, `4` much worse/high variance. |
| O04 | Wave-size verification | Keep | Confirmed `warpSize=32`. |
| O05 | Unrolled vs runtime inner loops | Keep (unrolled) | Runtime-loop mode slower/high variance. |
| O06 | Packed vs scalar INT8 I/O | Keep (scalar default) | INT8 improved ~31.84% vs packed reference. |
| O07 | Hoist scale/bias loads | Drop (forced in-loop variant) | In-loop load variant much slower. |
| O08 | Per-iter requant vs once-at-store | Drop (forced per-iter variant) | Large regression on both modes. |
| O09 | Interior/edge split dispatch | Unsure | Small INT8 gain, FP8 very noisy. |
| O10 | LDS stage input | Drop | Slower, high variance. |
| O11 | LDS stage input+weight | Drop | Slower than O10. |
| O12 | LDS padding/swizzle proxy | Drop | Stable but much slower. |
| O13 | LDS double-buffer proxy | Drop | Slightly better than O12, still much slower. |
| O14 | Occupancy/register sweep | Unsure | No confidence-grade win; `inner=32` hurt FP8. |
| O15 | Compile flags (`-O2`, `-Ofast/-ffast-math`) | Unsure | Tiny INT8 movement, FP8 too noisy; kept `-O3`. |
| O16 | Unfused post-op path | Drop | Clear regression. |
| O17 | Two-pass adjacent path | Drop | Clear regression. |
| O18 | Mixed INT8 subpath | Drop | Large INT8 regression. |
| O19 | FP8 quantized-IO fallback | Unsure | Median looked better, variance too high. |
| O20 | Final cleanup | Keep | Kept only net-positive defaults. |

## Where Details Live
- Full run-by-run log: `WORKLOG.md`
- Implementation decisions and commands: `IMPLEMENTATION.md`
- Detailed result tables: `OPTIMIZATION_RESULTS.md`
- Queue state: `OPTIMIZATION_QUEUE.md`

## Repo Structure and Doc Links
### Repository Layout
```text
.
├── benchmarks/                  # HIP benchmark kernels (baseline_kernels_bench.cpp)
├── baseline-benchmark.py        # Python benchmark harness (build/run/classify)
├── fsr4-src/
│   ├── baseline/                # Immutable FSR4 source snapshot (HLSL shaders + weights)
│   └── opt/                     # Optimization workspace
├── results/                     # Benchmark JSON outputs and latest symlink files
├── reference/                   # External reference material/checkouts
├── AGENTS.md                    # Environment + workflow instructions
├── README.md                    # Single-glance project summary (this file)
├── ANALYSIS.md                  # Extended analysis snapshot/history
├── OPTIMIZATION_QUEUE.md        # Optimization queue status
├── OPTIMIZATION_RESULTS.md      # Compact optimization results tables
├── IMPLEMENTATION.md            # Per-optimization implementation log
├── WORKLOG.md                   # Reverse-chronological session log
└── TODO.md                      # Punchlist tracking
```

### Key Docs
- [ANALYSIS.md](ANALYSIS.md)
- [AGENTS.md](AGENTS.md)
- [WORKLOG.md](WORKLOG.md)
- [IMPLEMENTATION.md](IMPLEMENTATION.md)
- [OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md)
- [OPTIMIZATION_QUEUE.md](OPTIMIZATION_QUEUE.md)
- [TODO.md](TODO.md)
