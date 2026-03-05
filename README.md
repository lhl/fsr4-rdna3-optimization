# FSR4 RDNA3.5 Benchmark Summary

## Scope

[AMD FidelityFX Super Resolution 4 (FSR4)](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) is AMD's ML-based upscaler that ships as quantized HLSL compute shaders generated from ONNX models via AMD's ML2Code toolchain. This repository benchmarks HIP microkernels that emulate FSR4-style INT8/FP8 compute behavior on Strix Halo (RDNA3.5, `gfx1151`), targeting the core dot-product and FMA operations that dominate the model's runtime.

This is not a full DX12 frame-time benchmark of shipping FSR4 -- it isolates the low-level compute primitives to understand how RDNA3.5 handles the quantized arithmetic that FSR4 depends on.

## FSR4 Model Overview

FSR4 uses a **quantized CNN** compiled from ONNX to HLSL shaders via ML2Code. In this repo, the generated v07 model is organized as an encoder/bottleneck/decoder pipeline with skip connections. Key details:

- **Model version**: v07 (from `fsr4_model_v07_*.onnx`)
- **Architecture**: Encoder-decoder CNN with skip connections
  - `encoder1`: Strided 2x2 convolution (DownscaleStridedConv2x2) for spatial downsampling
  - `encoder2`: 2x ConvNext-style residual blocks (`conv_dw` + `conv_pw_expand` + `conv_pw_contract`)
  - `encoder3`: 2x FasterNet-style residual blocks with spatial-mixing partial convolutions
  - `bottleneck`: 2x residual blocks with spatial mixing + transposed 2x2 convolution (UpscaleConvTranspose2x2) for upsampling
  - `decoder3`/`decoder2`/`decoder1`: Mirror the encoder stages, each with residual blocks and upscale convolutions
  - Postprocessing: RCAS (Robust Contrast Adaptive Sharpening) and SPD (Single Pass Downsampler) with auto exposure
- **Compute passes**: 14 model passes (`pass0..pass13`) plus 13 post passes (`pass0_post..pass12_post`) at 1080p
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
| Wave size | Verified | `warpSize=32` confirmed on gfx1151. The HIP harness default is 256 threads, while generated FSR4 model shaders use mixed groups (`32x1x1`, `64x1x1`, and `8x8x1`). |

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
| **INT8 dot product** | `amd_mixed_dot` (packed 4×INT8→INT32) or scalar INT8 multiply-accumulate | Mixed path: pass 0 uses FP16 `dot2add` (`Unpack4h`), while passes 1-13 are dominated by INT8 `dot4add_i8packed` (`DOT4_ENABLED=1`) in fused operators |
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
- **Scalar element-wise > packed `amd_mixed_dot`**: Our biggest INT8 win (~32%). The real INT8 HLSL is mixed: pass 0 uses `dot2add`, but passes 1-13 are mostly `dot4add_i8packed` (same instruction class as `amd_mixed_dot`). ML2Code should benchmark `dot4add_i8packed` vs scalar INT8 in the fused ConvNext/FasterNet operators on gfx1151; this is the highest-impact comparison for real FSR4 INT8 passes.
- **LDS staging is slow on Strix Halo iGPU**: All four LDS variants we tested regressed (O10-O13), likely due to shared memory subsystem contention on iGPU. This is concerning for the FP8/WMMA path, which requires LDS staging (`groupshared uint inputLDS[]`) for wave matrix input loading. The WMMA path may underperform on iGPU vs dGPU for this reason.

**Requires further investigation**:
- **Thread block sizing**: Our harness found 256 threads (8 waves) optimal, but real model shaders already use mixed multi-wave groups (`64x1x1` and `8x8x1`, plus `32x1x1` post passes). The tiling is different from the harness, so occupancy tradeoffs are not directly transferable. Worth profiling larger groups on selected bottleneck/fused passes.
- **Dispatch overhead**: The INT8 pipeline is 14 model passes + 13 post passes (27 dispatches total). At ~5us-level kernel times in the harness, dispatch overhead can be material on iGPU. Fusing adjacent passes or reducing post-pass work could improve end-to-end latency, but this requires ML2Code changes.

**Recommendations for RDNA3.5 optimization** (actionable for ML2Code / FSR4 shader development):
- Benchmark fused INT8 operators (`ConvNextBlock`, `FasterNetBlock`, `FNB_CT2D_ADD`, `CNB_CT2D`) with `dot4add_i8packed` vs scalar INT8 variants on gfx1151 -- this targets passes 1-13 where most INT8 compute happens
- Profile the FP8 WMMA path (`float8_NHWC/Conv2D_k2s2b`) on Strix Halo specifically -- LDS contention on iGPU may make WMMA less effective than on discrete RDNA3 GPUs like Navi 48
- Validate that all ML2Code-generated code paths use store-time (not per-iteration) quantization
- Consider `-O3` as the default compiler optimization level -- our testing found no benefit from `-O2` or `-Ofast/-ffast-math`, and `-O3` matched or beat all alternatives
- Investigate pass/post-pass fusion opportunities for the 27-dispatch INT8 pipeline (14 model + 13 post), especially where dispatch overhead may dominate compute time on low-power iGPUs

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

## Theoretical vs Practical: RDNA3 Quantized Arithmetic

RDNA3/3.5 provides native dot-product instructions for multiple integer precisions. The theoretical throughput scaling is straightforward -- each step down in precision doubles the elements packed per instruction:

| Data Type | Instruction | Elements/Cycle/CU | Theoretical vs FP16 |
|---|---|---|---|
| FP16 | `v_dot2_f16_f16` | 2 MADs (4 ops) | 1x |
| INT8 | `v_dot4_i32_iu8` | 4 MADs (8 ops) | 2x |
| INT4 | `v_dot8_i32_iu4` | 8 MADs (16 ops) | 4x |

FP8 (e4m3) takes a different path -- it requires WMMA (Wave Matrix Multiply Accumulate) for 16x16 matrix tiles with LDS staging, rather than simple per-lane dot products.

### What We Actually Measured (Strix Halo iGPU, gfx1151)

The practical results diverge significantly from theoretical expectations:

1. **Scalar INT8 > packed INT8 by ~32%**: The `v_dot4_i32_iu8` / `amd_mixed_dot` instruction that should be the fast path was actually slower than scalar element-wise multiply-accumulate on gfx1151. This was our single largest optimization finding and affects 93% of FSR4's compute passes (passes 1-13). The likely cause is microarchitectural scheduling differences on the RDNA3.5 iGPU -- the packed dot-product instruction may have higher latency or worse pipelining than four scalar multiplies on this specific target.

2. **FP8 is 3.7x slower than INT8** (after optimization): Despite FP8 being the same bit-width as INT8, the WMMA requirement introduces LDS staging overhead that dominates on iGPU. Before optimization the gap was 15x, mostly due to a catastrophic per-iteration requantization bug -- but even after fixing that, FP8 can't match INT8's simple ALU path on shared-memory-constrained hardware.

3. **LDS staging uniformly regresses on iGPU**: All four LDS strategies we tested (O10-O13) made things worse. The Strix Halo's shared memory subsystem is contended between CPU and GPU, making any LDS-dependent path (including FP8/WMMA) structurally disadvantaged vs register-only computation.

### INT4 and Beyond: Untapped Hardware Potential

RDNA3's `v_dot8_i32_iu4` instruction provides a theoretical 4x throughput advantage over FP16 and 2x over INT8 -- packing 8 four-bit elements into a single DWORD operand with INT32 accumulation. This instruction exists in hardware but is **not used by FSR4 today**.

Practical barriers to INT4 adoption:

- **Precision**: INT4 has only 16 discrete values (-8 to +7 signed). INT8's 256 levels are already aggressive for neural network weights/activations -- INT4 would almost certainly require **quantization-aware training (QAT)** to maintain acceptable image quality. The FSR4 model was trained for INT8/FP8 precision, not INT4.
- **Packed instruction penalty**: Our finding that packed `v_dot4_i32_iu8` underperforms scalar INT8 on gfx1151 is a cautionary signal. `v_dot8_i32_iu4` is the same class of packed dot-product instruction and may hit the same microarchitectural scheduling penalty on iGPU. This needs benchmarking before assuming the theoretical 2x over INT8 materializes.
- **Packing overhead**: Two INT4 values share each byte. If the model isn't natively INT4-quantized, runtime packing/unpacking costs erode the throughput advantage.

**What it would take**: AMD's ML2Code toolchain would need to support INT4 quantization as a target precision, and the FSR4 model would need to be retrained with INT4-aware quantization (QAT or GPTQ-style post-training quantization with calibration). If the image quality holds, INT4 could deliver a meaningful speedup on RDNA3 -- but only if the packed instruction penalty observed with INT8 doesn't also apply to `v_dot8_i32_iu4`.

### Key Takeaway

On paper, lower precision = proportionally higher throughput on RDNA3. In practice, the instruction scheduling, memory subsystem, and data movement costs dominate at microsecond-scale kernel times on iGPU. The "best" precision for a given target requires empirical validation -- our results show that the theoretically optimal instruction (`v_dot4_i32_iu8`) was beaten by a simpler scalar approach, and the theoretically equivalent-width FP8 path was 3.7x slower due to its WMMA/LDS dependency.

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
- [session-analysis/](session-analysis/) - Session analysis tooling and process docs

## AI-Assisted Optimization: Session Analysis

The entire optimization campaign -- from initial source code analysis through 20 optimization experiments to final documentation -- was conducted as a collaboration between a human developer and AI coding assistants. The workflow used Claude Code (Opus 4.6) for source code analysis and documentation, and Codex CLI (GPT-5.3-Codex) for the hands-on kernel optimization work. This section documents the actual session data in chronological order: how long things took, how many tokens were consumed, and what the human-AI interaction pattern looked like.

Analysis tooling and methodology are documented in [session-analysis/](session-analysis/).

### 1. Claude Code: FSR4 Source Code Analysis (Pre-Work)

Before any optimization began, Claude Code was used to analyze the full FSR4 source code repository (`FidelityFX-SDK-FSR4-SOURCE-CODE`) and produce a structured ANALYSIS.md. This informed the entire optimization strategy -- identifying INT8 vs FP8 paths, WMMA usage, quantization policy, and RDNA3.5-relevant optimization opportunities.

| Metric | Value |
|---|---|
| **Session ID** | `eedc02b0-93f2-4e0f-a237-4f63a495762e` |
| **Model** | `claude-opus-4-6` |
| **Wall time** | **16m 0s** (Feb 26 14:53 -> 15:09 UTC) |
| **Active time** | 16m 0s (100% -- no idle gaps) |
| **Turns** | 19 user / 35 assistant |
| **Tokens** | 62K input + 418K cache-create + 1.27M cache-read + 15.6K output |
| **Total API tokens** | **~1.76M** |

The initial user prompt was: *"AMD recently released their FSR4 source code. My understanding is there is an FP8 model, but can you review the repo. Is there also an FP16/BF16 model? My understanding is RDNA3 in particular has no FP8 acceleration but does have native INT8, so it'd be much better to quant to that if possible?"*

Over 19 turns, Claude explored the HLSL shader source, model ONNX files, and ML2Code toolchain to produce a comprehensive analysis covering model architecture (encoder-bottleneck-decoder U-Net), quantization paths (INT8 fixed-point vs FP8 e4m3), WMMA acceleration, and the key insight that RDNA3.5 does have native INT8 support via `amd_mixed_dot` but FP8 requires WMMA matrix ops with LDS staging. This analysis directly shaped which optimizations were attempted in the Codex sessions that followed.

### 2. Codex CLI: Main Optimization Session

The primary optimization work was done in a single long-running Codex CLI session using GPT-5.3-Codex. The developer worked interactively through setup and early benchmarks, then left Codex running autonomously to complete the optimization queue.

| Metric | Value |
|---|---|
| **Session ID** | `019c9a8d-bf44-7201-8831-be8c09e650c4` |
| **Model** | `gpt-5.3-codex` (CLI v0.105.0) |
| **Wall time** | **11h 50m** (Feb 26 15:26 -> Feb 27 03:16 UTC) |
| **Active time** | **2h 34m** (21.7% of wall time) |
| **Turns** | 438 turn contexts (model invocations) |
| **User messages** | ~19 unique |
| **Tool calls** | 400 |
| **Tokens** | 55.54M input (53.60M cached) + 181.5K output (74.9K reasoning) |
| **Total API tokens** | **~55.72M** |
| **Total events** | 3,944 |

#### Activity Breakdown

The session had distinct phases with idle gaps between them. Timestamps are UTC:

```
Phase 1 - Setup & benchmark scaffolding     15:26-15:40  (~14 min active)
  User messages: environment setup, hipcc verification, mamba env config,
  copying model code into repo, planning optimization list
    [idle 80 min - user away]

Phase 2 - Benchmark tuning                  17:00-17:46  (~46 min active)
  User messages: adjusting target durations (2m, 3m, 5m), checking variance,
  documenting the benchmark script and testing protocol
    [idle 67 min - user away]

Phase 3 - Re-benchmark on direct TTY        18:53-19:37  (~44 min active)
  User messages: re-running benchmarks after exiting GUI session for
  cleaner measurements, storing baseline stats
    [idle 30 min]

Phase 4 - Autonomous optimization            20:08-20:53  (~45 min active)
  User said "please continue" and left Codex to execute the optimization
  queue autonomously. Codex ran through O07-O20, building/benchmarking
  each variant, classifying results, and making keep/drop decisions.
    [idle 6h 18min - Codex finished, user was away overnight]

Phase 5 - User returned                     03:11-03:16  (~5 min active)
  User reviewed results, asked for README and before/after comparison.
```

#### Hourly Event Density

```
15:00 UTC  ████████████████████████████████████  703 events  (setup + scaffolding)
17:00 UTC  ████████████████████████████████████████████████  955 events  (benchmark tuning)
18:00 UTC  ██████  111 events  (re-bench start)
19:00 UTC  ███████████████████████████████████████████████  940 events  (benchmarking)
20:00 UTC  █████████████████████████████████████████████████████████  1143 events  (autonomous opt!)
03:00 UTC  █████  92 events  (user return)
```

#### Key Observation

The "overnight" autonomous optimization was actually only **~45 minutes** of active work. Codex completed all of O07-O20 (14 optimization experiments including build, benchmark, classify, and decide for each) in under an hour, then sat idle until the user returned 6+ hours later. The peak activity was in the 20:00 UTC hour with 1,143 events -- the densest period of the entire session.

### 3. Claude Code: Post-Optimization Documentation

After the Codex optimization work completed, a Claude Code session (`af1c788b`) was used to write the project README, synthesize results into the HIP-vs-HLSL comparison table, and draft the real-world implications sections. This session also included other unrelated work, so the numbers below overcount somewhat.

| Metric | Value |
|---|---|
| **Session ID** | `af1c788b-c174-4d2f-bf69-56e2d5ebda85` |
| **Model** | `claude-opus-4-6` |
| **Wall time** | ~20 min (of relevant work) |
| **Turns** | ~54 user / ~64 assistant |
| **Total API tokens** | ~4M (mostly prompt cache reads) |

### Supporting Codex Sessions

Several shorter sessions were used for setup, environment fixes, and post-optimization exploration:

| Session ID | Duration | Model | Purpose | Tokens |
|---|---|---|---|---|
| `019c9a81` | 1m 56s | gpt-5.3-codex-spark | Initial environment probe | 475K |
| `019c9a85` | 1m 58s | gpt-5.3-codex | First benchmark attempt | 530K |
| `019c9a8b` | 27s | gpt-5.3-codex | AGENTS.md env update | 52K |
| `019c9b4d` | 31s | gpt-5.3-codex | Brief env continuation | 71K |
| `019c9d1e` | 9m 29s | gpt-5.3-codex | Post-opt DX12/Linux compile discussion | 5.03M |

### Aggregate Token Usage

| Tool | Sessions | Total Tokens | Notes |
|---|---|---|---|
| Claude Code (pre-analysis) | 1 | ~1.76M | Source code analysis, produced ANALYSIS.md |
| Codex CLI (all fsr4) | 6 | ~61.9M | Dominated by 55.7M from main session |
| Claude Code (post-docs) | 1 | ~4M | README/docs synthesis, mostly cache reads |
| **Combined** | **8** | **~67.7M** | |

The vast majority of tokens (>96%) were Codex input tokens, which in turn were >96% prompt cache hits. The actual novel output from all tools was relatively modest (~200K from Codex, ~31K from Claude) -- the token cost is dominated by maintaining context across hundreds of tool-call turns.

### Timeline Summary

```
Feb 26  14:53 UTC  ── Claude Code: FSR4 source analysis ──────  15:09 UTC  (16 min)
        15:11 UTC  ── Codex: setup sessions (3 short) ────────  15:25 UTC  (14 min)
        15:26 UTC  ── Codex: main optimization session ───────────────────────────
                      (interactive phases with idle gaps)
        20:08 UTC     └─ autonomous optimization ─────────────  20:53 UTC  (45 min)
                        (idle 6h 18m - user away overnight)
Feb 27  03:11 UTC     └─ user returned, reviewed results ────  03:16 UTC  (5 min)
        03:18 UTC  ── Claude Code: documentation synthesis ───  03:38 UTC  (20 min)
```

Total active AI compute time: **~3h 10m** (16m analysis + 2h 34m optimization + 20m docs).

### What This Tells Us About AI-Assisted Optimization

1. **Analysis-first approach paid off**: The 16-minute Claude pre-analysis of the FSR4 source code identified the key optimization targets (INT8 dot product paths, FP8 conversion overhead, quantization policy) before any code was written. This front-loaded understanding meant the Codex optimization session could work from a structured plan rather than discovering the codebase as it went.

2. **The overnight run was fast**: 14 optimization experiments in ~45 minutes of autonomous work. Each experiment involved modifying C++ source, compiling a HIP kernel, running a 60-second benchmark, parsing results, classifying keep/drop, and updating documentation. A human doing this manually would need hours.

3. **Setup was the bottleneck**: Of the 2h 34m of active Codex time, about 1h 44m was interactive setup and benchmarking -- getting the environment right, tuning benchmark parameters, establishing stable baselines. The actual optimization execution was the fastest phase.

4. **Token costs are dominated by context**: 55.7M tokens for the main Codex session sounds large, but 53.6M were prompt cache hits (cheap). The actual model reasoning was 74.9K tokens. Long-running tool-heavy sessions amortize context loading across many operations.

5. **Human-AI handoff worked well**: The developer set up the environment, validated the methodology, then delegated execution. The AI autonomously made sound keep/drop decisions (verified by the developer upon return). The conservative decision gate (`min_uncertainty_pct=3%`) meant the AI erred toward caution rather than accepting noise as signal.
