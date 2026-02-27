# ANALYSIS-HLSL: FSR4 INT8 Shader Implementation vs HIP Harness

## Overview

This document analyzes the actual FSR4 INT8 HLSL shader implementation, compares it to our HIP microkernel harness, and estimates where our optimization findings could (and could not) apply to the real codebase.

**Key takeaway**: The FSR4 INT8 path uses two distinct dot-product strategies depending on the operator. Pass 0 (the input downscale) uses FP16 `dot2add`, while passes 1-13 are dominated by **native INT8 `dot4add_i8packed`** (including pass 13's `CNB_CT2D` compute path before FP16 writeback). This is much closer to what our HIP harness benchmarks.

## FP8 vs INT8: Why We Focus on INT8

| Aspect | INT8 | FP8 |
|---|---|---|
| HIP harness speed | 0.005376 ms | 0.019868 ms |
| Ratio | 1.0x (baseline) | 3.7x slower |
| Real HLSL approach | Mostly `dot4add_i8packed` + boundary `dot2add` | `AmdWaveMatrixMultiply` (WMMA, wave-level matrix ops) |
| WMMA required? | No | **Yes** -- FP8 HLSL has `#error` without `WMMA_ENABLED=1` |
| LDS required? | No | **Yes** -- `groupshared uint inputLDS[]` for wave matrix input staging |
| Our harness relevance | High -- same instruction class | Low -- completely different compute model |

The FP8 path (`float8_NHWC/Conv2D_k2s2b.hlsli:217`) explicitly errors without WMMA: `#error To use FP8 data type you need to provide WMMA_ENABLED=1. There is no FP8 without WMMA.` This means FP8 requires wave-level matrix operations and LDS staging, which our scalar FMA harness does not exercise at all.

In the `i8_balanced` model, `WMMA_ENABLED` is set to `0` and `DOT4_ENABLED` is set to `1` (see `passes_1080.hlsl:8-10`), confirming the INT8 DOT4 path is the primary execution path.

**Relevant source files**:
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/float8_NHWC/Conv2D_k2s2b.hlsli` -- FP8 WMMA-only path
- `fsr4-src/baseline/internal/shaders/fsr4_model_v07_i8_balanced/passes_1080.hlsl:8-10` -- `WMMA_ENABLED=0`, `DOT4_ENABLED=1`

## The INT8 Execution Pipeline (1080p, Balanced)

The model executes 14 sequential compute passes (0-13). Passes 0-12 are followed by padding reset post-passes; pass 13 has no post-pass. The pipeline forms a U-Net: encoder downsamples spatially while increasing channels, bottleneck processes at lowest resolution, decoder upsamples back.

Source: `fsr4-src/baseline/internal/shaders/fsr4_model_v07_i8_balanced/passes_1080.hlsl`

| Pass | Layer | Operator | Spatial Dims | Channels | Threads | Dot Instruction | Line |
|---:|---|---|---|---:|---|---|---:|
| 0 | encoder1 downscale | `Conv2D_k2s2b` | 1920x1080 -> 960x540 | 7 -> 16 | (8,8,1) | `dot2add` (FP16) | 94 |
| 1 | encoder2 ResBlock_0 | `ConvNextBlock` | 960x540 | 16 -> 16 | (64,1,1) | `dot4add_i8packed` | 421 |
| 2 | encoder2 ResBlock_1 | `ConvNextBlock` | 960x540 | 16 -> 16 | (64,1,1) | `dot4add_i8packed` | 797 |
| 3 | encoder2 downscale | `FusedConv2D_k2s2b_QuantizedOutput` | 960x540 -> 480x270 | 16 -> 32 | (64,1,1) | `dot4add_i8packed` | 1081 |
| 4 | encoder3 ResBlock_0 | `FasterNetBlock<32,1>` | 480x270 | 32 -> 32 | (64,1,1) | `dot4add_i8packed` | 1605 |
| 5 | encoder3 ResBlock_1 | `FasterNetBlock<32,1>` | 480x270 | 32 -> 32 | (64,1,1) | `dot4add_i8packed` | 2177 |
| 6 | encoder3 downscale | `FusedConv2D_k2s2b_QuantizedOutput` | 480x270 -> 240x135 | 32 -> 64 | (64,1,1) | `dot4add_i8packed` | 2335 |
| 7 | bottleneck ResBlock_0 | `FasterNetBlock<64,2>` | 240x135 | 64 -> 64 | (64,1,1) | `dot4add_i8packed` | 2466 |
| 8 | bottleneck ResBlock_1 | `FasterNetBlock<64,2>` | 240x135 | 64 -> 64 | (64,1,1) | `dot4add_i8packed` | 2643 |
| 9 | bottleneck ResBlock_2 + upscale + skip-add | `FNB_CT2D_ADD<64,2>` | 240x135 -> 480x270 | 64 -> 32 | (8,8,1) | `dot4add_i8packed` | 2826 |
| 10 | decoder3 ResBlock_1 | `FasterNetBlock<32,1>` | 480x270 | 32 -> 32 | (64,1,1) | `dot4add_i8packed` | 3430 |
| 11 | decoder3 ResBlock_2 + upscale + skip-add | `FNB_CT2D_ADD<32,1>` | 480x270 -> 960x540 | 32 -> 16 | (64,1,1) | `dot4add_i8packed` | 4136 |
| 12 | decoder2 ResBlock | `ConvNextBlock` | 960x540 | 16 -> 16 | (64,1,1) | `dot4add_i8packed` | 4548 |
| 13 | decoder2 upscale (FP16 output) | `CNB_CT2D` (float16 output) | 960x540 -> 1920x1080 | 16 -> 8 | (8,8,1) | `dot4add_i8packed` (INT8 internal compute) | 4962 |

**Spatial progression**: 1920x1080 -> 960x540 -> 480x270 -> 240x135 -> 480x270 -> 960x540 -> 1920x1080

**Channel progression**: 7 -> 16 -> 32 -> 64 -> 32 -> 16 -> 8 (RGB output)

## The Two Dot-Product Paths

### Path A: FP16 `dot2add` (Pass 0 boundary pass)

Used for the entry pass where FP16 frame input is quantized into INT8 activations.

**Source**: `int8_NHWC/Conv2D_k2s2b.hlsli:60-65` (specialized `Conv_8_16_k2s2b`)

```hlsl
// Load FP16 inputs and weights
half4 inputValues[2*4];                              // Pre-loaded FP16 register array
const uint4 inputDwords = input.storage.Load4(inputOffset);
inputValues[nInput++] = Unpack4h(inputDwords.xy);    // Unpack uint -> 4x half

// FP16 dot product with FP32 accumulation
accumulators[f] = dot2add(ws.xy, inputValues[nInput].xy, accumulators[f]);
accumulators[f] = dot2add(ws.zw, inputValues[nInput++].zw, accumulators[f]);

// Quantize to INT8 at store
int16_t4 quantized_vs = int16_t4(round((vs + Unpack4h(biasDwords.xy)) * rcpScale));
storageBytes.x = pack_clamp_s8(quantized_vs);
```

**Data flow**: FP16 input -> `Unpack4h` -> `dot2add(half2, half2, float)` -> FP32 accumulator -> `round(x * rcpScale)` -> `pack_clamp_s8` -> INT8 output

**Why FP16**: The input tensor is FP16 (the raw frame data). Pass 0 converts from FP16 to INT8, so it cannot use an INT8xINT8 dot path at input.

### Path B: Native INT8 `dot4add_i8packed` (Passes 1-13)

Used for the internal network passes and also pass 13's fused CNB+CT2D compute stages. This is the **dominant compute pattern**.

**Source**: `int8_NHWC/Fused/ConvNextBlock.hlsli:95-98` (representative example)

```hlsl
// Load packed INT8 inputs (16 bytes = 16 int8 values per Load4)
int8_t4_packed vs[16/4];
const uint4 inputDwords = input.storage.Load4(inputOffset);
vs[inputIndex++] = inputDwords.x;   // 4 packed INT8 values
vs[inputIndex++] = inputDwords.y;
vs[inputIndex++] = inputDwords.z;
vs[inputIndex++] = inputDwords.w;

// Native INT8 dot product: 4x INT8 multiply + INT32 accumulate, per instruction
accumulator[f] = dot4add_i8packed(vs[inputIndex++], weightsDwords.x, accumulator[f]);
accumulator[f] = dot4add_i8packed(vs[inputIndex++], weightsDwords.y, accumulator[f]);
accumulator[f] = dot4add_i8packed(vs[inputIndex++], weightsDwords.z, accumulator[f]);
accumulator[f] = dot4add_i8packed(vs[inputIndex++], weightsDwords.w, accumulator[f]);

// Scale and quantize at store
const int16_t4 result = round(acc * weights.quantizationScale *
                              input.quantizationScale * (1.0 / quantFactor));
storeDwords[f/4] = pack_clamp_s8(result);
```

**Data flow**: INT8 packed input -> `dot4add_i8packed(uint, uint, int)` -> INT32 accumulator -> float scale multiply -> `round()` -> `pack_clamp_s8` -> INT8 output

**Critical observation**: `dot4add_i8packed` is the HLSL equivalent of our HIP `amd_mixed_dot` -- both perform a packed 4-element INT8 dot product with INT32 accumulation. This is the same instruction class we benchmarked, and it dominates 13 of 14 passes.

### Instruction Mapping: HIP vs HLSL

| HIP Harness | HLSL Equivalent | Used In |
|---|---|---|
| `amd_mixed_dot(char4, char4, int32_t, false)` | `dot4add_i8packed(uint, uint, int)` | Passes 1-13 (dominant INT8 compute path) |
| Scalar `acc += av0*bv0 + av1*bv1 + av2*bv2 + av3*bv3` | No HLSL equivalent used | N/A (our optimization) |
| N/A | `dot2add(half2, half2, float)` | Pass 0 (FP16 input quantization) |

**Key source files for Path B operators**:
- `int8_NHWC/Fused/ConvNextBlock.hlsli` -- 3x3 spatial stage + 1x1 PW expand + 1x1 PW contract, all `dot4add_i8packed`
- `int8_NHWC/Fused/FasterNetBlock.hlsli` -- Same as ConvNextBlock but with grouped convolution and activation fusion
- `int8_NHWC/Fused/FusedConv2D_k2s2b_QuantizedOutput.hlsli` -- Strided 2x2 downscale, `dot4add_i8packed`
- `int8_NHWC/Fused/FNB_CT2D_ADD.hlsli` -- FasterNetBlock + ConvTranspose2D upscale + skip-connection add
- `float16_NHWC/Fused/CNB_CT2D.hlsli` -- Pass 13 fused CNB + CT2D path, `dot4add_i8packed` internal compute
- `int8_NHWC/Conv2D.hlsli:279-331` -- Generic quantized INT8 Conv2D template path (uses `dot4add_i8packed` when selected)

## Operator-by-Operator Analysis

### ConvNextBlock (Passes 1, 2, 12)

**Source**: `int8_NHWC/Fused/ConvNextBlock.hlsli`

Fuses three convolutions into a single dispatch:
1. **3x3 spatial conv stage** (weights0) -- `dot4add_i8packed`, lines 86-99
2. **1x1 pointwise expand** (weights1) -- `dot4add_i8packed`, lines 107-119 (after quantize+ReLU)
3. **1x1 pointwise contract** (weights2) -- `dot4add_i8packed`, lines 130-143 (after quantize+ReLU)
4. **Residual add** -- with skip connection input

**Inner loop structure** (3x3 spatial stage, `ConvNextBlock.hlsli:64-99`):
```
[unroll] for ky in 0..2:          // kernel height
  [unroll] for kx in 0..2:        // kernel width
    if (ValidPosition):            // boundary check
      load 16 INT8 values (Load4)
      [unroll] for f in 0..15:     // features
        load 16 weight values (Load4)
        4x dot4add_i8packed       // 16 INT8 MACs per feature
```

Each `dot4add_i8packed` processes 4 INT8 multiplies + 1 INT32 accumulate. With 16 input channels loaded per spatial position and 16 features, this is 16 * 9 * 16 = 2,304 INT8 MACs per output pixel for this stage.

**Quantization between fused ops** (`ConvNextBlock.hlsli:101-106`):
```hlsl
const int16_t4 r = round(int4(...) * weights0.quantizationScale *
                         input.quantizationScale * rcprOutputScale0);
int8_t4_packed quantized = pack_clamp_s8(r);
// ReLU: max(0, quantized)
conv_output[f] = max(pack_clamp_s8(int4(0,0,0,0)), quantized);
```
Requantization happens **between** each fused convolution, not per-iteration. This matches our finding (O08) that store-time quantization is correct.

### FasterNetBlock (Passes 4, 5, 7, 8, 10)

**Source**: `int8_NHWC/Fused/FasterNetBlock.hlsli`

Similar to ConvNextBlock but with a split-channel (grouped) convolution structure. Template parameters control channel count:
- `FasterNetBlock<32, 1>` for 32-channel layers (passes 4, 5, 10)
- `FasterNetBlock<64, 2>` for 64-channel layers with groups=2 (passes 7, 8)

**First convolution** (3x3 grouped spatial conv, `FasterNetBlock.hlsli:82-117`):
```hlsl
// Preload input channels
int preloadedInputs[numFeatures/4/2];  // Pre-staged in registers
...
// Load INT8 weights directly (no FP16 conversion)
int8_t4_packed weights = Load4i8A(weights0, uint4(kx, ky, c, f));
conv_accumulator[f] = dot4add_i8packed(vs[inputIndex++], weights, conv_accumulator[f]);
```

The grouped convolution means the first convolution operates on only `numFeatures/2` channels, then the result is concatenated with the untouched second half before the 1x1 pointwise expand.

**Data pre-loading** (lines 52-78): Input is loaded into register arrays before the convolution loop, enabling data reuse across the feature loop. This is a register-based staging strategy (not LDS).

### FusedConv2D_k2s2b_QuantizedOutput (Passes 3, 6)

**Source**: `int8_NHWC/Fused/FusedConv2D_k2s2b_QuantizedOutput.hlsli`

Downscale convolution with fused quantization output. Contains multiple specialized fast paths:

| Fast Path | Input Channels | Output Features | Source Line |
|---|---:|---:|---:|
| `Conv_16_32` | 16 | 32 | 12 |
| `Conv_16_48` | 16 | 48 | (after Conv_16_32) |
| `Conv_32_64` | 32 | 64 | (after Conv_16_48) |
| `Conv_48_64` | 48 | 64 | (after Conv_32_64) |

**Conv_16_32 inner loop** (`FusedConv2D_k2s2b_QuantizedOutput.hlsli:34-59`):
```hlsl
int accumulator[32];                               // 32 INT32 accumulators
...
[unroll] for ky in 0..1:                           // 2x2 kernel
  [unroll] for kx in 0..1:
    load 16 INT8 inputs (Load4 -> 4 packed dwords)
    [unroll] for f in 0..31:                       // all output features
      load 16 INT8 weights (Load4)
      4x dot4add_i8packed                          // 16 INT8 MACs per feature
```

**Split quantization** (`FusedConv2D_k2s2b_QuantizedOutput.hlsli:62-83`): Output is split into two halves, each quantized with a different scale factor (`quantFactor0` and `quantFactor1`). This is a per-group quantization scheme, more granular than a single per-tensor scale.

### FNB_CT2D_ADD (Passes 9, 11)

**Source**: `int8_NHWC/Fused/FNB_CT2D_ADD.hlsli`

The most complex fused operator: FasterNetBlock + ConvTranspose2D (upscale) + skip-connection Add, all in one dispatch. This bridges between resolution levels in the decoder.

**Operations fused**:
1. 3x3 grouped spatial conv (dot4add_i8packed)
2. Quantize + ReLU
3. 1x1 PW expand (dot4add_i8packed)
4. Quantize + ReLU
5. 1x1 PW contract (dot4add_i8packed)
6. Residual add
7. 2x2 ConvTranspose (upscale, dot4add_i8packed)
8. Skip-connection add from encoder
9. Final quantization

**ConvTranspose inner loop** (see `ConvTranspose2D_k2s2b.hlsli:58-69`):
```hlsl
// dot4add_i8packed used for transposed convolution too
accumulator.x = dot4add_i8packed(w0.x, packedVs.x, accumulator.x);
accumulator.x = dot4add_i8packed(w0.y, packedVs.y, accumulator.x);
accumulator.x = dot4add_i8packed(w0.z, packedVs.z, accumulator.x);
accumulator.x = dot4add_i8packed(w0.w, packedVs.w, accumulator.x);
```

### Conv2D_k2s2b (Pass 0 only)

**Source**: `int8_NHWC/Conv2D_k2s2b.hlsli`

The entry convolution that converts FP16 input to INT8. Uses `dot2add` because input is FP16 (the raw frame data), not INT8. Contains specialized fast paths:

- `Conv_8_16_k2s2b` (line 10): For 8-channel FP16 input -> 16 INT8 output (the actual pass 0 shape: 7 logical channels, padded to 8 storage)
- `Conv_10_16_k2s2b` (line 184): For 10-channel input
- Generic fallback via `TemplatedConv2D_3413_NHWC_44` (line 360)

### Conv2D generic (Template path, not used by generated i8 balanced passes 1-13)

**Source**: `int8_NHWC/Conv2D.hlsli:279-331`

When called with `QuantizedTensor3i8_NHWC` inputs/outputs, the Dot function resolves to:
```hlsl
int Dot(int8_t4_packed _ws, int8_t4_packed _vs, int acc) {
    return dot4add_i8packed(_ws, _vs, acc);
}
```
This feeds into `TemplatedConv2D_3413_NHWC_44` (`templates/Conv2D.hlsli:108-155`) which has the standard spatial tiling + feature loop with `[unroll]` on all dimensions. In this specific generated `passes_1080.hlsl`, passes 1-13 use fused operators instead of this generic Conv2D template.

## Applying Our Optimization Findings

### O06: Scalar INT8 > Packed `amd_mixed_dot` (32% gain in harness)

**Our finding**: Scalar element-wise multiply-accumulate outperformed `amd_mixed_dot` (the packed INT8 dot) by ~32% on gfx1151.

**HLSL relevance**: **HIGH for passes 1-13**. The real HLSL uses `dot4add_i8packed`, which maps to the same hardware instruction as `amd_mixed_dot`. If scalar element-wise is genuinely faster on gfx1151, then there may be an optimization opportunity for the ML2Code code generator to emit scalar INT8 multiplies instead of packed dot products.

**Potential impact**: Passes 1-13 account for 13 of 14 passes and contain the vast majority of compute. If the 32% scalar advantage holds for the actual convolution workload (not just our synthetic loop), this would be a significant improvement across the entire network.

**Caveats**:
- Our harness loop is simpler (single inner loop, flat arrays). The real HLSL has nested spatial + feature loops with memory access patterns that may affect instruction scheduling differently.
- The real code loads 16 bytes per `Load4` and processes 4 dwords sequentially through `dot4add_i8packed`. A scalar replacement would need to unpack each dword into 4 individual bytes, potentially increasing register pressure.
- DXC (the HLSL compiler) may optimize `dot4add_i8packed` differently than hipcc optimizes `amd_mixed_dot`. The hardware instruction may be the same, but the surrounding code generation could differ.

**How to test**: Modify one of the `ConvNextBlock` or `FasterNetBlock` operators to replace:
```hlsl
accumulator[f] = dot4add_i8packed(weightsDwords.x, vs[0], accumulator[f]);
```
with:
```hlsl
int4 w = unpack_s8s32(weightsDwords.x);
int4 v = unpack_s8s32(vs[0]);
accumulator[f] += w.x*v.x + w.y*v.y + w.z*v.z + w.w*v.w;
```
Then benchmark the modified shader on gfx1151.

**Relevant files to modify**:
- `int8_NHWC/Fused/ConvNextBlock.hlsli:95-98`
- `int8_NHWC/Fused/FasterNetBlock.hlsli:113-114`
- `int8_NHWC/Fused/FusedConv2D_k2s2b_QuantizedOutput.hlsli:54-57`

### O05: Compile-Time Loop Unrolling (12% gain in harness)

**Our finding**: Compile-time unrolled inner loops outperformed runtime loop control by ~12%.

**HLSL relevance**: **Already applied**. Every operator uses `[unroll]` on kernel spatial loops (kx, ky), channel loops, and feature loops. The HLSL compiler (DXC with `-O3`) will expand these at compile time.

**Potential concern**: Some loops are NOT marked `[unroll]` -- notably the first convolution spatial loop in `FasterNetBlock.hlsli:91-92`:
```hlsl
for (uint ky = 0; ky < weights0.logicalSize.y; ++ky)   // NOT [unroll]
    for (uint kx = 0; kx < weights0.logicalSize.x; ++kx)
```
The compiler may or may not unroll these since `weights0.logicalSize` is a runtime value (set from tensor metadata). If it doesn't unroll, this could be leaving performance on the table.

**How to test**: Add `[unroll]` to the first-conv spatial loop in FasterNetBlock and benchmark. If kernel dimensions are always 3x3 (they are for this model), the loop count is always known and unrolling is safe.

**Relevant files**:
- `int8_NHWC/Fused/FasterNetBlock.hlsli:91-92` -- missing `[unroll]` on first conv spatial loops

### O08: Store-Time Quantization (per-iter requant = +194% INT8 regression)

**Our finding**: Per-iteration requantization catastrophically regresses performance. Accumulating in full precision and quantizing once at output is essential.

**HLSL relevance**: **Already correctly implemented**. All operators accumulate in INT32 (for `dot4add_i8packed` path) or FP32 (for `dot2add` path) and only quantize at the very end of each convolution via `pack_clamp_s8(round(acc * scale))`.

Between fused operations (e.g., between the DW conv and PW expand inside a ConvNextBlock), there IS a requantization step -- but this is semantically required by the model architecture (it's a distinct quantized tensor boundary). Our finding confirms that the ML2Code generator should never insert gratuitous requantization within a single convolution's accumulation loop.

**Relevant examples**:
- `ConvNextBlock.hlsli:101-106` -- requant between DW and PW (correct, architecturally required)
- `FusedConv2D_k2s2b_QuantizedOutput.hlsli:62-83` -- store-time quant only (correct)
- `ConvTranspose2D_k2s2b.hlsli:120-130` -- store-time quant only (correct)

### O10-O13: LDS Staging (all regressed)

**Our finding**: All four LDS staging variants regressed or showed unacceptable variance on Strix Halo iGPU.

**HLSL relevance**: **The INT8 path does NOT use LDS**. Data staging in the real INT8 code is purely register-based:
- `ConvNextBlock.hlsli:73-84` -- input loaded into `int8_t4_packed vs[16/4]` register array
- `FasterNetBlock.hlsli:66-78` -- input pre-loaded into `int preloadedInputs[]` register array
- `Conv2D_k2s2b.hlsli:22-36` -- input pre-loaded into `half4 inputValues[2*4]` register array

This means our LDS regression finding does NOT affect the INT8 path. However, it IS relevant to the FP8/WMMA path, which uses `groupshared uint inputLDS[]` (see `float8_NHWC/Conv2D_k2s2b.hlsli:15`).

**Implication**: The INT8 path's register-based staging strategy is a better fit for Strix Halo's memory subsystem. If someone were to "optimize" the INT8 path by adding LDS staging (a common GPU optimization technique), our results strongly suggest it would regress on iGPU.

### O02: Thread Block Size (256 optimal in harness)

**Our finding**: 256 threads per block (8 waves of 32) was optimal. 64 and 128 were slower.

**HLSL relevance**: **Mixed**. The real shaders use two dispatch patterns:
- `numthreads(64, 1, 1)` -- 2 waves, used for most internal passes (1-8, 10-12)
- `numthreads(8, 8, 1)` -- 64 threads = 2 waves, used for passes 0, 9, 13

Neither uses 256 threads. The HLSL code uses 64 threads for a different reason than our harness: each thread processes one spatial position across all features, rather than processing one element of a flat array. The work distribution model is fundamentally different.

**However**: Our finding that 64 threads was slower than 256 could indicate insufficient occupancy at 2 waves. Testing with `numthreads(128, 1, 1)` or `numthreads(256, 1, 1)` (4 or 8 waves) on the FasterNetBlock/ConvNextBlock operators might improve occupancy on gfx1151, especially for the smaller bottleneck passes (7, 8) where each dispatch has fewer thread groups.

**Relevant dispatch calculations**: For pass 7 (bottleneck, 240x135, 64 channels), with numthreads(64,1,1):
- Thread groups: ~240 * 135 / (perThreadWork) -- depends on SplitWork calculation
- More threads per group would mean fewer groups but better occupancy per CU

### O09: Interior/Edge Split Dispatch (Unsure)

**Our finding**: Small INT8 gain, FP8 too noisy to call.

**HLSL relevance**: **Partially applied**. The template `Conv2D.hlsli` already has separate padded and unpadded paths:
```hlsl
// templates/Conv2D.hlsli:124-154
if (beginX != 0 || beginY != 0)
    // Padded Case -- includes ValidPosition checks inside inner loop
    TemplatedInnerConvPadded3413_44<...>(...)
else
    // Unpadded Case -- no boundary checks inside inner loop
    TemplatedInnerConv3413_44<...>(...)
```

The fused operators (ConvNextBlock, FasterNetBlock) do NOT split interior/edge -- they check `ValidPosition` inside the convolution loop for every pixel (`ConvNextBlock.hlsli:70-71`):
```hlsl
if (!ValidPosition(input, int3(piBase.xy + int2(kx, ky), 0)))
    continue;
```

This branch inside the hot loop could be eliminated for interior pixels. Given our harness showed a small INT8 gain from splitting, applying this to the fused operators might yield a similar modest improvement, especially for the large 960x540 passes where >99% of pixels are interior.

### O07: Hoisting Scale/Bias Loads (in-loop variant much slower)

**Our finding**: Loading scale/bias inside the inner loop was much slower than hoisting.

**HLSL relevance**: **Already correctly hoisted**. In all operators, bias is loaded once before the feature loop (e.g., `ConvNextBlock.hlsli:44-51`), and quantization scales are stored as float parameters, not reloaded per iteration.

### O14-O15: Occupancy/Register Pressure and Compiler Flags

**Our finding**: No confidence-grade improvement from register tuning. `-O3` matched or beat `-O2` and `-Ofast`.

**HLSL relevance**: The HLSL compilation command in `passes_1080.hlsl:3` already specifies `-O3`:
```
// Compile with dxc.exe -no-warnings -O3 -enable-16bit-types -HV 2021 -T cs_6_6
```
Our finding confirms this is the right choice.

**Register pressure concern**: The fused operators (FasterNetBlock, FNB_CT2D_ADD) maintain large register arrays:
- `int conv_accumulator[numFeatures/2]` -- 16 or 32 INT32 values
- `int8_t4_packed conv_result[numFeatures/4]` -- 8 or 16 packed values
- `int relu_output[numFeatures]` -- 32 or 64 INT32 values (for expand stage)

For `FasterNetBlock<64, 2>` (passes 7, 8), this is 32 + 16 + 64 = 112 registers just for accumulators, plus weights and input staging. On gfx1151 with its VGPR budget, this may limit occupancy. Our O14 finding that occupancy tuning didn't help could mean either (a) occupancy is already at an acceptable level, or (b) the benefit of keeping data in registers outweighs the occupancy loss.

## Estimated Impact Summary

| Optimization | Applicable? | Passes Affected | Estimated Real Impact | Confidence |
|---|---|---|---|---|
| O06: Scalar > packed INT8 | Yes | 1-13 (~93% of pipeline) | Likely ~5-15% per-pass; stretch ~20-25%; hard upper bound ~32% | Medium -- needs DXC codegen validation |
| O05: Loop unrolling | Partially -- some loops missing `[unroll]` | 4, 5, 7, 8, 10 (FasterNetBlock first conv loops) | ~5-12% on affected passes | High -- well-understood optimization |
| O08: Store-time quant | Already correct | N/A | N/A (would be catastrophic if violated) | High |
| O10-O13: Avoid LDS staging | INT8 already avoids LDS | N/A for INT8 | Validates current register-based approach | High |
| O09: Interior/edge split | Applicable to fused ops | 1, 2, 4, 5, 7, 8, 10, 12 | ~1-3% (branch removal in hot loop) | Low-Medium |
| O02: Thread block sizing | Testable | Most internal passes | Unknown -- different work distribution | Low |

## Expected Gain Ranges (gfx1151)

Based on the microbenchmarks and the HLSL pass mapping:

- **Instruction-level upper bound**: ~32% on kernels where packed INT8 dot math dominates and scalarized replacements compile efficiently.
- **Most likely early result in real HLSL dispatches**: **~5-15%** on affected passes after first scalar-vs-packed A/B.
- **Stretch target** (if DXC codegen, VGPR pressure, and occupancy all cooperate): **~20-25%** on key passes.

Why expected real gains are lower than the microbenchmark peak:

- The microbenchmark isolates inner math, while real HLSL includes tensor loads/stores, boundary checks, quantization, and fused epilog/prolog work.
- Scalar replacements may increase unpacking overhead and register pressure, which can reduce occupancy and hide less latency.
- Compiler behavior (DXC ISA generation) can materially change outcomes even when the source-level operation appears equivalent.

Frame-level implication:

- If FSR4 consumes fraction `S` of frame time, and FSR4 stage speedup is `G`, approximate frame uplift is `S * G`.
- Example: if FSR4 is 40% of frame time and FSR4 stage speedup is 10%, frame uplift is about 4%.

## Priority Recommendations

1. **Benchmark scalar vs `dot4add_i8packed` in DXC-compiled shaders on gfx1151**. This is the highest-impact finding. Modify `ConvNextBlock.hlsli` to use unpacked scalar INT8 multiply-accumulate and compare dispatch times. If the 32% advantage holds, it applies to 13 of 14 passes.

2. **Add `[unroll]` to first-conv spatial loops in FasterNetBlock**. The outer `for ky/kx` loops at `FasterNetBlock.hlsli:91-92` lack `[unroll]` annotations. Since kernel dimensions are always 3x3 for this model, this is a safe, low-risk optimization that could yield ~5-12% on passes 4, 5, 7, 8, 10.

3. **Profile per-pass dispatch time on Strix Halo**. With 14 sequential dispatches + 13 padding post-passes = 27 total dispatches per frame, the dispatch overhead may be significant relative to compute time on iGPU. Our harness measured ~5us per INT8 kernel invocation; 27 dispatches at ~5us each = ~135us of dispatch overhead alone.

4. **Test larger thread group sizes for bottleneck passes**. Passes 7-8 operate on 240x135 spatial dims with 64 channels. With `numthreads(64,1,1)`, occupancy may be low. Testing `numthreads(128,1,1)` or `numthreads(256,1,1)` could improve occupancy on gfx1151's wave32 architecture.

5. **Never add LDS staging to the INT8 path**. Our harness conclusively showed LDS is slower on Strix Halo iGPU. The current register-based input staging strategy is correct for this hardware.

## File Reference Index

### Model Pass Definitions
- `fsr4-src/baseline/internal/shaders/fsr4_model_v07_i8_balanced/passes_1080.hlsl` -- Complete 14-pass pipeline, 5090+ lines

### INT8 Operators (non-fused)
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Conv2D.hlsli` -- Generic Conv2D with 3 template specializations (FP16, INT8, UINT8 inputs)
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Conv2D_k2s2b.hlsli` -- Strided 2x2 Conv2D with fast paths for 8/10-channel inputs
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Conv2D_k3p1b.hlsli` -- 3x3 padded Conv2D (FP16 dot2add "slow path")
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Conv2D_k3p1gb.hlsli` -- 3x3 padded grouped Conv2D
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/ConvTranspose2D_k2s2b.hlsli` -- 2x2 transposed Conv (upscale), `dot4add_i8packed`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Add.hlsli` -- Element-wise add with mixed quantization
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Relu.hlsli` -- ReLU activation on INT8
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/QuantizeLinear.hlsli` -- FP16 -> INT8 quantization
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Concat.hlsli` -- Channel concatenation
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Pad.hlsli` -- Constant padding

### INT8 Fused Operators
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/ConvNextBlock.hlsli` -- 3x3 spatial stage + PW expand + PW contract + residual add
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FasterNetBlock.hlsli` -- Grouped ConvNextBlock variant
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FNB_CT2D_ADD.hlsli` -- FasterNetBlock + upscale + skip-add
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FusedConv2D_k2s2b_QuantizedOutput.hlsli` -- Strided downscale + fused quant output
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FusedConv2D_k1b_Relu.hlsli` -- 1x1 Conv + ReLU fusion
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FusedConvTranspose2D_k2s2b_Add.hlsli` -- ConvTranspose + add
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FusedAdd_QuantizedOutput.hlsli` -- Add + quantize output

### Template Infrastructure
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/templates/Conv2D.hlsli` -- Generic Conv2D template with padded/unpadded/grouped variants
- `fsr4-src/baseline/dx12/ml2code_runtime/tensor_int8.hlsli` -- INT8 tensor type definitions, pack/unpack functions
- `fsr4-src/baseline/dx12/ml2code_runtime/tensor_float16.hlsli` -- FP16 tensor types, `Unpack4h`, `dot2add`
- `fsr4-src/baseline/dx12/ml2code_runtime/scalar_functions.hlsli` -- `SplitWork`, `ValidPosition`, utility functions
- `fsr4-src/baseline/dx12/ml2code_runtime/storage.hlsli` -- `BufferStorage`, `RWBufferStorage`, `ConstantBufferStorage`

### FP8 Path (for contrast)
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/float8_NHWC/Conv2D_k2s2b.hlsli` -- FP8 WMMA-only Conv2D
- `fsr4-src/baseline/dx12/ml2code_runtime/tensor_float8.hlsli` -- FP8 tensor types

### HIP Harness (our benchmarks)
- `benchmarks/baseline_kernels_bench.cpp:288-313` -- `int8_dot4_kernel_unrolled` (packed `amd_mixed_dot`)
- `benchmarks/baseline_kernels_bench.cpp:316-346` -- `int8_dot4_kernel_unrolled_scalar` (scalar INT8, the winner)
- `benchmarks/baseline_kernels_bench.cpp` -- Full harness with all optimization flags

### DX12 Integration
- `fsr4-src/baseline/dx12/ffx_provider_fsr4_dx12.cpp` -- DX12 dispatch orchestration, per-quality weight selection
- `fsr4-src/baseline/internal/shader_selector.cpp` -- DOT4/FP8/WMMA path selection based on hardware caps
- `fsr4-src/baseline/dx12/BuildFSR4UpscalerShaders.bat` -- Shader compilation commands
