# RDNA4-ONLY: Actual Source-Visible RDNA4 and WMMA Dependencies in Released FSR4

## Scope

This note is a source-grounded audit of the released FSR4 HLSL and runtime-selection code in this repo plus the reference SDK checkout. The goal is narrower than "prove what silicon can or cannot do":

- Identify the actual intrinsics, wave ops, and optimization patterns visible in source.
- Separate source-level dependencies from backend or ISA-counter terminology.
- Distinguish the active INT8 path from the WMMA-based FP8 path.

The strongest conclusion is that the released source is not monolithically "RDNA4-only". It contains:

- an INT8 `DOT4` path selected when WMMA is not used
- a WMMA/FP8 path selected only when WMMA support is available and enabled

Relevant runtime-selection plumbing:

- `fsr4-src/baseline/dx12/ffx_provider_fsr4_dx12.cpp`
- `fsr4-src/baseline/internal/shader_selector.cpp`

## Executive Summary

The actual source-visible features that look genuinely WMMA or RDNA4-leaning are:

| Feature | Where it appears | Notes |
| --- | --- | --- |
| `AmdWaveMatrixMultiply` | `float8_NHWC/Conv2D_k2s2b.hlsli`, `float16_NHWC/Fused/CNB_CT2D.hlsli`, `pre_wmma.hlsl`, `post_wmma.hlsl` | Core wave-matrix multiply primitive in FP8 path and optional INT8 WMMA branches |
| `AmdWaveMatrixA/B/Accumulator` | same files | Source-visible matrix object model used by WMMA path |
| `AMD_GROUPSHARED_LOAD` | `float8_NHWC/Conv2D_k2s2b.hlsli` | Explicit LDS-to-wave-matrix staging |
| `groupshared` LDS staging | `float8_NHWC/Conv2D_k2s2b.hlsli`, `pre_wmma.hlsl`, `post_wmma.hlsl` | Strongest source-visible "special hardware path" signal |
| `WaveReadLaneFirst`, `WaveGetLaneIndex` | `float8_NHWC/Conv2D_k2s2b.hlsli` | Lane choreography for WMMA input staging |
| WMMA-specific blob selection | `shader_selector.cpp`, `ffx_provider_fsr4_dx12.cpp` | Runtime chooses WMMA blobs only when supported and enabled |

The actual source-visible features that dominate the active INT8 path are not WMMA-specific:

| Feature | Where it appears | Notes |
| --- | --- | --- |
| `dot2add` | `int8_NHWC/Conv2D_k2s2b.hlsli` | Entry pass: FP16 input downscale and quantization |
| `dot4add_i8packed` | `ConvNextBlock.hlsli`, `FasterNetBlock.hlsli`, `FNB_CT2D_ADD.hlsli`, `CNB_CT2D.hlsli`, `FusedConv2D_k2s2b_QuantizedOutput.hlsli` | Main INT8 compute primitive in passes 1-13 |
| `ByteAddressBuffer` and `RWByteAddressBuffer` | generated pass files | Main model-core memory path |
| `[unroll]` | almost all fused operators | Main code-shape optimization pattern used by ML2Code |
| `SampleLevel` texture sampling | pre/post passes, debug, reprojection | Real, but mostly outside the model core |

## Runtime and Blob Selection

The runtime and build plumbing already expose two distinct execution families.

### Runtime gate

`fsr4-src/baseline/dx12/ffx_provider_fsr4_dx12.cpp`:

- `supportsWmma = EnableAMDExtensions(...)`
- when `FSR4_ENABLE_DOT4` is enabled, WMMA is further gated by `MLSR-WMMA`

That means the checked-in provider does not always force the WMMA path. It can still select the non-WMMA INT8 path.

### Shader blob selection

`fsr4-src/baseline/internal/shader_selector.cpp`:

- if `options.WMMA || !FSR4_ENABLE_DOT4`, the selector chooses `fsr4_model_v07_fp8_no_scale_*`
- otherwise it chooses `fsr4_model_v07_i8_*`

### Build scripts

`fsr4-src/baseline/dx12/BuildFSR4UpscalerShaders.bat` builds the WMMA shader set explicitly with `-DWMMA_ENABLED=1`.

## Actual Source-Visible RDNA4 and WMMA Features

### 1. `AmdWaveMatrixMultiply`

This is the clearest source-visible "newer hardware path" primitive in the released code.

Examples:

- `fsr4-src/baseline/dx12/ml2code_runtime/operators/float8_NHWC/Conv2D_k2s2b.hlsli`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/float16_NHWC/Fused/CNB_CT2D.hlsli`
- `fsr4-src/baseline/internal/shaders/pre_wmma.hlsl`
- `fsr4-src/baseline/internal/shaders/post_wmma.hlsl`

What it means here:

- FP8 path depends on wave-matrix multiply in source, not just as a likely backend lowering.
- Some INT8 fused operators also contain WMMA variants, but those are not the active checked-in INT8 blob path.

### 2. `AmdWaveMatrixA`, `AmdWaveMatrixB`, `AmdWaveMatrixAccumulator`

These types appear throughout the WMMA code paths and make the dependency source-visible rather than inferred.

They are used with:

- `F16` data in FP8 entry conv staging
- `I8` data in optional INT8 WMMA branches
- `FP8` data in FP8 model and postprocessing paths

### 3. LDS staging via `groupshared` and `AMD_GROUPSHARED_LOAD`

The FP8 conv path does not just call WMMA. It explicitly stages through LDS:

- `groupshared uint inputLDS[(16 * 16)/2];`
- `AMD_GROUPSHARED_LOAD(inputMatrix, inputLDS, 0, 8, true);`

This is one of the few places where the Reddit claim is directionally aligned with the released source: the FP8 path really is tied to wave-matrix plus LDS staging.

### 4. Wave-lane coordination

The FP8 path also uses:

- `WaveReadLaneFirst`
- `WaveGetLaneIndex`

These are part of the source-visible lane choreography that feeds the WMMA tiles.

### 5. Optional INT8 WMMA branches

Some operators ship both:

- a straightforward `dot4add_i8packed` path
- a `WMMA_ENABLED` branch using `AmdWaveMatrixMultiply` on `I8`

Examples:

- `fsr4-src/baseline/dx12/ml2code_runtime/operators/float16_NHWC/Fused/CNB_CT2D.hlsli`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FusedConv2D_k2s2b_QuantizedOutput.hlsli`

Important caveat:

- the checked-in INT8 generated shaders have `WMMA_ENABLED 0`, so these optional branches are not the active path we audited and benchmarked

## INT8 Kernel Analysis

### What is active in the checked-in INT8 shaders

The generated INT8 model shaders define:

- `WMMA_ENABLED 0`
- `DOT4_ENABLED 1`

See:

- `fsr4-src/baseline/internal/shaders/fsr4_model_v07_i8_balanced/passes_1080.hlsl`

### INT8 pass structure

- Pass 0 uses `dot2add` on FP16 inputs in `int8_NHWC/Conv2D_k2s2b.hlsli`.
- Passes 1-13 are dominated by `dot4add_i8packed` in fused INT8 operators.

Representative files:

- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Conv2D_k2s2b.hlsli`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/ConvNextBlock.hlsli`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FasterNetBlock.hlsli`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FNB_CT2D_ADD.hlsli`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FusedConv2D_k2s2b_QuantizedOutput.hlsli`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/float16_NHWC/Fused/CNB_CT2D.hlsli`

### What actually matters for INT8 performance

From a source-level optimization perspective, the active INT8 path is mainly about:

- `dot4add_i8packed`
- register staging and packed load shape
- `ByteAddressBuffer` and scratch-buffer traffic
- fused operator structure
- `[unroll]` keeping the inner loops static
- quantize-on-store policy

### What does not show up as an explicit source dependency

The following Reddit-claim names do not appear in the released source as HLSL primitives:

- `loadcnt`
- `samplecnt`
- `storecnt`
- `bvhcnt`
- `kmcnt`
- `dscnt`
- `swmmac`

So even if some of them matter to final ISA scheduling or backend codegen, they are not the right source-level vocabulary for the active INT8 kernel path we optimized.

### Relevance to our measured results

Our largest real harness win lines up with this source reading:

- on `gfx1151`, scalar INT8 beat the packed-dot path by about `31.84%`
- on `gfx1100`, the same direction held, but with much smaller magnitude

Relevant repo notes:

- `OPTIMIZATION_RESULTS.md`
- `README.md`
- `gfx1100/README.md`

This is exactly the part of FSR4 that maps well to our INT8 kernel work: the active INT8 path is built around `dot4add_i8packed`, not around source-visible WMMA-only intrinsics.

## FP8 Kernel Analysis

### What is active in the checked-in FP8 shaders

The generated FP8 model shaders define:

- `WMMA_ENABLED 1`
- `DOT4_ENABLED 1`

See:

- `fsr4-src/baseline/internal/shaders/fsr4_model_v07_fp8_no_scale_passes_1080.hlsl`

### FP8 path is WMMA-only in source

`fsr4-src/baseline/dx12/ml2code_runtime/operators/float8_NHWC/Conv2D_k2s2b.hlsli` explicitly errors out when `WMMA_ENABLED` is absent:

- `#error To use FP8 data type you need to provide WMMA_ENABLED=1 in hlsl_defines. There is no FP8 without WMMA.`

This is the strongest source-visible dependency in the whole tree.

### FP8 path characteristics

The FP8 kernels use:

- `AmdWaveMatrixMultiply`
- `AmdWaveMatrixA/B/Accumulator`
- `groupshared` LDS staging
- wave-lane helpers (`WaveReadLaneFirst`, `WaveGetLaneIndex`)
- FP8 saturation/copy/store via wave-matrix objects

### Tile shape actually visible in source

The visible matrix tiles are `16x16`, not `32x32`.

Examples:

- `AmdWaveMatrixAccumulator<..., 16, 16>`
- `AmdWaveMatrixA<..., 16, 16>`
- `AmdWaveMatrixB<..., 16, 16>`

That makes the Reddit claim about "3.5 and earlier can't do a 32x32 matrix" look low-relevance for the released source as written. The source we audited is built around `16x16` wave-matrix tiles.

### Pre/post passes also use texture sampling

Outside the model core, the FP8 or WMMA path still interacts with conventional graphics-style operations:

- `SampleLevel` in `pre_common.hlsli`
- texture resources and samplers in `ffx_fsr4_upscale_resources.h`
- SPD and auto-exposure texture reads and writes in `spd_auto_exposure.hlsl`

So the FP8 path is not "only WMMA". It is WMMA plus ordinary texture and UAV operations.

## Reddit Permalink and Claim Assessment

Permalink:

- <https://www.reddit.com/r/Amd/comments/1hzhnsj/comment/m6w3uda/>

The comment claims that RDNA3.5 and earlier lack several specific counters or register families and therefore cannot execute the required FSR4 instructions as designed.

This repo does not let us prove or disprove the silicon statement directly. What we can assess is whether the released source explicitly depends on those named features.

### Claim-by-claim relevance to released source

| Claimed dependency | Source-visible evidence in released FSR4 | Accuracy or relevance assessment |
| --- | --- | --- |
| `loadcnt` | not present as a source primitive | Low source-level relevance. The code obviously performs loads, but not through a source-visible `loadcnt` intrinsic. |
| `samplecnt` | not present as a source primitive | Low source-level relevance. Texture sampling exists in pre/post paths, but not under that name. |
| `storecnt` | not present as a source primitive | Low source-level relevance. Stores are ubiquitous, but not exposed through that source term. |
| `bvhcnt` | no BVH, `TraceRay`, or `RayQuery` usage found | Not relevant to released FSR4 source. |
| `kmcnt` | not present | No source evidence that this is an explicit dependency. |
| `dscnt` | not present | No source evidence that this is an explicit dependency. |
| `swmmac` | token not present | The relevant source-visible primitive is `AmdWaveMatrixMultiply`, not `swmmac`. |
| `32x32 matrix` support | no `32x32` wave-matrix tile visible; audited code uses `16x16` | Low relevance to released HLSL as written. |
| `8/16-bit scalars` | no explicit dependency in active kernel audit | Low source-level relevance to the active INT8 DOT4 path we optimized. |

### Overall assessment of the permalink

The Reddit comment is most relevant to the WMMA family of paths, especially FP8.

It is much less relevant to the active INT8 DOT4 path we audited and benchmarked, because:

- the released INT8 model path is source-visible and non-WMMA
- the quoted counter names do not appear in the HLSL
- the released source already contains explicit fallback or alternate non-WMMA selection logic

In short:

- **FP8 path:** the comment is directionally relevant because the source is clearly WMMA-dependent.
- **INT8 path:** the comment overstates the relevance of those named dependencies for the active source path we actually optimized.

## What Seems Genuinely RDNA4-Only From Source

If this file is used as a strict "RDNA4-only" checklist, the strongest candidates are:

1. `AmdWaveMatrixMultiply`
2. `AmdWaveMatrixA/B/Accumulator`
3. `groupshared` plus `AMD_GROUPSHARED_LOAD` staging for wave-matrix inputs
4. FP8 model entry points that hard-require `WMMA_ENABLED=1`

What does **not** look RDNA4-only from released source alone:

1. `dot4add_i8packed`
2. `dot2add`
3. ordinary `ByteAddressBuffer` model-core traffic
4. `SampleLevel` texture sampling
5. the active INT8 blob set selected when WMMA is not used

## Updating the FP8 WMMA HLSL for RDNA3 or RDNA3.5

Assuming RDNA3 or RDNA3.5 exposes `F16` WMMA but not `FP8` WMMA, the released FP8 shaders do not have a cheap compatibility switch. The source-visible dependency is not just "WMMA exists". It is specifically:

- FP8 wave-matrix inputs and weights in the fused model core
- repeated `CopySat` back into FP8 wave-matrix objects between fused stages
- FP8 tensor types at operator boundaries (`QuantizedTensor3f8_*`, `QuantizedTensor4f8_*`)

So the first conclusion is: **yes, the compute path would need to become `F16` WMMA or a software-emulated FP8 path.** But that does **not** automatically mean the model must be stored only as `F16`.

### Why this is more than a format rename

The prepass is already a useful counterexample:

- `float8_NHWC/Conv2D_k2s2b.hlsli` uses `F16` wave-matrix inputs and weights
- it accumulates in `F32`
- then it `CopySat`s to FP8 output

That means the source already demonstrates one viable pattern for RDNA3-style WMMA compute: **`F16` WMMA math with FP8 export**.

The problem is that the fused FP8 body is not written that way. In the model core and post path, the code repeatedly does:

- FP8 input tile load
- FP8 weight tile load
- `AmdWaveMatrixMultiply`
- `CopySat` back into FP8 intermediates
- consume those FP8 intermediates in the next stage

So the FP8 quantization points are part of the shipped graph behavior, not just a storage detail.

### What an RDNA3-capable FP8-model path would actually need

There are really two different port targets.

#### 1. Preserve the shipped FP8 model semantics

If we want behavior close to the released FP8 path, RDNA3 needs an explicit conversion path around every current FP8 WMMA boundary:

- unpack FP8 weights and activations to `F16`
- feed `F16` WMMA
- accumulate in `F32`
- re-quantize or saturate back to FP8 anywhere the current code does `CopySat` or FP8 tensor stores

This is the numerically safest approach, but it is also the heaviest one:

- extra unpack cost on input and weight tiles
- extra repack cost on intermediate activations
- more LDS or register pressure to stage the converted `F16` tiles

In other words, "support the FP8 model" on RDNA3 is not the same as "run the current FP8 HLSL unchanged".

#### 2. Keep FP8 storage, but move compute to an RDNA3-friendly `F16` graph

If the goal is performance rather than exact parity with the released FP8 path, the more plausible route is:

- keep weights compressed as FP8 on disk or in the package
- expand them to `F16` at load time or once per threadgroup
- keep activations in `F16` inside each fused operator
- avoid repeated FP8 round-trips inside the hot loop
- only quantize back to FP8 at coarse boundaries, or regenerate the model to use `F16` intermediates

This is probably the only version with meaningful upside on RDNA3 or RDNA3.5, but it is no longer the exact shipped FP8 execution path. It needs validation as a distinct model or blob family.

### Is there a clever packing trick?

There is one plausible "clever packing" angle:

- keep FP8 weight storage for bandwidth efficiency
- unpack a tile once into LDS as packed `F16`
- use `AMD_GROUPSHARED_LOAD` or equivalent `F16` wave-matrix loads from LDS

That can amortize the weight-conversion cost across a whole tile. But it does not solve the deeper issue by itself:

- the current fused FP8 shaders also keep **activations** and many intermediates in FP8 wave-matrix objects
- if RDNA3 cannot do FP8 WMMA natively, those intermediates must either be converted repeatedly or the graph must change to `F16`-internal execution

So packing can help bandwidth, but it does not remove the need for a separate RDNA3 compute path.

### The strongest source-based conclusion

From the released source alone, the cleanest RDNA3 or RDNA3.5 strategy is:

1. Build a separate WMMA-`F16` shader family for the FP8-trained model.
2. Decide explicitly whether to preserve FP8 quantization boundaries or move fused intermediates to `F16`.
3. If preserving FP8 semantics, expect substantial unpack or repack overhead.
4. If prioritizing speed, use FP8 storage plus `F16` WMMA compute and minimize FP8 round-trips.

That is why the active INT8 path remains the much simpler and safer optimization target for RDNA3-class hardware. The FP8 path can be made to run there in principle, but it is a real retargeting effort, not a small HLSL toggle.

## References

- Reddit permalink: <https://www.reddit.com/r/Amd/comments/1hzhnsj/comment/m6w3uda/>
- `fsr4-src/baseline/dx12/ffx_provider_fsr4_dx12.cpp`
- `fsr4-src/baseline/internal/shader_selector.cpp`
- `fsr4-src/baseline/dx12/BuildFSR4UpscalerShaders.bat`
- `fsr4-src/baseline/internal/shaders/fsr4_model_v07_i8_balanced/passes_1080.hlsl`
- `fsr4-src/baseline/internal/shaders/fsr4_model_v07_fp8_no_scale_passes_1080.hlsl`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Conv2D_k2s2b.hlsli`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/ConvNextBlock.hlsli`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FasterNetBlock.hlsli`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FNB_CT2D_ADD.hlsli`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FusedConv2D_k2s2b_QuantizedOutput.hlsli`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/float8_NHWC/Conv2D_k2s2b.hlsli`
- `fsr4-src/baseline/dx12/ml2code_runtime/operators/float16_NHWC/Fused/CNB_CT2D.hlsli`
- `fsr4-src/baseline/include/gpu/fsr4/pre_common.hlsli`
- `fsr4-src/baseline/include/gpu/fsr4/ffx_fsr4_upscale_resources.h`
- `fsr4-src/baseline/internal/shaders/spd_auto_exposure.hlsl`
