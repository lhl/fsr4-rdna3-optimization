# HLSL Golfing

This document turns the HIP benchmark findings into a practical HLSL/DX12 attack plan for RDNA3 (`gfx1100`) and RDNA3.5 (`gfx1151`).

The goal is not "rewrite FSR4". The goal is narrower:

- keep `fsr4-src/baseline` untouched
- use `fsr4-src/opt` as the sandbox
- focus first on the INT8/DOT4 path that both GPUs actually share
- leave the WMMA/FP8/RDNA4-oriented path alone until the INT8 work is exhausted

## Basic HOWTO

- Open the repo in VS Code on Windows.
- Work only in `fsr4-src/opt`.
- Edit the reusable operator `.hlsli` files first, not the generated `passes_*.hlsl` files.
- Start with these INT8 operator files:
  - `fsr4-src/opt/dx12/ml2code_runtime/operators/int8_NHWC/Fused/ConvNextBlock.hlsli`
  - `fsr4-src/opt/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FasterNetBlock.hlsli`
  - `fsr4-src/opt/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FusedConv2D_k2s2b_QuantizedOutput.hlsli`
  - `fsr4-src/opt/dx12/ml2code_runtime/operators/int8_NHWC/Fused/FNB_CT2D_ADD.hlsli`
  - `fsr4-src/opt/dx12/ml2code_runtime/operators/int8_NHWC/ConvTranspose2D_k2s2b.hlsli`
  - `fsr4-src/opt/dx12/ml2code_runtime/operators/float16_NHWC/Fused/CNB_CT2D.hlsli`
- Do not start by editing:
  - `fsr4-src/opt/dx12/ml2code_runtime/operators/float8_NHWC/*`
  - `fsr4-src/opt/internal/shaders/pre_wmma.hlsl`
  - `fsr4-src/opt/internal/shaders/post_wmma.hlsl`
- If you just need the helper-generated initializer files, build:
  - `msbuild fsr4-src/opt/dx12/GenerateFR4Files.vcxproj /p:Configuration=Release /p:Platform=x64`
- If you need the full generated shader blob headers and permutation headers, run the FidelityFX shader/codegen step from the full SDK/API build that produces the files included by `fsr4-src/opt/internal/shader_selector.cpp`.
- Build the DX12 provider DLL from the full SDK/API solution that compiles `fsr4-src/opt/dx12/ffx_provider_fsr4_dx12.cpp`.
- Build that provider with `FSR4_ENABLE_DOT4=1` if you want the INT8 path compiled in at all.
- Leave the `MLSR-WMMA` environment variable unset when benchmarking the RDNA3/RDNA3.5 INT8/DOT4 path.
- Benchmark in two layers:
  - quick directional sanity checks with `./baseline-benchmark.py`
  - real DX12 profiling on actual `gfx1100` and `gfx1151` hardware with the host app that loads the provider DLL

## Reality Check

This tree is useful, but it is not a turnkey Windows shipping pipeline by itself.

- `fsr4-src/opt/internal/shader_selector.cpp` includes generated files like `fsr4_model_v07_i8_balanced.h`, `fsr4_model_v07_i8_balanced.cpp`, and many `*_permutations.h` headers that are not checked into this repo.
- `fsr4-src/opt/dx12/GenerateFR4Files.vcxproj` appears to cover initializer file generation, not the full shader blob/permutation generation pipeline.
- `fsr4-src/opt/dx12/ffx_provider_fsr4_dx12.cpp` is present, but this repo does not expose an obvious standalone provider-DLL project file.

Practical implication:

- you can do real HLSL editing here
- you can document and stage the changes here
- you will still need the surrounding FidelityFX SDK/API build to regenerate the full blob headers and build the final provider DLL

## Architecture Targeting

There is no obvious DXC-style `--arch gfx1100` or `--arch gfx1151` knob in this HLSL path.

- The generated HLSL comments in this tree say `This file was generated for navi48 SKU`.
- The checked-in shader compile commands use shader-model and macro switches like `-T cs_6_6` and `-DWMMA_ENABLED=1`, not `gfx1100` or `gfx1151`.
- In practice, the DX12 path here emits DXIL and the AMD driver does the final hardware-specific compilation on the test GPU.

So "target RDNA3/3.5 and leave RDNA4 alone" means:

- work in the non-WMMA INT8/DOT4 path
- do not change the WMMA/FP8 codepath unless you are explicitly running a separate FP8 experiment
- benchmark on real `gfx1100` and `gfx1151` hardware
- treat the `navi48`-generated source shape as a starting point, not a proof that the current code shape is right for RDNA3/RDNA3.5

## Path Selection

Two switches matter more than anything else for keeping the work on the INT8 path.

- `FSR4_ENABLE_DOT4`
  - `fsr4-src/opt/internal/shader_selector.cpp` only compiles the INT8 model blob tables when this define is enabled.
  - If it is off, the selector falls back to the FP8 model path.
- `MLSR-WMMA`
  - `fsr4-src/opt/dx12/ffx_provider_fsr4_dx12.cpp` uses this environment variable as a runtime gate when `FSR4_ENABLE_DOT4` is enabled.
  - If `MLSR-WMMA` is unset, the provider stays on the non-WMMA path.

Practical rule:

- For RDNA3/RDNA3.5 INT8 golfing, build with `FSR4_ENABLE_DOT4=1` and do not set `MLSR-WMMA`.

## Shared Findings

These are the signals that look real on both `gfx1100` and `gfx1151`.

| Topic | gfx1151 | gfx1100 | HLSL relevance |
|---|---:|---:|---|
| Scalar INT8 beats packed dot | about `+31.8%` | about `+7-8%` | Highest-value shared INT8 experiment |
| Compile-time unrolling | keep | keep | Already reflected in generated HLSL; do not regress it |
| Store-time quantization | keep | keep | Already reflected in real HLSL; do not move requant into inner loops |
| Hoisted scale/bias loads | keep | keep | Preserve or improve invariant load hoisting |
| LDS staging | drop | drop | Strong anti-target for INT8 path |
| Unfused / two-pass variants | drop | drop | Do not split fused operators just to "simplify" code |
| `threads=256` in harness | best | best | No evidence that thread-count golfing is the best next step |
| Interior/edge split | unsure | unsure | Secondary experiment only |

The most important shared conclusion is simple:

- the first serious HLSL experiment should be scalarizing hot INT8 `dot4add_i8packed` regions behind an A/B switch

## Divergent Findings

These are real, but not equally portable.

| Topic | gfx1151 | gfx1100 | Meaning |
|---|---|---|---|
| Scalar over packed INT8 | very large win | modest win | Still worth trying on both, but expect smaller gain on RDNA3 |
| `items_per_thread=4` | no meaningful win | about `+11.5%` INT8 | Good RDNA3-specific tiling experiment |
| `inner_fp8=8` | not meaningful | about `+6%` FP8 | FP8-only follow-up, not phase 1 |
| `fp8_quantized_io` | unsure | about `+17%` FP8 | FP8-only follow-up, not phase 1 |

Practical rule:

- use shared findings to set phase 1
- use the `gfx1100`-specific knobs as optional phase 2 work

## Operator Map

For the real HLSL, the hottest INT8 path is not pass 0.

- Pass 0 uses FP16 `dot2add`; it is not the packed-vs-scalar target.
- Passes 1-13 are dominated by INT8 `dot4add_i8packed`.

Best operator targets, in order:

1. `ConvNextBlock.hlsli`
   - simpler than the bigger fused blocks
   - used in passes 1, 2, and 12
   - good first scalarization target
2. `FasterNetBlock.hlsli`
   - used repeatedly in passes 4, 5, 7, 8, 10
   - high leverage once the `ConvNextBlock` experiment is understood
3. `FusedConv2D_k2s2b_QuantizedOutput.hlsli`
   - used in passes 3 and 6
   - good place to test a wider per-thread output tile
4. `FNB_CT2D_ADD.hlsli`
   - very important but more complex
   - good after the simpler fused blocks
5. `ConvTranspose2D_k2s2b.hlsli`
   - used by the larger fused decode path
6. `CNB_CT2D.hlsli`
   - post path still does INT8 internal compute before FP16 output

## Dispatch Coupling

If you change threadgroup shape or pixels-per-thread in HLSL, you probably also need to change the provider-side dispatch math.

The main places to audit are:

- `fsr4-src/opt/dx12/ffx_provider_fsr4_dx12.cpp`
  - pre-pass dispatch sizing
  - pass 1-12 dispatch sizing
  - pass 13 dispatch sizing

The non-WMMA dispatch logic currently assumes fixed group shapes for the INT8 path. If you change:

- `[numthreads(...)]`
- how many X/Y pixels a thread computes
- output tile width per lane

then you must re-check the corresponding `RoundUpDiv(...)` logic in the provider.

## Best HLSL Targets

### P0: Shared INT8 scalarization

Goal:

- replace or A/B test the packed INT8 inner MAC sequence in the shared INT8 fused operators

What to try:

- add a local helper for scalar INT8 MAC accumulation
- keep the surrounding dataflow the same
- preserve:
  - register preloads
  - loop unrolling
  - once-at-store requantization
- start in `ConvNextBlock.hlsli`

Why:

- this is the only major win that clearly points at the real HLSL INT8 path on both GPUs

Expected result:

- likely larger upside on `gfx1151`
- still possibly worth `10%`-ish class wins on `gfx1100` depending on operator and codegen

### P1: RDNA3 INT8 tile widening

Goal:

- emulate the `items_per_thread=4` win seen on `gfx1100`

What to try:

- have one lane compute multiple neighboring outputs in X before storing
- prefer experiments that do not require changing `numthreads` yet
- start in:
  - `FusedConv2D_k2s2b_QuantizedOutput.hlsli`
  - then `ConvNextBlock.hlsli`

Why:

- `gfx1100` showed a real INT8 gain here
- this is the next-most-interesting non-WMMA RDNA3 signal after scalarization

Risk:

- likely architecture-specific
- likely neutral or worse on `gfx1151`

### P2: Boundary specialization

Goal:

- split the interior fast path from the bounds-checked edge path

What to try:

- separate no-bounds helper for the interior region
- small fallback path for edges

Why:

- it was not proven, but it is still a plausible HLSL optimization once the bigger wins are exhausted

Risk:

- both GPUs classified this as `unsure`
- code complexity rises quickly

### P3: FP8-only follow-up

Goal:

- only after INT8 work is done, probe the FP8/WMMA path

What to try:

- reduce effective inner blocking similar to the `inner_fp8=8` signal
- inspect FP8 quantized I/O overhead

Why not first:

- the real FP8 path uses WMMA and LDS staging
- the HIP FP8 microkernel is much less representative of the shipping HLSL than the INT8 harness is
- you explicitly want to leave the RDNA4/WMMA path alone at the start

## What Not To Try First

- Do not start with LDS staging on the INT8 path.
- Do not start with compile-flag churn.
- Do not split fused operators into two passes.
- Do not move requantization into the accumulate loop.
- Do not start by changing the FP8 WMMA path if the goal is RDNA3/RDNA3.5 INT8 performance.
- Do not hand-edit every generated `passes_*.hlsl` file unless the operator-template route is impossible.

## Suggested Experiment Order

1. Add a compile-time A/B switch for scalar vs packed INT8 MAC in `ConvNextBlock.hlsli`.
2. Repeat the same pattern in `FasterNetBlock.hlsli`.
3. Apply the same idea to `FusedConv2D_k2s2b_QuantizedOutput.hlsli`.
4. Only then try an `items_per_thread`-style widening in one INT8 operator.
5. If any operator changes thread-to-output mapping, update provider dispatch sizing.
6. Keep FP8/WMMA untouched until the INT8 path has no more clear wins.

## Benchmarking Ladder

Use the fastest feedback loop that still answers the question you are asking.

- For "is this idea directionally sane?" use the Linux HIP harness in this repo.
- For "did the HLSL still compile and select the INT8 path?" regenerate the shader/codegen outputs and rebuild the provider.
- For "did this help the real DX12 implementation on RDNA3/3.5?" run the actual DX12 host app on `gfx1100` and `gfx1151` and profile dispatches.

Recommended comparison discipline:

- lock one preset first, preferably `Balanced`
- lock one resolution first, preferably `1080p`
- compare one operator family at a time
- keep WMMA disabled for the INT8 campaign
- preserve the exact same frame sequence / benchmark scene when comparing DLLs

## Short Version

If you only remember five things:

- edit `fsr4-src/opt`, not `baseline`
- stay on the INT8/DOT4 path first
- do not assume there is a DXC `gfx1100/gfx1151` target flag here
- scalarizing `dot4add_i8packed` is the best shared HLSL experiment
- `items_per_thread=4` is the best extra RDNA3-specific idea after that
