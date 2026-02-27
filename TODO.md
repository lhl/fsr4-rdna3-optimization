# TODO Punchlist

## Queue
- [ ] Add per-pass GPU timing capture for i8 balanced path (passes 0-13 + post 0-12) in DX12 provider
- [ ] Define measurement protocol for shader A/Bs on gfx1151 (warmup, sample count, pinned clocks/power mode, variance thresholds)
- [ ] Run baseline per-pass timing profile at 1080p/2160p across presets (Native/Quality/Balanced/Performance/DRS/UltraPerformance)
- [ ] Rank top compute-cost passes and publish hotspot table in `ANALYSIS-HLSL.md` and `IMPLEMENTATION.md`
- [ ] Implement O06 A/B in `ConvNextBlock.hlsli` (`dot4add_i8packed` vs scalarized int8 MAC) and benchmark
- [ ] Implement O06 A/B in `FasterNetBlock.hlsli` (`dot4add_i8packed` vs scalarized int8 MAC) and benchmark
- [ ] Implement O06 A/B in `FNB_CT2D_ADD.hlsli` (`dot4add_i8packed` vs scalarized int8 MAC) and benchmark
- [ ] Implement O06 A/B in pass-13 path `float16_NHWC/Fused/CNB_CT2D.hlsli` and benchmark
- [ ] Collect compiler/codegen evidence for each O06 variant (DXIL/ISA, VGPR/SGPR usage, occupancy) and correlate with timing
- [ ] Add correctness validation for each shader variant (output parity metrics, tolerance gates, regression screenshots)
- [ ] Apply O05 candidate: add `[unroll]` on first-conv spatial loops in `FasterNetBlock.hlsli` and benchmark
- [ ] Sweep threadgroup sizes for eligible passes (`64` vs `128` vs `256`) and record occupancy/timing tradeoffs
- [ ] Prototype interior/edge split dispatch for fused operators and measure gains vs branch-in-hot-loop baseline
- [ ] Evaluate combined "best-of" variant (O06 + O05 + any net-positive launch config) vs baseline end-to-end
- [ ] Build automation script to run the full A/B matrix and emit machine-readable summary (mean/stddev/cv + keep/drop decision)
- [ ] Add Linux runtime A/B check plan for Windows DX12 artifacts under Proton/VKD3D (relative baseline vs opt only)
- [ ] Record every run and decision in `WORKLOG.md` and `IMPLEMENTATION.md` after each experiment batch

## In Progress

## Blocked
- [ ] Verify stable DX12 extension support path under Wine/Proton for FSR4 WMMA-gated runtime comparisons

## Completed
- [x] Create AGENTS environment baseline
- [x] Add WORKLOG.md work recording requirement
- [x] Validate `therock` HIP/ROCm env vars and `hipcc` smoke compile
- [x] Document `rocm-sdk init` + env var bootstrap in README/AGENTS
- [x] Copy FSR4 source into split trees (`fsr4-src/baseline`, `fsr4-src/opt`)
- [x] Add benchmark harness with long-run sampling and stddev/cv tracking
- [x] Run 2m/3m/5m duration sweep and record variance + run counts
- [x] Add machine-readable stats output + auto keep/drop/unsure classification (`--reference-stats`)
- [x] Lock direct-TTY baseline at 60s with low variance
- [x] Complete `O02` threadgroup sweep (`64/128/256`)
- [x] Complete `O03` per-thread tile-size sweep (`items-per-thread=1/2/4`)
- [x] Complete `O04` wave-size verification (`warpSize=32`)
- [x] Complete `O05` inner-loop unroll A/B (`unrolled` vs `force-runtime-inner-loops`)
- [x] Complete `O06` INT8 I/O A/B (`force-packed-int8-io` vs scalar default)
- [x] Complete `O07` hoist invariants A/B (`force-inloop-scale-bias` vs default)
- [x] Complete `O08` requant policy A/B (`force-per-iter-requant` vs default)
- [x] Complete `O09` split interior/edge dispatch check (`split-interior-edge`)
- [x] Complete `O10` LDS input staging experiment (`lds-stage-input`)
- [x] Complete `O11` LDS weight staging experiment (`lds-stage-input + lds-stage-weight`)
- [x] Complete `O12` LDS padding experiment (`lds-padding`)
- [x] Complete `O13` LDS double-buffer experiment (`lds-double-buffer`)
- [x] Complete `O14` occupancy/register sweep (`threads` + `inner` sweep)
- [x] Complete `O15` compile-flag sweep (`-O2`, `-Ofast/-ffast-math`)
- [x] Complete `O16` unfused post-op A/B (`force-unfused-post`)
- [x] Complete `O17` two-pass A/B (`force-two-pass`)
- [x] Complete `O18` INT8 mixed-subpath A/B (`force-mixed-int8-path`)
- [x] Complete `O19` FP8 quantized-IO experiment (`fp8-quantized-io`)
- [x] Complete `O20` cleanup/keep-only-net-positive defaults
