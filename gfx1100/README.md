# FSR4 RDNA3 Benchmark: gfx1100 (W7900) vs gfx1151 (Strix Halo)

This folder replicates the HIP microkernel benchmark workflow from the repo root on a discrete Radeon Pro W7900 (`gfx1100`, RDNA3), enabling direct comparison against the Strix Halo iGPU (`gfx1151`, RDNA3.5) results in the parent project.

## Cross-Architecture Performance Summary

### Baseline Defaults (threads=256, items=1, unrolled, scalar INT8)

| Mode | gfx1100 Mean (ms) | gfx1151 Mean (ms) | Delta | Faster On |
|---|---:|---:|---:|---|
| INT8 | 0.007754 | 0.005376 | +44.2% | gfx1151 |
| FP8 | 0.012805 | 0.019868 | -35.6% | gfx1100 |

**Key observation**: The two GPUs have opposite strengths in this harness. The gfx1151 iGPU is substantially faster at INT8 scalar MAC (44% advantage), while the gfx1100 dGPU is substantially faster at FP8 conversion+FMA (36% advantage). This likely reflects differences in ALU scheduling, memory subsystem (shared DRAM on iGPU vs dedicated GDDR6 on dGPU), and FP8 conversion path latency.

### Before/After Performance (gfx1100)

The only confirmed optimization on gfx1100 was O19 (`--fp8-quantized-io`), which improved FP8 by ~17%.

- Before: `results/baseline-benchmark-20260305-185902.json` (3-trial)
- After: `results/gfx1100-final-default-optimized-trials3.json` (3-trial)

| Mode | Before Mean (ms) | After Mean (ms) | Improvement |
|---|---:|---:|---:|
| INT8 | 0.007754 | 0.007713 | 0.54% faster |
| FP8 | 0.012805 | 0.010641 | 16.90% faster |

Compare with gfx1151's optimization gains (where scalar INT8 and requant-policy fixes delivered dramatic improvement):

| Mode | gfx1151 Before (ms) | gfx1151 After (ms) | Improvement |
|---|---:|---:|---:|
| INT8 | 0.007743 | 0.005376 | 30.57% faster |
| FP8 | 0.117392 | 0.019868 | 83.08% faster |

The gfx1151 FP8 "before" number (0.117 ms) was inflated by a per-iteration requantization policy bug that was fixed during optimization. The gfx1100 baseline never had this issue (the harness already used store-time quantization by the time gfx1100 benchmarks were run).

### Final Cross-Architecture Comparison (After Optimization)

| Mode | gfx1100 Final (ms) | gfx1151 Final (ms) | Delta | Faster On |
|---|---:|---:|---:|---|
| INT8 | 0.007713 | 0.005376 | +43.5% | gfx1151 |
| FP8 | 0.010641 | 0.019868 | -46.4% | gfx1100 |

### INT8 vs FP8 Ratio (Within Each GPU)

| GPU | INT8 (ms) | FP8 (ms) | FP8/INT8 Ratio |
|---|---:|---:|---:|
| gfx1100 Before | 0.007754 | 0.012805 | 1.65x |
| gfx1100 After | 0.007713 | 0.010641 | 1.38x |
| gfx1151 After | 0.005376 | 0.019868 | 3.70x |

On gfx1100, the INT8-vs-FP8 gap is much narrower (1.38x after optimization vs 3.70x on gfx1151). FP8 conversion+FMA overhead is much less punishing on the discrete GPU.

## Optimization Comparison: gfx1100 vs gfx1151

20 optimization attempts were evaluated on both architectures. The table below shows which optimizations worked, failed, or differed between GPUs.

### Optimizations That Agreed (Same Outcome on Both)

| ID | Change | gfx1100 | gfx1151 | Notes |
|---|---|---|---|---|
| O01 | Protocol lock | Keep | Keep | Stable baselines on both |
| O02 | Threadgroup 64/128 | Unsure/worse | Unsure/worse | No confidence-grade win over 256 on either target |
| O04 | Wave-size check | warpSize=32 | warpSize=32 | Same hardware wave size |
| O05 | Runtime inner loops | Drop (FP8 -360%) | Drop (high variance) | Compile-time unrolling wins on both; gfx1100 shows even larger FP8 regression |
| O07 | In-loop scale/bias | Drop | Drop | Hoisted loads better on both |
| O08 | Per-iter requant | Drop | Drop | Catastrophic on both (gfx1100: INT8 +39%, FP8 +324%; gfx1151: INT8 +194%, FP8 +476%) |
| O09 | Interior/edge split | Unsure | Unsure | Noisy on both |
| O10 | LDS stage input | Drop | Drop | Slower on both |
| O11 | LDS stage input+weight | Drop | Drop | Worse than O10 on both |
| O12 | LDS padding | Drop | Drop | Much slower on both |
| O13 | LDS double-buffer | Drop | Drop | Still slower on both |
| O16 | Unfused post-op | Drop | Drop | Clear regression on both |
| O17 | Two-pass | Drop | Drop | Clear regression on both |
| O18 | Mixed INT8 subpath | Drop | Drop | Large INT8 regression on both |
| O20 | Final control | Unsure/Keep | Keep | Near baseline on both |

### Optimizations That Diverged

| ID | Change | gfx1100 | gfx1151 | What Differed |
|---|---|---|---|---|
| O03 | Items-per-thread 2/4 | **Unsure** (items=4: INT8 +11.8%, FP8 -9.6%) | **Keep (items=1)** | gfx1100 shows INT8 benefit from more items but FP8 degrades; gfx1151 items=1 was clearly best for both modes |
| O06 | Packed vs scalar INT8 | **Drop** (packed -7.3%) | **Keep** (scalar +31.8%) | Both prefer scalar, but the magnitude differs hugely: gfx1151 gained 32% from switching to scalar; gfx1100 only loses 7% going back to packed |
| O14 | Occupancy/register sweep | **Unsure** (inner=8 slight FP8 gain) | **Unsure** | gfx1100 shows `inner=8` FP8 mean 0.012047 ms (possible small win); gfx1151 had FP8 too noisy to classify |
| O15 | Compile flags (-O2, -Ofast) | **Unsure** (INT8 high variance) | **Unsure** | Both show no clear benefit; kept `-O3` |
| O19 | FP8 quantized-IO | **Keep** (+17.1% FP8) | **Unsure** (high variance) | The only confirmed keep unique to gfx1100. The dGPU's lower FP8 variance made the improvement measurable. |

### Key Divergence Analysis

**O06 (Packed vs Scalar INT8)** -- The biggest finding on gfx1151 (32% INT8 gain from scalar) was the smallest measurable effect on gfx1100 (only 7% packed penalty). Both GPUs prefer scalar, but the discrete GPU's wider memory bus and different ALU scheduler likely hide most of the packed-dot overhead that dominates on the iGPU. This means the `dot4add_i8packed` vs scalar INT8 question is even more target-dependent than expected.

**O19 (FP8 Quantized-IO)** -- This was `unsure` on gfx1151 (too much FP8 variance under iGPU contention) but a clear `keep` on gfx1100 (+17% FP8 improvement, cv < 1%). The discrete GPU's dedicated memory and stable power delivery enabled confident measurement of what was likely a real win on both targets.

**O03 (Items-Per-Thread)** -- gfx1100 showed a clear INT8 benefit from items=4 (+11.8%) that gfx1151 never exhibited. However, FP8 degraded (-9.6%), making the overall verdict mixed. This suggests gfx1100's register file and ALU can better exploit instruction-level parallelism from processing multiple items, but only for the simpler INT8 path.

## Optimization Attempts (Single-Glance)

| ID | Change | Outcome | Headline Effect |
|---|---|---|---|
| O01 | Protocol lock | Keep | Stable 60s baseline captured. |
| O02 | Threadgroup sweep (64/128/256) | Keep (256) | 64/128 unsure or worse. |
| O03 | Items-per-thread sweep (1/2/4) | Unsure | items=4 helps INT8 (+11.8%) but hurts FP8 (-9.6%). |
| O04 | Wave-size verification | Keep | Confirmed `warpSize=32`. |
| O05 | Unrolled vs runtime inner loops | Keep (unrolled) | Runtime-loop FP8 regression: -360%. |
| O06 | Packed vs scalar INT8 I/O | Keep (scalar) | Packed INT8 -7.3% (much smaller than gfx1151's -32%). |
| O07 | Hoist scale/bias loads | Drop | In-loop variant +31% slower. |
| O08 | Per-iter requant vs once-at-store | Drop | INT8 +39%, FP8 +324% regression. |
| O09 | Interior/edge split dispatch | Unsure | Small movements, noisy. |
| O10 | LDS stage input | Drop | +30% slower (INT8). |
| O11 | LDS stage input+weight | Drop | +32% slower (INT8). |
| O12 | LDS padding/swizzle proxy | Drop | +73% slower (INT8). |
| O13 | LDS double-buffer proxy | Drop | +57% slower (INT8). |
| O14 | Occupancy/register sweep | Unsure | No confident win; inner=32 hurt FP8. |
| O15 | Compile flags (-O2, -Ofast) | Unsure | Tiny movements within noise; kept -O3. |
| O16 | Unfused post-op path | Drop | +97% INT8, +61% FP8 regression. |
| O17 | Two-pass adjacent path | Drop | +97% INT8, +63% FP8 regression. |
| O18 | Mixed INT8 subpath | Drop | +25% INT8 regression. |
| O19 | FP8 quantized-IO | Keep | +17.1% FP8 improvement (only confirmed keep). |
| O20 | Final cleanup control | Unsure | Near baseline (confirms no drift). |

Batch summary: `drop=12`, `keep=1`, `unsure=10` (conservative gate: `min_uncertainty_pct=3%`, `cv_scale=0.5`).

## Final Optimized Defaults (gfx1100)

- Kernel default updated: `fp8_quantized_io = true` in `benchmarks/baseline_kernels_bench.cpp`
- Final 3-trial run: `results/gfx1100-final-default-optimized-trials3.json`
- Classification: `results/gfx1100-final-default-optimized-trials3-classification.json`
  - FP8: `keep` (+16.87% median improvement, 3-trial vs 3-trial)
  - INT8: `unsure` (+0.39% median, within noise)

## Environment

| Item | Value |
|---|---|
| GPU | AMD Radeon Pro W7900 |
| Target arch | `gfx1100` (RDNA3) |
| Mamba env | `therock` |
| HIP/ROCm | HIP `7.2.26043-9999` (clang 22) |
| ROCm SDK | `7.12.0a20260304` |

### gfx1151 Reference Environment (from parent project)

| Item | Value |
|---|---|
| GPU | Radeon 8060S Graphics (Strix Halo iGPU) |
| Target arch | `gfx1151` (RDNA3.5) |
| HIP/ROCm | HIP `7.12.60490-128c4eea36` |
| ROCm SDK | `7.12.0a20260226` |

## Attempted gfx1151 Run On W7900 (What Failed and Why)

We attempted to run the `gfx1151` final kernel path directly on this `gfx1100` host. There were two distinct outcomes:

1. **Initial attempt (misleading, not a true gfx1151 execution):**
   - Command used `--arch gfx1151` but **without** `--force-rebuild`.
   - The harness reused the existing `gfx1100` binary.
   - Artifact: `results/gfx1151-final-kernel-attempt-on-w7900.json`
   - Evidence: payload config says `arch=gfx1151`, but runtime `INFO` reports `gcnArchName=gfx1100`.
   - Interpretation: this file is a valid run, but it is **not** a true `gfx1151` code-object run.

2. **Forced true gfx1151 attempt (expected failure):**
   - Re-run with `--arch gfx1151 --force-rebuild`.
   - Build succeeded, runtime failed immediately with HIP code-object assertion (`hip_code_object.cpp`) and `SIGABRT`.
   - Root cause: a `gfx1151` code object cannot be dispatched on a `gfx1100` device.

Implication: direct `gfx1151` execution must be done on `gfx1151` hardware; this README uses recorded `gfx1151` final results from the parent project for cross-architecture comparison.

## Benchmark Protocol

Same protocol as gfx1151 for direct comparability:
- `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1`
- 60s target per mode; `mode=both` runs ~2 minutes total
- Stats: mean, stddev, median, p95, cv_pct
- Keep/drop gate: classification with `min_uncertainty_pct=3`, `cv_scale=0.5`
- Baseline: 3-trial aggregate (`--trials 3`) for compare-ready reference
- Note: gfx1100 benchmarks were run after the harness already included the requant-once fix from gfx1151 optimization, so the gfx1100 "before" FP8 number (0.012805 ms) is not inflated by the per-iter requant bug that affected the gfx1151 "before" (0.117392 ms).

## Conclusions

1. **The gfx1100 dGPU and gfx1151 iGPU have opposite performance profiles** in this harness: gfx1151 is 44% faster at INT8, gfx1100 is 36-46% faster at FP8.

2. **Most optimizations agreed across architectures**: LDS staging, per-iter requantization, runtime loops, unfused post-ops, and two-pass all regressed on both. Compile-time unrolling and store-time quantization are universally good practices.

3. **The biggest gfx1151 win (scalar INT8, O06) was much smaller on gfx1100**: 32% INT8 gain on iGPU vs only 7% packed penalty on dGPU. The scalar-vs-packed dot question is highly target-dependent.

4. **O19 (FP8 quantized-IO) is the one confirmed optimization for gfx1100**: +17% FP8 improvement. This was unmeasurable on gfx1151 due to variance, suggesting it's a real win that the iGPU's noise floor hid.

5. **No INT8 optimization was promoted to default on gfx1100**: The default configuration (scalar, unrolled, threads=256, items=1) remained the best balanced choice under mixed INT8+FP8 gating. Some INT8-only improvements appeared (for example `items=4`, +11.8% INT8) but came with FP8 regressions, so they were not adopted as defaults.

6. **gfx1100's narrower INT8/FP8 gap (1.38x after optimization vs 3.70x)** suggests the discrete GPU handles FP8 conversion overhead much better, likely due to dedicated VRAM bandwidth and more predictable memory latency.

## Tracking Files
- `WORKLOG.md` -- run-by-run log
- `IMPLEMENTATION.md` -- commands and per-optimization details
- `TODO.md` -- punchlist

Note: Root-level `gfx1151` artifacts are intentionally not modified by this workspace.
