# FSR4 RDNA3.5 Benchmark Summary

## Scope
This repository currently benchmarks HIP microkernels that emulate FSR4-style INT8/FP8 compute behavior on Strix Halo (RDNA3.5, `gfx1151`).

This is not a full DX12 frame-time benchmark of shipping FSR4.

## RDNA3.5 Support Snapshot
| Capability | Status In This Repo | Notes |
|---|---|---|
| INT8 dot compute | Benchmarked | INT8 kernels run on `gfx1151` (`amd_mixed_dot` and scalar INT8 paths). |
| FP8 (`e4m3`) compute | Benchmarked | FP8 conversion/FMA microkernels run on `gfx1151` (`hip_fp8`). |
| WMMA path | Present in FSR4 source, not benchmarked by this HIP harness | FSR4 source contains WMMA shader files (`pre_wmma.hlsl`, `post_wmma.hlsl`). |
| Wave size | Verified | Harness reports `warpSize=32` on this device. |

RDNA3.5 ISA reference (online):
- https://github.com/woct0rdho/rdna35-isa-markdown

## Before/After Performance (Main Summary)
Comparison of the stable direct-TTY baseline vs the final selected defaults after the optimization loop.

- Before file: `results/baseline-benchmark-20260227-040756.json`
- After file: `results/baseline-benchmark-20260227-052146.json`

| Mode | Before Mean (ms) | After Mean (ms) | Improvement |
|---|---:|---:|---:|
| INT8 | 0.007743 | 0.005376 | 30.57% faster |
| FP8 | 0.117392 | 0.019868 | 83.08% faster |

## INT8 vs FP8 Relative Performance
Lower time is better. `FP8/INT8` > 1.0 means INT8 is faster.

| Snapshot | INT8 Mean (ms) | FP8 Mean (ms) | FP8/INT8 Ratio | INT8 Speed Advantage |
|---|---:|---:|---:|---:|
| Before | 0.007743 | 0.117392 | 15.16x | 1416.10% |
| After | 0.005376 | 0.019868 | 3.70x | 269.57% |

Summary:
- INT8 remains faster than FP8 in this harness.
- The INT8-vs-FP8 gap narrowed significantly (about `4.10x` narrower ratio vs before), mostly because FP8 improved a lot in the updated harness path.

## Benchmark Methodology
### Protocol
- Primary command:
  - `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1`
- `60s` target is per mode (`int8` and `fp8`), so `mode=both` runs about 2 minutes total.
- Stats collected: `mean`, `stddev`, `median`, `p95`, `cv_pct`, run count.
- Keep/drop gate uses classification from `--reference-stats` with uncertainty fallback (`min_uncertainty_pct=3`, `cv_scale=0.5`).

### Jitter Notes
- Direct TTY control runs were very stable (`cv < 1%` for the selected control).
- Under interactive/system load, several FP8 variants showed high variance (`cv` often `25-60%`), and those were marked `Unsure` unless clearly outside uncertainty bounds.

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
