# FSR4 RDNA3 (gfx1100) Benchmark: Cross-Architecture Validation

Secondary benchmark run on Radeon Pro W7900 (`gfx1100`, discrete RDNA3) to test whether the optimizations found on Strix Halo (`gfx1151`, RDNA3.5 iGPU) carry over to a different GPU in the same architecture family. See the [root README](../README.md) for full context on FSR4, the benchmark harness, and what the kernels measure.

**Core question**: Do the same HIP microkernel optimizations that helped on RDNA3.5 also help on RDNA3 -- and where do the architectures diverge?

## Key Findings

The short answer is: most optimizations transfer, but the two architectures have opposite performance profiles for INT8 vs FP8, and the single biggest gfx1151 win did not reproduce on gfx1100.

- **INT8 is faster than FP8 on both GPUs.** This is the most important takeaway: regardless of architecture, INT8 dot-product compute is faster than FP8 conversion+FMA in this harness. On gfx1100, INT8 is 1.46x faster; on gfx1151, INT8 is 3.70x faster. If you're choosing a quantization precision for latency, INT8 wins everywhere we tested.
- **Between the two GPUs, gfx1100 is ~2x faster at FP8 while gfx1151 is ~27% faster at INT8.** The discrete GPU's dedicated VRAM and FP pipeline handle FP8 conversion+FMA much better than the iGPU's shared DRAM. The iGPU has better INT8 dot-product throughput per-kernel despite having far fewer CUs. But both GPUs still run INT8 faster than FP8 in absolute terms.
- **The biggest gfx1151 win didn't carry over.** On gfx1151, switching from packed `amd_mixed_dot` to scalar INT8 MAC improved performance by ~32% -- the single largest optimization. On gfx1100, scalar is still slightly better than packed, but the delta is small and not a significant win.
- **gfx1100 found its own wins not seen on gfx1151.** Processing 4 items per thread (instead of 1) improved INT8 by ~11.5%. Reducing the FP8 inner loop depth from 16 to 8 iterations improved FP8 by ~6%. Neither of these helped on gfx1151.
- **Both architectures agree on what hurts.** LDS staging, per-iteration requantization, runtime (non-unrolled) inner loops, unfused post-ops, and two-pass dispatch all regressed on both GPUs. These are robust anti-patterns for this workload class.
- **Modest per-mode improvements on gfx1100: INT8 +12%, FP8 +22%** over the raw baseline, achieved via architecture-tuned config for each mode rather than a one-size-fits-all setting.

## Cross-Architecture Comparison

All numbers are mean kernel execution time in milliseconds for a single dispatch of 262,144 logical vectors. Lower is better.

| Mode | gfx1100 (W7900) | gfx1151 (Strix Halo) | Faster GPU | Delta |
|---|---:|---:|---|---:|
| INT8 | 0.006834 ms | 0.005376 ms | gfx1151 | 27% faster |
| FP8 | 0.010006 ms | 0.019868 ms | gfx1100 | 50% faster |
| FP8/INT8 ratio | 1.46x | 3.70x | -- | -- |

### What the Numbers Mean

Each measurement is the wall-clock time for one GPU kernel dispatch that performs quantized dot-product (INT8) or FP8 conversion+FMA arithmetic on 262,144 input vectors. This emulates the per-pass compute pattern of FSR4's quantized convolution shaders (see [root README](../README.md#what-we-benchmark) for details on what the kernels do and how they relate to real FSR4).

The FP8/INT8 ratio tells you how much slower FP8 is relative to INT8 on each GPU. INT8 is faster than FP8 on both GPUs -- the question is by how much. On gfx1100, FP8 is 1.46x slower than INT8, a moderate gap. On gfx1151, FP8 is 3.70x slower, a much larger penalty. So while gfx1100 is the faster GPU for FP8 in cross-arch comparison, INT8 is still the faster precision on gfx1100 itself.

### Why the Architectures Differ

**FP8 gap (~2x, gfx1100 wins):** The W7900 is a discrete GPU with dedicated 48 GB GDDR6 VRAM, while Strix Halo's iGPU shares system DRAM with the CPU. The FP8 kernel is conversion-heavy (FP8-to-FP32 on load, FP32-to-FP8 on store) and the discrete GPU's dedicated memory controller handles this traffic with lower latency and higher bandwidth. The W7900 also has 96 CUs vs 16 on the iGPU, providing more FP32 ALU throughput for the FMA accumulation.

**INT8 gap (~27%, gfx1151 wins):** This is more surprising -- the iGPU with fewer CUs is faster per-kernel at INT8. RDNA3.5 may have improved integer dot-product throughput or scheduling relative to RDNA3. The ISA disassembly shows nearly identical instruction shapes across both architectures (same `v_dot4` and `v_mul/v_mad` counts), so the difference is likely in execution throughput or latency rather than codegen.

**Scalar vs packed INT8 delta (big on gfx1151, small on gfx1100):** On gfx1151, scalar element-wise INT8 MAC outperformed the packed `amd_mixed_dot` intrinsic by ~32%. On gfx1100, the same test showed scalar is still slightly better but within noise. The ISA is the same in both cases, so this is a microarchitectural throughput difference -- gfx1151 appears to have a larger penalty for the packed dot-product instruction relative to scalar arithmetic.

### What Transfers Across Architectures

| Optimization | gfx1151 | gfx1100 | Transfers? |
|---|---|---|---|
| Scalar INT8 > packed dot | +32% (big win) | Small/noise | Direction yes, magnitude no |
| Compile-time loop unrolling | +12% | Keep (default) | Yes |
| Store-time quantization (not per-iter) | Requant: +194% INT8, +476% FP8 regression | Requant: catastrophic regression | Yes |
| LDS staging (all variants) | All regressed | All regressed | Yes (anti-pattern) |
| 256 threads optimal | Yes | Yes | Yes |
| `items_per_thread=4` for INT8 | Not significant | +11.5% | gfx1100-specific |
| `inner_fp8=8` (shorter inner loop) | Not significant | +6% | gfx1100-specific |
| `fp8_quantized_io` (O19) | Unsure (noisy) | Keep (+17% FP8) | gfx1100 confirmed |

## Before / After Performance on gfx1100

The optimization sweep produced modest but consistent improvements over the untuned baseline: **+12% on INT8, +22% on FP8**. A real workload would use one precision or the other, not both simultaneously.

| Stage | What It Is | INT8 (ms) | FP8 (ms) |
|---|---|---:|---:|
| **Baseline** | Default config, no optimizations | 0.007754 | 0.012805 |
| **Tuned** | Best config per mode | 0.006834 | 0.010006 |
| **Improvement** | | +11.9% | +21.9% |

No single config improved both modes -- the knobs that helped INT8 were neutral or harmful for FP8 and vice versa. Tuning each mode independently let us capture wins for both:

- **INT8 tuning**: `items_per_thread=4` (processes 4 vectors per thread instead of 1) gave +11.9%. Higher values (8, 16) regressed; the sweet spot is 4.
- **FP8 tuning**: `fp8_quantized_io` (avoids redundant FP8 requantization on I/O) was the single largest FP8 win at ~17%. On top of that, `inner_fp8=8` (halves the inner loop depth from 16 to 8) added another ~6%.

Both modes share the same defaults for everything else (256 threads, scalar INT8 I/O, compile-time unrolled loops, store-time quantization).

## Benchmark Protocol

Protocol matches the [root project](../README.md#benchmark-methodology) so cross-architecture comparisons are apples-to-apples.

- **Core command**: `./baseline-benchmark.py --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --threads 256`
- **Trials**: Shared Config baseline used 3 trials; final Tuned Config confirmation used 5 trials per mode.
- **Classification gate**: median metric, `min_uncertainty_pct=3`, `cv_scale=0.5`. Verdicts: `keep` / `drop` / `unsure`. Only stable, directionally consistent improvements were accepted.
- **Decision policy**: For mixed INT8+FP8 behavior, per-mode tuning is allowed when a shared config forces a tradeoff.

## Optimization Sweep Overview

Two rounds of experiments were run:

- **O-series (O01-O20)**: Replicate the same 20 optimizations tested on gfx1151 in the [root project](../README.md#optimization-attempts-single-glance). These test whether gfx1151 findings carry over to gfx1100.
- **P-series (P0-P3)**: Additional gfx1100-specific follow-up experiments, organized by priority. P0 = rerun ambiguous O-series results with more trials. P1 = extended parameter sweeps. P2 = ISA-level investigations (disassembly, profiling). P3 = blocked experiments (WMMA prototype, cross-arch execution).

Summary: 5 of the 20 O-series optimizations were kept on gfx1100 (vs 5 on gfx1151), 11 dropped, 4 unsure. The P-series refined the winners into the final Tuned Config.

## Environment

### gfx1100 (This Machine)

| Item | Value |
|---|---|
| GPU | AMD Radeon Pro W7900 (discrete, 48 GB GDDR6) |
| Architecture | RDNA3 (`gfx1100`), 96 CUs |
| HIP/ROCm | HIP `7.2.26043-9999` (clang 22) |
| ROCm SDK | `7.12.0a20260304` |
| Mamba env | `therock` |

### gfx1151 (Reference, from Root Project)

| Item | Value |
|---|---|
| GPU | Radeon 8060S Graphics (Strix Halo iGPU, shared DRAM) |
| Architecture | RDNA3.5 (`gfx1151`), 16 CUs |
| HIP/ROCm | HIP `7.12.60490-128c4eea36` |
| ROCm SDK | `7.12.0a20260226` |

Cross-architecture numbers use recorded gfx1151 artifacts from the root project (`../results/baseline-benchmark-20260227-052146.json`), not local execution -- gfx1151 binaries cannot run on a gfx1100 host (see [attempted cross-execution](#attempted-gfx1151-execution-on-w7900-host) below).

---

## Detailed Tracking

Everything below is the full experiment log preserved for reproducibility. The sections above are sufficient for understanding the results.

### Attempted gfx1151 Execution on W7900 Host

Direct execution of true `gfx1151` binaries was attempted and failed as expected on this `gfx1100` host.

- `--arch gfx1151 --force-rebuild` builds successfully but aborts at runtime with HIP code-object assertion (`SIGABRT`) because `gfx1151` code objects cannot dispatch on `gfx1100`.
- Evidence:
  - `results/todo-p3-gfx1151-o19-trials5-attempt-status.json`
  - `results/todo-p3-gfx1151-o19-trials5-attempt.log`

### O-Series Sweep (Replicating gfx1151 Optimizations)

Baseline reference: `results/baseline-benchmark-20260305-185902.json` (3 trials).

| ID | Change | Outcome | Headline Result |
|---|---|---|---|
| O01 | Protocol lock | Keep | Stable 60s baseline captured. |
| O02 | Threadgroup sweep (`64/128/256`) | Unsure | `64/128` not a confident win over 256. |
| O03 | Items-per-thread sweep (`1/2/4`) | Unsure | `4` helped INT8 but hurt FP8 in shared mode. |
| O04 | Wave-size verification | Keep | Confirmed `warpSize=32`. |
| O05 | Unrolled vs runtime inner loops | Drop | Runtime loops strongly regressed FP8. |
| O06 | Packed vs scalar INT8 I/O | Drop | Packed regressed vs scalar (same direction as gfx1151, much smaller delta). |
| O07 | Hoist scale/bias loads | Drop | In-loop load path regressed. |
| O08 | Per-iter requant vs once-at-store | Drop | Catastrophic regression on both modes. |
| O09 | Interior/edge split dispatch | Unsure | Small/noisy movement. |
| O10 | LDS stage input | Drop | Slower, high variance. |
| O11 | LDS stage input+weight | Drop | Slower than O10. |
| O12 | LDS padding/swizzle proxy | Drop | Stable but much slower. |
| O13 | LDS double-buffer proxy | Drop | Slightly better than O12, still much slower. |
| O14 | Occupancy/register sweep | Unsure | Mixed; FP8 `inner=8` looked promising. |
| O15 | Compile flags (`-O2`, `-Ofast/-ffast-math`) | Unsure | Inconclusive vs `-O3`. |
| O16 | Unfused post-op path | Drop | Large regression. |
| O17 | Two-pass adjacent path | Drop | Large regression. |
| O18 | Mixed INT8 subpath | Drop | Large INT8 regression. |
| O19 | FP8 quantized-IO | Keep | Clear FP8 improvement (~17%). |
| O20 | Final control | Unsure | Near baseline (no drift, no clear win). |

### P-Series Sweep (gfx1100-Specific Follow-Up)

| Item | Status | Result |
|---|---|---|
| P0: Lock canonical reference | Completed | All classifications now reference `results/gfx1100-final-default-optimized-trials3.json`. |
| P0: Arch-specific build output | Completed | Binary names by arch (`build/baseline_kernels_bench.<arch>`). |
| P0: O03 rerun (`items=1/2/4`, trials=3) | Completed | `items=4` keeps INT8 but drops FP8; mode-specific policy needed. |
| P0: O14 independent sweeps | Completed | FP8 `inner=8` is a keep (~+6%); INT8 inner sweeps remain unsure. |
| P0: O09 rerun (trials=3) | Completed | Overall `unsure`. |
| P0: Mixed-policy decision | Completed | Selected mode-specific candidate C (`int8 items=4`, `fp8 inner=8`). |
| P1: Extended thread sweep | Completed | No confident winner vs threads=256 for either mode. |
| P1: INT8 items sweep (`8/16`) | Completed | `8` and `16` regress; `4` remains best. |
| P1: Scaling sanity sweep | Completed | Stable low-CV region at protocol size. |
| P1: High-noise reruns (trials=3) | Completed | O02 threads=64, O15 -O2, O15 -Ofast all remain `unsure`. |
| P1: Result summarizer | Completed | `summarize_results.py` + `results/todo-summary.md`. |
| P1: Candidate A/B/C evaluation | Completed | A keeps INT8, B keeps FP8, C keeps both (selected). |
| P2: ISA direct packed variant | Completed (blocked) | `__builtin_amdgcn_sdot4` compile fails on gfx1100 (`dot1-insts` feature). |
| P2: `restrict` + `assume_aligned` A/B | Completed | Overall `unsure` (no measurable gain). |
| P2: ILP2 INT8 | Completed | Overall `unsure`. |
| P2: Conv-like + LDS rerun | Completed | All variants dropped; worst case ~-209%. |
| P2: Packed INT8 + items=4 | Completed | Overall `unsure` / slight regression. |
| P2: Reps-per-run sweep | Completed | `reps>=200` stable; `50` shows overhead bias. |
| P2: ROCProfiler on FP8 candidates | Completed | inner=8 kernel avg ~7.5us vs baseline ~8.1us (~7.3% faster). |
| P2: Disassembly comparison gfx1100 vs gfx1151 | Completed | Similar instruction shapes; O06 delta is hardware throughput, not ISA divergence. |
| P3: FP8 WMMA/MFMA prototype | Blocked | `fp8-insts` missing for gfx1100/gfx1151; rocWMMA FP8 is gfx12-gated. |
| P3: gfx1151 O19 cross-validate | Blocked | gfx1151 code object cannot execute on W7900/gfx1100. |

### Key Artifacts

- Final policy runs:
  - `results/gfx1100-final-policy-int8-items4-inner16-trials5.json`
  - `results/gfx1100-final-policy-fp8-items1-inner8-trials5.json`
  - `results/gfx1100-final-policy-summary.json`
- TODO sweep summary:
  - `results/todo-summary.md`
  - `results/todo-summary.json`
- P2 profiling/disassembly:
  - `results/todo-p2-rocprof-status.json`
  - `results/todo-p2-rocprof-fp8-baseline_kernel_stats.csv`
  - `results/todo-p2-rocprof-fp8-inner8_kernel_stats.csv`
  - `results/todo-p2-disasm-gfx1100-summary.json`
  - `results/todo-p2-disasm-gfx1151-summary.json`
  - `results/todo-p2-disasm-int8-kernels-compare.json`
- P3 blocked evidence:
  - `results/todo-p3-fp8-wmma-mfma-prototype-status.json`
  - `results/todo-p3-gfx1151-o19-trials5-attempt-status.json`

### Tracking Files

- `WORKLOG.md` -- run-by-run log
- `IMPLEMENTATION.md` -- commands and per-optimization details
- `TODO.md` -- active punchlist
