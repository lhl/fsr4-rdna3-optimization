# TODO (gfx1100)

This is the active task tracker for the `gfx1100/` workstream.

## Queue (Priority Order)
- [ ] P0: Lock canonical comparison reference to `results/gfx1100-final-default-optimized-trials3.json` for all upcoming classifications.
- [ ] P0: Make benchmark build output arch-specific (e.g., `build/baseline_kernels_bench.<arch>`) to prevent stale binary reuse when `--arch` changes.
- [ ] P0: Re-run O03 (`items-per-thread=1/2/4`) under current default (`fp8_quantized_io=true`) with `--trials 3` and classify against canonical reference.
- [ ] P0: Re-run O14 with independent inner-loop sweeps (do not tie INT8/FP8 together), using `--trials 3` for top candidates.
- [ ] P0: Re-run O09 (`--split-interior-edge`) with `--trials 3` under current defaults.
- [ ] P0: Decide and document final default policy for mixed mode (`both`) vs per-mode (`int8`/`fp8`) launch knobs.
- [ ] P1: Extended threadgroup sweep beyond O02:
- [ ] P1: run `--threads 192/320/384/512/1024` for `--mode int8` (with canonical reference and at least top-candidate `--trials 3` rerun).
- [ ] P1: run `--threads 192/320/384/512/1024` for `--mode fp8` (with canonical reference and at least top-candidate `--trials 3` rerun).
- [ ] P1: Extended INT8-only tile sweep:
- [ ] P1: run `--mode int8` with `--items-per-thread 8` and `16` (compare against `1/2/4` on current defaults).
- [ ] P1: Scaling sanity sweep (overhead vs steady-state):
- [ ] P1: 2D sweep over `elements` x `inner_*` for both modes, then identify stable-throughput region and re-check candidate wins there.
- [ ] P1: Re-run high-noise candidates with `--trials 3` (`O02 threads=64`, `O15 -O2`, `O15 -Ofast/-ffast-math`) to close uncertainty.
- [ ] P1: Add a small result-summarizer script/table generator for `gfx1100/results/` to auto-rank candidate configs by mode and mixed policy.
- [ ] P1: Evaluate mode-specific defaults explicitly:
- [ ] P1: candidate A (`int8`: items=4, `fp8`: items=1)
- [ ] P1: candidate B (`int8`: inner=16, `fp8`: inner=8)
- [ ] P1: candidate C (combined A+B if stable)
- [ ] P2: Add ISA-direct packed INT8 dot variant using `__builtin_amdgcn_sdot4` + dword loads in `benchmarks/baseline_kernels_bench.cpp` and benchmark vs current packed/scalar paths.
- [ ] P2: Add `__restrict__` and alignment assumptions (`__builtin_assume_aligned`) on hot pointers, then run A/B to validate codegen/perf impact.
- [ ] P2: Add ILP variant (multiple accumulators per thread, e.g. `acc0/acc1`) and benchmark for latency-hiding gains.
- [ ] P2: Add a conv-like microkernel with real inter-thread tile reuse and re-run LDS experiments (O10-O13 style) on that kernel.
- [ ] P2: Profile top 2 FP8 candidates with ROCProfiler (`rocprofv3`) to verify if gains are compute- or memory-bound.
- [ ] P3: Prototype FP8 WMMA/MFMA-style microkernel (rocWMMA/MFMA path) to better match real FSR4 FP8 behavior than scalar `fmaf` path.
- [ ] P2: ISA disassembly comparison (`--save-temps` or `llvm-objdump`) of scalar vs packed INT8 on gfx1100 vs gfx1151 to understand why the O06 delta is 7% vs 32%.
- [ ] P2: Test `--force-packed-int8-io` combined with `items-per-thread=4` on gfx1100 -- the smaller packed penalty here might compose differently with the items=4 INT8 gain.
- [ ] P2: Sweep `reps-per-run` (50/100/200/400) to check if kernel launch overhead is material at these timescales on dGPU vs iGPU.
- [ ] P3: Cross-validate O19 (FP8 quantized-IO) on gfx1151 under direct TTY with `--trials 5` -- the gfx1100 result suggests it's a real win that iGPU noise hid.

## In Progress
- [ ] P0: Prepare and execute the post-default retest batch (O03/O09/O14) with canonical reference and `--trials 3`.

## Blocked
- [ ] None.

## Completed
- [x] Isolated `gfx1100/` workspace scaffolding.
- [x] Environment sanity checks for W7900 (`gfx1100`) in `therock`.
- [x] First smoke benchmark run (`target-seconds=1`).
- [x] First protocol-style baseline run (`target-seconds=60`, mode both).
- [x] Capture compare-ready `gfx1100` baseline (`--trials 3`).
- [x] Run O02 initial threadgroup sweep (`threads=64/128`).
- [x] Run O03 initial items-per-thread sweep (`2/4`).
- [x] Run O05 runtime inner-loops comparison.
- [x] Run O06 packed-vs-scalar INT8 I/O comparison.
- [x] Execute full optimization queue batch (`O02`-`O20`) with per-run stats/classification outputs.
- [x] Analyze full `O02`-`O20` batch outcomes and choose gfx1100 defaults.
- [x] Adopt `O19` (`fp8_quantized_io`) as default for gfx1100 FP8 path.
- [x] Publish gfx1100 README final results and cross-target comparison vs recorded gfx1151 final.
