# TODO (gfx1100)

This is the active task tracker for the `gfx1100/` workstream.

## Queue (Priority Order)
- [ ] None.

## In Progress
- [ ] None.

## Blocked
- [ ] P3: Prototype FP8 WMMA/MFMA-style microkernel (rocWMMA/MFMA path).
  - Status: blocked by toolchain/target feature availability.
  - Evidence: `results/todo-p3-fp8-wmma-mfma-prototype-status.json` (`fp8-insts` missing for `gfx1100`/`gfx1151`).
- [ ] P3: Cross-validate O19 (FP8 quantized-IO) on gfx1151 with `--trials 5`.
  - Status: blocked on this host (W7900/gfx1100 cannot execute gfx1151 code object).
  - Evidence: `results/todo-p3-gfx1151-o19-trials5-attempt-status.json`.

## Completed
- [x] P0: Lock canonical comparison reference to `results/gfx1100-final-default-optimized-trials3.json` for all classifications.
- [x] P0: Make benchmark build output arch-specific (`build/baseline_kernels_bench.<arch>`).
- [x] P0: Re-run O03 (`items-per-thread=1/2/4`) under current defaults with `--trials 3`.
  - Artifacts: `results/todo-p0-o03-items{1,2,4}-trials3*.json`.
- [x] P0: Re-run O14 with independent INT8/FP8 inner-loop sweeps.
  - Artifacts: `results/todo-p0-o14-*-classification.json`.
- [x] P0: Re-run O09 (`--split-interior-edge`) with `--trials 3`.
  - Artifacts: `results/todo-p0-o09-split-interior-edge-trials3*.json`.
- [x] P0: Decide and document final default policy for mixed mode (`both`) vs per-mode (`int8`/`fp8`) launch knobs.
  - Decision: per-mode policy wins (`int8: items=4`, `fp8: inner=8`), with shared-default retained as comparison baseline.

- [x] P1: Extended threadgroup sweep beyond O02.
  - INT8: `threads=192/320/384/512/1024` completed.
  - FP8: `threads=192/320/384/512/1024` completed.
- [x] P1: Extended INT8-only tile sweep (`items-per-thread=8/16`, compared vs `1/2/4`).
- [x] P1: Scaling sanity sweep (`elements x inner`) for both modes.
- [x] P1: Re-run high-noise candidates with `--trials 3` (`O02 threads=64`, `O15 -O2`, `O15 -Ofast/-ffast-math`).
- [x] P1: Add result summarizer for `gfx1100/results/` auto-ranking.
  - Script: `summarize_results.py`
  - Output: `results/todo-summary.md`, `results/todo-summary.json`.
- [x] P1: Evaluate mode-specific defaults candidates.
  - A: `int8 items=4`, `fp8 items=1`.
  - B: `int8 inner=16`, `fp8 inner=8`.
  - C: combined A+B (selected final mode-specific policy).

- [x] P2: Add ISA-direct packed INT8 dot variant (`__builtin_amdgcn_sdot4`) and benchmark vs packed/scalar paths.
  - Builtin path blocked for gfx1100 (`dot1-insts` target feature), fallback path benchmarked.
  - Artifacts: `results/todo-p2-isa-*.json`, `results/todo-p2-isa-builtin-sdot4-attempt-status.json`.
- [x] P2: Add `__restrict__` + `__builtin_assume_aligned` hot-pointer A/B.
  - Artifact: `results/todo-p2-assume-aligned-hot-ptrs-trials3-classification.json` (`unsure`).
- [x] P2: Add ILP2 variant and benchmark.
  - Artifact: `results/todo-p2-ilp2-int8-trials3-classification.json` (`unsure`).
- [x] P2: Add conv-like microkernel and rerun LDS experiments.
  - Artifacts: `results/todo-p2-convlike-*.json` (all `drop`, including trials=3 rerun).
- [x] P2: Profile top FP8 candidates with ROCProfiler.
  - Working method: `rocprofv3 --kernel-trace --stats --summary`.
  - Artifacts: `results/todo-p2-rocprof-fp8-*_kernel_stats.csv`, `results/todo-p2-rocprof-status.json`.
- [x] P2: ISA disassembly comparison (gfx1100 vs gfx1151).
  - Artifacts: `results/todo-p2-disasm-gfx1100-summary.json`, `results/todo-p2-disasm-gfx1151-summary.json`, `results/todo-p2-disasm-int8-kernels-compare.json`.
- [x] P2: Test packed INT8 + `items-per-thread=4` composition.
  - Artifact: `results/todo-p2-packed-items4-int8-trials3-classification.json` (`unsure`).
- [x] P2: Sweep `reps-per-run` (`50/100/200/400`) for launch-overhead sensitivity.
  - Artifacts: `results/todo-p2-reps-sweep-*.json`.

- [x] Isolated `gfx1100/` workspace scaffolding.
- [x] Environment sanity checks for W7900 (`gfx1100`) in `therock`.
- [x] First smoke benchmark run (`target-seconds=1`).
- [x] First protocol baseline run (`target-seconds=60`, mode both).
- [x] Capture compare-ready `gfx1100` baseline (`--trials 3`).
- [x] Run initial optimization queue (`O02`-`O20`) and choose defaults.
- [x] Adopt `O19` (`fp8_quantized_io`) as default.
- [x] Publish updated final numbers and extra-optimization status in README.
