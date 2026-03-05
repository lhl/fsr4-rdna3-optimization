# IMPLEMENTATION (gfx1100)

## 2026-03-06: Full TODO Sweep Closure + Final Policy Integration
- Goal:
  - Finish all queued `gfx1100/TODO.md` items, keep only proven wins, and publish final numbers/status.
- Core execution:
  - Continued and completed `./run_todo_p2.sh` (P2 queue).
  - Added recovery/resume path for failed profiling tail:
    - `./run_todo_p2_resume_after_reps.sh` (profiling + disassembly tail-only recovery).
- Script/tooling updates:
  - `gfx1100/run_todo_p2.sh`
    - fixed `rocprofv3` app separator usage (`--`)
    - switched profiling flow to:
      - `--kernel-trace --stats --summary`
      - csv outputs per candidate
      - status-json emission instead of hard-failing whole batch
    - fixed disassembly regex escaping for instruction counts
    - added `todo-p2-disasm-int8-kernels-compare.json` generation for scalar/packed kernel-shape comparison
  - Added final-policy runner:
    - `gfx1100/run_final_policy.sh`
- Final integrated policy validation commands:
  - INT8:
    - `mamba run -n therock ./baseline-benchmark.py --arch gfx1100 --mode int8 --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --threads 256 --items-per-thread 4 --fp8-quantized-io --trials 5 --reference-stats results/gfx1100-final-default-optimized-trials3.json --stats-out results/gfx1100-final-policy-int8-items4-inner16-trials5.json --classification-out results/gfx1100-final-policy-int8-items4-inner16-trials5-classification.json`
  - FP8:
    - `mamba run -n therock ./baseline-benchmark.py --arch gfx1100 --mode fp8 --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-fp8 8 --threads 256 --items-per-thread 1 --fp8-quantized-io --trials 5 --reference-stats results/gfx1100-final-default-optimized-trials3.json --stats-out results/gfx1100-final-policy-fp8-items1-inner8-trials5.json --classification-out results/gfx1100-final-policy-fp8-items1-inner8-trials5-classification.json`
- Final integrated results:
  - INT8 final policy:
    - mean-of-mean `0.006834 ms`
    - mean-cv `1.412%`
    - classification verdict `keep` (`+11.484%` median vs canonical reference)
  - FP8 final policy:
    - mean-of-mean `0.010006 ms`
    - mean-cv `1.568%`
    - classification verdict `keep` (`+6.124%` median vs canonical reference)
  - Combined summary:
    - `results/gfx1100-final-policy-summary.json`
    - canonical total mean `0.018353 ms` -> final policy total mean `0.016839 ms` (`+8.249%`)
- P2 notable conclusions:
  - ISA builtin packed INT8 probe:
    - `__builtin_amdgcn_sdot4` compile attempt fails on gfx1100 (`dot1-insts` feature gate)
    - artifact: `results/todo-p2-isa-builtin-sdot4-attempt-status.json`
  - Assume-aligned hot pointers:
    - `results/todo-p2-assume-aligned-hot-ptrs-trials3-classification.json` -> `unsure`
  - ILP2 INT8:
    - `results/todo-p2-ilp2-int8-trials3-classification.json` -> `unsure`
  - Conv-like + LDS stack:
    - all variants `drop` (worst ~`-208.8%`)
  - Packed+items=4 composition:
    - `results/todo-p2-packed-items4-int8-trials3-classification.json` -> `unsure`/negative
  - Reps-per-run sweep:
    - `50` shows overhead inflation; `200/400` stable near canonical.
  - ROCProfiler:
    - initial `--stats`-only usage failed on this rocprofv3 build
    - rerun with `--kernel-trace --stats --summary` succeeded
    - kernel stats show fp8 inner=8 average kernel duration below fp8 inner=16 baseline
- P3 closure:
  - FP8 WMMA/MFMA prototype: blocked
    - probe source/logs:
      - `results/todo-p3-fp8-mfma-probe.cpp`
      - `results/todo-p3-fp8-mfma-probe-gfx1100.log`
      - `results/todo-p3-fp8-mfma-probe-gfx1151.log`
    - status: `results/todo-p3-fp8-wmma-mfma-prototype-status.json`
  - gfx1151 O19 trials=5 cross-validate on this host: blocked at runtime dispatch
    - status/log:
      - `results/todo-p3-gfx1151-o19-trials5-attempt-status.json`
      - `results/todo-p3-gfx1151-o19-trials5-attempt.log`

## 2026-03-05: Bootstrap
- Goal: replicate benchmark workflow on W7900 (`gfx1100`) in an isolated folder.
- Scope: local harness copy, environment sanity checks, smoke benchmark, initial baseline run.
- Files created:
  - `gfx1100/baseline-benchmark.py`
  - `gfx1100/benchmarks/baseline_kernels_bench.cpp`
  - `gfx1100/README.md`
  - `gfx1100/WORKLOG.md`
  - `gfx1100/IMPLEMENTATION.md`
  - `gfx1100/TODO.md`
- File modified:
  - `gfx1100/baseline-benchmark.py` (default arch changed to `gfx1100`)
- Validation commands:
  - `mamba run -n therock rocm-sdk version`
  - `mamba run -n therock hipcc --version`
  - `mamba run -n therock rocminfo | rg -n "gfx1100|Radeon Pro W7900"`
  - `mamba run -n therock rocm-smi --showproductname --showbus --showdriverversion --showfwinfo`
  - `mamba run -n therock /usr/bin/bash -lc '... hipcc -O2 --offload-arch=gfx1100 ... && <binary>'`
- Smoke benchmark:
  - Command: `mamba run -n therock ./baseline-benchmark.py --mode both --target-seconds 1 --min-runs 5 --max-runs 30 --warmup-runs 2 --reps-per-run 20 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1`
  - Output: `gfx1100/results/baseline-benchmark-20260305-184537.json`
  - INT8: mean `0.004830 ms`, stddev `0.000056`, cv `1.154%`
  - FP8: mean `0.010916 ms`, stddev `0.000168`, cv `1.543%`
- Protocol baseline:
  - Command: `mamba run -n therock ./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1`
  - Output: `gfx1100/results/baseline-benchmark-20260305-184741.json`
  - INT8: mean `0.007771 ms`, stddev `0.000046`, cv `0.591%`
  - FP8: mean `0.012682 ms`, stddev `0.000059`, cv `0.467%`
- Immediate observations:
  - Both modes showed low variance (`cv < 1%`) in the 60s run.
  - Relative to existing gfx1151 final baseline, `gfx1100` is slower for INT8 but faster for FP8 in this harness.

## 2026-03-05: Compare-Ready Baseline (`trials=3`)
- Command:
  - `mamba run -n therock ./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --trials 3`
- Output:
  - `gfx1100/results/baseline-benchmark-20260305-185902.json`
- Aggregated results:
  - INT8: mean-of-mean `0.007754 ms`, stdev-of-mean `0.000032`, mean-cv `1.857%`
  - FP8: mean-of-mean `0.012805 ms`, stdev-of-mean `0.000067`, mean-cv `0.668%`

## 2026-03-05: O02 Threadgroup Sweep (`64/128` vs `256`)
- Reference used during run: `gfx1100/results/baseline-benchmark-20260305-184741.json`
- `threads=64`
  - Command: `mamba run -n therock ./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 64 --items-per-thread 1 --reference-stats results/baseline-benchmark-20260305-184741.json`
  - Stats: `gfx1100/results/baseline-benchmark-20260305-185050.json`
  - Classification: `gfx1100/results/baseline-benchmark-classification-20260305-185050.json`
  - Verdict: `unsure` overall (`int8` run had high variance, cv `6.418%`)
- `threads=128`
  - Command: `mamba run -n therock ./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 128 --items-per-thread 1 --reference-stats results/baseline-benchmark-20260305-184741.json`
  - Stats: `gfx1100/results/baseline-benchmark-20260305-185253.json`
  - Classification: `gfx1100/results/baseline-benchmark-classification-20260305-185253.json`
  - Verdict: `unsure` overall

## 2026-03-05: O06 Packed vs Scalar INT8 I/O
- Reference: `gfx1100/results/baseline-benchmark-20260305-185902.json`
- Command:
  - `mamba run -n therock ./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --force-packed-int8-io --reference-stats results/baseline-benchmark-20260305-185902.json`
- Stats: `gfx1100/results/baseline-benchmark-20260305-190114.json`
- Classification: `gfx1100/results/baseline-benchmark-classification-20260305-190114.json`
- Result:
  - INT8 median delta: `-7.320%` (regression)
  - Overall verdict: `drop`

## 2026-03-05: O05 Runtime Inner Loops
- Reference: `gfx1100/results/baseline-benchmark-20260305-185902.json`
- Command:
  - `mamba run -n therock ./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --force-runtime-inner-loops --reference-stats results/baseline-benchmark-20260305-185902.json`
- Stats: `gfx1100/results/baseline-benchmark-20260305-190318.json`
- Classification: `gfx1100/results/baseline-benchmark-classification-20260305-190318.json`
- Result:
  - FP8 median delta: `-360.084%` (major regression)
  - Overall verdict: `drop`

## 2026-03-05: O03 Items-Per-Thread Sweep (`2/4`)
- Reference: `gfx1100/results/baseline-benchmark-20260305-185902.json`
- `items-per-thread=2`
  - Command: `mamba run -n therock ./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 2 --reference-stats results/baseline-benchmark-20260305-185902.json`
  - Stats: `gfx1100/results/baseline-benchmark-20260305-190523.json`
  - Classification: `gfx1100/results/baseline-benchmark-classification-20260305-190523.json`
  - Result: INT8 `keep` (+4.699%), FP8 `unsure`, overall `unsure`
- `items-per-thread=4`
  - Command: `mamba run -n therock ./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 4 --reference-stats results/baseline-benchmark-20260305-185902.json`
  - Stats: `gfx1100/results/baseline-benchmark-20260305-190726.json`
  - Classification: `gfx1100/results/baseline-benchmark-classification-20260305-190726.json`
  - Result: INT8 `keep` (+11.761%), FP8 `drop` (-9.633%), overall `unsure`

## 2026-03-05: Full Optimization Queue Batch (`O02`-`O20`)
- Driver script:
  - `gfx1100/run_all_optimizations.sh`
- Reference baseline for classification:
  - `gfx1100/results/baseline-benchmark-20260305-185902.json`
- Batch coverage:
  - `O02`, `O03`, `O04`, `O05`, `O06`, `O07`, `O08`, `O09`, `O10`, `O11`, `O12`, `O13`, `O14`, `O15`, `O16`, `O17`, `O18`, `O19`, `O20`
- Outcome count across classification files:
  - `drop=12`, `keep=1`, `unsure=10`
- Per-run metrics and decisions:

| Run | INT8 mean / cv% | FP8 mean / cv% | Overall | Files |
|---|---:|---:|---|---|
| O02 threads=64 | `0.007702 / 6.487` | `0.012618 / 0.594` | `unsure` | `results/o02-threads64.json`, `results/o02-threads64-classification.json` |
| O02 threads=128 | `0.007782 / 0.575` | `0.012638 / 0.595` | `unsure` | `results/o02-threads128.json`, `results/o02-threads128-classification.json` |
| O03 items=2 | `0.007394 / 0.374` | `0.012771 / 0.539` | `unsure` | `results/o03-items2.json`, `results/o03-items2-classification.json` |
| O03 items=4 | `0.006821 / 0.496` | `0.013980 / 0.492` | `unsure` | `results/o03-items4.json`, `results/o03-items4-classification.json` |
| O04 wave check | `0.006877 / 3.944` | `-` | `n/a` | `results/o04-wavecheck.json` |
| O05 runtime loops | `0.007756 / 0.374` | `0.059102 / 0.771` | `drop` | `results/o05-runtime-loops.json`, `results/o05-runtime-loops-classification.json` |
| O06 packed INT8 I/O | `0.008353 / 0.374` | `0.012763 / 0.445` | `drop` | `results/o06-packed-int8-io.json`, `results/o06-packed-int8-io-classification.json` |
| O07 in-loop scale/bias | `0.010153 / 0.639` | `-` | `drop` | `results/o07-inloop-scale-bias.json`, `results/o07-inloop-scale-bias-classification.json` |
| O08 per-iter requant | `0.010771 / 0.436` | `0.054318 / 0.654` | `drop` | `results/o08-per-iter-requant.json`, `results/o08-per-iter-requant-classification.json` |
| O09 interior/edge split | `0.007637 / 0.592` | `0.012583 / 0.597` | `unsure` | `results/o09-split-interior-edge.json`, `results/o09-split-interior-edge-classification.json` |
| O10 LDS input | `0.010092 / 0.633` | `-` | `drop` | `results/o10-lds-stage-input.json`, `results/o10-lds-stage-input-classification.json` |
| O11 LDS input+weight | `0.010222 / 0.469` | `-` | `drop` | `results/o11-lds-stage-input-weight.json`, `results/o11-lds-stage-input-weight-classification.json` |
| O12 LDS padding | `0.013388 / 0.571` | `-` | `drop` | `results/o12-lds-padding.json`, `results/o12-lds-padding-classification.json` |
| O13 LDS double-buffer | `0.012207 / 0.663` | `-` | `drop` | `results/o13-lds-double-buffer.json`, `results/o13-lds-double-buffer-classification.json` |
| O14 threads=128 inner=16 | `0.007832 / 0.565` | `0.012742 / 0.780` | `unsure` | `results/o14-threads128-inner16.json`, `results/o14-threads128-inner16-classification.json` |
| O14 threads=256 inner=8 | `0.007765 / 0.588` | `0.012047 / 0.547` | `unsure` | `results/o14-threads256-inner8.json`, `results/o14-threads256-inner8-classification.json` |
| O14 threads=256 inner=32 | `0.007765 / 0.585` | `0.014328 / 0.592` | `drop` | `results/o14-threads256-inner32.json`, `results/o14-threads256-inner32-classification.json` |
| O15 flag `-O2` | `0.007609 / 6.289` | `0.012852 / 0.801` | `unsure` | `results/o15-flag-o2.json`, `results/o15-flag-o2-classification.json` |
| O15 flag `-Ofast -ffast-math` | `0.007684 / 3.372` | `0.012822 / 0.872` | `unsure` | `results/o15-flag-ofast-ffastmath.json`, `results/o15-flag-ofast-ffastmath-classification.json` |
| O15 reset default build | `0.008705 / 1.829` | `-` | `n/a` | `results/o15-reset-default-build.json` |
| O16 unfused post | `0.015247 / 3.916` | `0.020617 / 0.714` | `drop` | `results/o16-unfused-post.json`, `results/o16-unfused-post-classification.json` |
| O17 two-pass | `0.015291 / 0.568` | `0.020827 / 0.657` | `drop` | `results/o17-two-pass.json`, `results/o17-two-pass-classification.json` |
| O18 mixed INT8 path | `0.009673 / 0.551` | `-` | `drop` | `results/o18-mixed-int8-path.json`, `results/o18-mixed-int8-path-classification.json` |
| O19 FP8 quantized IO | `-` | `0.010615 / 0.851` | `keep` | `results/o19-fp8-quantized-io.json`, `results/o19-fp8-quantized-io-classification.json` |
| O20 final control | `0.007759 / 0.589` | `0.012779 / 0.591` | `unsure` | `results/o20-final-control.json`, `results/o20-final-control-classification.json` |

## 2026-03-05: Finalize gfx1100 Defaults + Cross-Arch Check
- Goal:
  - Align `gfx1100` defaults with selected optimization outcomes and publish README-level final results.
- Code change:
  - `gfx1100/benchmarks/baseline_kernels_bench.cpp`
    - `fp8_quantized_io` default changed from `false` to `true`
- Final default benchmark (no feature flags):
  - Command:
    - `mamba run -n therock ./baseline-benchmark.py --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --trials 3 --reference-stats results/baseline-benchmark-20260305-185902.json --stats-out results/gfx1100-final-default-optimized-trials3.json --classification-out results/gfx1100-final-default-optimized-trials3-classification.json`
  - Stats artifact:
    - `results/gfx1100-final-default-optimized-trials3.json`
  - Classification artifact:
    - `results/gfx1100-final-default-optimized-trials3-classification.json`
  - Aggregate results:
    - INT8: mean-of-mean `0.007713 ms`, mean-cv `2.565%`
    - FP8: mean-of-mean `0.010641 ms`, mean-cv `0.583%`
  - vs prior gfx1100 baseline (`results/baseline-benchmark-20260305-185902.json`):
    - INT8: `+0.54%` faster
    - FP8: `+16.90%` faster
- gfx1151 execution check on this host:
  - Forced command:
    - `mamba run -n therock ./baseline-benchmark.py --arch gfx1151 --force-rebuild --mode both --target-seconds 1 --min-runs 5 --max-runs 20 --warmup-runs 2 --reps-per-run 20 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1`
  - Result:
    - Build succeeds for `gfx1151`, but runtime aborts on W7900 (`gfx1100`) with HIP code-object assertion and `SIGABRT`.
  - Implication:
    - Direct execution of true `gfx1151` kernel binaries is not supported on this machine; cross-target comparison should use recorded `gfx1151` reference data.
- Recovery step:
  - Rebuilt benchmark binary back to `gfx1100` target:
    - `results/gfx1100-rebuild-after-gfx1151-attempt.json`
- Documentation update:
  - Added final before/after and `gfx1100` vs `gfx1151` comparison tables to `gfx1100/README.md`.
