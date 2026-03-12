# Implementation Log

This file records each optimization attempt end-to-end.

## Optimization Loop
1. Pick one optimization ID from `OPTIMIZATION_QUEUE.md`.
2. Implement only that change in `fsr4-src/opt`.
3. Run smoke/validation checks.
4. Run benchmark harness with fixed protocol.
5. Compare against current baseline and previous best (use `--reference-stats` auto-classification).
6. Keep only if net positive and variance is acceptable.
7. Record outcome in this file, then move to next item.

## Benchmark Protocol (default)

```bash
./baseline-benchmark.py --mode both --target-seconds 180 --min-runs 200 --reps-per-run 200
```

Recommended on a busy machine:

```bash
./baseline-benchmark.py --mode both --target-seconds 240 --min-runs 250 --reps-per-run 200
```

Variance gates:
- Prefer `cv_pct <= 3.0` for decision-quality runs.
- If `cv_pct > 3.0`, rerun with longer duration and/or `--trials 3` before deciding.

## Session Entries

### YYYY-MM-DD: OXX <short title>
- Scope:
- Files changed:
- Validation:
- Benchmark command:
- INT8 result: `mean_ms=`, `stddev_ms=`, `cv_pct=`, `median_ms=`, `p95_ms=`
- FP8 result: `mean_ms=`, `stddev_ms=`, `cv_pct=`, `median_ms=`, `p95_ms=`
- Decision: `Keep` or `Rollback`
- Notes:

### 2026-03-12: DOC RDNA4-only source audit
- Scope: Document which released FSR4 source features are actually WMMA or RDNA4-leaning, separate INT8 vs FP8 kernel dependencies, assess the relevance of the cited Reddit claim vocabulary against the released HLSL, and spell out what an RDNA3 or RDNA3.5 FP8-model port would require.
- Files changed: `RDNA4-ONLY.md`, `WORKLOG.md`, `IMPLEMENTATION.md`
- Validation: Re-audited runtime-selection plumbing and operator source files; verified that the active checked-in INT8 path is non-WMMA, that the FP8 path is WMMA-only in source, and that the prepass uses `F16` WMMA while the fused FP8 body keeps re-quantizing intermediates back into FP8 wave-matrix objects.
- Benchmark command: none (documentation-only session)
- INT8 result: none (no benchmark run)
- FP8 result: none (no benchmark run)
- Decision: `Keep`
- Notes: The updated doc now distinguishes source-visible dependencies (`dot2add`, `dot4add_i8packed`, `AmdWaveMatrixMultiply`, `groupshared`, `SampleLevel`) from backend-style counter names in the Reddit comment (`loadcnt`, `samplecnt`, `storecnt`, `bvhcnt`, `kmcnt`, `dscnt`), and concludes that an RDNA3-capable FP8-model path would need a separate `F16`-WMMA shader family rather than a trivial format toggle in the existing FP8 HLSL.

### 2026-02-27: O01 Benchmark Protocol Lock (initial capture)
- Scope: Establish baseline harness with minimum-duration sampling and variance metrics.
- Files changed: `baseline-benchmark.py`, `benchmarks/baseline_kernels_bench.cpp`, `README.md`, `OPTIMIZATION_QUEUE.md`
- Validation: Harness build+run succeeded; JSON artifact generated.
- Benchmark command: `./baseline-benchmark.py --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16`
- INT8 result: `mean_ms=0.008455`, `stddev_ms=0.000783`, `cv_pct=9.264`, `median_ms=0.008203`, `p95_ms=0.010122`
- FP8 result: `mean_ms=0.122223`, `stddev_ms=0.008459`, `cv_pct=6.921`, `median_ms=0.120998`, `p95_ms=0.129231`
- Decision: `Keep`
- Notes: Usable for progress tracking, but not decision-quality due high variance under interactive load; use longer runs and `--trials 3` for optimization keep/rollback calls.

### 2026-02-27: O01 Duration Sweep (2m/3m/5m target per mode)
- Scope: Measure how variance changes with longer target durations while machine is actively used.
- Files changed: none (benchmark-only run set).
- Validation: Three benchmark runs completed; artifacts saved.
- Benchmark command(s):
  - `./baseline-benchmark.py --no-build --mode both --target-seconds 120 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16`
  - `./baseline-benchmark.py --no-build --mode both --target-seconds 180 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16`
  - `./baseline-benchmark.py --no-build --mode both --target-seconds 300 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16`
- INT8 results:
  - `120s`: `runs=69893`, `mean_ms=0.008508`, `stddev_ms=0.002584`, `cv_pct=30.377`
  - `180s`: `runs=63577`, `mean_ms=0.014052`, `stddev_ms=0.008598`, `cv_pct=61.191`
  - `300s`: `runs=107432`, `mean_ms=0.013862`, `stddev_ms=0.008452`, `cv_pct=60.968`
- FP8 results:
  - `120s`: `runs=3922`, `mean_ms=0.152833`, `stddev_ms=0.030432`, `cv_pct=19.912`
  - `180s`: `runs=4499`, `mean_ms=0.199776`, `stddev_ms=0.030858`, `cv_pct=15.446`
  - `300s`: `runs=7909`, `mean_ms=0.189382`, `stddev_ms=0.029693`, `cv_pct=15.679`
- Decision: `Keep` (data captured), but not decision-quality for keep/rollback optimization gating.
- Notes: Under active interactive use, longer duration alone did not stabilize variance; next step should be multi-trial runs (`--trials 3`) and median-of-trials for decisions.

### 2026-02-27: O01 Baseline Lock (direct TTY, 60s)
- Scope: Lock a low-jitter baseline in direct TTY before optimization iterations.
- Files changed: none (benchmark-only run).
- Validation: Benchmark completed with low variance in both modes.
- Benchmark command: `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256`
- INT8 result: `runs=38233`, `mean_ms=0.007743`, `stddev_ms=0.000096`, `cv_pct=1.239`, `median_ms=0.007754`, `p95_ms=0.007888`
- FP8 result: `runs=2554`, `mean_ms=0.117392`, `stddev_ms=0.001498`, `cv_pct=1.276`, `median_ms=0.117125`, `p95_ms=0.120338`
- Decision: `Keep`
- Notes: This is the active reference baseline for optimization comparisons.

### 2026-02-27: O02 Threadgroup Sweep (`64`, `128`, `256`)
- Scope: Evaluate threadgroup size as a low-effort optimization candidate.
- Files changed: none (parameter sweep only).
- Validation: All three runs completed and artifacts saved.
- Benchmark command(s):
  - `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 64`
  - `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 128`
  - Baseline reference: `threads=256` (entry above)
- INT8 results:
  - `threads=64`: `mean_ms=0.009068`, `cv_pct=50.129`, `delta=+17.11%`
  - `threads=128`: `mean_ms=0.010435`, `cv_pct=62.409`, `delta=+34.77%`
  - `threads=256`: `mean_ms=0.007743`, `cv_pct=1.239`, `delta=baseline`
- FP8 results:
  - `threads=64`: `mean_ms=0.129159`, `cv_pct=9.803`, `delta=+10.02%`
  - `threads=128`: `mean_ms=0.146970`, `cv_pct=14.178`, `delta=+25.20%`
  - `threads=256`: `mean_ms=0.117392`, `cv_pct=1.276`, `delta=baseline`
- Decision: `Keep` `threads=256`
- Notes: `threads=64` and `128` appear slower and high-variance; treated as `Unsure (likely worse)` under variance rules.

### 2026-02-27: O03 Per-Thread Tile Sweep (`items-per-thread=1,2,4`)
- Scope: Approximate per-thread output tile sizing in the harness via `items-per-thread`.
- Files changed: `benchmarks/baseline_kernels_bench.cpp`, `baseline-benchmark.py`
- Validation: Build succeeded and sweep runs completed.
- Benchmark command(s):
  - `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 2`
  - `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 4`
  - Baseline reference: `items-per-thread=1` (O01 entry)
- INT8 results:
  - `items=2`: `mean_ms=0.008756`, `cv_pct=1.265`, `delta=+13.08%`
  - `items=4`: `mean_ms=0.011774`, `cv_pct=54.322`, `delta=+52.06%`
  - `items=1`: `mean_ms=0.007743`, `cv_pct=1.239` (baseline)
- FP8 results:
  - `items=2`: `mean_ms=0.121350`, `cv_pct=1.222`, `delta=+3.37%`
  - `items=4`: `mean_ms=0.143581`, `cv_pct=9.483`, `delta=+22.31%`
  - `items=1`: `mean_ms=0.117392`, `cv_pct=1.276` (baseline)
- Decision: `Keep` `items-per-thread=1`
- Notes: `items=2` is confidently worse; `items=4` is high-variance and much slower, treated as likely worse.

### 2026-02-27: O04 Wave-Size Verification
- Scope: Verify wave-size assumptions on current `gfx1151` target before deeper kernel changes.
- Files changed: `benchmarks/baseline_kernels_bench.cpp` (added `warpSize` to INFO output)
- Validation: Harness run and JSON info inspection completed.
- Benchmark command: `./baseline-benchmark.py --mode int8 --target-seconds 0.1 --min-runs 2 --max-runs 20 --warmup-runs 1 --reps-per-run 10 --elements 32768 --inner-int8 8 --threads 256 --items-per-thread 1`
- Result: `warpSize=32` reported by `hipDeviceProp_t` on `gfx1151`.
- Decision: `Keep` (assumption confirmed)
- Notes: No explicit wave64-only logic exists in current harness kernels; this pass is verification-only.

### 2026-02-27: O05 Inner-Loop Unroll Sweep
- Scope: Compare compile-time-unrolled inner loops vs forced runtime inner loops.
- Files changed:
  - `benchmarks/baseline_kernels_bench.cpp` (templated unrolled kernels + runtime fallback switch)
  - `baseline-benchmark.py` (`--force-runtime-inner-loops` pass-through)
- Validation: Build and both A/B runs succeeded.
- Benchmark command(s):
  - Unrolled default:
    - `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1`
  - Forced runtime:
    - `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --force-runtime-inner-loops`
- INT8 results:
  - unrolled: `mean_ms=0.007816`, `cv_pct=1.347`
  - runtime: `mean_ms=0.008767`, `cv_pct=38.897`, `delta=+12.17%`
- FP8 results:
  - unrolled: `mean_ms=0.113599`, `cv_pct=1.335`
  - runtime: `mean_ms=0.124402`, `cv_pct=6.198`, `delta=+9.51%`
- Decision: `Keep` unrolled path
- Notes: Runtime-loop leg showed high variance and slower means; treated as likely worse under uncertainty gate.

### 2026-02-27: O06 INT8 I/O Path Sweep (packed vs scalar)
- Scope: Compare packed INT8 path against scalar INT8 path while keeping FP8 path unchanged.
- Files changed:
  - `benchmarks/baseline_kernels_bench.cpp` (added scalar INT8 kernels and `--force-packed-int8-io`)
  - `baseline-benchmark.py` (`--force-packed-int8-io` pass-through)
- Validation: Build succeeded; packed and scalar legs completed, scalar leg repeated for confirmation.
- Benchmark command(s):
  - Packed reference:
    - `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --force-packed-int8-io`
  - Scalar candidate:
    - `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1`
- INT8 results:
  - packed: `mean_ms=0.007835`, `cv_pct=1.501`
  - scalar: `mean_ms=0.005340`, `cv_pct=0.397`, `delta=-31.84%`
- FP8 results:
  - packed run: `mean_ms=0.113683`, `cv_pct=1.409`
  - scalar-default run: `mean_ms=0.115609`, `cv_pct=1.828`, `delta=+1.69%`
- Decision: `Keep` scalar INT8 I/O as default
- Notes: INT8 gain is large and stable; FP8 shift is small and within uncertainty band for current protocol.

### 2026-02-27: O07 Hoist Quant/Bias Loads
- Scope: Validate in-loop invariant load hoisting by comparing against forced in-loop quant/bias reads.
- Files changed:
  - `benchmarks/baseline_kernels_bench.cpp` (`--force-inloop-scale-bias` + advanced scalar path)
  - `baseline-benchmark.py` (flag pass-through)
- Benchmark command:
  - `./baseline-benchmark.py --no-build --mode int8 --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --force-inloop-scale-bias --reference-stats results/baseline-benchmark-20260227-052146.json`
- INT8 result: `mean_ms=0.012061`, `stddev_ms=0.000121`, `cv_pct=1.001`, `median_ms=0.012083`, `p95_ms=0.012175`
- FP8 result: `n/a`
- Decision: `Keep` hoisted/default path (`force-inloop` classified `drop`)
- Notes: Forced in-loop invariant reads are >2x slower than control.

### 2026-02-27: O08 Requant Once vs Per-Iter
- Scope: Compare full-accumulate + once-at-store behavior against forced per-iteration requant.
- Files changed:
  - `benchmarks/baseline_kernels_bench.cpp` (`--force-per-iter-requant` + fp8 general kernel path)
  - `baseline-benchmark.py` (flag pass-through)
- Benchmark command:
  - `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --force-per-iter-requant --reference-stats results/baseline-benchmark-20260227-052146.json`
- INT8 result: `mean_ms=0.015792`, `stddev_ms=0.004207`, `cv_pct=26.640`, `median_ms=0.015153`, `p95_ms=0.015574`
- FP8 result: `mean_ms=0.114545`, `stddev_ms=0.001636`, `cv_pct=1.428`, `median_ms=0.113919`, `p95_ms=0.118762`
- Decision: `Keep` once-at-store/default (`force-per-iter` classified `drop`)
- Notes: Per-iteration requant heavily regresses both modes.

### 2026-02-27: O09 Interior/Edge Split
- Scope: Test split interior + tail launch to avoid bounds checks on the hot path.
- Files changed:
  - `benchmarks/baseline_kernels_bench.cpp` (`--split-interior-edge` + no-bounds kernel dispatch)
  - `baseline-benchmark.py` (flag pass-through)
- Benchmark command:
  - `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --split-interior-edge --reference-stats results/baseline-benchmark-20260227-052146.json`
- INT8 result: `mean_ms=0.005325`, `stddev_ms=0.000025`, `cv_pct=0.462`, `median_ms=0.005322`, `p95_ms=0.005343`
- FP8 result: `mean_ms=0.022557`, `stddev_ms=0.008393`, `cv_pct=37.209`, `median_ms=0.020119`, `p95_ms=0.045105`
- Decision: `Unsure`
- Notes: INT8 improvement is small; FP8 variance too high for keep/drop confidence.

### 2026-02-27: O10 LDS Input Staging
- Scope: Stage input tiles in LDS and compare with global-load baseline.
- Files changed: `benchmarks/baseline_kernels_bench.cpp`, `baseline-benchmark.py`
- Benchmark command:
  - `./baseline-benchmark.py --no-build --mode int8 --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --lds-stage-input --reference-stats results/baseline-benchmark-20260227-052146.json`
- INT8 result: `mean_ms=0.016294`, `stddev_ms=0.007592`, `cv_pct=46.595`, `median_ms=0.013885`, `p95_ms=0.038227`
- FP8 result: `n/a`
- Decision: `Drop`
- Notes: Large regression with high jitter.

### 2026-02-27: O11 LDS Weight Staging
- Scope: Stage both input and weight tiles in LDS.
- Files changed: `benchmarks/baseline_kernels_bench.cpp`, `baseline-benchmark.py`
- Benchmark command:
  - `./baseline-benchmark.py --no-build --mode int8 --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --lds-stage-input --lds-stage-weight --reference-stats results/baseline-benchmark-20260227-052146.json`
- INT8 result: `mean_ms=0.018153`, `stddev_ms=0.009728`, `cv_pct=53.590`, `median_ms=0.014779`, `p95_ms=0.045228`
- FP8 result: `n/a`
- Decision: `Drop`
- Notes: Regressed further relative to O10.

### 2026-02-27: O12 LDS Padding/Swizzle Proxy
- Scope: Add padded LDS layout on staged tiles.
- Files changed: `benchmarks/baseline_kernels_bench.cpp`, `baseline-benchmark.py`
- Benchmark command:
  - `./baseline-benchmark.py --no-build --mode int8 --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --lds-stage-input --lds-stage-weight --lds-padding --reference-stats results/baseline-benchmark-20260227-052146.json`
- INT8 result: `mean_ms=0.024804`, `stddev_ms=0.000046`, `cv_pct=0.184`, `median_ms=0.024810`, `p95_ms=0.024864`
- FP8 result: `n/a`
- Decision: `Drop`
- Notes: Stable but much slower.

### 2026-02-27: O13 LDS Double-Buffer
- Scope: Add software double-buffer style scheduling on staged LDS path.
- Files changed: `benchmarks/baseline_kernels_bench.cpp`, `baseline-benchmark.py`
- Benchmark command:
  - `./baseline-benchmark.py --no-build --mode int8 --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --lds-stage-input --lds-stage-weight --lds-padding --lds-double-buffer --reference-stats results/baseline-benchmark-20260227-052146.json`
- INT8 result: `mean_ms=0.023248`, `stddev_ms=0.000059`, `cv_pct=0.256`, `median_ms=0.023249`, `p95_ms=0.023320`
- FP8 result: `n/a`
- Decision: `Drop`
- Notes: Slightly better than O12 but still far slower than control.

### 2026-02-27: O14 Register Pressure vs Occupancy Sweep
- Scope: Re-sweep launch/unroll knobs under current code.
- Files changed: none (parameter sweep only)
- Benchmark command(s):
  - `threads=128, inner=16`
  - `threads=256, inner=8`
  - `threads=256, inner=32`
- INT8 results:
  - `128/16`: `mean_ms=0.005367`, `cv_pct=0.532`
  - `256/8`: `mean_ms=0.005354`, `cv_pct=0.554`
  - `256/32`: `mean_ms=0.005308`, `cv_pct=0.438`
- FP8 results:
  - `128/16`: `mean_ms=0.023782`, `cv_pct=48.440`
  - `256/8`: `mean_ms=0.024987`, `cv_pct=47.026`
  - `256/32`: `mean_ms=0.034923`, `cv_pct=43.665`
- Decision: `Keep` existing `threads=256`, `inner=16`
- Notes: No confidence-grade win due FP8 instability; `inner=32` is a clear FP8 regression.

### 2026-02-27: O15 Compile-Flag Sweep (`gfx1151`)
- Scope: Compare compiler flag variants against default `-O3` build.
- Files changed: `baseline-benchmark.py` (added repeatable `--hipcc-flag`)
- Benchmark command(s):
  - `--hipcc-flag=-O2`
  - `--hipcc-flag=-Ofast --hipcc-flag=-ffast-math`
- INT8 results:
  - `-O2`: `mean_ms=0.005359`, `cv_pct=0.589`
  - `-Ofast/-ffast-math`: `mean_ms=0.005309`, `cv_pct=0.569`
- FP8 results:
  - `-O2`: `mean_ms=0.025726`, `cv_pct=48.245`
  - `-Ofast/-ffast-math`: `mean_ms=0.025914`, `cv_pct=47.311`
- Decision: `Keep` default `-O3`
- Notes: INT8 movement was tiny; FP8 remained noisy/uncertain.

### 2026-02-27: O16 Post-Op Fusion A/B
- Scope: Compare fused default against explicit extra post pass (`--force-unfused-post`).
- Files changed: `benchmarks/baseline_kernels_bench.cpp`, `baseline-benchmark.py`
- Benchmark command:
  - `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --force-unfused-post --reference-stats results/baseline-benchmark-20260227-052146.json`
- INT8 result: `mean_ms=0.009843`, `stddev_ms=0.000050`, `cv_pct=0.506`, `median_ms=0.009841`, `p95_ms=0.009870`
- FP8 result: `mean_ms=0.025971`, `stddev_ms=0.007005`, `cv_pct=26.973`, `median_ms=0.024132`, `p95_ms=0.041818`
- Decision: `Drop` (`force-unfused`)
- Notes: Clear regression in both modes.

### 2026-02-27: O17 Adjacent-Pass Fusion A/B
- Scope: Compare single-pass default against forced two-pass path (`--force-two-pass`).
- Files changed: `benchmarks/baseline_kernels_bench.cpp`, `baseline-benchmark.py`
- Benchmark command:
  - `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --force-two-pass --reference-stats results/baseline-benchmark-20260227-052146.json`
- INT8 result: `mean_ms=0.009850`, `stddev_ms=0.000047`, `cv_pct=0.473`, `median_ms=0.009848`, `p95_ms=0.009868`
- FP8 result: `mean_ms=0.031611`, `stddev_ms=0.013649`, `cv_pct=43.177`, `median_ms=0.025100`, `p95_ms=0.064700`
- Decision: `Drop` (`force-two-pass`)
- Notes: Strong regression vs single-pass default.

### 2026-02-27: O18 INT8 Mixed Subpath
- Scope: Compare DOT-style INT8 path against mixed INT/FP compute (`--force-mixed-int8-path`).
- Files changed: `benchmarks/baseline_kernels_bench.cpp`, `baseline-benchmark.py`
- Benchmark command:
  - `./baseline-benchmark.py --no-build --mode int8 --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --force-mixed-int8-path --reference-stats results/baseline-benchmark-20260227-052146.json`
- INT8 result: `mean_ms=0.017437`, `stddev_ms=0.010509`, `cv_pct=60.269`, `median_ms=0.012356`, `p95_ms=0.041443`
- FP8 result: `n/a`
- Decision: `Drop`
- Notes: Mixed path significantly slower and unstable.

### 2026-02-27: O19 FP8 Quantized-IO Fallback
- Scope: Test quantized FP8 IO path (`--fp8-quantized-io`) for fallback-style compute flow.
- Files changed: `benchmarks/baseline_kernels_bench.cpp`, `baseline-benchmark.py`
- Benchmark command:
  - `./baseline-benchmark.py --no-build --mode fp8 --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 --fp8-quantized-io --reference-stats results/baseline-benchmark-20260227-052146.json`
- INT8 result: `n/a`
- FP8 result: `mean_ms=0.020277`, `stddev_ms=0.011747`, `cv_pct=57.935`, `median_ms=0.014781`, `p95_ms=0.048668`
- Decision: `Unsure`
- Notes: Potential upside on median, but very high variance blocks a keep call.

### 2026-02-27: O20 Cleanup (Net-Positive Only)
- Scope: Keep only net-positive defaults and preserve all experimental paths behind flags.
- Files changed:
  - `benchmarks/baseline_kernels_bench.cpp`
  - `baseline-benchmark.py`
  - `OPTIMIZATION_QUEUE.md`, `OPTIMIZATION_RESULTS.md`, `TODO.md`, `WORKLOG.md`, `IMPLEMENTATION.md`
- Validation: Default control run retained as final candidate baseline.
- Benchmark command: `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1`
- INT8 result: `mean_ms=0.005376`, `stddev_ms=0.000038`, `cv_pct=0.704`, `median_ms=0.005371`, `p95_ms=0.005392`
- FP8 result: `mean_ms=0.019868`, `stddev_ms=0.000115`, `cv_pct=0.581`, `median_ms=0.019860`, `p95_ms=0.020063`
- Decision: `Keep`
- Notes: Final selected baseline after exhausting O07-O19.
