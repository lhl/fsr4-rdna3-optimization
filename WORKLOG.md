# Work Log

## 2026-03-05

### Session Goal
- Audit the new README findings sections (first pass + grounding pass) for claim accuracy, evidence boundaries, and overreach.

### Changes / Commands
- Reviewed recent documentation commits:
  - `5ce5be9` (`optimization findings first pass`)
  - `49a735e` (`clarify theory vs measured results`)
- Cross-checked README claims against:
  - benchmark artifacts in `results/`
  - optimization tables in `OPTIMIZATION_RESULTS.md` / `IMPLEMENTATION.md`
  - source-path evidence in `fsr4-src/baseline/...` (INT8 dot path + FP8 WMMA path)
- Updated `README.md` to tighten grounding:
  - clarified loop-level requantization wording
  - replaced unverified intrinsic-name specifics in INT4 discussion with instruction-class wording
  - added explicit O08 FP8 penalty numbers (`0.114545 ms` vs `0.019868 ms`, `~5.76x`)
  - added an "Evidence Boundaries (Measured vs Inferred)" section
  - added a direct-TTY multi-trial follow-up recommendation for `Unsure` variants (`O09/O15/O19`)

### Benchmarks
- No new benchmark runs in this session (documentation/analysis audit only).

### Validation / Results
- README quantitative claims and ratios were re-checked against recorded benchmark JSON values and remain consistent after wording updates.

## 2026-02-27

### Session Goal
- Complete the remaining optimization queue (`O07`-`O20`) with 60s variance-aware decisions and finalize before/after comparison.

### Changes / Commands
- Extended benchmark kernel and harness controls for queue coverage:
  - `benchmarks/baseline_kernels_bench.cpp`
  - `baseline-benchmark.py`
- Added flags for O07-O19 experiments:
  - `--force-inloop-scale-bias`, `--force-per-iter-requant`, `--split-interior-edge`
  - `--lds-stage-input`, `--lds-stage-weight`, `--lds-padding`, `--lds-double-buffer`
  - `--force-unfused-post`, `--force-two-pass`, `--force-mixed-int8-path`, `--fp8-quantized-io`
  - compile sweep support: `--hipcc-flag`
- Ran 60s reference and all remaining optimization sweeps/classifications.
- Updated tracking docs:
  - `OPTIMIZATION_QUEUE.md`
  - `OPTIMIZATION_RESULTS.md`
  - `IMPLEMENTATION.md`
  - `TODO.md`

### Benchmarks (60s protocol)
- New phase reference/control:
  - `results/baseline-benchmark-20260227-052146.json`
  - INT8: `mean_ms=0.005376`, `cv_pct=0.704`
  - FP8: `mean_ms=0.019868`, `cv_pct=0.581`
- O07 result (`force-inloop-scale-bias`): `drop`
- O08 result (`force-per-iter-requant`): `drop`
- O09 result (`split-interior-edge`): `unsure`
- O10/O11/O12/O13 results (LDS staging/padding/double-buffer): all `drop`
- O14 sweep (`threads/inner`): `unsure` overall (no confidence-grade improvement)
- O15 compile sweep (`-O2`, `-Ofast/-ffast-math`): `unsure` (kept `-O3`)
- O16 (`force-unfused-post`): `drop`
- O17 (`force-two-pass`): `drop`
- O18 (`force-mixed-int8-path`): `drop`
- O19 (`fp8-quantized-io`): `unsure`

### Final Before/After Snapshot
- Before (`O01` direct-TTY baseline, `results/baseline-benchmark-20260227-040756.json`):
  - INT8: `0.007743 ms`
  - FP8: `0.117392 ms`
- After (`O20` final defaults, `results/baseline-benchmark-20260227-052146.json`):
  - INT8: `0.005376 ms` (`~30.57%` faster vs O01)
  - FP8: `0.019868 ms` (`~83.08%` faster vs O01)

### Issues / Follow-ups
- FP8 runs for many O09+ variants showed very high CV under current load; several entries remain `unsure` and would benefit from multi-trial confirmation (`--trials 3`).

## 2026-02-27

### Session Goal
- Set up optimization execution framework and a variance-aware FP8/INT8 benchmark harness.

### Environment Snapshot
- ROCm Version: `HIP version: 7.12.60490-128c4eea36`
- ROCm SDK Version: `7.12.0a20260226`
- GPU Target: `gfx1151`
- GPU: `Radeon 8060S Graphics`
- Driver: `6.19.0-rc6-1-mainline`
- Commit/Branch: `N/A (workspace not a git repo root)`

### Changes / Commands
- Created split source trees:
  - `fsr4-src/baseline` (immutable upstream copy)
  - `fsr4-src/opt` (optimization workspace)
- Added HIP benchmark binary source: `benchmarks/baseline_kernels_bench.cpp`
- Added wrapper harness: `baseline-benchmark.py`
- Added process docs:
  - `OPTIMIZATION_QUEUE.md`
  - `IMPLEMENTATION.md`
  - README section `Baseline Benchmark Harness`
- Extended harness outputs:
  - machine-readable stable stats file: `results/latest-benchmark-stats.json`
  - auto classification output: `results/latest-benchmark-classification.json`
  - CLI compare mode: `--reference-stats` with `keep/drop/unsure` verdicts

### Benchmarks
- Quick smoke benchmark command:
  - `./baseline-benchmark.py --mode both --target-seconds 1 --min-runs 5 --max-runs 20 --warmup-runs 5 --reps-per-run 20 --elements 262144 --inner-int8 16 --inner-fp8 16`
- Quick smoke output (non-decision run):
  - INT8: `mean_ms=0.007995`, `stddev_ms=0.000211`, `cv_pct=2.637`
  - FP8: `mean_ms=0.124184`, `stddev_ms=0.005807`, `cv_pct=4.676`
- Long-sample baseline command:
  - `./baseline-benchmark.py --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16`
- Long-sample output:
  - INT8: `mean_ms=0.008455`, `stddev_ms=0.000783`, `cv_pct=9.264`
  - FP8: `mean_ms=0.122223`, `stddev_ms=0.008459`, `cv_pct=6.921`
- Duration sweep output (target seconds per mode, reps_per_run=200):
  - `120s`: INT8 `runs=69893`, `cv_pct=30.377`; FP8 `runs=3922`, `cv_pct=19.912`
  - `180s`: INT8 `runs=63577`, `cv_pct=61.191`; FP8 `runs=4499`, `cv_pct=15.446`
  - `300s`: INT8 `runs=107432`, `cv_pct=60.968`; FP8 `runs=7909`, `cv_pct=15.679`
- Direct-TTY baseline lock (`60s`, `threads=256`, `items-per-thread=1`):
  - INT8: `runs=38233`, `mean_ms=0.007743`, `cv_pct=1.239`
  - FP8: `runs=2554`, `mean_ms=0.117392`, `cv_pct=1.276`
- `O02` threadgroup sweep (`60s` each):
  - `threads=64`: INT8 `mean_ms=0.009068`, `cv_pct=50.129`; FP8 `mean_ms=0.129159`, `cv_pct=9.803`
  - `threads=128`: INT8 `mean_ms=0.010435`, `cv_pct=62.409`; FP8 `mean_ms=0.146970`, `cv_pct=14.178`
  - `threads=256`: retained baseline best/stable.
- `O03` per-thread tile sweep (`60s` each, `threads=256`):
  - `items=2`: INT8 `mean_ms=0.008756`, `cv_pct=1.265`; FP8 `mean_ms=0.121350`, `cv_pct=1.222`
  - `items=4`: INT8 `mean_ms=0.011774`, `cv_pct=54.322`; FP8 `mean_ms=0.143581`, `cv_pct=9.483`
- `O05` unroll A/B (`60s` each, `threads=256`, `items=1`, `inner=16`):
  - unrolled(default): INT8 `mean_ms=0.007816`, `cv_pct=1.347`; FP8 `mean_ms=0.113599`, `cv_pct=1.335`
  - forced runtime loops: INT8 `mean_ms=0.008767`, `cv_pct=38.897`; FP8 `mean_ms=0.124402`, `cv_pct=6.198`
- `O06` INT8 I/O A/B (`60s` each, `threads=256`, `items=1`, `inner=16`):
  - packed INT8 (`--force-packed-int8-io`): INT8 `mean_ms=0.007835`, `cv_pct=1.501`; FP8 `mean_ms=0.113683`, `cv_pct=1.409`
  - scalar INT8 (default): INT8 `mean_ms=0.005340`, `cv_pct=0.397`; FP8 `mean_ms=0.115609`, `cv_pct=1.828`
- Classification smoke test (`results/test-reference.json` vs `results/test-current.json`):
  - FP8 verdict: `unsure`
  - INT8 verdict: `unsure`
  - Overall verdict: `unsure`

### Optimization Notes
- Harness now supports minimum benchmark wall time and large `max_runs` default to avoid premature stop on fast kernels.
- Added multi-trial support (`--trials N`) to improve confidence when system load is noisy.
- Added `items-per-thread` control to emulate per-thread tile sizing (`O03`).
- Added `warpSize` info emission for wave-size verification (`O04`).
- Added unrolled kernel variants and runtime fallback toggle for inner-loop A/B (`O05`).
- Added scalar INT8 I/O path and promoted scalar INT8 as default; packed path kept behind `--force-packed-int8-io` (`O06`).

### Validation / Results
- `hipcc` build and benchmark binary execution succeeded from harness.
- Result artifacts saved under `results/baseline-benchmark-*.json`.

### Issues / Follow-ups
- Current machine-load variance is high; require longer and/or multi-trial baselines (`--trials 3`) before keep/rollback decisions in optimization passes.
- In direct TTY, stable baseline is achievable at 60s, but some variants still show bursty jitter; keep `cv_pct` gate for decisions.

## 2026-02-27

### Session Goal
- Validate and fix `therock` HIP/ROCm environment variable setup for reliable `hipcc` use.

### Environment Snapshot
- ROCm Version: `HIP version: 7.12.60490-128c4eea36`
- ROCm SDK Version: `7.12.0a20260226`
- GPU Target: `gfx1151`
- GPU: `Radeon 8060S Graphics`
- Driver: `6.19.0-rc6-1-mainline`
- Commit/Branch: `N/A (workspace not a git repo root)`

### Changes / Commands
- Verified broken pre-init behavior was tied to `_rocm_sdk_devel` not being expanded.
- Ran `mamba run -n therock rocm-sdk init` to expand `_rocm_sdk_devel`.
- Re-stamped persisted env vars with `conda env config vars set -n therock ...` derived from `rocm-sdk path --root`.
- Added environment bootstrap and validation instructions to `AGENTS.md` and `README.md`, including the canonical `pip install ... rocm[libraries,devel]` and PyTorch nightly commands.

### Benchmarks
- None (environment setup session only).

### Optimization Notes
- None.

### Validation / Results
- `hipcc --version` resolves successfully.
- HIP smoke compile succeeds:
  - `hipcc -O2 --offload-arch=gfx1151 /tmp/hip_envcheck2.cpp -o /tmp/hip_envcheck2`
- No missing `clang++`/`rocm_agent_enumerator` errors after `rocm-sdk init`.

### Issues / Follow-ups
- `mamba run -n therock` still prints "overwriting environment variables" warnings due existing parent-shell vars; functionally okay but noisy.
- Keep `rocm-sdk init` as a documented one-time post-install/update step.
