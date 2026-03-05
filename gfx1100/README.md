# FSR4 RDNA3 Benchmark: gfx1100 (W7900) vs gfx1151 (Strix Halo)

This folder replicates the HIP microkernel benchmark workflow on Radeon Pro W7900 (`gfx1100`) and tracks direct comparison against recorded Strix Halo (`gfx1151`) results from the parent project.

## Final Outcome (After Full TODO Sweep)

Two layers are now tracked:

1. **Canonical shared-default baseline** (single config for both modes):
   - `fp8_quantized_io=true` (`O19` keep)
   - artifact: `results/gfx1100-final-default-optimized-trials3.json`

2. **Final integrated mode-specific policy** (selected after P0/P1/P2 reruns):
   - `int8`: `items_per_thread=4`, `inner_int8=16`
   - `fp8`: `items_per_thread=1`, `inner_fp8=8`
   - artifacts:
     - `results/gfx1100-final-policy-int8-items4-inner16-trials5.json`
     - `results/gfx1100-final-policy-fp8-items1-inner8-trials5.json`
     - `results/gfx1100-final-policy-summary.json`

### Before / Canonical / Final

| Mode | Raw Baseline (ms) | Canonical Shared Default (ms) | Final Mode-Specific (ms) | Final vs Raw | Final vs Canonical |
|---|---:|---:|---:|---:|---:|
| INT8 | 0.007754 | 0.007713 | 0.006834 | +11.87% | +11.40% |
| FP8 | 0.012805 | 0.010641 | 0.010006 | +21.86% | +5.97% |
| Total (INT8+FP8) | 0.020559 | 0.018353 | 0.016839 | +18.09% | +8.25% |

Notes:
- Raw baseline: `results/baseline-benchmark-20260305-185902.json` (3 trials).
- Canonical shared default: `results/gfx1100-final-default-optimized-trials3.json` (3 trials).
- Final mode-specific: two independent 5-trial runs (INT8+FP8), combined in `results/gfx1100-final-policy-summary.json`.

## Cross-Architecture Snapshot (Final)

| Mode | gfx1100 Final (ms) | gfx1151 Final (ms) | Delta | Faster On |
|---|---:|---:|---:|---|
| INT8 | 0.006834 | 0.005376 | +27.11% | gfx1151 |
| FP8 | 0.010006 | 0.019868 | -49.64% | gfx1100 |

`gfx1151` reference uses recorded parent-project final artifact: `../results/baseline-benchmark-20260227-052146.json`.

## Final Adopted Policy

- Always keep `fp8_quantized_io=true` (O19).
- Shared-default/canonical path retained for compare stability and regression checks.
- Performance path for deployments/experiments uses mode-specific knobs:
  - INT8: `items_per_thread=4`, `inner_int8=16`
  - FP8: `items_per_thread=1`, `inner_fp8=8`
- Runner helper added: `run_final_policy.sh`.

## Benchmark Protocol and Decision Rule

Protocol was kept aligned with the parent `gfx1151` workflow so cross-target comparisons stay meaningful.

- Core run shape:
  - `./baseline-benchmark.py --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1`
- Compare-ready references and final policy decisions were trial aggregates:
  - canonical shared default: 3 trials
  - final mode-specific policy confirmation: 5 trials per mode
- Classification gate:
  - metric: median
  - `min_uncertainty_pct=3`
  - `cv_scale=0.5`
  - verdicts: `keep` / `drop` / `unsure`
- Decision policy:
  - keep optimizations only when the classification is stable and directionally consistent
  - for mixed INT8+FP8 behavior, prefer mode-specific policy if shared knobs force a tradeoff

## Environment

### gfx1100 Execution Environment

| Item | Value |
|---|---|
| GPU | AMD Radeon Pro W7900 |
| Target arch | `gfx1100` (RDNA3) |
| Mamba env | `therock` |
| HIP/ROCm | HIP `7.2.26043-9999` (clang 22) |
| ROCm SDK | `7.12.0a20260304` |

### gfx1151 Reference Environment (Parent Project Record)

| Item | Value |
|---|---|
| GPU | Radeon 8060S Graphics (Strix Halo iGPU) |
| Target arch | `gfx1151` (RDNA3.5) |
| HIP/ROCm | HIP `7.12.60490-128c4eea36` |
| ROCm SDK | `7.12.0a20260226` |

## Attempted gfx1151 Execution on W7900 Host

Direct execution of true `gfx1151` binaries was attempted and failed as expected on this `gfx1100` host.

- `--arch gfx1151 --force-rebuild` builds successfully but aborts at runtime with HIP code-object assertion (`SIGABRT`) because `gfx1151` code objects cannot dispatch on `gfx1100`.
- Evidence:
  - `results/todo-p3-gfx1151-o19-trials5-attempt-status.json`
  - `results/todo-p3-gfx1151-o19-trials5-attempt.log`
- Implication:
  - all cross-arch tables use recorded `gfx1151` artifacts from the parent project, not local execution on this machine.

## Interpretation Notes

- `Canonical Shared Default` is the single-config control plane for apples-to-apples mixed-mode comparisons.
- `Final Mode-Specific` is the deployment/performance policy and intentionally allows different INT8 vs FP8 launch knobs.
- FP8/INT8 ratio evolution on gfx1100:
  - raw baseline: `1.65x`
  - canonical shared default: `1.38x`
  - final mode-specific: `1.46x`
- Final cross-target ratio remains highly asymmetric:
  - gfx1100 final FP8/INT8: `1.46x`
  - gfx1151 final FP8/INT8: `3.70x`

## Extra Optimization Status (Full TODO Sweep)

| Item | Status | Result |
|---|---|---|
| P0 canonical reference lock | Completed | All post-default classification runs now reference `results/gfx1100-final-default-optimized-trials3.json`. |
| P0 arch-specific build output | Completed | Binary now names by arch (`build/baseline_kernels_bench.<arch>`), avoiding stale cross-arch reuse. |
| P0 O03 rerun (`items=1/2/4`, trials=3) | Completed | `items=4` keeps INT8 but drops FP8; overall mixed. |
| P0 O14 independent sweeps | Completed | FP8 `inner=8` is a keep (~+6%); INT8 inner sweeps remain unsure. |
| P0 O09 rerun (`split-interior-edge`, trials=3) | Completed | Overall `unsure`. |
| P0 mixed-policy decision | Completed | Selected mode-specific candidate C (`int8 items=4`, `fp8 inner=8`). |
| P1 extended thread sweep | Completed | No confident winner vs threads=256 for either mode. |
| P1 INT8 items sweep (`8/16`) | Completed | `8` and `16` regress; `4` remains best. |
| P1 scaling sanity sweep | Completed | Stable low-CV region maintained at protocol size (`elements=262144`); FP8 `inner=8` remains favorable. |
| P1 high-noise reruns (trials=3) | Completed | O02 threads=64, O15 -O2, O15 -Ofast all remain `unsure`. |
| P1 result summarizer | Completed | `summarize_results.py` + `results/todo-summary.md` and `results/todo-summary.json` added. |
| P1 candidate A/B/C evaluation | Completed | A keeps INT8, B keeps FP8, C keeps both and is selected. |
| P2 ISA direct packed variant (`__builtin_amdgcn_sdot4`) | Completed (partial block) | Direct builtin compile fails on gfx1100 (`dot1-insts` feature). Fallback path benchmarked and dropped. |
| P2 restrict + assume aligned A/B | Completed | Overall `unsure` (no measurable gain). |
| P2 ILP2 INT8 | Completed | Overall `unsure` (no measurable gain). |
| P2 conv-like + LDS rerun | Completed | All variants dropped; worst case ~-209%. |
| P2 packed INT8 + items=4 | Completed | Overall `unsure` / slight regression. |
| P2 reps-per-run sweep | Completed | `reps>=200` gives stable timings; `50` shows measurable overhead bias. |
| P2 ROCProfiler on FP8 top candidates | Completed | Needed `--kernel-trace --stats --summary`; candidate inner=8 kernel avg ~7.5us vs baseline ~8.1us (~7.3% faster). |
| P2 disassembly comparison gfx1100 vs gfx1151 | Completed | Similar scalar/packed instruction shape across arch; O06 delta gap likely hardware throughput/latency behavior, not major ISA shape divergence. |
| P3 FP8 WMMA/MFMA prototype | Blocked | FP8 MFMA builtin probe fails for gfx1100/gfx1151 (`needs target feature fp8-insts`); rocWMMA FP8 WMMA path in installed headers is gfx12-gated. |
| P3 gfx1151 O19 trials=5 cross-validate | Blocked on this host | True gfx1151 code object cannot execute on W7900/gfx1100 (HIP code-object assertion). |

## Historical O Sweep (Initial Batch Record)

This section preserves the original `O02`-`O20` batch outcomes from `run_all_optimizations.sh` against the original compare-ready baseline reference:

- baseline reference: `results/baseline-benchmark-20260305-185902.json`
- batch artifacts: `results/oXX-*.json` and `results/oXX-*-classification.json`

These are kept as historical record; final integration decisions are based on the later `P`-series reruns with tighter controls/trials.

| ID | Initial O Outcome | Headline Result | Historical Artifacts |
|---|---|---|---|
| O01 | Keep | Protocol lock / stable reference capture. | See `IMPLEMENTATION.md` run log. |
| O02 | Unsure | `threads=64/128` not a confident win over 256. | `results/o02-threads64*.json`, `results/o02-threads128*.json` |
| O03 | Unsure | `items=4` helped INT8 but hurt FP8 in shared-mode view. | `results/o03-items2*.json`, `results/o03-items4*.json` |
| O04 | Keep | Wave check confirmed `warpSize=32`. | `results/o04-wavecheck.json` |
| O05 | Drop | Runtime inner loops strongly regressed FP8. | `results/o05-runtime-loops*.json` |
| O06 | Drop | Packed INT8 I/O regressed INT8 vs scalar path. | `results/o06-packed-int8-io*.json` |
| O07 | Drop | In-loop scale/bias path regressed. | `results/o07-inloop-scale-bias*.json` |
| O08 | Drop | Per-iter requantization catastrophic regression. | `results/o08-per-iter-requant*.json` |
| O09 | Unsure | Interior/edge split showed small/noisy movement. | `results/o09-split-interior-edge*.json` |
| O10 | Drop | LDS stage input regression. | `results/o10-lds-stage-input*.json` |
| O11 | Drop | LDS stage input+weight regression. | `results/o11-lds-stage-input-weight*.json` |
| O12 | Drop | LDS padding/swizzle proxy regression. | `results/o12-lds-padding*.json` |
| O13 | Drop | LDS double-buffer proxy regression. | `results/o13-lds-double-buffer*.json` |
| O14 | Unsure | Mixed signal; FP8 `inner=8` looked promising but needed rerun. | `results/o14-threads128-inner16*.json`, `results/o14-threads256-inner8*.json`, `results/o14-threads256-inner32*.json` |
| O15 | Unsure | `-O2` / `-Ofast -ffast-math` inconclusive vs `-O3`. | `results/o15-flag-o2*.json`, `results/o15-flag-ofast-ffastmath*.json` |
| O16 | Drop | Unfused post-op path large regression. | `results/o16-unfused-post*.json` |
| O17 | Drop | Two-pass adjacent path large regression. | `results/o17-two-pass*.json` |
| O18 | Drop | Mixed INT8 subpath regression. | `results/o18-mixed-int8-path*.json` |
| O19 | Keep | FP8 quantized-IO clear improvement. | `results/o19-fp8-quantized-io*.json` |
| O20 | Unsure | Final control near baseline (no drift, no clear win). | `results/o20-final-control*.json` |

## Conclusions

1. `gfx1100` and `gfx1151` still show opposite specialization in this harness: gfx1151 leads INT8, gfx1100 leads FP8.
2. The full `O` and `P` sweeps agree on the major regressions (runtime inner loops, per-iter requant, LDS staging variants, unfused/two-pass paths).
3. The practical integrated policy on gfx1100 is mode-specific, not one-size-fits-all:
   - INT8 likes `items_per_thread=4`
   - FP8 likes `inner_fp8=8`
4. This mode-specific integration improved total combined mean by `+8.25%` over canonical shared-default and `+18.09%` over the original raw baseline.

## Key Artifacts

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

## Tracking Files

- `WORKLOG.md` -- run-by-run log
- `IMPLEMENTATION.md` -- commands and per-optimization details
- `TODO.md` -- active punchlist
