# Optimization Results

Decision rule:
- `Better`/`Worse` only when variance is acceptable (`cv_pct <= 3.0` in both modes).
- Otherwise mark `Unsure` and keep baseline unless speedup is clearly outside error bounds.

## Baseline Reference

Protocol:
- `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256`

| ID | Variant | INT8 mean ms | INT8 cv% | FP8 mean ms | FP8 cv% | INT8 runs | FP8 runs | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| REF | threads=256 | 0.007743 | 1.239 | 0.117392 | 1.276 | 38233 | 2554 | Baseline |

## Optimization Trials

| ID | Variant | INT8 mean ms | INT8 cv% | FP8 mean ms | FP8 cv% | ΔINT8 vs REF | ΔFP8 vs REF | INT8 runs | FP8 runs | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| O02 | threads=64 | 0.009068 | 50.129 | 0.129159 | 9.803 | +17.11% | +10.02% | 32698 | 2321 | Unsure (high variance, likely worse) |
| O02 | threads=128 | 0.010435 | 62.409 | 0.146970 | 14.178 | +34.77% | +25.20% | 28443 | 2040 | Unsure (high variance, likely worse) |
| O02 | threads=256 | 0.007743 | 1.239 | 0.117392 | 1.276 | +0.00% | +0.00% | 38233 | 2554 | Keep (best stable result) |
| O03 | items/thread=2 | 0.008756 | 1.265 | 0.121350 | 1.222 | +13.08% | +3.37% | 33867 | 2471 | Worse |
| O03 | items/thread=4 | 0.011774 | 54.322 | 0.143581 | 9.483 | +52.06% | +22.31% | 25248 | 2088 | Unsure (high variance, likely worse) |
| O03 | items/thread=1 | 0.007743 | 1.239 | 0.117392 | 1.276 | +0.00% | +0.00% | 38233 | 2554 | Keep (best stable result) |
| O04 | warpSize check | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | Verified `warpSize=32` on `gfx1151` |
| O05 | unrolled (default) | 0.007816 | 1.347 | 0.113599 | 1.335 | +0.00%* | +0.00%* | 37888 | 2639 | Keep |
| O05 | force runtime loops | 0.008767 | 38.897 | 0.124402 | 6.198 | +12.17%* | +9.51%* | 33810 | 2410 | Unsure (high variance, likely worse) |
| O06 | packed INT8 I/O (`--force-packed-int8-io`) | 0.007835 | 1.501 | 0.113683 | 1.409 | +0.00%** | +0.00%** | 37795 | 2637 | Reference for O06 |
| O06 | scalar INT8 I/O (default) | 0.005340 | 0.397 | 0.115609 | 1.828 | -31.84%** | +1.69%** | 55189 | 2593 | Keep (INT8 win, FP8 neutral within uncertainty) |

`*` O05 deltas are relative to the O05 unrolled run, not the global REF row.
`**` O06 deltas are relative to the O06 packed-I/O reference row.

## Phase 2 Trials (`O07`-`O20`)

Reference for this phase:
- `C2` = `results/baseline-benchmark-20260227-052146.json`
- Command: `./baseline-benchmark.py --no-build --mode both --target-seconds 60 --min-runs 200 --reps-per-run 200 --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1`

| ID | Variant | INT8 mean ms | INT8 cv% | FP8 mean ms | FP8 cv% | Decision |
|---|---|---:|---:|---:|---:|---|
| C2 | control (post-O06 code, no O07+ flags) | 0.005376 | 0.704 | 0.019868 | 0.581 | Baseline |
| O07 | `--force-inloop-scale-bias` | 0.012061 | 1.001 | n/a | n/a | Drop |
| O08 | `--force-per-iter-requant` | 0.015792 | 26.640 | 0.114545 | 1.428 | Drop |
| O09 | `--split-interior-edge` | 0.005325 | 0.462 | 0.022557 | 37.209 | Unsure (FP8 high variance) |
| O10 | `--lds-stage-input` | 0.016294 | 46.595 | n/a | n/a | Drop |
| O11 | `--lds-stage-input --lds-stage-weight` | 0.018153 | 53.590 | n/a | n/a | Drop |
| O12 | `--lds-stage-input --lds-stage-weight --lds-padding` | 0.024804 | 0.184 | n/a | n/a | Drop |
| O13 | `--lds-stage-input --lds-stage-weight --lds-padding --lds-double-buffer` | 0.023248 | 0.256 | n/a | n/a | Drop |
| O14 | `threads=128, inner=16` | 0.005367 | 0.532 | 0.023782 | 48.440 | Unsure |
| O14 | `threads=256, inner=8` | 0.005354 | 0.554 | 0.024987 | 47.026 | Unsure |
| O14 | `threads=256, inner=32` | 0.005308 | 0.438 | 0.034923 | 43.665 | Drop (FP8) |
| O15 | compile `-O2` | 0.005359 | 0.589 | 0.025726 | 48.245 | Unsure |
| O15 | compile `-Ofast -ffast-math` | 0.005309 | 0.569 | 0.025914 | 47.311 | Unsure |
| O16 | `--force-unfused-post` | 0.009843 | 0.506 | 0.025971 | 26.973 | Drop |
| O17 | `--force-two-pass` | 0.009850 | 0.473 | 0.031611 | 43.177 | Drop |
| O18 | `--force-mixed-int8-path` | 0.017437 | 60.269 | n/a | n/a | Drop |
| O19 | `--fp8-quantized-io` | n/a | n/a | 0.020277 | 57.935 | Unsure |
| O20 | cleanup: keep net-positive defaults only | 0.005376 | 0.704 | 0.019868 | 0.581 | Keep |

## Current Best

| Mode | Best Variant | Mean ms | cv% |
|---|---|---:|---:|
| INT8 | scalar INT8 I/O + unrolled loops (`O06`) with O20 cleanup defaults | 0.005376 | 0.704 |
| FP8 | O20 cleanup defaults (includes O08 requant-once behavior) | 0.019868 | 0.581 |
