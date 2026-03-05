# TODO (gfx1100)

## Queue
- [ ] Resolve O03/O14 mixed results using weighted INT8+FP8 policy and possibly `--trials 3` confirmations.
- [ ] Optionally rerun high-noise candidates (`O02 threads=64`, `O15 -O2`) with `--trials 3`.

## In Progress
- [ ] Define final policy for `items-per-thread` and O14 (`inner=8` vs `16`) under mixed INT8/FP8 goals.

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
