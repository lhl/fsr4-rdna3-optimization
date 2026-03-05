#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ARCH="gfx1100"
REF="results/gfx1100-final-default-optimized-trials3.json"

run() {
  local label="$1"
  shift
  echo
  echo "===== ${label} ====="
  date -Iseconds
  mamba run -n therock ./baseline-benchmark.py --arch "$ARCH" "$@"
}

# Common long-run protocol knobs.
COMMON=(
  --target-seconds 60
  --min-runs 200
  --reps-per-run 200
  --elements 262144
  --inner-int8 16
  --inner-fp8 16
  --threads 256
  --items-per-thread 1
)

# O02 Threadgroup sweep
run "O02 threads=64" \
  --no-build --mode both "${COMMON[@]}" --threads 64 \
  --reference-stats "$REF" \
  --stats-out results/o02-threads64.json \
  --classification-out results/o02-threads64-classification.json

run "O02 threads=128" \
  --no-build --mode both "${COMMON[@]}" --threads 128 \
  --reference-stats "$REF" \
  --stats-out results/o02-threads128.json \
  --classification-out results/o02-threads128-classification.json

# O03 Items-per-thread sweep
run "O03 items-per-thread=2" \
  --no-build --mode both "${COMMON[@]}" --items-per-thread 2 \
  --reference-stats "$REF" \
  --stats-out results/o03-items2.json \
  --classification-out results/o03-items2-classification.json

run "O03 items-per-thread=4" \
  --no-build --mode both "${COMMON[@]}" --items-per-thread 4 \
  --reference-stats "$REF" \
  --stats-out results/o03-items4.json \
  --classification-out results/o03-items4-classification.json

# O04 Wave-size verification
run "O04 wave-size verification" \
  --no-build --mode int8 \
  --target-seconds 0.1 --min-runs 2 --max-runs 20 --warmup-runs 1 --reps-per-run 10 \
  --elements 32768 --inner-int8 8 --threads 256 --items-per-thread 1 \
  --stats-out results/o04-wavecheck.json

# O05 Runtime inner loops
run "O05 force-runtime-inner-loops" \
  --no-build --mode both "${COMMON[@]}" --force-runtime-inner-loops \
  --reference-stats "$REF" \
  --stats-out results/o05-runtime-loops.json \
  --classification-out results/o05-runtime-loops-classification.json

# O06 Packed INT8 path
run "O06 force-packed-int8-io" \
  --no-build --mode both "${COMMON[@]}" --force-packed-int8-io \
  --reference-stats "$REF" \
  --stats-out results/o06-packed-int8-io.json \
  --classification-out results/o06-packed-int8-io-classification.json

# O07 In-loop scale/bias
run "O07 force-inloop-scale-bias" \
  --no-build --mode int8 "${COMMON[@]}" --force-inloop-scale-bias \
  --reference-stats "$REF" \
  --stats-out results/o07-inloop-scale-bias.json \
  --classification-out results/o07-inloop-scale-bias-classification.json

# O08 Per-iter requant
run "O08 force-per-iter-requant" \
  --no-build --mode both "${COMMON[@]}" --force-per-iter-requant \
  --reference-stats "$REF" \
  --stats-out results/o08-per-iter-requant.json \
  --classification-out results/o08-per-iter-requant-classification.json

# O09 Interior/edge split
run "O09 split-interior-edge" \
  --no-build --mode both "${COMMON[@]}" --split-interior-edge \
  --reference-stats "$REF" \
  --stats-out results/o09-split-interior-edge.json \
  --classification-out results/o09-split-interior-edge-classification.json

# O10-O13 LDS variants
run "O10 lds-stage-input" \
  --no-build --mode int8 "${COMMON[@]}" --lds-stage-input \
  --reference-stats "$REF" \
  --stats-out results/o10-lds-stage-input.json \
  --classification-out results/o10-lds-stage-input-classification.json

run "O11 lds-stage-input+weight" \
  --no-build --mode int8 "${COMMON[@]}" --lds-stage-input --lds-stage-weight \
  --reference-stats "$REF" \
  --stats-out results/o11-lds-stage-input-weight.json \
  --classification-out results/o11-lds-stage-input-weight-classification.json

run "O12 lds-stage-input+weight+padding" \
  --no-build --mode int8 "${COMMON[@]}" --lds-stage-input --lds-stage-weight --lds-padding \
  --reference-stats "$REF" \
  --stats-out results/o12-lds-padding.json \
  --classification-out results/o12-lds-padding-classification.json

run "O13 lds-stage-input+weight+padding+double-buffer" \
  --no-build --mode int8 "${COMMON[@]}" --lds-stage-input --lds-stage-weight --lds-padding --lds-double-buffer \
  --reference-stats "$REF" \
  --stats-out results/o13-lds-double-buffer.json \
  --classification-out results/o13-lds-double-buffer-classification.json

# O14 Occupancy/register sweep
run "O14 threads=128 inner=16" \
  --no-build --mode both "${COMMON[@]}" --threads 128 --inner-int8 16 --inner-fp8 16 \
  --reference-stats "$REF" \
  --stats-out results/o14-threads128-inner16.json \
  --classification-out results/o14-threads128-inner16-classification.json

run "O14 threads=256 inner=8" \
  --no-build --mode both "${COMMON[@]}" --threads 256 --inner-int8 8 --inner-fp8 8 \
  --reference-stats "$REF" \
  --stats-out results/o14-threads256-inner8.json \
  --classification-out results/o14-threads256-inner8-classification.json

run "O14 threads=256 inner=32" \
  --no-build --mode both "${COMMON[@]}" --threads 256 --inner-int8 32 --inner-fp8 32 \
  --reference-stats "$REF" \
  --stats-out results/o14-threads256-inner32.json \
  --classification-out results/o14-threads256-inner32-classification.json

# O15 compile-flag sweep (rebuild required)
run "O15 hipcc -O2" \
  --mode both "${COMMON[@]}" --force-rebuild --hipcc-flag=-O2 \
  --reference-stats "$REF" \
  --stats-out results/o15-flag-o2.json \
  --classification-out results/o15-flag-o2-classification.json

run "O15 hipcc -Ofast -ffast-math" \
  --mode both "${COMMON[@]}" --force-rebuild --hipcc-flag=-Ofast --hipcc-flag=-ffast-math \
  --reference-stats "$REF" \
  --stats-out results/o15-flag-ofast-ffastmath.json \
  --classification-out results/o15-flag-ofast-ffastmath-classification.json

# Rebuild default -O3 binary for downstream O16+ runs.
run "Reset build to default -O3" \
  --mode int8 \
  --target-seconds 0.1 --min-runs 2 --max-runs 20 --warmup-runs 1 --reps-per-run 10 \
  --elements 32768 --inner-int8 8 --threads 256 --items-per-thread 1 \
  --force-rebuild \
  --stats-out results/o15-reset-default-build.json

# O16-O19
run "O16 force-unfused-post" \
  --no-build --mode both "${COMMON[@]}" --force-unfused-post \
  --reference-stats "$REF" \
  --stats-out results/o16-unfused-post.json \
  --classification-out results/o16-unfused-post-classification.json

run "O17 force-two-pass" \
  --no-build --mode both "${COMMON[@]}" --force-two-pass \
  --reference-stats "$REF" \
  --stats-out results/o17-two-pass.json \
  --classification-out results/o17-two-pass-classification.json

run "O18 force-mixed-int8-path" \
  --no-build --mode int8 "${COMMON[@]}" --force-mixed-int8-path \
  --reference-stats "$REF" \
  --stats-out results/o18-mixed-int8-path.json \
  --classification-out results/o18-mixed-int8-path-classification.json

run "O19 fp8-quantized-io" \
  --no-build --mode fp8 "${COMMON[@]}" --fp8-quantized-io \
  --reference-stats "$REF" \
  --stats-out results/o19-fp8-quantized-io.json \
  --classification-out results/o19-fp8-quantized-io-classification.json

# O20 final control
run "O20 final control" \
  --no-build --mode both "${COMMON[@]}" \
  --reference-stats "$REF" \
  --stats-out results/o20-final-control.json \
  --classification-out results/o20-final-control-classification.json

echo
printf 'All optimization runs complete at %s\n' "$(date -Iseconds)"
