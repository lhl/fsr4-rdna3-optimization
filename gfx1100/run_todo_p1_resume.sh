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

pick_best_items_from_existing() {
  python - <<'PY'
import glob
import json

best = None
for p in sorted(glob.glob("results/todo-p1-items-int8-*.json")):
    if p.endswith("-classification.json") or p.endswith("-trials3.json"):
        continue
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mean_ms = float(payload["mode_summary"]["int8"]["mean_of_mean_ms"])
    cmd = payload.get("trials", [{}])[0].get("command", [])
    if "--items-per-thread" not in cmd:
        continue
    idx = cmd.index("--items-per-thread")
    value = int(cmd[idx + 1])
    if best is None or mean_ms < best[1]:
        best = (value, mean_ms, p)
if best is None:
    raise SystemExit("could not determine best items-per-thread from existing files")
print(best[0])
PY
}

COMMON_INT8=(
  --mode int8
  --target-seconds 60
  --min-runs 200
  --reps-per-run 200
  --elements 262144
  --inner-int8 16
  --threads 256
  --items-per-thread 1
  --fp8-quantized-io
)

COMMON_FP8=(
  --mode fp8
  --target-seconds 60
  --min-runs 200
  --reps-per-run 200
  --elements 262144
  --inner-fp8 16
  --threads 256
  --items-per-thread 1
  --fp8-quantized-io
)

COMMON_BOTH=(
  --mode both
  --target-seconds 60
  --min-runs 200
  --reps-per-run 200
  --elements 262144
  --inner-int8 16
  --inner-fp8 16
  --threads 256
  --items-per-thread 1
  --fp8-quantized-io
)

best_int8_items="$(pick_best_items_from_existing)"
echo "Resuming with best int8 items-per-thread=${best_int8_items}"

run "P1 int8 items top=${best_int8_items} (trials=3)" \
  "${COMMON_INT8[@]}" --items-per-thread "$best_int8_items" --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p1-items-int8-${best_int8_items}-trials3.json" \
  --classification-out "results/todo-p1-items-int8-${best_int8_items}-trials3-classification.json"

# P1: scaling sanity sweep.
for elements in 65536 262144 1048576; do
  for inner in 8 16 32; do
    run "P1 scaling int8 elements=${elements} inner=${inner}" \
      --mode int8 --target-seconds 20 --min-runs 100 --reps-per-run 200 \
      --elements "$elements" --inner-int8 "$inner" --threads 256 --items-per-thread 1 \
      --fp8-quantized-io --trials 1 \
      --stats-out "results/todo-p1-scaling-int8-e${elements}-inner${inner}.json"
    run "P1 scaling fp8 elements=${elements} inner=${inner}" \
      --mode fp8 --target-seconds 20 --min-runs 100 --reps-per-run 200 \
      --elements "$elements" --inner-fp8 "$inner" --threads 256 --items-per-thread 1 \
      --fp8-quantized-io --trials 1 \
      --stats-out "results/todo-p1-scaling-fp8-e${elements}-inner${inner}.json"
  done
done

# Re-check candidate wins in stable-throughput region.
STABLE_ELEMENTS=1048576
run "P1 stable-region baseline int8 (trials=3)" \
  --mode int8 --target-seconds 60 --min-runs 200 --reps-per-run 200 \
  --elements "$STABLE_ELEMENTS" --inner-int8 16 --threads 256 --items-per-thread 1 \
  --fp8-quantized-io --trials 3 \
  --stats-out "results/todo-p1-stable-int8-baseline-trials3.json"
run "P1 stable-region candidate int8 items=4 (trials=3)" \
  --mode int8 --target-seconds 60 --min-runs 200 --reps-per-run 200 \
  --elements "$STABLE_ELEMENTS" --inner-int8 16 --threads 256 --items-per-thread 4 \
  --fp8-quantized-io --trials 3 \
  --stats-out "results/todo-p1-stable-int8-items4-trials3.json"
run "P1 stable-region baseline fp8 (trials=3)" \
  --mode fp8 --target-seconds 60 --min-runs 200 --reps-per-run 200 \
  --elements "$STABLE_ELEMENTS" --inner-fp8 16 --threads 256 --items-per-thread 1 \
  --fp8-quantized-io --trials 3 \
  --stats-out "results/todo-p1-stable-fp8-baseline-trials3.json"
run "P1 stable-region candidate fp8 inner=8 (trials=3)" \
  --mode fp8 --target-seconds 60 --min-runs 200 --reps-per-run 200 \
  --elements "$STABLE_ELEMENTS" --inner-fp8 8 --threads 256 --items-per-thread 1 \
  --fp8-quantized-io --trials 3 \
  --stats-out "results/todo-p1-stable-fp8-inner8-trials3.json"

# P1: high-noise candidate reruns.
run "P1 high-noise rerun O02 threads=64 (trials=3)" \
  "${COMMON_BOTH[@]}" --threads 64 --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p1-highnoise-o02-threads64-trials3.json" \
  --classification-out "results/todo-p1-highnoise-o02-threads64-trials3-classification.json"

run "P1 high-noise rerun O15 -O2 (trials=3)" \
  "${COMMON_BOTH[@]}" --trials 3 --force-rebuild --hipcc-flag=-O2 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p1-highnoise-o15-o2-trials3.json" \
  --classification-out "results/todo-p1-highnoise-o15-o2-trials3-classification.json"

run "P1 high-noise rerun O15 -Ofast -ffast-math (trials=3)" \
  "${COMMON_BOTH[@]}" --trials 3 --force-rebuild --hipcc-flag=-Ofast --hipcc-flag=-ffast-math \
  --reference-stats "$REF" \
  --stats-out "results/todo-p1-highnoise-o15-ofast-ffastmath-trials3.json" \
  --classification-out "results/todo-p1-highnoise-o15-ofast-ffastmath-trials3-classification.json"

# Rebuild default binary after compile-flag experiments.
run "P1 reset default -O3 binary" \
  --mode int8 --target-seconds 0.2 --min-runs 2 --max-runs 20 --warmup-runs 1 --reps-per-run 20 \
  --elements 32768 --inner-int8 8 --threads 256 --items-per-thread 1 \
  --fp8-quantized-io --force-rebuild \
  --stats-out "results/todo-p1-reset-default-o3.json"

# P1: explicit mode-specific defaults evaluation.
run "P1 candidate A int8 (items=4)" \
  "${COMMON_INT8[@]}" --items-per-thread 4 --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p1-candA-int8-items4-trials3.json" \
  --classification-out "results/todo-p1-candA-int8-items4-trials3-classification.json"
run "P1 candidate A fp8 (items=1)" \
  "${COMMON_FP8[@]}" --items-per-thread 1 --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p1-candA-fp8-items1-trials3.json" \
  --classification-out "results/todo-p1-candA-fp8-items1-trials3-classification.json"

run "P1 candidate B int8 (inner=16)" \
  "${COMMON_INT8[@]}" --inner-int8 16 --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p1-candB-int8-inner16-trials3.json" \
  --classification-out "results/todo-p1-candB-int8-inner16-trials3-classification.json"
run "P1 candidate B fp8 (inner=8)" \
  "${COMMON_FP8[@]}" --inner-fp8 8 --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p1-candB-fp8-inner8-trials3.json" \
  --classification-out "results/todo-p1-candB-fp8-inner8-trials3-classification.json"

run "P1 candidate C int8 (items=4, inner=16)" \
  "${COMMON_INT8[@]}" --items-per-thread 4 --inner-int8 16 --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p1-candC-int8-items4-inner16-trials3.json" \
  --classification-out "results/todo-p1-candC-int8-items4-inner16-trials3-classification.json"
run "P1 candidate C fp8 (items=1, inner=8)" \
  "${COMMON_FP8[@]}" --items-per-thread 1 --inner-fp8 8 --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p1-candC-fp8-items1-inner8-trials3.json" \
  --classification-out "results/todo-p1-candC-fp8-items1-inner8-trials3-classification.json"

echo
printf 'P1 TODO resume batch complete at %s\n' "$(date -Iseconds)"
