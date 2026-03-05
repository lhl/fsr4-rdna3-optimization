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

pick_best_cfg() {
  local mode="$1"
  shift
  python - "$mode" "$@" <<'PY'
import json
import sys

mode = sys.argv[1]
paths = sys.argv[2:]
best = None
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mean_ms = float(payload["mode_summary"][mode]["mean_of_mean_ms"])
    cfg = payload["config"]
    if mode == "int8":
        value = int(cfg["inner_int8"])
    else:
        value = int(cfg["inner_fp8"])
    if best is None or mean_ms < best[1]:
        best = (value, mean_ms, p)
print(best[0])
PY
}

pick_best_by_key() {
  local mode="$1"
  local key="$2"
  shift 2
  python - "$mode" "$key" "$@" <<'PY'
import json
import sys

mode = sys.argv[1]
key = sys.argv[2]
paths = sys.argv[3:]
best = None
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mean_ms = float(payload["mode_summary"][mode]["mean_of_mean_ms"])
    cfg = payload.get("config", {})
    value = cfg.get(key)
    if value is None:
        cmd = payload.get("trials", [{}])[0].get("command", [])
        flag = "--" + key.replace("_", "-")
        if flag in cmd:
            idx = cmd.index(flag)
            if idx + 1 < len(cmd):
                value = int(cmd[idx + 1])
    if value is None:
        raise KeyError(f"missing {key} in {p}")
    value = int(value)
    if best is None or mean_ms < best[1]:
        best = (value, mean_ms, p)
print(best[0])
PY
}

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

echo "Canonical reference: $REF"

# P0: O03 retest under current defaults.
for items in 1 2 4; do
  run "P0 O03 retest items=${items} (trials=3)" \
    "${COMMON_BOTH[@]}" --items-per-thread "$items" --trials 3 \
    --reference-stats "$REF" \
    --stats-out "results/todo-p0-o03-items${items}-trials3.json" \
    --classification-out "results/todo-p0-o03-items${items}-trials3-classification.json"
done

# P0: O14 independent inner-loop sweeps.
int8_inner_paths=()
for inner in 8 16 32 64; do
  out="results/todo-p0-o14-int8-inner${inner}.json"
  run "P0 O14 int8 inner=${inner}" \
    "${COMMON_INT8[@]}" --inner-int8 "$inner" --trials 1 \
    --reference-stats "$REF" \
    --stats-out "$out" \
    --classification-out "results/todo-p0-o14-int8-inner${inner}-classification.json"
  int8_inner_paths+=("$out")
done

fp8_inner_paths=()
for inner in 8 16 32 64; do
  out="results/todo-p0-o14-fp8-inner${inner}.json"
  run "P0 O14 fp8 inner=${inner}" \
    "${COMMON_FP8[@]}" --inner-fp8 "$inner" --trials 1 \
    --reference-stats "$REF" \
    --stats-out "$out" \
    --classification-out "results/todo-p0-o14-fp8-inner${inner}-classification.json"
  fp8_inner_paths+=("$out")
done

best_int8_inner="$(pick_best_cfg int8 "${int8_inner_paths[@]}")"
best_fp8_inner="$(pick_best_cfg fp8 "${fp8_inner_paths[@]}")"
echo "Best int8 inner from sweep: ${best_int8_inner}"
echo "Best fp8 inner from sweep: ${best_fp8_inner}"

run "P0 O14 int8 top candidate inner=${best_int8_inner} (trials=3)" \
  "${COMMON_INT8[@]}" --inner-int8 "$best_int8_inner" --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p0-o14-int8-inner${best_int8_inner}-trials3.json" \
  --classification-out "results/todo-p0-o14-int8-inner${best_int8_inner}-trials3-classification.json"

run "P0 O14 fp8 top candidate inner=${best_fp8_inner} (trials=3)" \
  "${COMMON_FP8[@]}" --inner-fp8 "$best_fp8_inner" --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p0-o14-fp8-inner${best_fp8_inner}-trials3.json" \
  --classification-out "results/todo-p0-o14-fp8-inner${best_fp8_inner}-trials3-classification.json"

# P0: O09 retest under current defaults.
run "P0 O09 split-interior-edge retest (trials=3)" \
  "${COMMON_BOTH[@]}" --split-interior-edge --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p0-o09-split-interior-edge-trials3.json" \
  --classification-out "results/todo-p0-o09-split-interior-edge-trials3-classification.json"

# P1: extended threads sweep per mode.
int8_thread_paths=()
for threads in 192 320 384 512 1024; do
  out="results/todo-p1-threads-int8-${threads}.json"
  run "P1 threads int8=${threads}" \
    "${COMMON_INT8[@]}" --threads "$threads" --trials 1 \
    --reference-stats "$REF" \
    --stats-out "$out" \
    --classification-out "results/todo-p1-threads-int8-${threads}-classification.json"
  int8_thread_paths+=("$out")
done
best_int8_threads="$(pick_best_by_key int8 threads "${int8_thread_paths[@]}")"
echo "Best int8 threads from sweep: ${best_int8_threads}"
run "P1 threads int8 top=${best_int8_threads} (trials=3)" \
  "${COMMON_INT8[@]}" --threads "$best_int8_threads" --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p1-threads-int8-${best_int8_threads}-trials3.json" \
  --classification-out "results/todo-p1-threads-int8-${best_int8_threads}-trials3-classification.json"

fp8_thread_paths=()
for threads in 192 320 384 512 1024; do
  out="results/todo-p1-threads-fp8-${threads}.json"
  run "P1 threads fp8=${threads}" \
    "${COMMON_FP8[@]}" --threads "$threads" --trials 1 \
    --reference-stats "$REF" \
    --stats-out "$out" \
    --classification-out "results/todo-p1-threads-fp8-${threads}-classification.json"
  fp8_thread_paths+=("$out")
done
best_fp8_threads="$(pick_best_by_key fp8 threads "${fp8_thread_paths[@]}")"
echo "Best fp8 threads from sweep: ${best_fp8_threads}"
run "P1 threads fp8 top=${best_fp8_threads} (trials=3)" \
  "${COMMON_FP8[@]}" --threads "$best_fp8_threads" --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p1-threads-fp8-${best_fp8_threads}-trials3.json" \
  --classification-out "results/todo-p1-threads-fp8-${best_fp8_threads}-trials3-classification.json"

# P1: extended INT8 tile sweep.
int8_items_paths=()
for items in 1 2 4 8 16; do
  out="results/todo-p1-items-int8-${items}.json"
  run "P1 int8 items-per-thread=${items}" \
    "${COMMON_INT8[@]}" --items-per-thread "$items" --trials 1 \
    --reference-stats "$REF" \
    --stats-out "$out" \
    --classification-out "results/todo-p1-items-int8-${items}-classification.json"
  int8_items_paths+=("$out")
done
best_int8_items="$(pick_best_by_key int8 items_per_thread "${int8_items_paths[@]}")"
echo "Best int8 items-per-thread from sweep: ${best_int8_items}"
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
printf 'P0/P1 TODO batch complete at %s\n' "$(date -Iseconds)"
