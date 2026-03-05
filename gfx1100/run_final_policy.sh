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

run "Final policy INT8 (items=4, inner=16, trials=5)" \
  --mode int8 --target-seconds 60 --min-runs 200 --reps-per-run 200 \
  --elements 262144 --threads 256 --items-per-thread 4 --inner-int8 16 \
  --fp8-quantized-io --trials 5 \
  --reference-stats "$REF" \
  --stats-out "results/gfx1100-final-policy-int8-items4-inner16-trials5.json" \
  --classification-out "results/gfx1100-final-policy-int8-items4-inner16-trials5-classification.json"

run "Final policy FP8 (items=1, inner=8, trials=5)" \
  --mode fp8 --target-seconds 60 --min-runs 200 --reps-per-run 200 \
  --elements 262144 --threads 256 --items-per-thread 1 --inner-fp8 8 \
  --fp8-quantized-io --trials 5 \
  --reference-stats "$REF" \
  --stats-out "results/gfx1100-final-policy-fp8-items1-inner8-trials5.json" \
  --classification-out "results/gfx1100-final-policy-fp8-items1-inner8-trials5-classification.json"

python - <<'PY'
import json
from pathlib import Path

root = Path("results")
ref = json.loads((root / "gfx1100-final-default-optimized-trials3.json").read_text())
int8 = json.loads((root / "gfx1100-final-policy-int8-items4-inner16-trials5.json").read_text())
fp8 = json.loads((root / "gfx1100-final-policy-fp8-items1-inner8-trials5.json").read_text())

ref_modes = ref["mode_summary"]
int8_modes = int8["mode_summary"]
fp8_modes = fp8["mode_summary"]

ref_total = ref_modes["int8"]["mean_of_mean_ms"] + ref_modes["fp8"]["mean_of_mean_ms"]
new_total = int8_modes["int8"]["mean_of_mean_ms"] + fp8_modes["fp8"]["mean_of_mean_ms"]

payload = {
    "reference_total_mean_ms": ref_total,
    "final_policy_total_mean_ms": new_total,
    "final_policy_total_delta_pct": 100.0 * (ref_total - new_total) / ref_total,
    "int8_mean_ms": int8_modes["int8"]["mean_of_mean_ms"],
    "fp8_mean_ms": fp8_modes["fp8"]["mean_of_mean_ms"],
}
out = root / "gfx1100-final-policy-summary.json"
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"[saved] {out}")
PY

echo
printf 'Final policy run complete at %s\n' "$(date -Iseconds)"
