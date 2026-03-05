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

# P2: ISA-direct packed dot variant (and control comparisons).
run "P2 ISA control scalar int8 (trials=3)" \
  "${COMMON_INT8[@]}" --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p2-isa-control-scalar-int8-trials3.json" \
  --classification-out "results/todo-p2-isa-control-scalar-int8-trials3-classification.json"

run "P2 ISA control packed-int8 (amd_mixed_dot) (trials=3)" \
  "${COMMON_INT8[@]}" --force-packed-int8-io --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p2-isa-control-packed-int8-trials3.json" \
  --classification-out "results/todo-p2-isa-control-packed-int8-trials3-classification.json"

set +e
echo
echo "===== P2 ISA builtin sdot4 compile attempt ====="
date -Iseconds
mamba run -n therock ./baseline-benchmark.py --arch "$ARCH" \
  "${COMMON_INT8[@]}" --force-isa-packed-int8-io --trials 1 --force-rebuild \
  --hipcc-flag=-DENABLE_ISA_SDOT4_VARIANT=1 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p2-isa-builtin-sdot4-attempt.json" \
  --classification-out "results/todo-p2-isa-builtin-sdot4-attempt-classification.json"
sdot4_rc=$?
set -e

python - "$sdot4_rc" <<'PY'
import datetime as dt
import json
import pathlib
import sys

rc = int(sys.argv[1])
payload = {
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": "ok" if rc == 0 else "failed",
    "exit_code": rc,
    "note": "gfx1100 compile attempt for __builtin_amdgcn_sdot4 with -DENABLE_ISA_SDOT4_VARIANT=1",
}
out = pathlib.Path("results/todo-p2-isa-builtin-sdot4-attempt-status.json")
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"[saved] {out}")
PY

# Fallback path keeps same interface but without builtin-enable macro.
run "P2 ISA fallback path (force-isa-packed-int8-io, trials=3)" \
  "${COMMON_INT8[@]}" --force-isa-packed-int8-io --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p2-isa-fallback-int8-trials3.json" \
  --classification-out "results/todo-p2-isa-fallback-int8-trials3-classification.json"

# P2: packed+items=4 composition check.
run "P2 packed int8 + items=4 (trials=3)" \
  "${COMMON_INT8[@]}" --force-packed-int8-io --items-per-thread 4 --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p2-packed-items4-int8-trials3.json" \
  --classification-out "results/todo-p2-packed-items4-int8-trials3-classification.json"

# P2: restrict + alignment assumptions A/B.
run "P2 assume-aligned hot ptrs A/B (trials=3)" \
  "${COMMON_BOTH[@]}" --trials 3 --force-rebuild --hipcc-flag=-DASSUME_ALIGNED_HOT_PTRS=1 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p2-assume-aligned-hot-ptrs-trials3.json" \
  --classification-out "results/todo-p2-assume-aligned-hot-ptrs-trials3-classification.json"

# Restore default build flags for downstream runs.
run "P2 reset default -O3 binary after ASSUME_ALIGNED" \
  --mode int8 --target-seconds 0.2 --min-runs 2 --max-runs 20 --warmup-runs 1 --reps-per-run 20 \
  --elements 32768 --inner-int8 8 --threads 256 --items-per-thread 1 \
  --fp8-quantized-io --force-rebuild \
  --stats-out "results/todo-p2-reset-default-after-assume.json"

# P2: ILP variant.
run "P2 ILP2 int8 (trials=3)" \
  "${COMMON_INT8[@]}" --force-ilp2-int8 --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p2-ilp2-int8-trials3.json" \
  --classification-out "results/todo-p2-ilp2-int8-trials3-classification.json"

# P2: conv-like microkernel + LDS rerun (O10-O13 style).
run "P2 convlike base (trials=1)" \
  "${COMMON_INT8[@]}" --force-convlike-int8 --trials 1 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p2-convlike-base.json" \
  --classification-out "results/todo-p2-convlike-base-classification.json"

run "P2 convlike + lds-stage-weight (trials=1)" \
  "${COMMON_INT8[@]}" --force-convlike-int8 --lds-stage-weight --trials 1 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p2-convlike-lds-weight.json" \
  --classification-out "results/todo-p2-convlike-lds-weight-classification.json"

run "P2 convlike + lds-weight+padding (trials=1)" \
  "${COMMON_INT8[@]}" --force-convlike-int8 --lds-stage-weight --lds-padding --trials 1 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p2-convlike-lds-padding.json" \
  --classification-out "results/todo-p2-convlike-lds-padding-classification.json"

run "P2 convlike + lds-weight+padding+double-buffer (trials=1)" \
  "${COMMON_INT8[@]}" --force-convlike-int8 --lds-stage-weight --lds-padding --lds-double-buffer \
  --trials 1 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p2-convlike-lds-double-buffer.json" \
  --classification-out "results/todo-p2-convlike-lds-double-buffer-classification.json"

run "P2 convlike top candidate lds-weight+padding+double-buffer (trials=3)" \
  "${COMMON_INT8[@]}" --force-convlike-int8 --lds-stage-weight --lds-padding --lds-double-buffer \
  --trials 3 \
  --reference-stats "$REF" \
  --stats-out "results/todo-p2-convlike-lds-double-buffer-trials3.json" \
  --classification-out "results/todo-p2-convlike-lds-double-buffer-trials3-classification.json"

# P2: reps-per-run overhead sweep.
for reps in 50 100 200 400; do
  run "P2 reps-per-run sweep reps=${reps}" \
    --mode both --target-seconds 20 --min-runs 100 --reps-per-run "${reps}" \
    --elements 262144 --inner-int8 16 --inner-fp8 16 --threads 256 --items-per-thread 1 \
    --fp8-quantized-io --trials 1 \
    --stats-out "results/todo-p2-reps-sweep-${reps}.json"
done

# P2: rocprofv3 top FP8 candidates (timing profile baseline vs inner=8 candidate).
if command -v rocprofv3 >/dev/null 2>&1; then
  base_rc=0
  cand_rc=0

  echo
  echo "===== P2 rocprofv3 FP8 baseline ====="
  date -Iseconds
  set +e
  mamba run -n therock rocprofv3 --kernel-trace --stats --summary \
    --summary-output-file results/todo-p2-rocprof-fp8-baseline-summary.txt \
    --output-format csv -o results/todo-p2-rocprof-fp8-baseline -- \
    ./build/baseline_kernels_bench.gfx1100 \
    --mode fp8 --elements 262144 --threads 256 --items-per-thread 1 \
    --inner-int8 64 --inner-fp8 16 --warmup-runs 20 --min-runs 200 --max-runs 200 \
    --reps-per-run 200 --target-seconds 1 --seed 1337 --fp8-quantized-io
  base_rc=$?

  echo
  echo "===== P2 rocprofv3 FP8 candidate inner=8 ====="
  date -Iseconds
  mamba run -n therock rocprofv3 --kernel-trace --stats --summary \
    --summary-output-file results/todo-p2-rocprof-fp8-inner8-summary.txt \
    --output-format csv -o results/todo-p2-rocprof-fp8-inner8 -- \
    ./build/baseline_kernels_bench.gfx1100 \
    --mode fp8 --elements 262144 --threads 256 --items-per-thread 1 \
    --inner-int8 64 --inner-fp8 8 --warmup-runs 20 --min-runs 200 --max-runs 200 \
    --reps-per-run 200 --target-seconds 1 --seed 1337 --fp8-quantized-io
  cand_rc=$?
  set -e

  python - "${base_rc}" "${cand_rc}" <<'PY'
import datetime as dt
import json
import pathlib
import sys

base_rc = int(sys.argv[1])
cand_rc = int(sys.argv[2])
status = "ok" if base_rc == 0 and cand_rc == 0 else "failed"
payload = {
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": status,
    "method": "rocprofv3 --kernel-trace --stats --summary",
    "baseline_exit_code": base_rc,
    "candidate_exit_code": cand_rc,
    "artifacts": {
        "baseline_kernel_stats_csv": "results/todo-p2-rocprof-fp8-baseline_kernel_stats.csv",
        "baseline_kernel_trace_csv": "results/todo-p2-rocprof-fp8-baseline_kernel_trace.csv",
        "baseline_summary": "results/todo-p2-rocprof-fp8-baseline_results/todo-p2-rocprof-fp8-baseline-summary.txt.txt",
        "candidate_kernel_stats_csv": "results/todo-p2-rocprof-fp8-inner8_kernel_stats.csv",
        "candidate_kernel_trace_csv": "results/todo-p2-rocprof-fp8-inner8_kernel_trace.csv",
        "candidate_summary": "results/todo-p2-rocprof-fp8-inner8_results/todo-p2-rocprof-fp8-inner8-summary.txt.txt",
    },
}
out = pathlib.Path("results/todo-p2-rocprof-status.json")
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"[saved] {out}")
PY
else
  python - <<'PY'
import datetime as dt
import json
import pathlib

payload = {
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": "skipped",
    "note": "rocprofv3 not found in PATH",
}
out = pathlib.Path("results/todo-p2-rocprof-status.json")
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"[saved] {out}")
PY
fi

# P2: ISA disassembly comparison scalar vs packed on gfx1100 and gfx1151.
echo
echo "===== P2 ISA disassembly capture ====="
date -Iseconds
mkdir -p build/disasm
mamba run -n therock hipcc -O3 --offload-arch=gfx1100 --save-temps \
  benchmarks/baseline_kernels_bench.cpp -o build/disasm/baseline_kernels_bench.gfx1100.disasm
mamba run -n therock hipcc -O3 --offload-arch=gfx1151 --save-temps \
  benchmarks/baseline_kernels_bench.cpp -o build/disasm/baseline_kernels_bench.gfx1151.disasm || true

for arch in gfx1100 gfx1151; do
  asm_file="$(ls -t baseline_kernels_bench*${arch}*.s 2>/dev/null | head -n1 || true)"
  if [[ -n "${asm_file}" ]]; then
    cp -f "${asm_file}" "build/disasm/${arch}.s"
    python - "${arch}" "build/disasm/${arch}.s" <<'PY'
import json
import pathlib
import re
import sys

arch = sys.argv[1]
asm_path = pathlib.Path(sys.argv[2])
text = asm_path.read_text(encoding="utf-8", errors="ignore")
dot4 = len(re.findall(r"\bv_dot4", text))
mad = len(re.findall(r"\bv_mad_|\bv_mac_", text))
mul = len(re.findall(r"\bv_mul_", text))
payload = {
    "arch": arch,
    "asm_path": str(asm_path),
    "counts": {
        "v_dot4_like": dot4,
        "v_mad_or_mac_like": mad,
        "v_mul_like": mul,
    },
}
out = pathlib.Path(f"results/todo-p2-disasm-{arch}-summary.json")
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"[saved] {out}")
PY
  else
    python - "${arch}" <<'PY'
import datetime as dt
import json
import pathlib
import sys

arch = sys.argv[1]
payload = {
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "arch": arch,
    "status": "missing_asm",
}
out = pathlib.Path(f"results/todo-p2-disasm-{arch}-summary.json")
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"[saved] {out}")
PY
  fi
done

python - <<'PY'
import json
import pathlib
import re

arch_payload = {}
for arch in ("gfx1100", "gfx1151"):
    asm_path = pathlib.Path(f"build/disasm/{arch}.s")
    if not asm_path.exists():
        continue
    lines = asm_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    functions = {}
    current = None
    buf = []
    for ln in lines:
        m = re.match(r"^(_Z\\S+):\\s*;\\s*@(_Z\\S+)", ln)
        if m:
            if current is not None:
                functions[current] = "\n".join(buf)
            current = m.group(1)
            buf = []
            continue
        if current is not None:
            buf.append(ln)
    if current is not None:
        functions[current] = "\n".join(buf)

    target_keys = {
        "runtime_packed": "int8_dot4_kernel_runtimePKaS0_Piiii",
        "runtime_scalar": "int8_dot4_kernel_runtime_scalarPKaS0_Piiii",
        "runtime_sdot4": "int8_dot4_kernel_runtime_sdot4PKaS0_Piiii",
    }
    per_func = {}
    for label, needle in target_keys.items():
        fn_name = next((k for k in functions if needle in k), None)
        if fn_name is None:
            per_func[label] = {"status": "missing"}
            continue
        text = functions[fn_name]
        per_func[label] = {
            "function": fn_name,
            "v_dot4_like": len(re.findall(r"\bv_dot4", text)),
            "v_mad_or_mac_like": len(re.findall(r"\bv_mad_|\bv_mac_", text)),
            "v_mul_like": len(re.findall(r"\bv_mul_", text)),
            "s_waitcnt": len(re.findall(r"\bs_waitcnt\b", text)),
            "buffer_or_global_loads": len(re.findall(r"\b(?:buffer_load|global_load)\w*", text)),
            "buffer_or_global_stores": len(re.findall(r"\b(?:buffer_store|global_store)\w*", text)),
        }

    arch_payload[arch] = {
        "asm_path": str(asm_path),
        "targets": per_func,
    }

out = pathlib.Path("results/todo-p2-disasm-int8-kernels-compare.json")
out.write_text(json.dumps(arch_payload, indent=2) + "\n", encoding="utf-8")
print(f"[saved] {out}")
PY

echo
printf 'P2 TODO batch complete at %s\n' "$(date -Iseconds)"
