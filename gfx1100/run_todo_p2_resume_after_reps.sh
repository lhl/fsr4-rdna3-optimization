#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ARCH="gfx1100"

# Ensure default benchmark binary exists before profiling/disassembly stages.
mamba run -n therock ./baseline-benchmark.py --arch "$ARCH" \
  --mode int8 --target-seconds 0.2 --min-runs 2 --max-runs 20 --warmup-runs 1 --reps-per-run 20 \
  --elements 32768 --inner-int8 8 --threads 256 --items-per-thread 1 \
  --fp8-quantized-io --force-rebuild \
  --stats-out "results/todo-p2-resume-reset-default.json"

if command -v rocprofv3 >/dev/null 2>&1; then
  base_rc=0
  cand_rc=0

  echo
  echo "===== P2 resume: rocprofv3 FP8 baseline ====="
  date -Iseconds
  set +e
  mamba run -n therock rocprofv3 --stats -o results/todo-p2-rocprof-fp8-baseline.csv -- \
    ./build/baseline_kernels_bench.gfx1100 \
    --mode fp8 --elements 262144 --threads 256 --items-per-thread 1 \
    --inner-int8 64 --inner-fp8 16 --warmup-runs 20 --min-runs 200 --max-runs 200 \
    --reps-per-run 200 --target-seconds 1 --seed 1337 --fp8-quantized-io
  base_rc=$?

  echo
  echo "===== P2 resume: rocprofv3 FP8 candidate inner=8 ====="
  date -Iseconds
  mamba run -n therock rocprofv3 --stats -o results/todo-p2-rocprof-fp8-inner8.csv -- \
    ./build/baseline_kernels_bench.gfx1100 \
    --mode fp8 --elements 262144 --threads 256 --items-per-thread 1 \
    --inner-int8 64 --inner-fp8 8 --warmup-runs 20 --min-runs 200 --max-runs 200 \
    --reps-per-run 200 --target-seconds 1 --seed 1337 --fp8-quantized-io
  cand_rc=$?
  set -e

  python - "$base_rc" "$cand_rc" <<'PY'
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
    "baseline_exit_code": base_rc,
    "candidate_exit_code": cand_rc,
    "baseline_csv": "results/todo-p2-rocprof-fp8-baseline.csv",
    "candidate_csv": "results/todo-p2-rocprof-fp8-inner8.csv",
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
echo "===== P2 resume: ISA disassembly capture ====="
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
dot4 = len(re.findall(r"\\bv_dot4", text))
mad = len(re.findall(r"\\bv_mad_|\\bv_mac_", text))
mul = len(re.findall(r"\\bv_mul_", text))
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

echo
printf 'P2 TODO resume batch complete at %s\n' "$(date -Iseconds)"
