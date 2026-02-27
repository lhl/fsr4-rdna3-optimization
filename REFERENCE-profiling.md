# ROCm/HIP Profiling Reference (FSR4 RDNA3.5)

This document is the profiling runbook for this repository:
- platform: Strix Halo iGPU (`gfx1151`)
- environment: `therock`
- workflow target: HIP microkernel profiling and bottleneck identification for FSR4-style INT8/FP8 paths

It is written for both humans and coding agents.

## 1) Environment and intent

Use this guide when you need to:
- identify which kernels are hot
- explain *why* a variant is faster/slower (utilization, memory, occupancy signals)
- correlate timing with power/clock behavior
- avoid false conclusions from GUI jitter

This repo already has a benchmark harness:
- `baseline-benchmark.py`
- `build/baseline_kernels_bench`

Profiling should generally be done in direct TTY (not desktop GUI) to keep variance low.

## 2) Tool availability snapshot (local)

The following was verified from `therock`:

### 2.1 Available now

- `rocprofv3`: `/home/lhl/mambaforge/envs/therock/bin/rocprofv3`
- `rocprof` (legacy): `/opt/rocm/bin/rocprof`
- `rocprof-compute`: `/opt/rocm/bin/rocprof-compute`
- `rocminfo`: `/home/lhl/mambaforge/envs/therock/bin/rocminfo`
- `amd-smi`: `/home/lhl/mambaforge/envs/therock/bin/amd-smi`
- `rocm-smi`: `/home/lhl/mambaforge/envs/therock/bin/rocm-smi`
- `rocgdb`: `/opt/rocm/bin/rocgdb`

### 2.2 Not installed in current env

- `omniperf`
- `omnitrace`
- `rocprofiler-systems` / `rocprof-sys`

### 2.3 Partially usable

- `rocprof-compute` currently errors due missing Python deps (`pandas`, `dash`, etc.).
- Required deps list path:
  - `/opt/rocm/libexec/rocprofiler-compute/requirements.txt`

## 3) Recommended tool stack (priority order)

Use tools in this order unless you have a reason to deviate:

1. `rocprofv3` (primary)
2. `amd-smi monitor` (telemetry during profiling)
3. `rocm-smi` (quick sanity/status and optional fixed-performance settings if needed)
4. `rocprof-compute` (optional deeper kernel analysis, only after dependency setup and compatibility check)

Why:
- `rocprofv3` gives timeline + kernel traces + counters in one consistent tool.
- `amd-smi` gives context for clock/power/thermal throttling that can otherwise look like random jitter.

## 4) Pre-flight checks

Run before profiling:

```bash
mamba run -n therock rocminfo | head -n 40
mamba run -n therock amd-smi list
mamba run -n therock /usr/bin/bash -lc 'command -v rocprofv3 amd-smi rocm-smi'
```

Optional benchmark smoke (very short):

```bash
./baseline-benchmark.py --mode both --target-seconds 1 --min-runs 5 --max-runs 20 --warmup-runs 2 --reps-per-run 20
```

If smoke fails, fix runtime first before collecting profiles.

## 5) Profiling workflow (standard)

Use a dedicated output root:

```bash
mkdir -p results/profiling
```

### Step A: Hotspot discovery (timeline + kernel summary)

```bash
mamba run -n therock rocprofv3 \
  --runtime-trace --kernel-trace --stats --summary \
  --output-format csv \
  --output-directory results/profiling/rocprofv3-hotspot \
  -- ./build/baseline_kernels_bench \
     --mode both --threads 256 --items-per-thread 1 \
     --inner-int8 16 --inner-fp8 16 \
     --target-seconds 5 --min-runs 20 --reps-per-run 50
```

What to read first:
- kernel total time
- average kernel dispatch time
- kernel call count
- large API overhead relative to kernel time

### Step B: Focus on specific kernel families

Use include regex to isolate kernels:

```bash
mamba run -n therock rocprofv3 \
  --kernel-trace --stats --summary \
  --kernel-include-regex 'int8_dot4|fp8_fma' \
  --output-format csv \
  --output-directory results/profiling/rocprofv3-kernel-filter \
  -- ./build/baseline_kernels_bench \
     --mode both --threads 256 --items-per-thread 1 \
     --inner-int8 16 --inner-fp8 16 \
     --target-seconds 5 --min-runs 20 --reps-per-run 50
```

Use this for A/B comparisons so data volume stays manageable.

### Step C: Counter collection (`--pmc`)

Use after hotspots are known. Keep runs small and focused.

```bash
mamba run -n therock rocprofv3 \
  --kernel-trace --stats --summary \
  --pmc 'SQ_WAVES GRBM_GUI_ACTIVE' \
  --output-format csv \
  --output-directory results/profiling/rocprofv3-pmc \
  -- ./build/baseline_kernels_bench \
     --mode int8 --threads 256 --items-per-thread 1 \
     --inner-int8 16 --target-seconds 5 --min-runs 20 --reps-per-run 50
```

Notes:
- Counter names are architecture/tool-version dependent.
- If a counter set cannot fit one pass, use multiple `--pmc` groups.

### Step D: Optional PC sampling (advanced)

Use only when needed for instruction-level hotspot mapping:

```bash
mamba run -n therock rocprofv3 \
  --kernel-trace --pc-sampling-beta-enabled \
  --pc-sampling-unit cycles \
  --pc-sampling-method stochastic \
  --pc-sampling-interval 50000 \
  --output-format csv \
  --output-directory results/profiling/rocprofv3-pcsampling \
  -- ./build/baseline_kernels_bench \
     --mode int8 --target-seconds 5 --min-runs 20 --reps-per-run 50
```

PC sampling has higher overhead; use it only after narrowing target kernels.

## 6) Run telemetry in parallel (clock/power/thermal context)

Start monitor in another terminal before profiling:

```bash
mamba run -n therock amd-smi monitor \
  -p -t -u -m -v \
  -w 1 --csv --file results/profiling/amd-smi-monitor.csv
```

Correlate spikes/dips in kernel time with:
- gfx clock changes
- memory clock changes
- temperature/power behavior

## 7) Comparing optimization variants

For each A/B pair:

1. keep all non-test flags fixed
2. collect `rocprofv3` summary for A and B
3. record benchmark stats (`mean/stddev/cv`) and profiler summary together
4. only call winner when both timing and profiler signals agree (or timing win is large/stable)

For this repo, tie results back to:
- `TODO.md`
- `IMPLEMENTATION.md`
- `WORKLOG.md`

## 8) Suggested experiment mapping to current TODOs

These are directly relevant now:

- `dot4add_i8packed` vs scalarized INT8 MAC in:
  - `ConvNextBlock.hlsli`
  - `FasterNetBlock.hlsli`
  - `FNB_CT2D_ADD.hlsli`
  - `float16_NHWC/Fused/CNB_CT2D.hlsli`
- `[unroll]` on first-conv spatial loops in `FasterNetBlock.hlsli`
- threadgroup sweep (`64/128/256`) with profile correlation
- interior/edge split dispatch profile validation

For each, collect:
- benchmark `mean/stddev/cv`
- `rocprofv3` kernel summary
- counter snapshot (if available)
- brief interpretation

## 9) Troubleshooting

### 9.1 `Invalid KFD descriptor: -1`

Usually indicates profiler/runtime cannot access KFD/GPU context correctly.

Check:
- running inside correct environment (`mamba run -n therock ...`)
- GPU device visibility and permissions
- ROCm runtime consistency between tools and driver

Sanity commands:

```bash
mamba run -n therock rocminfo | rg -n 'gfx|Name|Agent'
mamba run -n therock amd-smi list
```

### 9.2 `rocprof-compute` missing Python modules

Current env lacks required packages. Error mentions missing modules like `pandas`, `dash`, etc.

Reference requirements:
- `/opt/rocm/libexec/rocprofiler-compute/requirements.txt`

### 9.3 High variance in GUI sessions

Expected on iGPU shared-memory systems under desktop load.

Mitigation:
- run in direct TTY
- use longer runs and multiple trials
- pin workload conditions before comparing variants

## 10) Agent notes (operational guardrails)

For automated agents:

- always prefix tool commands with `mamba run -n therock`
- write outputs under `results/profiling/<run-id>/`
- include command line used in each report
- if profiler fails, record exact stderr and environment snapshot
- do not compare A/B runs collected under different system load conditions
- never claim a winner from one noisy GUI sample

Recommended profile run-id format:

```text
results/profiling/YYYYMMDD-HHMM-<tag>/
```

## 11) Quick command cheat sheet

```bash
# Availability
mamba run -n therock /usr/bin/bash -lc 'for t in rocprofv3 rocprof rocprof-compute amd-smi rocm-smi; do echo -n "$t: "; command -v "$t"; done'

# Primary profile (timeline + summary)
mamba run -n therock rocprofv3 --runtime-trace --kernel-trace --stats --summary --output-format csv --output-directory results/profiling/hotspot -- ./build/baseline_kernels_bench --mode both --target-seconds 5 --min-runs 20 --reps-per-run 50

# Kernel-focused profile
mamba run -n therock rocprofv3 --kernel-trace --stats --summary --kernel-include-regex 'int8_dot4|fp8_fma' --output-format csv --output-directory results/profiling/focused -- ./build/baseline_kernels_bench --mode both --target-seconds 5 --min-runs 20 --reps-per-run 50

# Telemetry
mamba run -n therock amd-smi monitor -p -t -u -m -v -w 1 --csv --file results/profiling/amd-smi-monitor.csv
```

## 12) References

- ROCProfilerV3 docs: `https://rocm.docs.amd.com/projects/rocprofiler/en/latest/how-to/using-rocprofv3.html`
- ROCProfiler SDK PC sampling docs: `https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html`
- AMD SMI CLI docs: `https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html`
- ROCProfiler Compute docs: `https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/`
- ROCProfiler Compute compatibility: `https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/compatible-accelerators.html`
- ROCProfiler Systems docs: `https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/`
