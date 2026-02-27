# FSR4 Research Environment

## Core Setup
- Target platform: `Strix Halo` (RDNA3.5 iGPU), ROCm GPU target `gfx1151`  
- Default mamba environment: `therock` (if not active, run commands with `mamba run -n therock <command>`)
- TheRock ROCm version: `7.12.0a20260226`  
- Primary workflow: benchmarking, optimization experiments, and validation of the FSR4 model paths

## Reference Sources
- `reference/FidelityFX-SDK-FSR4-SOURCE-CODE` -> `/home/lhl/github/other/FidelityFX-SDK-FSR4-SOURCE-CODE`
- `reference/rdna35-isa-markdown` -> `/home/lhl/github/other/rdna35-isa-markdown`

## Required Runtime / Libraries
- ROCm 7.12.0a20260226 runtime/toolchain expected in the active environment:
  - `hipcc` / HIP toolchain
  - ROCm system utilities (`rocminfo`, `rocm-smi`, `clinfo`)
  - Vulkan/driver access that can enumerate Strix Halo
- Optional quick check: `rocm-sdk version`
- GPU/compute stack should use the same checkout and driver/runtime versions for all benchmark runs to keep timing comparisons consistent.

## Conda ROCm/HIP Vars (therock)
- One-time after environment creation/update: run `mamba run -n therock rocm-sdk init` to expand `_rocm_sdk_devel` contents.
- Canonical ROCm root: `mamba run -n therock rocm-sdk path --root` (expected: `/home/lhl/mambaforge/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel`).
- Persisted conda vars should be derived from that root:
  - `ROCM_PATH`, `HIP_PATH`
  - `HIP_INCLUDE_PATH`, `HIP_LIB_PATH`, `HIP_CLANG_PATH`, `HIP_DEVICE_LIB_PATH`
  - `PATH`, `LD_LIBRARY_PATH`, `LIBRARY_PATH`, `CPATH`, `PKG_CONFIG_PATH`
- Example set command:
  - `conda env config vars set -n therock ROCM_PATH="$(mamba run -n therock rocm-sdk path --root)" HIP_PATH="$(mamba run -n therock rocm-sdk path --root)" HIP_INCLUDE_PATH="$(mamba run -n therock rocm-sdk path --root)/include" HIP_LIB_PATH="$(mamba run -n therock rocm-sdk path --root)/lib" HIP_CLANG_PATH="$(mamba run -n therock rocm-sdk path --root)/lib/llvm/bin" HIP_DEVICE_LIB_PATH="$(mamba run -n therock rocm-sdk path --root)/lib/llvm/amdgcn/bitcode"`
- Reactivate after changes: `mamba deactivate && mamba activate therock` (or open a new shell).
- Smoke test:
  - `mamba run -n therock hipcc --version`
  - `mamba run -n therock /usr/bin/bash -lc 'printf "#include <hip/hip_runtime.h>\n__global__ void k(){}\nint main(){return 0;}\n" >/tmp/hip_smoke.cpp && hipcc -O2 --offload-arch=gfx1151 /tmp/hip_smoke.cpp -o /tmp/hip_smoke'`

## Notes
- Keep README-level guidance for methodology, command invocations, and data interpretation here.
- This file is intentionally environment-focused; avoid duplicating experiment methodology.
- Keep `fsr4-src/baseline` immutable; make optimization edits only in `fsr4-src/opt`.

## Work Tracking
- Record every benchmark run, optimization change, and validation result in `[WORKLOG.md](/home/lhl/github/lhl/fsr4-rdna3/WORKLOG.md)`.
- Keep it in reverse chronological order with one entry per work session.
- Record each optimization attempt decision in `[IMPLEMENTATION.md](/home/lhl/github/lhl/fsr4-rdna3/IMPLEMENTATION.md)` including `mean/stddev/cv` metrics.

## Task Queue
- Maintain upcoming work in `[TODO.md](/home/lhl/github/lhl/fsr4-rdna3/TODO.md)` as a punchlist.
- Keep active tasks organized by status (`Queue`, `In Progress`, `Blocked`, `Completed`).
