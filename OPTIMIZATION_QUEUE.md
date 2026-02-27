# Optimization Queue

Status key: `Queue`, `In Progress`, `Done`, `Dropped`

## Queue
- [ ]

## In Progress
- [ ]

## Done
- [x] Source split: `fsr4-src/baseline` (immutable) and `fsr4-src/opt` (working tree).
- [x] Added repeatable baseline harness with long-run sampling and stddev/cv reporting.
- [x] `O01` Benchmark protocol lock: stable 60s reference captured in direct TTY.
- [x] `O02` Threadgroup sweep (`64`, `128`, `256`): keep `256` as best stable configuration.
- [x] `O03` Per-thread tile-size sweep (`items-per-thread=1,2,4`): keep `1`.
- [x] `O04` Wave-size verification: `warpSize=32` confirmed on `gfx1151`.
- [x] `O05` Inner-loop unroll variants: keep unrolled kernels.
- [x] `O06` INT8 I/O path sweep: scalar path outperformed packed path on `gfx1151`.
- [x] `O07` Hoist quant scales and bias loads out of innermost loops.
- [x] `O08` Accumulate in `int32`/`float32` for full tile, requantize once on store.
- [x] `O09` Split interior and edge kernels to avoid hot-path bounds checks.
- [x] `O10` LDS stage input tiles for reuse.
- [x] `O11` LDS stage weight tiles for reuse.
- [x] `O12` Add LDS padding/swizzle to reduce bank conflicts.
- [x] `O13` Double-buffer LDS (prefetch + compute overlap).
- [x] `O14` Tune register pressure vs occupancy (`threads`, unroll depth, temporary reuse).
- [x] `O15` Compile-flag sweep for `gfx1151` (`-O` level and backend flags).
- [x] `O16` Fuse conv+bias+activation+requant where legal.
- [x] `O17` Fuse adjacent passes to reduce global-memory round trips.
- [x] `O18` INT8 subpath comparison: DOT4-heavy vs mixed FP16/DOT2 in applicable kernels.
- [x] `O19` FP8 path experiments for RDNA3.5 fallback behavior (quantized IO + non-WMMA compute).
- [x] `O20` Final cleanup pass: keep only net-positive changes.

## Dropped
- [ ]
