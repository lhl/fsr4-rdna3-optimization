#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_math_functions.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifndef ASSUME_ALIGNED_HOT_PTRS
#define ASSUME_ALIGNED_HOT_PTRS 0
#endif

#ifndef ENABLE_ISA_SDOT4_VARIANT
#define ENABLE_ISA_SDOT4_VARIANT 0
#endif

#define HIP_CHECK(cmd)                                                                     \
  do {                                                                                     \
    hipError_t err__ = (cmd);                                                              \
    if (err__ != hipSuccess) {                                                             \
      std::cerr << "HIP error: " << hipGetErrorString(err__) << " at " << __FILE__ << ':' \
                << __LINE__ << '\n';                                                      \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

struct Config {
  std::string mode = "both";
  int elements = 1 << 20;      // logical vector count
  int threads = 256;
  int items_per_thread = 1;
  bool force_runtime_inner_loops = false;
  bool force_scalar_int8_io = true;
  bool force_isa_packed_int8_io = false;
  bool force_ilp2_int8 = false;
  bool force_convlike_int8 = false;
  bool force_inloop_scale_bias = false;
  bool force_per_iter_requant = false;
  bool split_interior_edge = false;
  bool lds_stage_input = false;
  bool lds_stage_weight = false;
  bool lds_padding = false;
  bool lds_double_buffer = false;
  bool force_unfused_post = false;
  bool force_two_pass = false;
  bool force_mixed_int8_path = false;
  bool fp8_quantized_io = true;
  int inner_int8 = 64;
  int inner_fp8 = 64;
  int warmup_runs = 40;
  int min_runs = 120;
  int max_runs = 200000;
  int reps_per_run = 200;
  double target_seconds = 120.0;
  int seed = 1337;
};

struct Stats {
  size_t runs = 0;
  double elapsed_seconds = 0.0;
  double mean_ms = 0.0;
  double stddev_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double median_ms = 0.0;
  double p95_ms = 0.0;
  double cv_pct = 0.0;
};

static bool parse_int_arg(const char* value, int* out) {
  if (value == nullptr || *value == '\0') {
    return false;
  }
  char* end = nullptr;
  long v = std::strtol(value, &end, 10);
  if (*end != '\0') {
    return false;
  }
  *out = static_cast<int>(v);
  return true;
}

static bool parse_double_arg(const char* value, double* out) {
  if (value == nullptr || *value == '\0') {
    return false;
  }
  char* end = nullptr;
  double v = std::strtod(value, &end);
  if (*end != '\0') {
    return false;
  }
  *out = v;
  return true;
}

static void usage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n"
            << "  --mode <both|int8|fp8>\n"
            << "  --elements <int>\n"
            << "  --threads <int>\n"
            << "  --items-per-thread <int>\n"
            << "  --force-runtime-inner-loops\n"
            << "  --force-scalar-int8-io\n"
            << "  --force-packed-int8-io\n"
            << "  --force-isa-packed-int8-io\n"
            << "  --force-ilp2-int8\n"
            << "  --force-convlike-int8\n"
            << "  --force-inloop-scale-bias\n"
            << "  --force-per-iter-requant\n"
            << "  --split-interior-edge\n"
            << "  --lds-stage-input\n"
            << "  --lds-stage-weight\n"
            << "  --lds-padding\n"
            << "  --lds-double-buffer\n"
            << "  --force-unfused-post\n"
            << "  --force-two-pass\n"
            << "  --force-mixed-int8-path\n"
            << "  --fp8-quantized-io\n"
            << "  --inner-int8 <int>\n"
            << "  --inner-fp8 <int>\n"
            << "  --warmup-runs <int>\n"
            << "  --min-runs <int>\n"
            << "  --max-runs <int>\n"
            << "  --reps-per-run <int>\n"
            << "  --target-seconds <double>\n"
            << "  --seed <int>\n";
}

static Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << '\n';
        std::exit(2);
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else if (arg == "--mode") {
      cfg.mode = require_value("--mode");
    } else if (arg == "--elements") {
      if (!parse_int_arg(require_value("--elements"), &cfg.elements)) {
        std::cerr << "Invalid --elements\n";
        std::exit(2);
      }
    } else if (arg == "--threads") {
      if (!parse_int_arg(require_value("--threads"), &cfg.threads)) {
        std::cerr << "Invalid --threads\n";
        std::exit(2);
      }
    } else if (arg == "--items-per-thread") {
      if (!parse_int_arg(require_value("--items-per-thread"), &cfg.items_per_thread)) {
        std::cerr << "Invalid --items-per-thread\n";
        std::exit(2);
      }
    } else if (arg == "--force-runtime-inner-loops") {
      cfg.force_runtime_inner_loops = true;
    } else if (arg == "--force-scalar-int8-io") {
      cfg.force_scalar_int8_io = true;
      cfg.force_isa_packed_int8_io = false;
    } else if (arg == "--force-packed-int8-io") {
      cfg.force_scalar_int8_io = false;
      cfg.force_isa_packed_int8_io = false;
      cfg.force_ilp2_int8 = false;
    } else if (arg == "--force-isa-packed-int8-io") {
      cfg.force_scalar_int8_io = false;
      cfg.force_isa_packed_int8_io = true;
      cfg.force_ilp2_int8 = false;
    } else if (arg == "--force-ilp2-int8") {
      cfg.force_ilp2_int8 = true;
      cfg.force_scalar_int8_io = true;
      cfg.force_isa_packed_int8_io = false;
    } else if (arg == "--force-convlike-int8") {
      cfg.force_convlike_int8 = true;
    } else if (arg == "--force-inloop-scale-bias") {
      cfg.force_inloop_scale_bias = true;
    } else if (arg == "--force-per-iter-requant") {
      cfg.force_per_iter_requant = true;
    } else if (arg == "--split-interior-edge") {
      cfg.split_interior_edge = true;
    } else if (arg == "--lds-stage-input") {
      cfg.lds_stage_input = true;
    } else if (arg == "--lds-stage-weight") {
      cfg.lds_stage_weight = true;
    } else if (arg == "--lds-padding") {
      cfg.lds_padding = true;
    } else if (arg == "--lds-double-buffer") {
      cfg.lds_double_buffer = true;
    } else if (arg == "--force-unfused-post") {
      cfg.force_unfused_post = true;
    } else if (arg == "--force-two-pass") {
      cfg.force_two_pass = true;
    } else if (arg == "--force-mixed-int8-path") {
      cfg.force_mixed_int8_path = true;
    } else if (arg == "--fp8-quantized-io") {
      cfg.fp8_quantized_io = true;
    } else if (arg == "--inner-int8") {
      if (!parse_int_arg(require_value("--inner-int8"), &cfg.inner_int8)) {
        std::cerr << "Invalid --inner-int8\n";
        std::exit(2);
      }
    } else if (arg == "--inner-fp8") {
      if (!parse_int_arg(require_value("--inner-fp8"), &cfg.inner_fp8)) {
        std::cerr << "Invalid --inner-fp8\n";
        std::exit(2);
      }
    } else if (arg == "--warmup-runs") {
      if (!parse_int_arg(require_value("--warmup-runs"), &cfg.warmup_runs)) {
        std::cerr << "Invalid --warmup-runs\n";
        std::exit(2);
      }
    } else if (arg == "--min-runs") {
      if (!parse_int_arg(require_value("--min-runs"), &cfg.min_runs)) {
        std::cerr << "Invalid --min-runs\n";
        std::exit(2);
      }
    } else if (arg == "--max-runs") {
      if (!parse_int_arg(require_value("--max-runs"), &cfg.max_runs)) {
        std::cerr << "Invalid --max-runs\n";
        std::exit(2);
      }
    } else if (arg == "--reps-per-run") {
      if (!parse_int_arg(require_value("--reps-per-run"), &cfg.reps_per_run)) {
        std::cerr << "Invalid --reps-per-run\n";
        std::exit(2);
      }
    } else if (arg == "--target-seconds") {
      if (!parse_double_arg(require_value("--target-seconds"), &cfg.target_seconds)) {
        std::cerr << "Invalid --target-seconds\n";
        std::exit(2);
      }
    } else if (arg == "--seed") {
      if (!parse_int_arg(require_value("--seed"), &cfg.seed)) {
        std::cerr << "Invalid --seed\n";
        std::exit(2);
      }
    } else {
      std::cerr << "Unknown argument: " << arg << '\n';
      usage(argv[0]);
      std::exit(2);
    }
  }

  if (cfg.mode != "both" && cfg.mode != "int8" && cfg.mode != "fp8") {
    std::cerr << "Invalid --mode: " << cfg.mode << '\n';
    std::exit(2);
  }
  if (cfg.elements <= 0 || cfg.threads <= 0 || cfg.items_per_thread <= 0 || cfg.inner_int8 <= 0 ||
      cfg.inner_fp8 <= 0 || cfg.warmup_runs < 0 || cfg.min_runs <= 0 ||
      cfg.max_runs < cfg.min_runs || cfg.reps_per_run <= 0 || cfg.target_seconds <= 0.0) {
    std::cerr << "Invalid numeric argument(s).\n";
    std::exit(2);
  }
  if (cfg.lds_padding && !cfg.lds_stage_input && !cfg.lds_stage_weight) {
    std::cerr << "--lds-padding requires --lds-stage-input and/or --lds-stage-weight\n";
    std::exit(2);
  }
  if (cfg.lds_double_buffer && !cfg.lds_stage_input && !cfg.lds_stage_weight) {
    std::cerr << "--lds-double-buffer requires --lds-stage-input and/or --lds-stage-weight\n";
    std::exit(2);
  }

  return cfg;
}

static Stats compute_stats(const std::vector<float>& samples, double elapsed_seconds) {
  Stats s;
  s.runs = samples.size();
  s.elapsed_seconds = elapsed_seconds;
  if (samples.empty()) {
    return s;
  }

  std::vector<float> sorted(samples.begin(), samples.end());
  std::sort(sorted.begin(), sorted.end());

  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  s.mean_ms = sum / static_cast<double>(samples.size());

  double var = 0.0;
  if (samples.size() > 1) {
    for (float x : samples) {
      const double d = static_cast<double>(x) - s.mean_ms;
      var += d * d;
    }
    var /= static_cast<double>(samples.size() - 1);
  }
  s.stddev_ms = std::sqrt(var);

  s.min_ms = sorted.front();
  s.max_ms = sorted.back();
  s.median_ms = sorted[sorted.size() / 2];

  const size_t p95_idx = static_cast<size_t>(std::ceil(0.95 * sorted.size())) - 1;
  s.p95_ms = sorted[std::min(p95_idx, sorted.size() - 1)];

  s.cv_pct = (s.mean_ms > 0.0) ? (100.0 * s.stddev_ms / s.mean_ms) : 0.0;
  return s;
}

template <int INNER_OPS>
__global__ void int8_dot4_kernel_unrolled(const int8_t* __restrict__ a,
                                          const int8_t* __restrict__ b,
                                          int32_t* __restrict__ out, int n_vec4,
                                          int items_per_thread) {
#if ASSUME_ALIGNED_HOT_PTRS
  a = static_cast<const int8_t*>(__builtin_assume_aligned(a, 4));
  b = static_cast<const int8_t*>(__builtin_assume_aligned(b, 4));
  out = static_cast<int32_t*>(__builtin_assume_aligned(out, 4));
#endif
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;
  const char4* a4 = reinterpret_cast<const char4*>(a);
  const char4* b4 = reinterpret_cast<const char4*>(b);

  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    if (idx >= n_vec4) {
      break;
    }

    char4 av = a4[idx];
    char4 bv = b4[idx];
    int32_t acc = 0;

#pragma unroll
    for (int i = 0; i < INNER_OPS; ++i) {
      acc = amd_mixed_dot(av, bv, acc, false);
      av.x = static_cast<signed char>(av.x + 1);
      bv.y = static_cast<signed char>(bv.y - 1);
    }

    out[idx] = acc;
  }
}

template <int INNER_OPS>
__global__ void int8_dot4_kernel_unrolled_scalar(const int8_t* __restrict__ a,
                                                 const int8_t* __restrict__ b,
                                                 int32_t* __restrict__ out, int n_vec4,
                                                 int items_per_thread) {
#if ASSUME_ALIGNED_HOT_PTRS
  a = static_cast<const int8_t*>(__builtin_assume_aligned(a, 4));
  b = static_cast<const int8_t*>(__builtin_assume_aligned(b, 4));
  out = static_cast<int32_t*>(__builtin_assume_aligned(out, 4));
#endif
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;

  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    if (idx >= n_vec4) {
      break;
    }

    int base = idx * 4;
    int av0 = static_cast<int>(a[base + 0]);
    int av1 = static_cast<int>(a[base + 1]);
    int av2 = static_cast<int>(a[base + 2]);
    int av3 = static_cast<int>(a[base + 3]);
    int bv0 = static_cast<int>(b[base + 0]);
    int bv1 = static_cast<int>(b[base + 1]);
    int bv2 = static_cast<int>(b[base + 2]);
    int bv3 = static_cast<int>(b[base + 3]);
    int32_t acc = 0;

#pragma unroll
    for (int i = 0; i < INNER_OPS; ++i) {
      acc += av0 * bv0 + av1 * bv1 + av2 * bv2 + av3 * bv3;
      av0 += 1;
      bv1 -= 1;
    }

    out[idx] = acc;
  }
}

__global__ void int8_dot4_kernel_runtime(const int8_t* __restrict__ a,
                                         const int8_t* __restrict__ b,
                                         int32_t* __restrict__ out, int n_vec4, int inner_ops,
                                         int items_per_thread) {
#if ASSUME_ALIGNED_HOT_PTRS
  a = static_cast<const int8_t*>(__builtin_assume_aligned(a, 4));
  b = static_cast<const int8_t*>(__builtin_assume_aligned(b, 4));
  out = static_cast<int32_t*>(__builtin_assume_aligned(out, 4));
#endif
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;
  const char4* a4 = reinterpret_cast<const char4*>(a);
  const char4* b4 = reinterpret_cast<const char4*>(b);

  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    if (idx >= n_vec4) {
      break;
    }

    char4 av = a4[idx];
    char4 bv = b4[idx];
    int32_t acc = 0;

    for (int i = 0; i < inner_ops; ++i) {
      acc = amd_mixed_dot(av, bv, acc, false);
      av.x = static_cast<signed char>(av.x + 1);
      bv.y = static_cast<signed char>(bv.y - 1);
    }

    out[idx] = acc;
  }
}

__global__ void int8_dot4_kernel_runtime_scalar(const int8_t* __restrict__ a,
                                                const int8_t* __restrict__ b,
                                                int32_t* __restrict__ out, int n_vec4,
                                                int inner_ops, int items_per_thread) {
#if ASSUME_ALIGNED_HOT_PTRS
  a = static_cast<const int8_t*>(__builtin_assume_aligned(a, 4));
  b = static_cast<const int8_t*>(__builtin_assume_aligned(b, 4));
  out = static_cast<int32_t*>(__builtin_assume_aligned(out, 4));
#endif
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;

  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    if (idx >= n_vec4) {
      break;
    }

    int base = idx * 4;
    int av0 = static_cast<int>(a[base + 0]);
    int av1 = static_cast<int>(a[base + 1]);
    int av2 = static_cast<int>(a[base + 2]);
    int av3 = static_cast<int>(a[base + 3]);
    int bv0 = static_cast<int>(b[base + 0]);
    int bv1 = static_cast<int>(b[base + 1]);
    int bv2 = static_cast<int>(b[base + 2]);
    int bv3 = static_cast<int>(b[base + 3]);
    int32_t acc = 0;

    for (int i = 0; i < inner_ops; ++i) {
      acc += av0 * bv0 + av1 * bv1 + av2 * bv2 + av3 * bv3;
      av0 += 1;
      bv1 -= 1;
    }

    out[idx] = acc;
  }
}

__device__ __forceinline__ uint32_t bump_low_byte(uint32_t x) {
  return (x & 0xFFFFFF00u) | ((x + 1u) & 0xFFu);
}

__device__ __forceinline__ uint32_t dec_byte1(uint32_t x) {
  const uint32_t b1 = ((x >> 8) - 1u) & 0xFFu;
  return (x & 0xFFFF00FFu) | (b1 << 8);
}

template <int INNER_OPS>
__global__ void int8_dot4_kernel_unrolled_sdot4(
    const int8_t* __restrict__ a, const int8_t* __restrict__ b, int32_t* __restrict__ out,
    int n_vec4, int items_per_thread) {
#if ASSUME_ALIGNED_HOT_PTRS
  a = static_cast<const int8_t*>(__builtin_assume_aligned(a, 4));
  b = static_cast<const int8_t*>(__builtin_assume_aligned(b, 4));
  out = static_cast<int32_t*>(__builtin_assume_aligned(out, 4));
#endif
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;
  const uint32_t* a32 = reinterpret_cast<const uint32_t*>(a);
  const uint32_t* b32 = reinterpret_cast<const uint32_t*>(b);

  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    if (idx >= n_vec4) {
      break;
    }

    uint32_t av = a32[idx];
    uint32_t bv = b32[idx];
    int32_t acc = 0;

#pragma unroll
    for (int i = 0; i < INNER_OPS; ++i) {
#if ENABLE_ISA_SDOT4_VARIANT
      acc = __builtin_amdgcn_sdot4(static_cast<int>(av), static_cast<int>(bv), acc, false);
#else
      const int a0 = static_cast<int>(static_cast<int8_t>(av & 0xFFu));
      const int a1 = static_cast<int>(static_cast<int8_t>((av >> 8) & 0xFFu));
      const int a2 = static_cast<int>(static_cast<int8_t>((av >> 16) & 0xFFu));
      const int a3 = static_cast<int>(static_cast<int8_t>((av >> 24) & 0xFFu));
      const int b0 = static_cast<int>(static_cast<int8_t>(bv & 0xFFu));
      const int b1 = static_cast<int>(static_cast<int8_t>((bv >> 8) & 0xFFu));
      const int b2 = static_cast<int>(static_cast<int8_t>((bv >> 16) & 0xFFu));
      const int b3 = static_cast<int>(static_cast<int8_t>((bv >> 24) & 0xFFu));
      acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
#endif
      av = bump_low_byte(av);
      bv = dec_byte1(bv);
    }
    out[idx] = acc;
  }
}

__global__ void int8_dot4_kernel_runtime_sdot4(
    const int8_t* __restrict__ a, const int8_t* __restrict__ b, int32_t* __restrict__ out,
    int n_vec4, int inner_ops, int items_per_thread) {
#if ASSUME_ALIGNED_HOT_PTRS
  a = static_cast<const int8_t*>(__builtin_assume_aligned(a, 4));
  b = static_cast<const int8_t*>(__builtin_assume_aligned(b, 4));
  out = static_cast<int32_t*>(__builtin_assume_aligned(out, 4));
#endif
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;
  const uint32_t* a32 = reinterpret_cast<const uint32_t*>(a);
  const uint32_t* b32 = reinterpret_cast<const uint32_t*>(b);

  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    if (idx >= n_vec4) {
      break;
    }

    uint32_t av = a32[idx];
    uint32_t bv = b32[idx];
    int32_t acc = 0;
    for (int i = 0; i < inner_ops; ++i) {
#if ENABLE_ISA_SDOT4_VARIANT
      acc = __builtin_amdgcn_sdot4(static_cast<int>(av), static_cast<int>(bv), acc, false);
#else
      const int a0 = static_cast<int>(static_cast<int8_t>(av & 0xFFu));
      const int a1 = static_cast<int>(static_cast<int8_t>((av >> 8) & 0xFFu));
      const int a2 = static_cast<int>(static_cast<int8_t>((av >> 16) & 0xFFu));
      const int a3 = static_cast<int>(static_cast<int8_t>((av >> 24) & 0xFFu));
      const int b0 = static_cast<int>(static_cast<int8_t>(bv & 0xFFu));
      const int b1 = static_cast<int>(static_cast<int8_t>((bv >> 8) & 0xFFu));
      const int b2 = static_cast<int>(static_cast<int8_t>((bv >> 16) & 0xFFu));
      const int b3 = static_cast<int>(static_cast<int8_t>((bv >> 24) & 0xFFu));
      acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
#endif
      av = bump_low_byte(av);
      bv = dec_byte1(bv);
    }
    out[idx] = acc;
  }
}

template <int INNER_OPS>
__global__ void int8_dot4_kernel_unrolled_scalar_ilp2(const int8_t* __restrict__ a,
                                                       const int8_t* __restrict__ b,
                                                       int32_t* __restrict__ out, int n_vec4,
                                                       int items_per_thread) {
#if ASSUME_ALIGNED_HOT_PTRS
  a = static_cast<const int8_t*>(__builtin_assume_aligned(a, 4));
  b = static_cast<const int8_t*>(__builtin_assume_aligned(b, 4));
  out = static_cast<int32_t*>(__builtin_assume_aligned(out, 4));
#endif
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;

  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    if (idx >= n_vec4) {
      break;
    }

    int base = idx * 4;
    int av0 = static_cast<int>(a[base + 0]);
    int av1 = static_cast<int>(a[base + 1]);
    int av2 = static_cast<int>(a[base + 2]);
    int av3 = static_cast<int>(a[base + 3]);
    int bv0 = static_cast<int>(b[base + 0]);
    int bv1 = static_cast<int>(b[base + 1]);
    int bv2 = static_cast<int>(b[base + 2]);
    int bv3 = static_cast<int>(b[base + 3]);
    int32_t acc0 = 0;
    int32_t acc1 = 0;

    int i = 0;
#pragma unroll
    for (; i + 1 < INNER_OPS; i += 2) {
      acc0 += av0 * bv0 + av1 * bv1 + av2 * bv2 + av3 * bv3;
      const int av0_next = av0 + 1;
      const int bv1_next = bv1 - 1;
      acc1 += av0_next * bv0 + av1 * bv1_next + av2 * bv2 + av3 * bv3;
      av0 = av0_next + 1;
      bv1 = bv1_next - 1;
    }
#pragma unroll
    for (; i < INNER_OPS; ++i) {
      acc0 += av0 * bv0 + av1 * bv1 + av2 * bv2 + av3 * bv3;
      av0 += 1;
      bv1 -= 1;
    }

    out[idx] = acc0 + acc1;
  }
}

__global__ void int8_dot4_kernel_runtime_scalar_ilp2(const int8_t* __restrict__ a,
                                                      const int8_t* __restrict__ b,
                                                      int32_t* __restrict__ out, int n_vec4,
                                                      int inner_ops, int items_per_thread) {
#if ASSUME_ALIGNED_HOT_PTRS
  a = static_cast<const int8_t*>(__builtin_assume_aligned(a, 4));
  b = static_cast<const int8_t*>(__builtin_assume_aligned(b, 4));
  out = static_cast<int32_t*>(__builtin_assume_aligned(out, 4));
#endif
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;

  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    if (idx >= n_vec4) {
      break;
    }

    int base = idx * 4;
    int av0 = static_cast<int>(a[base + 0]);
    int av1 = static_cast<int>(a[base + 1]);
    int av2 = static_cast<int>(a[base + 2]);
    int av3 = static_cast<int>(a[base + 3]);
    int bv0 = static_cast<int>(b[base + 0]);
    int bv1 = static_cast<int>(b[base + 1]);
    int bv2 = static_cast<int>(b[base + 2]);
    int bv3 = static_cast<int>(b[base + 3]);
    int32_t acc0 = 0;
    int32_t acc1 = 0;

    int i = 0;
    for (; i + 1 < inner_ops; i += 2) {
      acc0 += av0 * bv0 + av1 * bv1 + av2 * bv2 + av3 * bv3;
      const int av0_next = av0 + 1;
      const int bv1_next = bv1 - 1;
      acc1 += av0_next * bv0 + av1 * bv1_next + av2 * bv2 + av3 * bv3;
      av0 = av0_next + 1;
      bv1 = bv1_next - 1;
    }
    for (; i < inner_ops; ++i) {
      acc0 += av0 * bv0 + av1 * bv1 + av2 * bv2 + av3 * bv3;
      av0 += 1;
      bv1 -= 1;
    }

    out[idx] = acc0 + acc1;
  }
}

template <int INNER_OPS>
__global__ void int8_convlike_kernel_shared_weight(const int8_t* __restrict__ a,
                                                    const int8_t* __restrict__ b,
                                                    int32_t* __restrict__ out, int n_vec4,
                                                    int items_per_thread, bool lds_stage_weight,
                                                    bool lds_padding, bool lds_double_buffer) {
#if ASSUME_ALIGNED_HOT_PTRS
  a = static_cast<const int8_t*>(__builtin_assume_aligned(a, 4));
  b = static_cast<const int8_t*>(__builtin_assume_aligned(b, 4));
  out = static_cast<int32_t*>(__builtin_assume_aligned(out, 4));
#endif
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;

  extern __shared__ int8_t smem[];
  const int lds_stride = lds_padding ? 5 : 4;
  int8_t* s_w0 = smem;
  int8_t* s_w1 = lds_double_buffer ? (smem + lds_stride) : nullptr;

  const int weight_idx0 = blockIdx.x % n_vec4;
  const int weight_base0 = weight_idx0 * 4;
  int weight_idx1 = weight_idx0;
  int weight_base1 = weight_base0;
  if (lds_double_buffer) {
    weight_idx1 = (weight_idx0 + 1) % n_vec4;
    weight_base1 = weight_idx1 * 4;
  }

  if (lds_stage_weight) {
    if (threadIdx.x < 4) {
      s_w0[threadIdx.x] = b[weight_base0 + threadIdx.x];
      if (lds_double_buffer) {
        s_w1[threadIdx.x] = b[weight_base1 + threadIdx.x];
      }
    }
    if (lds_padding && threadIdx.x == 0) {
      s_w0[4] = 0;
      if (lds_double_buffer) {
        s_w1[4] = 0;
      }
    }
    __syncthreads();
  }

  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    if (idx >= n_vec4) {
      break;
    }

    const int base = idx * 4;
    int av0 = static_cast<int>(a[base + 0]);
    int av1 = static_cast<int>(a[base + 1]);
    int av2 = static_cast<int>(a[base + 2]);
    int av3 = static_cast<int>(a[base + 3]);
    int32_t acc = 0;

    int bv0, bv1, bv2, bv3;
    if (lds_stage_weight) {
      bv0 = static_cast<int>(s_w0[0]);
      bv1 = static_cast<int>(s_w0[1]);
      bv2 = static_cast<int>(s_w0[2]);
      bv3 = static_cast<int>(s_w0[3]);
    } else {
      bv0 = static_cast<int>(b[weight_base0 + 0]);
      bv1 = static_cast<int>(b[weight_base0 + 1]);
      bv2 = static_cast<int>(b[weight_base0 + 2]);
      bv3 = static_cast<int>(b[weight_base0 + 3]);
    }

#pragma unroll
    for (int i = 0; i < INNER_OPS; ++i) {
      if (lds_double_buffer && lds_stage_weight && (i & 1)) {
        const int8_t* src = s_w1;
        bv0 = static_cast<int>(src[0]);
        bv1 = static_cast<int>(src[1]);
        bv2 = static_cast<int>(src[2]);
        bv3 = static_cast<int>(src[3]);
      }
      acc += av0 * bv0 + av1 * bv1 + av2 * bv2 + av3 * bv3;
      av0 += 1;
      bv1 -= 1;
    }
    out[idx] = acc;
  }
}

__global__ void int8_convlike_kernel_shared_weight_runtime(
    const int8_t* __restrict__ a, const int8_t* __restrict__ b, int32_t* __restrict__ out,
    int n_vec4, int inner_ops, int items_per_thread, bool lds_stage_weight, bool lds_padding,
    bool lds_double_buffer) {
#if ASSUME_ALIGNED_HOT_PTRS
  a = static_cast<const int8_t*>(__builtin_assume_aligned(a, 4));
  b = static_cast<const int8_t*>(__builtin_assume_aligned(b, 4));
  out = static_cast<int32_t*>(__builtin_assume_aligned(out, 4));
#endif
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;

  extern __shared__ int8_t smem[];
  const int lds_stride = lds_padding ? 5 : 4;
  int8_t* s_w0 = smem;
  int8_t* s_w1 = lds_double_buffer ? (smem + lds_stride) : nullptr;

  const int weight_idx0 = blockIdx.x % n_vec4;
  const int weight_base0 = weight_idx0 * 4;
  int weight_idx1 = weight_idx0;
  int weight_base1 = weight_base0;
  if (lds_double_buffer) {
    weight_idx1 = (weight_idx0 + 1) % n_vec4;
    weight_base1 = weight_idx1 * 4;
  }

  if (lds_stage_weight) {
    if (threadIdx.x < 4) {
      s_w0[threadIdx.x] = b[weight_base0 + threadIdx.x];
      if (lds_double_buffer) {
        s_w1[threadIdx.x] = b[weight_base1 + threadIdx.x];
      }
    }
    if (lds_padding && threadIdx.x == 0) {
      s_w0[4] = 0;
      if (lds_double_buffer) {
        s_w1[4] = 0;
      }
    }
    __syncthreads();
  }

  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    if (idx >= n_vec4) {
      break;
    }

    const int base = idx * 4;
    int av0 = static_cast<int>(a[base + 0]);
    int av1 = static_cast<int>(a[base + 1]);
    int av2 = static_cast<int>(a[base + 2]);
    int av3 = static_cast<int>(a[base + 3]);
    int32_t acc = 0;

    int bv0, bv1, bv2, bv3;
    if (lds_stage_weight) {
      bv0 = static_cast<int>(s_w0[0]);
      bv1 = static_cast<int>(s_w0[1]);
      bv2 = static_cast<int>(s_w0[2]);
      bv3 = static_cast<int>(s_w0[3]);
    } else {
      bv0 = static_cast<int>(b[weight_base0 + 0]);
      bv1 = static_cast<int>(b[weight_base0 + 1]);
      bv2 = static_cast<int>(b[weight_base0 + 2]);
      bv3 = static_cast<int>(b[weight_base0 + 3]);
    }

    for (int i = 0; i < inner_ops; ++i) {
      if (lds_double_buffer && lds_stage_weight && (i & 1)) {
        const int8_t* src = s_w1;
        bv0 = static_cast<int>(src[0]);
        bv1 = static_cast<int>(src[1]);
        bv2 = static_cast<int>(src[2]);
        bv3 = static_cast<int>(src[3]);
      }
      acc += av0 * bv0 + av1 * bv1 + av2 * bv2 + av3 * bv3;
      av0 += 1;
      bv1 -= 1;
    }
    out[idx] = acc;
  }
}

__device__ __forceinline__ int quantize_i8(float x) {
  int q = static_cast<int>(x >= 0.0f ? (x + 0.5f) : (x - 0.5f));
  q = max(-127, min(127, q));
  return q;
}

template <int INNER_OPS>
__global__ void int8_dot4_kernel_unrolled_scalar_nobounds(const int8_t* a, const int8_t* b,
                                                           int32_t* out, int items_per_thread) {
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;
  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    int base = idx * 4;
    int av0 = static_cast<int>(a[base + 0]);
    int av1 = static_cast<int>(a[base + 1]);
    int av2 = static_cast<int>(a[base + 2]);
    int av3 = static_cast<int>(a[base + 3]);
    int bv0 = static_cast<int>(b[base + 0]);
    int bv1 = static_cast<int>(b[base + 1]);
    int bv2 = static_cast<int>(b[base + 2]);
    int bv3 = static_cast<int>(b[base + 3]);
    int32_t acc = 0;

#pragma unroll
    for (int i = 0; i < INNER_OPS; ++i) {
      acc += av0 * bv0 + av1 * bv1 + av2 * bv2 + av3 * bv3;
      av0 += 1;
      bv1 -= 1;
    }
    out[idx] = acc;
  }
}

template <int INNER_OPS>
__global__ void int8_dot4_kernel_unrolled_scalar_advanced(
    const int8_t* a, const int8_t* b, int32_t* out, int n_vec4, int items_per_thread,
    const float* scales, const float* biases, bool force_inloop_scale_bias,
    bool force_per_iter_requant, bool lds_stage_input, bool lds_stage_weight, bool lds_padding,
    bool lds_double_buffer, bool force_mixed_int8_path) {
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;

  extern __shared__ int8_t smem[];
  const bool use_lds = lds_stage_input || lds_stage_weight;
  const int lds_stride = lds_padding ? 5 : 4;
  const int vectors_per_block = blockDim.x * items_per_thread;
  int8_t* s_a = lds_stage_input ? smem : nullptr;
  int8_t* s_b = lds_stage_weight
                    ? (smem + (lds_stage_input ? vectors_per_block * lds_stride : 0))
                    : nullptr;

  if (use_lds) {
    for (int item = 0; item < items_per_thread; ++item) {
      const int idx = idx0 + item;
      if (idx >= n_vec4) {
        continue;
      }
      const int base = idx * 4;
      const int local_vec = threadIdx.x * items_per_thread + item;
      const int local_base = local_vec * lds_stride;
      if (lds_stage_input) {
        s_a[local_base + 0] = a[base + 0];
        s_a[local_base + 1] = a[base + 1];
        s_a[local_base + 2] = a[base + 2];
        s_a[local_base + 3] = a[base + 3];
      }
      if (lds_stage_weight) {
        s_b[local_base + 0] = b[base + 0];
        s_b[local_base + 1] = b[base + 1];
        s_b[local_base + 2] = b[base + 2];
        s_b[local_base + 3] = b[base + 3];
      }
    }
    __syncthreads();
  }

  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    if (idx >= n_vec4) {
      break;
    }

    int av0, av1, av2, av3;
    int bv0, bv1, bv2, bv3;
    if (use_lds) {
      const int local_vec = threadIdx.x * items_per_thread + item;
      const int local_base = local_vec * lds_stride;
      if (lds_stage_input) {
        av0 = static_cast<int>(s_a[local_base + 0]);
        av1 = static_cast<int>(s_a[local_base + 1]);
        av2 = static_cast<int>(s_a[local_base + 2]);
        av3 = static_cast<int>(s_a[local_base + 3]);
      } else {
        const int base = idx * 4;
        av0 = static_cast<int>(a[base + 0]);
        av1 = static_cast<int>(a[base + 1]);
        av2 = static_cast<int>(a[base + 2]);
        av3 = static_cast<int>(a[base + 3]);
      }
      if (lds_stage_weight) {
        bv0 = static_cast<int>(s_b[local_base + 0]);
        bv1 = static_cast<int>(s_b[local_base + 1]);
        bv2 = static_cast<int>(s_b[local_base + 2]);
        bv3 = static_cast<int>(s_b[local_base + 3]);
      } else {
        const int base = idx * 4;
        bv0 = static_cast<int>(b[base + 0]);
        bv1 = static_cast<int>(b[base + 1]);
        bv2 = static_cast<int>(b[base + 2]);
        bv3 = static_cast<int>(b[base + 3]);
      }
    } else {
      const int base = idx * 4;
      av0 = static_cast<int>(a[base + 0]);
      av1 = static_cast<int>(a[base + 1]);
      av2 = static_cast<int>(a[base + 2]);
      av3 = static_cast<int>(a[base + 3]);
      bv0 = static_cast<int>(b[base + 0]);
      bv1 = static_cast<int>(b[base + 1]);
      bv2 = static_cast<int>(b[base + 2]);
      bv3 = static_cast<int>(b[base + 3]);
    }

    float scale = scales[idx];
    float bias = biases[idx];
    int32_t acc_i = 0;
    float acc_f = 0.0f;

    int i = 0;
    if (lds_double_buffer) {
#pragma unroll
      for (; i + 1 < INNER_OPS; i += 2) {
        if (force_inloop_scale_bias) {
          scale = scales[idx];
          bias = biases[idx];
        }
        if (force_mixed_int8_path) {
          const float dot0 = static_cast<float>(av0 * bv0 + av1 * bv1 + av2 * bv2 + av3 * bv3);
          if (force_per_iter_requant) {
            acc_f = static_cast<float>(quantize_i8((acc_f + dot0) * scale + bias));
          } else {
            acc_f += dot0;
          }
        } else {
          const int32_t dot0 = av0 * bv0 + av1 * bv1 + av2 * bv2 + av3 * bv3;
          if (force_per_iter_requant) {
            acc_i = quantize_i8(static_cast<float>(acc_i + dot0) * scale + bias);
          } else {
            acc_i += dot0;
          }
        }

        const int av0_next = av0 + 1;
        const int bv1_next = bv1 - 1;
        if (force_inloop_scale_bias) {
          scale = scales[idx];
          bias = biases[idx];
        }
        if (force_mixed_int8_path) {
          const float dot1 =
              static_cast<float>(av0_next * bv0 + av1 * bv1_next + av2 * bv2 + av3 * bv3);
          if (force_per_iter_requant) {
            acc_f = static_cast<float>(quantize_i8((acc_f + dot1) * scale + bias));
          } else {
            acc_f += dot1;
          }
        } else {
          const int32_t dot1 = av0_next * bv0 + av1 * bv1_next + av2 * bv2 + av3 * bv3;
          if (force_per_iter_requant) {
            acc_i = quantize_i8(static_cast<float>(acc_i + dot1) * scale + bias);
          } else {
            acc_i += dot1;
          }
        }
        av0 = av0_next + 1;
        bv1 = bv1_next - 1;
      }
    }

#pragma unroll
    for (; i < INNER_OPS; ++i) {
      if (force_inloop_scale_bias) {
        scale = scales[idx];
        bias = biases[idx];
      }
      if (force_mixed_int8_path) {
        const float dot = static_cast<float>(av0 * bv0 + av1 * bv1 + av2 * bv2 + av3 * bv3);
        if (force_per_iter_requant) {
          acc_f = static_cast<float>(quantize_i8((acc_f + dot) * scale + bias));
        } else {
          acc_f += dot;
        }
      } else {
        const int32_t dot = av0 * bv0 + av1 * bv1 + av2 * bv2 + av3 * bv3;
        if (force_per_iter_requant) {
          acc_i = quantize_i8(static_cast<float>(acc_i + dot) * scale + bias);
        } else {
          acc_i += dot;
        }
      }
      av0 += 1;
      bv1 -= 1;
    }

    if (force_mixed_int8_path) {
      if (force_per_iter_requant) {
        out[idx] = static_cast<int32_t>(acc_f);
      } else {
        out[idx] = quantize_i8(acc_f * scale + bias);
      }
    } else {
      if (force_per_iter_requant) {
        out[idx] = acc_i;
      } else {
        out[idx] = quantize_i8(static_cast<float>(acc_i) * scale + bias);
      }
    }
  }
}

__global__ void int8_post_kernel(int32_t* out, int n_vec4) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vec4) {
    return;
  }
  const int x = out[idx] + 3;
  out[idx] = max(0, x);
}

__global__ void int8_adjacent_pass_kernel(const int32_t* in, int32_t* out, int n_vec4) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vec4) {
    return;
  }
  const int x = in[idx];
  out[idx] = x + ((x >> 3) & 7);
}

template <int INNER_OPS>
__global__ void fp8_fma_kernel_unrolled(const float* __restrict__ a, const float* __restrict__ b,
                                        float* __restrict__ out, int n, int items_per_thread) {
#if ASSUME_ALIGNED_HOT_PTRS
  a = static_cast<const float*>(__builtin_assume_aligned(a, 4));
  b = static_cast<const float*>(__builtin_assume_aligned(b, 4));
  out = static_cast<float*>(__builtin_assume_aligned(out, 4));
#endif
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;

  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    if (idx >= n) {
      break;
    }

    __hip_fp8_e4m3 qa = __hip_fp8_e4m3(a[idx]);
    __hip_fp8_e4m3 qb = __hip_fp8_e4m3(b[idx]);

    float af = static_cast<float>(qa);
    float bf = static_cast<float>(qb);
    float acc = 0.0f;

#pragma unroll
    for (int i = 0; i < INNER_OPS; ++i) {
      acc = fmaf(af, bf, acc);
      qa = __hip_fp8_e4m3(af * 1.0001f + 0.00001f * static_cast<float>(i));
      qb = __hip_fp8_e4m3(bf * 0.9999f - 0.00001f * static_cast<float>(i));
      af = static_cast<float>(qa);
      bf = static_cast<float>(qb);
    }

    out[idx] = acc;
  }
}

__global__ void fp8_fma_kernel_runtime(const float* __restrict__ a, const float* __restrict__ b,
                                       float* __restrict__ out, int n, int inner_ops,
                                       int items_per_thread) {
#if ASSUME_ALIGNED_HOT_PTRS
  a = static_cast<const float*>(__builtin_assume_aligned(a, 4));
  b = static_cast<const float*>(__builtin_assume_aligned(b, 4));
  out = static_cast<float*>(__builtin_assume_aligned(out, 4));
#endif
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;

  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    if (idx >= n) {
      break;
    }

    __hip_fp8_e4m3 qa = __hip_fp8_e4m3(a[idx]);
    __hip_fp8_e4m3 qb = __hip_fp8_e4m3(b[idx]);

    float af = static_cast<float>(qa);
    float bf = static_cast<float>(qb);
    float acc = 0.0f;

    for (int i = 0; i < inner_ops; ++i) {
      acc = fmaf(af, bf, acc);
      // Keep explicit fp8 conversion in-loop to emulate quantized dataflow pressure.
      qa = __hip_fp8_e4m3(af * 1.0001f + 0.00001f * static_cast<float>(i));
      qb = __hip_fp8_e4m3(bf * 0.9999f - 0.00001f * static_cast<float>(i));
      af = static_cast<float>(qa);
      bf = static_cast<float>(qb);
    }

    out[idx] = acc;
  }
}

template <int INNER_OPS, bool CHECK_BOUNDS>
__global__ void fp8_fma_kernel_unrolled_general(const float* __restrict__ a,
                                                const float* __restrict__ b,
                                                const uint8_t* __restrict__ a_q,
                                                const uint8_t* __restrict__ b_q,
                                                float* __restrict__ out, int n,
                                                int items_per_thread, bool use_quantized_io,
                                                bool force_per_iter_requant) {
#if ASSUME_ALIGNED_HOT_PTRS
  a = static_cast<const float*>(__builtin_assume_aligned(a, 4));
  b = static_cast<const float*>(__builtin_assume_aligned(b, 4));
  a_q = static_cast<const uint8_t*>(__builtin_assume_aligned(a_q, 1));
  b_q = static_cast<const uint8_t*>(__builtin_assume_aligned(b_q, 1));
  out = static_cast<float*>(__builtin_assume_aligned(out, 4));
#endif
  int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;

  for (int item = 0; item < items_per_thread; ++item) {
    int idx = idx0 + item;
    if (CHECK_BOUNDS && idx >= n) {
      break;
    }

    __hip_fp8_e4m3 qa;
    __hip_fp8_e4m3 qb;
    if (use_quantized_io) {
      qa = *reinterpret_cast<const __hip_fp8_e4m3*>(a_q + idx);
      qb = *reinterpret_cast<const __hip_fp8_e4m3*>(b_q + idx);
    } else {
      qa = __hip_fp8_e4m3(a[idx]);
      qb = __hip_fp8_e4m3(b[idx]);
    }

    float af = static_cast<float>(qa);
    float bf = static_cast<float>(qb);
    float acc = 0.0f;

    if (force_per_iter_requant) {
#pragma unroll
      for (int i = 0; i < INNER_OPS; ++i) {
        acc = fmaf(af, bf, acc);
        qa = __hip_fp8_e4m3(af * 1.0001f + 0.00001f * static_cast<float>(i));
        qb = __hip_fp8_e4m3(bf * 0.9999f - 0.00001f * static_cast<float>(i));
        af = static_cast<float>(qa);
        bf = static_cast<float>(qb);
      }
    } else {
#pragma unroll
      for (int i = 0; i < INNER_OPS; ++i) {
        acc = fmaf(af, bf, acc);
        af = af * 1.0001f + 0.00001f * static_cast<float>(i);
        bf = bf * 0.9999f - 0.00001f * static_cast<float>(i);
      }
      qa = __hip_fp8_e4m3(af);
      qb = __hip_fp8_e4m3(bf);
      // Retain one quantization point to mimic epilogue requant behavior.
      acc += 0.0001f * (static_cast<float>(qa) + static_cast<float>(qb));
    }

    out[idx] = acc;
  }
}

__global__ void fp8_post_kernel(float* out, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = fmaxf(0.0f, out[idx] + 0.003f);
}

__global__ void fp8_adjacent_pass_kernel(const float* in, float* out, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  float x = in[idx];
  out[idx] = x + 0.125f * sinf(x);
}

static std::pair<Stats, double> run_int8(const Config& cfg) {
  const int n_vec4 = cfg.elements;
  const int n_scalar = n_vec4 * 4;

  std::vector<int8_t> h_a(n_scalar);
  std::vector<int8_t> h_b(n_scalar);
  std::vector<float> h_scales;
  std::vector<float> h_biases;

  std::mt19937 rng(cfg.seed);
  std::uniform_int_distribution<int> dist(-127, 127);
  for (int i = 0; i < n_scalar; ++i) {
    h_a[i] = static_cast<int8_t>(dist(rng));
    h_b[i] = static_cast<int8_t>(dist(rng));
  }

  const bool needs_advanced_scalar =
      cfg.force_scalar_int8_io &&
      (cfg.force_inloop_scale_bias || cfg.force_per_iter_requant || cfg.lds_stage_input ||
       cfg.lds_stage_weight || cfg.lds_padding || cfg.lds_double_buffer || cfg.force_mixed_int8_path);
  if (needs_advanced_scalar) {
    h_scales.resize(n_vec4);
    h_biases.resize(n_vec4);
    std::uniform_real_distribution<float> scale_dist(0.80f, 1.20f);
    std::uniform_real_distribution<float> bias_dist(-3.0f, 3.0f);
    for (int i = 0; i < n_vec4; ++i) {
      h_scales[i] = scale_dist(rng);
      h_biases[i] = bias_dist(rng);
    }
  }

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_out = nullptr;
  int32_t* d_tmp = nullptr;
  float* d_scales = nullptr;
  float* d_biases = nullptr;
  HIP_CHECK(hipMalloc(&d_a, h_a.size() * sizeof(int8_t)));
  HIP_CHECK(hipMalloc(&d_b, h_b.size() * sizeof(int8_t)));
  HIP_CHECK(hipMalloc(&d_out, n_vec4 * sizeof(int32_t)));
  if (cfg.force_two_pass) {
    HIP_CHECK(hipMalloc(&d_tmp, n_vec4 * sizeof(int32_t)));
  }
  if (needs_advanced_scalar) {
    HIP_CHECK(hipMalloc(&d_scales, h_scales.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_biases, h_biases.size() * sizeof(float)));
  }

  HIP_CHECK(hipMemcpy(d_a, h_a.data(), h_a.size() * sizeof(int8_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), h_b.size() * sizeof(int8_t), hipMemcpyHostToDevice));
  if (needs_advanced_scalar) {
    HIP_CHECK(
        hipMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_biases, h_biases.data(), h_biases.size() * sizeof(float), hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipMemset(d_out, 0, n_vec4 * sizeof(int32_t)));
  if (d_tmp) {
    HIP_CHECK(hipMemset(d_tmp, 0, n_vec4 * sizeof(int32_t)));
  }

  const dim3 block(cfg.threads);
  const int logical_threads = cfg.threads * cfg.items_per_thread;
  const dim3 grid((n_vec4 + logical_threads - 1) / logical_threads);
  const dim3 post_grid((n_vec4 + cfg.threads - 1) / cfg.threads);

  size_t lds_shared_bytes = 0;
  if (cfg.lds_stage_input || cfg.lds_stage_weight) {
    const int lds_stride = cfg.lds_padding ? 5 : 4;
    const int vectors_per_block = cfg.threads * cfg.items_per_thread;
    if (cfg.lds_stage_input) {
      lds_shared_bytes += vectors_per_block * lds_stride * sizeof(int8_t);
    }
    if (cfg.lds_stage_weight) {
      lds_shared_bytes += vectors_per_block * lds_stride * sizeof(int8_t);
    }
  }
  size_t conv_shared_bytes = 0;
  if (cfg.force_convlike_int8 && cfg.lds_stage_weight) {
    const int lds_stride = cfg.lds_padding ? 5 : 4;
    conv_shared_bytes = static_cast<size_t>(lds_stride) * sizeof(int8_t);
    if (cfg.lds_double_buffer) {
      conv_shared_bytes *= 2;
    }
  }

  auto launch_scalar_unrolled_checked = [&](const int8_t* a_ptr, const int8_t* b_ptr, int32_t* out_ptr,
                                            int n_local, dim3 grid_local) {
    switch (cfg.inner_int8) {
      case 8:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar<8>, grid_local, block, 0, 0, a_ptr, b_ptr,
                           out_ptr, n_local, cfg.items_per_thread);
        break;
      case 16:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar<16>, grid_local, block, 0, 0, a_ptr,
                           b_ptr, out_ptr, n_local, cfg.items_per_thread);
        break;
      case 32:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar<32>, grid_local, block, 0, 0, a_ptr,
                           b_ptr, out_ptr, n_local, cfg.items_per_thread);
        break;
      case 64:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar<64>, grid_local, block, 0, 0, a_ptr,
                           b_ptr, out_ptr, n_local, cfg.items_per_thread);
        break;
      default:
        hipLaunchKernelGGL(int8_dot4_kernel_runtime_scalar, grid_local, block, 0, 0, a_ptr, b_ptr,
                           out_ptr, n_local, cfg.inner_int8, cfg.items_per_thread);
        break;
    }
  };

  auto launch_scalar_unrolled_nobounds = [&](const int8_t* a_ptr, const int8_t* b_ptr,
                                             int32_t* out_ptr, dim3 grid_local) {
    switch (cfg.inner_int8) {
      case 8:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar_nobounds<8>, grid_local, block, 0, 0,
                           a_ptr, b_ptr, out_ptr, cfg.items_per_thread);
        break;
      case 16:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar_nobounds<16>, grid_local, block, 0, 0,
                           a_ptr, b_ptr, out_ptr, cfg.items_per_thread);
        break;
      case 32:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar_nobounds<32>, grid_local, block, 0, 0,
                           a_ptr, b_ptr, out_ptr, cfg.items_per_thread);
        break;
      case 64:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar_nobounds<64>, grid_local, block, 0, 0,
                           a_ptr, b_ptr, out_ptr, cfg.items_per_thread);
        break;
      default:
        // No no-bounds runtime variant; fall back to checked.
        break;
    }
  };

  auto launch_scalar_advanced = [&](const int8_t* a_ptr, const int8_t* b_ptr, int32_t* out_ptr,
                                    int n_local, dim3 grid_local, const float* scales_ptr,
                                    const float* biases_ptr) {
    switch (cfg.inner_int8) {
      case 8:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar_advanced<8>, grid_local, block,
                           lds_shared_bytes, 0, a_ptr, b_ptr, out_ptr, n_local,
                           cfg.items_per_thread, scales_ptr, biases_ptr,
                           cfg.force_inloop_scale_bias, cfg.force_per_iter_requant,
                           cfg.lds_stage_input, cfg.lds_stage_weight, cfg.lds_padding,
                           cfg.lds_double_buffer, cfg.force_mixed_int8_path);
        break;
      case 16:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar_advanced<16>, grid_local, block,
                           lds_shared_bytes, 0, a_ptr, b_ptr, out_ptr, n_local,
                           cfg.items_per_thread, scales_ptr, biases_ptr,
                           cfg.force_inloop_scale_bias, cfg.force_per_iter_requant,
                           cfg.lds_stage_input, cfg.lds_stage_weight, cfg.lds_padding,
                           cfg.lds_double_buffer, cfg.force_mixed_int8_path);
        break;
      case 32:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar_advanced<32>, grid_local, block,
                           lds_shared_bytes, 0, a_ptr, b_ptr, out_ptr, n_local,
                           cfg.items_per_thread, scales_ptr, biases_ptr,
                           cfg.force_inloop_scale_bias, cfg.force_per_iter_requant,
                           cfg.lds_stage_input, cfg.lds_stage_weight, cfg.lds_padding,
                           cfg.lds_double_buffer, cfg.force_mixed_int8_path);
        break;
      case 64:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar_advanced<64>, grid_local, block,
                           lds_shared_bytes, 0, a_ptr, b_ptr, out_ptr, n_local,
                           cfg.items_per_thread, scales_ptr, biases_ptr,
                           cfg.force_inloop_scale_bias, cfg.force_per_iter_requant,
                           cfg.lds_stage_input, cfg.lds_stage_weight, cfg.lds_padding,
                           cfg.lds_double_buffer, cfg.force_mixed_int8_path);
        break;
      default:
        hipLaunchKernelGGL(int8_dot4_kernel_runtime_scalar, grid_local, block, 0, 0, a_ptr, b_ptr,
                           out_ptr, n_local, cfg.inner_int8, cfg.items_per_thread);
        break;
    }
  };

  auto launch_scalar_ilp2 = [&](const int8_t* a_ptr, const int8_t* b_ptr, int32_t* out_ptr,
                                int n_local, dim3 grid_local) {
    switch (cfg.inner_int8) {
      case 8:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar_ilp2<8>, grid_local, block, 0, 0, a_ptr,
                           b_ptr, out_ptr, n_local, cfg.items_per_thread);
        break;
      case 16:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar_ilp2<16>, grid_local, block, 0, 0, a_ptr,
                           b_ptr, out_ptr, n_local, cfg.items_per_thread);
        break;
      case 32:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar_ilp2<32>, grid_local, block, 0, 0, a_ptr,
                           b_ptr, out_ptr, n_local, cfg.items_per_thread);
        break;
      case 64:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar_ilp2<64>, grid_local, block, 0, 0, a_ptr,
                           b_ptr, out_ptr, n_local, cfg.items_per_thread);
        break;
      default:
        hipLaunchKernelGGL(int8_dot4_kernel_runtime_scalar_ilp2, grid_local, block, 0, 0, a_ptr,
                           b_ptr, out_ptr, n_local, cfg.inner_int8, cfg.items_per_thread);
        break;
    }
  };

  auto launch_isa_packed = [&](const int8_t* a_ptr, const int8_t* b_ptr, int32_t* out_ptr,
                               int n_local, dim3 grid_local) {
    switch (cfg.inner_int8) {
      case 8:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_sdot4<8>, grid_local, block, 0, 0, a_ptr, b_ptr,
                           out_ptr, n_local, cfg.items_per_thread);
        break;
      case 16:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_sdot4<16>, grid_local, block, 0, 0, a_ptr,
                           b_ptr, out_ptr, n_local, cfg.items_per_thread);
        break;
      case 32:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_sdot4<32>, grid_local, block, 0, 0, a_ptr,
                           b_ptr, out_ptr, n_local, cfg.items_per_thread);
        break;
      case 64:
        hipLaunchKernelGGL(int8_dot4_kernel_unrolled_sdot4<64>, grid_local, block, 0, 0, a_ptr,
                           b_ptr, out_ptr, n_local, cfg.items_per_thread);
        break;
      default:
        hipLaunchKernelGGL(int8_dot4_kernel_runtime_sdot4, grid_local, block, 0, 0, a_ptr, b_ptr,
                           out_ptr, n_local, cfg.inner_int8, cfg.items_per_thread);
        break;
    }
  };

  auto launch_convlike = [&](const int8_t* a_ptr, const int8_t* b_ptr, int32_t* out_ptr,
                             int n_local, dim3 grid_local) {
    switch (cfg.inner_int8) {
      case 8:
        hipLaunchKernelGGL(int8_convlike_kernel_shared_weight<8>, grid_local, block, conv_shared_bytes,
                           0, a_ptr, b_ptr, out_ptr, n_local, cfg.items_per_thread,
                           cfg.lds_stage_weight, cfg.lds_padding, cfg.lds_double_buffer);
        break;
      case 16:
        hipLaunchKernelGGL(int8_convlike_kernel_shared_weight<16>, grid_local, block, conv_shared_bytes,
                           0, a_ptr, b_ptr, out_ptr, n_local, cfg.items_per_thread,
                           cfg.lds_stage_weight, cfg.lds_padding, cfg.lds_double_buffer);
        break;
      case 32:
        hipLaunchKernelGGL(int8_convlike_kernel_shared_weight<32>, grid_local, block, conv_shared_bytes,
                           0, a_ptr, b_ptr, out_ptr, n_local, cfg.items_per_thread,
                           cfg.lds_stage_weight, cfg.lds_padding, cfg.lds_double_buffer);
        break;
      case 64:
        hipLaunchKernelGGL(int8_convlike_kernel_shared_weight<64>, grid_local, block, conv_shared_bytes,
                           0, a_ptr, b_ptr, out_ptr, n_local, cfg.items_per_thread,
                           cfg.lds_stage_weight, cfg.lds_padding, cfg.lds_double_buffer);
        break;
      default:
        hipLaunchKernelGGL(int8_convlike_kernel_shared_weight_runtime, grid_local, block,
                           conv_shared_bytes, 0, a_ptr, b_ptr, out_ptr, n_local, cfg.inner_int8,
                           cfg.items_per_thread, cfg.lds_stage_weight, cfg.lds_padding,
                           cfg.lds_double_buffer);
        break;
    }
  };

  auto launch_int8_core = [&](int32_t* out_ptr) {
    if (cfg.force_convlike_int8) {
      launch_convlike(d_a, d_b, out_ptr, n_vec4, grid);
      return;
    }

    if (cfg.force_runtime_inner_loops) {
      if (cfg.force_scalar_int8_io) {
        if (cfg.force_ilp2_int8) {
          hipLaunchKernelGGL(int8_dot4_kernel_runtime_scalar_ilp2, grid, block, 0, 0, d_a, d_b,
                             out_ptr, n_vec4, cfg.inner_int8, cfg.items_per_thread);
        } else {
          hipLaunchKernelGGL(int8_dot4_kernel_runtime_scalar, grid, block, 0, 0, d_a, d_b, out_ptr,
                             n_vec4, cfg.inner_int8, cfg.items_per_thread);
        }
      } else {
        if (cfg.force_isa_packed_int8_io) {
          hipLaunchKernelGGL(int8_dot4_kernel_runtime_sdot4, grid, block, 0, 0, d_a, d_b, out_ptr,
                             n_vec4, cfg.inner_int8, cfg.items_per_thread);
        } else {
          hipLaunchKernelGGL(int8_dot4_kernel_runtime, grid, block, 0, 0, d_a, d_b, out_ptr, n_vec4,
                             cfg.inner_int8, cfg.items_per_thread);
        }
      }
      return;
    }

    if (cfg.force_scalar_int8_io && !cfg.force_ilp2_int8 && needs_advanced_scalar) {
      launch_scalar_advanced(d_a, d_b, out_ptr, n_vec4, grid, d_scales, d_biases);
      return;
    }

    const bool can_split_scalar = cfg.split_interior_edge && cfg.force_scalar_int8_io &&
                                  !cfg.force_runtime_inner_loops && !cfg.force_ilp2_int8 &&
                                  !needs_advanced_scalar &&
                                  (cfg.inner_int8 == 8 || cfg.inner_int8 == 16 ||
                                   cfg.inner_int8 == 32 || cfg.inner_int8 == 64);
    if (can_split_scalar) {
      const int interior_vec = (n_vec4 / logical_threads) * logical_threads;
      if (interior_vec > 0) {
        const dim3 grid_interior(interior_vec / logical_threads);
        launch_scalar_unrolled_nobounds(d_a, d_b, out_ptr, grid_interior);
      }
      const int tail_vec = n_vec4 - interior_vec;
      if (tail_vec > 0) {
        const dim3 grid_tail((tail_vec + logical_threads - 1) / logical_threads);
        launch_scalar_unrolled_checked(d_a + interior_vec * 4, d_b + interior_vec * 4,
                                       out_ptr + interior_vec, tail_vec, grid_tail);
      }
      return;
    }

    if (cfg.force_scalar_int8_io && cfg.force_ilp2_int8) {
      launch_scalar_ilp2(d_a, d_b, out_ptr, n_vec4, grid);
      return;
    }
    if (!cfg.force_scalar_int8_io && cfg.force_isa_packed_int8_io) {
      launch_isa_packed(d_a, d_b, out_ptr, n_vec4, grid);
      return;
    }

    switch (cfg.inner_int8) {
      case 8:
        if (cfg.force_scalar_int8_io) {
          hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar<8>, grid, block, 0, 0, d_a, d_b,
                             out_ptr, n_vec4, cfg.items_per_thread);
        } else {
          hipLaunchKernelGGL(int8_dot4_kernel_unrolled<8>, grid, block, 0, 0, d_a, d_b, out_ptr,
                             n_vec4, cfg.items_per_thread);
        }
        break;
      case 16:
        if (cfg.force_scalar_int8_io) {
          hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar<16>, grid, block, 0, 0, d_a, d_b,
                             out_ptr, n_vec4, cfg.items_per_thread);
        } else {
          hipLaunchKernelGGL(int8_dot4_kernel_unrolled<16>, grid, block, 0, 0, d_a, d_b, out_ptr,
                             n_vec4, cfg.items_per_thread);
        }
        break;
      case 32:
        if (cfg.force_scalar_int8_io) {
          hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar<32>, grid, block, 0, 0, d_a, d_b,
                             out_ptr, n_vec4, cfg.items_per_thread);
        } else {
          hipLaunchKernelGGL(int8_dot4_kernel_unrolled<32>, grid, block, 0, 0, d_a, d_b, out_ptr,
                             n_vec4, cfg.items_per_thread);
        }
        break;
      case 64:
        if (cfg.force_scalar_int8_io) {
          hipLaunchKernelGGL(int8_dot4_kernel_unrolled_scalar<64>, grid, block, 0, 0, d_a, d_b,
                             out_ptr, n_vec4, cfg.items_per_thread);
        } else {
          hipLaunchKernelGGL(int8_dot4_kernel_unrolled<64>, grid, block, 0, 0, d_a, d_b, out_ptr,
                             n_vec4, cfg.items_per_thread);
        }
        break;
      default:
        if (cfg.force_scalar_int8_io) {
          hipLaunchKernelGGL(int8_dot4_kernel_runtime_scalar, grid, block, 0, 0, d_a, d_b, out_ptr,
                             n_vec4, cfg.inner_int8, cfg.items_per_thread);
        } else {
          hipLaunchKernelGGL(int8_dot4_kernel_runtime, grid, block, 0, 0, d_a, d_b, out_ptr,
                             n_vec4, cfg.inner_int8, cfg.items_per_thread);
        }
        break;
    }
  };

  auto launch_int8 = [&]() {
    if (cfg.force_two_pass) {
      launch_int8_core(d_tmp ? d_tmp : d_out);
      if (cfg.force_unfused_post) {
        hipLaunchKernelGGL(int8_post_kernel, post_grid, block, 0, 0, d_tmp ? d_tmp : d_out,
                           n_vec4);
      }
      hipLaunchKernelGGL(int8_adjacent_pass_kernel, post_grid, block, 0, 0, d_tmp ? d_tmp : d_out,
                         d_out, n_vec4);
      if (cfg.force_unfused_post) {
        hipLaunchKernelGGL(int8_post_kernel, post_grid, block, 0, 0, d_out, n_vec4);
      }
      return;
    }
    launch_int8_core(d_out);
    if (cfg.force_unfused_post) {
      hipLaunchKernelGGL(int8_post_kernel, post_grid, block, 0, 0, d_out, n_vec4);
    }
  };

  for (int i = 0; i < cfg.warmup_runs; ++i) {
    launch_int8();
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t ev_start, ev_stop;
  HIP_CHECK(hipEventCreate(&ev_start));
  HIP_CHECK(hipEventCreate(&ev_stop));

  std::vector<float> samples;
  samples.reserve(cfg.max_runs);
  auto t0 = std::chrono::steady_clock::now();

  for (int run = 0; run < cfg.max_runs; ++run) {
    HIP_CHECK(hipEventRecord(ev_start));
    for (int rep = 0; rep < cfg.reps_per_run; ++rep) {
      launch_int8();
    }
    HIP_CHECK(hipEventRecord(ev_stop));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    samples.push_back(elapsed_ms / static_cast<float>(cfg.reps_per_run));

    auto now = std::chrono::steady_clock::now();
    const double elapsed_s =
        std::chrono::duration_cast<std::chrono::duration<double>>(now - t0).count();
    if (samples.size() >= static_cast<size_t>(cfg.min_runs) && elapsed_s >= cfg.target_seconds) {
      break;
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  const double elapsed_s =
      std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

  int32_t checksum = 0;
  HIP_CHECK(hipMemcpy(&checksum, d_out, sizeof(int32_t), hipMemcpyDeviceToHost));

  HIP_CHECK(hipEventDestroy(ev_start));
  HIP_CHECK(hipEventDestroy(ev_stop));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_out));
  if (d_tmp) {
    HIP_CHECK(hipFree(d_tmp));
  }
  if (d_scales) {
    HIP_CHECK(hipFree(d_scales));
  }
  if (d_biases) {
    HIP_CHECK(hipFree(d_biases));
  }

  return {compute_stats(samples, elapsed_s), static_cast<double>(checksum)};
}

static std::pair<Stats, double> run_fp8(const Config& cfg) {
  const int n = cfg.elements;

  std::vector<float> h_a(n);
  std::vector<float> h_b(n);
  std::vector<uint8_t> h_a_q;
  std::vector<uint8_t> h_b_q;
  std::mt19937 rng(cfg.seed + 11);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (int i = 0; i < n; ++i) {
    h_a[i] = dist(rng);
    h_b[i] = dist(rng);
  }
  if (cfg.fp8_quantized_io) {
    h_a_q.resize(n);
    h_b_q.resize(n);
    for (int i = 0; i < n; ++i) {
      __hip_fp8_e4m3 qa = __hip_fp8_e4m3(h_a[i]);
      __hip_fp8_e4m3 qb = __hip_fp8_e4m3(h_b[i]);
      std::memcpy(&h_a_q[i], &qa, sizeof(uint8_t));
      std::memcpy(&h_b_q[i], &qb, sizeof(uint8_t));
    }
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_out = nullptr;
  float* d_tmp = nullptr;
  uint8_t* d_a_q = nullptr;
  uint8_t* d_b_q = nullptr;
  HIP_CHECK(hipMalloc(&d_a, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_b, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, n * sizeof(float)));
  if (cfg.force_two_pass) {
    HIP_CHECK(hipMalloc(&d_tmp, n * sizeof(float)));
  }
  if (cfg.fp8_quantized_io) {
    HIP_CHECK(hipMalloc(&d_a_q, n * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc(&d_b_q, n * sizeof(uint8_t)));
  }

  HIP_CHECK(hipMemcpy(d_a, h_a.data(), n * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), n * sizeof(float), hipMemcpyHostToDevice));
  if (cfg.fp8_quantized_io) {
    HIP_CHECK(hipMemcpy(d_a_q, h_a_q.data(), n * sizeof(uint8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b_q, h_b_q.data(), n * sizeof(uint8_t), hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipMemset(d_out, 0, n * sizeof(float)));
  if (d_tmp) {
    HIP_CHECK(hipMemset(d_tmp, 0, n * sizeof(float)));
  }

  const dim3 block(cfg.threads);
  const int logical_threads = cfg.threads * cfg.items_per_thread;
  const dim3 grid((n + logical_threads - 1) / logical_threads);
  const dim3 post_grid((n + cfg.threads - 1) / cfg.threads);

  auto launch_fp8_checked = [&](const float* a_ptr, const float* b_ptr, const uint8_t* a_q_ptr,
                                const uint8_t* b_q_ptr, float* out_ptr, int n_local,
                                dim3 grid_local) {
    switch (cfg.inner_fp8) {
      case 8:
        {
          auto kernel = fp8_fma_kernel_unrolled_general<8, true>;
          hipLaunchKernelGGL(kernel, grid_local, block, 0, 0, a_ptr, b_ptr, a_q_ptr, b_q_ptr,
                             out_ptr, n_local, cfg.items_per_thread, cfg.fp8_quantized_io,
                             cfg.force_per_iter_requant);
        }
        break;
      case 16:
        {
          auto kernel = fp8_fma_kernel_unrolled_general<16, true>;
          hipLaunchKernelGGL(kernel, grid_local, block, 0, 0, a_ptr, b_ptr, a_q_ptr, b_q_ptr,
                             out_ptr, n_local, cfg.items_per_thread, cfg.fp8_quantized_io,
                             cfg.force_per_iter_requant);
        }
        break;
      case 32:
        {
          auto kernel = fp8_fma_kernel_unrolled_general<32, true>;
          hipLaunchKernelGGL(kernel, grid_local, block, 0, 0, a_ptr, b_ptr, a_q_ptr, b_q_ptr,
                             out_ptr, n_local, cfg.items_per_thread, cfg.fp8_quantized_io,
                             cfg.force_per_iter_requant);
        }
        break;
      case 64:
        {
          auto kernel = fp8_fma_kernel_unrolled_general<64, true>;
          hipLaunchKernelGGL(kernel, grid_local, block, 0, 0, a_ptr, b_ptr, a_q_ptr, b_q_ptr,
                             out_ptr, n_local, cfg.items_per_thread, cfg.fp8_quantized_io,
                             cfg.force_per_iter_requant);
        }
        break;
      default:
        hipLaunchKernelGGL(fp8_fma_kernel_runtime, grid_local, block, 0, 0, a_ptr, b_ptr, out_ptr,
                           n_local, cfg.inner_fp8, cfg.items_per_thread);
        break;
    }
  };

  auto launch_fp8_nobounds = [&](const float* a_ptr, const float* b_ptr, const uint8_t* a_q_ptr,
                                 const uint8_t* b_q_ptr, float* out_ptr, dim3 grid_local) {
    switch (cfg.inner_fp8) {
      case 8:
        {
          auto kernel = fp8_fma_kernel_unrolled_general<8, false>;
          hipLaunchKernelGGL(kernel, grid_local, block, 0, 0, a_ptr, b_ptr, a_q_ptr, b_q_ptr,
                             out_ptr, 0, cfg.items_per_thread, cfg.fp8_quantized_io,
                             cfg.force_per_iter_requant);
        }
        break;
      case 16:
        {
          auto kernel = fp8_fma_kernel_unrolled_general<16, false>;
          hipLaunchKernelGGL(kernel, grid_local, block, 0, 0, a_ptr, b_ptr, a_q_ptr, b_q_ptr,
                             out_ptr, 0, cfg.items_per_thread, cfg.fp8_quantized_io,
                             cfg.force_per_iter_requant);
        }
        break;
      case 32:
        {
          auto kernel = fp8_fma_kernel_unrolled_general<32, false>;
          hipLaunchKernelGGL(kernel, grid_local, block, 0, 0, a_ptr, b_ptr, a_q_ptr, b_q_ptr,
                             out_ptr, 0, cfg.items_per_thread, cfg.fp8_quantized_io,
                             cfg.force_per_iter_requant);
        }
        break;
      case 64:
        {
          auto kernel = fp8_fma_kernel_unrolled_general<64, false>;
          hipLaunchKernelGGL(kernel, grid_local, block, 0, 0, a_ptr, b_ptr, a_q_ptr, b_q_ptr,
                             out_ptr, 0, cfg.items_per_thread, cfg.fp8_quantized_io,
                             cfg.force_per_iter_requant);
        }
        break;
      default:
        break;
    }
  };

  auto launch_fp8_core = [&](float* out_ptr) {
    if (cfg.force_runtime_inner_loops) {
      hipLaunchKernelGGL(fp8_fma_kernel_runtime, grid, block, 0, 0, d_a, d_b, out_ptr, n,
                         cfg.inner_fp8, cfg.items_per_thread);
      return;
    }

    const bool can_split = cfg.split_interior_edge &&
                           (cfg.inner_fp8 == 8 || cfg.inner_fp8 == 16 || cfg.inner_fp8 == 32 ||
                            cfg.inner_fp8 == 64);
    if (can_split) {
      const int interior = (n / logical_threads) * logical_threads;
      if (interior > 0) {
        const dim3 grid_interior(interior / logical_threads);
        launch_fp8_nobounds(d_a, d_b, d_a_q, d_b_q, out_ptr, grid_interior);
      }
      const int tail = n - interior;
      if (tail > 0) {
        const dim3 grid_tail((tail + logical_threads - 1) / logical_threads);
        launch_fp8_checked(d_a + interior, d_b + interior, d_a_q ? d_a_q + interior : nullptr,
                           d_b_q ? d_b_q + interior : nullptr, out_ptr + interior, tail, grid_tail);
      }
      return;
    }

    launch_fp8_checked(d_a, d_b, d_a_q, d_b_q, out_ptr, n, grid);
  };

  auto launch_fp8 = [&]() {
    if (cfg.force_two_pass) {
      launch_fp8_core(d_tmp ? d_tmp : d_out);
      if (cfg.force_unfused_post) {
        hipLaunchKernelGGL(fp8_post_kernel, post_grid, block, 0, 0, d_tmp ? d_tmp : d_out, n);
      }
      hipLaunchKernelGGL(fp8_adjacent_pass_kernel, post_grid, block, 0, 0, d_tmp ? d_tmp : d_out,
                         d_out, n);
      if (cfg.force_unfused_post) {
        hipLaunchKernelGGL(fp8_post_kernel, post_grid, block, 0, 0, d_out, n);
      }
      return;
    }
    launch_fp8_core(d_out);
    if (cfg.force_unfused_post) {
      hipLaunchKernelGGL(fp8_post_kernel, post_grid, block, 0, 0, d_out, n);
    }
  };

  for (int i = 0; i < cfg.warmup_runs; ++i) {
    launch_fp8();
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t ev_start, ev_stop;
  HIP_CHECK(hipEventCreate(&ev_start));
  HIP_CHECK(hipEventCreate(&ev_stop));

  std::vector<float> samples;
  samples.reserve(cfg.max_runs);
  auto t0 = std::chrono::steady_clock::now();

  for (int run = 0; run < cfg.max_runs; ++run) {
    HIP_CHECK(hipEventRecord(ev_start));
    for (int rep = 0; rep < cfg.reps_per_run; ++rep) {
      launch_fp8();
    }
    HIP_CHECK(hipEventRecord(ev_stop));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    samples.push_back(elapsed_ms / static_cast<float>(cfg.reps_per_run));

    auto now = std::chrono::steady_clock::now();
    const double elapsed_s =
        std::chrono::duration_cast<std::chrono::duration<double>>(now - t0).count();
    if (samples.size() >= static_cast<size_t>(cfg.min_runs) && elapsed_s >= cfg.target_seconds) {
      break;
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  const double elapsed_s =
      std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

  float checksum = 0.0f;
  HIP_CHECK(hipMemcpy(&checksum, d_out, sizeof(float), hipMemcpyDeviceToHost));

  HIP_CHECK(hipEventDestroy(ev_start));
  HIP_CHECK(hipEventDestroy(ev_stop));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_out));
  if (d_tmp) {
    HIP_CHECK(hipFree(d_tmp));
  }
  if (d_a_q) {
    HIP_CHECK(hipFree(d_a_q));
  }
  if (d_b_q) {
    HIP_CHECK(hipFree(d_b_q));
  }

  return {compute_stats(samples, elapsed_s), static_cast<double>(checksum)};
}

static void print_result(const char* mode, const Stats& s, double checksum) {
  std::cout << std::fixed << std::setprecision(6)
            << "RESULT"
            << " mode=" << mode
            << " runs=" << s.runs
            << " elapsed_s=" << s.elapsed_seconds
            << " mean_ms=" << s.mean_ms
            << " stddev_ms=" << s.stddev_ms
            << " median_ms=" << s.median_ms
            << " p95_ms=" << s.p95_ms
            << " min_ms=" << s.min_ms
            << " max_ms=" << s.max_ms
            << " cv_pct=" << s.cv_pct
            << " checksum=" << checksum << '\n';
}

int main(int argc, char** argv) {
  Config cfg = parse_args(argc, argv);

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, device));

  std::cout << "INFO mode=" << cfg.mode << " device=" << device << " name=\"" << props.name
            << "\""
            << " gcnArchName=" << props.gcnArchName
            << " warpSize=" << props.warpSize
            << " elements=" << cfg.elements
            << " threads=" << cfg.threads
            << " items_per_thread=" << cfg.items_per_thread
            << " force_runtime_inner_loops=" << (cfg.force_runtime_inner_loops ? 1 : 0)
            << " force_scalar_int8_io=" << (cfg.force_scalar_int8_io ? 1 : 0)
            << " force_isa_packed_int8_io=" << (cfg.force_isa_packed_int8_io ? 1 : 0)
            << " force_ilp2_int8=" << (cfg.force_ilp2_int8 ? 1 : 0)
            << " force_convlike_int8=" << (cfg.force_convlike_int8 ? 1 : 0)
            << " force_inloop_scale_bias=" << (cfg.force_inloop_scale_bias ? 1 : 0)
            << " force_per_iter_requant=" << (cfg.force_per_iter_requant ? 1 : 0)
            << " split_interior_edge=" << (cfg.split_interior_edge ? 1 : 0)
            << " lds_stage_input=" << (cfg.lds_stage_input ? 1 : 0)
            << " lds_stage_weight=" << (cfg.lds_stage_weight ? 1 : 0)
            << " lds_padding=" << (cfg.lds_padding ? 1 : 0)
            << " lds_double_buffer=" << (cfg.lds_double_buffer ? 1 : 0)
            << " force_unfused_post=" << (cfg.force_unfused_post ? 1 : 0)
            << " force_two_pass=" << (cfg.force_two_pass ? 1 : 0)
            << " force_mixed_int8_path=" << (cfg.force_mixed_int8_path ? 1 : 0)
            << " fp8_quantized_io=" << (cfg.fp8_quantized_io ? 1 : 0)
            << " warmup_runs=" << cfg.warmup_runs
            << " min_runs=" << cfg.min_runs
            << " max_runs=" << cfg.max_runs
            << " reps_per_run=" << cfg.reps_per_run
            << " target_seconds=" << cfg.target_seconds
            << " inner_int8=" << cfg.inner_int8
            << " inner_fp8=" << cfg.inner_fp8 << '\n';

  if (cfg.mode == "int8" || cfg.mode == "both") {
    auto [stats, checksum] = run_int8(cfg);
    print_result("int8", stats, checksum);
  }

  if (cfg.mode == "fp8" || cfg.mode == "both") {
    auto [stats, checksum] = run_fp8(cfg);
    print_result("fp8", stats, checksum);
  }

  return 0;
}
