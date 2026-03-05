#include <hip/hip_runtime.h>

typedef float float4_t __attribute__((ext_vector_type(4)));

__global__ void probe_mfma_fp8(unsigned long long a, unsigned long long b, float4_t* out) {
  float4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
  acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, acc, 0, 0, 0);
  if (threadIdx.x == 0) {
    out[0] = acc;
  }
}

int main() {
  return 0;
}
