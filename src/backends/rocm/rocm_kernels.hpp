#pragma once

// Device kernels owned by the rocm backend (defined in
// rocm_kernels_*.cpp, launched through the registry in
// rocm_backend.cpp). No vendor type crosses the backend boundary;
// only this directory includes HIP headers.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace tessera::backends::rocm {

// IEEE binary16 (little-endian) -> fp32, exact (device port of the
// core Fp16ToFloat in src/core/numerics/quant.cpp). Inline so every
// kernel TU carries its own copy (no cross-TU device calls).
inline __host__ __device__ std::uint16_t FloatToFp16Dev(float value) {
  std::uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  int exponent = static_cast<int>((bits >> 23) & 0xFFu) - 127;
  std::uint32_t mantissa = bits & 0x7FFFFFu;
  if (exponent > 15) {
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }
  if (exponent < -24) {
    return static_cast<std::uint16_t>(sign);
  }
  if (exponent < -14) {
    mantissa |= 0x800000u;
    const std::uint32_t shift = static_cast<std::uint32_t>(-exponent - 14);
    return static_cast<std::uint16_t>(sign | (mantissa >> (shift + 13)));
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exponent + 15) << 10) |
      (mantissa >> 13));
}

inline __host__ __device__ float Fp16ToFloatDev(std::uint16_t half) {
  const std::uint32_t sign = half >> 15;
  const std::uint32_t exp = (half >> 10) & 0x1F;
  const std::uint32_t mant = half & 0x3FF;
  float value;
  if (exp == 0) {
    value = static_cast<float>(mant) * 0x1p-24f;
  } else if (exp == 31) {
    value = mant != 0
        ? std::numeric_limits<float>::quiet_NaN()
        : std::numeric_limits<float>::infinity();
  } else {
    value = (1.0f + static_cast<float>(mant) / 1024.0f) *
            std::ldexp(1.0f, static_cast<int>(exp) - 15);
  }
  return sign != 0 ? -value : value;
}

// The 6-bit scale/min pair of sub-block j from the 12-byte packing
// (device port of the core GetScaleMin; the vulkan kernel uses the
// same formula). Inline, same reason as above.
inline __host__ __device__ void GetScaleMinDev(
    std::size_t j, const unsigned char* scales, std::uint8_t* scale,
    std::uint8_t* min) {
  if (j < 4) {
    *scale = scales[j] & 63;
    *min = scales[j + 4] & 63;
  } else {
    *scale = (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4);
    *min = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
  }
}

// OCP MX E8M0 scale byte to fp32, 2^(scale - 127) (device port of the
// core E8M0ToFloat). The fp32 exponent field equals `scale`, so the bits
// are `scale << 23`; that is cheaper than the software `ldexp` the GPU
// would otherwise call. Only `scale == 0` differs (2^-127 is subnormal),
// so that one value keeps ldexp. Inline so every kernel TU carries its
// own copy.
inline __host__ __device__ float E8M0ToFloatDev(std::uint8_t scale) {
  if (scale == 0) {
    return static_cast<float>(std::ldexp(1.0, -127));
  }
  const int bits = static_cast<int>(scale) << 23;
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

// OCP MX E2M1 nibble values (port of the core F4E2M1ToFloat). A table
// lookup avoids the branch and the ldexp in the hot loop. Inline so
// every kernel TU carries its own copy.
inline __device__ const float kE2M1Dev[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
__global__ void FillKernel(int* out, unsigned long long value,
                           unsigned long long count);
__global__ void RopeKernel(float* data, unsigned long long rows,
                           unsigned long long heads,
                           unsigned long long head_dim,
                           unsigned long long rope_dim,
                           unsigned long long pos_base,
                           unsigned long long theta_bits);
__global__ void AttentionKernel(const float* q, const float* k,
                                const float* v, float* out,
                                unsigned long long m, unsigned long long n,
                                unsigned long long heads,
                                unsigned long long kv_heads,
                                unsigned long long head_dim,
                                unsigned long long q_base,
                                unsigned long long window,
                                unsigned long long kv_f16,
                                unsigned long long causal);
__global__ void GemmFp8BlockKernel(const float* a, const unsigned char* w,
                                   const float* s, float* c,
                                   unsigned long long m,
                                   unsigned long long n,
                                   unsigned long long k);
__global__ void GemmBf16Kernel(const float* a, const unsigned short* w,
                               float* c, unsigned long long m,
                               unsigned long long n,
                               unsigned long long k);
__global__ void GemmF32Kernel(const float* a, const float* w, float* c,
                              unsigned long long m,
                              unsigned long long n,
                              unsigned long long k);
__global__ void GemmFp8Kernel(const float* a, const unsigned char* w,
                              const float* s, float* c,
                              unsigned long long m, unsigned long long n,
                              unsigned long long k);
__global__ void GemmMxFp4Kernel(const float* a, const unsigned char* w,
                                float* c, unsigned long long m,
                                unsigned long long n, unsigned long long k);
__global__ void GemmMxFp4BatchedKernel(const float* a, const unsigned char* w,
                                       float* c, unsigned long long m,
                                       unsigned long long n,
                                       unsigned long long k);
__global__ void GemmQ4KKernel(const float* a, const unsigned char* w,
                              float* c, unsigned long long m,
                              unsigned long long n, unsigned long long k);
__global__ void GemmQ4KBatchedKernel(const float* a, const unsigned char* w,
                                     float* c, unsigned long long m,
                                     unsigned long long n,
                                     unsigned long long k);
__global__ void GemmQ5KKernel(const float* a, const unsigned char* w,
                              float* c, unsigned long long m,
                              unsigned long long n, unsigned long long k);
__global__ void GemmQ6KKernel(const float* a, const unsigned char* w,
                              float* c, unsigned long long m,
                              unsigned long long n, unsigned long long k);
__global__ void GemmQ3KKernel(const float* a, const unsigned char* w,
                              float* c, unsigned long long m,
                              unsigned long long n, unsigned long long k);
__global__ void GemmIq4NlKernel(const float* a, const unsigned char* w,
                                float* c, unsigned long long m,
                                unsigned long long n, unsigned long long k);
__global__ void GemmIq4XsKernel(const float* a, const unsigned char* w,
                                float* c, unsigned long long m,
                                unsigned long long n, unsigned long long k);
__global__ void GemmIq3SKernel(const float* a, const unsigned char* w,
                                float* c, unsigned long long m,
                                unsigned long long n, unsigned long long k);
__global__ void GemmQ80Kernel(const float* a, const unsigned char* w,
                              float* c, unsigned long long m,
                              unsigned long long n, unsigned long long k);
__global__ void RmsnormKernel(const float* x, const float* w, float* y,
                              unsigned long long rows, unsigned long long cols,
                              unsigned long long eps_bits);
__global__ void L2NormKernel(const float* x, float* y,
                             unsigned long long rows, unsigned long long cols,
                             unsigned long long eps_bits,
                             unsigned long long scale_bits);
__global__ void RmsnormGatedKernel(const float* x, const float* w,
                                   const float* gate, float* y,
                                   unsigned long long rows,
                                   unsigned long long cols,
                                   unsigned long long eps_bits);
__global__ void SigmoidGateKernel(const float* a, const float* g, float* o,
                                  unsigned long long n);
__global__ void Conv1dKernel(const float* x, const float* w, float* y,
                             unsigned long long channels,
                             unsigned long long length,
                             unsigned long long width);
__global__ void DeltaStepKernel(float* s, const float* k, const float* v,
                                const float* q, float* o,
                                unsigned long long dk, unsigned long long dv,
                                unsigned long long alpha_bits,
                                unsigned long long beta_bits);
__global__ void MropeKernel(float* data, const unsigned long long* pos,
                            unsigned long long rows, unsigned long long heads,
                            unsigned long long head_dim,
                            unsigned long long rope_dim,
                            unsigned long long theta_bits,
                            unsigned long long sec_t, unsigned long long sec_h,
                            unsigned long long sec_w);
__global__ void QGateSplitKernel(const float* fused, float* q, float* gate,
                                 unsigned long long heads,
                                 unsigned long long head_dim,
                                 unsigned long long rows);
__global__ void AddKernel(const float* a, const float* b, float* o,
                          unsigned long long n);
__global__ void RepeatHeadsKernel(const float* in, float* out,
                                  unsigned long long num_v_heads,
                                  unsigned long long head_k_dim,
                                  unsigned long long factor);
__global__ void QuantizeQ8Kernel(const float* in, unsigned int* packed,
                                 float* scale, unsigned long long rows,
                                 unsigned long long cols);
__global__ void AttentionQ8Kernel(const float* q, const unsigned char* k,
                                  const unsigned char* v, const float* ks,
                                  const float* vs, float* out,
                                  unsigned long long m, unsigned long long n,
                                  unsigned long long heads,
                                  unsigned long long kv_heads,
                                  unsigned long long head_dim,
                                  unsigned long long q_base,
                                  unsigned long long window);
__global__ void SpatialMergeKernel(const float* x, float* y,
                                   unsigned long long embed,
                                   unsigned long long grid_w,
                                   unsigned long long grid_h,
                                   unsigned long long merge);
__global__ void BiasAddKernel(const float* x, const float* b, float* y,
                              unsigned long long rows,
                              unsigned long long cols);
__global__ void ImagePatchifyKernel(const float* image, const float* mean,
                                    const float* sd, float* out_values,
                                    unsigned long long h,
                                    unsigned long long w,
                                    unsigned long long patch);
__global__ void LayerNormKernel(const float* x, const float* w, const float* b,
                                float* y, unsigned long long rows,
                                unsigned long long cols, float eps);
__global__ void GeluKernel(const float* x, float* y, unsigned long long n);
__global__ void QuantizeQ4Kernel(const float* in, unsigned int* packed,
                                 float* scale, unsigned long long rows,
                                 unsigned long long cols);
__global__ void AttentionQ4Kernel(const float* q, const unsigned int* k,
                                  const unsigned int* v, const float* ks,
                                  const float* vs, float* out,
                                  unsigned long long m, unsigned long long n,
                                  unsigned long long heads,
                                  unsigned long long kv_heads,
                                  unsigned long long head_dim,
                                  unsigned long long q_base,
                                  unsigned long long window);
__global__ void CastF32F16Kernel(const float* in, unsigned int* out,
                                 unsigned long long n);
__global__ void ConcatFeaturesKernel(const float* in, float* out,
                                     unsigned long long n,
                                     unsigned long long rows,
                                     unsigned long long features);
__global__ void SelectorEdgeScoreKernel(const float* predecessor_codebook,
                                        const float* successor_codebook,
                                        const float* hidden,
                                        const int* candidate_ids,
                                        const int* anchor_ids,
                                        const float* unary,
                                        float* out_scores,
                                        unsigned long long batch,
                                        unsigned long long seq,
                                        unsigned long long top_k,
                                        unsigned long long rank);
__global__ void DflashConvKernel(const float* x, const float* delta,
                                 const float* base, float* y,
                                 unsigned long long rows,
                                 unsigned long long channels,
                                 unsigned long long taps,
                                 unsigned long long group_size,
                                 unsigned long long block_size,
                                 unsigned long long delta_row_stride,
                                 unsigned long long delta_offset);
__global__ void Conv1dStepKernel(const float* x, const float* w, float* y,
                                 unsigned long long channels,
                                 unsigned long long width);
// Current-step causal depthwise conv with an on-device history, the SiLU
// and the q/k/v split (the Vulkan "conv1d_state" contract).
__global__ void Conv1dStateKernel(const float* qkv, const float* w,
                                  float* hist, float* q, float* k, float* v,
                                  unsigned long long conv_dim,
                                  unsigned long long width,
                                  unsigned long long key_dim,
                                  unsigned long long qkv_offset);
__global__ void SsmGateKernel(const float* a_log, const float* dt,
                              const float* alpha_raw, const float* beta_raw,
                              float* alpha, float* beta,
                              unsigned long long heads,
                              unsigned long long rows);
__global__ void DeltaStepHeadsKernel(
    float* s, const float* k, const float* v, const float* q, float* o,
    const float* alpha, const float* beta, unsigned long long heads,
    unsigned long long dk, unsigned long long dv);
__global__ void SiluMulKernel(const float* g, const float* u, float* o,
                              unsigned long long n);
// Built-in "embedding_f32"/"embedding_bf16"/"embedding_q4k": buffer 0
// token ids (u32, rows), buffer 1 the embedding table (vocab x cols),
// buffer 2 the fp32 output (rows x cols); scalars are rows, cols, vocab.
__global__ void EmbeddingF32Kernel(const unsigned int* ids, const float* w,
                                   float* out, unsigned long long rows,
                                   unsigned long long cols,
                                   unsigned long long vocab);
__global__ void EmbeddingBf16Kernel(const unsigned int* ids,
                                    const unsigned short* w, float* out,
                                    unsigned long long rows,
                                    unsigned long long cols,
                                    unsigned long long vocab);
__global__ void EmbeddingQ4KKernel(const unsigned int* ids,
                                   const unsigned char* w, float* out,
                                   unsigned long long rows,
                                   unsigned long long cols,
                                   unsigned long long vocab);

}  // namespace tessera::backends::rocm
