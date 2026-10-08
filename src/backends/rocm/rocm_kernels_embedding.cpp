#include "backends/rocm/rocm_kernels.hpp"

#include <cstdint>
#include <cstring>

// Token-embedding gather kernels. One thread per output element; the
// dispatch is ceil(rows * cols / 256). Buffer 0 is the token ids
// (u32, rows), buffer 1 the embedding table (vocab x cols), buffer 2
// the fp32 output (rows x cols); scalars are rows, cols, vocab. The
// table layout matches the core GatherEmbedding (row t is `t * cols`
// elements in). An out-of-range id writes a zero row.
namespace tessera::backends::rocm {

// Built-in "embedding_f32": fp32 embedding table.
__global__ void EmbeddingF32Kernel(const unsigned int* ids, const float* w,
                                   float* out, unsigned long long rows,
                                   unsigned long long cols,
                                   unsigned long long vocab) {
  const unsigned long long g =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= rows * cols) {
    return;
  }
  const unsigned long long row = g / cols;
  const unsigned long long i = g % cols;
  const unsigned long long id = ids[row];
  out[g] = id < vocab ? w[id * cols + i] : 0.0f;
}

// Built-in "embedding_bf16": bf16 embedding table (the MXFP4 target).
// bf16 shares the fp32 exponent, so the conversion is the high half.
__global__ void EmbeddingBf16Kernel(const unsigned int* ids,
                                    const unsigned short* w, float* out,
                                    unsigned long long rows,
                                    unsigned long long cols,
                                    unsigned long long vocab) {
  const unsigned long long g =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= rows * cols) {
    return;
  }
  const unsigned long long row = g / cols;
  const unsigned long long i = g % cols;
  const unsigned long long id = ids[row];
  float value = 0.0f;
  if (id < vocab) {
    const unsigned int bits =
        static_cast<unsigned int>(w[id * cols + i]) << 16;
    std::memcpy(&value, &bits, sizeof(value));
  }
  out[g] = value;
}

// Built-in "embedding_q4k": Q4_K embedding table (the GGUF target).
// Element `e` of a 256-element block reads sub-block `e / 64`, half
// `(e % 64) / 32` and lane `e % 32`; sub-block 2*g is the low nibble
// and 2*g+1 the high, matching the gemm_q4k dequant.
__global__ void EmbeddingQ4KKernel(const unsigned int* ids,
                                   const unsigned char* w, float* out,
                                   unsigned long long rows,
                                   unsigned long long cols,
                                   unsigned long long vocab) {
  const unsigned long long g =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= rows * cols) {
    return;
  }
  const unsigned long long row = g / cols;
  const unsigned long long i = g % cols;
  const unsigned long long id = ids[row];
  if (id >= vocab) {
    out[g] = 0.0f;
    return;
  }
  const unsigned long long blocks = cols / 256;
  const unsigned long long b = i / 256;
  const unsigned long long e = i % 256;
  const unsigned char* base = w + (id * blocks + b) * 144;
  std::uint16_t d_bits = 0;
  std::uint16_t dm_bits = 0;
  std::memcpy(&d_bits, base, 2);
  std::memcpy(&dm_bits, base + 2, 2);
  const float d = Fp16ToFloatDev(d_bits);
  const float dm = Fp16ToFloatDev(dm_bits);
  const unsigned long long grp = e / 64;
  const unsigned long long half = (e % 64) / 32;
  const unsigned long long l = e % 32;
  std::uint8_t sc = 0;
  std::uint8_t mn = 0;
  GetScaleMinDev(2 * grp + half, base + 4, &sc, &mn);
  const std::uint8_t q = base[16 + grp * 32 + l];
  const std::uint8_t nib = half != 0 ? (q >> 4) : (q & 15);
  out[g] = d * static_cast<float>(sc) * static_cast<float>(nib) -
           dm * static_cast<float>(mn);
}

}  // namespace tessera::backends::rocm
