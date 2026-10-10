#include "backends/rocm/rocm_kernels.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace tessera::backends::rocm {

// Built-in "fill": writes scalar 0 to every int32 element (scalar 1 is
// the element count). Contract: CheckBuiltInArgs (tessera API).
__global__ void FillKernel(int* out, unsigned long long value,
                          unsigned long long count) {
  unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = static_cast<int>(value);
  }
}

// Built-in "rope": NeoX-style rotary embedding over rows x heads x
// head_dim fp32 in place (buffer 0); scalars are rows, heads,
// head_dim, rope_dim, pos_base, theta fp32 bits. One thread per pair.
__global__ void RopeKernel(float* data, unsigned long long rows,
                           unsigned long long heads,
                           unsigned long long head_dim,
                           unsigned long long rope_dim,
                           unsigned long long pos_base,
                           unsigned long long theta_bits) {
  const unsigned long long pairs = rope_dim / 2;
  const unsigned long long t =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= rows * heads * pairs) {
    return;
  }
  const unsigned long long row = t / (heads * pairs);
  const unsigned long long rem = t % (heads * pairs);
  const unsigned long long head = rem / pairs;
  const unsigned long long j = rem % pairs;
  float theta = 0.0f;
  static_assert(sizeof(theta) == 4);
  std::uint32_t bits = static_cast<std::uint32_t>(theta_bits);
  std::memcpy(&theta, &bits, 4);
  const float pos = static_cast<float>(pos_base + row);
  const float angle =
      pos * powf(theta, -2.0f * static_cast<float>(j) /
                            static_cast<float>(rope_dim));
  const float c = cosf(angle);
  const float s = sinf(angle);
  float* base = data + (row * heads + head) * head_dim;
  const float x1 = base[j];
  const float x2 = base[j + pairs];
  base[j] = x1 * c - x2 * s;
  base[j + pairs] = x1 * s + x2 * c;
}

// Built-in "attention": causal grouped-query attention, scale
// 1/sqrt(head_dim), fp32 sequential accumulation. Buffers are q, k, v
// and out; scalars are m, n, heads, kv_heads, head_dim, q_base. One
// thread per output element (two score passes, no scratch buffer).
__global__ void AttentionKernel(const float* q, const float* k, const float* v,
                                float* out, unsigned long long m,
                                unsigned long long n,
                                unsigned long long heads,
                                unsigned long long kv_heads,
                                unsigned long long head_dim,
                                unsigned long long q_base,
                                unsigned long long window,
                                 unsigned long long kv_f16,
                                 unsigned long long causal) {
  // One workgroup per (query row, head); the query/key dot product is
  // computed once per key, so the cost is O(n * head_dim) per query/head.
  constexpr unsigned long long kTile = 256;
  __shared__ float q_s[256];
  __shared__ float sc[256];
  __shared__ float wt[256];
  // Tile reduction broadcast: thread 0 computes the max/weights once.
  __shared__ float tile_bc[3];
  const std::uint16_t* kh = reinterpret_cast<const std::uint16_t*>(k);
  const std::uint16_t* vh = reinterpret_cast<const std::uint16_t*>(v);
  const auto kat = [&](unsigned long long idx) {
    return kv_f16 ? Fp16ToFloatDev(kh[idx]) : k[idx];
  };
  const auto vat = [&](unsigned long long idx) {
    return kv_f16 ? Fp16ToFloatDev(vh[idx]) : v[idx];
  };
  const unsigned long long g = blockIdx.x;
  if (g >= m * heads) {
    return;
  }
  const unsigned long long e = threadIdx.x;
  const unsigned long long i = g / heads;
  const unsigned long long h = g % heads;
  const unsigned long long kv = h / (heads / kv_heads);
  const unsigned long long pos = q_base + i;
  const unsigned long long last =
      causal != 0 ? (pos >= n ? n - 1 : pos) : n - 1;
  unsigned long long start = 0;
  if (causal != 0) {
    start = (window != 0 && pos + 1 > window) ? pos + 1 - window : 0;
    if (start > last) {
      start = last;
    }
  }
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const unsigned long long qb = (i * heads + h) * head_dim;
  if (e < head_dim) {
    q_s[e] = q[qb + e];
  }
  __syncthreads();
  float run_max = -1e30f;
  float run_sum = 0.0f;
  float acc = 0.0f;
  unsigned long long tile = start;
  while (true) {
    const unsigned long long tile_last =
        (tile + kTile - 1 < last) ? tile + kTile - 1 : last;
    float s = -1e30f;
    if (tile + e <= last) {
      const unsigned long long j = tile + e;
      const unsigned long long kb = (j * kv_heads + kv) * head_dim;
      // Four independent partial sums keep the FMA pipeline full;
      // one chain stalls on its own latency (see the tiled GEMMs).
      float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
      unsigned long long d = 0;
      for (; d + 3 < head_dim; d += 4) {
        dot0 = fmaf(q_s[d], kat(kb + d), dot0);
        dot1 = fmaf(q_s[d + 1], kat(kb + d + 1), dot1);
        dot2 = fmaf(q_s[d + 2], kat(kb + d + 2), dot2);
        dot3 = fmaf(q_s[d + 3], kat(kb + d + 3), dot3);
      }
      float dot = (dot0 + dot1) + (dot2 + dot3);
      for (; d < head_dim; ++d) {
        dot = fmaf(q_s[d], kat(kb + d), dot);
      }
      s = dot * scale;
    }
    sc[e] = s;
    __syncthreads();
    // One thread reduces the tile so a long context pays one scan,
    // not one scan per thread. Same order as before, same numerics.
    if (e == 0) {
      float tmax = -1e30f;
      for (unsigned long long t = 0; t < kTile; ++t) {
        tmax = fmaxf(tmax, sc[t]);
      }
      const float m_new = fmaxf(run_max, tmax);
      const float corr = expf(run_max - m_new);
      float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
      for (unsigned long long t = 0; t < kTile; t += 4) {
        const float w0 = expf(sc[t] - m_new);
        const float w1 = expf(sc[t + 1] - m_new);
        const float w2 = expf(sc[t + 2] - m_new);
        const float w3 = expf(sc[t + 3] - m_new);
        wt[t] = w0;
        wt[t + 1] = w1;
        wt[t + 2] = w2;
        wt[t + 3] = w3;
        l0 += w0;
        l1 += w1;
        l2 += w2;
        l3 += w3;
      }
      tile_bc[0] = m_new;
      tile_bc[1] = corr;
      tile_bc[2] = (l0 + l1) + (l2 + l3);
    }
    __syncthreads();
    const float m_new = tile_bc[0];
    const float corr = tile_bc[1];
    const float l = tile_bc[2];
    // Four independent accumulators, as in the score dot above.
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a = 0.0f;
    // Only lanes below head_dim have an output element; the others read
    // past the value row if they run this loop, so guard it.
    if (e < head_dim) {
      const unsigned long long count =
          (tile + kTile - 1 <= last) ? kTile : (last - tile + 1);
      unsigned long long t = 0;
      for (; t + 3 < count; t += 4) {
        const unsigned long long b0 = (tile + t) * kv_heads + kv;
        const unsigned long long b1 = (tile + t + 1) * kv_heads + kv;
        const unsigned long long b2 = (tile + t + 2) * kv_heads + kv;
        const unsigned long long b3 = (tile + t + 3) * kv_heads + kv;
        a0 = fmaf(wt[t], vat((b0 * head_dim) + e), a0);
        a1 = fmaf(wt[t + 1], vat((b1 * head_dim) + e), a1);
        a2 = fmaf(wt[t + 2], vat((b2 * head_dim) + e), a2);
        a3 = fmaf(wt[t + 3], vat((b3 * head_dim) + e), a3);
      }
      a = (a0 + a1) + (a2 + a3);
      for (; t < count; ++t) {
        const unsigned long long vb = (tile + t) * kv_heads + kv;
        a = fmaf(wt[t], vat(vb * head_dim + e), a);
      }
    }
    acc = fmaf(corr, acc, a);
    run_sum = fmaf(corr, run_sum, l);
    run_max = m_new;
    __syncthreads();
    if (tile_last >= last) {
      break;
    }
    tile += kTile;
  }
  if (e < head_dim) {
    out[qb + e] = acc / run_sum;
  }
}

// OCP FP8 E4M3 byte to fp32 (device port of the core Fp8E4M3ToFloat).
__host__ __device__ float Fp8E4M3ToFloatDev(std::uint8_t bits) {
  const std::uint32_t sign = bits >> 7;
  const std::uint32_t exp = (bits >> 3) & 0xF;
  const std::uint32_t mant = bits & 0x7;
  float value;
  if (exp == 0) {
    value = static_cast<float>(mant) * 0x1p-9f;
  } else if (exp == 15) {
    if (mant == 7) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    value = (1.0f + static_cast<float>(mant) / 8.0f) * 256.0f;
  } else {
    value = (1.0f + static_cast<float>(mant) / 8.0f) *
            std::ldexp(1.0f, static_cast<int>(exp) - 7);
  }
  return sign != 0 ? -value : value;
}

// fp32 -> OCP FP8 E4M3 byte, round to nearest even (device port of the
// core Fp32ToFp8E4M3Bits). NaN and values above 448 saturate to the
// all-ones byte.
__device__ std::uint8_t Fp8E4M3FromFloatDev(float value) {
  unsigned int bits;
  memcpy(&bits, &value, sizeof(bits));
  const unsigned int sign = bits >> 31;
  if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
    return static_cast<std::uint8_t>((sign << 7) | 0x7Fu);
  }
  const float a = value < 0.0f ? -value : value;
  if (a == 0.0f) {
    return static_cast<std::uint8_t>(sign << 7);
  }
  if (a > 448.0f) {
    return static_cast<std::uint8_t>((sign << 7) | 0x7Fu);
  }
  int e = 0;
  const float frac = frexpf(a, &e);
  int field = e + 6;
  long long mant;
  if (field < 1) {
    mant = llrintf(a * 512.0f);
    if (mant > 7) {
      mant = 7;
    }
    return static_cast<std::uint8_t>((sign << 7) | static_cast<unsigned int>(mant));
  }
  mant = llrintf((frac * 2.0f - 1.0f) * 8.0f);
  if (mant == 8) {
    mant = 0;
    ++field;
  }
  if (field > 15) {
    field = 15;
  }
  if (field == 15 && mant > 6) {
    mant = 6;
  }
  return static_cast<std::uint8_t>(
      (sign << 7) | (static_cast<unsigned int>(field) << 3) |
      static_cast<unsigned int>(mant));
}

// Built-in "quantize_fp8": symmetric per-row E4M3 quantize-dequantize. One
// workgroup per row so a one-row call (the DFlash2 context hidden) uses the
// whole block; the dispatch is `rows` workgroups of 256.
__global__ void QuantizeFp8Kernel(float* data, float* scale,
                                  unsigned long long rows,
                                  unsigned long long cols) {
  __shared__ float red[256];
  const unsigned long long r =
      static_cast<unsigned long long>(blockIdx.x);
  if (r >= rows) {
    return;
  }
  const unsigned long long tid = threadIdx.x;
  const unsigned long long base = r * cols;
  float amax = 0.0f;
  for (unsigned long long c = tid; c < cols; c += blockDim.x) {
    amax = fmaxf(amax, fabsf(data[base + c]));
  }
  red[tid] = amax;
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s) {
      red[tid] = fmaxf(red[tid], red[tid + s]);
    }
  }
  __syncthreads();
  float sc = red[0] / 448.0f;
  const float min_s = 1.0f / (448.0f * 512.0f);
  if (sc < min_s) {
    sc = min_s;
  }
  if (tid == 0) {
    scale[r] = sc;
  }
  for (unsigned long long c = tid; c < cols; c += blockDim.x) {
    const float q = fminf(fmaxf(data[base + c] / sc, -448.0f), 448.0f);
    const std::uint8_t bits = Fp8E4M3FromFloatDev(q);
    data[base + c] = Fp8E4M3ToFloatDev(bits) * sc;
  }
}

// E8M0ToFloatDev and kE2M1Dev (the MXFP4 codec) now live in
// rocm_kernels.hpp so the batched MXFP4 kernel in this translation unit
// and any other share one definition.

// bf16 -> fp32 on the device: shift the 16 bits into the high half.
__device__ float Bf16ToFloatDev(unsigned short bits) {
  unsigned int wide = static_cast<unsigned int>(bits) << 16;
  float value;
  memcpy(&value, &wide, sizeof(value));
  return value;
}

// fp32 -> bf16 round-to-nearest-even, the activation precision the bf16
// tensor-core GEMM stages.
__device__ unsigned short Bf16FromFloatDev(float value) {
  unsigned int bits;
  memcpy(&bits, &value, sizeof(bits));
  const unsigned int rounding = (bits >> 16) & 1u;
  bits += 0x7FFFu + rounding;
  return static_cast<unsigned short>(bits >> 16);
}

// Built-in "gemm_bf16": C = A x W^T with bf16 weights and fp32
// sequential accumulation.
__global__ void GemmBf16Kernel(const float* a, const unsigned short* w,
                               float* c, unsigned long long m,
                               unsigned long long n, unsigned long long k) {
  const unsigned long long warp =
      (static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
       threadIdx.x) /
      32;
  const unsigned long long lane = threadIdx.x & 31u;
  if (warp >= m * n) {
    return;
  }
  const unsigned long long row_a = warp / n;
  const unsigned long long row_w = warp % n;
  const float* a_row = a + row_a * k;
  const unsigned short* w_row = w + row_w * k;
  float acc = 0.0f;
  for (unsigned long long t = lane; t < k; t += 32) {
    acc = fmaf(a_row[t], Bf16ToFloatDev(w_row[t]), acc);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down(acc, offset);
  }
  if (lane == 0) {
    c[warp] = acc;
  }
}

// Built-in "round_bf16": one thread per element; round each fp32 value to
// bfloat16 (round to nearest even), keeping it in fp32. This is the bf16
// activation rounding the served target's fused epilogues apply.
__global__ void RoundBf16Kernel(float* x, unsigned long long n) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  unsigned int bits;
  memcpy(&bits, &x[i], sizeof(bits));
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  bits &= 0xFFFF0000u;
  memcpy(&x[i], &bits, sizeof(bits));
}

// Built-in "gemm_f32": C = A x W^T with fp32 reconstruction. One 32-lane
// warp per output element; lane l sums t = l, l+32, ..., so consecutive
// lanes read consecutive weights (coalesced) instead of striding by k, then
// the lane partials reduce with a shuffle. Grid: ceil(m*n / 8) workgroups.
__global__ void GemmF32Kernel(const float* a, const float* w, float* c,
                              unsigned long long m, unsigned long long n,
                              unsigned long long k) {
  const unsigned long long warp =
      (static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
       threadIdx.x) /
      32;
  const unsigned long long lane = threadIdx.x & 31u;
  if (warp >= m * n) {
    return;
  }
  const unsigned long long row_a = warp / n;
  const unsigned long long row_w = warp % n;
  const float* a_row = a + row_a * k;
  const float* w_row = w + row_w * k;
  float acc = 0.0f;
  for (unsigned long long t = lane; t < k; t += 32) {
    acc = fmaf(a_row[t], w_row[t], acc);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down(acc, offset);
  }
  if (lane == 0) {
    c[warp] = acc;
  }
}

// Built-in "gemm_f32_batched": C = A x W^T for fp32 weights with a 32-row x
// 8-column output tile, so a batch (the DFlash2 draft block) reads each
// weight row once instead of once per row. Four independent partial sums
// per thread keep the FMA chain shallow. Grid: ceil(m / 32) * ceil(n / 8).
__global__ void GemmF32BatchedKernel(const float* a, const float* w, float* c,
                                     unsigned long long m,
                                     unsigned long long n,
                                     unsigned long long k) {
  const unsigned long long tid = static_cast<unsigned long long>(threadIdx.x);
  const unsigned long long col_tiles = (n + 7) / 8;
  const unsigned long long wg = static_cast<unsigned long long>(blockIdx.x);
  const unsigned long long row_a = (wg / col_tiles) * 32 + tid / 8;
  const unsigned long long row_w = (wg % col_tiles) * 8 + tid % 8;
  if (row_a >= m || row_w >= n) {
    return;
  }
  const float* ar = a + row_a * k;
  const float* wr = w + row_w * k;
  float p0 = 0.0f;
  float p1 = 0.0f;
  float p2 = 0.0f;
  float p3 = 0.0f;
  unsigned long long t = 0;
  for (; t + 4 <= k; t += 4) {
    p0 = fmaf(ar[t], wr[t], p0);
    p1 = fmaf(ar[t + 1], wr[t + 1], p1);
    p2 = fmaf(ar[t + 2], wr[t + 2], p2);
    p3 = fmaf(ar[t + 3], wr[t + 3], p3);
  }
  for (; t < k; ++t) {
    p0 = fmaf(ar[t], wr[t], p0);
  }
  c[row_a * n + row_w] = (p0 + p1) + (p2 + p3);
}

// Built-in "gemm_bf16_batched": as gemm_f32_batched for bf16 weights.
__global__ void GemmBf16BatchedKernel(const float* a, const unsigned short* w,
                                      float* c, unsigned long long m,
                                      unsigned long long n,
                                      unsigned long long k) {
  const unsigned long long tid = static_cast<unsigned long long>(threadIdx.x);
  const unsigned long long col_tiles = (n + 7) / 8;
  const unsigned long long wg = static_cast<unsigned long long>(blockIdx.x);
  const unsigned long long row_a = (wg / col_tiles) * 32 + tid / 8;
  const unsigned long long row_w = (wg % col_tiles) * 8 + tid % 8;
  if (row_a >= m || row_w >= n) {
    return;
  }
  const float* ar = a + row_a * k;
  const unsigned short* wr = w + row_w * k;
  float p0 = 0.0f;
  float p1 = 0.0f;
  float p2 = 0.0f;
  float p3 = 0.0f;
  unsigned long long t = 0;
  for (; t + 4 <= k; t += 4) {
    p0 = fmaf(ar[t], Bf16ToFloatDev(wr[t]), p0);
    p1 = fmaf(ar[t + 1], Bf16ToFloatDev(wr[t + 1]), p1);
    p2 = fmaf(ar[t + 2], Bf16ToFloatDev(wr[t + 2]), p2);
    p3 = fmaf(ar[t + 3], Bf16ToFloatDev(wr[t + 3]), p3);
  }
  for (; t < k; ++t) {
    p0 = fmaf(ar[t], Bf16ToFloatDev(wr[t]), p0);
  }
  c[row_a * n + row_w] = (p0 + p1) + (p2 + p3);
}

// Built-in "gemm_fp8": C = A x (diag(s) x W)^T with fp32 sequential
// accumulation; W holds FP8 E4M3 bytes, s one fp32 scale per row.
__global__ void GemmFp8Kernel(const float* a, const unsigned char* w,
                              const float* s, float* c,
                              unsigned long long m, unsigned long long n,
                              unsigned long long k) {
  unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= m * n) {
    return;
  }
  const unsigned long long row_a = idx / n;
  const unsigned long long row_w = idx % n;
  const float scale = s[row_w];
  float acc = 0.0f;
  for (unsigned long long t = 0; t < k; ++t) {
    acc = fmaf(a[row_a * k + t],
               scale * Fp8E4M3ToFloatDev(w[row_w * k + t]), acc);
  }
  c[idx] = acc;
}

// Built-in "gemm_fp8_block": one fp32 scale per 128 x 128 weight block
// (the DFlash2 draft quantization).
__global__ void GemmFp8BlockKernel(const float* a, const unsigned char* w,
                                   const float* s, float* c,
                                   unsigned long long m,
                                   unsigned long long n,
                                   unsigned long long k) {
  unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= m * n) {
    return;
  }
  const unsigned long long row_a = idx / n;
  const unsigned long long row_w = idx % n;
  const unsigned long long k_blocks = k / 128ULL;
  const float* s_row = &s[(row_w / 128ULL) * k_blocks];
  float acc = 0.0f;
  for (unsigned long long t = 0; t < k; ++t) {
    const float wgt =
        s_row[t / 128ULL] *
        Fp8E4M3ToFloatDev(w[row_w * k + t]);
    acc = fmaf(a[row_a * k + t], wgt, acc);
  }
  c[idx] = acc;
}

// Built-in "gemm_mxfp4": C = A x W'^T with fp32 sequential
// accumulation; W' dequantizes MXFP4 nibbles (low nibble first) with
// one E8M0 scale byte per 32 elements. `w` packs the n x k/2 blob bytes
// followed by the n x k/32 scale bytes (the scale starts at n*k/2). k is
// a multiple of 32.
__global__ void GemmMxFp4Kernel(const float* a, const unsigned char* w,
                                float* c, unsigned long long m,
                                unsigned long long n, unsigned long long k) {
  // One 32-lane warp per output element; lane l sums blocks l, l+32, ...,
  // so the activation and weight reads are coalesced across the warp
  // (consecutive lanes read consecutive blocks), then the lane partials
  // are reduced with a shuffle. Grid: ceil(m*n / 8) workgroups of 256.
  const unsigned long long warp =
      (static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
       threadIdx.x) /
      32;
  const unsigned long long lane = threadIdx.x & 31u;
  const unsigned long long total = m * n;
  if (warp >= total) {
    return;
  }
  const unsigned char* s = w + (n * k) / 2;
  const unsigned long long row_a = warp / n;
  const unsigned long long row_w = warp % n;
  const unsigned long long blocks = k / 32;
  const float* a_row = a + row_a * k;
  // A 32-bit word holds eight nibbles. Row starts (row_w * k) and block
  // starts (b * 32) are multiples of 32 elements, so the byte offset is a
  // multiple of 16 and every word load is aligned.
  const unsigned int* words = reinterpret_cast<const unsigned int*>(w);
  float acc = 0.0f;
  for (unsigned long long b = lane; b < blocks; b += 32) {
    const float scale = E8M0ToFloatDev(s[row_w * blocks + b]);
    const unsigned long long base = b * 32;
    // Element index / 8 gives the word index.
    const unsigned long long word_base = (row_w * k + base) / 8;
    for (unsigned long long q = 0; q < 4; ++q) {
      unsigned int bits = words[word_base + q];
      for (unsigned long long j = 0; j < 8; ++j) {
        acc = fmaf(a_row[base + q * 8 + j], scale * kE2M1Dev[bits & 15u], acc);
        bits >>= 4;
      }
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down(acc, offset);
  }
  if (lane == 0) {
    c[warp] = acc;
  }
}

// Built-in "gemm_mxfp4_rows": the small-batch MXFP4 GEMV. One warp owns one
// weight column (row_w) and accumulates every one of the m activation rows,
// so the weight block is decoded and read once per column instead of once
// per (row, column) as the single-row "gemm_mxfp4" does. This is the shape
// the speculative verifier hits (m about 2 to 7), where reading the weight
// matrix once instead of m times is the win. Grid: ceil(n / 8) workgroups
// of 256. m must be 1..16.
constexpr int kGemmRowsMax = 16;

__global__ void GemmMxFp4RowsKernel(const float* a, const unsigned char* w,
                                    float* c, unsigned long long m,
                                    unsigned long long n, unsigned long long k) {
  const unsigned long long warp =
      (static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
       threadIdx.x) /
      32;
  const unsigned long long lane = threadIdx.x & 31u;
  if (warp >= n) {
    return;
  }
  const unsigned long long row_w = warp;
  const unsigned long long rows = m < kGemmRowsMax ? m : kGemmRowsMax;
  const unsigned char* s = w + (n * k) / 2;
  const unsigned long long blocks = k / 32;
  const unsigned int* words = reinterpret_cast<const unsigned int*>(w);
  float acc[kGemmRowsMax];
#pragma unroll
  for (int r = 0; r < kGemmRowsMax; ++r) {
    acc[r] = 0.0f;
  }
  for (unsigned long long b = lane; b < blocks; b += 32) {
    const float scale = E8M0ToFloatDev(s[row_w * blocks + b]);
    const unsigned long long base = b * 32;
    const unsigned long long word_base = (row_w * k + base) / 8;
    float wv[32];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      unsigned int bits = words[word_base + q];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        wv[q * 8 + j] = scale * kE2M1Dev[bits & 15u];
        bits >>= 4;
      }
    }
#pragma unroll
    for (int r = 0; r < kGemmRowsMax; ++r) {
      if (static_cast<unsigned long long>(r) >= rows) {
        break;
      }
      const float* a_row = a + static_cast<unsigned long long>(r) * k + base;
#pragma unroll
      for (int e = 0; e < 32; ++e) {
        acc[r] = fmaf(a_row[e], wv[e], acc[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kGemmRowsMax; ++r) {
    if (static_cast<unsigned long long>(r) >= rows) {
      break;
    }
    float v = acc[r];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_down(v, offset);
    }
    if (lane == 0) {
      c[static_cast<unsigned long long>(r) * n + row_w] = v;
    }
  }
}

// Built-in "gemm_mxfp4_batched": tiled C = A x dequant(W)^T for a batch.
// One workgroup handles 8 activation rows x one weight column; the 32
// MXFP4 elements of a block (and its E8M0 scale) are dequantized once
// into shared memory and reused across the 8 rows. Buffers and scalars
// match gemm_mxfp4 (m, n, k, k a multiple of 32); grid_x is
// ceil(m / 8) * n, block_x is 256. The dot runs elements 0..31 in
// order, the gemm_mxfp4 accumulation order. Threads 8..255 still help
// dequantize but write no output.
__global__ void GemmMxFp4BatchedKernel(const float* a, const unsigned char* w,
                                       float* c, unsigned long long m,
                                       unsigned long long n,
                                       unsigned long long k) {
  // Output tile: 32 rows x 8 columns per 256-thread workgroup. Threads in a
  // warp share a row (eight each) and a column (four each), so the
  // activation and weight reads coalesce or broadcast in cache instead of
  // streaming one cache line per output element. Grid: ceil(m / 32) *
  // ceil(n / 8) workgroups of 256 (see ProjectTiledDevice).
  const unsigned long long tid = static_cast<unsigned long long>(threadIdx.x);
  const unsigned long long col_tiles = (n + 7) / 8;
  const unsigned long long wg = static_cast<unsigned long long>(blockIdx.x);
  const unsigned long long row_a = (wg / col_tiles) * 32 + tid / 8;
  const unsigned long long row_w = (wg % col_tiles) * 8 + tid % 8;
  if (row_a >= m || row_w >= n) {
    return;
  }
  const unsigned long long blocks = k / 32;
  const unsigned char* s = w + (n * k) / 2;
  const float* ar = a + row_a * k;
  const unsigned int* wr =
      reinterpret_cast<const unsigned int*>(w) + (row_w * k) / 8;
  // Four independent partial sums (one per 8-element word of the block) so
  // the FMA chain is four deep instead of one; a single accumulator left
  // the kernel latency-bound (the prefill's dominant cost). Eight partial
  // sums regressed: the extra live registers cost more occupancy than the
  // added ILP bought.
  float p0 = 0.0f;
  float p1 = 0.0f;
  float p2 = 0.0f;
  float p3 = 0.0f;
  for (unsigned long long b = 0; b < blocks; ++b) {
    const float scale = E8M0ToFloatDev(s[row_w * blocks + b]);
    const float* ab = ar + b * 32;
    const unsigned int* wb = wr + b * 4;
    unsigned int b0 = wb[0];
    unsigned int b1 = wb[1];
    unsigned int b2 = wb[2];
    unsigned int b3 = wb[3];
    for (unsigned long long j = 0; j < 8; ++j) {
      p0 = fmaf(ab[j], scale * kE2M1Dev[b0 & 15u], p0);
      p1 = fmaf(ab[8 + j], scale * kE2M1Dev[b1 & 15u], p1);
      p2 = fmaf(ab[16 + j], scale * kE2M1Dev[b2 & 15u], p2);
      p3 = fmaf(ab[24 + j], scale * kE2M1Dev[b3 & 15u], p3);
      b0 >>= 4;
      b1 >>= 4;
      b2 >>= 4;
      b3 >>= 4;
    }
  }
  c[row_a * n + row_w] = (p0 + p1) + (p2 + p3);
}

// fp8 tensor-core WMMA: C[m,n] fp32 = A_fp8[m,k] . dequant(W_mxfp4)[n,k]^T
// with per-token As[m]. 4 waves per workgroup, each a 16x16 output tile; the
// A and W tiles (kWmmaDbk deep) are staged in padded shared memory and the
// fragments read from LDS (raw global fragment reads are ~10x slower). A is
// e4m3. W is MXFP4 (n*k/2 packed nibbles then n*k/32 E8M0 scales); each
// nibble is upconverted to e4m3 with the block exponent folded in relative to
// the row max (the kMag table, see FoldMxFp4ToFp8), so DRAM stays 0.5 B. The
// served reference's small-m path (radiance_mxfp4_fp8_gemm_decode). Buffers:
// 0 A (m x k fp8), 1 W (packed MXFP4), 2 As (m f32), 3 Wref (n bytes), 4 C
// (m x n f32); scalars m, n, k. Grid ceil(n/64), block 128.
typedef int WmmaFp8A __attribute__((ext_vector_type(2)));
typedef float WmmaFp8C __attribute__((ext_vector_type(8)));
typedef unsigned int WmmaU4 __attribute__((ext_vector_type(4)));
typedef unsigned int WmmaU2 __attribute__((ext_vector_type(2)));

constexpr int kWmmaDbk = 128;
constexpr int kWmmaPad = 16;
constexpr int kWmmaAstr = kWmmaDbk + kWmmaPad;
constexpr int kWmmaDwn = 4;

__device__ __constant__ unsigned char kMagDev[16][8] = {
    {0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c},
    {0x00, 0x28, 0x30, 0x34, 0x38, 0x3c, 0x40, 0x44},
    {0x00, 0x20, 0x28, 0x2c, 0x30, 0x34, 0x38, 0x3c},
    {0x00, 0x18, 0x20, 0x24, 0x28, 0x2c, 0x30, 0x34},
    {0x00, 0x10, 0x18, 0x1c, 0x20, 0x24, 0x28, 0x2c},
    {0x00, 0x08, 0x10, 0x14, 0x18, 0x1c, 0x20, 0x24},
    {0x00, 0x04, 0x08, 0x0c, 0x10, 0x14, 0x18, 0x1c},
    {0x00, 0x02, 0x04, 0x06, 0x08, 0x0c, 0x10, 0x14},
    {0x00, 0x01, 0x02, 0x03, 0x04, 0x06, 0x08, 0x0c},
    {0x00, 0x00, 0x01, 0x01, 0x02, 0x03, 0x04, 0x06},
    {0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x02, 0x03},
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01},
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01},
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
};

// Built-in "mxfp4_row_ref": per-row max of the E8M0 block scales of a packed
// MXFP4 weight (n*k/2 nibbles then n*k/32 scales), the Wref the fp8 GEMM
// folds against. One thread per row. Buffers: 0 W (packed), 1 Wref (n bytes);
// scalars n, k. Grid ceil(n/256), block 256.
__global__ void MxFp4RowRefKernel(const unsigned char* w, unsigned char* wref,
                                  unsigned long long n, unsigned long long k) {
  const unsigned long long r =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= n) {
    return;
  }
  const unsigned char* scales = w + (n * k) / 2;
  const unsigned long long blocks = k / 32;
  unsigned char ref = 0;
  for (unsigned long long b = 0; b < blocks; ++b) {
    const unsigned char s = scales[r * blocks + b];
    ref = s > ref ? s : ref;
  }
  wref[r] = ref;
}

__global__ void GemmMxFp4WmmaKernel(const unsigned char* a,
                                    const unsigned char* w,
                                    const float* as,
                                    const unsigned char* wref, float* c,
                                    unsigned long long m, unsigned long long n,
                                    unsigned long long k,
                                    unsigned long long split,
                                    unsigned long long wperm) {
  __shared__ unsigned char sA[16 * kWmmaAstr];
  __shared__ unsigned char sW[kWmmaDwn * 16 * kWmmaAstr];
  const int tid = threadIdx.x;
  const int wave = tid >> 5, lane = tid & 31;
  const unsigned long long n0 =
      static_cast<unsigned long long>(blockIdx.x) * (kWmmaDwn * 16ull) +
      static_cast<unsigned long long>(wave) * 16ull;
  const unsigned char* scales = w + (n * k) / 2;
  const unsigned long long blocks = k / 32;
  const int acol8 = (lane >> 4) * 8;
  const int wrow8 = (lane >> 4) * 8, wcol = lane & 15;
  WmmaFp8C acc;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    acc[e] = 0.0f;
  }
  // Split-K: blockIdx.y selects the K chunk range; split == 1 keeps the
  // whole K in one block (the original behaviour).
  const unsigned long long chunks = (k + kWmmaDbk - 1) / kWmmaDbk;
  const unsigned long long per = (chunks + split - 1) / split;
  const unsigned long long c0 =
      static_cast<unsigned long long>(blockIdx.y) * per;
  const unsigned long long c1 = (c0 + per) < chunks ? (c0 + per) : chunks;
  for (unsigned long long ci = c0; ci < c1; ++ci) {
    const unsigned long long k0 = ci * kWmmaDbk;
    // A tile: 16 rows x DBK bytes, vectorized stores.
    for (int u = tid; u < 16 * kWmmaDbk / 16; u += kWmmaDwn * 32) {
      const int pos = u * 16, r = pos / kWmmaDbk, col = pos % kWmmaDbk;
      const unsigned long long ar =
          static_cast<unsigned long long>(r) < m ? r : m - 1;
      *reinterpret_cast<WmmaU4*>(&sA[r * kWmmaAstr + col]) =
          *reinterpret_cast<const WmmaU4*>(&a[ar * k + k0 + col]);
    }
    // W tile: DBK/16 groups of 16 nibbles per row, folded to 16 fp8 bytes.
    // Each wave stages its own 16 columns (its 32 lanes cover the groups).
    if (n0 < n) {
      // With wperm the weight is fragment order [n-tile][k-step][half][row][4];
      // a row's 16 nibbles are two 4-byte halves 64 bytes apart, and
      // consecutive lanes take consecutive rows so the read coalesces.
      const unsigned int* wp = reinterpret_cast<const unsigned int*>(w);
      const unsigned long long ks_total = k / 16;
      const unsigned long long nt = n0 / 16;
      // Each lane stages four 8-byte weight groups. Load all four (and their
      // scale/fold index) first, then fold and store, so the cold weight
      // reads are in flight together instead of one-at-a-time behind the
      // fold (which left the verify latency-bound at ~193 GB/s).
      WmmaU2 wvs[4];
      int d0s[4];
      int dst[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int g = lane + u * 32;
        int r, cg;
        if (wperm != 0) {
          r = g % 16;
          cg = g / 16;
        } else {
          r = g / (kWmmaDbk / 16);
          cg = g % (kWmmaDbk / 16);
        }
        const int nr = static_cast<int>(n0) + r;
        const int nrc = nr < static_cast<int>(n) ? nr : static_cast<int>(n) - 1;
        const unsigned long long kt0 =
            k0 + static_cast<unsigned long long>(cg * 16);
        if (wperm != 0) {
          const unsigned long long base =
              ((nt * ks_total + kt0 / 16) * 2ull) * 16ull;
          reinterpret_cast<unsigned int*>(&wvs[u])[0] = wp[base + r];
          reinterpret_cast<unsigned int*>(&wvs[u])[1] = wp[base + 16 + r];
        } else {
          wvs[u] = *reinterpret_cast<const WmmaU2*>(
              &w[(static_cast<unsigned long long>(nrc) * k + kt0) / 2]);
        }
        // One E8M0 scale covers this 16-wide group (a scale spans 32
        // elements, the group starts on a 16 boundary).
        const int wref_n = static_cast<int>(wref[nrc]);
        const int sbyte = wref_n - static_cast<int>(
            scales[static_cast<unsigned long long>(nrc) * blocks +
                   (k0 + cg * 16) / 32]);
        d0s[u] = sbyte < 0 ? 0 : (sbyte > 15 ? 15 : sbyte);
        dst[u] = (wave * 16 + r) * kWmmaAstr + cg * 16;
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        // Fold the 16 nibbles to 16 fp8 bytes with three v_perm per eight
        // nibbles (the served reference's trick) instead of a per-element
        // kMag load. d0's row of kMagDev holds the eight magnitude bytes;
        // the sign bit rides in each nibble's bit 3.
        const unsigned int t0 =
            *reinterpret_cast<const unsigned int*>(&kMagDev[d0s[u]][0]);
        const unsigned int t1 =
            *reinterpret_cast<const unsigned int*>(&kMagDev[d0s[u]][4]);
        unsigned int outw[4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const unsigned int w32 =
              reinterpret_cast<const unsigned int*>(&wvs[u])[h];
          const unsigned int ev = w32 & 0x0F0F0F0Fu;
          const unsigned int od = (w32 >> 4) & 0x0F0F0F0Fu;
          const unsigned int be =
              __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) |
              ((ev & 0x08080808u) << 4);
          const unsigned int bo =
              __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) |
              ((od & 0x08080808u) << 4);
          outw[2 * h] = __builtin_amdgcn_perm(bo, be, 0x05010400u);
          outw[2 * h + 1] = __builtin_amdgcn_perm(bo, be, 0x07030602u);
        }
        *reinterpret_cast<WmmaU4*>(&sW[dst[u]]) =
            *reinterpret_cast<WmmaU4*>(outw);
      }
    }
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < kWmmaDbk / 16; ++ks) {
      WmmaFp8A af, wf;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        reinterpret_cast<unsigned char*>(&af)[j] =
            sA[(lane & 15) * kWmmaAstr + ks * 16 + acol8 + j];
        reinterpret_cast<unsigned char*>(&wf)[j] =
            sW[(wave * 16 + (lane & 15)) * kWmmaAstr + ks * 16 + wrow8 + j];
      }
      acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af, wf, acc);
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const unsigned long long r = static_cast<unsigned long long>(wrow8 + j);
    const unsigned long long nn = n0 + static_cast<unsigned long long>(wcol);
    if (r < m && nn < n) {
      // Split-K writes each split's partial to its own region of c; the
      // reduce kernel sums the regions into the output.
      const unsigned long long base =
          static_cast<unsigned long long>(blockIdx.y) * (m * n);
      c[base + r * n + nn] =
          acc[j] * as[r] * __int_as_float(static_cast<int>(wref[nn]) << 23);
    }
  }
}

// Sums the split partials of GemmMxFp4WmmaKernel: c[i] = sum_p part[p*total+i].
// One thread per output element; buffers 0 part, 1 c; scalars total, split.
__global__ void GemmMxFp4WmmaReduceKernel(const float* part, float* c,
                                          unsigned long long total,
                                          unsigned long long split) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) {
    return;
  }
  float sum = 0.0f;
  for (unsigned long long p = 0; p < split; ++p) {
    sum += part[p * total + i];
  }
  c[i] = sum;
}
// Built-in "gemm_bf16_wmma": bf16 tensor-core GEMM for the small-m batched
// bf16 projections (the DFlash2 drafter and the bf16 shared head). A is fp32
// (rounded to bf16 while it is staged), W is bf16. Reads each weight column
// once, coalesced through shared memory, unlike the per-output
// gemm_bf16_batched. Buffers 0 A, 1 W, 2 C; scalars m, n, k. Grid ceil(n/64),
// block 128; m is 1..16.
typedef short WmmaBf16A __attribute__((ext_vector_type(8)));
constexpr int kWmmaBstr = kWmmaDbk + kWmmaPad;

__global__ void GemmBf16WmmaKernel(const float* a, const unsigned short* w,
                                   float* c, unsigned long long m,
                                   unsigned long long n, unsigned long long k) {
  __shared__ unsigned short sA[16 * kWmmaBstr];
  __shared__ unsigned short sW[kWmmaDwn * 16 * kWmmaBstr];
  const int tid = threadIdx.x;
  const int wave = tid >> 5, lane = tid & 31;
  const unsigned long long n0 = blockIdx.x * (kWmmaDwn * 16ull) + wave * 16ull;
  const int acol8 = (lane >> 4) * 8;
  WmmaFp8C acc;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    acc[e] = 0.0f;
  }
  for (unsigned long long k0 = 0; k0 < k; k0 += kWmmaDbk) {
    // A tile: 16 rows x DBK, fp32 rounded to bf16.
    for (int u = tid; u < 16 * kWmmaDbk / 8; u += kWmmaDwn * 32) {
      const int pos = u * 8, r = pos / kWmmaDbk, col = pos % kWmmaDbk;
      const unsigned long long ar =
          static_cast<unsigned long long>(r) < m ? r : m - 1;
      const float* src = &a[ar * k + k0 + col];
      unsigned short* dst = &sA[r * kWmmaBstr + col];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        dst[e] = Bf16FromFloatDev(src[e]);
      }
    }
    // W tile: the wave's 16 columns, DBK bf16 each, already bf16 in DRAM.
    if (n0 < n) {
      for (int g = lane; g < 16 * (kWmmaDbk / 8); g += 32) {
        const int r = g / (kWmmaDbk / 8), cg = g % (kWmmaDbk / 8);
        const int nr = static_cast<int>(n0) + r;
        const int nrc = nr < static_cast<int>(n) ? nr : static_cast<int>(n) - 1;
        *reinterpret_cast<WmmaU4*>(&sW[(wave * 16 + r) * kWmmaBstr + cg * 8]) =
            *reinterpret_cast<const WmmaU4*>(
                &w[static_cast<unsigned long long>(nrc) * k + k0 + cg * 8]);
      }
    }
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < kWmmaDbk / 16; ++ks) {
      WmmaBf16A af, wf;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        af[j] = sA[(lane & 15) * kWmmaBstr + ks * 16 + acol8 + j];
        wf[j] =
            sW[(wave * 16 + (lane & 15)) * kWmmaBstr + ks * 16 + acol8 + j];
      }
      acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(af, wf, acc);
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const unsigned long long r = static_cast<unsigned long long>(acol8 + j);
    const unsigned long long nn = n0 + static_cast<unsigned long long>(lane & 15);
    if (r < m && nn < n) {
      c[r * n + nn] = acc[j];
    }
  }
}
// Built-in "rmsnorm": row-wise RMS norm over rows x cols fp32. One block
// per row; the row sum reduces across the block so a single-row decode
// step does not serialize the row on one thread.
__global__ void RmsnormKernel(const float* x, const float* w, float* y,
                              unsigned long long rows, unsigned long long cols,
                              unsigned long long eps_bits) {
  __shared__ float partial[1024];
  const unsigned long long r = blockIdx.x;
  if (r >= rows) {
    return;
  }
  const unsigned int tid = threadIdx.x;
  float eps = 0.0f;
  static_assert(sizeof(eps) == 4);
  std::uint32_t bits = static_cast<std::uint32_t>(eps_bits);
  std::memcpy(&eps, &bits, 4);
  float sum = 0.0f;
  for (unsigned long long c = tid; c < cols; c += blockDim.x) {
    sum = fmaf(x[r * cols + c], x[r * cols + c], sum);
  }
  partial[tid] = sum;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      partial[tid] += partial[tid + s];
    }
    __syncthreads();
  }
  const float mean = partial[0] / static_cast<float>(cols);
  const float gain = 1.0f / sqrtf(mean + eps);
  for (unsigned long long c = tid; c < cols; c += blockDim.x) {
    y[r * cols + c] = x[r * cols + c] * gain * w[c];
  }
}

// Built-in "sigmoid_gate": elementwise out = a * sigmoid(g) over n
// fp32. One thread per element.
__global__ void SigmoidGateKernel(const float* a, const float* g, float* o,
                                  unsigned long long n) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  o[i] = a[i] / (1.0f + expf(-g[i]));
}

// Built-in "l2norm": row-wise L2 normalization over rows x cols fp32. One
// workgroup per row (shared-memory sum reduction), dispatch `rows`
// workgroups of 256.
__global__ void L2NormKernel(const float* x, float* y,
                             unsigned long long rows, unsigned long long cols,
                             unsigned long long eps_bits,
                             unsigned long long scale_bits) {
  __shared__ float red[1024];
  const unsigned long long r = static_cast<unsigned long long>(blockIdx.x);
  if (r >= rows) {
    return;
  }
  const unsigned long long tid = threadIdx.x;
  const unsigned long long base = r * cols;
  float eps = 0.0f;
  float scale = 0.0f;
  static_assert(sizeof(eps) == 4);
  std::uint32_t eps_bits32 = static_cast<std::uint32_t>(eps_bits);
  std::memcpy(&eps, &eps_bits32, 4);
  std::uint32_t scale_bits32 = static_cast<std::uint32_t>(scale_bits);
  std::memcpy(&scale, &scale_bits32, 4);
  float sum = 0.0f;
  for (unsigned long long c = tid; c < cols; c += blockDim.x) {
    sum = fmaf(x[base + c], x[base + c], sum);
  }
  red[tid] = sum;
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s) {
      red[tid] += red[tid + s];
    }
  }
  __syncthreads();
  const float gain = scale / sqrtf(red[0] + eps);
  for (unsigned long long c = tid; c < cols; c += blockDim.x) {
    y[base + c] = x[base + c] * gain;
  }
}

// Built-in "rmsnorm_gated": row-wise RMS norm scaled by a SiLU gate
// over rows x cols fp32. One thread per row with sequential
// accumulation.
__global__ void RmsnormGatedKernel(const float* x, const float* w,
                                   const float* gate, float* y,
                                   unsigned long long rows,
                                   unsigned long long cols,
                                   unsigned long long eps_bits,
                                   unsigned long long gbase,
                                   unsigned long long ybase) {
  __shared__ float red[256];
  const unsigned long long r = static_cast<unsigned long long>(blockIdx.x);
  if (r >= rows) {
    return;
  }
  const unsigned long long tid = threadIdx.x;
  const unsigned long long base = r * cols;
  float eps = 0.0f;
  static_assert(sizeof(eps) == 4);
  std::uint32_t bits = static_cast<std::uint32_t>(eps_bits);
  std::memcpy(&eps, &bits, 4);
  float sum = 0.0f;
  for (unsigned long long c = tid; c < cols; c += blockDim.x) {
    sum = fmaf(x[base + c], x[base + c], sum);
  }
  red[tid] = sum;
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s) {
      red[tid] += red[tid + s];
    }
  }
  __syncthreads();
  const float mean = red[0] / static_cast<float>(cols);
  const float gain = 1.0f / sqrtf(mean + eps);
  for (unsigned long long c = tid; c < cols; c += blockDim.x) {
    const float g = gate[gbase + base + c];
    const float silu = g / (1.0f + expf(-g));
    y[ybase + base + c] = x[base + c] * gain * w[c] * silu;
  }
}
// Built-in "conv1d": causal depthwise convolution over channels x
// length fp32. One thread per output element, sequential accumulation.
__global__ void Conv1dKernel(const float* x, const float* w, float* y,
                             unsigned long long channels,
                             unsigned long long length,
                             unsigned long long width) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= channels * length) {
    return;
  }
  const unsigned long long c = i / length;
  const unsigned long long t = i % length;
  float acc = 0.0f;
  for (unsigned long long k = 0; k < width && k <= t; ++k) {
    acc = fmaf(w[c * width + k], x[c * length + t - k], acc);
  }
  y[i] = acc;
}
// Built-in "delta_step": one gated delta-rule recurrent step over a
// dk x dv state updated in place. One thread per state column with
// sequential accumulation, so columns never race.
__global__ void DeltaStepKernel(float* s, const float* k, const float* v,
                                const float* q, float* o,
                                unsigned long long dk, unsigned long long dv,
                                unsigned long long alpha_bits,
                                unsigned long long beta_bits) {
  const unsigned long long d =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (d >= dv) {
    return;
  }
  float alpha = 0.0f;
  float beta = 0.0f;
  static_assert(sizeof(alpha) == 4);
  std::uint32_t alpha_raw = static_cast<std::uint32_t>(alpha_bits);
  std::uint32_t beta_raw = static_cast<std::uint32_t>(beta_bits);
  std::memcpy(&alpha, &alpha_raw, 4);
  std::memcpy(&beta, &beta_raw, 4);
  float read = 0.0f;
  for (unsigned long long j = 0; j < dk; ++j) {
    read = fmaf(s[j * dv + d], k[j], read);
  }
  float out = 0.0f;
  for (unsigned long long j = 0; j < dk; ++j) {
    const float updated = alpha * (s[j * dv + d] - beta * k[j] * read) +
                          beta * v[d] * k[j];
    s[j * dv + d] = updated;
    out = fmaf(updated, q[j], out);
  }
  o[d] = out;
}
// Built-in "mrope": multimodal rotary embedding over rows x heads x
// head_dim fp32 in place; pos holds one (t, h, w) u64 triple per row.
// One thread per rotated pair with the global pair index.
__global__ void MropeKernel(float* data, const unsigned long long* pos,
                            unsigned long long rows, unsigned long long heads,
                            unsigned long long head_dim,
                            unsigned long long rope_dim,
                            unsigned long long theta_bits,
                            unsigned long long sec_t, unsigned long long sec_h,
                            unsigned long long sec_w) {
  const unsigned long long pairs = rope_dim / 2;
  const unsigned long long t =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= rows * heads * pairs) {
    return;
  }
  (void)sec_w;  // pairs past temporal+height reuse the width id
  const unsigned long long row = t / (heads * pairs);
  const unsigned long long rem = t % (heads * pairs);
  const unsigned long long head = rem / pairs;
  const unsigned long long j = rem % pairs;
  float theta = 0.0f;
  static_assert(sizeof(theta) == 4);
  std::uint32_t bits = static_cast<std::uint32_t>(theta_bits);
  std::memcpy(&theta, &bits, 4);
  const unsigned long long section =
      j < sec_t ? 0 : (j < sec_t + sec_h ? 1 : 2);
  const float p = static_cast<float>(pos[row * 3 + section]);
  const float angle = p * powf(theta, -2.0f * static_cast<float>(j) /
                                          static_cast<float>(rope_dim));
  const float c = cosf(angle);
  const float s = sinf(angle);
  float* base = data + (row * heads + head) * head_dim;
  const float x1 = base[j];
  const float x2 = base[j + pairs];
  base[j] = x1 * c - x2 * s;
  base[j + pairs] = x1 * s + x2 * c;
}

// Built-in "qgate_split": split a fused gated-attention projection
// into queries and gates (per row, per head query then gate). One thread
// per (row, head, element).
__global__ void QGateSplitKernel(const float* fused, float* q, float* gate,
                                 unsigned long long heads,
                                 unsigned long long head_dim,
                                 unsigned long long rows) {
  const unsigned long long t =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= rows * heads * head_dim) {
    return;
  }
  const unsigned long long row = t / (heads * head_dim);
  const unsigned long long rem = t % (heads * head_dim);
  const unsigned long long head = rem / head_dim;
  const unsigned long long e = rem % head_dim;
  const unsigned long long fbase = (row * heads + head) * 2 * head_dim;
  const unsigned long long obase = row * heads * head_dim + head * head_dim;
  q[obase + e] = fused[fbase + e];
  gate[obase + e] = fused[fbase + head_dim + e];
}
// Built-in "add": elementwise o = a + b over n fp32. One thread per
// element.
__global__ void AddKernel(const float* a, const float* b, float* o,
                          unsigned long long n) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  o[i] = a[i] + b[i];
}

// Built-in "silu_mul": elementwise o = silu(g) * u over n fp32. One
// thread per element.
__global__ void SiluMulKernel(const float* g, const float* u, float* o,
                              unsigned long long n) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float gate = g[i];
  o[i] = (gate / (1.0f + expf(-gate))) * u[i];
}

// Built-in "repeat_heads": expand q/k to the value heads. One thread
// per output element.
__global__ void RepeatHeadsKernel(const float* in, float* out,
                                  unsigned long long num_v_heads,
                                  unsigned long long head_k_dim,
                                  unsigned long long factor,
                                  unsigned long long rows) {
  const unsigned long long row_stride = num_v_heads * head_k_dim;
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * row_stride) {
    return;
  }
  const unsigned long long row = i / row_stride;
  const unsigned long long j = i % row_stride;
  const unsigned long long h = j / head_k_dim;
  const unsigned long long e = j % head_k_dim;
  const unsigned long long k_heads = num_v_heads / factor;
  out[i] =
      in[row * k_heads * head_k_dim + (h % k_heads) * head_k_dim + e];
}

// Built-in "ssm_gate": the gated-delta decay/write gates per value head.
__global__ void SsmGateKernel(const float* a_log, const float* dt,
                              const float* alpha_raw, const float* beta_raw,
                              float* alpha, float* beta,
                              unsigned long long heads,
                              unsigned long long rows) {
  const unsigned long long t =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= rows * heads) {
    return;
  }
  const unsigned long long h = t % heads;
  const float raw = alpha_raw[t] + dt[h];
  const float softplus = raw > 20.0f ? raw : log1pf(expf(raw));
  alpha[t] = expf(a_log[h] * softplus);
  beta[t] = 1.0f / (1.0f + expf(-beta_raw[t]));
}

// Built-in "delta_step_heads": one gated-delta step for all value
// heads. One thread per (head, state column).
__global__ void DeltaStepHeadsKernel(
    float* s, const float* k, const float* v, const float* q, float* o,
    const float* alpha, const float* beta, unsigned long long heads,
    unsigned long long dk, unsigned long long dv, unsigned long long rows,
    unsigned long long sbase, unsigned long long sstride,
    unsigned long long abase) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= heads * dv) {
    return;
  }
  const unsigned long long h = idx / dv;
  const unsigned long long d = idx % dv;
  const unsigned long long s_base = h * dk * dv;
  // The input state is at sbase and the updated state at sbase + sstride; a
  // zero stride advances in place, a non-zero one appends each row's
  // post-step state to its own history slot (row t reads slot t, writes slot
  // t+1) so a verify can restore any accepted prefix without a per-row
  // device copy.
  const unsigned long long s_base_in = sbase + s_base;
  const unsigned long long k_base = h * dk;
  const unsigned long long v_base = h * dv;
  // Rows share the recurrent state and advance it in turn, so a whole
  // sequence costs one launch (rows = 1 is the single-token step).
  for (unsigned long long t = 0; t < rows; ++t) {
    const unsigned long long s_rd = s_base_in + t * sstride;
    const unsigned long long s_wr = s_base_in + (t + 1) * sstride;
    const float* kt = k + t * heads * dk;
    const float* vt = v + t * heads * dv;
    const float* qt = q + t * heads * dk;
    const float a = alpha[abase + t * heads + h];
    const float b = beta[abase + t * heads + h];
    // Four partial sums keep the dk-deep read and output chains short.
    float r0 = 0.0f;
    float r1 = 0.0f;
    float r2 = 0.0f;
    float r3 = 0.0f;
    unsigned long long j = 0;
    for (; j + 4 <= dk; j += 4) {
      r0 = fmaf(s[s_rd + (j + 0) * dv + d], kt[k_base + j + 0], r0);
      r1 = fmaf(s[s_rd + (j + 1) * dv + d], kt[k_base + j + 1], r1);
      r2 = fmaf(s[s_rd + (j + 2) * dv + d], kt[k_base + j + 2], r2);
      r3 = fmaf(s[s_rd + (j + 3) * dv + d], kt[k_base + j + 3], r3);
    }
    for (; j < dk; ++j) {
      r0 = fmaf(s[s_rd + j * dv + d], kt[k_base + j], r0);
    }
    const float read = (r0 + r1) + (r2 + r3);
    float o0 = 0.0f;
    float o1 = 0.0f;
    float o2 = 0.0f;
    float o3 = 0.0f;
    j = 0;
    for (; j + 4 <= dk; j += 4) {
      const float u0 =
          a * (s[s_rd + (j + 0) * dv + d] - b * kt[k_base + j + 0] * read) +
          b * vt[v_base + d] * kt[k_base + j + 0];
      const float u1 =
          a * (s[s_rd + (j + 1) * dv + d] - b * kt[k_base + j + 1] * read) +
          b * vt[v_base + d] * kt[k_base + j + 1];
      const float u2 =
          a * (s[s_rd + (j + 2) * dv + d] - b * kt[k_base + j + 2] * read) +
          b * vt[v_base + d] * kt[k_base + j + 2];
      const float u3 =
          a * (s[s_rd + (j + 3) * dv + d] - b * kt[k_base + j + 3] * read) +
          b * vt[v_base + d] * kt[k_base + j + 3];
      s[s_wr + (j + 0) * dv + d] = u0;
      s[s_wr + (j + 1) * dv + d] = u1;
      s[s_wr + (j + 2) * dv + d] = u2;
      s[s_wr + (j + 3) * dv + d] = u3;
      o0 = fmaf(u0, qt[k_base + j + 0], o0);
      o1 = fmaf(u1, qt[k_base + j + 1], o1);
      o2 = fmaf(u2, qt[k_base + j + 2], o2);
      o3 = fmaf(u3, qt[k_base + j + 3], o3);
    }
    float out = (o0 + o1) + (o2 + o3);
    for (; j < dk; ++j) {
      const float updated =
          a * (s[s_rd + j * dv + d] - b * kt[k_base + j] * read) +
          b * vt[v_base + d] * kt[k_base + j];
      s[s_wr + j * dv + d] = updated;
      out = fmaf(updated, qt[k_base + j], out);
    }
    o[t * heads * dv + idx] = out;
    // The next row reads what this one just wrote.
  }
}

// Built-in "conv1d_step": current-step causal depthwise conv. One
// thread per channel.
__global__ void Conv1dStepKernel(const float* x, const float* w, float* y,
                                 unsigned long long channels,
                                 unsigned long long width) {
  const unsigned long long c =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= channels) {
    return;
  }
  float acc = 0.0f;
  for (unsigned long long i = 0; i < width; ++i) {
    acc = fmaf(w[c * width + i], x[c * width + i], acc);
  }
  y[c] = acc;
}

// Built-in "conv1d_state": current-step causal depthwise conv with the
// history kept on the device, the SiLU and the q/k/v split. One thread
// per channel. W is in checkpoint tap order (tap 0 oldest), the history
// is newest first.
__global__ void Conv1dStateKernel(const float* qkv, const float* w,
                                  float* hist, float* q, float* k, float* v,
                                  float* hist_hist,
                                  unsigned long long conv_dim,
                                  unsigned long long width,
                                  unsigned long long key_dim,
                                  unsigned long long qkv_offset,
                                  unsigned long long rows,
                                  unsigned long long hist_stride) {
  const unsigned long long c =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= conv_dim) {
    return;
  }
  const unsigned long long stride = width - 1;
  const unsigned long long hbase = c * stride;
  const unsigned long long value_dim = conv_dim - 2 * key_dim;
  // Rows advance the shared history in turn, so a sequence costs one launch
  // (rows = 1 with qkv_offset = t * conv_dim is the single-token step). With
  // a non-zero hist_stride each row's post-conv history is written to its own
  // slot, so a batched verify can restore any accepted prefix.
  for (unsigned long long t = 0; t < rows; ++t) {
    const float current = qkv[qkv_offset + t * conv_dim + c];
    float acc = w[c * width + (width - 1)] * current;
    for (unsigned long long i = 1; i < width; ++i) {
      acc = fmaf(w[c * width + (width - 1 - i)], hist[hbase + (i - 1)], acc);
    }
    const float out = acc / (1.0f + expf(-acc));
    if (c < key_dim) {
      q[t * key_dim + c] = out;
    } else if (c < 2 * key_dim) {
      k[t * key_dim + (c - key_dim)] = out;
    } else {
      v[t * value_dim + (c - 2 * key_dim)] = out;
    }
    if (width > 1) {
      for (unsigned long long i = stride; i > 1; --i) {
        hist[hbase + (i - 1)] = hist[hbase + (i - 2)];
      }
      hist[hbase] = current;
    }
    if (hist_stride != 0) {
      float* dst = hist_hist + (t + 1) * hist_stride + hbase;
      for (unsigned long long i = 0; i < stride; ++i) {
        dst[i] = hist[hbase + i];
      }
    }
  }
}

// Built-in "dflash_conv": DFlash2 grouped dynamic convolution. One
// thread per output element.
__global__ void DflashConvKernel(const float* x, const float* delta,
                                 const float* base, float* y,
                                 unsigned long long rows,
                                 unsigned long long channels,
                                 unsigned long long taps,
                                 unsigned long long group_size,
                                 unsigned long long block_size,
                                 unsigned long long delta_row_stride,
                                 unsigned long long delta_offset) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * channels) {
    return;
  }
  const unsigned long long num_groups = channels / group_size;
  const unsigned long long row = i / channels;
  const unsigned long long c = i % channels;
  const unsigned long long grp = c / group_size;
  const unsigned long long position = row % block_size;
  const unsigned long long delta_base = delta_offset + row * delta_row_stride + grp;
  float acc = (base[c] + delta[delta_base]) * x[i];
  for (unsigned long long tap = 1; tap < taps; ++tap) {
    if (position < tap) {
      continue;
    }
    const float coeff =
        base[tap * channels + c] + delta[delta_base + tap * num_groups];
    acc = fmaf(coeff, x[(row - tap) * channels + c], acc);
  }
  y[i] = acc;
}

// Built-in "quantize_q8": one thread per row; symmetric int8 with a
// per-row absmax scale, packed four bytes per word.
__global__ void QuantizeQ8Kernel(const float* in, unsigned int* packed,
                                 float* scale, unsigned long long rows,
                                 unsigned long long cols) {
  const unsigned long long r =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) {
    return;
  }
  const unsigned long long base = r * cols;
  float amax = 0.0f;
  for (unsigned long long c = 0; c < cols; ++c) {
    amax = fmaxf(amax, fabsf(in[base + c]));
  }
  const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
  scale[r] = s;
  const unsigned long long words = cols / 4;
  for (unsigned long long w = 0; w < words; ++w) {
    unsigned int word = 0u;
    for (unsigned int b = 0u; b < 4u; ++b) {
      const float x = in[base + w * 4 + b] / s;
      int q = static_cast<int>(x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f));
      q = q < -127 ? -127 : (q > 127 ? 127 : q);
      word |= (static_cast<unsigned int>(q) & 0xFFu) << (b * 8u);
    }
    packed[r * words + w] = word;
  }
}

// Built-in "attention_q8": GQA with symmetric int8 keys/values.
// Quantized-KV decode for the tiled attention: Kind 0 int8, 1 4-bit
// (two per byte), 2 OCP FP8 E4M3. `s` is the key/value row scale.
enum QuantKvKind { kQuantKvQ8 = 0, kQuantKvQ4 = 1, kQuantKvFp8 = 2 };

template <int Kind>
__device__ __forceinline__ float QuantKvAt(const unsigned char* data,
                                           unsigned long long idx, float s) {
  if (Kind == kQuantKvQ8) {
    return static_cast<float>(static_cast<signed char>(data[idx])) * s;
  }
  if (Kind == kQuantKvQ4) {
    const unsigned int word =
        reinterpret_cast<const unsigned int*>(data)[idx >> 3];
    int value = static_cast<int>((word >> ((idx & 7u) << 2u)) & 0xFu);
    value = (value & 0x8) != 0 ? value - 16 : value;
    return static_cast<float>(value) * s;
  }
  return Fp8E4M3ToFloatDev(data[idx]) * s;
}

// Tiled online-softmax for the quantized KV caches: one workgroup per
// (query row, head), the query staged in shared memory, so the dot product
// is computed once per key and the cost is O(n * head_dim) per query/head.
template <int Kind>
__device__ void AttentionQuantTiled(const float* q, const unsigned char* k,
                                    const unsigned char* v, const float* ks,
                                    const float* vs, float* out,
                                    unsigned long long m, unsigned long long n,
                                    unsigned long long heads,
                                    unsigned long long kv_heads,
                                    unsigned long long head_dim,
                                    unsigned long long q_base,
                                    unsigned long long window) {
  constexpr unsigned long long kTile = 256;
  __shared__ float q_s[256];
  __shared__ float sc[256];
  __shared__ float wt[256];
  // Tile reduction broadcast: thread 0 computes the max/weights once.
  __shared__ float tile_bc[3];
  const unsigned long long g = blockIdx.x;
  if (g >= m * heads) {
    return;
  }
  const unsigned long long e = threadIdx.x;
  const unsigned long long i = g / heads;
  const unsigned long long h = g % heads;
  const unsigned long long kv = h / (heads / kv_heads);
  const unsigned long long pos = q_base + i;
  const unsigned long long last = pos >= n ? n - 1 : pos;
  unsigned long long start =
      (window != 0 && pos + 1 > window) ? pos + 1 - window : 0;
  if (start > last) {
    start = last;
  }
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const unsigned long long qb = (i * heads + h) * head_dim;
  if (e < head_dim) {
    q_s[e] = q[qb + e];
  }
  __syncthreads();
  float run_max = -1e30f;
  float run_sum = 0.0f;
  float acc = 0.0f;
  unsigned long long tile = start;
  while (true) {
    const unsigned long long tile_last =
        (tile + kTile - 1 < last) ? tile + kTile - 1 : last;
    float s = -1e30f;
    if (tile + e <= last) {
      const unsigned long long j = tile + e;
      const unsigned long long kb = (j * kv_heads + kv) * head_dim;
      // Four independent partial sums keep the FMA pipeline full;
      // one chain stalls on its own latency (see the tiled GEMMs).
      float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
      unsigned long long d = 0;
      for (; d + 3 < head_dim; d += 4) {
        dot0 = fmaf(q_s[d], QuantKvAt<Kind>(k, kb + d, ks[j]), dot0);
        dot1 = fmaf(q_s[d + 1], QuantKvAt<Kind>(k, kb + d + 1, ks[j]), dot1);
        dot2 = fmaf(q_s[d + 2], QuantKvAt<Kind>(k, kb + d + 2, ks[j]), dot2);
        dot3 = fmaf(q_s[d + 3], QuantKvAt<Kind>(k, kb + d + 3, ks[j]), dot3);
      }
      float dot = (dot0 + dot1) + (dot2 + dot3);
      for (; d < head_dim; ++d) {
        dot = fmaf(q_s[d], QuantKvAt<Kind>(k, kb + d, ks[j]), dot);
      }
      s = dot * scale;
    }
    sc[e] = s;
    __syncthreads();
    // One thread reduces the tile so a long context pays one scan,
    // not one scan per thread. Same order as before, same numerics.
    if (e == 0) {
      float tmax = -1e30f;
      for (unsigned long long t = 0; t < kTile; ++t) {
        tmax = fmaxf(tmax, sc[t]);
      }
      const float m_new = fmaxf(run_max, tmax);
      const float corr = expf(run_max - m_new);
      float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
      for (unsigned long long t = 0; t < kTile; t += 4) {
        const float w0 = expf(sc[t] - m_new);
        const float w1 = expf(sc[t + 1] - m_new);
        const float w2 = expf(sc[t + 2] - m_new);
        const float w3 = expf(sc[t + 3] - m_new);
        wt[t] = w0;
        wt[t + 1] = w1;
        wt[t + 2] = w2;
        wt[t + 3] = w3;
        l0 += w0;
        l1 += w1;
        l2 += w2;
        l3 += w3;
      }
      tile_bc[0] = m_new;
      tile_bc[1] = corr;
      tile_bc[2] = (l0 + l1) + (l2 + l3);
    }
    __syncthreads();
    const float m_new = tile_bc[0];
    const float corr = tile_bc[1];
    const float l = tile_bc[2];
    // Four independent accumulators, as in the score dot above.
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a = 0.0f;
    if (e < head_dim) {
      const unsigned long long count =
          (tile + kTile - 1 <= last) ? kTile : (last - tile + 1);
      unsigned long long t = 0;
      for (; t + 3 < count; t += 4) {
        const unsigned long long b0 = (tile + t) * kv_heads + kv;
        const unsigned long long b1 = (tile + t + 1) * kv_heads + kv;
        const unsigned long long b2 = (tile + t + 2) * kv_heads + kv;
        const unsigned long long b3 = (tile + t + 3) * kv_heads + kv;
        a0 = fmaf(wt[t], QuantKvAt<Kind>(v, b0 * head_dim + e, vs[tile + t]),
                  a0);
        a1 = fmaf(wt[t + 1],
                  QuantKvAt<Kind>(v, b1 * head_dim + e, vs[tile + t + 1]), a1);
        a2 = fmaf(wt[t + 2],
                  QuantKvAt<Kind>(v, b2 * head_dim + e, vs[tile + t + 2]), a2);
        a3 = fmaf(wt[t + 3],
                  QuantKvAt<Kind>(v, b3 * head_dim + e, vs[tile + t + 3]), a3);
      }
      a = (a0 + a1) + (a2 + a3);
      for (; t < count; ++t) {
        const unsigned long long vb = (tile + t) * kv_heads + kv;
        a = fmaf(wt[t], QuantKvAt<Kind>(v, vb * head_dim + e, vs[tile + t]),
                 a);
      }
    }
    acc = fmaf(corr, acc, a);
    run_sum = fmaf(corr, run_sum, l);
    run_max = m_new;
    __syncthreads();
    if (tile_last >= last) {
      break;
    }
    tile += kTile;
  }
  if (e < head_dim) {
    out[qb + e] = acc / run_sum;
  }
}

__global__ void AttentionQ8Kernel(const float* q, const unsigned char* k,
                                  const unsigned char* v, const float* ks,
                                  const float* vs, float* out,
                                  unsigned long long m, unsigned long long n,
                                  unsigned long long heads,
                                  unsigned long long kv_heads,
                                  unsigned long long head_dim,
                                  unsigned long long q_base,
                                  unsigned long long window,
                                  unsigned long long kind) {
  (void)kind;
  AttentionQuantTiled<kQuantKvQ8>(q, k, v, ks, vs, out, m, n, heads, kv_heads,
                                  head_dim, q_base, window);
}

// Built-in "quantize_fp8_pack": one thread per row; OCP FP8 E4M3 bytes
// packed four per word with the dynamic per-token W4A8 scale.
__global__ void QuantizeFp8PackKernel(const float* in, unsigned int* packed,
                                      float* scale, unsigned long long rows,
                                      unsigned long long cols) {
  const unsigned long long r =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) {
    return;
  }
  const unsigned long long base = r * cols;
  float amax = 0.0f;
  for (unsigned long long c = 0; c < cols; ++c) {
    amax = fmaxf(amax, fabsf(in[base + c]));
  }
  float s = amax / 448.0f;
  const float min_s = 1.0f / (448.0f * 512.0f);
  if (s < min_s) {
    s = min_s;
  }
  scale[r] = s;
  const unsigned long long words = cols / 4;
  for (unsigned long long w = 0; w < words; ++w) {
    unsigned int word = 0u;
    for (unsigned int b = 0u; b < 4u; ++b) {
      const float q =
          fminf(fmaxf(in[base + w * 4 + b] / s, -448.0f), 448.0f);
      const std::uint8_t bits = Fp8E4M3FromFloatDev(q);
      word |= static_cast<unsigned int>(bits) << (b * 8u);
    }
    packed[r * words + w] = word;
  }
}

// Built-in "quantize_fp8_pack_rows": as quantize_fp8_pack but one workgroup
// per row (shared-memory amax reduction), so a few huge rows (the fp8-WMMA
// activation) do not serialize on one thread. Buffers: 0 in (f32 rows x
// cols), 1 packed (rows x cols/4 uint), 2 scale (rows). Scalars rows, cols
// (a multiple of 4). Grid `rows`, block 256.
__global__ void QuantizeFp8PackRowsKernel(const float* in, unsigned int* packed,
                                          float* scale, unsigned long long rows,
                                          unsigned long long cols) {
  __shared__ float red[1024];
  const unsigned long long r = blockIdx.x;
  if (r >= rows) {
    return;
  }
  const unsigned long long tid = threadIdx.x;
  const unsigned long long base = r * cols;
  float amax = 0.0f;
  for (unsigned long long c = tid; c < cols; c += blockDim.x) {
    amax = fmaxf(amax, fabsf(in[base + c]));
  }
  red[tid] = amax;
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s) {
      red[tid] = fmaxf(red[tid], red[tid + s]);
    }
  }
  __syncthreads();
  float sc = red[0] / 448.0f;
  const float min_s = 1.0f / (448.0f * 512.0f);
  if (sc < min_s) {
    sc = min_s;
  }
  if (tid == 0) {
    scale[r] = sc;
  }
  const unsigned long long words = cols / 4;
  for (unsigned long long w = tid; w < words; w += blockDim.x) {
    unsigned int word = 0u;
    for (unsigned int b = 0u; b < 4u; ++b) {
      const float q = fminf(fmaxf(in[base + w * 4 + b] / sc, -448.0f), 448.0f);
      const std::uint8_t bits = Fp8E4M3FromFloatDev(q);
      word |= static_cast<unsigned int>(bits) << (b * 8u);
    }
    packed[r * words + w] = word;
  }
}

// Built-in "attention_fp8": GQA with OCP FP8 E4M3 keys/values.
__global__ void AttentionFp8Kernel(const float* q, const unsigned char* k,
                                   const unsigned char* v, const float* ks,
                                   const float* vs, float* out,
                                   unsigned long long m, unsigned long long n,
                                   unsigned long long heads,
                                   unsigned long long kv_heads,
                                   unsigned long long head_dim,
                                   unsigned long long q_base,
                                   unsigned long long window,
                                   unsigned long long kind) {
  (void)kind;
  AttentionQuantTiled<kQuantKvFp8>(q, k, v, ks, vs, out, m, n, heads, kv_heads,
                                   head_dim, q_base, window);
}

// Built-in "spatial_merge": group merge x merge patches, concatenating
// their embeddings.
__global__ void SpatialMergeKernel(const float* x, float* y,
                                   unsigned long long embed,
                                   unsigned long long grid_w,
                                   unsigned long long grid_h,
                                   unsigned long long merge) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long m2 = merge * merge;
  const unsigned long long out_w = grid_w / merge;
  const unsigned long long out_tokens = (grid_h / merge) * out_w;
  if (idx >= out_tokens * m2 * embed) {
    return;
  }
  const unsigned long long per_out = m2 * embed;
  const unsigned long long b = idx / per_out;
  const unsigned long long rem = idx % per_out;
  const unsigned long long q = rem / embed;
  const unsigned long long c = rem % embed;
  const unsigned long long mh = q / merge;
  const unsigned long long mw = q % merge;
  const unsigned long long gbh = b / out_w;
  const unsigned long long gbw = b % out_w;
  const unsigned long long in_row =
      (gbh * merge + mh) * grid_w + (gbw * merge + mw);
  y[idx] = x[in_row * embed + c];
}

// Built-in "bias_add": row-broadcast bias add.
__global__ void BiasAddKernel(const float* x, const float* b, float* y,
                              unsigned long long rows,
                              unsigned long long cols) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= rows * cols) {
    return;
  }
  y[idx] = x[idx] + b[idx % cols];
}

// Built-in "image_patchify": normalize and split an image into patches.
__global__ void ImagePatchifyKernel(const float* image, const float* mean,
                                    const float* sd, float* out_values,
                                    unsigned long long h,
                                    unsigned long long w,
                                    unsigned long long patch) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long nx = w / patch;
  const unsigned long long ny = h / patch;
  const unsigned long long per_patch = 3 * patch * patch;
  if (idx >= nx * ny * per_patch) {
    return;
  }
  const unsigned long long p = idx / per_patch;
  const unsigned long long rem = idx % per_patch;
  const unsigned long long c = rem / (patch * patch);
  const unsigned long long pos = rem % (patch * patch);
  const unsigned long long ph = pos / patch;
  const unsigned long long pw = pos % patch;
  const unsigned long long py = p / nx;
  const unsigned long long px = p % nx;
  const unsigned long long y = py * patch + ph;
  const unsigned long long x = px * patch + pw;
  out_values[idx] = (image[(y * w + x) * 3 + c] - mean[c]) / sd[c];
}

// Built-in "layernorm": one thread per row.
__global__ void LayerNormKernel(const float* x, const float* w, const float* b,
                                float* y, unsigned long long rows,
                                unsigned long long cols, float eps) {
  const unsigned long long r =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) {
    return;
  }
  const unsigned long long base = r * cols;
  float mean = 0.0f;
  for (unsigned long long c = 0; c < cols; ++c) {
    mean += x[base + c];
  }
  mean /= static_cast<float>(cols);
  float var = 0.0f;
  for (unsigned long long c = 0; c < cols; ++c) {
    const float d = x[base + c] - mean;
    var = fmaf(d, d, var);
  }
  var /= static_cast<float>(cols);
  const float inv = rsqrtf(var + eps);
  for (unsigned long long c = 0; c < cols; ++c) {
    y[base + c] = (x[base + c] - mean) * inv * w[c] + b[c];
  }
}

// Built-in "gelu": elementwise tanh approximation.
__global__ void GeluKernel(const float* x, float* y, unsigned long long n) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float t = x[i];
  y[i] = 0.5f * t *
         (1.0f + tanhf(0.7978845608f * (t + 0.044715f * t * t * t)));
}

// Built-in "quantize_q4": one thread per row; symmetric 4-bit with a
// per-row absmax scale, packed eight nibbles per word.
__global__ void QuantizeQ4Kernel(const float* in, unsigned int* packed,
                                 float* scale, unsigned long long rows,
                                 unsigned long long cols) {
  const unsigned long long r =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) {
    return;
  }
  const unsigned long long base = r * cols;
  float amax = 0.0f;
  for (unsigned long long c = 0; c < cols; ++c) {
    amax = fmaxf(amax, fabsf(in[base + c]));
  }
  const float s = amax > 0.0f ? amax / 7.0f : 1.0f;
  scale[r] = s;
  const unsigned long long words = cols / 8;
  for (unsigned long long w = 0; w < words; ++w) {
    unsigned int word = 0u;
    for (unsigned int b = 0u; b < 8u; ++b) {
      const float x = in[base + w * 8 + b] / s;
      int q = static_cast<int>(x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f));
      q = q < -7 ? -7 : (q > 7 ? 7 : q);
      word |= (static_cast<unsigned int>(q) & 0xFu) << (b * 4u);
    }
    packed[r * words + w] = word;
  }
}

// Built-in "attention_q4": GQA with symmetric 4-bit keys/values.
__global__ void AttentionQ4Kernel(const float* q, const unsigned char* k,
                                  const unsigned char* v, const float* ks,
                                  const float* vs, float* out,
                                  unsigned long long m, unsigned long long n,
                                  unsigned long long heads,
                                  unsigned long long kv_heads,
                                  unsigned long long head_dim,
                                  unsigned long long q_base,
                                  unsigned long long window,
                                  unsigned long long kind) {
  (void)kind;
  AttentionQuantTiled<kQuantKvQ4>(q, k, v, ks, vs, out, m, n, heads, kv_heads,
                                  head_dim, q_base, window);
}

// Built-in "cast_f32_f16": two fp32 -> one packed fp16 word.
__global__ void CastF32F16Kernel(const float* in, unsigned int* out,
                                 unsigned long long n) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx * 2 >= n) {
    return;
  }
  const std::uint16_t lo = FloatToFp16Dev(in[idx * 2]);
  const std::uint16_t hi = FloatToFp16Dev(in[idx * 2 + 1]);
  out[idx] = static_cast<unsigned int>(lo) |
             (static_cast<unsigned int>(hi) << 16);
}

// Built-in "concat_features": stack n tensors of rows x features along
// the feature axis.
__global__ void ConcatFeaturesKernel(const float* in, float* out,
                                     unsigned long long n,
                                     unsigned long long rows,
                                     unsigned long long features) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= rows * n * features) {
    return;
  }
  const unsigned long long width = n * features;
  const unsigned long long r = idx / width;
  const unsigned long long rem = idx % width;
  const unsigned long long i = rem / features;
  const unsigned long long j = rem % features;
  out[idx] = in[(i * rows + r) * features + j];
}

// Built-in "selector_edge_score": DFlash2 candidate-selector transition
// score. One thread per output element.
__global__ void SelectorEdgeScoreKernel(
    const float* predecessor_codebook, const float* successor_codebook,
    const float* hidden, const int* candidate_ids, const int* anchor_ids,
    const float* unary, float* out_scores, unsigned long long batch,
    unsigned long long seq, unsigned long long top_k,
    unsigned long long rank) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long positions = batch * seq;
  if (index >= positions * top_k * top_k) {
    return;
  }
  const unsigned long long c = index % top_k;
  const unsigned long long p = (index / top_k) % top_k;
  const unsigned long long pos = index / (top_k * top_k);
  const unsigned long long l = pos % seq;
  const int pred_id =
      (l == 0) ? anchor_ids[pos] : candidate_ids[pos * top_k - top_k + p];
  const int succ_id = candidate_ids[pos * top_k + c];
  float dot = 0.0f;
  const unsigned long long hidden_row = pos * rank;
  const unsigned long long pred_row =
      static_cast<unsigned long long>(pred_id) * rank;
  const unsigned long long succ_row =
      static_cast<unsigned long long>(succ_id) * rank;
  for (unsigned long long r = 0; r < rank; ++r) {
    dot = fmaf(predecessor_codebook[pred_row + r] * hidden[hidden_row + r],
               successor_codebook[succ_row + r], dot);
  }
  out_scores[index] = unary[pos * top_k + c] + dot;
}

// Built-in "top_k_rows": per-row top-k over `vocab` fp32 logits (spec/DFlash2
// candidate selection). One workgroup per row, 256 threads. Each thread keeps
// a descending local top-k in scratch, then the workgroup merges the `count`
// sorted lists with a k-way merge: each pass reduces the current head of every
// list to the largest, records it and advances that list.
__global__ void TopKRowsKernel(const float* logits, unsigned int* ids,
                               float* vals, unsigned long long rows,
                               unsigned long long vocab,
                               unsigned long long top_k,
                               unsigned long long row_base) {
  constexpr unsigned long long k = 32;
  __shared__ float rv[256];
  __shared__ unsigned int ri[256];
  __shared__ unsigned int rown[256];
  const unsigned long long r = blockIdx.x;
  if (r >= rows || top_k == 0 || top_k > k) {
    return;
  }
  const unsigned long long tid = threadIdx.x;
  const unsigned long long dim = blockDim.x;
  const float* row = logits + (row_base + r) * vocab;
  float lv[k];
  unsigned int li[k];
  unsigned long long count = 0;
  for (unsigned long long c = tid; c < vocab; c += dim) {
    const float v = row[c];
    if (count < top_k) {
      unsigned long long j = count;
      while (j > 0 && lv[j - 1] < v) {
        lv[j] = lv[j - 1];
        li[j] = li[j - 1];
        --j;
      }
      lv[j] = v;
      li[j] = static_cast<unsigned int>(c);
      ++count;
    } else if (v > lv[top_k - 1]) {
      unsigned long long j = top_k - 1;
      while (j > 0 && lv[j - 1] < v) {
        lv[j] = lv[j - 1];
        li[j] = li[j - 1];
        --j;
      }
      lv[j] = v;
      li[j] = static_cast<unsigned int>(c);
    }
  }
  unsigned long long head = 0;
  for (unsigned long long p = 0; p < top_k; ++p) {
    rv[tid] = (head < count) ? lv[head] : -INFINITY;
    ri[tid] = (head < count) ? li[head] : 0u;
    rown[tid] = static_cast<unsigned int>(tid);
    __syncthreads();
    for (unsigned long long s = dim / 2; s > 0; s >>= 1) {
      if (tid < s && rv[tid + s] > rv[tid]) {
        rv[tid] = rv[tid + s];
        ri[tid] = ri[tid + s];
        rown[tid] = rown[tid + s];
      }
      __syncthreads();
    }
    if (tid == 0) {
      vals[r * top_k + p] = rv[0];
      ids[r * top_k + p] = ri[0];
    }
    const unsigned int owner = rown[0];
    __syncthreads();
    if (tid == owner) {
      ++head;
    }
    __syncthreads();
  }
}

}  // namespace tessera::backends::rocm
