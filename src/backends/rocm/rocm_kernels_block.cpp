#include "backends/rocm/rocm_kernels.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace tessera::backends::rocm {
// Built-in "gemm_q4k": buffer 0 is the activation A (fp32, m x k),
// buffer 1 the quantized weights W (Q4_K, n x k), buffer 2 the output
// C (fp32, m x n); scalars are m, n, k. One workgroup per 256 output
// elements (block_x = 256).
__global__ void GemmQ4KKernel(const float* a, const unsigned char* w,
                              float* c, unsigned long long m,
                              unsigned long long n,
                              unsigned long long k) {
  unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= m * n) {
    return;
  }
  const unsigned long long row_a = idx / n;
  const unsigned long long row_w = idx % n;
  const unsigned long long blocks = k / 256;
  float acc = 0.0f;
  for (unsigned long long b = 0; b < blocks; ++b) {
    const unsigned char* base =
        w + (row_w * blocks + b) * 144;
    std::uint16_t d_bits = 0;
    std::uint16_t dm_bits = 0;
    std::memcpy(&d_bits, base, 2);
    std::memcpy(&dm_bits, base + 2, 2);
    const float d = Fp16ToFloatDev(d_bits);
    const float dm = Fp16ToFloatDev(dm_bits);
    const unsigned char* scales = base + 4;
    // The 128 nibble bytes start at base + 16; the block is 144 bytes so
    // base is 16-byte aligned and the word loads are aligned. Read four
    // bytes (eight nibbles) per load instead of one byte (two).
    const unsigned int* qwords =
        reinterpret_cast<const unsigned int*>(base + 16);
    for (unsigned long long grp = 0; grp < 4; ++grp) {
      std::uint8_t sc0, mn0, sc1, mn1;
      GetScaleMinDev(2 * grp, scales, &sc0, &mn0);
      GetScaleMinDev(2 * grp + 1, scales, &sc1, &mn1);
      const float a0 = d * static_cast<float>(sc0);
      const float o0 = dm * static_cast<float>(mn0);
      const float a1 = d * static_cast<float>(sc1);
      const float o1 = dm * static_cast<float>(mn1);
      for (unsigned long long w32 = 0; w32 < 8; ++w32) {
        unsigned int word = qwords[grp * 8 + w32];
        for (unsigned long long kk = 0; kk < 4; ++kk) {
          const std::uint8_t q = static_cast<std::uint8_t>(word & 0xFFu);
          word >>= 8;
          const unsigned long long l = w32 * 4 + kk;
          const unsigned long long t = b * 256 + grp * 64 + l;
          acc += a[row_a * k + t] * (a0 * static_cast<float>(q & 15) - o0);
          acc += a[row_a * k + t + 32] *
                 (a1 * static_cast<float>(q >> 4) - o1);
        }
      }
    }
  }
  c[idx] = acc;
}

// Built-in "gemm_q4k_row": C = A x dequant(W)^T with one workgroup per
// output element (row_a, row_w), so the weight and activation reads are
// coalesced across the workgroup instead of strided per thread. Thread t
// owns element `b * 256 + t` of every block: the activation read is 256
// consecutive floats and the Q4_K byte read hits sub-block `t / 32`
// (lane `t % 32`). Partial sums reduce in shared memory. Dispatch is
// m * n workgroups of 256 (block_x = 256); wave-size independent.
__global__ void GemmQ4KRowKernel(const float* a, const unsigned char* w,
                                 float* c, unsigned long long m,
                                 unsigned long long n, unsigned long long k) {
  const unsigned long long out =
      static_cast<unsigned long long>(blockIdx.x);
  if (out >= m * n) {
    return;
  }
  const unsigned long long row_a = out / n;
  const unsigned long long row_w = out % n;
  const unsigned long long blocks = k / 256;
  const unsigned int t = threadIdx.x;
  const unsigned int s = t / 32;
  const unsigned int lane = t % 32;
  const unsigned int byte_off = 16 + (s / 2) * 32 + lane;
  const float* a_row = a + row_a * k;
  float acc = 0.0f;
  for (unsigned long long b = 0; b < blocks; ++b) {
    const unsigned char* base = w + (row_w * blocks + b) * 144;
    std::uint16_t d_bits = 0;
    std::uint16_t dm_bits = 0;
    std::memcpy(&d_bits, base, 2);
    std::memcpy(&dm_bits, base + 2, 2);
    const float d = Fp16ToFloatDev(d_bits);
    const float dm = Fp16ToFloatDev(dm_bits);
    std::uint8_t sc = 0;
    std::uint8_t mn = 0;
    GetScaleMinDev(s, base + 4, &sc, &mn);
    const std::uint8_t byte = base[byte_off];
    const std::uint8_t nib = (s & 1u) == 0 ? (byte & 15u) : (byte >> 4);
    const float wv = d * static_cast<float>(sc) * static_cast<float>(nib) -
                     dm * static_cast<float>(mn);
    acc = fmaf(a_row[b * 256 + t], wv, acc);
  }
  __shared__ float sh[256];
  sh[t] = acc;
  __syncthreads();
  for (unsigned int off = 128; off > 0; off >>= 1) {
    if (t < off) {
      sh[t] += sh[t + off];
    }
    __syncthreads();
  }
  if (t == 0u) {
    c[row_a * n + row_w] = sh[0];
  }
}

// Built-in "gemm_q5k": C = A x dequant(W)^T, fp32 sequential
// accumulation; W holds Q5_K blocks (176 bytes per 256).
__global__ void GemmQ5KKernel(const float* a, const unsigned char* w,
                              float* c, unsigned long long m,
                              unsigned long long n, unsigned long long k) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= m * n) {
    return;
  }
  const unsigned long long row_a = idx / n;
  const unsigned long long row_w = idx % n;
  const unsigned long long blocks = k / 256;
  float acc = 0.0f;
  for (unsigned long long b = 0; b < blocks; ++b) {
    const unsigned char* base = w + (row_w * blocks + b) * 176;
    float d = 0.0f;
    float dmin = 0.0f;
    std::uint16_t d_bits = 0;
    std::uint16_t dm_bits = 0;
    std::memcpy(&d_bits, base, 2);
    std::memcpy(&dm_bits, base + 2, 2);
    d = Fp16ToFloatDev(d_bits);
    dmin = Fp16ToFloatDev(dm_bits);
    unsigned int u1 = 1;
    unsigned int u2 = 2;
    for (unsigned long long g = 0; g < 4; ++g) {
      std::uint8_t sc0, mn0, sc1, mn1;
      GetScaleMinDev(2 * g, base + 4, &sc0, &mn0);
      GetScaleMinDev(2 * g + 1, base + 4, &sc1, &mn1);
      const float d1 = d * sc0;
      const float o1 = dmin * mn0;
      const float d2 = d * sc1;
      const float o2 = dmin * mn1;
      for (unsigned long long l = 0; l < 32; ++l) {
        const unsigned long long t = b * 256 + g * 64 + l;
        const unsigned char low = base[48 + g * 32 + l];
        const unsigned char high = base[16 + l];
        acc = fmaf(a[row_a * k + t],
                   d1 * ((low & 15) + (high & u1 ? 16 : 0)) - o1, acc);
        acc = fmaf(a[row_a * k + t + 32],
                   d2 * ((low >> 4) + (high & u2 ? 16 : 0)) - o2, acc);
      }
      u1 <<= 2;
      u2 <<= 2;
    }
  }
  c[idx] = acc;
}

// Built-in "gemm_q6k": C = A x dequant(W)^T; W holds Q6_K blocks
// (210 bytes per 256).
__global__ void GemmQ6KKernel(const float* a, const unsigned char* w,
                              float* c, unsigned long long m,
                              unsigned long long n, unsigned long long k) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= m * n) {
    return;
  }
  const unsigned long long row_a = idx / n;
  const unsigned long long row_w = idx % n;
  const unsigned long long blocks = k / 256;
  float acc = 0.0f;
  for (unsigned long long b = 0; b < blocks; ++b) {
    const unsigned char* base = w + (row_w * blocks + b) * 210;
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, base + 208, 2);
    const float d = Fp16ToFloatDev(d_bits);
    for (unsigned long long n2 = 0; n2 < 2; ++n2) {
      for (unsigned long long l = 0; l < 32; ++l) {
        const unsigned long long is = l / 16;
        const unsigned char ql0 = base[n2 * 64 + l];
        const unsigned char ql1 = base[n2 * 64 + 32 + l];
        const unsigned char qh = base[128 + n2 * 32 + l];
        const auto s = [&](unsigned long long o) {
          int v = base[192 + n2 * 8 + is + o];
          return v >= 128 ? v - 256 : v;
        };
        const int q1 = (ql0 & 15) | (((qh >> 0) & 3) << 4);
        const int q2 = (ql1 & 15) | (((qh >> 2) & 3) << 4);
        const int q3 = (ql0 >> 4) | (((qh >> 4) & 3) << 4);
        const int q4 = (ql1 >> 4) | (((qh >> 6) & 3) << 4);
        const unsigned long long t = b * 256 + n2 * 128 + l;
        acc = fmaf(a[row_a * k + t], d * s(0) * (q1 - 32), acc);
        acc = fmaf(a[row_a * k + t + 32], d * s(2) * (q2 - 32), acc);
        acc = fmaf(a[row_a * k + t + 64], d * s(4) * (q3 - 32), acc);
        acc = fmaf(a[row_a * k + t + 96], d * s(6) * (q4 - 32), acc);
      }
    }
  }
  c[idx] = acc;
}

// Built-in "gemm_q3k": C = A x dequant(W)^T; W holds Q3_K blocks
// (110 bytes per 256).
__global__ void GemmQ3KKernel(const float* a, const unsigned char* w,
                              float* c, unsigned long long m,
                              unsigned long long n, unsigned long long k) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= m * n) {
    return;
  }
  const unsigned long long row_a = idx / n;
  const unsigned long long row_w = idx % n;
  const unsigned long long blocks = k / 256;
  float acc = 0.0f;
  for (unsigned long long b = 0; b < blocks; ++b) {
    const unsigned char* base = w + (row_w * blocks + b) * 110;
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, base + 108, 2);
    const float d_all = Fp16ToFloatDev(d_bits);
    std::uint32_t aux[4] = {};
    for (int e = 0; e < 3; ++e) {
      std::uint32_t word = 0;
      for (int f = 0; f < 4; ++f) {
        word |= static_cast<std::uint32_t>(base[96 + e * 4 + f]) << (8 * f);
      }
      aux[e] = word;
    }
    const std::uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4);
    aux[0] = (aux[0] & 0x0f0f0f0f) | (((tmp >> 0) & 0x03030303) << 4);
    aux[1] = (aux[1] & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4);
    const std::int8_t* scales = reinterpret_cast<const std::int8_t*>(aux);
    unsigned int is = 0;
    unsigned int mask = 1;
    for (unsigned long long n2 = 0; n2 < 2; ++n2) {
      unsigned int shift = 0;
      for (unsigned long long j = 0; j < 4; ++j) {
        const float dl = d_all * (scales[is++] - 32);
        for (unsigned long long l = 0; l < 16; ++l) {
          const float qv = static_cast<float>(
              (base[32 + n2 * 32 + l] >> shift) & 3);
          const float hb = (base[l] & mask) ? 0.0f : 4.0f;
          const unsigned long long t = b * 256 + n2 * 128 + j * 32 + l;
          acc = fmaf(a[row_a * k + t], dl * (qv - hb), acc);
        }
        const float dl2 = d_all * (scales[is++] - 32);
        for (unsigned long long l = 0; l < 16; ++l) {
          const float qv = static_cast<float>(
              (base[32 + n2 * 32 + 16 + l] >> shift) & 3);
          const float hb = (base[16 + l] & mask) ? 0.0f : 4.0f;
          const unsigned long long t = b * 256 + n2 * 128 + j * 32 + 16 + l;
          acc = fmaf(a[row_a * k + t], dl2 * (qv - hb), acc);
        }
        shift += 2;
        mask <<= 1;
      }
    }
  }
  c[idx] = acc;
}

// Built-in "gemm_q4k_batched": tiled C = A x dequant(W)^T for a batch
// of rows. One workgroup handles 8 activation rows x one weight column;
// the Q4_K block is dequantized once into shared memory and reused
// across the 8 rows, so a weight block is dequantized once per tile
// instead of once per row. The 8 dot threads accumulate sequentially
// over k in the same order as gemm_q4k, so the result matches the GEMV
// kernel up to floating-point reassociation (none here: one chain).
// Buffers and scalars match gemm_q4k (m, n, k, k a multiple of 256);
// grid_x is ceil(m / 8) * n, block_x is 256. The 8-row tile matches
// core::detail::kGemmTileRows. Threads outside the 8 dot rows still
// help dequantize but write no output.
__global__ void GemmQ4KBatchedKernel(const float* a, const unsigned char* w,
                                     float* c, unsigned long long m,
                                     unsigned long long n,
                                     unsigned long long k) {
  const unsigned long long wg =
      static_cast<unsigned long long>(blockIdx.x);
  const unsigned long long tid =
      static_cast<unsigned long long>(threadIdx.x);
  const unsigned long long tm = wg / n;
  const unsigned long long row_w = wg % n;
  const unsigned long long base_row = tm * 8;
  const bool dot_valid = tid < 8 && base_row + tid < m;
  const unsigned long long row_a = base_row + tid;
  const unsigned long long blocks = k / 256;
  __shared__ float wq[256];
  float acc = 0.0f;
  for (unsigned long long b = 0; b < blocks; ++b) {
    // Cooperatively dequantize block (row_w, b): thread tid writes
    // element tid, using the same sub-block map as gemm_q4k.
    const unsigned char* base = w + (row_w * blocks + b) * 144;
    std::uint16_t d_bits = 0;
    std::uint16_t dm_bits = 0;
    std::memcpy(&d_bits, base, 2);
    std::memcpy(&dm_bits, base + 2, 2);
    const float d = Fp16ToFloatDev(d_bits);
    const float dm = Fp16ToFloatDev(dm_bits);
    const unsigned char* scales = base + 4;
    const unsigned char* qs = base + 16;
    const unsigned long long e = tid;
    const unsigned long long grp = e / 64;
    const unsigned long long half = (e % 64) / 32;
    const unsigned long long l = e % 32;
    std::uint8_t sc = 0;
    std::uint8_t mn = 0;
    GetScaleMinDev(2 * grp + half, scales, &sc, &mn);
    const std::uint8_t q = qs[grp * 32 + l];
    const std::uint8_t nib = half != 0 ? (q >> 4) : (q & 15);
    wq[e] = d * static_cast<float>(sc) * static_cast<float>(nib) -
            dm * static_cast<float>(mn);
    __syncthreads();
    // One dot per tile row in the gemm_q4k element order (per 64-element
    // group, the low 32 then the high 32, interleaved per lane) so the
    // accumulation matches the GEMV kernel.
    if (dot_valid) {
      const float* ar = a + row_a * k + b * 256;
      for (unsigned long long grp = 0; grp < 4; ++grp) {
        for (unsigned long long l = 0; l < 32; ++l) {
          const unsigned long long t0 = grp * 64 + l;
          const unsigned long long t1 = t0 + 32;
          acc += ar[t0] * wq[t0];
          acc += ar[t1] * wq[t1];
        }
      }
    }
    __syncthreads();
  }
  if (dot_valid) {
    c[row_a * n + row_w] = acc;
  }
}

// Built-in "gemm_q5k_batched": tiled C = A x dequant(W)^T for a batch.
// One workgroup handles 8 activation rows x one weight column; the 256
// Q5_K elements of a block are dequantized once into shared memory and
// reused across the rows. Element `e` is sub-block `e / 32`; sub-block
// `s` uses scale/min pair `s`, low nibble byte `48 + (s/2)*32 + e%32`,
// the high bit `1 << s` of byte `16 + e%32`, and the high nibble for odd
// `s`. The dot uses the gemm_q5k order (per 64-element group, low then
// high half). Dispatch is ceil(m / 8) * n workgroups of 256.
__global__ void GemmQ5KBatchedKernel(const float* a, const unsigned char* w,
                                     float* c, unsigned long long m,
                                     unsigned long long n,
                                     unsigned long long k) {
  const unsigned long long wg =
      static_cast<unsigned long long>(blockIdx.x);
  const unsigned long long tid =
      static_cast<unsigned long long>(threadIdx.x);
  const unsigned long long tm = wg / n;
  const unsigned long long row_w = wg % n;
  const unsigned long long base_row = tm * 8;
  const bool dot_valid = tid < 8 && base_row + tid < m;
  const unsigned long long row_a = base_row + tid;
  const unsigned long long blocks = k / 256;
  __shared__ float wq[256];
  float acc = 0.0f;
  for (unsigned long long b = 0; b < blocks; ++b) {
    const unsigned char* base = w + (row_w * blocks + b) * 176;
    std::uint16_t d_bits = 0;
    std::uint16_t dm_bits = 0;
    std::memcpy(&d_bits, base, 2);
    std::memcpy(&dm_bits, base + 2, 2);
    const float d = Fp16ToFloatDev(d_bits);
    const float dm = Fp16ToFloatDev(dm_bits);
    const unsigned long long e = tid;
    const unsigned long long s = e / 32;
    const unsigned long long lane = e % 32;
    const unsigned long long grp = s / 2;
    const unsigned long long parity = s % 2;
    std::uint8_t sc = 0;
    std::uint8_t mn = 0;
    GetScaleMinDev(s, base + 4, &sc, &mn);
    const std::uint8_t low = base[48 + grp * 32 + lane];
    const std::uint8_t high = base[16 + lane];
    const std::uint8_t nib = parity != 0 ? (low >> 4) : (low & 15);
    const std::uint8_t hb = (high & (1u << s)) != 0 ? 16u : 0u;
    wq[e] = d * static_cast<float>(sc) * static_cast<float>(nib + hb) -
            dm * static_cast<float>(mn);
    __syncthreads();
    if (dot_valid) {
      const float* ar = a + row_a * k + b * 256;
      for (unsigned long long g = 0; g < 4; ++g) {
        for (unsigned long long l = 0; l < 32; ++l) {
          const unsigned long long t0 = g * 64 + l;
          const unsigned long long t1 = t0 + 32;
          acc += ar[t0] * wq[t0];
          acc += ar[t1] * wq[t1];
        }
      }
    }
    __syncthreads();
  }
  if (dot_valid) {
    c[row_a * n + row_w] = acc;
  }
}

// Built-in "gemm_q80": buffer 0 is the activation A (fp32, m x k),
// buffer 1 the quantized weights W (Q8_0, n x k), buffer 2 the output
// C (fp32, m x n); scalars are m, n, k. A Q8_0 block is 34 bytes:
// d fp16 then 32 signed bytes. One thread per output element.
__global__ void GemmQ80Kernel(const float* a, const unsigned char* w,
                              float* c, unsigned long long m,
                              unsigned long long n, unsigned long long k) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= m * n) {
    return;
  }
  const unsigned long long row_a = idx / n;
  const unsigned long long row_w = idx % n;
  const unsigned long long blocks = k / 32;
  float acc = 0.0f;
  for (unsigned long long b = 0; b < blocks; ++b) {
    const unsigned char* base = w + row_w * (blocks * 34) + b * 34;
    const std::uint16_t half = static_cast<std::uint16_t>(
        base[0] | (static_cast<std::uint16_t>(base[1]) << 8));
    const float d = Fp16ToFloatDev(half);
    for (unsigned long long l = 0; l < 32; ++l) {
      const auto q = static_cast<std::int8_t>(base[2 + l]);
      acc = fmaf(a[row_a * k + b * 32 + l], d * static_cast<float>(q), acc);
    }
  }
  c[idx] = acc;
}
}  // namespace tessera::backends::rocm
