#include "backends/rocm/rocm_kernels.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace tessera::backends::rocm {
// Built-in "gemm_q4k": buffer 0 is the activation A (fp32, m x k),
// buffer 1 the quantized weights W (Q4_K, n x k), buffer 2 the output
// C (fp32, m x n); scalar 0 is k, scalar 1 n, scalar 2 m. One
// workgroup per 256 output elements (block_x = 256).
__global__ void GemmQ4KKernel(const float* a, const unsigned char* w,
                              float* c, unsigned long long k,
                              unsigned long long n,
                              unsigned long long m) {
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
    const unsigned char* qs = base + 16;
    for (unsigned long long grp = 0; grp < 4; ++grp) {
      std::uint8_t sc0, mn0, sc1, mn1;
      GetScaleMinDev(2 * grp, scales, &sc0, &mn0);
      GetScaleMinDev(2 * grp + 1, scales, &sc1, &mn1);
      const float a0 = d * static_cast<float>(sc0);
      const float o0 = dm * static_cast<float>(mn0);
      const float a1 = d * static_cast<float>(sc1);
      const float o1 = dm * static_cast<float>(mn1);
      for (unsigned long long l = 0; l < 32; ++l) {
        const std::uint8_t q = qs[32 * grp + l];
        const unsigned long long t = b * 256 + grp * 64 + l;
        acc += a[row_a * k + t] * (a0 * static_cast<float>(q & 15) - o0);
        acc += a[row_a * k + t + 32] *
               (a1 * static_cast<float>(q >> 4) - o1);
      }
    }
  }
  c[idx] = acc;
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
