#pragma once

// Shared block-quantized GEMV codecs for the ROCm kernels. Each codec
// dequantizes one superblock format and accumulates the dot product for a
// single output lane of a 32-lane warp-per-output GEMV. The warp-per-output
// family ("gemm_<fmt>_vec") is one 32-lane group per output element and
// eight groups per 256-thread workgroup. The fused mixture-of-experts
// kernels reuse the same codecs so the expert and dense paths share one
// decode.

#include "backends/rocm/rocm_kernels.hpp"

namespace tessera::backends::rocm {
namespace detail {

__device__ __forceinline__ std::uint16_t LoadU16(const unsigned char* p) {
  std::uint16_t v = 0;
  std::memcpy(&v, p, 2);
  return v;
}

// The warp-per-output GEMV family ("gemm_<fmt>_vec"): one 32-lane group per
// output element and eight groups per 256-thread workgroup, so the grid is
// ceil(m n / 8). Every lane owns eight elements of each quant block and
// reads them with aligned loads; the block scale/min is broadcast within
// the lanes that share it; the partial sums reduce with five warp shuffles
// instead of a shared-memory tree over 256 threads. The coalesced column
// GEMV that came before read the weight stream through one workgroup per
// output with that 256-thread shared reduction and stalled at 40-150 GB/s;
// these kernels stream the same bytes at the memory rate. Element order
// per lane is the low half of the block first, then the high half, so ROCm
// and Vulkan agree when both use this mapping.

// Q4_K: lane l owns byte l of each of the four 32-byte qs groups of the
// 144-byte superblock. For group g the low nibble is element g*64+l and
// the high nibble is g*64+l+32; the two sub-block scales (2g, 2g+1) have
// the same index on every lane, so the scale extraction stays uniform
// across the warp.
struct Q4KCodec {
  static constexpr unsigned kBlockElements = 256;
  static constexpr unsigned kBlockBytes = 144;

  __device__ __forceinline__ static float Dot(const float* av,
                                              const unsigned char* wr,
                                              unsigned long long blocks,
                                              unsigned int lane) {
    float acc = 0.0f;
    for (unsigned long long b = 0; b < blocks; ++b) {
      const unsigned char* base = wr + b * kBlockBytes;
      const float d = Fp16ToFloatDev(LoadU16(base));
      const float dm = Fp16ToFloatDev(LoadU16(base + 2));
      const float* e = av + b * kBlockElements + lane;
#pragma unroll
      for (unsigned int g = 0; g < 4; ++g) {
        const unsigned char q = base[16 + g * 32 + lane];
        std::uint8_t sc0 = 0;
        std::uint8_t mn0 = 0;
        std::uint8_t sc1 = 0;
        std::uint8_t mn1 = 0;
        GetScaleMinDev(2 * g, base + 4, &sc0, &mn0);
        GetScaleMinDev(2 * g + 1, base + 4, &sc1, &mn1);
        acc = fmaf(e[g * 64],
                   d * static_cast<float>(sc0) * static_cast<float>(q & 15u) -
                       dm * static_cast<float>(mn0),
                   acc);
        acc = fmaf(e[g * 64 + 32],
                   d * static_cast<float>(sc1) * static_cast<float>(q >> 4) -
                       dm * static_cast<float>(mn1),
                   acc);
      }
    }
    return acc;
  }
};

// Q5_K: lane l owns byte l of each of the four 32-byte ql groups and of the
// shared 32-byte qh group (176-byte superblock). For group g the low
// element is g*64+lane, with its 5th bit from qh bit 2g; the high element
// is +32, with bit 2g+1.
struct Q5KCodec {
  static constexpr unsigned kBlockElements = 256;
  static constexpr unsigned kBlockBytes = 176;

  __device__ __forceinline__ static float Dot(const float* av,
                                              const unsigned char* wr,
                                              unsigned long long blocks,
                                              unsigned int lane) {
    float acc = 0.0f;
    for (unsigned long long b = 0; b < blocks; ++b) {
      const unsigned char* base = wr + b * kBlockBytes;
      const float d = Fp16ToFloatDev(LoadU16(base));
      const float dm = Fp16ToFloatDev(LoadU16(base + 2));
      const unsigned char qh = base[16 + lane];
      const float* e0 = av + b * kBlockElements + lane;
#pragma unroll
      for (unsigned int g = 0; g < 4; ++g) {
        const unsigned char ql = base[48 + g * 32 + lane];
        std::uint8_t sc0 = 0;
        std::uint8_t mn0 = 0;
        std::uint8_t sc1 = 0;
        std::uint8_t mn1 = 0;
        GetScaleMinDev(2 * g, base + 4, &sc0, &mn0);
        GetScaleMinDev(2 * g + 1, base + 4, &sc1, &mn1);
        const float v0 = static_cast<float>((ql & 15u) +
                                            (((qh >> (2 * g)) & 1u) << 4));
        const float v1 = static_cast<float>((ql >> 4) +
                                            (((qh >> (2 * g + 1)) & 1u) << 4));
        acc = fmaf(e0[g * 64],
                   d * static_cast<float>(sc0) * v0 -
                       dm * static_cast<float>(mn0),
                   acc);
        acc = fmaf(e0[g * 64 + 32],
                   d * static_cast<float>(sc1) * v1 -
                       dm * static_cast<float>(mn1),
                   acc);
      }
    }
    return acc;
  }
};

// Q6_K: lane l owns byte l of the two 32-byte qh groups and of each ql
// group (210-byte superblock). Per 128-element half it produces four
// elements (t, t+32, t+64, t+96) sharing the byte's low/high nibbles and
// two qh bit pairs, with four int8 sub-block scales.
struct Q6KCodec {
  static constexpr unsigned kBlockElements = 256;
  static constexpr unsigned kBlockBytes = 210;

  __device__ __forceinline__ static float Dot(const float* av,
                                              const unsigned char* wr,
                                              unsigned long long blocks,
                                              unsigned int lane) {
    const unsigned int is = lane >> 4;
    float acc = 0.0f;
    for (unsigned long long b = 0; b < blocks; ++b) {
      const unsigned char* base = wr + b * kBlockBytes;
      const float d = Fp16ToFloatDev(LoadU16(base + 208));
#pragma unroll
      for (unsigned int n2 = 0; n2 < 2; ++n2) {
        const unsigned char* s = base + 192 + n2 * 8 + is;
        const float s0 = d * static_cast<float>(static_cast<int>(
                                 static_cast<std::int8_t>(s[0])));
        const float s1 = d * static_cast<float>(static_cast<int>(
                                 static_cast<std::int8_t>(s[2])));
        const float s2 = d * static_cast<float>(static_cast<int>(
                                 static_cast<std::int8_t>(s[4])));
        const float s3 = d * static_cast<float>(static_cast<int>(
                                 static_cast<std::int8_t>(s[6])));
        const unsigned char ql0 = base[n2 * 64 + lane];
        const unsigned char ql1 = base[n2 * 64 + 32 + lane];
        const unsigned char qh = base[128 + n2 * 32 + lane];
        const float q1 =
            static_cast<float>((ql0 & 15u) | (((qh >> 0) & 3u) << 4)) - 32.0f;
        const float q2 =
            static_cast<float>((ql1 & 15u) | (((qh >> 2) & 3u) << 4)) - 32.0f;
        const float q3 =
            static_cast<float>((ql0 >> 4) | (((qh >> 4) & 3u) << 4)) - 32.0f;
        const float q4 =
            static_cast<float>((ql1 >> 4) | (((qh >> 6) & 3u) << 4)) - 32.0f;
        const float* e = av + b * kBlockElements + n2 * 128 + lane;
        acc = fmaf(e[0], s0 * q1, acc);
        acc = fmaf(e[32], s1 * q2, acc);
        acc = fmaf(e[64], s2 * q3, acc);
        acc = fmaf(e[96], s3 * q4, acc);
      }
    }
    return acc;
  }
};

}  // namespace detail
}  // namespace tessera::backends::rocm
