#include "backends/rocm/rocm_kernels.hpp"

#include <cstdint>

#include "backends/rocm/rocm_kernels_codec.hpp"
#include "backends/rocm/rocm_quant_tables.hpp"

// fp16 tensor-core batched GEMM for the block-quantized weight formats, the
// prefill path for m >= a tile of rows. C[m,n] fp32 = A_fp32[m,k] x
// dequant(W)[n,k]^T with the activations rounded to fp16 and the weights
// dequantized to fp16 in kernel. The dequantized weight is the only place a
// format shows: one TileCodec per format decodes eight consecutive k
// elements of one column into the shared staging tile. The k-tile is staged
// through shared memory because a raw global fragment read is about 10x
// slower (same reason as gemm_mxfp4_wmma).
//
// Tile: 64 rows x 64 columns per workgroup over 256 threads. Eight warps
// each own a 16-row x 32-column region of the tile, so the A fragment of a
// k-step is loaded once and feeds both 16-column MMA tiles: the shared
// memory traffic per FLOP stays near the minimum the 16x16x16 builtin needs.
// The row tile is deliberately deep: the dequant ALU cost per weight element
// is amortized over the tile rows, and a 64-row tile keeps the fp16 MMA busy
// (measured 46 TFLOPS against 5 in the scalar tiled kernel).

namespace tessera::backends::rocm {
#if defined(TESSERA_ROCM_WMMA)

// Tile geometry in fp16 elements. The 8-element pad on each staged row keeps
// the fragment rows off each other's banks.
constexpr int kTileRows = 128;
constexpr int kTileCols = 64;
constexpr int kTileK = 64;
constexpr int kTileStride = kTileK + 8;
constexpr int kTileThreads = 256;
constexpr int kStageGroups = kTileK / 8;

using F16x8 = short __attribute__((ext_vector_type(8)));
using F32x8 = float __attribute__((ext_vector_type(8)));

// A tile codec with no lookup table.
struct NoTileLut {
  static constexpr unsigned kLutWords = 1;

  __device__ __forceinline__ static void FillLut(std::uint32_t* lut, int tid) {
    if (tid == 0) {
      lut[0] = 0u;
    }
  }
};

// Q4_K: 144 bytes per 256 elements. Element t of the superblock is sub-block
// g = t / 64, half hf = (t / 32) % 2 and index l = t % 32; the low nibble of
// qs[g*32 + l] is sub-block 2g and the high nibble is 2g+1.
struct Q4KTileCodec : NoTileLut {
  static constexpr unsigned kBlockElements = 256;
  static constexpr unsigned kBlockBytes = 144;

  // `lut` is the shared lookup window; this format needs no table.
  __device__ __forceinline__ static void Decode8(const unsigned char* base,
                                                 unsigned int t,
                                                 const std::uint32_t* /*lut*/,
                                                 __half* dst) {
    const unsigned int g = t >> 6;
    const unsigned int hf = (t >> 5) & 1u;
    const float d = Fp16ToFloatDev(detail::LoadU16(base));
    const float dm = Fp16ToFloatDev(detail::LoadU16(base + 2));
    std::uint8_t sc = 0;
    std::uint8_t mn = 0;
    GetScaleMinDev(g * 2 + hf, base + 4, &sc, &mn);
    const float dsc = d * static_cast<float>(sc);
    const float dmm = dm * static_cast<float>(mn);
    const unsigned char* q = base + 16 + g * 32 + (t & 31u);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const unsigned int qb = q[e];
      const unsigned int nib = hf == 0 ? (qb & 15u) : (qb >> 4);
      dst[e] = __float2half_rn(dsc * static_cast<float>(nib) - dmm);
    }
  }
};

// Q5_K: 176 bytes per 256 elements. The fifth bit of sub-block s lives in
// bit s of qh[l]; the low four bits are the nibble of qs[s/2*32 + l].
struct Q5KTileCodec : NoTileLut {
  static constexpr unsigned kBlockElements = 256;
  static constexpr unsigned kBlockBytes = 176;

  // `lut` is the shared lookup window; this format needs no table.
  __device__ __forceinline__ static void Decode8(const unsigned char* base,
                                                 unsigned int t,
                                                 const std::uint32_t* /*lut*/,
                                                 __half* dst) {
    const unsigned int s = t >> 5;
    const unsigned int l = t & 31u;
    const float d = Fp16ToFloatDev(detail::LoadU16(base));
    const float dm = Fp16ToFloatDev(detail::LoadU16(base + 2));
    std::uint8_t sc = 0;
    std::uint8_t mn = 0;
    GetScaleMinDev(s, base + 4, &sc, &mn);
    const float dsc = d * static_cast<float>(sc);
    const float dmm = dm * static_cast<float>(mn);
    const unsigned char* low = base + 48 + (s >> 1) * 32 + l;
    const unsigned char* high = base + 16 + l;
    const unsigned int bit = 1u << s;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const unsigned int qb = low[e];
      const unsigned int nib = (s & 1u) != 0 ? (qb >> 4) : (qb & 15u);
      const unsigned int hb = (high[e] & bit) != 0 ? 16u : 0u;
      dst[e] = __float2half_rn(dsc * static_cast<float>(nib + hb) - dmm);
    }
  }
};

// Q6_K: 210 bytes per 256 elements. Element t = n2*128 + q*32 + l has four
// low bits in ql[n2*64 + (q%2)*32 + l], two high bits in the 2*q bits of
// qh[n2*32 + l], and the signed scale at 192 + n2*8 + l/16 + 2*q.
struct Q6KTileCodec : NoTileLut {
  static constexpr unsigned kBlockElements = 256;
  static constexpr unsigned kBlockBytes = 210;

  // `lut` is the shared lookup window; this format needs no table.
  __device__ __forceinline__ static void Decode8(const unsigned char* base,
                                                 unsigned int t,
                                                 const std::uint32_t* /*lut*/,
                                                 __half* dst) {
    const unsigned int n2 = t >> 7;
    const unsigned int q = (t >> 5) & 3u;
    const unsigned int l = t & 31u;
    const float d = Fp16ToFloatDev(detail::LoadU16(base + 208));
    const int scale =
        static_cast<int>(static_cast<std::int8_t>(
            base[192 + n2 * 8 + (l >> 4) + 2 * q]));
    const float ds = d * static_cast<float>(scale);
    const unsigned char* ql = base + n2 * 64 + (q & 1u) * 32 + l;
    const unsigned char* qh = base + 128 + n2 * 32 + l;
    const unsigned int shift = 2 * q;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const unsigned int low = q < 2 ? (ql[e] & 15u) : (ql[e] >> 4);
      const unsigned int high = ((qh[e] >> shift) & 3u) << 4;
      const int qv = static_cast<int>(low | high) - 32;
      dst[e] = __float2half_rn(ds * static_cast<float>(qv));
    }
  }
};

// Q3_K: 110 bytes per 256 elements. Element t = n2*128 + q*32 + part*16 + l
// selects two of the three bits from the 2*q bits of qs[32 + n2*32 +
// part*16 + l] and the high bit from hm[part*16 + l], inverted: a clear bit
// subtracts 4. Sixteen 6-bit scales are packed in the twelve bytes at 96,
// split as two 4-bit halves in two words plus two high bits in a third.
struct Q3KTileCodec : NoTileLut {
  static constexpr unsigned kBlockElements = 256;
  static constexpr unsigned kBlockBytes = 110;

  // `lut` is the shared lookup window; this format needs no table.
  __device__ __forceinline__ static void Decode8(const unsigned char* base,
                                                 unsigned int t,
                                                 const std::uint32_t* /*lut*/,
                                                 __half* dst) {
    const unsigned int n2 = t >> 7;
    const unsigned int q = (t >> 5) & 3u;
    const unsigned int part = (t >> 4) & 1u;
    const unsigned int l = t & 15u;
    const unsigned int is = (n2 * 4 + q) * 2 + part;
    const std::uint32_t w0 = detail::LoadU32(base + 96);
    const std::uint32_t w1 = detail::LoadU32(base + 100);
    const std::uint32_t w2 = detail::LoadU32(base + 104);
    const unsigned int byte = is & 3u;
    const unsigned int half = is >> 2;
    const std::uint32_t src = (half & 1u) != 0 ? w1 : w0;
    const std::uint32_t b = (src >> (8 * byte)) & 0xFFu;
    const unsigned int low = half < 2 ? (b & 15u) : (b >> 4);
    const unsigned int high = ((w2 >> (8 * byte + 2 * half)) & 3u) << 4;
    const float d = Fp16ToFloatDev(detail::LoadU16(base + 108));
    const float dl = d * static_cast<float>(
                             static_cast<int>(low | high) - 32);
    const unsigned char* qs = base + 32 + n2 * 32 + part * 16 + l;
    const unsigned char* hm = base + part * 16 + l;
    const unsigned int shift = 2 * q;
    const unsigned int mask = 1u << (n2 * 4 + q);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float qv = static_cast<float>((qs[e] >> shift) & 3u);
      const float hb = (hm[e] & mask) != 0 ? 0.0f : 4.0f;
      dst[e] = __float2half_rn(dl * (qv - hb));
    }
  }
};

// IQ4_XS: 136 bytes per 256 elements. Sub-block ib = t / 32 takes its 6-bit
// scale from the packed nibble at 4 + ib/2 and the two high bits of the
// word at 2; each nibble of qs[8 + ib*16 + (t%16)] indexes the IQ4 codebook.
struct Iq4XsTileCodec {
  static constexpr unsigned kBlockElements = 256;
  static constexpr unsigned kBlockBytes = 136;
  static constexpr unsigned kLutWords = 16;

  __device__ __forceinline__ static void FillLut(std::uint32_t* lut,
                                                 int tid) {
    if (tid < 16) {
      lut[tid] =
          __float_as_uint(static_cast<float>(kIq4NlValuesDev[tid]));
    }
  }

  __device__ __forceinline__ static void Decode8(const unsigned char* base,
                                                 unsigned int t,
                                                 const std::uint32_t* lut,
                                                 __half* dst) {
    const unsigned int ib = t >> 5;
    const unsigned int l = t & 31u;
    const float d = Fp16ToFloatDev(detail::LoadU16(base));
    const unsigned int packed = base[4 + (ib >> 1)];
    const unsigned int high = detail::LoadU16(base + 2);
    const int ls = static_cast<int>(((packed >> (4 * (ib & 1u))) & 15u) |
                                    (((high >> (2 * ib)) & 3u) << 4));
    const float dl = d * static_cast<float>(ls - 32);
    const unsigned char* q = base + 8 + ib * 16 + (l & 15u);
    const bool hi = l >= 16;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const unsigned int qb = q[e];
      const unsigned int nib = hi ? (qb >> 4) : (qb & 15u);
      dst[e] = __float2half_rn(dl * __uint_as_float(lut[nib]));
    }
  }
};

// IQ4_NL: 18 bytes per 32 elements. The nibble of block byte j is element j
// (low) or j+16 (high) and indexes the IQ4 codebook.
struct Iq4NlTileCodec {
  static constexpr unsigned kBlockElements = 32;
  static constexpr unsigned kBlockBytes = 18;
  static constexpr unsigned kLutWords = 16;

  __device__ __forceinline__ static void FillLut(std::uint32_t* lut,
                                                 int tid) {
    if (tid < 16) {
      lut[tid] =
          __float_as_uint(static_cast<float>(kIq4NlValuesDev[tid]));
    }
  }

  __device__ __forceinline__ static void Decode8(const unsigned char* base,
                                                 unsigned int t,
                                                 const std::uint32_t* lut,
                                                 __half* dst) {
    const float d = Fp16ToFloatDev(detail::LoadU16(base));
    const unsigned char* q = base + 2 + (t & 15u);
    const bool hi = t >= 16;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const unsigned int qb = q[e];
      const unsigned int nib = hi ? (qb >> 4) : (qb & 15u);
      dst[e] = __float2half_rn(d * __uint_as_float(lut[nib]));
    }
  }
};

// IQ3_S: 110 bytes per 256 elements. Element t = s*32 + l0*8 + j*2 + mm is
// the byte j of the grid word selected by qs[2 + ib*8 + 2*l0] plus the qh
// bit at 66 + ib + part, scaled by db and signed by bit j (mm 0) or j+4
// (mm 1) of signs[74 + ib*4 + part*4 + l0]; ib = (s/2)*2, part = s%2, and
// db = d * (1 + 2*scale nibble).
struct Iq3STileCodec {
  static constexpr unsigned kBlockElements = 256;
  static constexpr unsigned kBlockBytes = 110;
  static constexpr unsigned kLutWords = 512;

  __device__ __forceinline__ static void FillLut(std::uint32_t* lut,
                                                 int tid) {
    for (int i = tid; i < 512; i += kTileThreads) {
      lut[i] = kIq3sGridDev[i];
    }
  }

  __device__ __forceinline__ static void Decode8(const unsigned char* base,
                                                 unsigned int t,
                                                 const std::uint32_t* lut,
                                                 __half* dst) {
    const unsigned int s = t >> 5;
    const unsigned int l0 = (t >> 3) & 3u;
    const unsigned int ib = (s >> 1) * 2;
    const unsigned int part = s & 1u;
    const float d = Fp16ToFloatDev(detail::LoadU16(base));
    const unsigned char db_byte = base[106 + (s >> 1)];
    const unsigned int nib = part == 0 ? (db_byte & 15u) : (db_byte >> 4);
    const float db = d * static_cast<float>(1 + 2 * nib);
    const unsigned int qh = base[66 + ib + part];
    const unsigned int g1 =
        lut[base[2 + ib * 8 + 2 * l0] | ((qh << (8 - 2 * l0)) & 256u)];
    const unsigned int g2 =
        lut[base[2 + ib * 8 + 2 * l0 + 1] | ((qh << (7 - 2 * l0)) & 256u)];
    const unsigned int signs = base[74 + ib * 4 + part * 4 + l0];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      // Element e of the group is k offset e of the sub-block: the low two
      // bits select the grid byte and the third selects the grid word.
      const unsigned int j = static_cast<unsigned int>(e) & 3u;
      const unsigned int mm = static_cast<unsigned int>(e) >> 2;
      const unsigned int gv =
          mm == 0 ? ((g1 >> (8 * j)) & 255u) : ((g2 >> (8 * j)) & 255u);
      const unsigned int bit = mm == 0 ? j : j + 4;
      const float sign = (signs & (1u << bit)) != 0 ? -1.0f : 1.0f;
      dst[e] = __float2half_rn(db * static_cast<float>(gv) * sign);
    }
  }
};

// Q8_0: 34 bytes per 32 elements. d fp16 then 32 signed bytes.
struct Q80TileCodec : NoTileLut {
  static constexpr unsigned kBlockElements = 32;
  static constexpr unsigned kBlockBytes = 34;

  // `lut` is the shared lookup window; this format needs no table.
  __device__ __forceinline__ static void Decode8(const unsigned char* base,
                                                 unsigned int t,
                                                 const std::uint32_t* /*lut*/,
                                                 __half* dst) {
    const float d = Fp16ToFloatDev(detail::LoadU16(base));
    const unsigned char* q = base + 2 + (t & 31u);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      dst[e] = __float2half_rn(
          d * static_cast<float>(static_cast<int>(
                  static_cast<std::int8_t>(q[e]))));
    }
  }
};

// The shared tile staging and MMA loop, shared by every format.
template <class Codec>
__device__ __forceinline__ void WmmaGemmBody(const float* a,
                                             const unsigned char* w, float* c,
                                             unsigned long long m,
                                             unsigned long long n,
                                             unsigned long long k) {
  __shared__ __half sA[kTileRows * kTileStride];
  __shared__ __half sW[kTileCols * kTileStride];
  __shared__ std::uint32_t sLut[Codec::kLutWords];
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const unsigned long long row0 =
      static_cast<unsigned long long>(blockIdx.y) * kTileRows;
  const unsigned long long col0 =
      static_cast<unsigned long long>(blockIdx.x) * kTileCols;
  const int wr = warp * 16;
  const int wc = 0;
  const int acol8 = (lane >> 4) * 8;
  const unsigned long long blocks = k / Codec::kBlockElements;
  Codec::FillLut(sLut, tid);
  LdsBarrier();
  F32x8 acc0 = F32x8(0.0f);
  F32x8 acc1 = F32x8(0.0f);
  F32x8 acc2 = F32x8(0.0f);
  F32x8 acc3 = F32x8(0.0f);
  for (unsigned long long k0 = 0; k0 < k; k0 += kTileK) {
    for (int u = tid; u < kTileRows * kStageGroups; u += kTileThreads) {
      const int r = u / kStageGroups;
      const int k8 = (u % kStageGroups) * 8;
      const unsigned long long ar =
          static_cast<unsigned long long>(row0 + r) < m ? row0 + r : m - 1;
      const float* src = &a[ar * k + k0 + k8];
      __half* dst = &sA[r * kTileStride + k8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        dst[e] = __float2half_rn(src[e]);
      }
    }
    for (int u = tid; u < kTileCols * kStageGroups; u += kTileThreads) {
      const int cc = u / kStageGroups;
      const int k8 = (u % kStageGroups) * 8;
      const unsigned long long gc =
          static_cast<unsigned long long>(col0 + cc) < n ? col0 + cc : n - 1;
      const unsigned long long e = k0 + k8;
      const unsigned char* base =
          w + (gc * blocks + e / Codec::kBlockElements) * Codec::kBlockBytes;
      Codec::Decode8(base, static_cast<unsigned int>(e % Codec::kBlockElements),
                     sLut, &sW[cc * kTileStride + k8]);
    }
    LdsBarrier();
#pragma unroll
    for (int ks = 0; ks < kTileK / 16; ++ks) {
      F16x8 af;
      F16x8 wf0;
      F16x8 wf1;
      F16x8 wf2;
      F16x8 wf3;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        af[j] = *reinterpret_cast<const short*>(
            &sA[(wr + (lane & 15)) * kTileStride + ks * 16 + acol8 + j]);
        wf0[j] = *reinterpret_cast<const short*>(
            &sW[(wc + (lane & 15)) * kTileStride + ks * 16 + acol8 + j]);
        wf1[j] = *reinterpret_cast<const short*>(
            &sW[(wc + 16 + (lane & 15)) * kTileStride + ks * 16 + acol8 + j]);
        wf2[j] = *reinterpret_cast<const short*>(
            &sW[(wc + 32 + (lane & 15)) * kTileStride + ks * 16 + acol8 + j]);
        wf3[j] = *reinterpret_cast<const short*>(
            &sW[(wc + 48 + (lane & 15)) * kTileStride + ks * 16 + acol8 + j]);
      }
      acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, wf0, acc0);
      acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, wf1, acc1);
      acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, wf2, acc2);
      acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, wf3, acc3);
    }
    LdsBarrier();
  }
  const unsigned long long r =
      static_cast<unsigned long long>(row0 + wr + acol8);
  const unsigned long long n0 =
      static_cast<unsigned long long>(col0 + wc + (lane & 15));
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    if (r + j < m && n0 < n) {
      c[(r + j) * n + n0] = acc0[j];
    }
    if (r + j < m && n0 + 16 < n) {
      c[(r + j) * n + n0 + 16] = acc1[j];
    }
    if (r + j < m && n0 + 32 < n) {
      c[(r + j) * n + n0 + 32] = acc2[j];
    }
    if (r + j < m && n0 + 48 < n) {
      c[(r + j) * n + n0 + 48] = acc3[j];
    }
  }
}

// One entry point per format, so the host registry can name each kernel.
__global__ void GemmQ4KWmmaKernel(const float* a, const unsigned char* w,
                                  float* c, unsigned long long m,
                                  unsigned long long n, unsigned long long k) {
  WmmaGemmBody<Q4KTileCodec>(a, w, c, m, n, k);
}

__global__ void GemmQ5KWmmaKernel(const float* a, const unsigned char* w,
                                  float* c, unsigned long long m,
                                  unsigned long long n, unsigned long long k) {
  WmmaGemmBody<Q5KTileCodec>(a, w, c, m, n, k);
}

__global__ void GemmQ6KWmmaKernel(const float* a, const unsigned char* w,
                                  float* c, unsigned long long m,
                                  unsigned long long n, unsigned long long k) {
  WmmaGemmBody<Q6KTileCodec>(a, w, c, m, n, k);
}

__global__ void GemmQ3KWmmaKernel(const float* a, const unsigned char* w,
                                  float* c, unsigned long long m,
                                  unsigned long long n, unsigned long long k) {
  WmmaGemmBody<Q3KTileCodec>(a, w, c, m, n, k);
}

__global__ void GemmIq4XsWmmaKernel(const float* a, const unsigned char* w,
                                    float* c, unsigned long long m,
                                    unsigned long long n,
                                    unsigned long long k) {
  WmmaGemmBody<Iq4XsTileCodec>(a, w, c, m, n, k);
}

__global__ void GemmIq4NlWmmaKernel(const float* a, const unsigned char* w,
                                    float* c, unsigned long long m,
                                    unsigned long long n,
                                    unsigned long long k) {
  WmmaGemmBody<Iq4NlTileCodec>(a, w, c, m, n, k);
}

__global__ void GemmIq3SWmmaKernel(const float* a, const unsigned char* w,
                                   float* c, unsigned long long m,
                                   unsigned long long n, unsigned long long k) {
  WmmaGemmBody<Iq3STileCodec>(a, w, c, m, n, k);
}

__global__ void GemmQ80WmmaKernel(const float* a, const unsigned char* w,
                                  float* c, unsigned long long m,
                                  unsigned long long n, unsigned long long k) {
  WmmaGemmBody<Q80TileCodec>(a, w, c, m, n, k);
}

#endif  // defined(TESSERA_ROCM_WMMA)
}  // namespace tessera::backends::rocm
