#include "backends/rocm/rocm_kernels_codec.hpp"
#include "backends/rocm/rocm_kernels.hpp"

namespace tessera::backends::rocm {

namespace {

using detail::LoadU16;
using detail::Q4KCodec;
using detail::Q5KCodec;
using detail::Q6KCodec;

// Shared body of the warp-per-output family: grid covers m*n 32-lane
// outputs, eight per 256-thread workgroup, five shuffles to reduce.
// grid_y selects the split-K chunk when split > 1: each chunk covers
// k/split elements and writes its partial to its own region of `c` (the
// caller passes the partial buffer); split == 1 writes the output
// directly at chunk 0. Small projections (n = 5120 is under one wave)
// use the split so enough workgroups exist to saturate memory.
template <typename Codec>
__device__ __forceinline__ void GemvVecBody(const float* a,
                                            const unsigned char* w, float* c,
                                            unsigned long long m,
                                            unsigned long long n,
                                            unsigned long long k,
                                            unsigned long long split) {
  const unsigned long long warps = blockDim.x >> 5;
  const unsigned long long total = m * n;
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * warps + (threadIdx.x >> 5);
  if (idx >= total) {
    return;
  }
  const unsigned long long blocks = k / Codec::kBlockElements;
  const unsigned long long per = blocks / split;
  const unsigned long long b0 =
      static_cast<unsigned long long>(blockIdx.y) * per;
  const unsigned int lane = threadIdx.x & 31u;
  const float* av = a + (idx / n) * k + b0 * Codec::kBlockElements;
  const unsigned char* wr =
      w + (idx % n) * blocks * Codec::kBlockBytes + b0 * Codec::kBlockBytes;
  float acc = Codec::Dot(av, wr, per, lane);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(~0ull, acc, off, 32);
  }
  if (lane == 0u) {
    c[static_cast<unsigned long long>(blockIdx.y) * total + idx] = acc;
  }
}

}  // namespace

// Built-in "gemm_q4k_vec": warp-per-output Q4_K GEMV (see Q4KCodec).
// Scalars: m, n, k, split; grid_y is the split-K chunk.
__global__ void GemmQ4KVecKernel(const float* a, const unsigned char* w,
                                 float* c, unsigned long long m,
                                 unsigned long long n, unsigned long long k,
                                 unsigned long long split) {
  GemvVecBody<Q4KCodec>(a, w, c, m, n, k, split);
}

// Built-in "gemm_q5k_vec": warp-per-output Q5_K GEMV (see Q5KCodec).
__global__ void GemmQ5KVecKernel(const float* a, const unsigned char* w,
                                 float* c, unsigned long long m,
                                 unsigned long long n, unsigned long long k,
                                 unsigned long long split) {
  GemvVecBody<Q5KCodec>(a, w, c, m, n, k, split);
}

// Built-in "gemm_q6k_vec": warp-per-output Q6_K GEMV (see Q6KCodec).
__global__ void GemmQ6KVecKernel(const float* a, const unsigned char* w,
                                 float* c, unsigned long long m,
                                 unsigned long long n, unsigned long long k,
                                 unsigned long long split) {
  GemvVecBody<Q6KCodec>(a, w, c, m, n, k, split);
}

// Built-in "gemm_vec_reduce": sums the `split` partial regions written by
// the "_vec" kernels into c. Ordered accumulation, so the result is
// deterministic. Scalars: total (m*n outputs), split.
__global__ void GemmVecReduceKernel(const float* part, float* c,
                                    unsigned long long total,
                                    unsigned long long split) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  float acc = 0.0f;
  for (unsigned long long s = 0; s < split; ++s) {
    acc += part[s * total + idx];
  }
  c[idx] = acc;
}

namespace {

// The multi-row GEMV family ("gemm_<fmt>_rowsR"): one warp per weight
// column decodes each quant block once and applies it to all R activation
// rows, so a small verify batch (the folding MTP verify at R = 2, or a
// chained block up to R = 4) reads the weight matrix about once instead of
// once per row. R is a template parameter so the row loop unrolls into
// named accumulators with no dynamic indexing; the earlier runtime-row
// version was twice as slow as the per-row GEMV because of that.
//
// grid_x is ceil(n / 8), grid_y is the split-K chunk. Each chunk writes its
// partial to region grid_y of c for "gemm_vec_reduce" (split == 1 writes
// the output directly).

// Five-shuffle reduction and store of the R accumulators. `base` is the
// split chunk's offset into c (blockIdx.y * m * n; 0 when split is 1).
template <int R>
__device__ __forceinline__ void RowsStore(float* c, const float (&acc)[R],
                                          unsigned long long n,
                                          unsigned long long base,
                                          unsigned long long warp,
                                          unsigned int lane) {
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float v = acc[r];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(~0ull, v, off, 32);
    }
    if (lane == 0u) {
      c[base + static_cast<unsigned long long>(r) * n + warp] = v;
    }
  }
}

}  // namespace

// Built-in "gemm_q4k_rows{R}": R-row Q4_K GEMV. Scalars: m (= R), n, k,
// split; grid_x is ceil(n / 8).
template <int R>
__global__ void GemmQ4KRowsKernel(const float* a, const unsigned char* w,
                                  float* c, unsigned long long m,
                                  unsigned long long n, unsigned long long k,
                                  unsigned long long split) {
  const unsigned long long warp =
      (static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
       threadIdx.x) /
      32;
  const unsigned int lane = threadIdx.x & 31u;
  if (warp >= n) {
    return;
  }
  const unsigned long long blocks = k / 256;
  const unsigned long long per = blocks / split;
  const unsigned long long b0 =
      static_cast<unsigned long long>(blockIdx.y) * per;
  const unsigned char* wr = w + warp * blocks * 144 + b0 * 144;
  float acc[R] = {};
  for (unsigned long long b = 0; b < per; ++b) {
    const unsigned char* base = wr + b * 144;
    const float d = Fp16ToFloatDev(LoadU16(base));
    const float dm = Fp16ToFloatDev(LoadU16(base + 2));
#pragma unroll
    for (unsigned int g = 0; g < 4; ++g) {
      const unsigned char q = base[16 + g * 32 + lane];
      std::uint8_t sc0 = 0;
      std::uint8_t mn0 = 0;
      std::uint8_t sc1 = 0;
      std::uint8_t mn1 = 0;
      GetScaleMinDev(2 * g, base + 4, &sc0, &mn0);
      GetScaleMinDev(2 * g + 1, base + 4, &sc1, &mn1);
      const float wl = d * static_cast<float>(sc0) *
                           static_cast<float>(q & 15u) -
                       dm * static_cast<float>(mn0);
      const float wh = d * static_cast<float>(sc1) *
                           static_cast<float>(q >> 4) -
                       dm * static_cast<float>(mn1);
      const unsigned long long tb = (b0 + b) * 256 + g * 64 + lane;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float* e = a + static_cast<unsigned long long>(r) * k + tb;
        acc[r] = fmaf(e[0], wl, acc[r]);
        acc[r] = fmaf(e[32], wh, acc[r]);
      }
    }
  }
  RowsStore<R>(c, acc, n, static_cast<unsigned long long>(blockIdx.y) * m * n,
               warp, lane);
}

// Built-in "gemm_q5k_rows{R}": R-row Q5_K GEMV.
template <int R>
__global__ void GemmQ5KRowsKernel(const float* a, const unsigned char* w,
                                  float* c, unsigned long long m,
                                  unsigned long long n, unsigned long long k,
                                  unsigned long long split) {
  const unsigned long long warp =
      (static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
       threadIdx.x) /
      32;
  const unsigned int lane = threadIdx.x & 31u;
  if (warp >= n) {
    return;
  }
  const unsigned long long blocks = k / 256;
  const unsigned long long per = blocks / split;
  const unsigned long long b0 =
      static_cast<unsigned long long>(blockIdx.y) * per;
  const unsigned char* wr = w + warp * blocks * 176 + b0 * 176;
  float acc[R] = {};
  for (unsigned long long b = 0; b < per; ++b) {
    const unsigned char* base = wr + b * 176;
    const float d = Fp16ToFloatDev(LoadU16(base));
    const float dm = Fp16ToFloatDev(LoadU16(base + 2));
    const unsigned char qh = base[16 + lane];
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
      const float wl =
          d * static_cast<float>(sc0) * v0 - dm * static_cast<float>(mn0);
      const float wh =
          d * static_cast<float>(sc1) * v1 - dm * static_cast<float>(mn1);
      const unsigned long long tb = (b0 + b) * 256 + g * 64 + lane;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float* e = a + static_cast<unsigned long long>(r) * k + tb;
        acc[r] = fmaf(e[0], wl, acc[r]);
        acc[r] = fmaf(e[32], wh, acc[r]);
      }
    }
  }
  RowsStore<R>(c, acc, n, static_cast<unsigned long long>(blockIdx.y) * m * n,
               warp, lane);
}

// Built-in "gemm_q6k_rows{R}": R-row Q6_K GEMV.
template <int R>
__global__ void GemmQ6KRowsKernel(const float* a, const unsigned char* w,
                                  float* c, unsigned long long m,
                                  unsigned long long n, unsigned long long k,
                                  unsigned long long split) {
  const unsigned long long warp =
      (static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
       threadIdx.x) /
      32;
  const unsigned int lane = threadIdx.x & 31u;
  if (warp >= n) {
    return;
  }
  const unsigned long long blocks = k / 256;
  const unsigned long long per = blocks / split;
  const unsigned long long b0 =
      static_cast<unsigned long long>(blockIdx.y) * per;
  const unsigned char* wr = w + warp * blocks * 210 + b0 * 210;
  const unsigned int is = lane >> 4;
  float acc[R] = {};
  for (unsigned long long b = 0; b < per; ++b) {
    const unsigned char* base = wr + b * 210;
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
      const unsigned long long tb = (b0 + b) * 256 + n2 * 128 + lane;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float* e = a + static_cast<unsigned long long>(r) * k + tb;
        acc[r] = fmaf(e[0], s0 * q1, acc[r]);
        acc[r] = fmaf(e[32], s1 * q2, acc[r]);
        acc[r] = fmaf(e[64], s2 * q3, acc[r]);
        acc[r] = fmaf(e[96], s3 * q4, acc[r]);
      }
    }
  }
  RowsStore<R>(c, acc, n, static_cast<unsigned long long>(blockIdx.y) * m * n,
               warp, lane);
}

template __global__ void GemmQ4KRowsKernel<2>(
    const float*, const unsigned char*, float*, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
template __global__ void GemmQ4KRowsKernel<3>(
    const float*, const unsigned char*, float*, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
template __global__ void GemmQ4KRowsKernel<4>(
    const float*, const unsigned char*, float*, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
template __global__ void GemmQ5KRowsKernel<2>(
    const float*, const unsigned char*, float*, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
template __global__ void GemmQ5KRowsKernel<3>(
    const float*, const unsigned char*, float*, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
template __global__ void GemmQ5KRowsKernel<4>(
    const float*, const unsigned char*, float*, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
template __global__ void GemmQ6KRowsKernel<2>(
    const float*, const unsigned char*, float*, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
template __global__ void GemmQ6KRowsKernel<3>(
    const float*, const unsigned char*, float*, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
template __global__ void GemmQ6KRowsKernel<4>(
    const float*, const unsigned char*, float*, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);

}  // namespace tessera::backends::rocm
