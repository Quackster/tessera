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

// Built-in "scatter_d2d": up to kScatterSlices device-to-device slices in
// one launch. Buffers: src, dst. Scalars: n (<= kScatterSlices), then n
// triples (src_offset, dst_offset, bytes). Copies 32-bit words, one block
// per (slice, y) pair.
constexpr unsigned kScatterSlices = 5;
constexpr unsigned kScatterWordBytes = 4;

__global__ void ScatterD2DKernel(
    const unsigned char* src, unsigned char* dst, unsigned long long n,
    unsigned long long s0, unsigned long long d0, unsigned long long l0,
    unsigned long long s1, unsigned long long d1, unsigned long long l1,
    unsigned long long s2, unsigned long long d2, unsigned long long l2,
    unsigned long long s3, unsigned long long d3, unsigned long long l3,
    unsigned long long s4, unsigned long long d4, unsigned long long l4) {
  const unsigned long long off[kScatterSlices][3] = {
      {s0, d0, l0}, {s1, d1, l1}, {s2, d2, l2}, {s3, d3, l3}, {s4, d4, l4}};
  const unsigned long long op = blockIdx.x;
  if (op >= n) {
    return;
  }
  const unsigned long long s = off[op][0];
  const unsigned long long d = off[op][1];
  const unsigned long long len = off[op][2];
  const unsigned long long words = len / kScatterWordBytes;
  const unsigned long long tail = len % kScatterWordBytes;
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.y) * blockDim.x;
  for (unsigned long long w = static_cast<unsigned long long>(blockIdx.y) *
                                   blockDim.x +
                               threadIdx.x;
       w < words; w += stride) {
    unsigned int v = 0;
    __builtin_memcpy(&v, src + s + w * kScatterWordBytes, kScatterWordBytes);
    __builtin_memcpy(dst + d + w * kScatterWordBytes, &v, kScatterWordBytes);
  }
  if (blockIdx.y == 0u && threadIdx.x < tail) {
    dst[d + words * kScatterWordBytes + threadIdx.x] =
        src[s + words * kScatterWordBytes + threadIdx.x];
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

// Tile softmax front end shared by the tiled attention kernels (base and
// split, plain and quantized). Thread e owns score sc[e] of the tile; the
// run max takes an order-free fmaxf fold and every thread then exps its
// own score, so the one-thread scan that serialized 256 expf calls
// disappears. Each caller keeps its own summation order.
__device__ void AttentionTileMaxExp(const unsigned long long e,
                                    const float* sc, float* wt, float* red,
                                    float* tile_bc, const float run_max) {
  if (e < 16) {
    float tmax = sc[e * 16];
    for (unsigned long long t = 1; t < 16; ++t) {
      tmax = fmaxf(tmax, sc[e * 16 + t]);
    }
    red[e] = tmax;
  }
  LdsBarrier();
  if (e < 8) {
    red[e] = fmaxf(red[e], red[e + 8]);
  }
  LdsBarrier();
  if (e < 4) {
    red[e] = fmaxf(red[e], red[e + 4]);
  }
  LdsBarrier();
  if (e < 2) {
    red[e] = fmaxf(red[e], red[e + 2]);
  }
  LdsBarrier();
  if (e == 0) {
    tile_bc[0] = fmaxf(run_max, fmaxf(red[0], red[1]));
    tile_bc[1] = expf(run_max - tile_bc[0]);
  }
  LdsBarrier();
  wt[e] = expf(sc[e] - tile_bc[0]);
  LdsBarrier();
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
                                unsigned long long row_base,
                                unsigned long long window,
                                 unsigned long long kv_f16,
                                 unsigned long long causal) {
  // One workgroup per (query row, head); the query/key dot product is
  // computed once per key, so the cost is O(n * head_dim) per query/head.
  constexpr unsigned long long kTile = 256;
  __shared__ float q_s[256];
  __shared__ float sc[256];
  __shared__ float wt[256];
  // Tile reduction scratch plus the broadcast {m_new, corr, l}.
  __shared__ float red[256];
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
  const unsigned long long qb = ((row_base + i) * heads + h) * head_dim;
  if (e < head_dim) {
    q_s[e] = q[qb + e];
  }
  LdsBarrier();
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
    LdsBarrier();
    AttentionTileMaxExp(e, sc, wt, red, tile_bc, run_max);
    if (e == 0) {
      // Four ordered chains keep the sequential partial sums.
      float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
      for (unsigned long long t = 0; t < kTile; t += 4) {
        l0 += wt[t];
        l1 += wt[t + 1];
        l2 += wt[t + 2];
        l3 += wt[t + 3];
      }
      tile_bc[2] = (l0 + l1) + (l2 + l3);
    }
    LdsBarrier();
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
    LdsBarrier();
    if (tile_last >= last) {
      break;
    }
    tile += kTile;
  }
  if (e < head_dim) {
    out[qb + e] = acc / run_sum;
  }
}

// Built-in "attention_split": one workgroup per (query row, head, key
// chunk) runs the tiled online softmax over its chunk and writes the
// unnormalized partial (acc, max, sum); "attention_combine" merges them.
__global__ void AttentionSplitKernel(
    const float* q, const float* k, const float* v, float* pacc, float* pmax,
    float* psum, unsigned long long m, unsigned long long n,
    unsigned long long heads, unsigned long long kv_heads,
    unsigned long long head_dim, unsigned long long q_base,
    unsigned long long window, unsigned long long kv_f16,
    unsigned long long causal, unsigned long long split) {
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
  const unsigned long long total = m * heads * split;
  if (g >= total) {
    return;
  }
  const unsigned long long s_idx = g % split;
  const unsigned long long rest = g / split;
  const unsigned long long e = threadIdx.x;
  const unsigned long long i = rest / heads;
  const unsigned long long h = rest % heads;
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
  const unsigned long long base = rest * split + s_idx;
  // Even chunk of [start, last] for this split (empty chunks stay null).
  const unsigned long long span = last + 1 - start;
  const unsigned long long chunk = (span + split - 1) / split;
  const unsigned long long cstart = start + s_idx * chunk;
  unsigned long long cend = cstart + chunk - 1;
  if (cend > last) {
    cend = last;
  }
  if (cstart > last) {
    if (e < head_dim) {
      pacc[base * head_dim + e] = 0.0f;
    }
    if (e == 0) {
      pmax[base] = -1e30f;
      psum[base] = 0.0f;
    }
    return;
  }
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const unsigned long long qb = (i * heads + h) * head_dim;
  if (e < head_dim) {
    q_s[e] = q[qb + e];
  }
  LdsBarrier();
  float run_max = -1e30f;
  float run_sum = 0.0f;
  float acc = 0.0f;
  unsigned long long tile = cstart;
  while (true) {
    const unsigned long long tile_last =
        (tile + kTile - 1 < cend) ? tile + kTile - 1 : cend;
    float s = -1e30f;
    if (tile + e <= cend) {
      const unsigned long long j = tile + e;
      const unsigned long long kb = (j * kv_heads + kv) * head_dim;
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
    LdsBarrier();
    if (e == 0) {
      float tmax = -1e30f;
      for (unsigned long long t = 0; t < kTile; ++t) {
        tmax = fmaxf(tmax, sc[t]);
      }
      const float m_new = fmaxf(run_max, tmax);
      const float corr = expf(run_max - m_new);
      float l = 0.0f;
      for (unsigned long long t = 0; t < kTile; ++t) {
        const float w = expf(sc[t] - m_new);
        wt[t] = w;
        l += w;
      }
      tile_bc[0] = m_new;
      tile_bc[1] = corr;
      tile_bc[2] = l;
    }
    LdsBarrier();
    const float m_new = tile_bc[0];
    const float corr = tile_bc[1];
    const float l = tile_bc[2];
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a = 0.0f;
    if (e < head_dim) {
      const unsigned long long count =
          (tile + kTile - 1 <= cend) ? kTile : (cend - tile + 1);
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
    LdsBarrier();
    if (tile_last >= cend) {
      break;
    }
    tile += kTile;
  }
  if (e < head_dim) {
    pacc[base * head_dim + e] = acc;
  }
  if (e == 0) {
    pmax[base] = run_max;
    psum[base] = run_sum;
  }
}

// Built-in "attention_split_gqa": flash-decoding variant of
// "attention_split" for one query row where one workgroup serves every
// query head of one kv head. One K row load and one V element load then
// feed heads/kv_heads heads instead of one, which is what makes long
// context decode fast: the KV traffic drops by that factor. Workgroup
// (kv, chunk) writes the same per (query row, head, chunk) partials as
// "attention_split", so "attention_combine" is unchanged. Buffers and
// scalars are those of "attention_split"; m must be 1, head_dim at most
// 256 and heads/kv_heads at most kMaxGqaHeads. The dispatch is
// kv_heads * split workgroups of 256.
__global__ void AttentionSplitGqaKernel(
    const float* q, const float* k, const float* v, float* pacc, float* pmax,
    float* psum, unsigned long long m, unsigned long long n,
    unsigned long long heads, unsigned long long kv_heads,
    unsigned long long head_dim, unsigned long long q_base,
    unsigned long long window, unsigned long long kv_f16,
    unsigned long long causal, unsigned long long split) {
  constexpr unsigned long long kTile = 256;
  constexpr unsigned long long kHeads = kMaxGqaHeads;
  __shared__ float q_s[kHeads * kTile];
  __shared__ float sc[kHeads * kTile];
  __shared__ float red[kHeads * 32];
  // Per-head tile broadcast: {m_new, corr, sum}.
  __shared__ float tile_bc[3 * kHeads];
  const std::uint16_t* kh = reinterpret_cast<const std::uint16_t*>(k);
  const std::uint16_t* vh = reinterpret_cast<const std::uint16_t*>(v);
  const auto kat = [&](unsigned long long idx) {
    return kv_f16 ? Fp16ToFloatDev(kh[idx]) : k[idx];
  };
  const auto vat = [&](unsigned long long idx) {
    return kv_f16 ? Fp16ToFloatDev(vh[idx]) : v[idx];
  };
  const unsigned long long e = threadIdx.x;
  const unsigned long long ratio = kv_heads == 0 ? 0 : heads / kv_heads;
  const unsigned long long kv = blockIdx.x / split;
  const unsigned long long s_idx = blockIdx.x % split;
  if (m != 1 || ratio == 0 || ratio > kHeads || kv >= kv_heads) {
    return;
  }
  const unsigned long long pos = q_base;
  const unsigned long long last =
      causal != 0 ? (pos >= n ? n - 1 : pos) : n - 1;
  unsigned long long start = 0;
  if (causal != 0) {
    start = (window != 0 && pos + 1 > window) ? pos + 1 - window : 0;
    if (start > last) {
      start = last;
    }
  }
  const unsigned long long span = last + 1 - start;
  const unsigned long long chunk = (span + split - 1) / split;
  const unsigned long long cstart = start + s_idx * chunk;
  unsigned long long cend = cstart + chunk - 1;
  if (cend > last) {
    cend = last;
  }
  if (cstart > last) {
#pragma unroll
    for (unsigned long long r = 0; r < kHeads; ++r) {
      if (r < ratio) {
        const unsigned long long base = (kv * ratio + r) * split + s_idx;
        if (e == 0) {
          pmax[base] = -1e30f;
          psum[base] = 0.0f;
        }
        if (e < head_dim) {
          pacc[base * head_dim + e] = 0.0f;
        }
      }
    }
    return;
  }
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  float run_max[kHeads];
  float run_sum[kHeads];
  float acc[kHeads];
#pragma unroll
  for (unsigned long long r = 0; r < kHeads; ++r) {
    if (r < ratio) {
      if (e < head_dim) {
        q_s[r * kTile + e] = q[(kv * ratio + r) * head_dim + e];
      }
      run_max[r] = -1e30f;
      run_sum[r] = 0.0f;
      acc[r] = 0.0f;
    }
  }
  LdsBarrier();
  unsigned long long tile = cstart;
  while (true) {
    const unsigned long long tile_last =
        (tile + kTile - 1 < cend) ? tile + kTile - 1 : cend;
    // Score pass: thread e owns key tile+e of the chunk and reuses its K
    // row for every head of the group.
    float s[kHeads];
#pragma unroll
    for (unsigned long long r = 0; r < kHeads; ++r) {
      if (r < ratio) {
        s[r] = -1e30f;
      }
    }
    if (tile + e <= cend) {
      const unsigned long long kb = ((tile + e) * kv_heads + kv) * head_dim;
      float dot[kHeads];
#pragma unroll
      for (unsigned long long r = 0; r < kHeads; ++r) {
        if (r < ratio) {
          dot[r] = 0.0f;
        }
      }
      for (unsigned long long d = 0; d < head_dim; ++d) {
        const float kd = kat(kb + d);
#pragma unroll
        for (unsigned long long r = 0; r < kHeads; ++r) {
          if (r < ratio) {
            dot[r] = fmaf(q_s[r * kTile + d], kd, dot[r]);
          }
        }
      }
#pragma unroll
      for (unsigned long long r = 0; r < kHeads; ++r) {
        if (r < ratio) {
          s[r] = dot[r] * scale;
        }
      }
    }
#pragma unroll
    for (unsigned long long r = 0; r < kHeads; ++r) {
      if (r < ratio) {
        sc[r * kTile + e] = s[r];
      }
    }
    LdsBarrier();
    // Exact per-head tile max (fmaxf is order free), then a parallel exp
    // and four ordered partial-sum chains per head, as in "attention".
    if (e < ratio * 32) {
      const unsigned long long r = e / 32;
      const unsigned long long c = (e % 32) * 8;
      float tmax = sc[r * kTile + c];
      for (unsigned long long t = 1; t < 8; ++t) {
        tmax = fmaxf(tmax, sc[r * kTile + c + t]);
      }
      red[e] = tmax;
    }
    LdsBarrier();
    if (e < ratio) {
      float tmax = red[e * 32];
      for (unsigned long long t = 1; t < 32; ++t) {
        tmax = fmaxf(tmax, red[e * 32 + t]);
      }
      const float m_new = fmaxf(run_max[e], tmax);
      tile_bc[e] = m_new;
      tile_bc[kHeads + e] = expf(run_max[e] - m_new);
    }
    LdsBarrier();
#pragma unroll
    for (unsigned long long r = 0; r < kHeads; ++r) {
      if (r < ratio) {
        sc[r * kTile + e] = expf(sc[r * kTile + e] - tile_bc[r]);
      }
    }
    LdsBarrier();
    if (e < ratio * 4) {
      const unsigned long long r = e / 4;
      const unsigned long long c = e % 4;
      float chain = 0.0f;
      for (unsigned long long t = c; t < kTile; t += 4) {
        chain += sc[r * kTile + t];
      }
      red[e] = chain;
    }
    LdsBarrier();
    if (e < ratio) {
      tile_bc[2 * kHeads + e] =
          (red[e * 4] + red[e * 4 + 1]) + (red[e * 4 + 2] + red[e * 4 + 3]);
    }
    LdsBarrier();
    // Value pass: thread e owns output element e and reuses each V
    // element for every head of the group.
    float a[kHeads];
#pragma unroll
    for (unsigned long long r = 0; r < kHeads; ++r) {
      if (r < ratio) {
        a[r] = 0.0f;
      }
    }
    if (e < head_dim) {
      const unsigned long long count =
          (tile + kTile - 1 <= cend) ? kTile : (cend - tile + 1);
      for (unsigned long long t = 0; t < count; ++t) {
        const float vv = vat(((tile + t) * kv_heads + kv) * head_dim + e);
#pragma unroll
        for (unsigned long long r = 0; r < kHeads; ++r) {
          if (r < ratio) {
            a[r] = fmaf(sc[r * kTile + t], vv, a[r]);
          }
        }
      }
    }
#pragma unroll
    for (unsigned long long r = 0; r < kHeads; ++r) {
      if (r < ratio) {
        const float corr = tile_bc[kHeads + r];
        acc[r] = fmaf(corr, acc[r], a[r]);
        run_sum[r] = fmaf(corr, run_sum[r], tile_bc[2 * kHeads + r]);
        run_max[r] = tile_bc[r];
      }
    }
    LdsBarrier();
    if (tile_last >= cend) {
      break;
    }
    tile += kTile;
  }
#pragma unroll
  for (unsigned long long r = 0; r < kHeads; ++r) {
    if (r < ratio) {
      const unsigned long long base = (kv * ratio + r) * split + s_idx;
      if (e < head_dim) {
        pacc[base * head_dim + e] = acc[r];
      }
      if (e == 0) {
        pmax[base] = run_max[r];
        psum[base] = run_sum[r];
      }
    }
  }
}

// Built-in "attention_combine": merges one split-N partial set per
// (query row, head) into the normalized output.
__global__ void AttentionCombineKernel(const float* pacc, const float* pmax,
                                       const float* psum, float* out,
                                       unsigned long long m,
                                       unsigned long long heads,
                                       unsigned long long head_dim,
                                       unsigned long long split) {
  const unsigned long long g = blockIdx.x;
  if (g >= m * heads) {
    return;
  }
  const unsigned long long e = threadIdx.x;
  const unsigned long long base = g * split;
  float M = -1e30f;
  for (unsigned long long s = 0; s < split; ++s) {
    M = fmaxf(M, pmax[base + s]);
  }
  if (e < head_dim) {
    float num = 0.0f, den = 0.0f;
    for (unsigned long long s = 0; s < split; ++s) {
      const float w = expf(pmax[base + s] - M);
      num = fmaf(w, pacc[(base + s) * head_dim + e], num);
      den = fmaf(w, psum[base + s], den);
    }
    out[g * head_dim + e] = num / den;
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
    LdsBarrier();
    if (tid < s) {
      red[tid] = fmaxf(red[tid], red[tid + s]);
    }
  }
  LdsBarrier();
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
    acc = fmaf(a_row[t], Bf16ToF32(w_row[t]), acc);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down(acc, offset);
  }
  if (lane == 0) {
    c[warp] = acc;
  }
}

// Built-in "gemm_bf16_rows": the small-batch bf16 GEMV. One warp owns one
// weight column (row_w) and accumulates every one of the m activation
// rows, so the weight column is read once per column instead of once per
// (row, column) as repeated "gemm_bf16" does. This is the shape the
// speculative verifier and the draft block hit (m about 2 to 8), where
// reading the weight matrix once instead of m times is the win. Grid:
// ceil(n / 8) workgroups of 256. m must be 1..16.
constexpr int kGemmBf16RowsMax = 16;

__global__ void GemmBf16RowsKernel(const float* a, const unsigned short* w,
                                   float* c, unsigned long long m,
                                   unsigned long long n,
                                   unsigned long long k) {
  const unsigned long long warp =
      (static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
       threadIdx.x) /
      32;
  const unsigned long long lane = threadIdx.x & 31u;
  if (warp >= n) {
    return;
  }
  const unsigned long long row_w = warp;
  const unsigned long long rows = m < kGemmBf16RowsMax ? m : kGemmBf16RowsMax;
  const unsigned short* w_col = w + row_w * k;
  float acc[kGemmBf16RowsMax];
#pragma unroll
  for (int r = 0; r < kGemmBf16RowsMax; ++r) {
    acc[r] = 0.0f;
  }
  for (unsigned long long t = lane; t < k; t += 32) {
    const float wt = Bf16ToF32(w_col[t]);
#pragma unroll
    for (int r = 0; r < kGemmBf16RowsMax; ++r) {
      if (static_cast<unsigned long long>(r) >= rows) {
        break;
      }
      acc[r] =
          fmaf(a[static_cast<unsigned long long>(r) * k + t], wt, acc[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < kGemmBf16RowsMax; ++r) {
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

// Built-in "gemm_f32_batched": one workgroup covers 32 rows x 8 columns;
// each 128-element k-slice is staged once into shared memory and reused
// across the rows. Four partial sums keep the FMA chain shallow; the tail
// folds into the first. Grid: ceil(m / 32) * ceil(n / 8).
__global__ void GemmF32BatchedKernel(const float* a, const float* w, float* c,
                                     unsigned long long m,
                                     unsigned long long n,
                                     unsigned long long k) {
  const unsigned long long tid = static_cast<unsigned long long>(threadIdx.x);
  const unsigned long long col_tiles = (n + 7) / 8;
  const unsigned long long wg = static_cast<unsigned long long>(blockIdx.x);
  const unsigned long long row_base = (wg / col_tiles) * 32;
  const unsigned long long col_base = (wg % col_tiles) * 8;
  const unsigned long long my_row = tid / 8;
  const unsigned long long my_col = tid % 8;
  const unsigned long long row_a = row_base + my_row;
  const unsigned long long row_w = col_base + my_col;
  const bool active = row_a < m && row_w < n;
  __shared__ float tile[8 * 128];
  float p0 = 0.0f;
  float p1 = 0.0f;
  float p2 = 0.0f;
  float p3 = 0.0f;
  const unsigned long long slices = (k + 127) / 128;
  for (unsigned long long s = 0; s < slices; ++s) {
    const unsigned long long base = s * 128;
    const unsigned long long len = base + 128 <= k ? 128 : k - base;
    for (unsigned long long j = 0; j < 4; ++j) {
      const unsigned long long e = tid * 4 + j;
      const unsigned long long col = e >> 7;
      const unsigned long long el = e & 127;
      const unsigned long long gc = col_base + col;
      if (gc < n && el < len) {
        tile[el * 8 + col] = w[gc * k + base + el];
      }
    }
    LdsBarrier();
    if (active) {
      const float* ab = a + row_a * k + base;
      unsigned long long t = 0;
      for (; t + 4 <= len; t += 4) {
        p0 = fmaf(ab[t], tile[t * 8 + my_col], p0);
        p1 = fmaf(ab[t + 1], tile[(t + 1) * 8 + my_col], p1);
        p2 = fmaf(ab[t + 2], tile[(t + 2) * 8 + my_col], p2);
        p3 = fmaf(ab[t + 3], tile[(t + 3) * 8 + my_col], p3);
      }
      for (; t < len; ++t) {
        p0 = fmaf(ab[t], tile[t * 8 + my_col], p0);
      }
    }
    LdsBarrier();
  }
  if (active) {
    c[row_a * n + row_w] = (p0 + p1) + (p2 + p3);
  }
}

// Built-in "gemm_bf16_batched": as gemm_f32_batched for bf16 weights, with
// the same 32-row x 8-column shared staging and partial-sum order.
__global__ void GemmBf16BatchedKernel(const float* a, const unsigned short* w,
                                      float* c, unsigned long long m,
                                      unsigned long long n,
                                      unsigned long long k) {
  const unsigned long long tid = static_cast<unsigned long long>(threadIdx.x);
  const unsigned long long col_tiles = (n + 7) / 8;
  const unsigned long long wg = static_cast<unsigned long long>(blockIdx.x);
  const unsigned long long row_base = (wg / col_tiles) * 32;
  const unsigned long long col_base = (wg % col_tiles) * 8;
  const unsigned long long my_row = tid / 8;
  const unsigned long long my_col = tid % 8;
  const unsigned long long row_a = row_base + my_row;
  const unsigned long long row_w = col_base + my_col;
  const bool active = row_a < m && row_w < n;
  __shared__ float tile[8 * 128];
  float p0 = 0.0f;
  float p1 = 0.0f;
  float p2 = 0.0f;
  float p3 = 0.0f;
  const unsigned long long slices = (k + 127) / 128;
  for (unsigned long long s = 0; s < slices; ++s) {
    const unsigned long long base = s * 128;
    const unsigned long long len = base + 128 <= k ? 128 : k - base;
    for (unsigned long long j = 0; j < 4; ++j) {
      const unsigned long long e = tid * 4 + j;
      const unsigned long long col = e >> 7;
      const unsigned long long el = e & 127;
      const unsigned long long gc = col_base + col;
      if (gc < n && el < len) {
        tile[el * 8 + col] = Bf16ToF32(w[gc * k + base + el]);
      }
    }
    LdsBarrier();
    if (active) {
      const float* ab = a + row_a * k + base;
      unsigned long long t = 0;
      for (; t + 4 <= len; t += 4) {
        p0 = fmaf(ab[t], tile[t * 8 + my_col], p0);
        p1 = fmaf(ab[t + 1], tile[(t + 1) * 8 + my_col], p1);
        p2 = fmaf(ab[t + 2], tile[(t + 2) * 8 + my_col], p2);
        p3 = fmaf(ab[t + 3], tile[(t + 3) * 8 + my_col], p3);
      }
      for (; t < len; ++t) {
        p0 = fmaf(ab[t], tile[t * 8 + my_col], p0);
      }
    }
    LdsBarrier();
  }
  if (active) {
    c[row_a * n + row_w] = (p0 + p1) + (p2 + p3);
  }
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

// Built-in "gemm_mxfp4_batched": one workgroup covers 32 rows x 32 columns
// over 1024 threads; eight 32-element blocks form one 256-element tile
// that is dequantized once into shared memory and reused across the rows.
// Four partial sums keep the FMA chain shallow. Grid: ceil(m / 32) *
// ceil(n / 32).
__global__ void GemmMxFp4BatchedKernel(const float* a, const unsigned char* w,
                                       float* c, unsigned long long m,
                                       unsigned long long n,
                                       unsigned long long k) {
  const unsigned long long tid = static_cast<unsigned long long>(threadIdx.x);
  const unsigned long long col_tiles = (n + 31) / 32;
  const unsigned long long wg = static_cast<unsigned long long>(blockIdx.x);
  const unsigned long long row_base = (wg / col_tiles) * 32;
  const unsigned long long col_base = (wg % col_tiles) * 32;
  const unsigned long long my_row = tid / 32;
  const unsigned long long my_col = tid % 32;
  const unsigned long long row_a = row_base + my_row;
  const unsigned long long row_w = col_base + my_col;
  const bool active = row_a < m && row_w < n;
  const unsigned long long blocks = k / 32;
  const unsigned char* s = w + (n * k) / 2;
  __shared__ float tile[8 * 32 * 32];
  float p0 = 0.0f;
  float p1 = 0.0f;
  float p2 = 0.0f;
  float p3 = 0.0f;
  const unsigned long long pair = tid >> 2;
  const unsigned long long col = pair & 31;
  const unsigned long long sub0 = (pair >> 5) & 7;
  const unsigned long long lane0 = (tid & 3) * 8;
  const unsigned long long gc = col_base + col;
  const unsigned long long qq = lane0 >> 3;
  for (unsigned long long bb = 0; bb < blocks; bb += 8) {
    const unsigned long long subs =
        blocks - bb < 8 ? blocks - bb : 8;
    if (gc < n && sub0 < subs) {
      const unsigned int* wr =
          reinterpret_cast<const unsigned int*>(w) + (gc * k) / 8;
      const unsigned int word = wr[(bb + sub0) * 4 + qq];
      const float scale =
          E8M0ToFloatDev(s[gc * blocks + bb + sub0]);
      for (unsigned long long jj = 0; jj < 8; ++jj) {
        const unsigned char nib =
            static_cast<unsigned char>((word >> (4 * jj)) & 15u);
        tile[(sub0 * 32 + qq * 8 + jj) * 32 + col] =
            scale * kE2M1Dev[nib];
      }
    }
    LdsBarrier();
    if (active) {
      const float* ab = a + row_a * k + bb * 32;
      for (unsigned long long sub = 0; sub < subs; ++sub) {
        for (unsigned long long jj = 0; jj < 8; ++jj) {
          const unsigned long long tb = (sub * 32 + jj) * 32 + my_col;
          p0 = fmaf(ab[sub * 32 + jj], tile[tb], p0);
          p1 = fmaf(ab[sub * 32 + 8 + jj], tile[tb + 8 * 32], p1);
          p2 = fmaf(ab[sub * 32 + 16 + jj], tile[tb + 16 * 32], p2);
          p3 = fmaf(ab[sub * 32 + 24 + jj], tile[tb + 24 * 32], p3);
        }
      }
    }
    LdsBarrier();
  }
  if (active) {
    c[row_a * n + row_w] = (p0 + p1) + (p2 + p3);
  }
}

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
#if defined(TESSERA_ROCM_WMMA)
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
    LdsBarrier();
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
    LdsBarrier();
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
        dst[e] = F32ToBf16(src[e]);
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
    LdsBarrier();
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
    LdsBarrier();
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
#endif  // defined(TESSERA_ROCM_WMMA)

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
  LdsBarrier();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      partial[tid] += partial[tid + s];
    }
    LdsBarrier();
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
    LdsBarrier();
    if (tid < s) {
      red[tid] += red[tid + s];
    }
  }
  LdsBarrier();
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
    LdsBarrier();
    if (tid < s) {
      red[tid] += red[tid + s];
    }
  }
  LdsBarrier();
  const float mean = red[0] / static_cast<float>(cols);
  const float gain = 1.0f / sqrtf(mean + eps);
  for (unsigned long long c = tid; c < cols; c += blockDim.x) {
    const float g = gate[gbase + base + c];
    const float silu = g / (1.0f + expf(-g));
    y[ybase + base + c] = x[base + c] * gain * w[c] * silu;
  }
}
// Built-in "gdn_out_norm": the qwen4exp gated-delta output norm. One block
// per value head over `dim` fp32, block-reduced: y = o * rsqrt(mean(o^2)
// + eps) * w * sigmoid(z). The gate is a sigmoid, not the SiLU of
// "rmsnorm_gated".
__global__ void GdnOutNormKernel(const float* o, const float* z,
                                 const float* w, float* y,
                                 unsigned long long heads,
                                 unsigned long long dim,
                                 unsigned long long eps_bits) {
  __shared__ float red[256];
  const unsigned long long h = static_cast<unsigned long long>(blockIdx.x);
  if (h >= heads) {
    return;
  }
  const unsigned long long tid = threadIdx.x;
  const unsigned long long base = h * dim;
  float eps = 0.0f;
  static_assert(sizeof(eps) == 4);
  const std::uint32_t bits = static_cast<std::uint32_t>(eps_bits);
  std::memcpy(&eps, &bits, 4);
  float sum = 0.0f;
  for (unsigned long long i = tid; i < dim; i += blockDim.x) {
    sum = fmaf(o[base + i], o[base + i], sum);
  }
  red[tid] = sum;
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    LdsBarrier();
    if (tid < s) {
      red[tid] += red[tid + s];
    }
  }
  LdsBarrier();
  const float inv = rsqrtf(red[0] / static_cast<float>(dim) + eps);
  for (unsigned long long i = tid; i < dim; i += blockDim.x) {
    const float zz = z[base + i];
    const float sig = 1.0f / (1.0f + expf(-zz));
    y[base + i] = o[base + i] * inv * w[i] * sig;
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

// Built-in "moe_gate": one workgroup. Softmaxes the selected router logits
// and computes the shared-expert gate from the normed hidden and the
// shared-expert weight. See rocm_kernels.hpp for the contract.
__global__ void MoeGateKernel(const float* vals, const float* x,
                              const float* w, float* wts, float* sig,
                              unsigned long long top_k,
                              unsigned long long hidden) {
  __shared__ float red[256];
  const unsigned int tid = threadIdx.x;
  const unsigned int stride = blockDim.x;
  float dot = 0.0f;
  for (unsigned long long t = tid; t < hidden; t += stride) {
    dot += x[t] * w[t];
  }
  red[tid] = dot;
  LdsBarrier();
  for (unsigned int s = stride / 2u; s > 0u; s >>= 1u) {
    if (tid < s) {
      red[tid] += red[tid + s];
    }
    LdsBarrier();
  }
  if (tid == 0u) {
    float m = -3.402823466e+38f;
    for (unsigned long long p = 0; p < top_k; ++p) {
      m = fmaxf(m, vals[p]);
    }
    float sum = 0.0f;
    for (unsigned long long p = 0; p < top_k; ++p) {
      sum += expf(vals[p] - m);
    }
    for (unsigned long long p = 0; p < top_k; ++p) {
      wts[p] = expf(vals[p] - m) / sum;
    }
    sig[0] = 1.0f / (1.0f + expf(-red[0]));
  }
}

// Built-in "moe_scale_add": o = a + f[idx] * b, elementwise.
__global__ void MoeScaleAddKernel(const float* a, const float* b,
                                  const float* f, float* o,
                                  unsigned long long n,
                                  unsigned long long idx) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  o[i] = a[i] + f[idx] * b[i];
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

// Built-in "delta_step_heads": one gated-delta step for all value heads.
// Four lanes of one warp own one state column: a lane keeps its dk/4 slice of
// the column in registers for the whole row loop, so the state element is
// loaded once and stored once per call instead of being walked twice per row
// through global memory. The four lane partials of the read and of the
// output fold through warp shuffles. The dispatch keeps the documented
// heads * dv threads: a grid-stride loop gives each thread four column
// slices, and the workgroup size keeps every lane quad inside one warp.
constexpr unsigned long long kDeltaStepLanes = 4;
// State rows per lane the register slice holds: dk up to lanes * this.
constexpr unsigned long long kDeltaStepJptMax = 64;

// One (head, column, lane) item over all rows. `item` indexes lane quads, so
// four consecutive items share a column and their lanes reduce together.
template <unsigned long long kJpt>
__device__ __forceinline__ void DeltaStepHeadsSlice(
    float* s, const float* k, const float* v, const float* q, float* o,
    const float* alpha, const float* beta, unsigned long long heads,
    unsigned long long dk, unsigned long long dv, unsigned long long rows,
    unsigned long long sbase, unsigned long long sstride,
    unsigned long long abase, unsigned long long item) {
  const unsigned long long total = heads * dv * kDeltaStepLanes;
  const bool valid = item < total;
  // An item past the end repeats column 0 so its quad still reduces; its
  // stores are gated by `valid` and its reads stay in bounds.
  const unsigned long long c = valid ? item / kDeltaStepLanes : 0;
  const unsigned long long lane = valid ? item % kDeltaStepLanes : 0;
  const unsigned long long h = c / dv;
  const unsigned long long d = c % dv;
  const unsigned long long jpt = (dk + kDeltaStepLanes - 1) / kDeltaStepLanes;
  const unsigned long long j0 = lane * jpt;
  const unsigned long long jn = (j0 + jpt < dk ? j0 + jpt : dk) - j0;
  const unsigned long long s_base = h * dk * dv;
  // The input state is at sbase and the updated state at sbase + sstride; a
  // zero stride advances in place, a non-zero one appends each row's
  // post-step state to its own history slot (row t reads slot t, writes slot
  // t+1) so a verify can restore any accepted prefix without a per-row
  // device copy.
  const unsigned long long s_base_in = sbase + s_base;
  const unsigned long long k_base = h * dk;
  const unsigned long long v_base = h * dv;
  float st[kJpt];
  // The row's k, then its q: staging the loads keeps them independent of the
  // FMA chain, so a row pays two memory round trips instead of one per element.
  float rq[kJpt];
#pragma unroll
  for (unsigned long long jj = 0; jj < kJpt; ++jj) {
    const unsigned long long j = j0 + jj < dk ? j0 + jj : dk - 1;
    st[jj] = s[s_base_in + j * dv + d];
  }
  // Rows share the recurrent state and advance it in turn, so a whole
  // sequence costs one launch (rows = 1 is the single-token step).
  for (unsigned long long t = 0; t < rows; ++t) {
    const unsigned long long ko = t * heads * dk;
    const unsigned long long vo = t * heads * dv;
    const float a = alpha[abase + t * heads + h];
    const float b = beta[abase + t * heads + h];
    const float vb = v[vo + v_base + d];
#pragma unroll
    for (unsigned long long jj = 0; jj < kJpt; ++jj) {
      const unsigned long long j = j0 + jj < dk ? j0 + jj : dk - 1;
      rq[jj] = k[ko + k_base + j];
    }
    // Four partial sums keep the dk/4-deep read and output chains short.
    float r0 = 0.0f;
    float r1 = 0.0f;
    float r2 = 0.0f;
    float r3 = 0.0f;
#pragma unroll
    for (unsigned long long jj = 0; jj < kJpt; jj += 4) {
      if (jj + 0 < jn) {
        r0 = fmaf(st[jj + 0], rq[jj + 0], r0);
      }
      if (jj + 1 < jn) {
        r1 = fmaf(st[jj + 1], rq[jj + 1], r1);
      }
      if (jj + 2 < jn) {
        r2 = fmaf(st[jj + 2], rq[jj + 2], r2);
      }
      if (jj + 3 < jn) {
        r3 = fmaf(st[jj + 3], rq[jj + 3], r3);
      }
    }
    float read = (r0 + r1) + (r2 + r3);
    read += __shfl_xor_sync(~0ull, read, 1, 32);
    read += __shfl_xor_sync(~0ull, read, 2, 32);
#pragma unroll
    for (unsigned long long jj = 0; jj < kJpt; jj += 4) {
      if (jj + 0 < jn) {
        const float kk = rq[jj + 0];
        st[jj + 0] = a * (st[jj + 0] - b * kk * read) + b * vb * kk;
      }
      if (jj + 1 < jn) {
        const float kk = rq[jj + 1];
        st[jj + 1] = a * (st[jj + 1] - b * kk * read) + b * vb * kk;
      }
      if (jj + 2 < jn) {
        const float kk = rq[jj + 2];
        st[jj + 2] = a * (st[jj + 2] - b * kk * read) + b * vb * kk;
      }
      if (jj + 3 < jn) {
        const float kk = rq[jj + 3];
        st[jj + 3] = a * (st[jj + 3] - b * kk * read) + b * vb * kk;
      }
    }
#pragma unroll
    for (unsigned long long jj = 0; jj < kJpt; ++jj) {
      const unsigned long long j = j0 + jj < dk ? j0 + jj : dk - 1;
      rq[jj] = q[ko + k_base + j];
    }
    float o0 = 0.0f;
    float o1 = 0.0f;
    float o2 = 0.0f;
    float o3 = 0.0f;
#pragma unroll
    for (unsigned long long jj = 0; jj < kJpt; jj += 4) {
      if (jj + 0 < jn) {
        o0 = fmaf(st[jj + 0], rq[jj + 0], o0);
      }
      if (jj + 1 < jn) {
        o1 = fmaf(st[jj + 1], rq[jj + 1], o1);
      }
      if (jj + 2 < jn) {
        o2 = fmaf(st[jj + 2], rq[jj + 2], o2);
      }
      if (jj + 3 < jn) {
        o3 = fmaf(st[jj + 3], rq[jj + 3], o3);
      }
    }
    float out = (o0 + o1) + (o2 + o3);
    out += __shfl_xor_sync(~0ull, out, 1, 32);
    out += __shfl_xor_sync(~0ull, out, 2, 32);
    if (valid && lane == 0) {
      o[t * heads * dv + c] = out;
    }
    if (sstride != 0) {
      // Row t's post-state goes to slot t+1 for a rollback.
#pragma unroll
      for (unsigned long long jj = 0; jj < kJpt; ++jj) {
        if (valid && jj < jn) {
          s[s_base_in + (t + 1) * sstride + (j0 + jj) * dv + d] = st[jj];
        }
      }
    }
  }
  if (sstride == 0) {
    // In-place mode keeps the final state in registers until the last row.
#pragma unroll
    for (unsigned long long jj = 0; jj < kJpt; ++jj) {
      if (valid && jj < jn) {
        s[s_base_in + (j0 + jj) * dv + d] = st[jj];
      }
    }
  }
}

__global__ void DeltaStepHeadsKernel(
    float* s, const float* k, const float* v, const float* q, float* o,
    const float* alpha, const float* beta, unsigned long long heads,
    unsigned long long dk, unsigned long long dv, unsigned long long rows,
    unsigned long long sbase, unsigned long long sstride,
    unsigned long long abase) {
  if (heads == 0 || dk == 0 || dv == 0 || rows == 0) {
    return;
  }
  const unsigned long long jpt = (dk + kDeltaStepLanes - 1) / kDeltaStepLanes;
  if (jpt > kDeltaStepJptMax) {
    // The launcher rejects dk above kDeltaStepLanes * kDeltaStepJptMax; this
    // guard keeps a stale launcher from writing a partial state.
    return;
  }
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long threads =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  const unsigned long long items = heads * dv * kDeltaStepLanes;
  const unsigned long long rounds = (items + threads - 1) / threads;
  if (jpt <= 32) {
    for (unsigned long long round = 0; round < rounds; ++round) {
      DeltaStepHeadsSlice<32>(s, k, v, q, o, alpha, beta, heads, dk, dv, rows,
                              sbase, sstride, abase, idx + round * threads);
    }
    return;
  }
  for (unsigned long long round = 0; round < rounds; ++round) {
    DeltaStepHeadsSlice<kDeltaStepJptMax>(s, k, v, q, o, alpha, beta, heads, dk,
                                          dv, rows, sbase, sstride, abase,
                                          idx + round * threads);
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

// Row-parallel KV quantizer: one workgroup per row. The workgroup reduces
// the row absmax in shared memory and writes the codes in parallel, so a
// single appended row costs a workgroup, not one thread. A one-thread-per-
// row launch left a single lane to walk `cols` serially; at cols 1024 that
// measured 231 us per row and dominated fp8 KV decode (a verify step pays
// 16 full-attention layers times 2 tensors times the block width in rows).
enum KvRowQuantKind { kKvRowQ8 = 0, kKvRowQ4 = 1, kKvRowFp8 = 2 };

template <int Kind>
__device__ void QuantizeKvRowBody(const float* in, unsigned int* packed,
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
    LdsBarrier();
    if (tid < s) {
      red[tid] = fmaxf(red[tid], red[tid + s]);
    }
  }
  LdsBarrier();
  float sc = 1.0f;
  if (Kind == kKvRowFp8) {
    sc = red[0] / 448.0f;
    const float min_s = 1.0f / (448.0f * 512.0f);
    if (sc < min_s) {
      sc = min_s;
    }
  } else if (red[0] > 0.0f) {
    sc = Kind == kKvRowQ4 ? red[0] / 7.0f : red[0] / 127.0f;
  }
  if (tid == 0) {
    scale[r] = sc;
  }
  const unsigned long long words = Kind == kKvRowQ4 ? cols / 8 : cols / 4;
  for (unsigned long long w = tid; w < words; w += blockDim.x) {
    unsigned int word = 0u;
    if (Kind == kKvRowQ4) {
      for (unsigned int b = 0u; b < 8u; ++b) {
        const float x = in[base + w * 8 + b] / sc;
        int q = static_cast<int>(x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f));
        q = q < -7 ? -7 : (q > 7 ? 7 : q);
        word |= (static_cast<unsigned int>(q) & 0xFu) << (b * 4u);
      }
    } else if (Kind == kKvRowQ8) {
      for (unsigned int b = 0u; b < 4u; ++b) {
        const float x = in[base + w * 4 + b] / sc;
        int q = static_cast<int>(x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f));
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        word |= (static_cast<unsigned int>(q) & 0xFFu) << (b * 8u);
      }
    } else {
      for (unsigned int b = 0u; b < 4u; ++b) {
        const float q = fminf(fmaxf(in[base + w * 4 + b] / sc, -448.0f), 448.0f);
        const std::uint8_t bits = Fp8E4M3FromFloatDev(q);
        word |= static_cast<unsigned int>(bits) << (b * 8u);
      }
    }
    packed[r * words + w] = word;
  }
}

// Built-in "quantize_q8": one workgroup per row; symmetric int8 with a
// per-row absmax scale, packed four bytes per word.
__global__ void QuantizeQ8Kernel(const float* in, unsigned int* packed,
                                 float* scale, unsigned long long rows,
                                 unsigned long long cols) {
  QuantizeKvRowBody<kKvRowQ8>(in, packed, scale, rows, cols);
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

// Eight consecutive codes of a quantized KV row, without the row scale: the
// caller folds the scale into the score once per key. One wide load covers
// eight codes, because the per-key byte gather (one 32-byte sector per lane
// per element) is what makes the tiled attention memory bound. head_dim must
// be a multiple of 8 so every key row stays naturally aligned; the caller
// keeps the scalar path otherwise.
template <int Kind>
__device__ __forceinline__ void QuantKvCodes8(const unsigned char* data,
                                              unsigned long long d,
                                              float* out) {
  if (Kind == kQuantKvQ8) {
    const unsigned long long w =
        *reinterpret_cast<const unsigned long long*>(data + d);
#pragma unroll
    for (unsigned long long i = 0; i < 8; ++i) {
      out[i] = static_cast<float>(static_cast<signed char>(w >> (8 * i)));
    }
    return;
  }
  if (Kind == kQuantKvQ4) {
    const unsigned int word =
        reinterpret_cast<const unsigned int*>(data)[d >> 3];
#pragma unroll
    for (unsigned long long i = 0; i < 8; ++i) {
      int value = static_cast<int>((word >> (4 * i)) & 0xFu);
      value = (value & 0x8) != 0 ? value - 16 : value;
      out[i] = static_cast<float>(value);
    }
    return;
  }
  const unsigned long long w =
      *reinterpret_cast<const unsigned long long*>(data + d);
#pragma unroll
  for (unsigned long long i = 0; i < 8; ++i) {
    out[i] = Fp8E4M3ToFloatDev(
        static_cast<unsigned char>(w >> (8 * i)));
  }
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
                                    unsigned long long row_base,
                                    unsigned long long window) {
  constexpr unsigned long long kTile = 256;
  __shared__ float q_s[256];
  __shared__ float sc[256];
  __shared__ float wt[256];
  // Tile reduction scratch plus the broadcast {m_new, corr, l}.
  __shared__ float red[256];
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
  const unsigned long long qb = ((row_base + i) * heads + h) * head_dim;
  if (e < head_dim) {
    q_s[e] = q[qb + e];
  }
  LdsBarrier();
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
      if (head_dim % 8 == 0) {
        // Eight codes per wide load (see QuantKvCodes8).
        float c[8];
        for (; d + 8 <= head_dim; d += 8) {
          QuantKvCodes8<Kind>(k, kb + d, c);
          dot0 = fmaf(q_s[d + 0], c[0], dot0);
          dot1 = fmaf(q_s[d + 1], c[1], dot1);
          dot2 = fmaf(q_s[d + 2], c[2], dot2);
          dot3 = fmaf(q_s[d + 3], c[3], dot3);
          dot0 = fmaf(q_s[d + 4], c[4], dot0);
          dot1 = fmaf(q_s[d + 5], c[5], dot1);
          dot2 = fmaf(q_s[d + 6], c[6], dot2);
          dot3 = fmaf(q_s[d + 7], c[7], dot3);
        }
        float dot = (dot0 + dot1) + (dot2 + dot3);
        dot *= ks[j];
        s = dot * scale;
      } else {
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
    }
    sc[e] = s;
    LdsBarrier();
    AttentionTileMaxExp(e, sc, wt, red, tile_bc, run_max);
    if (e == 0) {
      // Four ordered chains keep the sequential partial sums.
      float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
      for (unsigned long long t = 0; t < kTile; t += 4) {
        l0 += wt[t];
        l1 += wt[t + 1];
        l2 += wt[t + 2];
        l3 += wt[t + 3];
      }
      tile_bc[2] = (l0 + l1) + (l2 + l3);
    }
    LdsBarrier();
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
    LdsBarrier();
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
                                  unsigned long long row_base,
                                  unsigned long long window,
                                  unsigned long long kind) {
  (void)kind;
  AttentionQuantTiled<kQuantKvQ8>(q, k, v, ks, vs, out, m, n, heads, kv_heads,
                                  head_dim, q_base, row_base, window);
}

// Split-N variant of AttentionQuantTiled: one workgroup per (query row,
// head, key chunk) writes the unnormalized partial for its chunk.
template <int Kind>
__device__ void AttentionQuantSplit(const float* q, const unsigned char* k,
                                    const unsigned char* v, const float* ks,
                                    const float* vs, float* pacc, float* pmax,
                                    float* psum, unsigned long long m,
                                    unsigned long long n,
                                    unsigned long long heads,
                                    unsigned long long kv_heads,
                                    unsigned long long head_dim,
                                    unsigned long long q_base,
                                    unsigned long long window,
                                    unsigned long long split) {
  constexpr unsigned long long kTile = 256;
  __shared__ float q_s[256];
  __shared__ float sc[256];
  __shared__ float wt[256];
  // Tile reduction broadcast: thread 0 computes the max/weights once.
  __shared__ float tile_bc[3];
  const unsigned long long g = blockIdx.x;
  if (g >= m * heads * split) {
    return;
  }
  const unsigned long long s_idx = g % split;
  const unsigned long long rest = g / split;
  const unsigned long long e = threadIdx.x;
  const unsigned long long i = rest / heads;
  const unsigned long long h = rest % heads;
  const unsigned long long kv = h / (heads / kv_heads);
  const unsigned long long pos = q_base + i;
  const unsigned long long last = pos >= n ? n - 1 : pos;
  unsigned long long start =
      (window != 0 && pos + 1 > window) ? pos + 1 - window : 0;
  if (start > last) {
    start = last;
  }
  const unsigned long long base = rest * split + s_idx;
  const unsigned long long span = last + 1 - start;
  const unsigned long long chunk = (span + split - 1) / split;
  const unsigned long long cstart = start + s_idx * chunk;
  unsigned long long cend = cstart + chunk - 1;
  if (cend > last) {
    cend = last;
  }
  if (cstart > last) {
    if (e < head_dim) {
      pacc[base * head_dim + e] = 0.0f;
    }
    if (e == 0) {
      pmax[base] = -1e30f;
      psum[base] = 0.0f;
    }
    return;
  }
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const unsigned long long qb = (i * heads + h) * head_dim;
  if (e < head_dim) {
    q_s[e] = q[qb + e];
  }
  LdsBarrier();
  float run_max = -1e30f;
  float run_sum = 0.0f;
  float acc = 0.0f;
  unsigned long long tile = cstart;
  while (true) {
    const unsigned long long tile_last =
        (tile + kTile - 1 < cend) ? tile + kTile - 1 : cend;
    float s = -1e30f;
    if (tile + e <= cend) {
      const unsigned long long j = tile + e;
      const unsigned long long kb = (j * kv_heads + kv) * head_dim;
      float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
      unsigned long long d = 0;
      if (head_dim % 8 == 0) {
        // Eight codes per wide load (see QuantKvCodes8).
        float c[8];
        for (; d + 8 <= head_dim; d += 8) {
          QuantKvCodes8<Kind>(k, kb + d, c);
          dot0 = fmaf(q_s[d + 0], c[0], dot0);
          dot1 = fmaf(q_s[d + 1], c[1], dot1);
          dot2 = fmaf(q_s[d + 2], c[2], dot2);
          dot3 = fmaf(q_s[d + 3], c[3], dot3);
          dot0 = fmaf(q_s[d + 4], c[4], dot0);
          dot1 = fmaf(q_s[d + 5], c[5], dot1);
          dot2 = fmaf(q_s[d + 6], c[6], dot2);
          dot3 = fmaf(q_s[d + 7], c[7], dot3);
        }
        float dot = (dot0 + dot1) + (dot2 + dot3);
        dot *= ks[j];
        s = dot * scale;
      } else {
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
    }
    sc[e] = s;
    LdsBarrier();
    if (e == 0) {
      float tmax = -1e30f;
      for (unsigned long long t = 0; t < kTile; ++t) {
        tmax = fmaxf(tmax, sc[t]);
      }
      const float m_new = fmaxf(run_max, tmax);
      const float corr = expf(run_max - m_new);
      float l = 0.0f;
      for (unsigned long long t = 0; t < kTile; ++t) {
        const float w = expf(sc[t] - m_new);
        wt[t] = w;
        l += w;
      }
      tile_bc[0] = m_new;
      tile_bc[1] = corr;
      tile_bc[2] = l;
    }
    LdsBarrier();
    const float m_new = tile_bc[0];
    const float corr = tile_bc[1];
    const float l = tile_bc[2];
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a = 0.0f;
    if (e < head_dim) {
      const unsigned long long count =
          (tile + kTile - 1 <= cend) ? kTile : (cend - tile + 1);
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
    LdsBarrier();
    if (tile_last >= cend) {
      break;
    }
    tile += kTile;
  }
  if (e < head_dim) {
    pacc[base * head_dim + e] = acc;
  }
  if (e == 0) {
    pmax[base] = run_max;
    psum[base] = run_sum;
  }
}

__global__ void AttentionQ8SplitKernel(
    const float* q, const unsigned char* k, const unsigned char* v,
    const float* ks, const float* vs, float* pacc, float* pmax, float* psum,
    unsigned long long m, unsigned long long n, unsigned long long heads,
    unsigned long long kv_heads, unsigned long long head_dim,
    unsigned long long q_base, unsigned long long window,
    unsigned long long kind, unsigned long long split) {
  (void)kind;
  AttentionQuantSplit<kQuantKvQ8>(q, k, v, ks, vs, pacc, pmax, psum, m, n,
                                  heads, kv_heads, head_dim, q_base, window,
                                  split);
}

// Built-in "quantize_fp8_pack" and "quantize_fp8_pack_rows": one workgroup
// per row; OCP FP8 E4M3 bytes packed four per word with the dynamic
// per-token scale. The W4A8 activation and the fp8 KV cache share this row
// body (the KV merge calls it with one row and a 256-thread workgroup).
__global__ void QuantizeFp8PackRowsKernel(const float* in, unsigned int* packed,
                                          float* scale, unsigned long long rows,
                                          unsigned long long cols) {
  QuantizeKvRowBody<kKvRowFp8>(in, packed, scale, rows, cols);
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
                                   unsigned long long row_base,
                                   unsigned long long window,
                                   unsigned long long kind) {
  (void)kind;
  AttentionQuantTiled<kQuantKvFp8>(q, k, v, ks, vs, out, m, n, heads, kv_heads,
                                   head_dim, q_base, row_base, window);
}

__global__ void AttentionFp8SplitKernel(
    const float* q, const unsigned char* k, const unsigned char* v,
    const float* ks, const float* vs, float* pacc, float* pmax, float* psum,
    unsigned long long m, unsigned long long n, unsigned long long heads,
    unsigned long long kv_heads, unsigned long long head_dim,
    unsigned long long q_base, unsigned long long window,
    unsigned long long kind, unsigned long long split) {
  (void)kind;
  AttentionQuantSplit<kQuantKvFp8>(q, k, v, ks, vs, pacc, pmax, psum, m, n,
                                   heads, kv_heads, head_dim, q_base, window,
                                   split);
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

// Built-in "quantize_q4": one workgroup per row; symmetric 4-bit with a
// per-row absmax scale, packed eight nibbles per word.
__global__ void QuantizeQ4Kernel(const float* in, unsigned int* packed,
                                 float* scale, unsigned long long rows,
                                 unsigned long long cols) {
  QuantizeKvRowBody<kKvRowQ4>(in, packed, scale, rows, cols);
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
                                   unsigned long long row_base,
                                   unsigned long long window,
                                   unsigned long long kind) {
  (void)kind;
  AttentionQuantTiled<kQuantKvQ4>(q, k, v, ks, vs, out, m, n, heads, kv_heads,
                                   head_dim, q_base, row_base, window);
}

__global__ void AttentionQ4SplitKernel(
    const float* q, const unsigned char* k, const unsigned char* v,
    const float* ks, const float* vs, float* pacc, float* pmax, float* psum,
    unsigned long long m, unsigned long long n, unsigned long long heads,
    unsigned long long kv_heads, unsigned long long head_dim,
    unsigned long long q_base, unsigned long long window,
    unsigned long long kind, unsigned long long split) {
  (void)kind;
  AttentionQuantSplit<kQuantKvQ4>(q, k, v, ks, vs, pacc, pmax, psum, m, n,
                                  heads, kv_heads, head_dim, q_base, window,
                                  split);
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

// Built-in "cast_f16_f32": the reverse of cast_f32_f16. n packed fp16
// values (two per unsigned int) become n fp32.
__global__ void CastF16F32Kernel(const unsigned int* in, float* out,
                                 unsigned long long n) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx * 2 >= n) {
    return;
  }
  const std::uint32_t packed = in[idx];
  out[idx * 2] = Fp16ToFloatDev(static_cast<std::uint16_t>(packed & 0xFFFFu));
  out[idx * 2 + 1] = Fp16ToFloatDev(static_cast<std::uint16_t>(packed >> 16));
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
    LdsBarrier();
    for (unsigned long long s = dim / 2; s > 0; s >>= 1) {
      if (tid < s && rv[tid + s] > rv[tid]) {
        rv[tid] = rv[tid + s];
        ri[tid] = ri[tid + s];
        rown[tid] = rown[tid + s];
      }
      LdsBarrier();
    }
    if (tid == 0) {
      vals[r * top_k + p] = rv[0];
      ids[r * top_k + p] = ri[0];
    }
    const unsigned int owner = rown[0];
    LdsBarrier();
    if (tid == owner) {
      ++head;
    }
    LdsBarrier();
  }
}

}  // namespace tessera::backends::rocm
