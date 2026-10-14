#include "backends/rocm/rocm_kernels.hpp"

#include <hip/hip_fp16.h>

// QSA indexer scoring, paged FP16 KV append and the output gate.

namespace tessera::backends::rocm {

// fp32 -> fp16 bits, round to nearest even (matches core Fp32ToHalfBits).
__device__ __forceinline__ std::uint16_t ToF16Rne(float value) {
  return __half_as_ushort(__float2half_rn(value));
}

// Built-in "qsa_index": score[j] = bias[cb] + sum_h Relu(dot(pooled[cb],
// q_idx[h])), cb = min(j / idx_block, n_bid); incomplete-tail cells get a
// large positive bias so the selection never keeps them ahead of whole
// blocks.
__global__ void QsaIndexKernel(const float* pooled, const float* q_idx,
                               const float* bias, float* scores,
                               unsigned long long idx_n_head,
                               unsigned long long idx_dim,
                               unsigned long long idx_block,
                               unsigned long long n_bid,
                               unsigned long long n_kv) {
  const unsigned long long j =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= n_kv) {
    return;
  }
  unsigned long long cb = j / idx_block;
  if (cb > n_bid) {
    cb = n_bid;
  }
  const float* p = pooled + cb * idx_dim;
  float score = bias[cb];
  for (unsigned long long h = 0; h < idx_n_head; ++h) {
    const float* q = q_idx + h * idx_dim;
    float dot = 0.0f;
    for (unsigned long long d = 0; d < idx_dim; ++d) {
      dot += p[d] * q[d];
    }
    score += dot > 0.0f ? dot : 0.0f;
  }
  if (j >= (n_kv / idx_block) * idx_block) {
    score += 1e9f;
  }
  scores[j] = score;
}

// Built-in "kv_append": one token's K and V into the paged FP16 cache.
// One thread per pair of consecutive head dims (head_dim is even), so the
// two fp16 stores land in one contiguous cell slot.
__global__ void KvAppendKernel(unsigned short* k_pool, unsigned short* v_pool,
                               const int* page_table, const float* kcur,
                               const float* vcur, unsigned long long pos,
                               unsigned long long n_head_kv,
                               unsigned long long head_dim,
                               unsigned long long page_size) {
  const unsigned long long pairs = n_head_kv * head_dim / 2;
  const unsigned long long p =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= pairs) {
    return;
  }
  const unsigned long long e = 2 * p;
  const unsigned long long h = e / head_dim;
  const unsigned long long d = e % head_dim;
  const unsigned long long physical =
      static_cast<unsigned long long>(page_table[pos / page_size]);
  const unsigned long long cell =
      (physical * n_head_kv + h) * page_size + (pos % page_size);
  const unsigned long long off = cell * head_dim + d;
  k_pool[off] = ToF16Rne(kcur[e]);
  k_pool[off + 1] = ToF16Rne(kcur[e + 1]);
  v_pool[off] = ToF16Rne(vcur[e]);
  v_pool[off + 1] = ToF16Rne(vcur[e + 1]);
}

// Built-in "qsa_gate_apply": out[h][d] = fp16(attn[h][d] *
// sigmoid(q_full[h][head_dim + d])). One thread per pair of head dims.
__global__ void QsaGateApplyKernel(const float* attn, const float* q_full,
                                   unsigned short* out,
                                   unsigned long long n_head,
                                   unsigned long long head_dim) {
  const unsigned long long pairs = n_head * head_dim / 2;
  const unsigned long long p =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= pairs) {
    return;
  }
  const unsigned long long e = 2 * p;
  const unsigned long long h = e / head_dim;
  const unsigned long long d = e % head_dim;
  for (unsigned long long j = 0; j < 2; ++j) {
    const unsigned long long idx = h * head_dim + d + j;
    const float gate = q_full[h * 2 * head_dim + head_dim + d + j];
    out[idx] = ToF16Rne(attn[idx] / (1.0f + __expf(-gate)));
  }
}

// Built-in "kv_gather": selected cells through the page table into a
// contiguous fp16 scratch. One thread per output element.
__global__ void KvGatherKernel(const unsigned short* k_pool,
                               const unsigned short* v_pool,
                               const int* page_table, const int* ids,
                               unsigned short* k_scratch,
                               unsigned short* v_scratch,
                               unsigned long long n_ids,
                               unsigned long long n_head_kv,
                               unsigned long long head_dim,
                               unsigned long long page_size) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_ids * n_head_kv * head_dim) {
    return;
  }
  const unsigned long long d = i % head_dim;
  const unsigned long long t = i / head_dim;
  const unsigned long long h = t % n_head_kv;
  const unsigned long long j = t / n_head_kv;
  const unsigned long long id = static_cast<unsigned long long>(ids[j]);
  const unsigned long long physical =
      static_cast<unsigned long long>(page_table[id / page_size]);
  const unsigned long long base =
      (physical * n_head_kv + h) * page_size + (id % page_size);
  k_scratch[i] = k_pool[base * head_dim + d];
  v_scratch[i] = v_pool[base * head_dim + d];
}

// The largest selection the attention works over (idx_top_k + idx_block -
// 1); the softmax stages one score per id in shared memory.
constexpr unsigned long long kQsaMaxIds = 4096;

__device__ __forceinline__ float QsaWarpMax(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_down(v, off));
  }
  return __shfl(v, 0);
}

__device__ __forceinline__ float QsaWarpSum(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down(v, off);
  }
  return __shfl(v, 0);
}

__device__ float QsaBlockMax(float v, float* red) {
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = QsaWarpMax(v);
  if (lane == 0) {
    red[warp] = v;
  }
  __syncthreads();
  const int warps = (static_cast<int>(blockDim.x) + 31) >> 5;
  v = (threadIdx.x < warps) ? red[threadIdx.x] : -3.0e38f;
  if (warp == 0) {
    v = QsaWarpMax(v);
  }
  if (threadIdx.x == 0) {
    red[0] = v;
  }
  __syncthreads();
  return red[0];
}

__device__ float QsaBlockSum(float v, float* red) {
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = QsaWarpSum(v);
  if (lane == 0) {
    red[warp] = v;
  }
  __syncthreads();
  const int warps = (static_cast<int>(blockDim.x) + 31) >> 5;
  v = (threadIdx.x < warps) ? red[threadIdx.x] : 0.0f;
  if (warp == 0) {
    v = QsaWarpSum(v);
  }
  if (threadIdx.x == 0) {
    red[0] = v;
  }
  __syncthreads();
  return red[0];
}

// Built-in "qsa_attend": softmax GQA over the gathered cells. One block
// per query head.
__global__ void QsaAttendKernel(const float* q, const unsigned short* k_scratch,
                                const unsigned short* v_scratch, float* attn,
                                unsigned long long n_ids,
                                unsigned long long n_head,
                                unsigned long long n_head_kv,
                                unsigned long long head_dim) {
  __shared__ float scores[kQsaMaxIds];
  __shared__ float red[8];
  const unsigned long long h = blockIdx.x;
  if (h >= n_head) {
    return;
  }
  const unsigned long long ratio = n_head / n_head_kv;
  const unsigned long long kv = h / ratio;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  if (n_ids == 0) {
    for (unsigned long long d = threadIdx.x; d < head_dim; d += blockDim.x) {
      attn[h * head_dim + d] = 0.0f;
    }
    return;
  }
  if (n_ids > kQsaMaxIds) {
    return;
  }
  for (unsigned long long j = threadIdx.x; j < n_ids; j += blockDim.x) {
    const unsigned short* kk = k_scratch + (j * n_head_kv + kv) * head_dim;
    float dot = 0.0f;
    for (unsigned long long d = 0; d < head_dim; ++d) {
      dot += q[h * head_dim + d] * Fp16ToFloatDev(kk[d]);
    }
    scores[j] = dot * scale;
  }
  __syncthreads();
  float local_max = -3.0e38f;
  for (unsigned long long j = threadIdx.x; j < n_ids; j += blockDim.x) {
    local_max = fmaxf(local_max, scores[j]);
  }
  const float best = QsaBlockMax(local_max, red);
  float local_sum = 0.0f;
  for (unsigned long long j = threadIdx.x; j < n_ids; j += blockDim.x) {
    local_sum += __expf(scores[j] - best);
  }
  const float total = QsaBlockSum(local_sum, red);
  for (unsigned long long j = threadIdx.x; j < n_ids; j += blockDim.x) {
    scores[j] = __expf(scores[j] - best) / total;
  }
  __syncthreads();
  for (unsigned long long d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (unsigned long long j = 0; j < n_ids; ++j) {
      acc += scores[j] *
             Fp16ToFloatDev(v_scratch[(j * n_head_kv + kv) * head_dim + d]);
    }
    attn[h * head_dim + d] = acc;
  }
}

// Built-in "indexer_key_append": append one raw indexer key and, on a
// block completion, pool the block's raw keys, RMS-normalize, and rotate
// the pooled row NeoX-wise at the block's first-cell position. The spare
// slot (row b+1) is rewritten with the constant cell-0 key.
__global__ void IndexerKeyAppendKernel(
    const float* raw, const float* w_k_norm, float* tail,
    float* dead, float* pooled, int* block_pos, const float* cos_tab,
    const float* sin_tab, long long pos_base, unsigned long long eps_bits,
    unsigned long long idx_dim, unsigned long long r, unsigned long long n_rot,
    long long pos_in) {
  __shared__ double s_mean[256];
  std::uint32_t eps_raw = static_cast<std::uint32_t>(eps_bits);
  float eps_f = 0.0f;
  std::memcpy(&eps_f, &eps_raw, sizeof(eps_f));
  const double eps = static_cast<double>(eps_f);
  const int base_i = static_cast<int>(pos_base);
  const unsigned long long d = threadIdx.x;
  const int pos = static_cast<int>(pos_in);
  const unsigned long long slot = static_cast<unsigned long long>(pos) % r;
  const unsigned long long b = static_cast<unsigned long long>(pos) / r;
  if (d < idx_dim && slot < r - 1) {
    tail[slot * idx_dim + d] = raw[d];
  }
  if (pos == 0) {
    for (unsigned long long dd = d; dd < idx_dim; dd += blockDim.x) {
      double ss = 0.0;
      for (unsigned long long i = 0; i < idx_dim; ++i) {
        const double v = raw[i];
        ss += v * v;
      }
      const double inv =
          1.0 / sqrt(ss / static_cast<double>(idx_dim) + static_cast<double>(eps));
      float y = static_cast<float>(static_cast<double>(raw[dd]) * inv *
                                   static_cast<double>(w_k_norm[dd]));
      if (dd < n_rot) {
        y *= cos_tab[dd % (n_rot / 2)];
      }
      dead[dd] = y;
      pooled[dd] = y;
    }
  }
  if (slot != r - 1) {
    return;
  }
  __syncthreads();
  for (unsigned long long dd = d; dd < idx_dim; dd += blockDim.x) {
    double m = 0.0;
    for (unsigned long long j = 0; j + 1 < r; ++j) {
      m += static_cast<double>(tail[j * idx_dim + dd]);
    }
    m += static_cast<double>(raw[dd]);
    s_mean[dd] = m / static_cast<double>(r);
  }
  __syncthreads();
  for (unsigned long long dd = d; dd < idx_dim; dd += blockDim.x) {
    double ss = 0.0;
    for (unsigned long long i = 0; i < idx_dim; ++i) {
      const double v = s_mean[i];
      ss += v * v;
    }
    const double inv =
        1.0 / sqrt(ss / static_cast<double>(idx_dim) + static_cast<double>(eps));
    pooled[b * idx_dim + dd] = static_cast<float>(
        s_mean[dd] * inv * static_cast<double>(w_k_norm[dd]));
    pooled[(b + 1) * idx_dim + dd] = dead[dd];
  }
  if (d == 0) {
    block_pos[0] = base_i + static_cast<int>(b * r);
  }
  __syncthreads();
  const unsigned long long half = n_rot / 2;
  for (unsigned long long dd = d; dd < half; dd += blockDim.x) {
    float* row = pooled + b * idx_dim;
    const unsigned long long toff =
        static_cast<unsigned long long>(base_i + static_cast<int>(b * r)) *
            half +
        dd;
    const float a = row[dd];
    const float c = row[half + dd];
    row[dd] = a * cos_tab[toff] - c * sin_tab[toff];
    row[half + dd] = a * sin_tab[toff] + c * cos_tab[toff];
  }
}

// The largest `width` cells, ascending, ties by ascending index. One
// block; the threshold is found by a 32-bit greedy search on a monotonic
// key, then thread 0 gathers in index order so the result is sorted.
__device__ __forceinline__ unsigned int QsaKeyOf(float value) {
  unsigned int bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ unsigned int QsaBlockSumU(unsigned int v,
                                                     unsigned int* red) {
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down(v, off);
  }
  v = __shfl(v, 0);
  if (lane == 0) {
    red[warp] = v;
  }
  __syncthreads();
  const int warps = (static_cast<int>(blockDim.x) + 31) >> 5;
  v = (threadIdx.x < warps) ? red[threadIdx.x] : 0u;
  if (warp == 0) {
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down(v, off);
    }
    v = __shfl(v, 0);
  }
  if (threadIdx.x == 0) {
    red[0] = v;
  }
  __syncthreads();
  return red[0];
}

// Built-in "qsa_topk": buffers scores (f32, n_kv), ids (i32, width).
// Scalars: n_kv, width.
__global__ void QsaTopKKernel(const float* scores, int* ids,
                              unsigned long long n_kv,
                              unsigned long long width) {
  __shared__ unsigned int red[8];
  if (width == 0) {
    return;
  }
  if (n_kv <= width) {
    for (unsigned long long j = threadIdx.x; j < n_kv; j += blockDim.x) {
      ids[j] = static_cast<int>(j);
    }
    return;
  }
  unsigned int threshold = 0;
  for (int bit = 31; bit >= 0; --bit) {
    const unsigned int candidate = threshold | (1u << bit);
    unsigned int count = 0;
    for (unsigned long long j = threadIdx.x; j < n_kv; j += blockDim.x) {
      if (QsaKeyOf(scores[j]) >= candidate) {
        ++count;
      }
    }
    count = QsaBlockSumU(count, red);
    if (count >= static_cast<unsigned int>(width)) {
      threshold = candidate;
    }
    __syncthreads();
  }
  unsigned int above = 0;
  for (unsigned long long j = threadIdx.x; j < n_kv; j += blockDim.x) {
    if (QsaKeyOf(scores[j]) > threshold) {
      ++above;
    }
  }
  above = QsaBlockSumU(above, red);
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long need = width - static_cast<unsigned long long>(above);
    unsigned long long out = 0;
    for (unsigned long long j = 0; j < n_kv && out < width; ++j) {
      const unsigned int key = QsaKeyOf(scores[j]);
      if (key > threshold) {
        ids[out++] = static_cast<int>(j);
      } else if (key == threshold && need > 0) {
        ids[out++] = static_cast<int>(j);
        --need;
      }
    }
  }
}

}  // namespace tessera::backends::rocm
