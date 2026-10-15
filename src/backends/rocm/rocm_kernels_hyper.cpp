#include "backends/rocm/rocm_kernels.hpp"

#include <cstdint>
#include <cstring>

// Gated residual stack ("hyper-connection") kernels. The read path is
// gr_norm, gr_down, gr_gate and gr_mean; gr_inject produces the
// per-stream injection; gr_write folds the block output back into the
// stack. The default activation contract rounds the normalized streams
// and the bottleneck to bfloat16 and carries them as f32.

namespace tessera::backends::rocm {

// fp32 -> bfloat16 bits, round to nearest even (matches round_bf16 and
// the core HyperConnectionReadRef).
__device__ __forceinline__ std::uint16_t F32ToBf16Dev(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>(bits >> 16);
}

// bfloat16 bits -> fp32.
__device__ __forceinline__ float Bf16ToF32Dev(std::uint16_t bits) {
  const std::uint32_t value = static_cast<std::uint32_t>(bits) << 16;
  float out = 0.0f;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

// Round an f32 to the nearest bf16 value, carried as f32.
__device__ __forceinline__ float Bf16RoundF32Dev(float value) {
  return Bf16ToF32Dev(F32ToBf16Dev(value));
}

__device__ __forceinline__ float SigmoidDev(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float SiluDev(float x) {
  return x / (1.0f + __expf(-x));
}

// Warp-tree sum, broadcast to every lane.
__device__ __forceinline__ float WarpSumDev(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down(v, off);
  }
  return __shfl(v, 0);
}

// Block-wide fp32 sum through shared memory, broadcast to every thread.
__device__ float BlockSumDev(float v, float* scratch) {
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = WarpSumDev(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  const int warps = (static_cast<int>(blockDim.x) + 31) >> 5;
  v = (threadIdx.x < warps) ? scratch[threadIdx.x] : 0.0f;
  if (warp == 0) v = WarpSumDev(v);
  if (threadIdx.x == 0) scratch[0] = v;
  __syncthreads();
  return scratch[0];
}

// Built-in "gr_norm": per-stream RMSNorm of the residual stack into xn
// and xq, both f32; xq is rounded to bf16. One block per stream.
__global__ void GrNormKernel(const float* r, const float* w_norm, float* xn,
                             float* xq, unsigned long long n_embd,
                             unsigned long long hc,
                             unsigned long long eps_bits) {
  const unsigned long long c = blockIdx.x;
  if (c >= hc) {
    return;
  }
  const std::uint32_t bits = static_cast<std::uint32_t>(eps_bits);
  float eps = 0.0f;
  std::memcpy(&eps, &bits, sizeof(eps));
  __shared__ float scratch[8];
  const float* rc = r + c * n_embd;
  float ss = 0.0f;
  for (unsigned long long d = threadIdx.x; d < n_embd; d += blockDim.x) {
    const float v = rc[d];
    ss += v * v;
  }
  const float mean = BlockSumDev(ss, scratch) / static_cast<float>(n_embd);
  const float rs = rsqrtf(mean + eps);
  for (unsigned long long d = threadIdx.x; d < n_embd; d += blockDim.x) {
    const unsigned long long i = c * n_embd + d;
    const float x = rc[d] * rs * w_norm[i];
    xn[i] = x;
    xq[i] = Bf16RoundF32Dev(x);
  }
}

// Built-in "gr_down": lq = bf16(silu((xq @ w_down.T) / hc)). One block
// per output row, its warps splitting the reduction.
__global__ void GrDownKernel(const float* xq, const unsigned short* w_down,
                             float* lq, unsigned long long hc_dim,
                             unsigned long long hc_lr,
                             unsigned long long hc) {
  const unsigned long long k = blockIdx.x;
  if (k >= hc_lr) {
    return;
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nw = static_cast<int>(blockDim.x >> 5);
  const unsigned short* row = w_down + k * hc_dim;
  float acc = 0.0f;
  for (unsigned long long i = warp * 32 + lane; i < hc_dim;
       i += static_cast<unsigned long long>(nw) * 32) {
    acc += xq[i] * Bf16ToF32Dev(row[i]);
  }
  acc = WarpSumDev(acc);
  __shared__ float part[8];
  if (lane == 0) {
    part[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    float t = (lane < nw) ? part[lane] : 0.0f;
    t = WarpSumDev(t);
    if (lane == 0) {
      lq[k] = Bf16RoundF32Dev(SiluDev(t / static_cast<float>(hc)));
    }
  }
}

// Built-in "gr_gate": gated = xn * sigmoid(lq @ w_up.T). One warp per
// output element, so a warp reads the weight row coalesced (lane k, k+32,
// ...) instead of one strided read per thread.
__global__ void GrGateKernel(const float* lq, const unsigned short* w_up,
                             const float* xn, float* gated,
                             unsigned long long hc_dim,
                             unsigned long long hc_lr) {
  const int lane = threadIdx.x & 31;
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * (blockDim.x >> 5) +
      (threadIdx.x >> 5);
  if (i >= hc_dim) {
    return;
  }
  const unsigned short* row = w_up + i * hc_lr;
  float acc = 0.0f;
  for (unsigned long long k = lane; k < hc_lr; k += 32) {
    acc += lq[k] * Bf16ToF32Dev(row[k]);
  }
  acc = WarpSumDev(acc);
  if (lane == 0) {
    gated[i] = xn[i] * SigmoidDev(acc);
  }
}

// Built-in "gr_mean": mixed[d] = mean over streams of gated.
__global__ void GrMeanKernel(const float* gated, float* mixed,
                             unsigned long long n_embd,
                             unsigned long long hc) {
  const unsigned long long d =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (d >= n_embd) {
    return;
  }
  float m = 0.0f;
  for (unsigned long long c = 0; c < hc; ++c) {
    m += gated[c * n_embd + d];
  }
  mixed[d] = m / static_cast<float>(hc);
}

// Built-in "gr_inject": inject[c] = xq @ w_inject[c]. One block per
// stream, block-reduced.
__global__ void GrInjectKernel(const float* xq,
                               const unsigned short* w_inject, float* inject,
                               unsigned long long hc_dim,
                               unsigned long long hc) {
  const unsigned long long c = blockIdx.x;
  if (c >= hc) {
    return;
  }
  __shared__ float scratch[8];
  const unsigned short* row = w_inject + c * hc_dim;
  float acc = 0.0f;
  for (unsigned long long i = threadIdx.x; i < hc_dim; i += blockDim.x) {
    acc += xq[i] * Bf16ToF32Dev(row[i]);
  }
  const float total = BlockSumDev(acc, scratch);
  if (threadIdx.x == 0) {
    inject[c] = total;
  }
}

// Built-in "gr_write": out[i] = R[i] + block_out[d] * w[c],
// w[c] = 2*sigmoid(inject[c] / hc). `out` may alias `r`.
__global__ void GrWriteKernel(const float* r, const float* block_out,
                              const float* inject, float* out,
                              unsigned long long n_embd,
                              unsigned long long hc) {
  __shared__ float w[32];
  if (threadIdx.x < hc && threadIdx.x < 32) {
    w[threadIdx.x] =
        2.0f * SigmoidDev(inject[threadIdx.x] / static_cast<float>(hc));
  }
  __syncthreads();
  const unsigned long long n = hc * n_embd;
  for (unsigned long long i =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       i < n;
       i += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    const unsigned long long c = i / n_embd;
    const unsigned long long d = i % n_embd;
    out[i] = r[i] + block_out[d] * w[c];
  }
}

}  // namespace tessera::backends::rocm
