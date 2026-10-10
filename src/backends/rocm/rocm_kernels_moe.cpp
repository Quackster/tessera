#include "backends/rocm/rocm_kernels_codec.hpp"
#include "backends/rocm/rocm_kernels.hpp"

// Fused routed-expert kernels for the mixture-of-experts feed-forward. Each
// workgroup covers eight 32-lane GEMVs (warp-per-output). The selected
// expert id is read from device memory, so several experts are computed in
// one launch with no host round-trip and no weight copy. The block decode
// is the shared codec (see rocm_kernels_codec.hpp).

namespace tessera::backends::rocm {

// The warp-per-output gate/up pair: for output j of expert slot, it
// computes silu(G_e x)[j] * (U_e x)[j] with e = ids[slot]. Both gate and
// up are Q4_K [inter, hidden] per expert.
__global__ void MoeExpertsGateUpQ4KKernel(
    const float* x, const unsigned char* gate, const unsigned char* up,
    const unsigned int* ids, float* phi, unsigned long long hidden,
    unsigned long long inter, unsigned long long top_k,
    unsigned long long gate_stride, unsigned long long up_stride) {
  const unsigned long long warps = blockDim.x >> 5;
  const unsigned long long total = inter * top_k;
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * warps + (threadIdx.x >> 5);
  if (idx >= total) {
    return;
  }
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned long long j = idx % inter;
  const unsigned long long slot = idx / inter;
  const unsigned long long expert = ids[slot];
  const unsigned long long blocks = hidden / detail::Q4KCodec::kBlockElements;
  const unsigned char* g_row =
      gate + expert * gate_stride + j * blocks * detail::Q4KCodec::kBlockBytes;
  const unsigned char* u_row =
      up + expert * up_stride + j * blocks * detail::Q4KCodec::kBlockBytes;
  float gate_acc = detail::Q4KCodec::Dot(x, g_row, blocks, lane);
  float up_acc = detail::Q4KCodec::Dot(x, u_row, blocks, lane);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    gate_acc += __shfl_down_sync(~0ull, gate_acc, off, 32);
    up_acc += __shfl_down_sync(~0ull, up_acc, off, 32);
  }
  if (lane == 0u) {
    phi[slot * inter + j] = (gate_acc / (1.0f + expf(-gate_acc))) * up_acc;
  }
}

// The warp-per-output down accumulation: out[j] = sum_slot wts[slot] *
// (D_e phi[slot])[j] with e = ids[slot]. Down is Q4_K [hidden, inter] per
// expert.
__global__ void MoeExpertsDownQ4KKernel(
    const float* phi, const unsigned char* down, const unsigned int* ids,
    const float* wts, float* out, unsigned long long hidden,
    unsigned long long inter, unsigned long long top_k,
    unsigned long long down_stride, unsigned long long accumulate) {
  const unsigned long long warps = blockDim.x >> 5;
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * warps + (threadIdx.x >> 5);
  if (idx >= hidden) {
    return;
  }
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned long long blocks = inter / detail::Q4KCodec::kBlockElements;
  float acc = 0.0f;
  for (unsigned long long slot = 0; slot < top_k; ++slot) {
    const unsigned long long expert = ids[slot];
    const unsigned char* d_row =
        down + expert * down_stride + idx * blocks * detail::Q4KCodec::kBlockBytes;
    float dot = detail::Q4KCodec::Dot(phi + slot * inter, d_row, blocks, lane);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      dot += __shfl_down_sync(~0ull, dot, off, 32);
    }
    acc += wts[slot] * dot;
  }
  if (lane == 0u) {
    out[idx] = accumulate != 0ull ? out[idx] + acc : acc;
  }
}

// The Q6_K down variant: same contract, Q6_K [hidden, inter] per expert.
__global__ void MoeExpertsDownQ6KKernel(
    const float* phi, const unsigned char* down, const unsigned int* ids,
    const float* wts, float* out, unsigned long long hidden,
    unsigned long long inter, unsigned long long top_k,
    unsigned long long down_stride, unsigned long long accumulate) {
  const unsigned long long warps = blockDim.x >> 5;
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * warps + (threadIdx.x >> 5);
  if (idx >= hidden) {
    return;
  }
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned long long blocks = inter / detail::Q6KCodec::kBlockElements;
  float acc = 0.0f;
  for (unsigned long long slot = 0; slot < top_k; ++slot) {
    const unsigned long long expert = ids[slot];
    const unsigned char* d_row =
        down + expert * down_stride + idx * blocks * detail::Q6KCodec::kBlockBytes;
    float dot = detail::Q6KCodec::Dot(phi + slot * inter, d_row, blocks, lane);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      dot += __shfl_down_sync(~0ull, dot, off, 32);
    }
    acc += wts[slot] * dot;
  }
  if (lane == 0u) {
    out[idx] = accumulate != 0ull ? out[idx] + acc : acc;
  }
}

}  // namespace tessera::backends::rocm
