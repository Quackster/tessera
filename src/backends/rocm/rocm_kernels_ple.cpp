#include "backends/rocm/rocm_kernels.hpp"

// PLE block post-ops: the gate, the per-stream broadcast of the value,
// the dilated causal convolution with the residual add, and the history
// shift. The key and value projections and the three grouped RMS norms
// reuse existing kernels (gemm_bf16, rmsnorm).

namespace tessera::backends::rocm {

// Local block reduction (the hyper-connection file's helper is private
// to its translation unit).
__device__ __forceinline__ float PleBlockSum(float v, float* scratch) {
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down(v, off);
  }
  v = __shfl(v, 0);
  if (lane == 0) {
    scratch[warp] = v;
  }
  __syncthreads();
  const int warps = (static_cast<int>(blockDim.x) + 31) >> 5;
  v = (threadIdx.x < warps) ? scratch[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down(v, off);
    }
    v = __shfl(v, 0);
  }
  if (threadIdx.x == 0) {
    scratch[0] = v;
  }
  __syncthreads();
  return scratch[0];
}

// Built-in "rmsnorm_rows": grouped RMS norm with a per-row gamma. One
// block per row of `cols`, gamma row r at w[r*cols].
__global__ void RmsnormRowsKernel(const float* x, const float* w, float* y,
                                  unsigned long long rows,
                                  unsigned long long cols,
                                  unsigned long long eps_bits) {
  const unsigned long long row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const std::uint32_t bits = static_cast<std::uint32_t>(eps_bits);
  float eps = 0.0f;
  std::memcpy(&eps, &bits, sizeof(eps));
  __shared__ float scratch[8];
  const float* xr = x + row * cols;
  float* yr = y + row * cols;
  const float* wr = w + row * cols;
  float ss = 0.0f;
  for (unsigned long long d = threadIdx.x; d < cols; d += blockDim.x) {
    const float v = xr[d];
    ss += v * v;
  }
  const float mean = PleBlockSum(ss, scratch) / static_cast<float>(cols);
  const float scale = rsqrtf(mean + eps);
  for (unsigned long long d = threadIdx.x; d < cols; d += blockDim.x) {
    yr[d] = scale * xr[d] * wr[d];
  }
}

// Built-in "ple_gate": one block per stream. gate[c] is
// sigmoid(sign(s) * sqrt(max(|s|, 1e-6))) with s the scaled dot of the
// normalized key and query for stream c.
__global__ void PleGateKernel(const float* key, const float* query,
                              float* gate, unsigned long long n_embd,
                              unsigned long long hc) {
  const unsigned long long c = blockIdx.x;
  if (c >= hc) {
    return;
  }
  __shared__ float scratch[8];
  float acc = 0.0f;
  for (unsigned long long d = threadIdx.x; d < n_embd; d += blockDim.x) {
    acc += key[c * n_embd + d] * query[c * n_embd + d];
  }
  const float total = PleBlockSum(acc, scratch);
  if (threadIdx.x == 0) {
    const float value = total / sqrtf(static_cast<float>(n_embd));
    const float mag = sqrtf(fmaxf(fabsf(value), 1e-6f));
    const float sign = (value > 0.0f) - (value < 0.0f);
    gate[c] = 1.0f / (1.0f + __expf(-sign * mag));
  }
}

// Built-in "ple_broadcast": gated[i] = value[i % n_embd] * gate[i / n_embd].
__global__ void PleBroadcastKernel(const float* value, const float* gate,
                                   float* gated, unsigned long long dim,
                                   unsigned long long n_embd) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= dim) {
    return;
  }
  gated[i] = value[i % n_embd] * gate[i / n_embd];
}

// Built-in "ple_conv": the dilated causal convolution (taps from the
// history, the last tap this token's normalized value) plus the residual.
__global__ void PleConvKernel(const float* history, const float* normalized,
                              const unsigned short* conv1d, const float* hidden,
                              const float* gated, float* conv, float* result,
                              unsigned long long dim, unsigned long long hist,
                              unsigned long long taps) {
  const unsigned long long c =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= dim) {
    return;
  }
  const unsigned long long stride = taps - 1;
  float sum = 0.0f;
  for (unsigned long long k = 0; k < taps; ++k) {
    const float x =
        k == taps - 1 ? normalized[c] : history[c * hist + stride * k];
    const float term = x * Fp16ToFloatDev(conv1d[c * taps + k]);
    sum = k == 0 ? term : sum + term;
  }
  const float activation = sum / (1.0f + __expf(-sum));
  conv[c] = activation;
  result[c] = hidden[c] + (gated[c] + activation);
}

// Built-in "ple_hist_advance": shift each channel's history to make room
// and append this token's normalized value.
__global__ void PleHistAdvanceKernel(float* history, const float* normalized,
                                     unsigned long long dim,
                                     unsigned long long hist) {
  const unsigned long long c =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= dim) {
    return;
  }
  for (unsigned long long r = 0; r + 1 < hist; ++r) {
    history[c * hist + r] = history[c * hist + r + 1];
  }
  history[c * hist + hist - 1] = normalized[c];
}

}  // namespace tessera::backends::rocm
