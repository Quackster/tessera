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
                                unsigned long long window,
                                unsigned long long kv_f16) {
  const std::uint16_t* kh = reinterpret_cast<const std::uint16_t*>(k);
  const std::uint16_t* vh = reinterpret_cast<const std::uint16_t*>(v);
  const auto kat = [&](unsigned long long idx) {
    return kv_f16 ? Fp16ToFloatDev(kh[idx]) : k[idx];
  };
  const auto vat = [&](unsigned long long idx) {
    return kv_f16 ? Fp16ToFloatDev(vh[idx]) : v[idx];
  };
  const unsigned long long t =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= m * heads * head_dim) {
    return;
  }
  const unsigned long long i = t / (heads * head_dim);
  const unsigned long long rem = t % (heads * head_dim);
  const unsigned long long h = rem / head_dim;
  const unsigned long long e = rem % head_dim;
  const unsigned long long kv = h / (heads / kv_heads);
  const unsigned long long pos = q_base + i;
  const unsigned long long last = pos >= n ? n - 1 : pos;
  unsigned long long start =
      (window != 0 && pos + 1 > window) ? pos + 1 - window : 0;
  if (start > last) {
    start = last;
  }
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const unsigned long long q_base_idx = (i * heads + h) * head_dim;
  float row_max = 0.0f;
  bool first = true;
  for (unsigned long long j = start; j <= last; ++j) {
    const unsigned long long k_base = (j * kv_heads + kv) * head_dim;
    float dot = 0.0f;
    for (unsigned long long d = 0; d < head_dim; ++d) {
      dot = fmaf(q[q_base_idx + d], kat(k_base + d), dot);
    }
    dot *= scale;
    if (first || dot > row_max) {
      row_max = dot;
      first = false;
    }
  }
  float acc = 0.0f;
  float denom = 0.0f;
  for (unsigned long long j = start; j <= last; ++j) {
    const unsigned long long k_base = (j * kv_heads + kv) * head_dim;
    float dot = 0.0f;
    for (unsigned long long d = 0; d < head_dim; ++d) {
      dot = fmaf(q[q_base_idx + d], kat(k_base + d), dot);
    }
    const float w = expf(dot * scale - row_max);
    denom += w;
    acc = fmaf(w, vat(k_base + e), acc);
  }
  out[t] = acc / denom;
}

// OCP FP8 E4M3 byte to fp32 (device port of the core Fp8E4M3ToFloat).
__host__ __device__ float Fp8E4M3ToFloatDev(std::uint8_t bits) {
  const std::uint32_t sign = bits >> 7;
  const std::uint32_t exp = (bits >> 3) & 0xF;
  const std::uint32_t mant = bits & 0x7;
  float value;
  if (exp == 0) {
    value = static_cast<float>(mant) * 0x1p-10f;
  } else if (exp == 15) {
    if (mant == 7) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    value = (1.0f + static_cast<float>(mant) / 8.0f) * 256.0f;
  } else {
    value = (1.0f + static_cast<float>(mant) / 8.0f) *
            std::ldexp(1.0f, static_cast<int>(exp) - 8);
  }
  return sign != 0 ? -value : value;
}

// OCP MX E8M0 scale byte to fp32 (device port of E8M0ToFloat).
__host__ __device__ float E8M0ToFloatDev(std::uint8_t scale) {
  return static_cast<float>(
      std::ldexp(1.0, static_cast<int>(scale) - 127));
}

// OCP MX E2M1 nibble to fp32 (device port of F4E2M1ToFloat).
__host__ __device__ float F4E2M1ToFloatDev(std::uint8_t nibble) {
  const std::uint32_t sign = (nibble >> 3) & 1;
  const std::uint32_t exp = (nibble >> 1) & 0x3;
  const std::uint32_t mant = nibble & 1;
  float value;
  if (exp == 0) {
    value = static_cast<float>(mant) * 0.5f;
  } else if (exp == 3 && mant == 1) {
    return std::numeric_limits<float>::quiet_NaN();
  } else {
    value = (1.0f + static_cast<float>(mant) / 2.0f) *
            std::ldexp(1.0f, static_cast<int>(exp) - 1);
  }
  return sign != 0 ? -value : value;
}

// bf16 -> fp32 on the device: shift the 16 bits into the high half.
__device__ float Bf16ToFloatDev(unsigned short bits) {
  unsigned int wide = static_cast<unsigned int>(bits) << 16;
  float value;
  memcpy(&value, &wide, sizeof(value));
  return value;
}

// Built-in "gemm_bf16": C = A x W^T with bf16 weights and fp32
// sequential accumulation.
__global__ void GemmBf16Kernel(const float* a, const unsigned short* w,
                               float* c, unsigned long long m,
                               unsigned long long n, unsigned long long k) {
  unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= m * n) {
    return;
  }
  const unsigned long long row_a = idx / n;
  const unsigned long long row_w = idx % n;
  float acc = 0.0f;
  for (unsigned long long t = 0; t < k; ++t) {
    acc = fmaf(a[row_a * k + t], Bf16ToFloatDev(w[row_w * k + t]), acc);
  }
  c[idx] = acc;
}

// Built-in "gemm_f32": C = A x W^T with fp32 sequential accumulation.
__global__ void GemmF32Kernel(const float* a, const float* w, float* c,
                              unsigned long long m, unsigned long long n,
                              unsigned long long k) {
  unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= m * n) {
    return;
  }
  const unsigned long long row_a = idx / n;
  const unsigned long long row_w = idx % n;
  float acc = 0.0f;
  for (unsigned long long t = 0; t < k; ++t) {
    acc = fmaf(a[row_a * k + t], w[row_w * k + t], acc);
  }
  c[idx] = acc;
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
// one E8M0 scale byte per 32 elements. k is a multiple of 32.
__global__ void GemmMxFp4Kernel(const float* a, const unsigned char* w,
                                const unsigned char* s, float* c,
                                unsigned long long m, unsigned long long n,
                                unsigned long long k) {
  unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= m * n) {
    return;
  }
  const unsigned long long row_a = idx / n;
  const unsigned long long row_w = idx % n;
  const unsigned long long blocks = k / 32;
  float acc = 0.0f;
  for (unsigned long long b = 0; b < blocks; ++b) {
    const float scale = E8M0ToFloatDev(s[row_w * blocks + b]);
    for (unsigned long long l = 0; l < 32; ++l) {
      const unsigned long long t = b * 32 + l;
      const std::uint8_t packed = w[(row_w * k + t) / 2];
      const std::uint8_t nibble =
          (t % 2 == 0) ? (packed & 0xF) : (packed >> 4);
      acc = fmaf(a[row_a * k + t], scale * F4E2M1ToFloatDev(nibble), acc);
    }
  }
  c[idx] = acc;
}
// Built-in "rmsnorm": row-wise RMS norm over rows x cols fp32.
// One thread per row with sequential accumulation.
__global__ void RmsnormKernel(const float* x, const float* w, float* y,
                              unsigned long long rows, unsigned long long cols,
                              unsigned long long eps_bits) {
  const unsigned long long r =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) {
    return;
  }
  float eps = 0.0f;
  static_assert(sizeof(eps) == 4);
  std::uint32_t bits = static_cast<std::uint32_t>(eps_bits);
  std::memcpy(&eps, &bits, 4);
  float mean = 0.0f;
  for (unsigned long long c = 0; c < cols; ++c) {
    mean = fmaf(x[r * cols + c], x[r * cols + c], mean);
  }
  mean /= static_cast<float>(cols);
  const float gain = 1.0f / sqrtf(mean + eps);
  for (unsigned long long c = 0; c < cols; ++c) {
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

// Built-in "l2norm": row-wise L2 normalization over rows x cols fp32.
// One thread per row with sequential accumulation.
__global__ void L2NormKernel(const float* x, float* y,
                             unsigned long long rows, unsigned long long cols,
                             unsigned long long eps_bits) {
  const unsigned long long r =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) {
    return;
  }
  float eps = 0.0f;
  static_assert(sizeof(eps) == 4);
  std::uint32_t bits = static_cast<std::uint32_t>(eps_bits);
  std::memcpy(&eps, &bits, 4);
  float sum = 0.0f;
  for (unsigned long long c = 0; c < cols; ++c) {
    sum = fmaf(x[r * cols + c], x[r * cols + c], sum);
  }
  const float gain = 1.0f / sqrtf(sum + eps);
  for (unsigned long long c = 0; c < cols; ++c) {
    y[r * cols + c] = x[r * cols + c] * gain;
  }
}

// Built-in "rmsnorm_gated": row-wise RMS norm scaled by a SiLU gate
// over rows x cols fp32. One thread per row with sequential
// accumulation.
__global__ void RmsnormGatedKernel(const float* x, const float* w,
                                   const float* gate, float* y,
                                   unsigned long long rows,
                                   unsigned long long cols,
                                   unsigned long long eps_bits) {
  const unsigned long long r =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) {
    return;
  }
  float eps = 0.0f;
  static_assert(sizeof(eps) == 4);
  std::uint32_t bits = static_cast<std::uint32_t>(eps_bits);
  std::memcpy(&eps, &bits, 4);
  float mean = 0.0f;
  for (unsigned long long c = 0; c < cols; ++c) {
    mean = fmaf(x[r * cols + c], x[r * cols + c], mean);
  }
  mean /= static_cast<float>(cols);
  const float gain = 1.0f / sqrtf(mean + eps);
  for (unsigned long long c = 0; c < cols; ++c) {
    const float g = gate[r * cols + c];
    const float silu = g / (1.0f + expf(-g));
    y[r * cols + c] = x[r * cols + c] * gain * w[c] * silu;
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
// into queries and gates (per head query then gate). One thread per
// (head, element).
__global__ void QGateSplitKernel(const float* fused, float* q, float* gate,
                                 unsigned long long heads,
                                 unsigned long long head_dim) {
  const unsigned long long t =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= heads * head_dim) {
    return;
  }
  const unsigned long long head = t / head_dim;
  const unsigned long long e = t % head_dim;
  const unsigned long long src = head * 2 * head_dim;
  q[t] = fused[src + e];
  gate[t] = fused[src + head_dim + e];
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

// Built-in "repeat_heads": expand q/k to the value heads. One thread
// per output element.
__global__ void RepeatHeadsKernel(const float* in, float* out,
                                  unsigned long long num_v_heads,
                                  unsigned long long head_k_dim,
                                  unsigned long long factor) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= num_v_heads * head_k_dim) {
    return;
  }
  const unsigned long long h = i / head_k_dim;
  const unsigned long long e = i % head_k_dim;
  out[i] = in[(h / factor) * head_k_dim + e];
}

// Built-in "ssm_gate": the gated-delta decay/write gates per value head.
__global__ void SsmGateKernel(const float* a_log, const float* dt,
                              const float* alpha_raw, const float* beta_raw,
                              float* alpha, float* beta,
                              unsigned long long heads) {
  const unsigned long long h =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (h >= heads) {
    return;
  }
  const float raw = alpha_raw[h] + dt[h];
  const float softplus = raw > 20.0f ? raw : log1pf(expf(raw));
  alpha[h] = expf(-expf(a_log[h]) * softplus);
  beta[h] = 1.0f / (1.0f + expf(-beta_raw[h]));
}

// Built-in "delta_step_heads": one gated-delta step for all value
// heads. One thread per (head, state column).
__global__ void DeltaStepHeadsKernel(
    float* s, const float* k, const float* v, const float* q, float* o,
    const float* alpha, const float* beta, unsigned long long heads,
    unsigned long long dk, unsigned long long dv) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= heads * dv) {
    return;
  }
  const unsigned long long h = idx / dv;
  const unsigned long long d = idx % dv;
  const unsigned long long s_base = h * dk * dv;
  const unsigned long long k_base = h * dk;
  const unsigned long long v_base = h * dv;
  const float a = alpha[h];
  const float b = beta[h];
  float read = 0.0f;
  for (unsigned long long j = 0; j < dk; ++j) {
    read = fmaf(s[s_base + j * dv + d], k[k_base + j], read);
  }
  float out = 0.0f;
  for (unsigned long long j = 0; j < dk; ++j) {
    const float updated =
        a * (s[s_base + j * dv + d] - b * k[k_base + j] * read) +
        b * v[v_base + d] * k[k_base + j];
    s[s_base + j * dv + d] = updated;
    out = fmaf(updated, q[k_base + j], out);
  }
  o[idx] = out;
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

// Built-in "quantize_q8": one thread per row; symmetric int8 with a
// per-row absmax scale, packed four bytes per word.
__global__ void QuantizeQ8Kernel(const float* in, unsigned int* packed,
                                 float* scale, unsigned long long rows,
                                 unsigned long long cols) {
  const unsigned long long r =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) {
    return;
  }
  const unsigned long long base = r * cols;
  float amax = 0.0f;
  for (unsigned long long c = 0; c < cols; ++c) {
    amax = fmaxf(amax, fabsf(in[base + c]));
  }
  const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
  scale[r] = s;
  const unsigned long long words = cols / 4;
  for (unsigned long long w = 0; w < words; ++w) {
    unsigned int word = 0u;
    for (unsigned int b = 0u; b < 4u; ++b) {
      const float x = in[base + w * 4 + b] / s;
      int q = static_cast<int>(x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f));
      q = q < -127 ? -127 : (q > 127 ? 127 : q);
      word |= (static_cast<unsigned int>(q) & 0xFFu) << (b * 8u);
    }
    packed[r * words + w] = word;
  }
}

// Built-in "attention_q8": GQA with symmetric int8 keys/values.
__global__ void AttentionQ8Kernel(const float* q, const unsigned char* k,
                                  const unsigned char* v, const float* ks,
                                  const float* vs, float* out,
                                  unsigned long long m, unsigned long long n,
                                  unsigned long long heads,
                                  unsigned long long kv_heads,
                                  unsigned long long head_dim,
                                  unsigned long long q_base,
                                  unsigned long long window) {
  const unsigned long long t =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= m * heads * head_dim) {
    return;
  }
  const unsigned long long i = t / (heads * head_dim);
  const unsigned long long rem = t % (heads * head_dim);
  const unsigned long long h = rem / head_dim;
  const unsigned long long e = rem % head_dim;
  const unsigned long long kv = h / (heads / kv_heads);
  const unsigned long long pos = q_base + i;
  const unsigned long long last = pos >= n ? n - 1 : pos;
  unsigned long long start =
      (window != 0 && pos + 1 > window) ? pos + 1 - window : 0;
  if (start > last) {
    start = last;
  }
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const unsigned long long q_base_idx = (i * heads + h) * head_dim;
  const auto kat = [&](unsigned long long idx, float s) {
    return static_cast<float>(static_cast<signed char>(k[idx])) * s;
  };
  const auto vat = [&](unsigned long long idx, float s) {
    return static_cast<float>(static_cast<signed char>(v[idx])) * s;
  };
  float row_max = 0.0f;
  bool first = true;
  for (unsigned long long j = start; j <= last; ++j) {
    const unsigned long long k_base = (j * kv_heads + kv) * head_dim;
    float dot = 0.0f;
    for (unsigned long long d = 0; d < head_dim; ++d) {
      dot = fmaf(q[q_base_idx + d], kat(k_base + d, ks[j]), dot);
    }
    dot *= scale;
    if (first || dot > row_max) {
      row_max = dot;
      first = false;
    }
  }
  float acc = 0.0f;
  float denom = 0.0f;
  for (unsigned long long j = start; j <= last; ++j) {
    const unsigned long long k_base = (j * kv_heads + kv) * head_dim;
    float dot = 0.0f;
    for (unsigned long long d = 0; d < head_dim; ++d) {
      dot = fmaf(q[q_base_idx + d], kat(k_base + d, ks[j]), dot);
    }
    const float w = expf(dot * scale - row_max);
    denom += w;
    acc = fmaf(w, vat(k_base + e, vs[j]), acc);
  }
  out[t] = acc / denom;
}

// Built-in "quantize_q4": one thread per row; symmetric 4-bit with a
// per-row absmax scale, packed eight nibbles per word.
__global__ void QuantizeQ4Kernel(const float* in, unsigned int* packed,
                                 float* scale, unsigned long long rows,
                                 unsigned long long cols) {
  const unsigned long long r =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) {
    return;
  }
  const unsigned long long base = r * cols;
  float amax = 0.0f;
  for (unsigned long long c = 0; c < cols; ++c) {
    amax = fmaxf(amax, fabsf(in[base + c]));
  }
  const float s = amax > 0.0f ? amax / 7.0f : 1.0f;
  scale[r] = s;
  const unsigned long long words = cols / 8;
  for (unsigned long long w = 0; w < words; ++w) {
    unsigned int word = 0u;
    for (unsigned int b = 0u; b < 8u; ++b) {
      const float x = in[base + w * 8 + b] / s;
      int q = static_cast<int>(x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f));
      q = q < -7 ? -7 : (q > 7 ? 7 : q);
      word |= (static_cast<unsigned int>(q) & 0xFu) << (b * 4u);
    }
    packed[r * words + w] = word;
  }
}

// Built-in "attention_q4": GQA with symmetric 4-bit keys/values.
__global__ void AttentionQ4Kernel(const float* q, const unsigned int* k,
                                  const unsigned int* v, const float* ks,
                                  const float* vs, float* out,
                                  unsigned long long m, unsigned long long n,
                                  unsigned long long heads,
                                  unsigned long long kv_heads,
                                  unsigned long long head_dim,
                                  unsigned long long q_base,
                                  unsigned long long window) {
  const unsigned long long t =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= m * heads * head_dim) {
    return;
  }
  const unsigned long long i = t / (heads * head_dim);
  const unsigned long long rem = t % (heads * head_dim);
  const unsigned long long h = rem / head_dim;
  const unsigned long long e = rem % head_dim;
  const unsigned long long kv = h / (heads / kv_heads);
  const unsigned long long pos = q_base + i;
  const unsigned long long last = pos >= n ? n - 1 : pos;
  unsigned long long start =
      (window != 0 && pos + 1 > window) ? pos + 1 - window : 0;
  if (start > last) {
    start = last;
  }
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const unsigned long long q_base_idx = (i * heads + h) * head_dim;
  const auto nib = [](unsigned long long idx, unsigned int word, float s) {
    int value = static_cast<int>((word >> ((idx & 7u) << 2u)) & 0xFu);
    value = (value & 0x8) != 0 ? value - 16 : value;
    return static_cast<float>(value) * s;
  };
  float row_max = 0.0f;
  bool first = true;
  for (unsigned long long j = start; j <= last; ++j) {
    const unsigned long long k_base = (j * kv_heads + kv) * head_dim;
    float dot = 0.0f;
    for (unsigned long long d = 0; d < head_dim; ++d) {
      dot = fmaf(q[q_base_idx + d],
                 nib(k_base + d, k[(k_base + d) >> 3], ks[j]), dot);
    }
    dot *= scale;
    if (first || dot > row_max) {
      row_max = dot;
      first = false;
    }
  }
  float acc = 0.0f;
  float denom = 0.0f;
  for (unsigned long long j = start; j <= last; ++j) {
    const unsigned long long k_base = (j * kv_heads + kv) * head_dim;
    float dot = 0.0f;
    for (unsigned long long d = 0; d < head_dim; ++d) {
      dot = fmaf(q[q_base_idx + d],
                 nib(k_base + d, k[(k_base + d) >> 3], ks[j]), dot);
    }
    const float w = expf(dot * scale - row_max);
    denom += w;
    acc = fmaf(w, nib(k_base + e, v[(k_base + e) >> 3], vs[j]), acc);
  }
  out[t] = acc / denom;
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
  out_scores[index] = unary[pos * top_k + p] + dot;
}

}  // namespace tessera::backends::rocm
