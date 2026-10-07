#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tessera/backend.hpp"
#include "tessera/model.hpp"
#include "tessera/types.hpp"

#include "core/numerics/quant.hpp"

// Host-side decode helpers shared by the vanilla and hybrid paths
// (single implementation, not one copy per path; AGENTS.md rule 2).
// They move host vectors through the generic device kernels, so the
// glue stays identical wherever a given op is used.

namespace tessera::core::detail {

// A required weight: missing is MalformedFile, a wrong layout is
// UnsupportedFeature.
[[nodiscard]] inline std::expected<const DeviceTensor*, StatusCode>
NeedWeight(const Model& model, std::string_view name, DType dtype) {
  for (const auto& weight : model.Weights()) {
    if (weight.manifest.name == name) {
      if (weight.manifest.dtype != dtype) {
        return std::unexpected(StatusCode::UnsupportedFeature);
      }
      return &weight;
    }
  }
  return std::unexpected(StatusCode::MalformedFile);
}

// Like NeedWeight but accepts any dtype (the caller selects the kernel
// from the weight's format). Missing is MalformedFile.
[[nodiscard]] inline std::expected<const DeviceTensor*, StatusCode>
NeedWeightAny(const Model& model, std::string_view name) {
  for (const auto& weight : model.Weights()) {
    if (weight.manifest.name == name) {
      return &weight;
    }
  }
  return std::unexpected(StatusCode::MalformedFile);
}

// Row-wise RMS norm on the host (norms are cheap vectors); the decode
// loop uses this for x, QK-norm and the gated norm's RMS part while
// their device kernels cover the hot paths.
inline void RmsNormInto(const std::vector<float>& x, const float* w,
                        double eps, std::vector<float>& out) {
  float mean = 0.0f;
  for (float v : x) {
    mean += v * v;
  }
  mean /= static_cast<float>(x.size());
  const float gain = 1.0f / std::sqrt(mean + static_cast<float>(eps));
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = x[i] * gain * w[i];
  }
}

inline void SiluMulInto(const std::vector<float>& gate,
                        const std::vector<float>& up,
                        std::vector<float>& out) {
  for (std::size_t i = 0; i < gate.size(); ++i) {
    const float silu = gate[i] / (1.0f + std::exp(-gate[i]));
    out[i] = silu * up[i];
  }
}

inline std::expected<std::vector<float>, StatusCode> DownloadF32(
    Backend& backend, const Buffer& buf) {
  std::vector<std::byte> raw(buf.Size());
  auto down = backend.CopyD2H(buf, raw.data(), raw.size());
  if (!down) {
    return std::unexpected(down.error());
  }
  std::vector<float> out(buf.Size() / 4);
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

// Append one key/value row to a preallocated KV cache. Works for both KV
// cache types (DeviceDecodeState::Kv and HybridDecodeCache::FullKv) so the
// append logic has one implementation. The buffer grows geometrically, so
// a decode step copies one row instead of reallocating the whole cache.
constexpr std::size_t kInitialKvRows = 4;

template <typename Kv>
inline std::expected<void, StatusCode> AppendKv(Backend& backend, Kv& kv,
                                                const Buffer& k_row,
                                                const Buffer& v_row,
                                                std::size_t kv_bytes) {
  if (kv.rows == kv.capacity) {
    const std::size_t new_capacity =
        kv.capacity == 0 ? kInitialKvRows : kv.capacity * 2;
    auto grown_k = backend.AllocateBuffer(new_capacity * kv_bytes,
                                          MemoryKind::Device);
    auto grown_v = backend.AllocateBuffer(new_capacity * kv_bytes,
                                          MemoryKind::Device);
    if (!grown_k || !grown_v) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    if (kv.rows > 0) {
      if (!backend.CopyD2D(*kv.k, 0, **grown_k, 0, kv.rows * kv_bytes) ||
          !backend.CopyD2D(*kv.v, 0, **grown_v, 0, kv.rows * kv_bytes)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
    kv.k = std::move(*grown_k);
    kv.v = std::move(*grown_v);
    kv.capacity = new_capacity;
  }
  if (!backend.CopyD2D(k_row, 0, *kv.k, kv.rows * kv_bytes, kv_bytes) ||
      !backend.CopyD2D(v_row, 0, *kv.v, kv.rows * kv_bytes, kv_bytes)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  ++kv.rows;
  return {};
}

// Greedy argmax over a logits row (first maximum wins, matching the
// decode loops). Empty input is undefined; callers pass a nonempty row.
inline std::uint32_t ArgMax(std::span<const float> logits) {
  std::uint32_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) {
      best = static_cast<std::uint32_t>(i);
    }
  }
  return best;
}

// Download an F32 buffer once and cache it by name (constant weights:
// norm vectors, SSM scalars). Saves a per-step host copy.
inline std::expected<const std::vector<float>*, StatusCode> DownloadF32Cached(
    Backend& backend,
    std::unordered_map<std::string, std::vector<float>>& cache,
    const std::string& name, const Buffer& buf) {
  auto it = cache.find(name);
  if (it != cache.end()) {
    return &it->second;
  }
  auto value = DownloadF32(backend, buf);
  if (!value) {
    return std::unexpected(value.error());
  }
  auto inserted = cache.emplace(name, std::move(*value));
  return &inserted.first->second;
}

inline std::expected<void, StatusCode> UploadF32(Backend& backend,
                                                 Buffer& buf,
                                                 const std::vector<float>& x) {
  if (x.size() * 4 > buf.Size()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return backend.CopyH2D(
      buf, std::span<const std::byte>(
               reinterpret_cast<const std::byte*>(x.data()), x.size() * 4));
}

// Read token row `token` of an embedding tensor (F32 or a supported
// quant format) into x (hidden floats); the row is dequantized on the
// host when the tensor is quantized.
inline std::expected<void, StatusCode> GatherEmbedding(
    Backend& backend, const DeviceTensor& embed, std::size_t token,
    std::size_t hidden, std::vector<float>& x) {
  if (embed.manifest.dtype == DType::F32) {
    auto row = backend.CopyD2HAt(*embed.device, token * hidden * 4,
                                 reinterpret_cast<std::byte*>(x.data()),
                                 hidden * 4);
    if (!row) {
      return std::unexpected(row.error());
    }
    return {};
  }
  const auto layout = BlockLayout(embed.manifest.dtype);
  if (!layout) {
    return std::unexpected(layout.error());
  }
  if (layout->elements == 0 || hidden % layout->elements != 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const std::size_t row_bytes = hidden / layout->elements * layout->bytes;
  std::vector<std::byte> raw(row_bytes);
  auto row = backend.CopyD2HAt(*embed.device, token * row_bytes, raw.data(),
                               row_bytes);
  if (!row) {
    return std::unexpected(row.error());
  }
  return DequantizeBlocks(embed.manifest.dtype,
                          std::span<const std::byte>(raw),
                          std::span<float>(x));
}

// Device-to-device launch helpers (no host copies): keep activations on
// the device and chain kernels. The device-resident block forward uses
// these instead of Project (which round-trips through the host).

// C = A (m x k) times dequant(W) with A and C on the device; scalars
// follow the shared (m, n, k) order.
inline std::expected<void, StatusCode> ProjectDevice(
    Backend& backend, const Kernel& gemm, const Buffer& a, const Buffer& w,
    Buffer& c, std::size_t m, std::size_t n, std::size_t k) {
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((m * n + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&a, &w, &c};
  launch.scalars = {m, n, k};
  return backend.LaunchKernel(gemm, launch);
}

// Y = rmsnorm(X, W) with X, W, Y on the device (rows x cols).
inline std::expected<void, StatusCode> RmsNormDevice(
    Backend& backend, const Kernel& kernel, const Buffer& x, const Buffer& w,
    Buffer& y, std::size_t rows, std::size_t cols, double eps) {
  const float eps_f = static_cast<float>(eps);
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(eps_f));
  std::memcpy(&bits, &eps_f, sizeof(bits));
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((rows + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&x, &w, &y};
  launch.scalars = {rows, cols, bits};
  return backend.LaunchKernel(kernel, launch);
}

// O = A + B (residual) on the device.
inline std::expected<void, StatusCode> AddDevice(Backend& backend,
                                                 const Kernel& kernel,
                                                 const Buffer& a,
                                                 const Buffer& b, Buffer& o,
                                                 std::size_t n) {
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((n + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&a, &b, &o};
  launch.scalars = {n};
  return backend.LaunchKernel(kernel, launch);
}

// O = silu(G) * U (gated MLP) on the device.
inline std::expected<void, StatusCode> SiluMulDevice(
    Backend& backend, const Kernel& kernel, const Buffer& gate,
    const Buffer& up, Buffer& o, std::size_t n) {
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((n + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&gate, &up, &o};
  launch.scalars = {n};
  return backend.LaunchKernel(kernel, launch);
}

// O = A * sigmoid(G) on the device.
inline std::expected<void, StatusCode> SigmoidGateDevice(
    Backend& backend, const Kernel& kernel, const Buffer& a, const Buffer& g,
    Buffer& o, std::size_t n) {
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((n + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&a, &g, &o};
  launch.scalars = {n};
  return backend.LaunchKernel(kernel, launch);
}

// Split a fused gated-attention projection into queries and gates.
inline std::expected<void, StatusCode> QGateSplitDevice(
    Backend& backend, const Kernel& kernel, const Buffer& fused, Buffer& q,
    Buffer& gate, std::size_t heads, std::size_t head_dim) {
  KernelLaunch launch;
  launch.grid_x =
      static_cast<std::uint32_t>((heads * head_dim + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&fused, &q, &gate};
  launch.scalars = {heads, head_dim};
  return backend.LaunchKernel(kernel, launch);
}

// In-place multimodal RoPE with per-row (t, h, w) position triples.
inline std::expected<void, StatusCode> MropeDevice(
    Backend& backend, const Kernel& kernel, Buffer& io, const Buffer& pos,
    std::size_t rows, std::size_t heads, std::size_t head_dim,
    std::size_t rope_dim, std::size_t sec_t, std::size_t sec_h,
    std::size_t sec_w, double theta) {
  const float theta_f = static_cast<float>(theta);
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(theta_f));
  std::memcpy(&bits, &theta_f, sizeof(bits));
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>(
      (rows * heads * (rope_dim / 2) + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&io, &pos};
  launch.scalars = {rows, heads, head_dim, rope_dim, bits, sec_t, sec_h, sec_w};
  return backend.LaunchKernel(kernel, launch);
}

// Y = L2-normalize(X) per row on the device.
inline std::expected<void, StatusCode> L2NormDevice(
    Backend& backend, const Kernel& kernel, const Buffer& x, Buffer& y,
    std::size_t rows, std::size_t cols, double eps) {
  const float eps_f = static_cast<float>(eps);
  std::uint32_t bits = 0;
  std::memcpy(&bits, &eps_f, sizeof(bits));
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((rows + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&x, &y};
  launch.scalars = {rows, cols, bits};
  return backend.LaunchKernel(kernel, launch);
}

// Y = rmsnorm(X, W) * silu(gate) per row on the device.
inline std::expected<void, StatusCode> RmsNormGatedDevice(
    Backend& backend, const Kernel& kernel, const Buffer& x, const Buffer& w,
    const Buffer& gate, Buffer& y, std::size_t rows, std::size_t cols,
    double eps) {
  const float eps_f = static_cast<float>(eps);
  std::uint32_t bits = 0;
  std::memcpy(&bits, &eps_f, sizeof(bits));
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((rows + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&x, &w, &gate, &y};
  launch.scalars = {rows, cols, bits};
  return backend.LaunchKernel(kernel, launch);
}

// Y = causal depthwise conv1d(X, W) on the device.
inline std::expected<void, StatusCode> Conv1dDevice(
    Backend& backend, const Kernel& kernel, const Buffer& x, const Buffer& w,
    Buffer& y, std::size_t channels, std::size_t length, std::size_t width) {
  KernelLaunch launch;
  launch.grid_x =
      static_cast<std::uint32_t>((channels * length + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&x, &w, &y};
  launch.scalars = {channels, length, width};
  return backend.LaunchKernel(kernel, launch);
}

// One gated-delta step updating the device state in place.
inline std::expected<void, StatusCode> DeltaStepDevice(
    Backend& backend, const Kernel& kernel, Buffer& state, const Buffer& k,
    const Buffer& v, const Buffer& q, Buffer& o, std::size_t dk,
    std::size_t dv, float alpha, float beta) {
  std::uint32_t alpha_bits = 0;
  std::uint32_t beta_bits = 0;
  std::memcpy(&alpha_bits, &alpha, 4);
  std::memcpy(&beta_bits, &beta, 4);
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((dv + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&state, &k, &v, &q, &o};
  launch.scalars = {dk, dv, alpha_bits, beta_bits};
  return backend.LaunchKernel(kernel, launch);
}

// One gated-delta step for all value heads with the device state updated// in place (`k`/`q` are heads x dk, `v`/`o` heads x dv, `alpha`/`beta`
// one value per head).
inline std::expected<void, StatusCode> DeltaStepHeadsDevice(
    Backend& backend, const Kernel& kernel, Buffer& state, const Buffer& k,
    const Buffer& v, const Buffer& q, Buffer& o, const Buffer& alpha,
    const Buffer& beta, std::size_t heads, std::size_t dk, std::size_t dv) {
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((heads * dv + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&state, &k, &v, &q, &o, &alpha, &beta};
  launch.scalars = {heads, dk, dv};
  return backend.LaunchKernel(kernel, launch);
}

// Causal GQA attention over `n` key/value rows on the device.
inline std::expected<void, StatusCode> AttentionDevice(
    Backend& backend, const Kernel& kernel, const Buffer& q, const Buffer& k,
    const Buffer& v, Buffer& out, std::size_t n, std::size_t heads,
    std::size_t kv_heads, std::size_t head_dim, std::uint64_t q_base,
    std::uint64_t window = 0) {
  KernelLaunch launch;
  launch.grid_x =
      static_cast<std::uint32_t>((heads * head_dim + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&q, &k, &v, &out};
  launch.scalars = {1, n, heads, kv_heads, head_dim, q_base, window};
  return backend.LaunchKernel(kernel, launch);
}

// In-place NeoX RoPE over rows x heads x head_dim on the device.
inline std::expected<void, StatusCode> RopeDevice(
    Backend& backend, const Kernel& kernel, Buffer& io, std::size_t heads,
    std::size_t head_dim, std::size_t rope_dim, std::uint64_t pos_base,
    double theta) {
  const float theta_f = static_cast<float>(theta);
  std::uint32_t bits = 0;
  std::memcpy(&bits, &theta_f, sizeof(bits));
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>(
      (heads * (rope_dim / 2) + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&io};
  launch.scalars = {1, heads, head_dim, rope_dim, pos_base, bits};
  return backend.LaunchKernel(kernel, launch);
}

// Gated-delta decay/write gates per value head on the device.
inline std::expected<void, StatusCode> SsmGateDevice(
    Backend& backend, const Kernel& kernel, const Buffer& a_log,
    const Buffer& dt, const Buffer& alpha_raw, const Buffer& beta_raw,
    Buffer& alpha, Buffer& beta, std::size_t heads) {
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((heads + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&a_log, &dt, &alpha_raw, &beta_raw, &alpha, &beta};
  launch.scalars = {heads};
  return backend.LaunchKernel(kernel, launch);
}

// Expand q/k head vectors to the value heads on the device.
inline std::expected<void, StatusCode> RepeatHeadsDevice(
    Backend& backend, const Kernel& kernel, const Buffer& in, Buffer& out,
    std::size_t num_v_heads, std::size_t head_k_dim, std::size_t factor) {
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>(
      (num_v_heads * head_k_dim + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&in, &out};
  launch.scalars = {num_v_heads, head_k_dim, factor};
  return backend.LaunchKernel(kernel, launch);
}

// GEMM kernel id for a projection dtype (the formats the GGUF targets// use). Empty when the dtype has no GEMM kernel.
inline std::string_view GemmKernelName(DType dtype) {
  switch (dtype) {
    case DType::Q4K: return "gemm_q4k";
    case DType::Q5K: return "gemm_q5k";
    case DType::Q6K: return "gemm_q6k";
    case DType::Q3K: return "gemm_q3k";
    case DType::Q80: return "gemm_q80";
    case DType::IQ4_NL: return "gemm_iq4nl";
    case DType::IQ4_XS: return "gemm_iq4xs";
    case DType::IQ3_S: return "gemm_iq3s";
    case DType::F32: return "gemm_f32";
    case DType::BF16: return "gemm_bf16";
    default: return {};
  }
}

// Load (once, into `cache`) and return the GEMM kernel for a projection
// dtype.
inline std::expected<Kernel*, StatusCode> GemmFor(
    Backend& backend, std::unordered_map<int, std::unique_ptr<Kernel>>& cache,
    DType dtype) {
  auto it = cache.find(static_cast<int>(dtype));
  if (it != cache.end()) {
    return it->second.get();
  }
  const std::string_view name = GemmKernelName(dtype);
  if (name.empty()) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  auto kernel = backend.LoadKernel(name, {});
  if (!kernel) {
    return std::unexpected(kernel.error());
  }
  Kernel* raw = kernel->get();
  cache.emplace(static_cast<int>(dtype), std::move(*kernel));
  return raw;
}

}  // namespace tessera::core::detail
