#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
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

// y = A(1 x k) times dequant(W) on the device; `weights` is n x k in
// the format the loaded `gemm` kernel expects.
inline std::expected<std::vector<float>, StatusCode> Project(    Backend& backend, const Kernel& gemm, const std::vector<float>& x,
    const Buffer& weights, std::size_t n) {
  const std::size_t k = x.size();
  auto x_buf = backend.AllocateBuffer(k * 4, MemoryKind::Device);
  auto out_buf = backend.AllocateBuffer(n * 4, MemoryKind::Device);
  if (!x_buf || !out_buf) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  auto up = UploadF32(backend, **x_buf, x);
  if (!up) {
    return std::unexpected(up.error());
  }
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((n + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {(*x_buf).get(), &weights, (*out_buf).get()};
  launch.scalars = {1, n, k};
  auto ran = backend.LaunchKernel(gemm, launch);
  if (!ran) {
    return std::unexpected(ran.error());
  }
  backend.Synchronize();
  return DownloadF32(backend, **out_buf);
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

// Causal GQA attention over one query row; the key/value matrices
// upload fresh every step.
inline std::expected<std::vector<float>, StatusCode> Attend(
    Backend& backend, const Kernel& attention, const std::vector<float>& q,
    const std::vector<float>& k, const std::vector<float>& v,
    std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
    std::uint64_t q_base) {
  const std::size_t kv_dim = kv_heads * head_dim;
  const std::size_t n = k.size() / kv_dim;
  auto q_buf = backend.AllocateBuffer(q.size() * 4, MemoryKind::Device);
  auto k_buf = backend.AllocateBuffer(k.size() * 4, MemoryKind::Device);
  auto v_buf = backend.AllocateBuffer(v.size() * 4, MemoryKind::Device);
  auto out_buf =
      backend.AllocateBuffer(heads * head_dim * 4, MemoryKind::Device);
  if (!q_buf || !k_buf || !v_buf || !out_buf) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  if (!UploadF32(backend, **q_buf, q) || !UploadF32(backend, **k_buf, k) ||
      !UploadF32(backend, **v_buf, v)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((heads * head_dim + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {(*q_buf).get(), (*k_buf).get(), (*v_buf).get(),
                    (*out_buf).get()};
  launch.scalars = {1, n, heads, kv_heads, head_dim, q_base};
  auto ran = backend.LaunchKernel(attention, launch);
  if (!ran) {
    return std::unexpected(ran.error());
  }
  backend.Synchronize();
  return DownloadF32(backend, **out_buf);
}

}  // namespace tessera::core::detail
