#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <string_view>
#include <vector>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"

// Device-op glue for the hybrid decode paths: kernel selection by
// weight dtype, mRoPE, QK-Norm, causal conv1d and the gated-delta scan
// step. Shared by the attention and recurrent mixers.

namespace tessera::core::detail {

// Gemm kernel id for a projection dtype (the formats the GGUF target
// uses). Empty string when the dtype has no GEMM kernel.
std::string_view GemmKernelName(DType dtype) {
  switch (dtype) {
    case DType::Q4K: return "gemm_q4k";
    case DType::Q5K: return "gemm_q5k";
    case DType::Q6K: return "gemm_q6k";
    case DType::Q3K: return "gemm_q3k";
    case DType::Q80: return "gemm_q80";
    case DType::IQ4_NL: return "gemm_iq4nl";
    case DType::IQ4_XS: return "gemm_iq4xs";
    case DType::IQ3_S: return "gemm_iq3s";
    default: return {};
  }
}

std::expected<Kernel*, StatusCode> GemmFor(Backend& backend,
                                           HybridDecodeCache& cache,
                                           DType dtype) {
  auto it = cache.gemms.find(static_cast<int>(dtype));
  if (it != cache.gemms.end()) {
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
  cache.gemms.emplace(static_cast<int>(dtype), std::move(*kernel));
  return raw;
}

// In-place mRoPE over rows x heads x head_dim with per-row (t, h, w)
// triples, on the device.
std::expected<void, StatusCode> RunMrope(
    Backend& backend, const Kernel& mrope, std::vector<float>& io,
    const std::vector<std::uint64_t>& pos, std::size_t rows, std::size_t heads,
    std::size_t head_dim, std::size_t rope_dim, std::size_t sec_t,
    std::size_t sec_h, std::size_t sec_w, double theta) {
  auto io_buf = backend.AllocateBuffer(io.size() * 4, MemoryKind::Device);
  auto pos_buf = backend.AllocateBuffer(pos.size() * 8, MemoryKind::Device);
  if (!io_buf || !pos_buf) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  if (!UploadF32(backend, **io_buf, io) ||
      !backend.CopyH2D(**pos_buf,
                       std::span<const std::byte>(
                           reinterpret_cast<const std::byte*>(pos.data()),
                           pos.size() * 8))) {
    return std::unexpected(StatusCode::DeviceError);
  }
  const float theta_f = static_cast<float>(theta);
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(theta_f));
  std::memcpy(&bits, &theta_f, sizeof(bits));
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>(
      (rows * heads * (rope_dim / 2) + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {(*io_buf).get(), (*pos_buf).get()};
  launch.scalars = {rows, heads, head_dim, rope_dim, bits,
                    sec_t, sec_h, sec_w};
  auto ran = backend.LaunchKernel(mrope, launch);
  if (!ran) {
    return std::unexpected(ran.error());
  }
  backend.Synchronize();
  auto down = DownloadF32(backend, **io_buf);
  if (!down) {
    return std::unexpected(down.error());
  }
  io = std::move(*down);
  return {};
}

// QK-Norm: RMSNorm over head_dim applied per head (host, ascending).
void NormHeads(std::vector<float>& x, std::size_t heads, std::size_t head_dim,
               const std::vector<float>& w, double eps) {
  std::vector<float> slice(head_dim);
  std::vector<float> out(head_dim);
  for (std::size_t h = 0; h < heads; ++h) {
    std::memcpy(slice.data(), x.data() + h * head_dim, head_dim * 4);
    RmsNormInto(slice, w.data(), eps, out);
    std::memcpy(x.data() + h * head_dim, out.data(), head_dim * 4);
  }
}

// Depthwise causal conv1d over X (channels x length) with device
// weights W (channels x width); returns the full Y.
std::expected<std::vector<float>, StatusCode> RunConv1d(
    Backend& backend, const Kernel& conv, const std::vector<float>& x,
    const Buffer& w, std::size_t channels, std::size_t length,
    std::size_t width) {
  auto x_buf = backend.AllocateBuffer(x.size() * 4, MemoryKind::Device);
  auto y_buf = backend.AllocateBuffer(x.size() * 4, MemoryKind::Device);
  if (!x_buf || !y_buf) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  if (!UploadF32(backend, **x_buf, x)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((channels * length + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {(*x_buf).get(), &w, (*y_buf).get()};
  launch.scalars = {channels, length, width};
  auto ran = backend.LaunchKernel(conv, launch);
  if (!ran) {
    return std::unexpected(ran.error());
  }
  backend.Synchronize();
  return DownloadF32(backend, **y_buf);
}

// One gated-delta step for one head: updates `state` in place and
// returns the read-out o.
std::expected<std::vector<float>, StatusCode> RunDeltaStep(
    Backend& backend, const Kernel& delta, std::vector<float>& state,
    const std::vector<float>& k, const std::vector<float>& v,
    const std::vector<float>& q, std::size_t dk, std::size_t dv, float alpha,
    float beta) {
  auto s_buf = backend.AllocateBuffer(dk * dv * 4, MemoryKind::Device);
  auto k_buf = backend.AllocateBuffer(dk * 4, MemoryKind::Device);
  auto v_buf = backend.AllocateBuffer(dv * 4, MemoryKind::Device);
  auto q_buf = backend.AllocateBuffer(dk * 4, MemoryKind::Device);
  auto o_buf = backend.AllocateBuffer(dv * 4, MemoryKind::Device);
  if (!s_buf || !k_buf || !v_buf || !q_buf || !o_buf) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  if (!UploadF32(backend, **s_buf, state) ||
      !UploadF32(backend, **k_buf, k) || !UploadF32(backend, **v_buf, v) ||
      !UploadF32(backend, **q_buf, q)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  std::uint32_t alpha_bits = 0;
  std::uint32_t beta_bits = 0;
  std::memcpy(&alpha_bits, &alpha, 4);
  std::memcpy(&beta_bits, &beta, 4);
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((dv + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {(*s_buf).get(), (*k_buf).get(), (*v_buf).get(),
                    (*q_buf).get(), (*o_buf).get()};
  launch.scalars = {dk, dv, alpha_bits, beta_bits};
  auto ran = backend.LaunchKernel(delta, launch);
  if (!ran) {
    return std::unexpected(ran.error());
  }
  backend.Synchronize();
  auto new_state = DownloadF32(backend, **s_buf);
  auto out = DownloadF32(backend, **o_buf);
  if (!new_state || !out) {
    return std::unexpected(StatusCode::DeviceError);
  }
  state = std::move(*new_state);
  return std::move(*out);
}

// Derived recurrent-layer geometry from the SSM config.
struct LinearGeometry {
  std::size_t key_dim = 0;
  std::size_t value_dim = 0;
  std::size_t num_v_heads = 0;
  std::size_t head_v_dim = 0;
  std::size_t num_k_heads = 0;
  std::size_t head_k_dim = 0;
  std::size_t conv_dim = 0;
  std::size_t width = 0;
  std::size_t factor = 0;
};

std::expected<LinearGeometry, StatusCode> DeriveGeometry(
    const TransformerConfig& cfg) {
  LinearGeometry g;
  g.head_k_dim = cfg.ssm.state_size;
  g.num_k_heads = cfg.ssm.group_count;
  g.value_dim = cfg.ssm.inner_size;
  g.num_v_heads = cfg.ssm.time_step_rank;
  g.width = cfg.ssm.conv_kernel;
  if (g.head_k_dim == 0 || g.num_k_heads == 0 || g.value_dim == 0 ||
      g.num_v_heads == 0 || g.width == 0 ||
      g.value_dim % g.num_v_heads != 0 ||
      g.num_v_heads % g.num_k_heads != 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  g.key_dim = g.head_k_dim * g.num_k_heads;
  g.head_v_dim = g.value_dim / g.num_v_heads;
  g.factor = g.num_v_heads / g.num_k_heads;
  g.conv_dim = g.key_dim * 2 + g.value_dim;
  return g;
}

}  // namespace tessera::core::detail
