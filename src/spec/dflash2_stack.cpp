#include "spec/dflash2_stack.hpp"

#include <memory>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/norm.hpp"

namespace tessera::spec {

namespace {

using core::detail::AddDevice;
using core::detail::RmsNormDevice;

}  // namespace

std::expected<void, StatusCode> DraftStackRef(
    std::span<const float> embed,
    const std::vector<DraftLayerWeights>& layers,
    std::span<const float> final_norm, std::span<float> out,
    std::size_t rows, std::size_t hidden_dim, std::size_t heads,
    std::size_t kv_heads, std::size_t head_dim, std::size_t ffn,
    std::size_t taps, std::size_t group_size, std::size_t block_size,
    std::size_t window, std::uint64_t pos_base, double theta, float eps,
    std::span<const float> context_hidden) {
  const std::size_t elements = rows * hidden_dim;
  if (embed.size() != elements || out.size() != elements ||
      final_norm.size() != hidden_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> hidden(embed.begin(), embed.end());
  std::vector<float> residual;
  std::vector<float> layer_out(elements);
  std::vector<float> layer_res(elements);
  for (const DraftLayerWeights& layer : layers) {
    auto status = DraftLayerRef(
        std::span<const float>(hidden),
        residual.empty() ? std::span<const float>()
                         : std::span<const float>(residual),
        layer, std::span<float>(layer_out), std::span<float>(layer_res), rows,
        hidden_dim, heads, kv_heads, head_dim, ffn, taps, group_size,
        block_size, window, pos_base, theta, eps, context_hidden);
    if (!status) {
      return std::unexpected(status.error());
    }
    hidden = layer_out;
    residual = layer_res;
  }
  std::vector<float> fused(elements);
  if (!core::AddRef(std::span<const float>(hidden),
                    std::span<const float>(residual), std::span<float>(fused),
                    elements) ||
      !core::RmsNormRef(std::span<const float>(fused), final_norm, out, rows,
                        hidden_dim, eps)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return {};
}

std::expected<void, StatusCode> DraftStackDevice(
    Backend& backend, const Kernel& rmsnorm, const Kernel& gemm,
    const Kernel& conv, const Kernel& rope, const Kernel& attention,
    const Kernel& silu, const Kernel& add, const Buffer& embed,
    const std::vector<DraftLayerBuffers>& layers, const Buffer& final_norm,
    Buffer& out, std::size_t rows, std::size_t hidden_dim, std::size_t heads,
    std::size_t kv_heads, std::size_t head_dim, std::size_t ffn,
    std::size_t taps, std::size_t group_size, std::size_t block_size,
    std::size_t window, std::uint64_t pos_base, double theta, float eps,
    const Buffer* context_hidden, std::size_t ctx) {
  if (rows == 0 || hidden_dim == 0 || layers.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t elements = rows * hidden_dim;
  auto alloc = [&backend](std::size_t count) {
    return backend.AllocateBuffer(count * 4, MemoryKind::Device);
  };
  auto h_a = alloc(elements);
  auto h_b = alloc(elements);
  auto residual = alloc(elements);
  auto fused = alloc(elements);
  if (!h_a || !h_b || !residual || !fused) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  if (!backend.CopyD2D(embed, 0, **h_a, 0, elements * 4)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  Buffer* in = h_a->get();
  Buffer* out_swap = h_b->get();
  for (std::size_t l = 0; l < layers.size(); ++l) {
    auto status = DraftLayerDevice(
        backend, rmsnorm, gemm, conv, rope, attention, silu, add, *in,
        l == 0 ? nullptr : residual->get(), layers[l], *out_swap, **residual,
        rows, hidden_dim, heads, kv_heads, head_dim, ffn, taps, group_size,
        block_size, window, pos_base, theta, eps, context_hidden, ctx);
    if (!status) {
      return std::unexpected(status.error());
    }
    std::swap(in, out_swap);
  }
  if (!AddDevice(backend, add, *in, **residual, **fused, elements) ||
      !RmsNormDevice(backend, rmsnorm, **fused, final_norm, out, rows,
                     hidden_dim, eps)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

}  // namespace tessera::spec
