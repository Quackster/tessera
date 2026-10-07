#include "core/vision_stack.hpp"

#include <memory>

#include "core/decode_internal.hpp"
#include "core/numerics/gemm.hpp"
#include "core/numerics/norm.hpp"

namespace tessera::core {

std::expected<void, StatusCode> VisionStackRef(
    std::span<const float> patches, std::span<const float> patch_w,
    std::span<const float> patch_bias, std::span<const float> position,
    const std::vector<VisionBlockWeights>& blocks,
    std::span<const float> post_ln_weight,
    std::span<const float> post_ln_bias, std::span<float> out,
    std::size_t tokens, std::size_t embed, std::size_t patch_dim,
    std::size_t heads, std::size_t head_dim, std::size_t ffn, float eps) {
  if (tokens == 0 || embed == 0 || patch_dim == 0 || heads * head_dim != embed ||
      patches.size() != tokens * patch_dim ||
      patch_w.size() != embed * patch_dim || patch_bias.size() != embed ||
      position.size() != tokens * embed || out.size() != tokens * embed) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> x(tokens * embed);
  std::vector<float> y(tokens * embed);
  if (!GemmF32Ref(patches, patch_w, std::span<float>(x), tokens, embed,
                  patch_dim) ||
      !BiasAddRef(std::span<const float>(x), patch_bias, std::span<float>(x),
                  tokens, embed) ||
      !AddRef(std::span<const float>(x), position, std::span<float>(x),
              tokens * embed)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (const VisionBlockWeights& block : blocks) {
    auto status = VisionBlockRef(std::span<const float>(x), block,
                                 std::span<float>(y), tokens, embed, heads,
                                 head_dim, ffn, eps);
    if (!status) {
      return std::unexpected(status.error());
    }
    x.swap(y);
  }
  return LayerNormRef(std::span<const float>(x), post_ln_weight, post_ln_bias,
                      out, tokens, embed, eps);
}

std::expected<void, StatusCode> VisionStackDevice(
    Backend& backend, const Kernel& layernorm, const Kernel& gemm,
    const Kernel& attention, const Kernel& gelu, const Kernel& bias_add,
    const Kernel& add, const Buffer& patches, const Buffer& patch_w,
    const Buffer& patch_bias, const Buffer& position,
    const std::vector<VisionBlockBuffers>& blocks, const Buffer& post_ln_weight,
    const Buffer& post_ln_bias, Buffer& out, std::size_t tokens,
    std::size_t embed, std::size_t patch_dim, std::size_t heads,
    std::size_t head_dim, std::size_t ffn, float eps) {
  if (tokens == 0 || embed == 0 || patch_dim == 0 || heads * head_dim != embed) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t elements = tokens * embed;
  auto x = backend.AllocateBuffer(elements * 4, MemoryKind::Device);
  auto y = backend.AllocateBuffer(elements * 4, MemoryKind::Device);
  if (!x || !y) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  if (!detail::ProjectDevice(backend, gemm, patches, patch_w, **x, tokens,
                             embed, patch_dim) ||
      !detail::BiasAddDevice(backend, bias_add, **x, patch_bias, **x, tokens,
                             embed) ||
      !detail::AddDevice(backend, add, **x, position, **x, elements)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  Buffer* in = x->get();
  Buffer* swap = y->get();
  for (const VisionBlockBuffers& block : blocks) {
    auto status = VisionBlockDevice(backend, layernorm, gemm, attention, gelu,
                                    bias_add, add, *in, block, *swap, tokens,
                                    embed, heads, head_dim, ffn, eps);
    if (!status) {
      return std::unexpected(status.error());
    }
    std::swap(in, swap);
  }
  return detail::LayerNormDevice(backend, layernorm, *in, post_ln_weight,
                                 post_ln_bias, out, tokens, embed, eps);
}

}  // namespace tessera::core
