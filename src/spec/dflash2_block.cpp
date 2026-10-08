#include "spec/dflash2_block.hpp"

#include <memory>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/gemm.hpp"
#include "spec/dflash2_fuse.hpp"
#include "spec/dflash2_stack.hpp"

namespace tessera::spec {

std::expected<void, StatusCode> DraftBlockRef(
    std::span<const float> mask_embeds, std::span<const float> aux,
    std::span<const float> fc_w,
    const std::vector<DraftLayerWeights>& layers,
    std::span<const float> final_norm, std::span<const float> output_w,
    std::span<float> logits, std::size_t rows, std::size_t ctx,
    std::size_t hidden_dim, std::size_t n, std::size_t features,
    std::size_t vocab, std::size_t heads, std::size_t kv_heads,
    std::size_t head_dim, std::size_t ffn, std::size_t taps,
    std::size_t group_size, std::size_t block_size, std::size_t window,
    std::uint64_t pos_base, double theta, float eps, bool causal) {
  if (mask_embeds.size() != rows * hidden_dim ||
      logits.size() != rows * vocab || output_w.size() != vocab * hidden_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> fused(ctx * hidden_dim);
  auto fuse = DraftFuseRef(aux, fc_w, std::span<float>(fused), n, ctx, features,
                           hidden_dim);
  if (!fuse) {
    return std::unexpected(fuse.error());
  }
  std::vector<float> hidden(rows * hidden_dim);
  auto stack = DraftStackRef(mask_embeds, layers, final_norm,
                             std::span<float>(hidden), rows, hidden_dim, heads,
                             kv_heads, head_dim, ffn, taps, group_size,
                             block_size, window, pos_base, theta, eps,
                             std::span<const float>(fused), causal);
  if (!stack) {
    return std::unexpected(stack.error());
  }
  if (!core::GemmF32Ref(std::span<const float>(hidden), output_w, logits, rows,
                        vocab, hidden_dim)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return {};
}

std::expected<void, StatusCode> DraftBlockDevice(
    Backend& backend, const Kernel& rmsnorm, const Kernel& gemm,
    const Kernel& head_gemm, const Kernel& conv, const Kernel& rope,
    const Kernel& attention,
    const Kernel& silu, const Kernel& add, const Kernel& concat,
    const Kernel& quantize,
    const Buffer& mask_embeds, const Buffer& aux, const Buffer& fc_w,
    const std::vector<DraftLayerBuffers>& layers, const Buffer& final_norm,
    const Buffer& output_w, Buffer& logits, std::size_t rows, std::size_t ctx,
    std::size_t hidden_dim, std::size_t n, std::size_t features,
    std::size_t vocab, std::size_t heads, std::size_t kv_heads,
    std::size_t head_dim, std::size_t ffn, std::size_t taps,
    std::size_t group_size, std::size_t block_size, std::size_t window,
    std::uint64_t pos_base, double theta, float eps, bool causal,
    Buffer* hidden_out) {
  auto fused = backend.AllocateBuffer(ctx * hidden_dim * 4, MemoryKind::Device);
  auto concat_scratch =
      backend.AllocateBuffer(ctx * n * features * 4, MemoryKind::Device);
  auto qscale = backend.AllocateBuffer(ctx * 4, MemoryKind::Device);
  auto hidden = backend.AllocateBuffer(rows * hidden_dim * 4, MemoryKind::Device);
  if (!fused || !concat_scratch || !qscale || !hidden) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  auto fuse = DraftFuseDevice(backend, gemm, concat, quantize, **concat_scratch,
                              **qscale, aux, fc_w, **fused, n, ctx, features,
                              hidden_dim);
  if (!fuse) {
    return std::unexpected(fuse.error());
  }
  auto stack = DraftStackDevice(
      backend, rmsnorm, gemm, conv, rope, attention, silu, add, mask_embeds,
      layers, final_norm, **hidden, rows, hidden_dim, heads, kv_heads, head_dim,
      ffn, taps, group_size, block_size, window, pos_base, theta, eps,
      fused->get(), ctx, causal);
  if (!stack) {
    return std::unexpected(stack.error());
  }
  if (hidden_out != nullptr &&
      !backend.CopyD2D(**hidden, 0, *hidden_out, 0,
                       rows * hidden_dim * 4)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return core::detail::ProjectDevice(backend, head_gemm, **hidden, output_w,
                                     logits, rows, vocab, hidden_dim);
}

}  // namespace tessera::spec
