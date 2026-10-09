#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"
#include "spec/dflash2_context.hpp"
#include "spec/dflash2_layer.hpp"

namespace tessera::spec {

// The full DFlash2 draft block: fuse the target hidden states with `fc`,
// run the draft layer stack on the mask-token embeddings with that fused
// hidden as context, apply the final norm (inside the stack) and project
// to logits with the shared output weight.
//   fused = fc(cat(aux))          (ctx x hidden)
//   hidden = stack(mask_embeds, layers, final_norm, context=fused)
//   logits = output_w @ hidden    (rows x vocab)
// `aux` is n x ctx x features (layer-major), `fc_w` is hidden x (n*features),
// `mask_embeds` and the returned `logits` are rows x hidden and rows x vocab.
[[nodiscard]] std::expected<void, StatusCode> DraftBlockRef(
    std::span<const float> mask_embeds, std::span<const float> aux,
    std::span<const float> fc_w,
    const std::vector<DraftLayerWeights>& layers,
    std::span<const float> final_norm, std::span<const float> output_w,
    std::span<float> logits, std::size_t rows, std::size_t ctx,
    std::size_t hidden_dim, std::size_t n, std::size_t features,
    std::size_t vocab, std::size_t heads, std::size_t kv_heads,
    std::size_t head_dim, std::size_t ffn, std::size_t taps,
    std::size_t group_size, std::size_t block_size, std::size_t window,
    std::uint64_t pos_base, double theta, float eps, bool causal = true);

// Device version. All kernels are built-ins: gemm (gemm_f32), conv
// (dflash_conv), rmsnorm, rope, attention, silu (silu_mul), add,
// concat (concat_features) and quantize (quantize_fp8, the per-token
// activation QDQ the drafter was trained on). Scratch is allocated
// internally.
// When `context_kv` is non-null the draft attends over the precomputed
// per-layer context K/V (ctx rows) and `aux`/`fc_w` are ignored; otherwise
// the fused context is computed from `aux` (n x ctx x features) and `fc_w`.
[[nodiscard]] std::expected<void, StatusCode> DraftBlockDevice(
    Backend& backend, const Kernel& rmsnorm, const Kernel& gemm,
    const Kernel& head_gemm, const Kernel& conv, const Kernel& rope,
    const Kernel& attention,
    const Kernel& silu, const Kernel& add, const Kernel& concat,
    const Kernel& quantize,
    const Buffer& mask_embeds, const Buffer* aux, const Buffer* fc_w,
    const std::vector<DraftLayerBuffers>& layers, const Buffer& final_norm,
    const Buffer& output_w, Buffer& logits, std::size_t rows,
    std::size_t ctx, std::size_t hidden_dim, std::size_t n,
    std::size_t features, std::size_t vocab, std::size_t heads,
    std::size_t kv_heads, std::size_t head_dim, std::size_t ffn,
    std::size_t taps, std::size_t group_size, std::size_t block_size,
    std::size_t window, std::uint64_t pos_base, double theta, float eps,
    bool causal = true, Buffer* hidden_out = nullptr,
    const std::vector<DraftContextKvView>* context_kv = nullptr);

}  // namespace tessera::spec
