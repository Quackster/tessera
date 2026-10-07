#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <vector>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"
#include "core/vision_block.hpp"

namespace tessera::core {

// The CLIP vision encoder body: patch embedding (+ bias + position
// embedding), the transformer blocks, and the final LayerNorm. `patches`
// is tokens x patch_dim (3*patch*patch), `patch_w` embed x patch_dim,
// `position` tokens x embed, `out` tokens x embed.
[[nodiscard]] std::expected<void, StatusCode> VisionStackRef(
    std::span<const float> patches, std::span<const float> patch_w,
    std::span<const float> patch_bias, std::span<const float> position,
    const std::vector<VisionBlockWeights>& blocks,
    std::span<const float> post_ln_weight,
    std::span<const float> post_ln_bias, std::span<float> out,
    std::size_t tokens, std::size_t embed, std::size_t patch_dim,
    std::size_t heads, std::size_t head_dim, std::size_t ffn, float eps);

[[nodiscard]] std::expected<void, StatusCode> VisionStackDevice(
    Backend& backend, const Kernel& layernorm, const Kernel& gemm,
    const Kernel& attention, const Kernel& gelu, const Kernel& bias_add,
    const Kernel& add, const Buffer& patches, const Buffer& patch_w,
    const Buffer& patch_bias, const Buffer& position,
    const std::vector<VisionBlockBuffers>& blocks, const Buffer& post_ln_weight,
    const Buffer& post_ln_bias, Buffer& out, std::size_t tokens,
    std::size_t embed, std::size_t patch_dim, std::size_t heads,
    std::size_t head_dim, std::size_t ffn, float eps);

}  // namespace tessera::core
