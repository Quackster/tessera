#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"

namespace tessera::core {

// The Qwen3VL vision merger: group the patches into merge x merge blocks
// (concatenating their embeddings), then mm.0 (+ bias) -> GELU -> mm.2
// (+ bias) into the language model's hidden space. `x` is
// (grid_h*grid_w) x embed, mm0 is (m2*embed) x (m2*embed), mm2 is
// proj x (m2*embed) with m2 = merge*merge; `out` is
// ((grid_h/merge)*(grid_w/merge)) x proj.
[[nodiscard]] std::expected<void, StatusCode> VisionMergerRef(
    std::span<const float> x, std::span<const float> mm0_weight,
    std::span<const float> mm0_bias, std::span<const float> mm2_weight,
    std::span<const float> mm2_bias, std::span<float> out, std::size_t grid_h,
    std::size_t grid_w, std::size_t embed, std::size_t merge,
    std::size_t projection_dim);

[[nodiscard]] std::expected<void, StatusCode> VisionMergerDevice(
    Backend& backend, const Kernel& gemm, const Kernel& gelu,
    const Kernel& bias_add, const Kernel& spatial_merge, const Buffer& x,
    const Buffer& mm0_weight, const Buffer& mm0_bias, const Buffer& mm2_weight,
    const Buffer& mm2_bias, Buffer& out, std::size_t grid_h,
    std::size_t grid_w, std::size_t embed, std::size_t merge,
    std::size_t projection_dim);

}  // namespace tessera::core
