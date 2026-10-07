#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "tessera/types.hpp"

namespace tessera::core {

// Host reference for the "image_patchify" built-in: normalize an
// [h, w, 3] fp32 image (values in [0, 1]) by the per-channel mean and
// std, then split it into non-overlapping patch x patch patches. `out` is
// num_patches x (3*patch*patch); patch p indexes (py, px) row-major and
// within a patch the element is at c*patch*patch + ph*patch + pw (the
// CLIP conv weight layout). h and w must be nonzero multiples of patch.
[[nodiscard]] std::expected<void, StatusCode> PatchifyRef(
    std::span<const float> image, std::span<const float> mean,
    std::span<const float> std, std::span<float> out, std::size_t h,
    std::size_t w, std::size_t patch);

// Host reference for the "spatial_merge" built-in: group a grid_h x
// grid_w grid of `embed`-wide tokens into (grid_h/merge) x
// (grid_w/merge) blocks, concatenating each block's merge x merge
// embeddings. `x` is (grid_h*grid_w) x embed; `out` is
// ((grid_h/merge)*(grid_w/merge)) x (merge*merge*embed), block-major with
// the element at (mh*merge+mw)*embed + c. Sizes must be nonzero and
// divisible; else InvalidArgument.
[[nodiscard]] std::expected<void, StatusCode> SpaceMergeRef(
    std::span<const float> x, std::span<float> out, std::size_t grid_h,
    std::size_t grid_w, std::size_t embed, std::size_t merge);

}  // namespace tessera::core
