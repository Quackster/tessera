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

}  // namespace tessera::core
