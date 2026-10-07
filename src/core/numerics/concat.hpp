#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "tessera/types.hpp"

namespace tessera::core {

// Host reference for the "concat_features" built-in: stack `n` tensors of
// rows x features along the feature axis. `in` is n x rows x features
// (layer-major, row-major within a layer) and `out` is rows x (n*features)
// with out[r, i*features + j] = in[i, r, j]. This is the DFlash2 target
// hidden concatenation before the `fc` projection. Every size must be
// nonzero and match; else InvalidArgument.
[[nodiscard]] std::expected<void, StatusCode> ConcatFeaturesRef(
    std::span<const float> in, std::span<float> out, std::size_t n,
    std::size_t rows, std::size_t features);

}  // namespace tessera::core
