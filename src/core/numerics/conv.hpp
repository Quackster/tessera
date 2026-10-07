#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "tessera/types.hpp"

namespace tessera::core {

// Host reference for the "conv1d" built-in: causal depthwise convolution
// over channels x length fp32 (y[c,t] = sum of w[c,i] * x[c,t-i] for
// i < width with t-i >= 0, ascending accumulation, the same order the
// kernels use). Linear-attention blocks run the qkv mix through this
// before the SiLU and the recurrent scan. Channels, length and width
// must be nonzero and the spans must hold channels*length (x, y) and
// channels*width (w) elements; else InvalidArgument.
//
// Usage:
//   auto status = ConvRef(x, w, y, channels, length, width);
[[nodiscard]] std::expected<void, StatusCode> ConvRef(
    std::span<const float> x, std::span<const float> w, std::span<float> y,
    std::size_t channels, std::size_t length, std::size_t width);

}  // namespace tessera::core
