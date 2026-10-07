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

// Host reference for the "conv1d_step" built-in: the current-step causal
// depthwise conv output. `x` is channels x width with the newest sample
// first (x[c*width] is time t, x[c*width+i] is time t-i), `w` is
// channels x width (w[c*width+i] is the tap for lag i) and `y` is
// channels. y[c] = sum_i w[c*width+i] * x[c*width+i]. The gated-delta
// block feeds the conv history plus the current qkv. Channels and width
// must be nonzero and the spans must match; else InvalidArgument.
[[nodiscard]] std::expected<void, StatusCode> Conv1dStepRef(
    std::span<const float> x, std::span<const float> w, std::span<float> y,
    std::size_t channels, std::size_t width);

}  // namespace tessera::core
