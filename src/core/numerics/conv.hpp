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

// Host reference for the "dflash_conv" built-in: the DFlash2 grouped
// dynamic convolution (token-position local, reset every block). `x` is
// rows x channels, `base` is taps x channels (a static per-tap kernel),
// `delta` is rows x taps x (channels/group_size) (per-token per-group
// offsets). With num_groups = channels / group_size, position =
// row % block_size and group = channel / group_size:
//   y[r,c] = (base[0,c] + delta[r,0,g]) * x[r,c]
//          + sum_{tap=1..taps-1, position>=tap}
//                (base[tap,c] + delta[r,tap,g]) * x[r-tap,c]
// channels must be a multiple of group_size; every size must match; else
// InvalidArgument.
[[nodiscard]] std::expected<void, StatusCode> DflashConvRef(
    std::span<const float> x, std::span<const float> delta,
    std::span<const float> base, std::span<float> y, std::size_t rows,
    std::size_t channels, std::size_t taps, std::size_t group_size,
    std::size_t block_size);

}  // namespace tessera::core
