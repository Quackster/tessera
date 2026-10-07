#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"

namespace tessera::spec {

// One stage ("prepare" or "finish") of a DFlash2 grouped dynamic
// convolution layer. It is the kernel_projection GEMM followed by the
// grouped dynamic convolution of one side:
//   proj = hidden @ projection_w^T          (rows x 2*taps*num_groups)
//   out  = dflash_conv(hidden, proj side `side`, base_side)
// `projection_w` is (2*taps*num_groups) x channels, `base_side` is
// taps x channels for the requested side, `out` is rows x channels.
// num_groups = channels / group_size. `side` must be 0 or 1.
//
// Usage:
//   auto status = GroupedConvRef(hidden, w, base, out, rows, channels, 2,
//                                16, 8, /*side=*/0);
[[nodiscard]] std::expected<void, StatusCode> GroupedConvRef(
    std::span<const float> hidden, std::span<const float> projection_w,
    std::span<const float> base_side, std::span<float> out, std::size_t rows,
    std::size_t channels, std::size_t taps, std::size_t group_size,
    std::size_t block_size, std::size_t side);

// Device version: `projection_w` is fp32 (rows x channels), `base_kernel`
// holds both sides (2 x taps x channels); the requested side is copied
// into `scratch_base`. `scratch_proj` is rows x 2*taps*num_groups and
// `scratch_base` is taps x channels. `gemm` runs the projection
// (gemm_f32) and `conv` the grouped convolution (dflash_conv).
[[nodiscard]] std::expected<void, StatusCode> GroupedConvDevice(
    Backend& backend, const Kernel& gemm, const Kernel& conv,
    Buffer& scratch_proj, Buffer& scratch_base, const Buffer& hidden,
    const Buffer& projection_w, const Buffer& base_kernel, Buffer& out,
    std::size_t rows, std::size_t channels, std::size_t taps,
    std::size_t group_size, std::size_t block_size, std::size_t side);

}  // namespace tessera::spec
