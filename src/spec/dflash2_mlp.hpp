#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"

namespace tessera::spec {

// The MLP half of a DFlash2 draft layer: post_attention RMSNorm, one
// grouped dynamic convolution (prepare, side 0), the gated SiLU MLP, and
// the matching grouped dynamic convolution (finish, side 1) that reuses
// the projection computed in prepare. `x` and `out` are rows x hidden,
// the projection weight is (2*taps*num_groups) x hidden, the base kernel
// is 2 x taps x hidden, gate/up are ffn x hidden and down is
// hidden x ffn; num_groups = hidden / group_size.
//
// Usage:
//   auto status = DraftMlpRef(x, norm, proj, base, gate, up, down, out,
//                              rows, hidden, ffn, 2, 16, 8, 1e-6f);
[[nodiscard]] std::expected<void, StatusCode> DraftMlpRef(
    std::span<const float> x, std::span<const float> post_norm_w,
    std::span<const float> conv_proj_w, std::span<const float> conv_base,
    std::span<const float> gate_w, std::span<const float> up_w,
    std::span<const float> down_w, std::span<float> out, std::size_t rows,
    std::size_t hidden, std::size_t ffn, std::size_t taps,
    std::size_t group_size, std::size_t block_size, float eps);

// Device version. `gemm` is gemm_f32, `conv` is dflash_conv, `rmsnorm`
// and `silu` are their built-ins. Scratch buffers (fp32): xn (rows x
// hidden), proj (rows x 2*taps*num_groups), h1 (rows x hidden), gate/up/
// act (rows x ffn), down (rows x hidden) and side_base (taps x hidden).
[[nodiscard]] std::expected<void, StatusCode> DraftMlpDevice(
    Backend& backend, const Kernel& rmsnorm, const Kernel& gemm,
    const Kernel& conv, const Kernel& silu, Buffer& xn, Buffer& proj,
    Buffer& h1, Buffer& gate, Buffer& up, Buffer& act, Buffer& down,
    Buffer& side_base, const Buffer& x, const Buffer& post_norm_w,
    const Buffer& conv_proj_w, const Buffer& conv_base, const Buffer& gate_w,
    const Buffer& up_w, const Buffer& down_w, Buffer& out, std::size_t rows,
    std::size_t hidden, std::size_t ffn, std::size_t taps,
    std::size_t group_size, std::size_t block_size, float eps);

}  // namespace tessera::spec
