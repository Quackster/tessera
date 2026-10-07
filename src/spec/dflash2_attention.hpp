#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"

namespace tessera::spec {

// The attention half of a DFlash2 draft layer: input_layernorm RMSNorm,
// one grouped dynamic convolution (attention_conv prepare, side 0),
// grouped-query sliding attention with per-head QK-RMSNorm and RoPE
// (queries start at `pos_base`), the output projection, and the matching
// attention_conv finish (side 1) that reuses the prepare projection.
// `x` and `out` are rows x hidden; the projection weight is
// (2*taps*num_groups) x hidden; the base kernel is 2 x taps x hidden;
// q/o are (heads*head_dim) wide and k/v (kv_heads*head_dim); q_norm and
// k_norm are head_dim; num_groups = hidden / group_size. `window` is the
// sliding window (0 keeps all past keys).
//
// Usage:
//   auto status = DraftAttentionRef(x, ...out, rows, 5120, 32, 8, 128,
//                                   2, 16, 8, 2048, 0, 1e7, 1e-6f);
[[nodiscard]] std::expected<void, StatusCode> DraftAttentionRef(
    std::span<const float> x, std::span<const float> input_norm_w,
    std::span<const float> conv_proj_w, std::span<const float> conv_base,
    std::span<const float> q_w, std::span<const float> k_w,
    std::span<const float> v_w, std::span<const float> o_w,
    std::span<const float> q_norm_w, std::span<const float> k_norm_w,
    std::span<float> out, std::size_t rows, std::size_t hidden,
    std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
    std::size_t taps, std::size_t group_size, std::size_t block_size,
    std::size_t window, std::uint64_t pos_base, double theta, float eps);

// Device version. `gemm` is gemm_f32, `conv` dflash_conv, `rmsnorm`,
// `rope` and `attention` their built-ins. Scratch (fp32): xn (rows x
// hidden), proj (rows x 2*taps*num_groups), h1 (rows x hidden), q
// (rows x heads*head_dim), k/v (rows x kv_heads*head_dim), attn (rows x
// heads*head_dim), oproj (rows x hidden) and side_base (taps x hidden).
[[nodiscard]] std::expected<void, StatusCode> DraftAttentionDevice(
    Backend& backend, const Kernel& rmsnorm, const Kernel& gemm,
    const Kernel& conv, const Kernel& rope, const Kernel& attention,
    Buffer& xn, Buffer& proj, Buffer& h1, Buffer& q, Buffer& k, Buffer& v,
    Buffer& attn, Buffer& oproj, Buffer& side_base, const Buffer& x,
    const Buffer& input_norm_w, const Buffer& conv_proj_w,
    const Buffer& conv_base, const Buffer& q_w, const Buffer& k_w,
    const Buffer& v_w, const Buffer& o_w, const Buffer& q_norm_w,
    const Buffer& k_norm_w, Buffer& out, std::size_t rows, std::size_t hidden,
    std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
    std::size_t taps, std::size_t group_size, std::size_t block_size,
    std::size_t window, std::uint64_t pos_base, double theta, float eps);

}  // namespace tessera::spec
