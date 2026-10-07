#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "tessera/types.hpp"

namespace tessera::core {

// Host reference for the "rope" built-in: NeoX-style rotary embedding
// applied in place over rows x heads x head_dim fp32. Pair j of each
// head rotates by pos * theta^(-2j/rope_dim) with pos = pos_base + row;
// elements past rope_dim are left alone. head_dim and rope_dim must be
// even, rope_dim <= head_dim, theta > 0; a size mismatch or a zero dim
// is InvalidArgument. Deterministic: ascending pair order.
//
// Usage:
//   auto status = RopeRef(io, rows, heads, head_dim, rope_dim, 0, 1e4);
[[nodiscard]] std::expected<void, StatusCode> RopeRef(
    std::span<float> io, std::size_t rows, std::size_t heads,
    std::size_t head_dim, std::size_t rope_dim, std::uint64_t pos_base,
    double theta);

// Host reference for the "mrope" built-in: multimodal rotary embedding
// applied in place over rows x heads x head_dim fp32. The rope_dim/2
// NeoX pairs split into temporal, height and width sections of sec_t,
// sec_h and sec_w pairs; pair j rotates by its section position times
// theta^(-2j/rope_dim) with the global pair index j, so frequencies
// stay continuous across sections. `pos` holds one (t, h, w) triple
// per row; text rows repeat the same id three times, which reduces to
// plain RoPE. Pairs past the three sections reuse the width id.
// Sections must sum to at most rope_dim/2, rope_dim must be even and
// within head_dim, theta > 0; a size mismatch or a zero dim is
// InvalidArgument. Deterministic: ascending pair order.
//
// Usage:
//   auto status = MropeRef(io, pos, rows, heads, dim, rdim, 3, 3, 2,
//                          1e7);
[[nodiscard]] std::expected<void, StatusCode> MropeRef(
    std::span<float> io, std::span<const std::uint64_t> pos,
    std::size_t rows, std::size_t heads, std::size_t head_dim,
    std::size_t rope_dim, std::size_t sec_t, std::size_t sec_h,
    std::size_t sec_w, double theta);

// Host reference for the "attention" built-in: causal grouped-query
// attention with scale 1/sqrt(head_dim) and fp32 sequential
// accumulation (deterministic, the same order the kernels use). Query
// row i sits at position q_base + i and attends key rows 0..pos
// (clamped to n - 1). `q` is m x heads*head_dim fp32 row-major, `k`
// and `v` are n x kv_heads*head_dim, `out` is m x heads*head_dim.
// heads must be a nonzero multiple of kv_heads; a size mismatch or a
// zero dim is InvalidArgument.
//
// Usage:
//   auto status = AttentionRef(q, k, v, out, m, n, h, kv, d, base);
[[nodiscard]] std::expected<void, StatusCode> AttentionRef(
    std::span<const float> q, std::span<const float> k,
    std::span<const float> v, std::span<float> out, std::size_t m,
    std::size_t n, std::size_t heads, std::size_t kv_heads,
    std::size_t head_dim, std::uint64_t q_base);

// Host reference for the "delta_step" built-in: one gated delta-rule
// recurrent step over a dk x dv fp32 state (linear attention without
// softmax). With read r = S^T k, the state moves to
// S' = alpha * (S - beta * k r^T) + beta * v k^T and reports
// o = S'^T q. The state buffer updates in place with ascending,
// per-column accumulation (deterministic, the order the kernels use).
// dk and dv must be nonzero and the spans must hold dk*dv (s),
// dk (k, q) and dv (v, o) elements; else InvalidArgument.
//
// Usage:
//   auto status = DeltaStepRef(s, k, v, q, o, dk, dv, 0.9f, 0.5f);
[[nodiscard]] std::expected<void, StatusCode> DeltaStepRef(
    std::span<float> s, std::span<const float> k, std::span<const float> v,
    std::span<const float> q, std::span<float> o, std::size_t dk,
    std::size_t dv, float alpha, float beta);

}  // namespace tessera::core
