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

}  // namespace tessera::core
