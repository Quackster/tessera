#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "tessera/types.hpp"

// Gated residual stack ("hyper-connection") numeric contracts. This is
// the mechanism qwen4exp wraps around its attention and FFN sub-blocks:
// the residual is a stack of `hc` streams, `read` collapses it to the one
// vector a sub-block consumes and produces a per-stream injection, and
// `write` puts the sub-block output back into every stream, weighted per
// stream.
//
// The weights are bfloat16 (tessera engine form BF16, two bytes each):
// w_norm is f32; w_down is (hc_lr, hc*n_embd) row-major with the reduction
// axis contiguous; w_up is (hc*n_embd, hc_lr) with the rank axis
// contiguous; w_inject is (hc, hc*n_embd). The default activation contract
// rounds the normalized streams and the bottleneck to bfloat16, matching
// the served reference. All fields come from model data.

namespace tessera::core {

// Geometry of one gated residual stack.
struct HyperConnectionShapes {
  std::size_t n_embd = 0;  // width of one residual stream
  std::size_t hc = 0;      // number of streams
  std::size_t hc_lr = 0;   // rank of the down/up bottleneck
};

// Device and host scratch (in floats and bytes) one read needs:
// xn (xn_floats), xq (xq_u16 bf16), lq (lq_u16 bf16), gated
// (gated_floats), lo (lo_floats). Reported as element counts so a caller
// can allocate the matching buffers.
struct HyperConnectionScratch {
  std::size_t xn_floats = 0;
  std::size_t xq_u16 = 0;
  std::size_t lq_u16 = 0;
  std::size_t gated_floats = 0;
  std::size_t lo_floats = 0;
};

// Element counts of the scratch the read path needs for `s`.
[[nodiscard]] HyperConnectionScratch HyperConnectionScratchFor(
    const HyperConnectionShapes& s);

// Host reference for the read path. `r` is (hc, n_embd) row-major,
// `w_norm` (hc*n_embd) f32, `w_down` (hc_lr, hc*n_embd) bf16, `w_up`
// (hc*n_embd, hc_lr) bf16, `w_inject` (hc, hc*n_embd) bf16 or empty.
// `mixed` receives n_embd floats and `inject` hc floats. Rounding and
// reduction follow the default bfloat16 contract. A size mismatch is
// InvalidArgument.
//
// Usage:
//   HyperConnectionReadRef(r, norm, down, up, inject, eps, shapes, mixed, out);
[[nodiscard]] std::expected<void, StatusCode> HyperConnectionReadRef(
    std::span<const float> r, std::span<const float> w_norm,
    std::span<const std::byte> w_down, std::span<const std::byte> w_up,
    std::span<const std::byte> w_inject, double eps,
    const HyperConnectionShapes& s, std::span<float> mixed,
    std::span<float> inject);

// Host reference for the write path: out[i] = r[i] + block_out[d] * w[c]
// with w[c] = 2*sigmoid(inject[c] / hc). `out` may alias `r`.
[[nodiscard]] std::expected<void, StatusCode> HyperConnectionWriteRef(
    std::span<const float> r, std::span<const float> block_out,
    std::span<const float> inject, const HyperConnectionShapes& s,
    std::span<float> out);

}  // namespace tessera::core
