#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "tessera/types.hpp"

namespace tessera::core {

// Host reference for the "gemm_q4k" built-in: C = A x dequant(W)^T
// with fp32 sequential accumulation (deterministic, the same order the
// kernels use). `a` is m x k fp32 row-major, `w` is n rows of
// k/256 Q4_K blocks, `c` is m x n fp32. k must be a positive multiple
// of 256 and m, n nonzero; a size mismatch is InvalidArgument.
//
// Usage:
//   auto status = GemmQ4KRef(a, w, c, m, n, k);
[[nodiscard]] std::expected<void, StatusCode> GemmQ4KRef(
    std::span<const float> a, std::span<const std::byte> w,
    std::span<float> c, std::size_t m, std::size_t n, std::size_t k);

}  // namespace tessera::core
