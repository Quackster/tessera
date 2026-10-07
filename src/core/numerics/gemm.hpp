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

// Host reference for the "gemm_fp8" built-in: C = A x (diag(s) x W)^T
// with fp32 sequential accumulation (deterministic, the same order
// the kernels use). `a` is m x k fp32 row-major, `w` is n x k FP8
// E4M3 bytes, `s` holds one fp32 scale per row of W, `c` is m x n
// fp32. m, n, k must be nonzero; a size mismatch is InvalidArgument.
//
// Usage:
//   auto status = GemmFp8Ref(a, w, s, c, m, n, k);
[[nodiscard]] std::expected<void, StatusCode> GemmFp8Ref(
    std::span<const float> a, std::span<const std::byte> w,
    std::span<const float> s, std::span<float> c, std::size_t m,
    std::size_t n, std::size_t k);

// Host reference for the "gemm_mxfp4" built-in: C = A x W'^T with
// fp32 sequential accumulation, where W' dequantizes the MXFP4
// microscale layout (F4E2M1 nibbles in `w`, one E8M0 byte per 32
// elements in `s`). `a` is m x k fp32, `w` holds n x k/2 bytes, `s`
// holds n x k/32 bytes, `c` is m x n fp32. k must be a positive
// multiple of 32 and m, n nonzero; a size mismatch is
// InvalidArgument.
//
// Usage:
//   auto status = GemmMxFp4Ref(a, w, s, c, m, n, k);
[[nodiscard]] std::expected<void, StatusCode> GemmMxFp4Ref(
    std::span<const float> a, std::span<const std::byte> w,
    std::span<const std::byte> s, std::span<float> c, std::size_t m,
    std::size_t n, std::size_t k);

}  // namespace tessera::core
