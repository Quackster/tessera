#include "core/numerics/gemm.hpp"

#include <cmath>
#include <vector>

#include "core/numerics/quant.hpp"

namespace tessera::core {

std::expected<void, StatusCode> GemmQ4KRef(std::span<const float> a,
                                          std::span<const std::byte> w,
                                          std::span<float> c, std::size_t m,
                                          std::size_t n, std::size_t k) {
  if (m == 0 || n == 0 || k == 0 || k % kQ4KBlockElements != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (a.size() != m * k) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (w.size() != n * (k / kQ4KBlockElements) * kQ4KBlockBytes) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (c.size() != m * n) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t blocks_per_row = k / kQ4KBlockElements;
  std::vector<float> row(k);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t b = 0; b < blocks_per_row; ++b) {
      DequantizeQ4K(w.subspan((j * blocks_per_row + b) * kQ4KBlockBytes),
                    std::span<float>(row.data() + b * kQ4KBlockElements,
                                     kQ4KBlockElements));
    }
    for (std::size_t i = 0; i < m; ++i) {
      float acc = 0.0f;
      const float* a_row = a.data() + i * k;
      for (std::size_t t = 0; t < k; ++t) {
        acc = std::fma(a_row[t], row[t], acc);
      }
      c[i * n + j] = acc;
    }
  }
  return {};
}

std::expected<void, StatusCode> GemmFp8Ref(
    std::span<const float> a, std::span<const std::byte> w,
    std::span<const float> s, std::span<float> c, std::size_t m,
    std::size_t n, std::size_t k) {
  if (m == 0 || n == 0 || k == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (a.size() != m * k || w.size() != n * k || s.size() != n ||
      c.size() != m * n) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const float scale = s[j];
      float acc = 0.0f;
      for (std::size_t t = 0; t < k; ++t) {
        const float wgt =
            scale * Fp8E4M3ToFloat(static_cast<std::uint8_t>(w[j * k + t]));
        acc = std::fma(a[i * k + t], wgt, acc);
      }
      c[i * n + j] = acc;
    }
  }
  return {};
}

std::expected<void, StatusCode> GemmMxFp4Ref(
    std::span<const float> a, std::span<const std::byte> w,
    std::span<const std::byte> s, std::span<float> c, std::size_t m,
    std::size_t n, std::size_t k) {
  constexpr std::size_t kBlock = 32;
  if (m == 0 || n == 0 || k == 0 || k % kBlock != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (a.size() != m * k || w.size() != n * (k / 2) ||
      s.size() != n * (k / kBlock) || c.size() != m * n) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t blocks = k / kBlock;
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (std::size_t b = 0; b < blocks; ++b) {
        const float scale = E8M0ToFloat(
            static_cast<std::uint8_t>(s[j * blocks + b]));
        for (std::size_t l = 0; l < kBlock; ++l) {
          const std::size_t t = b * kBlock + l;
          const std::uint8_t packed = static_cast<std::uint8_t>(
              w[(j * k + t) / 2]);
          const std::uint8_t nibble =
              (t % 2 == 0) ? (packed & 0xF) : (packed >> 4);
          const float wgt = scale * F4E2M1ToFloat(nibble);
          acc = std::fma(a[i * k + t], wgt, acc);
        }
      }
      c[i * n + j] = acc;
    }
  }
  return {};
}

}  // namespace tessera::core
