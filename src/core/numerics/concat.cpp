#include "core/numerics/concat.hpp"

namespace tessera::core {

std::expected<void, StatusCode> ConcatFeaturesRef(std::span<const float> in,
                                                  std::span<float> out,
                                                  std::size_t n,
                                                  std::size_t rows,
                                                  std::size_t features) {
  if (n == 0 || rows == 0 || features == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (in.size() != n * rows * features || out.size() != rows * n * features) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    float* row = out.data() + r * n * features;
    for (std::size_t i = 0; i < n; ++i) {
      const float* src = in.data() + (i * rows + r) * features;
      for (std::size_t j = 0; j < features; ++j) {
        row[i * features + j] = src[j];
      }
    }
  }
  return {};
}

}  // namespace tessera::core
