#include "core/numerics/conv.hpp"

#include <cmath>

namespace tessera::core {

std::expected<void, StatusCode> ConvRef(std::span<const float> x,
                                         std::span<const float> w,
                                         std::span<float> y,
                                         std::size_t channels,
                                         std::size_t length,
                                         std::size_t width) {
  if (channels == 0 || length == 0 || width == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (x.size() != channels * length || w.size() != channels * width ||
      y.size() != channels * length) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t c = 0; c < channels; ++c) {
    const float* x_row = x.data() + c * length;
    const float* w_row = w.data() + c * width;
    float* y_row = y.data() + c * length;
    for (std::size_t t = 0; t < length; ++t) {
      float acc = 0.0f;
      for (std::size_t i = 0; i < width && i <= t; ++i) {
        acc = std::fma(w_row[i], x_row[t - i], acc);
      }
      y_row[t] = acc;
    }
  }
  return {};
}

}  // namespace tessera::core
