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

std::expected<void, StatusCode> Conv1dStepRef(std::span<const float> x,
                                              std::span<const float> w,
                                              std::span<float> y,
                                              std::size_t channels,
                                              std::size_t width) {
  if (channels == 0 || width == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (x.size() != channels * width || w.size() != channels * width ||
      y.size() != channels) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t c = 0; c < channels; ++c) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < width; ++i) {
      acc = std::fma(w[c * width + i], x[c * width + i], acc);
    }
    y[c] = acc;
  }
  return {};
}

std::expected<void, StatusCode> DflashConvRef(
    std::span<const float> x, std::span<const float> delta,
    std::span<const float> base, std::span<float> y, std::size_t rows,
    std::size_t channels, std::size_t taps, std::size_t group_size,
    std::size_t block_size, std::size_t delta_row_stride,
    std::size_t delta_offset) {
  if (rows == 0 || channels == 0 || taps == 0 || group_size == 0 ||
      block_size == 0 || channels % group_size != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t num_groups = channels / group_size;
  // The highest delta index read is on the last row's last tap/group.
  const std::size_t delta_required =
      delta_offset + (rows - 1) * delta_row_stride + taps * num_groups;
  if (x.size() != rows * channels || y.size() != rows * channels ||
      base.size() != taps * channels ||
      delta_row_stride < taps * num_groups || delta.size() < delta_required) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    const std::size_t position = r % block_size;
    const std::size_t delta_row = delta_offset + r * delta_row_stride;
    for (std::size_t c = 0; c < channels; ++c) {
      const std::size_t grp = c / group_size;
      float acc = (base[c] + delta[delta_row + grp]) * x[r * channels + c];
      for (std::size_t tap = 1; tap < taps; ++tap) {
        if (position < tap) {
          continue;
        }
        const float coeff =
            base[tap * channels + c] +
            delta[delta_row + tap * num_groups + grp];
        acc = std::fma(coeff, x[(r - tap) * channels + c], acc);
      }
      y[r * channels + c] = acc;
    }
  }
  return {};
}

}  // namespace tessera::core
