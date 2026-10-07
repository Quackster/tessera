#include "core/numerics/norm.hpp"

#include <cmath>

namespace tessera::core {

std::expected<void, StatusCode> RmsNormRef(std::span<const float> x,
                                           std::span<const float> w,
                                           std::span<float> y,
                                           std::size_t rows,
                                           std::size_t cols, float eps) {
  if (rows == 0 || cols == 0 || eps < 0.0f) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (x.size() != rows * cols || w.size() != cols ||
      y.size() != rows * cols) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    const float* x_row = x.data() + r * cols;
    float mean = 0.0f;
    for (std::size_t c = 0; c < cols; ++c) {
      mean = std::fma(x_row[c], x_row[c], mean);
    }
    mean /= static_cast<float>(cols);
    const float gain = 1.0f / std::sqrt(mean + eps);
    float* y_row = y.data() + r * cols;
    for (std::size_t c = 0; c < cols; ++c) {
      y_row[c] = x_row[c] * gain * w[c];
    }
  }
  return {};
}

std::expected<void, StatusCode> SigmoidGateRef(std::span<const float> a,
                                               std::span<const float> g,
                                               std::span<float> out,
                                               std::size_t n) {
  if (n == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (a.size() != n || g.size() != n || out.size() != n) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = a[i] / (1.0f + std::exp(-g[i]));
  }
  return {};
}

std::expected<void, StatusCode> L2NormRef(std::span<const float> x,
                                          std::span<float> y,
                                          std::size_t rows, std::size_t cols,
                                          float eps) {
  if (rows == 0 || cols == 0 || eps < 0.0f) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (x.size() != rows * cols || y.size() != rows * cols) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    const float* x_row = x.data() + r * cols;
    float sum = 0.0f;
    for (std::size_t c = 0; c < cols; ++c) {
      sum = std::fma(x_row[c], x_row[c], sum);
    }
    const float gain = 1.0f / std::sqrt(sum + eps);
    float* y_row = y.data() + r * cols;
    for (std::size_t c = 0; c < cols; ++c) {
      y_row[c] = x_row[c] * gain;
    }
  }
  return {};
}

std::expected<void, StatusCode> RmsNormGatedRef(
    std::span<const float> x, std::span<const float> w,
    std::span<const float> gate, std::span<float> y, std::size_t rows,
    std::size_t cols, float eps) {
  if (rows == 0 || cols == 0 || eps < 0.0f) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (x.size() != rows * cols || gate.size() != rows * cols ||
      w.size() != cols || y.size() != rows * cols) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    const float* x_row = x.data() + r * cols;
    const float* gate_row = gate.data() + r * cols;
    float mean = 0.0f;
    for (std::size_t c = 0; c < cols; ++c) {
      mean = std::fma(x_row[c], x_row[c], mean);
    }
    mean /= static_cast<float>(cols);
    const float gain = 1.0f / std::sqrt(mean + eps);
    float* y_row = y.data() + r * cols;
    for (std::size_t c = 0; c < cols; ++c) {
      const float silu = gate_row[c] / (1.0f + std::exp(-gate_row[c]));
      y_row[c] = x_row[c] * gain * w[c] * silu;
    }
  }
  return {};
}

std::expected<void, StatusCode> AddRef(std::span<const float> a,
                                       std::span<const float> b,
                                       std::span<float> out, std::size_t n) {
  if (n == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (a.size() != n || b.size() != n || out.size() != n) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
  return {};
}

std::expected<void, StatusCode> SiluMulRef(std::span<const float> gate,
                                           std::span<const float> up,
                                           std::span<float> out,
                                           std::size_t n) {
  if (n == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (gate.size() != n || up.size() != n || out.size() != n) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t i = 0; i < n; ++i) {
    const float silu = gate[i] / (1.0f + std::exp(-gate[i]));
    out[i] = silu * up[i];
  }
  return {};
}

std::expected<void, StatusCode> SsmGateRef(
    std::span<const float> a_log, std::span<const float> dt,
    std::span<const float> alpha_raw, std::span<const float> beta_raw,
    std::span<float> alpha, std::span<float> beta, std::size_t heads) {
  if (heads == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (a_log.size() != heads || dt.size() != heads ||
      alpha_raw.size() != heads || beta_raw.size() != heads ||
      alpha.size() != heads || beta.size() != heads) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t h = 0; h < heads; ++h) {
    const float raw = alpha_raw[h] + dt[h];
    const float softplus = raw > 20.0f ? raw : std::log1p(std::exp(raw));
    alpha[h] = std::exp(-std::exp(a_log[h]) * softplus);
    beta[h] = 1.0f / (1.0f + std::exp(-beta_raw[h]));
  }
  return {};
}

}  // namespace tessera::core
