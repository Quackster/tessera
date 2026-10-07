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

std::expected<void, StatusCode> LayerNormRef(std::span<const float> x,
                                             std::span<const float> w,
                                             std::span<const float> b,
                                             std::span<float> y,
                                             std::size_t rows,
                                             std::size_t cols, float eps) {
  if (rows == 0 || cols == 0 || eps < 0.0f || x.size() != rows * cols ||
      w.size() != cols || b.size() != cols || y.size() != rows * cols) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = x.data() + r * cols;
    float mean = 0.0f;
    for (std::size_t c = 0; c < cols; ++c) {
      mean += xr[c];
    }
    mean /= static_cast<float>(cols);
    float var = 0.0f;
    for (std::size_t c = 0; c < cols; ++c) {
      const float d = xr[c] - mean;
      var = std::fma(d, d, var);
    }
    var /= static_cast<float>(cols);
    const float inv = 1.0f / std::sqrt(var + eps);
    float* yr = y.data() + r * cols;
    for (std::size_t c = 0; c < cols; ++c) {
      yr[c] = (xr[c] - mean) * inv * w[c] + b[c];
    }
  }
  return {};
}

std::expected<void, StatusCode> BiasAddRef(std::span<const float> x,
                                           std::span<const float> b,
                                           std::span<float> y,
                                           std::size_t rows,
                                           std::size_t cols) {
  if (rows == 0 || cols == 0 || x.size() != rows * cols || b.size() != cols ||
      y.size() != rows * cols) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      y[r * cols + c] = x[r * cols + c] + b[c];
    }
  }
  return {};
}

std::expected<void, StatusCode> GeluRef(std::span<const float> x,
                                        std::span<float> y, std::size_t n) {
  if (n == 0 || x.size() != n || y.size() != n) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t i = 0; i < n; ++i) {
    const float t = x[i];
    y[i] = 0.5f * t *
           (1.0f + std::tanh(0.7978845608f * (t + 0.044715f * t * t * t)));
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
                                          float eps, float scale) {
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
    const float gain = scale / std::sqrt(sum + eps);
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
    std::span<float> alpha, std::span<float> beta, std::size_t heads,
    std::size_t rows) {
  if (heads == 0 || rows == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (a_log.size() != heads || dt.size() != heads ||
      alpha_raw.size() != rows * heads || beta_raw.size() != rows * heads ||
      alpha.size() != rows * heads || beta.size() != rows * heads) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t h = 0; h < heads; ++h) {
      const std::size_t i = r * heads + h;
      const float raw = alpha_raw[i] + dt[h];
      const float softplus = raw > 20.0f ? raw : std::log1p(std::exp(raw));
      alpha[i] = std::exp(a_log[h] * softplus);
      beta[i] = 1.0f / (1.0f + std::exp(-beta_raw[i]));
    }
  }
  return {};
}

}  // namespace tessera::core
