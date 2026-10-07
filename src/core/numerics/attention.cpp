#include "core/numerics/attention.hpp"

#include <cmath>
#include <vector>

namespace tessera::core {

std::expected<void, StatusCode> RopeRef(std::span<float> io,
                                        std::size_t rows, std::size_t heads,
                                        std::size_t head_dim,
                                        std::size_t rope_dim,
                                        std::uint64_t pos_base,
                                        double theta) {
  if (rows == 0 || heads == 0 || head_dim == 0 || rope_dim == 0 ||
      rope_dim > head_dim || (rope_dim % 2) != 0 || !(theta > 0.0)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (io.size() != rows * heads * head_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t half = rope_dim / 2;
  for (std::size_t r = 0; r < rows; ++r) {
    const double pos = static_cast<double>(pos_base + r);
    for (std::size_t h = 0; h < heads; ++h) {
      float* base = io.data() + (r * heads + h) * head_dim;
      for (std::size_t j = 0; j < half; ++j) {
        const double angle =
            pos * std::pow(theta, -2.0 * static_cast<double>(j) /
                                      static_cast<double>(rope_dim));
        const float c = static_cast<float>(std::cos(angle));
        const float s = static_cast<float>(std::sin(angle));
        const float x1 = base[j];
        const float x2 = base[j + half];
        base[j] = x1 * c - x2 * s;
        base[j + half] = x1 * s + x2 * c;
      }
    }
  }
  return {};
}

std::expected<void, StatusCode> AttentionRef(
    std::span<const float> q, std::span<const float> k,
    std::span<const float> v, std::span<float> out, std::size_t m,
    std::size_t n, std::size_t heads, std::size_t kv_heads,
    std::size_t head_dim, std::uint64_t q_base) {
  if (m == 0 || n == 0 || heads == 0 || kv_heads == 0 || head_dim == 0 ||
      (heads % kv_heads) != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (q.size() != m * heads * head_dim || out.size() != m * heads * head_dim ||
      k.size() != n * kv_heads * head_dim ||
      v.size() != n * kv_heads * head_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t group = heads / kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> scores(n);
  for (std::size_t i = 0; i < m; ++i) {
    const std::size_t last =
        q_base + i >= n ? n - 1 : static_cast<std::size_t>(q_base + i);
    for (std::size_t h = 0; h < heads; ++h) {
      const std::size_t kv = h / group;
      const float* q_row = q.data() + (i * heads + h) * head_dim;
      float row_max = 0.0f;
      bool first = true;
      for (std::size_t j = 0; j <= last; ++j) {
        const float* k_row = k.data() + (j * kv_heads + kv) * head_dim;
        float dot = 0.0f;
        for (std::size_t e = 0; e < head_dim; ++e) {
          dot = std::fma(q_row[e], k_row[e], dot);
        }
        dot *= scale;
        scores[j] = dot;
        if (first || dot > row_max) {
          row_max = dot;
          first = false;
        }
      }
      float denom = 0.0f;
      for (std::size_t j = 0; j <= last; ++j) {
        scores[j] = std::exp(scores[j] - row_max);
        denom += scores[j];
      }
      float* o_row = out.data() + (i * heads + h) * head_dim;
      for (std::size_t e = 0; e < head_dim; ++e) {
        float acc = 0.0f;
        for (std::size_t j = 0; j <= last; ++j) {
          const float* v_row = v.data() + (j * kv_heads + kv) * head_dim;
          acc = std::fma(scores[j] / denom, v_row[e], acc);
        }
        o_row[e] = acc;
      }
    }
  }
  return {};
}

}  // namespace tessera::core
