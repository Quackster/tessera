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

std::expected<void, StatusCode> MropeRef(
    std::span<float> io, std::span<const std::uint64_t> pos,
    std::size_t rows, std::size_t heads, std::size_t head_dim,
    std::size_t rope_dim, std::size_t sec_t, std::size_t sec_h,
    std::size_t sec_w, double theta) {
  if (rows == 0 || heads == 0 || head_dim == 0 || rope_dim == 0 ||
      rope_dim > head_dim || (rope_dim % 2) != 0 || !(theta > 0.0)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t pairs = rope_dim / 2;
  if (sec_t + sec_h + sec_w > pairs) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (io.size() != rows * heads * head_dim || pos.size() != rows * 3) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t h = 0; h < heads; ++h) {
      float* base = io.data() + (r * heads + h) * head_dim;
      for (std::size_t j = 0; j < pairs; ++j) {
        const std::uint64_t section_pos = j < sec_t ? pos[r * 3]
                                          : j < sec_t + sec_h ? pos[r * 3 + 1]
                                                              : pos[r * 3 + 2];
        const double angle =
            static_cast<double>(section_pos) *
            std::pow(theta, -2.0 * static_cast<double>(j) /
                                static_cast<double>(rope_dim));
        const float c = static_cast<float>(std::cos(angle));
        const float s = static_cast<float>(std::sin(angle));
        const float x1 = base[j];
        const float x2 = base[j + pairs];
        base[j] = x1 * c - x2 * s;
        base[j + pairs] = x1 * s + x2 * c;
      }
    }
  }
  return {};
}

std::expected<void, StatusCode> DeltaStepRef(
    std::span<float> s, std::span<const float> k, std::span<const float> v,
    std::span<const float> q, std::span<float> o, std::size_t dk,
    std::size_t dv, float alpha, float beta) {  if (dk == 0 || dv == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (s.size() != dk * dv || k.size() != dk || v.size() != dv ||
      q.size() != dk || o.size() != dv) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t d = 0; d < dv; ++d) {
    float read = 0.0f;
    for (std::size_t j = 0; j < dk; ++j) {
      read = std::fma(s[j * dv + d], k[j], read);
    }
    float out = 0.0f;
    for (std::size_t j = 0; j < dk; ++j) {
      const float updated = alpha * (s[j * dv + d] - beta * k[j] * read) +
                            beta * v[d] * k[j];
      s[j * dv + d] = updated;
      out = std::fma(updated, q[j], out);
    }
    o[d] = out;
  }
  return {};
}

std::expected<void, StatusCode> QGateSplitRef(
    std::span<const float> fused, std::span<float> q, std::span<float> gate,
    std::size_t heads, std::size_t head_dim) {
  if (heads == 0 || head_dim == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t per_head = heads * head_dim;
  if (fused.size() != per_head * 2 || q.size() != per_head ||
      gate.size() != per_head) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t h = 0; h < heads; ++h) {
    const float* src = fused.data() + h * 2 * head_dim;
    float* q_row = q.data() + h * head_dim;
    float* gate_row = gate.data() + h * head_dim;
    for (std::size_t e = 0; e < head_dim; ++e) {
      q_row[e] = src[e];
      gate_row[e] = src[head_dim + e];
    }
  }
  return {};
}

std::expected<void, StatusCode> RepeatHeadsRef(
    std::span<const float> in, std::span<float> out, std::size_t num_v_heads,
    std::size_t head_k_dim, std::size_t factor) {
  if (num_v_heads == 0 || head_k_dim == 0 || factor == 0 ||
      num_v_heads % factor != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t num_k_heads = num_v_heads / factor;
  if (in.size() != num_k_heads * head_k_dim ||
      out.size() != num_v_heads * head_k_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t h = 0; h < num_v_heads; ++h) {
    const std::size_t src_head = h / factor;
    for (std::size_t e = 0; e < head_k_dim; ++e) {
      out[h * head_k_dim + e] = in[src_head * head_k_dim + e];
    }
  }
  return {};
}

std::expected<void, StatusCode> DeltaStepHeadsRef(
    std::span<float> s, std::span<const float> k, std::span<const float> v,
    std::span<const float> q, std::span<float> o,
    std::span<const float> alpha, std::span<const float> beta,
    std::size_t heads, std::size_t dk, std::size_t dv) {
  if (heads == 0 || dk == 0 || dv == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (s.size() != heads * dk * dv || k.size() != heads * dk ||
      q.size() != heads * dk || v.size() != heads * dv ||
      o.size() != heads * dv || alpha.size() != heads ||
      beta.size() != heads) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t h = 0; h < heads; ++h) {
    float* state = s.data() + h * dk * dv;
    const float* k_h = k.data() + h * dk;
    const float* q_h = q.data() + h * dk;
    const float* v_h = v.data() + h * dv;
    float* o_h = o.data() + h * dv;
    for (std::size_t d = 0; d < dv; ++d) {
      float read = 0.0f;
      for (std::size_t j = 0; j < dk; ++j) {
        read = std::fma(state[j * dv + d], k_h[j], read);
      }
      float out = 0.0f;
      for (std::size_t j = 0; j < dk; ++j) {
        const float updated =
            alpha[h] * (state[j * dv + d] - beta[h] * k_h[j] * read) +
            beta[h] * v_h[d] * k_h[j];
        state[j * dv + d] = updated;
        out = std::fma(updated, q_h[j], out);
      }
      o_h[d] = out;
    }
  }
  return {};
}

}  // namespace tessera::core
