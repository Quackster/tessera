#include "core/numerics/ple.hpp"

#include <cmath>
#include <vector>

#include "core/numerics/quant.hpp"

namespace tessera::core {

namespace {

float Silu(float x) { return static_cast<float>(x / (1.0 + std::exp(-x))); }

// Per-stream RMS norm: rows of `n_embd`, gamma indexed per row.
void GroupedRms(std::span<const float> input, std::span<const float> gamma,
                std::size_t n_embd, std::size_t hc, double eps,
                std::span<float> out) {
  for (std::size_t r = 0; r < hc; ++r) {
    double ss = 0.0;
    for (std::size_t d = 0; d < n_embd; ++d) {
      const float v = input[r * n_embd + d];
      ss += static_cast<double>(v) * v;
    }
    const float mean = static_cast<float>(ss / static_cast<double>(n_embd));
    const float scale = 1.0f / std::sqrt(mean + static_cast<float>(eps));
    for (std::size_t d = 0; d < n_embd; ++d) {
      out[r * n_embd + d] = scale * input[r * n_embd + d] *
                            gamma[r * n_embd + d];
    }
  }
}

}  // namespace

std::expected<void, StatusCode> PlePostopsRef(
    std::span<const float> projected_key, std::span<const float> hidden,
    std::span<const float> value, std::span<const float> history,
    std::span<const float> norm_key, std::span<const float> norm_query,
    std::span<const float> norm_conv, std::span<const std::byte> conv1d,
    const PlePostopsShapes& s, double eps, std::span<float> result,
    std::span<float> gate, std::span<float> history_out) {
  const std::size_t n_embd = s.n_embd;
  const std::size_t hc = s.hc;
  const std::size_t taps = s.conv_taps;
  const std::size_t hist = s.history_rows;
  if (n_embd == 0 || hc == 0 || taps < 2 || hist == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t dim = hc * n_embd;
  if (projected_key.size() != dim || hidden.size() != dim ||
      value.size() != n_embd || history.size() != dim * hist ||
      norm_key.size() != dim || norm_query.size() != dim ||
      norm_conv.size() != dim ||
      conv1d.size() != dim * taps * 2 || result.size() != dim ||
      gate.size() != hc || history_out.size() != dim * hist) {
    return std::unexpected(StatusCode::InvalidArgument);
  }

  std::vector<float> key(dim), query(dim), gated(dim), normalized(dim);
  GroupedRms(projected_key, norm_key, n_embd, hc, eps,
             std::span<float>(key));
  GroupedRms(hidden, norm_query, n_embd, hc, eps, std::span<float>(query));

  const float scale = 1.0f / std::sqrt(static_cast<float>(n_embd));
  for (std::size_t c = 0; c < hc; ++c) {
    double sum = 0.0;
    for (std::size_t d = 0; d < n_embd; ++d) {
      sum += static_cast<double>(key[c * n_embd + d]) * query[c * n_embd + d];
    }
    const float s_c = scale * static_cast<float>(sum);
    const float mag = std::sqrt(std::max(std::abs(s_c), 1e-6f));
    const float sign = static_cast<float>((s_c > 0.0f) - (s_c < 0.0f));
    gate[c] = 1.0f / (1.0f + std::exp(-sign * mag));
  }
  for (std::size_t i = 0; i < dim; ++i) {
    gated[i] = value[i % n_embd] * gate[i / n_embd];
  }
  GroupedRms(std::span<const float>(gated), norm_conv, n_embd, hc, eps,
             std::span<float>(normalized));

  const std::size_t stride = taps - 1;
  for (std::size_t c = 0; c < dim; ++c) {
    float sum = 0.0f;
    for (std::size_t k = 0; k < taps; ++k) {
      const float x =
          k == taps - 1 ? normalized[c] : history[c * hist + stride * k];
      const std::uint16_t bits = static_cast<std::uint16_t>(
          static_cast<std::uint8_t>(conv1d[(c * taps + k) * 2]) |
          (static_cast<std::uint16_t>(static_cast<std::uint8_t>(
               conv1d[(c * taps + k) * 2 + 1]))
           << 8));
      const float term = x * Fp16ToFloat(bits);
      sum = k == 0 ? term : sum + term;
    }
    const float activation = Silu(sum);
    result[c] = hidden[c] + (gated[c] + activation);
  }

  for (std::size_t c = 0; c < dim; ++c) {
    for (std::size_t r = 0; r + 1 < hist; ++r) {
      history_out[c * hist + r] = history[c * hist + r + 1];
    }
    history_out[c * hist + hist - 1] = normalized[c];
  }
  return {};
}

}  // namespace tessera::core
