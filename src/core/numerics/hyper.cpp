#include "core/numerics/hyper.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace tessera::core {

namespace {

// fp32 to bfloat16 bits, round to nearest even (the "round_bf16" kernel
// and the reference rounding of the default activation contract).
std::uint16_t F32ToBf16Bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>(bits >> 16);
}

float Bf16BitsToF32(std::uint16_t bits) {
  const std::uint32_t value = static_cast<std::uint32_t>(bits) << 16;
  float out = 0.0f;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

std::uint16_t ReadU16(std::span<const std::byte> bytes, std::size_t index) {
  return static_cast<std::uint16_t>(
      static_cast<std::uint8_t>(bytes[index * 2]) |
      (static_cast<std::uint16_t>(static_cast<std::uint8_t>(bytes[index * 2 + 1]))
       << 8));
}

float Bf16At(std::span<const std::byte> bytes, std::size_t index) {
  return Bf16BitsToF32(ReadU16(bytes, index));
}

float Sigmoid(double x) {
  return static_cast<float>(1.0 / (1.0 + std::exp(-x)));
}

float Silu(double x) {
  return static_cast<float>(x / (1.0 + std::exp(-x)));
}

}  // namespace

HyperConnectionScratch HyperConnectionScratchFor(
    const HyperConnectionShapes& s) {
  const std::size_t hc_dim = s.hc * s.n_embd;
  HyperConnectionScratch out;
  out.xn_floats = hc_dim;
  out.xq_u16 = hc_dim;
  out.lq_u16 = s.hc_lr;
  out.gated_floats = hc_dim;
  out.lo_floats = s.hc_lr;
  return out;
}

std::expected<void, StatusCode> HyperConnectionReadRef(
    std::span<const float> r, std::span<const float> w_norm,
    std::span<const std::byte> w_down, std::span<const std::byte> w_up,
    std::span<const std::byte> w_inject, double eps,
    const HyperConnectionShapes& s, std::span<float> mixed,
    std::span<float> inject) {
  if (s.n_embd == 0 || s.hc == 0 || s.hc_lr == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t n_embd = s.n_embd;
  const std::size_t hc = s.hc;
  const std::size_t hc_lr = s.hc_lr;
  const std::size_t hc_dim = hc * n_embd;
  if (r.size() != hc_dim || w_norm.size() != hc_dim ||
      w_down.size() != hc_lr * hc_dim * 2 ||
      w_up.size() != hc_dim * hc_lr * 2 || mixed.size() != n_embd ||
      inject.size() != hc ||
      (!w_inject.empty() && w_inject.size() != hc * hc_dim * 2)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }

  std::vector<float> xn(hc_dim, 0.0f);
  std::vector<std::uint16_t> xq(hc_dim, 0);
  for (std::size_t c = 0; c < hc; ++c) {
    const float* rc = r.data() + c * n_embd;
    double ss = 0.0;
    for (std::size_t d = 0; d < n_embd; ++d) {
      ss += static_cast<double>(rc[d]) * rc[d];
    }
    const float mean = static_cast<float>(ss / static_cast<double>(n_embd));
    const float rs = 1.0f / std::sqrt(mean + static_cast<float>(eps));
    for (std::size_t d = 0; d < n_embd; ++d) {
      const std::size_t i = c * n_embd + d;
      const float x = rc[d] * rs * w_norm[i];
      xn[i] = x;
      xq[i] = F32ToBf16Bits(x);
    }
  }

  std::vector<float> lo(hc_lr, 0.0f);
  std::vector<std::uint16_t> lq(hc_lr, 0);
  for (std::size_t k = 0; k < hc_lr; ++k) {
    double acc = 0.0;
    for (std::size_t i = 0; i < hc_dim; ++i) {
      acc += static_cast<double>(Bf16BitsToF32(xq[i])) *
             Bf16At(w_down, k * hc_dim + i);
    }
    lo[k] = Silu(acc / static_cast<double>(hc));
    lq[k] = F32ToBf16Bits(lo[k]);
  }

  std::vector<float> gated(hc_dim, 0.0f);
  for (std::size_t i = 0; i < hc_dim; ++i) {
    double acc = 0.0;
    for (std::size_t k = 0; k < hc_lr; ++k) {
      acc += static_cast<double>(Bf16BitsToF32(lq[k])) *
             Bf16At(w_up, i * hc_lr + k);
    }
    gated[i] = xn[i] * Sigmoid(acc);
  }
  for (std::size_t d = 0; d < n_embd; ++d) {
    double m = 0.0;
    for (std::size_t c = 0; c < hc; ++c) {
      m += gated[c * n_embd + d];
    }
    mixed[d] = static_cast<float>(m / static_cast<double>(hc));
  }

  if (!w_inject.empty()) {
    for (std::size_t c = 0; c < hc; ++c) {
      double acc = 0.0;
      for (std::size_t i = 0; i < hc_dim; ++i) {
        acc += static_cast<double>(Bf16BitsToF32(xq[i])) *
               Bf16At(w_inject, c * hc_dim + i);
      }
      inject[c] = static_cast<float>(acc);
    }
  }
  return {};
}

std::expected<void, StatusCode> HyperConnectionWriteRef(
    std::span<const float> r, std::span<const float> block_out,
    std::span<const float> inject, const HyperConnectionShapes& s,
    std::span<float> out) {
  if (s.n_embd == 0 || s.hc == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t hc_dim = s.hc * s.n_embd;
  if (r.size() != hc_dim || block_out.size() != s.n_embd ||
      inject.size() != s.hc || out.size() != hc_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t c = 0; c < s.hc; ++c) {
    const float w =
        2.0f * Sigmoid(static_cast<double>(inject[c]) /
                       static_cast<double>(s.hc));
    for (std::size_t d = 0; d < s.n_embd; ++d) {
      const std::size_t i = c * s.n_embd + d;
      out[i] = r[i] + block_out[d] * w;
    }
  }
  return {};
}

}  // namespace tessera::core
