#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "tessera/types.hpp"

namespace tessera::core {

// Host reference for the "rmsnorm" built-in: row-wise RMS normalization
// over rows x cols fp32 (y = x / sqrt(mean(x^2) + eps) * w, ascending
// accumulation, the same order the kernels use). QK-Norm applies this
// per attention head with the head dim as cols. Rows and cols must be
// nonzero, eps must not be negative, and the spans must hold rows*cols
// (x, y) and cols (w) elements; else InvalidArgument.
//
// Usage:
//   auto status = RmsNormRef(x, w, y, rows, cols, 1e-6f);
[[nodiscard]] std::expected<void, StatusCode> RmsNormRef(
    std::span<const float> x, std::span<const float> w, std::span<float> y,
    std::size_t rows, std::size_t cols, float eps);

// Host reference for the "layernorm" built-in: row-wise LayerNorm over
// rows x cols fp32 (y = (x - mean) / sqrt(var + eps) * w + b, biased
// variance, ascending accumulation, the order the kernels use). The
// CLIP vision encoder uses it. Rows and cols must be nonzero, eps must
// not be negative, and the spans must hold rows*cols (x, y) and cols
// (w, b); else InvalidArgument.
[[nodiscard]] std::expected<void, StatusCode> LayerNormRef(
    std::span<const float> x, std::span<const float> w,
    std::span<const float> b, std::span<float> y, std::size_t rows,
    std::size_t cols, float eps);

// Host reference for the "gelu" built-in: elementwise tanh-approximation
// GELU over n fp32. n must be nonzero; else InvalidArgument.
[[nodiscard]] std::expected<void, StatusCode> GeluRef(
    std::span<const float> x, std::span<float> y, std::size_t n);

// Host reference for the "sigmoid_gate" built-in: elementwise gated
// scale over n fp32 (out = a * sigmoid(g)). Gated attention applies
// this to the SDPA output with the fused gate projection. n must be
// nonzero and every span must hold n elements; else InvalidArgument.
//
// Usage:
//   auto status = SigmoidGateRef(a, g, out, n);
[[nodiscard]] std::expected<void, StatusCode> SigmoidGateRef(
    std::span<const float> a, std::span<const float> g, std::span<float> out,
    std::size_t n);

// Host reference for the "l2norm" built-in: row-wise L2 normalization
// over rows x cols fp32 (y = x / sqrt(sum(x^2) + eps), ascending
// accumulation, the order the kernels use). The gated delta rule
// normalizes its queries and keys this way before the scan. Rows and
// cols must be nonzero, eps must not be negative, and the spans must
// hold rows*cols elements; else InvalidArgument.
//
// Usage:
//   auto status = L2NormRef(x, y, rows, cols, 1e-6f);
[[nodiscard]] std::expected<void, StatusCode> L2NormRef(
    std::span<const float> x, std::span<float> y, std::size_t rows,
    std::size_t cols, float eps);

// Host reference for the "rmsnorm_gated" built-in: row-wise RMS
// normalization scaled by a SiLU gate over rows x cols fp32
// (y = x / sqrt(mean(x^2) + eps) * w * silu(gate)). The linear-
// attention block normalizes the scan output and gates it with the z
// projection. Rows and cols must be nonzero, eps must not be negative,
// and the spans must hold rows*cols (x, gate) and cols (w); else
// InvalidArgument.
//
// Usage:
//   auto status = RmsNormGatedRef(x, w, gate, y, rows, cols, 1e-6f);
[[nodiscard]] std::expected<void, StatusCode> RmsNormGatedRef(
    std::span<const float> x, std::span<const float> w,
    std::span<const float> gate, std::span<float> y, std::size_t rows,
    std::size_t cols, float eps);

// Host reference for the "add" built-in: elementwise a + b over n fp32.
// Residual connections use this on the device. n must be nonzero and
// every span must hold n elements; else InvalidArgument.
[[nodiscard]] std::expected<void, StatusCode> AddRef(
    std::span<const float> a, std::span<const float> b, std::span<float> out,
    std::size_t n);

// Host reference for the "silu_mul" built-in: out = silu(gate) * up over
// n fp32. The gated MLP uses this on the device. n must be nonzero and
// every span must hold n elements; else InvalidArgument.
[[nodiscard]] std::expected<void, StatusCode> SiluMulRef(
    std::span<const float> gate, std::span<const float> up,
    std::span<float> out, std::size_t n);

// Host reference for the "ssm_gate" built-in: the gated-delta decay and
// write gates over `heads` value heads.
//   alpha = exp(-exp(a_log) * softplus(alpha_raw + dt))
//   beta  = sigmoid(beta_raw)
// Each span holds `heads` elements; heads must be nonzero.
[[nodiscard]] std::expected<void, StatusCode> SsmGateRef(
    std::span<const float> a_log, std::span<const float> dt,
    std::span<const float> alpha_raw, std::span<const float> beta_raw,
    std::span<float> alpha, std::span<float> beta, std::size_t heads);

}  // namespace tessera::core
