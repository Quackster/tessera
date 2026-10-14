#include "core/numerics/moe_cpu.hpp"

#include <cmath>
#include <vector>

#include "core/numerics/gemm.hpp"

namespace tessera::core {
namespace {

// Dispatch one expert projection to the host GEMM reference for its dtype.
// The CPU+GPU hybrid MoE uses this for the experts the device cache does not
// hold.
std::expected<void, StatusCode> ExpertGemm(DType dtype,
                                           std::span<const float> a,
                                           std::span<const std::byte> w,
                                           std::span<float> c, std::size_t m,
                                           std::size_t n, std::size_t k) {
  switch (dtype) {
  case DType::IQ2_S: return GemmIq2SRef(a, w, c, m, n, k);
  case DType::IQ3_XXS: return GemmIq3XxsRef(a, w, c, m, n, k);
  case DType::IQ3_S: return GemmIq3SRef(a, w, c, m, n, k);
  case DType::IQ4_XS: return GemmIq4XsRef(a, w, c, m, n, k);
  case DType::IQ4_NL: return GemmIq4NlRef(a, w, c, m, n, k);
  case DType::Q2_0: return GemmQ20Ref(a, w, c, m, n, k);
  case DType::Q4K: return GemmQ4KRef(a, w, c, m, n, k);
  case DType::Q5K: return GemmQ5KRef(a, w, c, m, n, k);
  case DType::Q6K: return GemmQ6KRef(a, w, c, m, n, k);
  case DType::Q80: return GemmQ80Ref(a, w, c, m, n, k);
  default: return std::unexpected(StatusCode::UnsupportedFeature);
  }
}

}  // namespace

std::expected<void, StatusCode> CpuMoeExpertForward(
    std::span<const float> x, DType gate_dtype,
    std::span<const std::byte> gate, DType up_dtype,
    std::span<const std::byte> up, DType down_dtype,
    std::span<const std::byte> down, std::size_t hidden, std::size_t inter,
    std::span<float> out) {
  if (hidden == 0 || inter == 0 || x.size() != hidden ||
      out.size() != hidden) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> gate_out(inter);
  std::vector<float> up_out(inter);
  if (auto s = ExpertGemm(gate_dtype, x, gate, gate_out, 1, inter, hidden);
      !s) {
    return s;
  }
  if (auto s = ExpertGemm(up_dtype, x, up, up_out, 1, inter, hidden); !s) {
    return s;
  }
  std::vector<float> phi(inter);
  for (std::size_t j = 0; j < inter; ++j) {
    const float g = gate_out[j];
    phi[j] = (g / (1.0f + std::exp(-g))) * up_out[j];
  }
  return ExpertGemm(down_dtype, phi, down, out, 1, hidden, inter);
}

}  // namespace tessera::core
