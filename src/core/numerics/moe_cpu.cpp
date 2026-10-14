#include "core/numerics/moe_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
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

std::expected<void, StatusCode> CpuMoeExpertsForward(
    std::span<const float> x, std::span<const std::uint32_t> experts,
    std::span<const float> wts, DType gate_dtype,
    std::span<const std::byte> gate, std::size_t gate_per, DType up_dtype,
    std::span<const std::byte> up, std::size_t up_per, DType down_dtype,
    std::span<const std::byte> down, std::size_t down_per, std::size_t hidden,
    std::size_t inter, std::span<float> acc) {
  if (experts.size() != wts.size() || acc.size() != hidden ||
      x.size() != hidden) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (experts.empty()) {
    return {};
  }
  const unsigned hw = std::thread::hardware_concurrency();
  const unsigned cap = hw == 0 ? 1u : hw;
  const unsigned nthreads =
      std::max(1u, std::min<unsigned>(static_cast<unsigned>(experts.size()),
                                      cap));
  // One partial accumulator per thread; combined in thread order so the
  // result does not depend on the thread schedule.
  std::vector<std::vector<float>> partial(nthreads,
                                          std::vector<float>(hidden, 0.0f));
  std::atomic<bool> failed{false};
  std::vector<std::thread> pool;
  pool.reserve(nthreads);
  for (unsigned t = 0; t < nthreads; ++t) {
    pool.emplace_back([&, t]() {
      std::vector<float> out(hidden);
      for (std::size_t c = t; c < experts.size(); c += nthreads) {
        if (failed.load()) {
          return;
        }
        const std::size_t e = experts[c];
        auto s = CpuMoeExpertForward(
            x, gate_dtype, gate.subspan(e * gate_per, gate_per), up_dtype,
            up.subspan(e * up_per, up_per), down_dtype,
            down.subspan(e * down_per, down_per), hidden, inter, out);
        if (!s) {
          failed.store(true);
          return;
        }
        const float w = wts[c];
        for (std::size_t j = 0; j < hidden; ++j) {
          partial[t][j] += w * out[j];
        }
      }
    });
  }
  for (std::thread& thread : pool) {
    thread.join();
  }
  if (failed.load()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t j = 0; j < hidden; ++j) {
    float sum = acc[j];
    for (unsigned t = 0; t < nthreads; ++t) {
      sum += partial[t][j];
    }
    acc[j] = sum;
  }
  return {};
}

}  // namespace tessera::core
