#include "models/qwen3_5/internal.hpp"

#include <cstdlib>

#include "core/decode_internal.hpp"
#include "models/qwen3_5/state.hpp"

// Qwen3.5 projection helper. With TESSERA_MXFP4_W4A8=1 an MXFP4 (F4E2M1)
// weight runs as W4A8: its activation is quantize-dequantized per token to
// FP8 E4M3 in place before the GEMM, matching the quantize step the served
// Qwen3.8 target applies to every MXFP4 linear input. It is off by default:
// on the short DFlash2 fixture it did not raise acceptance and it costs
// about 2.5x decode time, so it is opt-in pending a long-generation
// measurement. Non-MXFP4 weights are always unchanged.

namespace tessera::models::qwen3_5 {

namespace detail = ::tessera::core::detail;

std::expected<void, StatusCode> QuantizeMxFp4Input(
    Backend& backend, Qwen35State& h, DType dtype, Buffer& data,
    std::size_t rows, std::size_t cols) {
  if (dtype != DType::F4E2M1) {
    return {};
  }
  const char* enabled = std::getenv("TESSERA_MXFP4_W4A8");
  if (enabled == nullptr || std::atoi(enabled) == 0) {
    return {};
  }
  if (h.fp8_quant_kernel == nullptr) {
    auto kernel = backend.LoadKernel("quantize_fp8", {});
    if (!kernel) {
      return std::unexpected(kernel.error());
    }
    h.fp8_quant_kernel = std::move(*kernel);
  }
  if (h.fp8_scale == nullptr || h.fp8_scale->Size() < rows * 4) {
    auto scale = backend.AllocateBuffer(rows * 4, MemoryKind::Device);
    if (!scale) {
      return std::unexpected(scale.error());
    }
    h.fp8_scale = std::move(*scale);
  }
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((rows + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&data, h.fp8_scale.get()};
  launch.scalars = {rows, cols};
  return backend.LaunchKernel(*h.fp8_quant_kernel, launch);
}

std::expected<void, StatusCode> ProjectBatch(
    Backend& backend, Qwen35State& h, DType dtype, Buffer& a,
    const Buffer& w, Buffer& out, std::size_t m, std::size_t n,
    std::size_t k) {
  if (auto quantized = QuantizeMxFp4Input(backend, h, dtype, a, m, k);
      !quantized) {
    return quantized;
  }
  if (m > 1 && !detail::GemmTiledKernelName(dtype).empty()) {
    auto tiled = detail::GemmTiledFor(backend, h.gemm_tiled, dtype);
    if (tiled) {
      return detail::ProjectTiledDevice(backend, **tiled, a, w, out, m, n, k);
    }
    if (tiled.error() != StatusCode::UnsupportedFeature) {
      return std::unexpected(tiled.error());
    }
    // The backend has no tiled kernel for this dtype; use the GEMV one.
  }
  auto gemm = detail::GemmFor(backend, h.gemms, dtype);
  if (!gemm) {
    return std::unexpected(gemm.error());
  }
  return detail::ProjectDevice(backend, **gemm, a, w, out, m, n, k);
}

}  // namespace tessera::models::qwen3_5
