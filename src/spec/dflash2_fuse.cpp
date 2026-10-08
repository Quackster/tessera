#include "spec/dflash2_fuse.hpp"

#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/concat.hpp"
#include "core/numerics/gemm.hpp"
#include "core/numerics/quant.hpp"

namespace tessera::spec {

std::expected<void, StatusCode> DraftFuseRef(
    std::span<const float> aux, std::span<const float> fc_w,
    std::span<float> out, std::size_t n, std::size_t rows,
    std::size_t features, std::size_t hidden) {
  if (n == 0 || rows == 0 || features == 0 || hidden == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t width = n * features;
  if (aux.size() != n * rows * features || fc_w.size() != hidden * width ||
      out.size() != rows * hidden) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> concat(rows * width);
  std::vector<float> scale(rows);
  if (!core::ConcatFeaturesRef(aux, std::span<float>(concat), n, rows,
                               features) ||
      !core::QuantizeFp8Ref(std::span<float>(concat), std::span<float>(scale),
                            rows, width) ||
      !core::GemmF32Ref(std::span<const float>(concat), fc_w, out, rows,
                        hidden, width)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return {};
}

std::expected<void, StatusCode> DraftFuseDevice(
    Backend& backend, const Kernel& gemm, const Kernel& concat,
    const Kernel& quantize, Buffer& scratch, Buffer& scale, const Buffer& aux,
    const Buffer& fc_w, Buffer& out, std::size_t n, std::size_t rows,
    std::size_t features, std::size_t hidden) {
  if (n == 0 || rows == 0 || features == 0 || hidden == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t width = n * features;
  if (scratch.Size() < rows * width * 4 || scale.Size() < rows * 4) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  KernelLaunch concat_launch;
  concat_launch.grid_x =
      static_cast<std::uint32_t>((rows * width + 255) / 256);
  concat_launch.block_x = 256;
  concat_launch.buffers = {&aux, &scratch};
  concat_launch.scalars = {n, rows, features};
  auto concatenated = backend.LaunchKernel(concat, concat_launch);
  if (!concatenated) {
    return std::unexpected(concatenated.error());
  }
  KernelLaunch quantize_launch;
  quantize_launch.grid_x = static_cast<std::uint32_t>((rows + 255) / 256);
  quantize_launch.block_x = 256;
  quantize_launch.buffers = {&scratch, &scale};
  quantize_launch.scalars = {rows, width};
  auto quantized = backend.LaunchKernel(quantize, quantize_launch);
  if (!quantized) {
    return std::unexpected(quantized.error());
  }
  return core::detail::ProjectDevice(backend, gemm, scratch, fc_w, out, rows,
                                     hidden, width);
}

}  // namespace tessera::spec
