#include "core/vision_merger.hpp"

#include "core/decode_internal.hpp"
#include "core/numerics/gemm.hpp"
#include "core/numerics/norm.hpp"
#include "core/numerics/vision.hpp"

namespace tessera::core {

namespace {

using detail::BiasAddDevice;
using detail::ProjectDevice;

std::expected<void, StatusCode> GeluDevice(Backend& backend,
                                           const Kernel& kernel,
                                           const Buffer& x, Buffer& y,
                                           std::size_t n) {
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((n + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&x, &y};
  launch.scalars = {n};
  return backend.LaunchKernel(kernel, launch);
}

std::expected<void, StatusCode> SpatialMergeDevice(
    Backend& backend, const Kernel& kernel, const Buffer& x, Buffer& y,
    std::size_t embed, std::size_t grid_w, std::size_t grid_h,
    std::size_t merge) {
  const std::size_t out_tokens = (grid_h / merge) * (grid_w / merge);
  const std::size_t n = out_tokens * merge * merge * embed;
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((n + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&x, &y};
  launch.scalars = {embed, grid_w, grid_h, merge};
  return backend.LaunchKernel(kernel, launch);
}

}  // namespace

std::expected<void, StatusCode> VisionMergerRef(
    std::span<const float> x, std::span<const float> mm0_weight,
    std::span<const float> mm0_bias, std::span<const float> mm2_weight,
    std::span<const float> mm2_bias, std::span<float> out, std::size_t grid_h,
    std::size_t grid_w, std::size_t embed, std::size_t merge,
    std::size_t projection_dim) {
  const std::size_t m2 = merge * merge;
  const std::size_t out_tokens = (grid_h / merge) * (grid_w / merge);
  const std::size_t merged_dim = m2 * embed;
  if (merge == 0 || embed == 0 || projection_dim == 0 ||
      mm0_weight.size() != merged_dim * merged_dim ||
      mm0_bias.size() != merged_dim ||
      mm2_weight.size() != projection_dim * merged_dim ||
      mm2_bias.size() != projection_dim || out.size() != out_tokens * projection_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> merged(out_tokens * merged_dim);
  std::vector<float> mid(out_tokens * merged_dim);
  if (!SpaceMergeRef(x, std::span<float>(merged), grid_h, grid_w, embed,
                     merge) ||
      !GemmF32Ref(std::span<const float>(merged), mm0_weight,
                  std::span<float>(mid), out_tokens, merged_dim, merged_dim) ||
      !BiasAddRef(std::span<const float>(mid), mm0_bias, std::span<float>(mid),
                  out_tokens, merged_dim) ||
      !GeluRef(std::span<const float>(mid), std::span<float>(mid),
               out_tokens * merged_dim) ||
      !GemmF32Ref(std::span<const float>(mid), mm2_weight, out, out_tokens,
                  projection_dim, merged_dim) ||
      !BiasAddRef(out, mm2_bias, out, out_tokens, projection_dim)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return {};
}

std::expected<void, StatusCode> VisionMergerDevice(
    Backend& backend, const Kernel& gemm, const Kernel& gelu,
    const Kernel& bias_add, const Kernel& spatial_merge, const Buffer& x,
    const Buffer& mm0_weight, const Buffer& mm0_bias, const Buffer& mm2_weight,
    const Buffer& mm2_bias, Buffer& out, std::size_t grid_h,
    std::size_t grid_w, std::size_t embed, std::size_t merge,
    std::size_t projection_dim) {
  const std::size_t m2 = merge * merge;
  const std::size_t out_tokens = (grid_h / merge) * (grid_w / merge);
  const std::size_t merged_dim = m2 * embed;
  auto merged = backend.AllocateBuffer(out_tokens * merged_dim * 4,
                                       MemoryKind::Device);
  auto mid = backend.AllocateBuffer(out_tokens * merged_dim * 4,
                                    MemoryKind::Device);
  if (!merged || !mid) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  if (!SpatialMergeDevice(backend, spatial_merge, x, **merged, embed, grid_w,
                          grid_h, merge) ||
      !ProjectDevice(backend, gemm, **merged, mm0_weight, **mid, out_tokens,
                     merged_dim, merged_dim) ||
      !BiasAddDevice(backend, bias_add, **mid, mm0_bias, **mid, out_tokens,
                     merged_dim) ||
      !GeluDevice(backend, gelu, **mid, **mid, out_tokens * merged_dim) ||
      !ProjectDevice(backend, gemm, **mid, mm2_weight, out, out_tokens,
                     projection_dim, merged_dim) ||
      !BiasAddDevice(backend, bias_add, out, mm2_bias, out, out_tokens,
                     projection_dim)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

}  // namespace tessera::core
