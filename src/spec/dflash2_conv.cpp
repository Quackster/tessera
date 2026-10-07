#include "spec/dflash2_conv.hpp"

#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/conv.hpp"
#include "core/numerics/gemm.hpp"

namespace tessera::spec {

std::expected<void, StatusCode> GroupedConvRef(
    std::span<const float> hidden, std::span<const float> projection_w,
    std::span<const float> base_side, std::span<float> out, std::size_t rows,
    std::size_t channels, std::size_t taps, std::size_t group_size,
    std::size_t block_size, std::size_t side) {
  if (side > 1 || rows == 0 || channels == 0 || taps == 0 ||
      group_size == 0 || block_size == 0 || channels % group_size != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t num_groups = channels / group_size;
  const std::size_t proj_n = 2 * taps * num_groups;
  if (hidden.size() != rows * channels ||
      projection_w.size() != proj_n * channels ||
      base_side.size() != taps * channels || out.size() != rows * channels) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> proj(rows * proj_n);
  auto gemm = core::GemmF32Ref(hidden, projection_w, std::span<float>(proj),
                               rows, proj_n, channels);
  if (!gemm) {
    return std::unexpected(gemm.error());
  }
  return core::DflashConvRef(hidden, std::span<const float>(proj), base_side,
                             out, rows, channels, taps, group_size, block_size,
                             proj_n, side * taps * num_groups);
}

std::expected<void, StatusCode> GroupedConvDevice(
    Backend& backend, const Kernel& gemm, const Kernel& conv,
    Buffer& scratch_proj, Buffer& scratch_base, const Buffer& hidden,
    const Buffer& projection_w, const Buffer& base_kernel, Buffer& out,
    std::size_t rows, std::size_t channels, std::size_t taps,
    std::size_t group_size, std::size_t block_size, std::size_t side) {
  if (side > 1 || rows == 0 || channels == 0 || taps == 0 ||
      group_size == 0 || block_size == 0 || channels % group_size != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t num_groups = channels / group_size;
  const std::size_t proj_n = 2 * taps * num_groups;
  const std::size_t base_bytes = taps * channels * 4;
  if (scratch_proj.Size() < rows * proj_n * 4 ||
      scratch_base.Size() < base_bytes) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto projected = core::detail::ProjectDevice(backend, gemm, hidden,
                                               projection_w, scratch_proj,
                                               rows, proj_n, channels);
  if (!projected) {
    return std::unexpected(projected.error());
  }
  auto copy = backend.CopyD2D(base_kernel, side * base_bytes, scratch_base, 0,
                              base_bytes);
  if (!copy) {
    return std::unexpected(copy.error());
  }
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((rows * channels + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&hidden, &scratch_proj, &scratch_base, &out};
  launch.scalars = {rows,             channels,   taps,        group_size,
                    block_size,       proj_n,     side * taps * num_groups};
  return backend.LaunchKernel(conv, launch);
}

std::expected<void, StatusCode> GroupedConvFinishRef(
    std::span<const float> hidden, std::span<const float> delta,
    std::span<const float> base_side, std::span<float> out, std::size_t rows,
    std::size_t channels, std::size_t taps, std::size_t group_size,
    std::size_t block_size, std::size_t side) {
  if (side > 1 || rows == 0 || channels == 0 || taps == 0 ||
      group_size == 0 || block_size == 0 || channels % group_size != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t num_groups = channels / group_size;
  const std::size_t stride = 2 * taps * num_groups;
  return core::DflashConvRef(hidden, delta, base_side, out, rows, channels,
                             taps, group_size, block_size, stride,
                             side * taps * num_groups);
}

std::expected<void, StatusCode> GroupedConvFinishDevice(
    Backend& backend, const Kernel& conv, const Buffer& hidden,
    const Buffer& delta, const Buffer& base_kernel, Buffer& scratch_base,
    Buffer& out, std::size_t rows, std::size_t channels, std::size_t taps,
    std::size_t group_size, std::size_t block_size, std::size_t side) {
  if (side > 1 || rows == 0 || channels == 0 || taps == 0 ||
      group_size == 0 || block_size == 0 || channels % group_size != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t num_groups = channels / group_size;
  const std::size_t stride = 2 * taps * num_groups;
  const std::size_t base_bytes = taps * channels * 4;
  if (scratch_base.Size() < base_bytes) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto copy = backend.CopyD2D(base_kernel, side * base_bytes, scratch_base, 0,
                              base_bytes);
  if (!copy) {
    return std::unexpected(copy.error());
  }
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((rows * channels + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&hidden, &delta, &scratch_base, &out};
  launch.scalars = {rows,  channels, taps,
                    group_size, block_size, stride,
                    side * taps * num_groups};
  return backend.LaunchKernel(conv, launch);
}

}  // namespace tessera::spec
