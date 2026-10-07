#include "core/numerics/vision.hpp"

namespace tessera::core {

std::expected<void, StatusCode> PatchifyRef(
    std::span<const float> image, std::span<const float> mean,
    std::span<const float> std, std::span<float> out, std::size_t h,
    std::size_t w, std::size_t patch) {
  if (h == 0 || w == 0 || patch == 0 || h % patch != 0 || w % patch != 0 ||
      image.size() != h * w * 3 || mean.size() != 3 || std.size() != 3) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t nx = w / patch;
  const std::size_t ny = h / patch;
  const std::size_t per_patch = 3 * patch * patch;
  if (out.size() != nx * ny * per_patch) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t py = 0; py < ny; ++py) {
    for (std::size_t px = 0; px < nx; ++px) {
      float* patch_out = out.data() + (py * nx + px) * per_patch;
      for (std::size_t c = 0; c < 3; ++c) {
        for (std::size_t ph = 0; ph < patch; ++ph) {
          for (std::size_t pw = 0; pw < patch; ++pw) {
            const std::size_t y = py * patch + ph;
            const std::size_t x = px * patch + pw;
            const float value = image[(y * w + x) * 3 + c];
            patch_out[c * patch * patch + ph * patch + pw] =
                (value - mean[c]) / std[c];
          }
        }
      }
    }
  }
  return {};
}

std::expected<void, StatusCode> SpaceMergeRef(std::span<const float> x,
                                              std::span<float> out,
                                              std::size_t grid_h,
                                              std::size_t grid_w,
                                              std::size_t embed,
                                              std::size_t merge) {
  if (grid_h == 0 || grid_w == 0 || embed == 0 || merge == 0 ||
      grid_h % merge != 0 || grid_w % merge != 0 ||
      x.size() != grid_h * grid_w * embed) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t out_w = grid_w / merge;
  const std::size_t out_tokens = (grid_h / merge) * out_w;
  const std::size_t m2 = merge * merge;
  if (out.size() != out_tokens * m2 * embed) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t b = 0; b < out_tokens; ++b) {
    const std::size_t gbh = b / out_w;
    const std::size_t gbw = b % out_w;
    for (std::size_t mh = 0; mh < merge; ++mh) {
      for (std::size_t mw = 0; mw < merge; ++mw) {
        const std::size_t q = mh * merge + mw;
        const std::size_t in_row = (gbh * merge + mh) * grid_w + (gbw * merge + mw);
        for (std::size_t c = 0; c < embed; ++c) {
          out[b * m2 * embed + q * embed + c] = x[in_row * embed + c];
        }
      }
    }
  }
  return {};
}

}  // namespace tessera::core
