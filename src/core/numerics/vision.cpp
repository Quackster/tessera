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

}  // namespace tessera::core
