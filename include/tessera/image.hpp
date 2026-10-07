#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <vector>

#include "tessera/types.hpp"

namespace tessera {

// An RGB image with fp32 pixels in [0, 1], row-major (r, c, channel).
struct RgbImage {
  std::size_t width = 0;
  std::size_t height = 0;
  std::vector<float> pixels;
};

// Load a binary PPM (P6) image. A small dependency-free format for the
// vision path; no decompression is involved. FileNotFound when missing,
// MalformedFile for a bad header or truncated pixels.
[[nodiscard]] std::expected<RgbImage, StatusCode> LoadPpm(
    const std::filesystem::path& path);

// Bilinear resize `image` to `out_width` x `out_height`.
[[nodiscard]] RgbImage ResizeBilinear(const RgbImage& image,
                                      std::size_t out_width,
                                      std::size_t out_height);

}  // namespace tessera
