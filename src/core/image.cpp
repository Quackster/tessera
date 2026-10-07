#include "tessera/image.hpp"

#include <cctype>
#include <cstdint>

#include "core/files.hpp"

namespace tessera {

namespace {

// Read the next whitespace-delimited unsigned integer, skipping comments.
char CharAt(const std::byte* data, std::size_t pos) {
  return static_cast<char>(std::to_integer<unsigned char>(data[pos]));
}

bool ReadInt(const std::byte* data, std::size_t size, std::size_t& pos,
             std::size_t& value) {
  while (pos < size) {
    const char c = CharAt(data, pos);
    if (c == '#') {
      while (pos < size && CharAt(data, pos) != '\n') {
        ++pos;
      }
    } else if (std::isspace(static_cast<unsigned char>(c))) {
      ++pos;
    } else {
      break;
    }
  }
  if (pos >= size || !std::isdigit(static_cast<unsigned char>(CharAt(data, pos)))) {
    return false;
  }
  value = 0;
  while (pos < size &&
         std::isdigit(static_cast<unsigned char>(CharAt(data, pos)))) {
    value = value * 10 + static_cast<std::size_t>(CharAt(data, pos) - '0');
    ++pos;
  }
  return true;
}

}  // namespace

std::expected<RgbImage, StatusCode> LoadPpm(
    const std::filesystem::path& path) {
  auto bytes = core::ReadFile(path);
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  const std::byte* data = bytes->data();
  const std::size_t size = bytes->size();
  if (size < 2 || data[0] != std::byte{'P'} || data[1] != std::byte{'6'}) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::size_t pos = 2;
  std::size_t width = 0, height = 0, maxval = 0;
  if (!ReadInt(data, size, pos, width) || !ReadInt(data, size, pos, height) ||
      !ReadInt(data, size, pos, maxval) || width == 0 || height == 0 ||
      maxval == 0 || maxval > 255) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (pos >= size || !std::isspace(static_cast<unsigned char>(CharAt(data, pos)))) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  ++pos;  // one whitespace byte after maxval
  const std::size_t needed = width * height * 3;
  if (size - pos < needed) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  RgbImage image;
  image.width = width;
  image.height = height;
  image.pixels.resize(needed);
  const float scale = 1.0f / static_cast<float>(maxval);
  for (std::size_t i = 0; i < needed; ++i) {
    image.pixels[i] = static_cast<float>(static_cast<std::uint8_t>(data[pos + i])) *
                      scale;
  }
  return image;
}

RgbImage ResizeBilinear(const RgbImage& image, std::size_t out_width,
                        std::size_t out_height) {
  RgbImage out;
  out.width = out_width;
  out.height = out_height;
  out.pixels.resize(out_width * out_height * 3);
  if (image.width == 0 || image.height == 0 || out_width == 0 ||
      out_height == 0) {
    return out;
  }
  const float sx = static_cast<float>(image.width) / static_cast<float>(out_width);
  const float sy =
      static_cast<float>(image.height) / static_cast<float>(out_height);
  for (std::size_t y = 0; y < out_height; ++y) {
    for (std::size_t x = 0; x < out_width; ++x) {
      const float fx = (static_cast<float>(x) + 0.5f) * sx - 0.5f;
      const float fy = (static_cast<float>(y) + 0.5f) * sy - 0.5f;
      const float cx = fx < 0.0f ? 0.0f
                                 : (fx > static_cast<float>(image.width - 1)
                                        ? static_cast<float>(image.width - 1)
                                        : fx);
      const float cy = fy < 0.0f ? 0.0f
                                 : (fy > static_cast<float>(image.height - 1)
                                        ? static_cast<float>(image.height - 1)
                                        : fy);
      const std::size_t x0 = static_cast<std::size_t>(cx);
      const std::size_t y0 = static_cast<std::size_t>(cy);
      const std::size_t x1 = x0 + 1 < image.width ? x0 + 1 : x0;
      const std::size_t y1 = y0 + 1 < image.height ? y0 + 1 : y0;
      const float tx = cx - static_cast<float>(x0);
      const float ty = cy - static_cast<float>(y0);
      for (std::size_t c = 0; c < 3; ++c) {
        const auto at = [&](std::size_t r, std::size_t col) {
          return image.pixels[(r * image.width + col) * 3 + c];
        };
        const float top = at(y0, x0) * (1.0f - tx) + at(y0, x1) * tx;
        const float bottom = at(y1, x0) * (1.0f - tx) + at(y1, x1) * tx;
        out.pixels[(y * out_width + x) * 3 + c] =
            top * (1.0f - ty) + bottom * ty;
      }
    }
  }
  return out;
}

}  // namespace tessera
