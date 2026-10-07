#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "tessera/types.hpp"

namespace tessera::core {

// Result of validating one safetensors container.
struct SafetensorsFileCheck {
  std::size_t file_size = 0;
  std::uint64_t header_len = 0;
};

// Validate a safetensors container: an 8-byte little-endian header
// length, the header within the file bounds, and a header that starts
// with '{' (JSON). Returns MalformedFile on any violation, FileNotFound
// when the path is missing.
[[nodiscard]] std::expected<SafetensorsFileCheck, StatusCode>
InspectSafetensorsFile(const std::filesystem::path& path);

// One tensor from the header map: manifest entry plus absolute file
// offsets of its payload (offsets count from the file start, so the
// payload is data[begin..end]).
struct SafetensorsTensor {
  std::string name;
  TensorEntry entry;
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
};

// Parse the header tensor map from whole file bytes: `{name:
// {dtype, shape, data_offsets}}`. Dtype strings map to DType (U8
// blobs with a paired scale become F4E2M1, U8 scales become F8E8M0).
// MalformedFile for schema violations, out-of-range offsets, or
// uninterpretable entries; UnsupportedFeature for unknown dtypes.
//
// Usage:
//   auto bytes = ReadFile(path);
//   auto map = ParseSafetensorsMap(*bytes);
[[nodiscard]] std::expected<std::vector<SafetensorsTensor>, StatusCode>
ParseSafetensorsMap(std::span<const std::byte> data);

// Layout of an MXFP4 (or DFlash2 draft) model directory.
struct MxFp4Layout {
  std::string weights_path;
  std::uint64_t header_len = 0;
  std::size_t file_size = 0;
};

// Validate a model directory: config.json present and exactly one
// *.safetensors (several weight files are a malformed layout).
// FileNotFound for a missing directory, MalformedFile for a bad layout.
[[nodiscard]] std::expected<MxFp4Layout, StatusCode> InspectMxFp4Directory(
    const std::filesystem::path& dir);

}  // namespace tessera::core
