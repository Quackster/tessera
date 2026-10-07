#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>

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
//
// The tensor map inside the JSON is parsed in milestone 6
// (docs/PROGRESS.md); this check only establishes the container layout.
[[nodiscard]] std::expected<SafetensorsFileCheck, StatusCode>
InspectSafetensorsFile(const std::filesystem::path& path);

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
