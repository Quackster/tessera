#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <span>
#include <vector>

#include "tessera/types.hpp"

namespace tessera::core {

// Read a whole file into memory.
// Returns FileNotFound when the path does not exist; MalformedFile when
// the file cannot be read as a regular file.
[[nodiscard]] std::expected<std::vector<std::byte>, StatusCode> ReadFile(
    const std::filesystem::path& path);

// Write bytes to a file, creating it if needed. MalformedFile on I/O
// error (e.g. the path points at a directory).
[[nodiscard]] std::expected<void, StatusCode> WriteFile(
    const std::filesystem::path& path, std::span<const std::byte> data);

}  // namespace tessera::core
