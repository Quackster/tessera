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

// A read-only memory map of a whole file. It avoids the copy and the
// anonymous allocation of ReadFile, which matters for a multi-gigabyte
// checkpoint. Move-only; the mapping stays valid until destruction.
class MappedFile {
 public:
  MappedFile() = default;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;
  ~MappedFile();

  [[nodiscard]] static std::expected<MappedFile, StatusCode> Open(
      const std::filesystem::path& path);

  [[nodiscard]] std::span<const std::byte> bytes() const {
    return std::span<const std::byte>(data_, size_);
  }

 private:
  void Reset();
  const std::byte* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace tessera::core
