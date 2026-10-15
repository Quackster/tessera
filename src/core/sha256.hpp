#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>

#include "tessera/types.hpp"

// SHA-256 (FIPS 180-4), used by the MTP head fetch to check each downloaded
// tensor against the pinned checkpoint's digest. Small and dependency-free
// so the engine keeps no crypto library dependency.
//
// Thread-safety: one hasher is used by one thread; the static helpers are
// safe for concurrent use.
namespace tessera::core {

class Sha256 {
 public:
  Sha256();

  // Feed `data` into the hash.
  void Update(std::span<const std::byte> data);

  // Finalize and return the lowercase hex digest. The hasher must not be
  // used afterwards.
  [[nodiscard]] std::string HexDigest();

  // Hash a whole byte range in one call.
  [[nodiscard]] static std::string Hex(std::span<const std::byte> data);

 private:
  void Transform(const std::uint8_t* block);

  std::array<std::uint32_t, 8> state_;
  std::array<std::uint8_t, 64> buffer_{};
  std::uint64_t bit_count_ = 0;
  std::size_t buffer_len_ = 0;
};

// SHA-256 of a whole file, streamed so a multi-gigabyte tensor is never
// held in memory.
//
// Usage:
//   auto digest = Sha256File(path);
[[nodiscard]] std::expected<std::string, StatusCode> Sha256File(
    const std::filesystem::path& path);

}  // namespace tessera::core
