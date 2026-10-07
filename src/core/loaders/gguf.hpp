#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "tessera/types.hpp"

namespace tessera::core {

// Scalar metadata values the GGUF parser supports. 8/16-bit values are
// widened to 32-bit. Array values are dropped with a warning (see
// ParseGguf).
using GgufValue = std::variant<bool, std::int32_t, std::uint32_t,
                              std::int64_t, std::uint64_t, float, double,
                              std::string>;

// Parsed GGUF file: header fields + metadata + tensor manifest.
struct GgufFile {
  std::uint32_t version = 0;
  std::unordered_map<std::string, GgufValue> metadata;
  std::vector<TensorEntry> tensors;
  // Byte offsets of each tensor relative to the start of the tensor
  // data (the aligned region after the tensor infos); parallel to
  // `tensors`.
  std::vector<std::uint64_t> tensor_offsets;
  // File offset where the tensor data starts (tensor_offsets are
  // relative to it); data.size() when the file holds no payload.
  std::uint64_t tensor_data_start = 0;
  // Keys of array metadata values dropped by the parser (not supported).
  std::vector<std::string> dropped_array_keys;
  // Array metadata retained by the parser: short arrays plus the
  // tokenizer definition arrays (tokens, token_type, merges) whatever
  // their size. Other large arrays stay dropped; see
  // dropped_array_keys.
  std::unordered_map<std::string, std::vector<GgufValue>> small_arrays;

  // Look up a metadata value by key; nullptr when absent.
  [[nodiscard]] const GgufValue* Find(std::string_view key) const;
};

// Parse a GGUF v2/v3 image from memory (little-endian; v2 and v3 share
// the byte layout, v3 additionally defines a big-endian encoding that
// this reader does not accept). Scalar metadata lands in metadata;
// arrays with few elements land in small_arrays, longer ones in
// dropped_array_keys.
//
// Rejects malformed input with MalformedFile: bad magic, truncated
// sections, out-of-bounds lengths, more than 4 dims, a bool value other
// than 0/1, an alignment that is not a power of two, a tensor offset
// that is not a multiple of the file alignment, a non-monotonic offset
// chain, or tensor data beyond the end of the file. Unsupported
// versions and unknown (but well-formed) tensor types return
// UnsupportedFeature.
//
// Usage:
//   auto bytes = ReadFile(path);
//   auto file = ParseGguf(bytes);
[[nodiscard]] std::expected<GgufFile, StatusCode> ParseGguf(
    std::span<const std::byte> data);

// Parse a .gguf file from disk (ReadFile + ParseGguf).
[[nodiscard]] std::expected<GgufFile, StatusCode> ParseGgufFile(
    const std::filesystem::path& path);

}  // namespace tessera::core
