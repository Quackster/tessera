#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tessera/types.hpp"

namespace tessera {

// Byte-level BPE tokenizer (the GPT-2/Qwen family). Built from a model
// definition: the vocabulary (id -> token bytes in the byte-level
// unicode form), the per-token GGUF types, and the merge rules in
// rank order. The Qwen pre-tokenization split is applied before merges.
//
// Preconditions: `vocab` is non-empty and `merges` holds "left right"
// pairs of byte-level tokens. Text must be UTF-8. Control/special
// tokens are not matched inside the input; Encode treats them as
// ordinary text. The split classifies bytes (bytes >= 0x80 count as
// letters) and does not apply NFC normalization; this matches the
// reference for ASCII, Latin and CJK text but not for combining marks
// or emoji.
//
// Usage:
//   Tokenizer tok(vocab, types, merges);
//   auto ids = tok.Encode("Hello, world!");
//   auto text = tok.Decode(*ids);
class Tokenizer {
 public:
  Tokenizer(std::vector<std::string> vocab,
            std::vector<std::int32_t> token_types,
            std::vector<std::string> merges);

  // Encode UTF-8 text to token ids (byte-level BPE over the Qwen
  // pre-tokenization split). InvalidArgument for malformed UTF-8 is not
  // raised; invalid bytes are passed through.
  [[nodiscard]] std::expected<std::vector<std::uint32_t>, StatusCode> Encode(
      std::string_view text) const;

  // Decode token ids back to UTF-8 bytes. InvalidArgument when an id is
  // out of range.
  [[nodiscard]] std::expected<std::string, StatusCode> Decode(
      std::span<const std::uint32_t> ids) const;

  [[nodiscard]] std::size_t VocabSize() const;

 private:
  std::vector<std::string> vocab_;
  std::vector<std::int32_t> token_types_;
  std::unordered_map<std::string, std::uint32_t> token_to_id_;
  // Merge rank keyed by "left\x01right".
  std::unordered_map<std::string, std::int32_t> merge_rank_;
};

}  // namespace tessera
