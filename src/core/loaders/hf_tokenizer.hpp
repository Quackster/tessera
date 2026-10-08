#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "tessera/tokenizer.hpp"
#include "tessera/types.hpp"

namespace tessera::core {

// Build a byte-level BPE tokenizer from a HuggingFace `tokenizer.json`.
// The Qwen export stores byte-level tokens in `model.vocab` (token string
// to id), the merge rules in `model.merges` ("left right" in rank order)
// and the special tokens in `added_tokens` (control type). Returns
// nullopt when the model type is not BPE. MalformedFile for malformed or
// inconsistent JSON, for example a negative or non-integral id.
[[nodiscard]] std::expected<std::optional<Tokenizer>, StatusCode>
ParseHfTokenizer(std::string_view text);

// Read `dir/tokenizer.json` and parse it. nullopt when the file is absent
// (a model without a tokenizer), FileNotFound for an unreadable dir.
[[nodiscard]] std::expected<std::optional<Tokenizer>, StatusCode>
LoadHfTokenizer(const std::filesystem::path& dir);

// Read the chat template for a HuggingFace model directory: the
// `chat_template` string in `tokenizer_config.json`, else the contents of
// `chat_template.jinja`. Empty when neither is present or parsable.
[[nodiscard]] std::string LoadHfChatTemplate(const std::filesystem::path& dir);

}  // namespace tessera::core
