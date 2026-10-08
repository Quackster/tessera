#include "core/loaders/hf_tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "core/files.hpp"
#include "core/json.hpp"

namespace tessera::core {

namespace {

// A nonnegative integral JSON number as an index.
std::expected<std::size_t, StatusCode> AsIndex(const Json& value) {
  if (value.type() != Json::Type::Number) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const double number = value.AsNumber();
  if (number < 0.0 || number != std::floor(number) || number > 1e15) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return static_cast<std::size_t>(number);
}

// Append the eos token ids in one document (`eos_token_id`, a number or an
// array of numbers) to `out`; invalid entries are skipped and duplicates
// are dropped.
void AppendEosIds(const Json& document, std::vector<std::uint32_t>& out) {
  const Json* eos = document.Find("eos_token_id");
  if (eos == nullptr) {
    return;
  }
  const auto append = [&out](const Json& value) {
    if (value.type() != Json::Type::Number) {
      return;
    }
    const double number = value.AsNumber();
    if (number < 0.0 || number != std::floor(number) ||
        number > 4294967295.0) {
      return;
    }
    const auto id = static_cast<std::uint32_t>(number);
    if (std::find(out.begin(), out.end(), id) == out.end()) {
      out.push_back(id);
    }
  };
  if (eos->isArray()) {
    for (const Json& value : eos->AsArray()) {
      append(value);
    }
  } else {
    append(*eos);
  }
}

}  // namespace

std::expected<std::optional<Tokenizer>, StatusCode> ParseHfTokenizer(
    std::string_view text) {
  auto document = Json::Parse(text);
  if (!document || !document->isObject()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const Json* model = document->Find("model");
  if (model == nullptr || !model->isObject()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const Json* type = model->Find("type");
  if (type == nullptr || !type->isString()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (type->AsString() != "BPE") {
    return std::optional<Tokenizer>{};
  }
  const Json* vocab = model->Find("vocab");
  if (vocab == nullptr || !vocab->isObject() || vocab->AsObject().empty()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  // Invert token to id. Ids are dense in practice but may have gaps.
  std::size_t max_id = 0;
  for (const auto& entry : vocab->AsObject()) {
    auto index = AsIndex(entry.second);
    if (!index) {
      return std::unexpected(index.error());
    }
    max_id = std::max(max_id, *index);
  }
  std::vector<std::string> tokens(max_id + 1);
  for (const auto& entry : vocab->AsObject()) {
    const std::size_t index = static_cast<std::size_t>(entry.second.AsNumber());
    tokens[index] = entry.first;
  }
  std::vector<std::int32_t> types(max_id + 1, 1);
  // The added tokens are the special strings (control type). They extend
  // the id range past the BPE vocabulary.
  const Json* added = document->Find("added_tokens");
  if (added != nullptr && added->isArray()) {
    for (const Json& entry : added->AsArray()) {
      const Json* content = entry.isObject() ? entry.Find("content") : nullptr;
      const Json* id = entry.isObject() ? entry.Find("id") : nullptr;
      if (content == nullptr || id == nullptr || !content->isString()) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      auto index = AsIndex(*id);
      if (!index) {
        return std::unexpected(index.error());
      }
      if (*index >= tokens.size()) {
        tokens.resize(*index + 1);
        types.resize(*index + 1, 1);
      }
      tokens[*index] = content->AsString();
      types[*index] = 3;
    }
  }
  std::vector<std::string> merges;
  const Json* merge_json = model->Find("merges");
  if (merge_json != nullptr && merge_json->isArray()) {
    merges.reserve(merge_json->AsArray().size());
    for (const Json& merge : merge_json->AsArray()) {
      if (merge.isString()) {
        merges.push_back(merge.AsString());
      } else if (merge.isArray() && merge.AsArray().size() == 2 &&
                 merge.AsArray()[0].isString() &&
                 merge.AsArray()[1].isString()) {
        merges.push_back(merge.AsArray()[0].AsString() + " " +
                         merge.AsArray()[1].AsString());
      } else {
        return std::unexpected(StatusCode::MalformedFile);
      }
    }
  }
  return std::optional<Tokenizer>{
      std::in_place, std::move(tokens), std::move(types), std::move(merges)};
}

std::expected<std::optional<Tokenizer>, StatusCode> LoadHfTokenizer(
    const std::filesystem::path& dir) {
  const std::filesystem::path path = dir / "tokenizer.json";
  std::error_code ec;
  if (!std::filesystem::exists(path, ec) || ec) {
    return std::optional<Tokenizer>{};
  }
  auto bytes = ReadFile(path);
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  const std::string_view text(reinterpret_cast<const char*>(bytes->data()),
                              bytes->size());
  return ParseHfTokenizer(text);
}

std::string LoadHfChatTemplate(const std::filesystem::path& dir) {
  auto config_bytes = ReadFile(dir / "tokenizer_config.json");
  if (config_bytes) {
    const std::string_view text(
        reinterpret_cast<const char*>(config_bytes->data()),
        config_bytes->size());
    auto document = Json::Parse(text);
    if (document && document->isObject()) {
      const Json* value = document->Find("chat_template");
      if (value != nullptr && value->isString()) {
        return value->AsString();
      }
    }
  }
  auto jinja = ReadFile(dir / "chat_template.jinja");
  if (jinja) {
    return std::string(reinterpret_cast<const char*>(jinja->data()),
                       jinja->size());
  }
  return std::string{};
}

std::vector<std::uint32_t> LoadHfStopTokens(
    const std::filesystem::path& dir) {
  std::vector<std::uint32_t> stops;
  for (const char* name : {"generation_config.json", "config.json"}) {
    auto bytes = ReadFile(dir / name);
    if (!bytes) {
      continue;
    }
    const std::string_view text(reinterpret_cast<const char*>(bytes->data()),
                                bytes->size());
    auto document = Json::Parse(text);
    if (document && document->isObject()) {
      AppendEosIds(*document, stops);
    }
  }
  return stops;
}

}  // namespace tessera::core
