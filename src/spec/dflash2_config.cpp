#include "spec/dflash2_config.hpp"

#include <cmath>
#include <memory>

#include "core/files.hpp"
#include "core/json.hpp"

namespace tessera::spec {

namespace {

using core::Json;

// A positive integer field; MalformedFile when absent, not a number, not
// integral, or not positive.
std::expected<std::size_t, StatusCode> Positive(
    const Json& object, std::string_view key) {
  const Json* value = object.Find(key);
  if (value == nullptr || value->type() != Json::Type::Number) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const double number = value->AsNumber();
  if (!(number > 0.0) || number != std::floor(number) ||
      number > 1e15) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return static_cast<std::size_t>(number);
}

// A nonnegative integer field (zero allowed).
std::expected<std::size_t, StatusCode> NonNegative(
    const Json& object, std::string_view key) {
  const Json* value = object.Find(key);
  if (value == nullptr || value->type() != Json::Type::Number) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const double number = value->AsNumber();
  if (number < 0.0 || number != std::floor(number) || number > 1e15) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return static_cast<std::size_t>(number);
}

std::expected<double, StatusCode> PositiveReal(const Json& object,
                                               std::string_view key) {
  const Json* value = object.Find(key);
  if (value == nullptr || value->type() != Json::Type::Number ||
      !(value->AsNumber() > 0.0)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return value->AsNumber();
}

}  // namespace

std::expected<DFlash2Config, StatusCode> ParseDFlash2Config(
    std::string_view text) {
  auto document = Json::Parse(text);
  if (!document || !document->isObject()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const Json& root = *document;
  DFlash2Config config;

  auto hidden = Positive(root, "hidden_size");
  auto layers = Positive(root, "num_hidden_layers");
  auto heads = Positive(root, "num_attention_heads");
  auto kv_heads = Positive(root, "num_key_value_heads");
  auto head_dim = Positive(root, "head_dim");
  auto intermediate = Positive(root, "intermediate_size");
  auto vocab = Positive(root, "vocab_size");
  auto eps = PositiveReal(root, "rms_norm_eps");
  if (!hidden || !layers || !heads || !kv_heads || !head_dim ||
      !intermediate || !vocab || !eps) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  config.hidden_size = *hidden;
  config.num_layers = *layers;
  config.num_heads = *heads;
  config.num_kv_heads = *kv_heads;
  config.head_dim = *head_dim;
  config.intermediate_size = *intermediate;
  config.vocab_size = *vocab;
  config.rms_norm_eps = *eps;
  if (config.num_heads % config.num_kv_heads != 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto window = NonNegative(root, "sliding_window");
  if (!window) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  config.sliding_window = *window;
  const Json* rope = root.Find("rope_parameters");
  if (rope == nullptr || !rope->isObject()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto theta = PositiveReal(*rope, "rope_theta");
  if (!theta) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  config.rope_theta = *theta;

  const Json* dflash = root.Find("dflash_config");
  if (dflash == nullptr || !dflash->isObject()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto block = Positive(*dflash, "block_size");
  auto group = Positive(*dflash, "conv_group_size");
  auto taps = Positive(*dflash, "conv_kernel_size");
  auto mask = NonNegative(*dflash, "mask_token_id");
  auto rank = Positive(*dflash, "selector_rank");
  auto top_k = Positive(*dflash, "selector_top_k");
  if (!block || !group || !taps || !mask || !rank || !top_k) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  config.block_size = *block;
  config.conv_group_size = *group;
  config.conv_kernel_size = *taps;
  config.mask_token_id = static_cast<std::uint32_t>(*mask);
  config.selector_rank = *rank;
  config.selector_top_k = *top_k;
  if (config.hidden_size % config.conv_group_size != 0 ||
      config.hidden_size / config.conv_group_size < config.selector_top_k) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const Json* targets = dflash->Find("target_layer_ids");
  if (targets == nullptr || !targets->isArray() || targets->AsArray().empty()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  for (const Json& id : targets->AsArray()) {
    if (id.type() != Json::Type::Number || id.AsNumber() < 0.0 ||
        id.AsNumber() != std::floor(id.AsNumber())) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    config.target_layer_ids.push_back(
        static_cast<std::uint32_t>(id.AsNumber()));
  }
  const Json* types = root.Find("layer_types");
  if (types == nullptr || !types->isArray() ||
      types->AsArray().size() != config.num_layers) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  for (const Json& type : types->AsArray()) {
    if (type.type() != Json::Type::String) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    config.layer_types.push_back(type.AsString());
  }
  // Draft attention causality, mirroring vLLM's `_dflash_layer_causal`: an
  // explicit `is_causal` wins, else `dflash_config.causal`, else sliding
  // attention is causal and full attention is not.
  const Json* is_causal = root.Find("is_causal");
  const Json* causal_override = dflash->Find("causal");
  if (is_causal != nullptr && is_causal->type() == Json::Type::Bool) {
    config.attn_causal = is_causal->AsBool();
  } else if (causal_override != nullptr &&
             causal_override->type() == Json::Type::Bool) {
    config.attn_causal = causal_override->AsBool();
  } else {
    config.attn_causal = config.layer_types.front() == "sliding_attention";
  }
  return config;
}

std::expected<DFlash2Config, StatusCode> LoadDFlash2Config(
    const std::filesystem::path& dir) {
  auto bytes = core::ReadFile(dir / "config.json");
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  const std::string_view text(reinterpret_cast<const char*>(bytes->data()),
                              bytes->size());
  return ParseDFlash2Config(text);
}

}  // namespace tessera::spec
