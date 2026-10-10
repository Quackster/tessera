#include <cmath>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "core/json.hpp"
#include "models/qwen3_5/architecture.hpp"

// Parses the HuggingFace config.json of a Qwen3.5 MXFP4 checkpoint into
// the generic TransformerConfig. The text model lives under
// `text_config` in the multimodal config; a text-only config carries the
// same keys at the root. Every field is required; a malformed or missing
// field is MalformedFile.

namespace tessera::models::qwen3_5 {

namespace {

using core::Json;

std::expected<std::uint64_t, StatusCode> U64(const Json& object,
                                             std::string_view key) {
  const Json* value = object.Find(key);
  if (value == nullptr || value->type() != Json::Type::Number) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const double number = value->AsNumber();
  if (!(number > 0.0) || number != std::floor(number) || number > 1e15) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return static_cast<std::uint64_t>(number);
}

std::expected<double, StatusCode> Real(const Json& object,
                                       std::string_view key) {
  const Json* value = object.Find(key);
  if (value == nullptr || value->type() != Json::Type::Number ||
      !(value->AsNumber() > 0.0)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return value->AsNumber();
}

// The text-model sub-object; the root when the config is text-only.
const Json& TextConfig(const Json& root) {
  const Json* text = root.Find("text_config");
  if (text != nullptr && text->isObject()) {
    return *text;
  }
  return root;
}

// mRoPE section pair counts: three integers, or four with a zero pad (the
// file convention). The leading three cover at most rope_dim/2 pairs.
std::expected<std::vector<std::uint64_t>, StatusCode> RopeSections(
    const Json& text, std::size_t rope_dim) {
  const Json* rope = text.Find("rope_parameters");
  if (rope == nullptr || !rope->isObject()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const Json* sections = rope->Find("mrope_section");
  if (sections == nullptr || !sections->isArray()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::vector<std::uint64_t> counts;
  for (const Json& entry : sections->AsArray()) {
    if (entry.type() != Json::Type::Number || entry.AsNumber() < 0.0 ||
        entry.AsNumber() != std::floor(entry.AsNumber())) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    counts.push_back(static_cast<std::uint64_t>(entry.AsNumber()));
  }
  if (counts.size() == 4) {
    if (counts[3] != 0) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    counts.pop_back();
  }
  if (counts.size() != 3 || counts[0] + counts[1] + counts[2] > rope_dim / 2) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return counts;
}

}  // namespace

std::expected<TransformerConfig, StatusCode>
Qwen35Architecture::ParseConfigJson(std::string_view json) const {
  auto document = Json::Parse(json);
  if (!document || !document->isObject()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const Json& text = TextConfig(*document);

  auto hidden = U64(text, "hidden_size");
  auto layers = U64(text, "num_hidden_layers");
  auto heads = U64(text, "num_attention_heads");
  auto kv_heads = U64(text, "num_key_value_heads");
  auto head_dim = U64(text, "head_dim");
  auto vocab = U64(text, "vocab_size");
  // A dense definition carries intermediate_size; an MoE definition
  // carries moe_intermediate_size plus the expert counts instead.
  std::optional<std::uint64_t> ffn;
  if (text.Find("intermediate_size") != nullptr) {
    auto parsed = U64(text, "intermediate_size");
    if (!parsed) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    ffn = *parsed;
  }
  std::optional<std::uint64_t> moe_inter;
  std::optional<std::uint64_t> num_experts;
  std::optional<std::uint64_t> experts_per_tok;
  std::optional<std::uint64_t> shared_inter;
  const bool moe = text.Find("num_experts") != nullptr ||
                   text.Find("moe_intermediate_size") != nullptr;
  if (moe) {
    auto mi = U64(text, "moe_intermediate_size");
    auto ne = U64(text, "num_experts");
    auto nt = U64(text, "num_experts_per_tok");
    auto si = U64(text, "shared_expert_intermediate_size");
    if (!mi || !ne || !nt || !si || *nt > *ne) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    moe_inter = *mi;
    num_experts = *ne;
    experts_per_tok = *nt;
    shared_inter = *si;
  } else if (!ffn) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto eps = Real(text, "rms_norm_eps");
  auto interval = U64(text, "full_attention_interval");
  auto partial = Real(text, "partial_rotary_factor");
  auto conv = U64(text, "linear_conv_kernel_dim");
  auto state = U64(text, "linear_key_head_dim");
  auto key_heads = U64(text, "linear_num_key_heads");
  auto value_heads = U64(text, "linear_num_value_heads");
  auto value_dim = U64(text, "linear_value_head_dim");
  if (!hidden || !layers || !heads || !kv_heads || !head_dim ||
      !vocab || !eps || !interval || !partial || !conv || !state ||
      !key_heads || !value_heads || !value_dim) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (*heads % *kv_heads != 0 || *key_heads == 0 ||
      *value_heads % *key_heads != 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const double rope_exact = static_cast<double>(*head_dim) * *partial;
  const auto rope_dim = static_cast<std::size_t>(std::llround(rope_exact));
  if (rope_dim == 0 || rope_dim > *head_dim || (rope_dim % 2) != 0 ||
      std::abs(rope_exact - static_cast<double>(rope_dim)) > 1e-6) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const Json* rope = text.Find("rope_parameters");
  if (rope == nullptr || !rope->isObject()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto theta = Real(*rope, "rope_theta");
  if (!theta) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto sections = RopeSections(text, rope_dim);
  if (!sections) {
    return std::unexpected(sections.error());
  }

  TransformerConfig config;
  config.attention.heads = static_cast<std::size_t>(*heads);
  config.attention.kv_heads = static_cast<std::size_t>(*kv_heads);
  config.attention.head_dim = static_cast<std::size_t>(*head_dim);
  config.attention.rope_dim = rope_dim;
  config.attention.rope_theta = *theta;
  config.layers = static_cast<std::size_t>(*layers);
  config.hidden_dim = static_cast<std::size_t>(*hidden);
  config.ffn_dim = ffn ? static_cast<std::size_t>(*ffn) : 0;
  config.vocab_size = static_cast<std::size_t>(*vocab);
  config.norm_eps = *eps;
  if (moe) {
    config.num_experts = static_cast<std::size_t>(*num_experts);
    config.experts_per_tok = static_cast<std::size_t>(*experts_per_tok);
    config.moe_intermediate = static_cast<std::size_t>(*moe_inter);
    config.shared_expert_intermediate = static_cast<std::size_t>(*shared_inter);
  }
  config.hybrid = true;
  config.ssm.conv_kernel = static_cast<std::size_t>(*conv);
  config.ssm.state_size = static_cast<std::size_t>(*state);
  config.ssm.group_count = static_cast<std::size_t>(*key_heads);
  config.ssm.time_step_rank = static_cast<std::size_t>(*value_heads);
  config.ssm.inner_size =
      static_cast<std::size_t>(*value_heads * *value_dim);
  config.full_attention_interval = static_cast<std::size_t>(*interval);
  config.rope_sections = *sections;
  return config;
}

}  // namespace tessera::models::qwen3_5
