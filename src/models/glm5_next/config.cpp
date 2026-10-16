#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/json.hpp"
#include "models/glm5_next/architecture.hpp"
#include "models/glm5_next/internal.hpp"

// Parses the HuggingFace config.json of a GLM-5-Next checkpoint into the
// generic TransformerConfig. The GLM-5.3-Flash config is multimodal and
// keeps the text model under `text_config`; a text-only config carries
// the same keys at the root. Every GLM-specific group (MLA, KDA, indexer,
// mHC, MoE) is required: the module only runs this architecture.

namespace tessera::models::glm5_next {

namespace {

using core::Json;
using core::JsonBool;
using core::JsonReal;
using core::JsonString;
using core::JsonTextConfig;
using core::JsonU64;

// The indexer k_norm is a LayerNorm with a fixed epsilon (1e-6) in the
// checkpoint contract (transformers Glm5NextIndexer); the GGUF carries
// the same value as attention.layer_norm_epsilon.
constexpr double kIndexerLayerNormEps = 1e-6;

// Config numbers above 2^53 cannot hold every integer exactly; the cap
// keeps the conversion to uint64 exact and rejects absurd dimensions.
constexpr double kMaxExactInteger = 1e15;

// GLM-5.3-Flash is NoPE (qk_rope_head_dim is 0), so this base is inert;
// the shared attention struct still requires a positive value.
constexpr double kNoPeRopeTheta = 10000.0;

// Required integral value that may be zero (dims such as
// qk_rope_head_dim are 0 for NoPE definitions).
std::expected<std::uint64_t, StatusCode> NonNegativeU64(const Json& object,
                                                        std::string_view key) {
  const Json* value = object.Find(key);
  if (value == nullptr || value->type() != Json::Type::Number) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const double number = value->AsNumber();
  if (number < 0.0 || number != std::floor(number) ||
      number > kMaxExactInteger) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return static_cast<std::uint64_t>(number);
}

// Required finite number; sign is not restricted (gate bounds are
// negative).
std::expected<double, StatusCode> Number(const Json& object,
                                         std::string_view key) {
  const Json* value = object.Find(key);
  if (value == nullptr || value->type() != Json::Type::Number ||
      !std::isfinite(value->AsNumber())) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return value->AsNumber();
}

// Optional integral value with a default.
std::expected<std::uint64_t, StatusCode> OptionalU64(const Json& object,
                                                     std::string_view key,
                                                     std::uint64_t fallback) {
  if (object.Find(key) == nullptr) {
    return fallback;
  }
  return JsonU64(object, key);
}

// The `layer_types` array as per-layer full-attention flags:
// "deepseek_sparse_attention" selects the MLA path, "linear_attention"
// the KDA path. MalformedFile on any other entry or a wrong length.
std::expected<std::vector<std::uint8_t>, StatusCode> LayerFlags(
    const Json& text, std::size_t layers) {
  const Json* types = text.Find("layer_types");
  if (types == nullptr || !types->isArray() ||
      types->AsArray().size() != layers) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::vector<std::uint8_t> flags;
  flags.reserve(layers);
  for (const Json& entry : types->AsArray()) {
    if (entry.type() != Json::Type::String) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    const std::string& name = entry.AsString();
    if (name == "linear_attention") {
      flags.push_back(0);
    } else if (name == "deepseek_sparse_attention") {
      flags.push_back(1);
    } else {
      return std::unexpected(StatusCode::MalformedFile);
    }
  }
  return flags;
}

// Check an optional index list (kda_layers / full_attn_layers) against
// the layer flags: every listed index carries `want`.
std::expected<void, StatusCode> ValidateLayerList(
    const Json& object, std::string_view key,
    std::span<const std::uint8_t> flags, std::uint8_t want) {
  const Json* list = object.Find(key);
  if (list == nullptr) {
    return {};
  }
  if (!list->isArray()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  for (const Json& entry : list->AsArray()) {
    if (entry.type() != Json::Type::Number) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    const double number = entry.AsNumber();
    if (number < 0.0 || number != std::floor(number) ||
        number >= static_cast<double>(flags.size())) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    if (flags[static_cast<std::size_t>(number)] != want) {
      return std::unexpected(StatusCode::MalformedFile);
    }
  }
  return {};
}

// A string array must exist, have `size` entries and hold only `value`.
std::expected<void, StatusCode> ValidateStringArray(const Json& text,
                                                    std::string_view key,
                                                    std::size_t size,
                                                    std::string_view value) {
  const Json* array = text.Find(key);
  if (array == nullptr || !array->isArray() ||
      array->AsArray().size() != size) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  for (const Json& entry : array->AsArray()) {
    if (entry.type() != Json::Type::String || entry.AsString() != value) {
      return std::unexpected(StatusCode::MalformedFile);
    }
  }
  return {};
}

// The `mlp_layer_types` schedule: `dense` for the leading dense layers,
// `sparse` for the mixture layers.
std::expected<void, StatusCode> ValidateMlpTypes(
    const Json& text, std::size_t layers, std::size_t leading_dense) {
  const Json* types = text.Find("mlp_layer_types");
  if (types == nullptr) {
    return {};
  }
  if (!types->isArray() || types->AsArray().size() != layers) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  for (std::size_t i = 0; i < layers; ++i) {
    const Json& entry = types->AsArray()[i];
    if (entry.type() != Json::Type::String) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    const std::string& want = i < leading_dense ? std::string("dense")
                                                : std::string("sparse");
    if (entry.AsString() != want) {
      return std::unexpected(StatusCode::MalformedFile);
    }
  }
  return {};
}

}  // namespace

std::expected<Glm5NextGeometry, StatusCode> GeometryOf(
    const TransformerConfig& config) {
  if (config.layers == 0 || config.hidden_dim == 0 || config.vocab_size == 0 ||
      !(config.norm_eps > 0.0)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (!config.mla.IsPresent() || !config.kda.IsPresent() ||
      !config.hyper_connection.IsManifold() || config.indexer.heads == 0 ||
      config.indexer.kpool == 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (config.layer_full_attention.size() != config.layers) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (config.attention.heads == 0 || config.attention.kv_heads != 1 ||
      config.attention.head_dim !=
          config.mla.kv_lora_rank + config.mla.qk_rope_head_dim ||
      config.attention.rope_dim != config.mla.qk_rope_head_dim ||
      config.mla.q_lora_rank == 0 || config.mla.kv_lora_rank == 0 ||
      config.mla.qk_nope_head_dim == 0 || config.mla.v_head_dim == 0 ||
      (config.mla.qk_rope_head_dim % 2) != 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (config.kda.heads == 0 || config.kda.head_dim == 0 ||
      config.kda.conv_kernel < 2 || !(config.kda.gate_lower_bound < 0.0)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (config.indexer.key_length == 0 || config.indexer.top_k == 0 ||
      config.indexer.top_k % config.indexer.kpool != 0 ||
      !(config.indexer.norm_eps > 0.0)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const HyperConnectionParams& hc = config.hyper_connection;
  if (hc.sinkhorn_iterations == 0 || !(hc.epsilon > 0.0) ||
      !(hc.epsilon < 1.0)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (config.num_experts == 0 || config.experts_per_tok == 0 ||
      config.experts_per_tok > config.num_experts ||
      config.moe_intermediate == 0 || config.shared_expert_intermediate == 0 ||
      config.leading_dense_layers > config.layers ||
      !(config.moe_weights_scale >= 0.0)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (config.leading_dense_layers > 0 && config.ffn_dim == 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (config.swiglu_clamp.size() != config.layers) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (!config.swiglu_clamp_shared.empty() &&
      config.swiglu_clamp_shared.size() != config.layers) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  Glm5NextGeometry geometry;
  geometry.trunk_layers = config.layers;
  geometry.hidden = config.hidden_dim;
  geometry.norm_eps = config.norm_eps;
  geometry.heads = config.attention.heads;
  geometry.q_lora_rank = config.mla.q_lora_rank;
  geometry.kv_lora_rank = config.mla.kv_lora_rank;
  geometry.qk_nope_head_dim = config.mla.qk_nope_head_dim;
  geometry.qk_rope_head_dim = config.mla.qk_rope_head_dim;
  geometry.v_head_dim = config.mla.v_head_dim;
  geometry.kda_heads = config.kda.heads;
  geometry.kda_head_dim = config.kda.head_dim;
  geometry.kda_inner = config.kda.heads * config.kda.head_dim;
  geometry.kda_conv_kernel = config.kda.conv_kernel;
  geometry.kda_gate_lower_bound = config.kda.gate_lower_bound;
  geometry.indexer_heads = config.indexer.heads;
  geometry.indexer_key_length = config.indexer.key_length;
  geometry.indexer_top_k = config.indexer.top_k;
  geometry.indexer_kpool = config.indexer.kpool;
  geometry.indexer_select_tail = config.indexer.select_tail;
  geometry.indexer_norm_eps = config.indexer.norm_eps;
  geometry.hc_streams = hc.count;
  geometry.hc_mix_dim = (2 + hc.count) * hc.count;
  geometry.hc_sinkhorn_iterations = hc.sinkhorn_iterations;
  geometry.hc_eps = hc.epsilon;
  geometry.num_experts = config.num_experts;
  geometry.experts_per_tok = config.experts_per_tok;
  geometry.moe_intermediate = config.moe_intermediate;
  geometry.shared_intermediate = config.shared_expert_intermediate;
  geometry.leading_dense_layers = config.leading_dense_layers;
  geometry.moe_gating = config.moe_gating;
  geometry.moe_weights_norm = config.moe_weights_norm;
  geometry.moe_weights_scale = config.moe_weights_scale;
  geometry.ffn_dim = config.ffn_dim;
  return geometry;
}

std::expected<TransformerConfig, StatusCode>
Glm5NextArchitecture::ParseConfigJson(std::string_view json) const {
  auto document = Json::Parse(json);
  if (!document || !document->isObject()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const Json& text = JsonTextConfig(*document);

  auto hidden = JsonU64(text, "hidden_size");
  auto layers = JsonU64(text, "num_hidden_layers");
  auto vocab = JsonU64(text, "vocab_size");
  auto heads = JsonU64(text, "num_attention_heads");
  auto intermediate = JsonU64(text, "intermediate_size");
  auto eps = JsonReal(text, "rms_norm_eps");
  auto q_lora = JsonU64(text, "q_lora_rank");
  auto kv_lora = JsonU64(text, "kv_lora_rank");
  auto qk_nope = JsonU64(text, "qk_nope_head_dim");
  auto qk_rope = NonNegativeU64(text, "qk_rope_head_dim");
  auto v_head = JsonU64(text, "v_head_dim");
  auto nope = JsonBool(text, "mla_use_nope");
  if (!hidden || !layers || !vocab || !heads || !intermediate || !eps ||
      !q_lora || !kv_lora || !qk_nope || !qk_rope || !v_head || !nope) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  // The module runs the NoPE variant; a rotary MLA definition needs the
  // rotary kernels and a rope_parameters contract it does not have yet.
  if (!*nope || *qk_rope != 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const std::size_t num_layers = static_cast<std::size_t>(*layers);

  auto flags = LayerFlags(text, num_layers);
  if (!flags) {
    return std::unexpected(flags.error());
  }

  const Json* linear = text.Find("linear_attn_config");
  if (linear == nullptr || !linear->isObject()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto kda_heads = JsonU64(*linear, "num_heads");
  auto kda_head_dim = JsonU64(*linear, "head_dim");
  auto kda_conv = JsonU64(*linear, "short_conv_kernel_size");
  auto kda_bound = Number(*linear, "gate_lower_bound");
  if (!kda_heads || !kda_head_dim || !kda_conv || !kda_bound ||
      !(*kda_bound < 0.0)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (auto ok = ValidateLayerList(*linear, "kda_layers", *flags, 0); !ok) {
    return std::unexpected(ok.error());
  }
  if (auto ok = ValidateLayerList(*linear, "full_attn_layers", *flags, 1);
      !ok) {
    return std::unexpected(ok.error());
  }

  auto index_heads = JsonU64(text, "index_n_heads");
  auto index_key = JsonU64(text, "index_head_dim");
  auto index_topk = JsonU64(text, "index_topk");
  auto index_kpool = JsonU64(text, "index_kpool");
  auto index_tail = JsonBool(text, "index_kpool_always_select_tail");
  auto index_compress = JsonBool(text, "index_kpool_compress");
  if (!index_heads || !index_key || !index_topk || !index_kpool ||
      !index_tail || !index_compress) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  // The indexer math is implemented for the compressed k-pool variant.
  if (!*index_compress) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (auto ok = ValidateStringArray(text, "indexer_types", num_layers, "full");
      !ok) {
    return std::unexpected(ok.error());
  }

  auto num_experts = JsonU64(text, "n_routed_experts");
  auto experts_per_tok = JsonU64(text, "num_experts_per_tok");
  auto moe_inter = JsonU64(text, "moe_intermediate_size");
  auto shared_experts = JsonU64(text, "n_shared_experts");
  auto leading_dense = NonNegativeU64(text, "first_k_dense_replace");
  auto norm_topk = JsonBool(text, "norm_topk_prob");
  auto weights_scale = JsonReal(text, "routed_scaling_factor");
  auto scoring = JsonString(text, "scoring_func");
  if (!num_experts || !experts_per_tok || !moe_inter || !shared_experts ||
      !leading_dense || !norm_topk || !weights_scale || !scoring) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const auto n_group = OptionalU64(text, "n_group", 1);
  const auto topk_group = OptionalU64(text, "topk_group", 1);
  if (!n_group || !topk_group || *n_group != 1 || *topk_group != 1) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (*experts_per_tok > *num_experts || *moe_inter > 1e12 ||
      *shared_experts > 1e6) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (*leading_dense > *layers) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (auto ok = ValidateMlpTypes(text, num_layers,
                                 static_cast<std::size_t>(*leading_dense));
      !ok) {
    return std::unexpected(ok.error());
  }

  auto mhc = JsonBool(text, "mhc");
  auto hc_mult = JsonU64(text, "hc_mult");
  auto hc_iters = JsonU64(text, "hc_sinkhorn_iters");
  auto hc_eps = JsonReal(text, "hc_eps");
  auto swiglu = JsonReal(text, "swiglu_limit");
  if (!mhc || !hc_mult || !hc_iters || !hc_eps || !swiglu || !*mhc ||
      !(*hc_eps < 1.0)) {
    return std::unexpected(StatusCode::MalformedFile);
  }

  TransformerConfig config;
  config.attention.heads = static_cast<std::size_t>(*heads);
  config.attention.kv_heads = 1;
  config.attention.head_dim =
      static_cast<std::size_t>(*kv_lora + *qk_rope);
  config.attention.rope_dim = static_cast<std::size_t>(*qk_rope);
  config.attention.rope_theta = kNoPeRopeTheta;
  config.mla.q_lora_rank = static_cast<std::size_t>(*q_lora);
  config.mla.kv_lora_rank = static_cast<std::size_t>(*kv_lora);
  config.mla.qk_nope_head_dim = static_cast<std::size_t>(*qk_nope);
  config.mla.qk_rope_head_dim = static_cast<std::size_t>(*qk_rope);
  config.mla.v_head_dim = static_cast<std::size_t>(*v_head);
  config.layers = num_layers;
  config.hidden_dim = static_cast<std::size_t>(*hidden);
  config.ffn_dim = static_cast<std::size_t>(*intermediate);
  config.vocab_size = static_cast<std::size_t>(*vocab);
  config.norm_eps = *eps;
  config.hybrid = true;
  config.kda.heads = static_cast<std::size_t>(*kda_heads);
  config.kda.head_dim = static_cast<std::size_t>(*kda_head_dim);
  config.kda.conv_kernel = static_cast<std::size_t>(*kda_conv);
  config.kda.gate_lower_bound = *kda_bound;
  config.indexer.heads = static_cast<std::size_t>(*index_heads);
  config.indexer.key_length = static_cast<std::size_t>(*index_key);
  config.indexer.top_k = static_cast<std::size_t>(*index_topk);
  config.indexer.kpool = static_cast<std::size_t>(*index_kpool);
  config.indexer.select_tail = *index_tail;
  config.indexer.norm_eps = kIndexerLayerNormEps;
  config.hyper_connection.count = static_cast<std::size_t>(*hc_mult);
  config.hyper_connection.low_rank = 0;
  config.hyper_connection.sinkhorn_iterations =
      static_cast<std::size_t>(*hc_iters);
  config.hyper_connection.epsilon = *hc_eps;
  config.num_experts = static_cast<std::size_t>(*num_experts);
  config.experts_per_tok = static_cast<std::size_t>(*experts_per_tok);
  config.moe_intermediate = static_cast<std::size_t>(*moe_inter);
  config.shared_expert_intermediate =
      static_cast<std::size_t>(*shared_experts * *moe_inter);
  config.moe_weights_norm = *norm_topk;
  config.moe_weights_scale = *weights_scale;
  if (*scoring == "sigmoid") {
    config.moe_gating = MoeGating::kSigmoid;
  } else if (*scoring == "softmax") {
    config.moe_gating = MoeGating::kSoftmax;
  } else {
    return std::unexpected(StatusCode::MalformedFile);
  }
  config.leading_dense_layers = static_cast<std::size_t>(*leading_dense);
  config.layer_full_attention = std::move(*flags);
  config.swiglu_clamp.assign(num_layers, static_cast<float>(*swiglu));
  config.swiglu_clamp_shared.assign(num_layers, static_cast<float>(*swiglu));

  auto geometry = GeometryOf(config);
  if (!geometry) {
    return std::unexpected(geometry.error());
  }
  return config;
}

}  // namespace tessera::models::glm5_next
