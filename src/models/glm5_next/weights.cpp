#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "models/glm5_next/architecture.hpp"

// Maps HuggingFace safetensors names (a GLM-5-Next checkpoint, for
// example GLM-5.3-Flash) to the internal "blk.N.*" names. The trunk
// layers map "model.language_model.layers.N.<suffix>" to "blk.N.*";
// unknown names return nullopt. Three checkpoint tensors have no 1:1
// mapping and are ignored until the loader can split them:
//   kv_b_proj.weight      -> attn_k_b + attn_v_b
//   conv1d.weight         -> ssm_conv1d_q/k/v (when stored fused)
//   mlp.experts.E.*       -> ffn_{gate,up,down}_exps (packing)
// Name() and ConvertWeight serve the GGUF path (identity mapping; the
// loader applies exp_negate to ssm_a) and the same HF path.

namespace tessera::models::glm5_next {

namespace {

constexpr std::string_view kLmPrefix = "model.language_model.";
constexpr std::string_view kLayers = "layers.";

std::optional<std::string_view> MapLayerSuffix(std::string_view suffix) {
  static constexpr std::pair<std::string_view, std::string_view> kMap[] = {
      {"input_layernorm.weight", "attn_norm.weight"},
      {"post_attention_layernorm.weight", "ffn_norm.weight"},
      {"hc_attn_fn", "hc_attn_fn.weight"},
      {"hc_attn_base", "hc_attn_base.weight"},
      {"hc_attn_scale", "hc_attn_scale.weight"},
      {"hc_ffn_fn", "hc_ffn_fn.weight"},
      {"hc_ffn_base", "hc_ffn_base.weight"},
      {"hc_ffn_scale", "hc_ffn_scale.weight"},
      // KDA layers (q/k/v are the gated delta projections; the q/k/v
      // convs are depthwise over one projection each).
      {"self_attn.q_proj.weight", "attn_q.weight"},
      {"self_attn.k_proj.weight", "attn_k.weight"},
      {"self_attn.v_proj.weight", "attn_v.weight"},
      {"self_attn.o_proj.weight", "attn_output.weight"},
      {"self_attn.q_conv1d.weight", "ssm_conv1d_q.weight"},
      {"self_attn.k_conv1d.weight", "ssm_conv1d_k.weight"},
      {"self_attn.v_conv1d.weight", "ssm_conv1d_v.weight"},
      {"self_attn.f_a_proj.weight", "ssm_f_a.weight"},
      {"self_attn.f_b_proj.weight", "ssm_f_b.weight"},
      {"self_attn.dt_bias", "ssm_dt.bias"},
      {"self_attn.A_log", "ssm_a"},
      {"self_attn.b_proj.weight", "ssm_beta.weight"},
      {"self_attn.g_a_proj.weight", "ssm_g_a.weight"},
      {"self_attn.g_b_proj.weight", "ssm_g_b.weight"},
      {"self_attn.o_norm.weight", "ssm_norm.weight"},
      // MLA layers.
      {"self_attn.q_a_proj.weight", "attn_q_a.weight"},
      {"self_attn.q_a_layernorm.weight", "attn_q_a_norm.weight"},
      {"self_attn.q_b_proj.weight", "attn_q_b.weight"},
      {"self_attn.kv_a_proj_with_mqa.weight", "attn_kv_a_mqa.weight"},
      {"self_attn.kv_a_layernorm.weight", "attn_kv_a_norm.weight"},
      // Sparse-attention indexer.
      {"self_attn.indexer.wk.weight", "indexer.attn_k.weight"},
      {"self_attn.indexer.wq_b.weight", "indexer.attn_q_b.weight"},
      {"self_attn.indexer.k_norm.weight", "indexer.k_norm.weight"},
      {"self_attn.indexer.k_norm.bias", "indexer.k_norm.bias"},
      {"self_attn.indexer.weights_proj.weight", "indexer.proj.weight"},
      {"self_attn.indexer.index_kpool_compress_gate",
       "indexer_compressor_gate.weight"},
      {"self_attn.indexer.index_kpool_compress_ape",
       "indexer_compressor_ape.weight"},
      // Dense and mixture MLP.
      {"mlp.gate_proj.weight", "ffn_gate.weight"},
      {"mlp.up_proj.weight", "ffn_up.weight"},
      {"mlp.down_proj.weight", "ffn_down.weight"},
      {"mlp.gate.weight", "ffn_gate_inp.weight"},
      {"mlp.gate.e_score_correction_bias", "exp_probs_b.bias"},
      {"mlp.shared_experts.gate_proj.weight", "ffn_gate_shexp.weight"},
      {"mlp.shared_experts.up_proj.weight", "ffn_up_shexp.weight"},
      {"mlp.shared_experts.down_proj.weight", "ffn_down_shexp.weight"},
  };
  for (const auto& [from, to] : kMap) {
    if (suffix == from) {
      return to;
    }
  }
  return std::nullopt;
}

std::string LayerName(std::string_view index, std::string_view suffix) {
  return "blk." + std::string(index) + "." + std::string(suffix);
}

}  // namespace

std::string_view Glm5NextArchitecture::Name() const { return "glm5-next"; }

std::optional<std::string> Glm5NextArchitecture::MapWeightName(
    std::string_view name, const TransformerConfig& config,
    const WeightLoadOptions& options) const {
  (void)config;
  (void)options;
  if (name == "lm_head.weight") {
    return std::string("output.weight");
  }
  if (name.rfind("model.visual.", 0) == 0) {
    return std::nullopt;  // vision tower; the text model does not use it
  }
  if (name.rfind(kLmPrefix, 0) != 0) {
    return std::nullopt;  // unknown (for example the MTP head)
  }
  const std::string_view rest = name.substr(kLmPrefix.size());
  if (rest == "embed_tokens.weight") {
    return std::string("token_embd.weight");
  }
  if (rest == "norm.weight") {
    return std::string("output_norm.weight");
  }
  if (rest.rfind(kLayers, 0) != 0) {
    return std::nullopt;
  }
  const std::string_view tail = rest.substr(kLayers.size());
  const std::size_t dot = tail.find('.');
  if (dot == std::string_view::npos || dot == 0) {
    return std::nullopt;
  }
  const std::string_view index = tail.substr(0, dot);
  for (const char c : index) {
    if (c < '0' || c > '9') {
      return std::nullopt;
    }
  }
  auto mapped = MapLayerSuffix(tail.substr(dot + 1));
  if (!mapped) {
    return std::nullopt;
  }
  return LayerName(index, *mapped);
}

std::optional<WeightConversion> Glm5NextArchitecture::ConvertWeight(
    std::string_view internal_name, const TransformerConfig& config) const {
  (void)config;
  // GGUF only: A_log is stored in log space; the internal form is F32
  // -exp(A_log). The names are already internal here, so match on "blk."
  // plus the bare "ssm_a" tail (no ".weight" suffix).
  if (internal_name.rfind("blk.", 0) != 0) {
    return std::nullopt;
  }
  const std::size_t dot = internal_name.find('.', 4);
  if (dot == std::string_view::npos ||
      internal_name.substr(dot + 1) != "ssm_a") {
    return std::nullopt;
  }
  WeightConversion conversion;
  conversion.exp_negate = true;
  return conversion;
}

std::unique_ptr<Architecture> MakeGlm5NextArchitecture() {
  return std::make_unique<Glm5NextArchitecture>();
}

}  // namespace tessera::models::glm5_next
