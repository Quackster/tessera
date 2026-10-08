#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "models/qwen3_5/architecture.hpp"

// Maps HuggingFace safetensors names (the MXFP4 checkpoint) to the
// internal names the Qwen3.5 module looks up. The trunk layers map
// "model.language_model.layers.N.<suffix>" to "blk.N.<internal>"; the
// next-token-prediction head maps "mtp.*" to "blk.<trunk>.*". Vision
// tensors and unknown names are ignored (nullopt).

namespace tessera::models::qwen3_5 {

namespace {

constexpr std::string_view kLmPrefix = "model.language_model.";
constexpr std::string_view kLayers = "layers.";

std::optional<std::string_view> MapLayerSuffix(std::string_view suffix) {
  static constexpr std::pair<std::string_view, std::string_view> kMap[] = {
      {"input_layernorm.weight", "attn_norm.weight"},
      {"post_attention_layernorm.weight", "post_attention_norm.weight"},
      {"self_attn.q_proj.weight", "attn_q.weight"},
      {"self_attn.k_proj.weight", "attn_k.weight"},
      {"self_attn.v_proj.weight", "attn_v.weight"},
      {"self_attn.o_proj.weight", "attn_output.weight"},
      {"self_attn.q_norm.weight", "attn_q_norm.weight"},
      {"self_attn.k_norm.weight", "attn_k_norm.weight"},
      {"linear_attn.in_proj_qkv.weight", "attn_qkv.weight"},
      {"linear_attn.in_proj_z.weight", "attn_gate.weight"},
      {"linear_attn.in_proj_a.weight", "ssm_alpha.weight"},
      {"linear_attn.in_proj_b.weight", "ssm_beta.weight"},
      {"linear_attn.conv1d.weight", "ssm_conv1d.weight"},
      {"linear_attn.A_log", "ssm_a"},
      {"linear_attn.dt_bias", "ssm_dt.bias"},
      {"linear_attn.norm.weight", "ssm_norm.weight"},
      {"linear_attn.out_proj.weight", "ssm_out.weight"},
      {"mlp.gate_proj.weight", "ffn_gate.weight"},
      {"mlp.up_proj.weight", "ffn_up.weight"},
      {"mlp.down_proj.weight", "ffn_down.weight"},
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

std::optional<std::string> Qwen35Architecture::MapWeightName(
    std::string_view name, const TransformerConfig& config) const {
  if (name == "lm_head.weight") {
    return std::string("output.weight");
  }
  if (name.rfind("model.visual.", 0) == 0) {
    return std::nullopt;  // vision tower; the text model does not use it
  }
  if (name.rfind("mtp.", 0) == 0) {
    const std::string base = std::to_string(config.layers);
    std::string_view rest = name.substr(4);
    if (rest == "fc.weight") {
      return LayerName(base, "nextn.eh_proj.weight");
    }
    if (rest == "pre_fc_norm_embedding.weight") {
      return LayerName(base, "nextn.enorm.weight");
    }
    if (rest == "pre_fc_norm_hidden.weight") {
      return LayerName(base, "nextn.hnorm.weight");
    }
    if (rest == "norm.weight") {
      return LayerName(base, "nextn.shared_head_norm.weight");
    }
    if (rest.rfind(kLayers, 0) == 0) {
      const std::string_view tail = rest.substr(kLayers.size());
      const std::size_t dot = tail.find('.');
      if (dot == std::string_view::npos || tail.substr(0, dot) != "0") {
        return std::nullopt;
      }
      auto mapped = MapLayerSuffix(tail.substr(dot + 1));
      if (!mapped) {
        return std::nullopt;
      }
      return LayerName(base, *mapped);
    }
    return std::nullopt;
  }
  if (name.rfind(kLmPrefix, 0) != 0) {
    return std::nullopt;
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

}  // namespace tessera::models::qwen3_5
