#pragma once

#include <cstddef>
#include <expected>

#include "tessera/types.hpp"

// Geometry of a GLM-5-Next (GLM-5.3-Flash) model: the shapes the module
// derives from the generic TransformerConfig, validated once at parse
// time so a definition the module cannot run rejects with MalformedFile
// instead of failing mid-decode. A definition mixes KDA linear-attention
// layers with MLA full-attention layers that select their keys through a
// k-pool indexer, wraps attention and FFN in manifold-constrained
// hyper-connections (mHC), and feeds the block FFN through a sparse MoE
// with leading dense layers.

namespace tessera::models::glm5_next {

struct Glm5NextGeometry {
  std::size_t trunk_layers = 0;
  std::size_t hidden = 0;
  double norm_eps = 0.0;
  // MLA full-attention layers.
  std::size_t heads = 0;
  std::size_t q_lora_rank = 0;
  std::size_t kv_lora_rank = 0;
  std::size_t qk_nope_head_dim = 0;
  std::size_t qk_rope_head_dim = 0;
  std::size_t v_head_dim = 0;
  // KDA recurrent layers; inner is the q/k/v projection width.
  std::size_t kda_heads = 0;
  std::size_t kda_head_dim = 0;
  std::size_t kda_inner = 0;
  std::size_t kda_conv_kernel = 0;
  double kda_gate_lower_bound = -5.0;
  // Sparse-attention indexer (k-pool).
  std::size_t indexer_heads = 0;
  std::size_t indexer_key_length = 0;
  std::size_t indexer_top_k = 0;
  std::size_t indexer_kpool = 1;
  bool indexer_select_tail = false;
  double indexer_norm_eps = 0.0;
  // Manifold-constrained hyper-connections (mHC).
  std::size_t hc_streams = 0;
  std::size_t hc_mix_dim = 0;  // (2 + streams) * streams
  std::size_t hc_sinkhorn_iterations = 0;
  double hc_eps = 0.0;
  // Mixture of experts; the first leading_dense_layers trunk layers run
  // a dense FFN of width ffn_dim instead.
  std::size_t num_experts = 0;
  std::size_t experts_per_tok = 0;
  std::size_t moe_intermediate = 0;
  std::size_t shared_intermediate = 0;
  std::size_t leading_dense_layers = 0;
  MoeGating moe_gating = MoeGating::kSoftmax;
  bool moe_weights_norm = false;
  double moe_weights_scale = 0.0;
  std::size_t ffn_dim = 0;
};

// Derive and validate the geometry of `config`; MalformedFile when a
// shape the module needs is absent or the definition mixes incompatible
// attention kinds. Both parse paths (GGUF metadata, HuggingFace JSON)
// funnel through here, so every GLM definition is validated once.
[[nodiscard]] std::expected<Glm5NextGeometry, StatusCode> GeometryOf(
    const TransformerConfig& config);

}  // namespace tessera::models::glm5_next
