#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>

#include "core/decode.hpp"
#include "tessera/model.hpp"
#include "tessera/types.hpp"

// Internal hybrid helpers shared by the hybrid decoder and the MTP head
// (both run full-attention Qwen3.5 blocks on the device).

namespace tessera::core {

// Derived recurrent-layer geometry from the SSM config.
struct LinearGeometry {
  std::size_t key_dim = 0;
  std::size_t value_dim = 0;
  std::size_t num_v_heads = 0;
  std::size_t head_v_dim = 0;
  std::size_t num_k_heads = 0;
  std::size_t head_k_dim = 0;
  std::size_t conv_dim = 0;
  std::size_t width = 0;
  std::size_t factor = 0;
};

inline std::expected<LinearGeometry, StatusCode> DeriveGeometry(
    const TransformerConfig& cfg) {
  LinearGeometry g;
  g.head_k_dim = cfg.ssm.state_size;
  g.num_k_heads = cfg.ssm.group_count;
  g.value_dim = cfg.ssm.inner_size;
  g.num_v_heads = cfg.ssm.time_step_rank;
  g.width = cfg.ssm.conv_kernel;
  if (g.head_k_dim == 0 || g.num_k_heads == 0 || g.value_dim == 0 ||
      g.num_v_heads == 0 || g.width == 0 ||
      g.value_dim % g.num_v_heads != 0 ||
      g.num_v_heads % g.num_k_heads != 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  g.key_dim = g.head_k_dim * g.num_k_heads;
  g.head_v_dim = g.value_dim / g.num_v_heads;
  g.factor = g.num_v_heads / g.num_k_heads;
  g.conv_dim = g.key_dim * 2 + g.value_dim;
  return g;
}

// Loads the hybrid kernels and allocates the per-step scratch on first
// use; idempotent. Requires the derived linear geometry.
[[nodiscard]] std::expected<void, StatusCode> EnsureHybridReady(
    Backend& backend, const TransformerConfig& cfg, DecodeCache& cache,
    const LinearGeometry& g);

// Runs the block's gated MLP (post-attention norm, gate/up/down, residual
// add) on h.x in place. The attention output must already be added to h.x.
[[nodiscard]] std::expected<void, StatusCode> RunFfn(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    HybridDecodeCache& h, std::size_t layer);

// Runs one full-attention block (attn_norm, gated attention, gated MLP) on
// h.x in place, using `kv` for the key/value cache at position `pos`.
[[nodiscard]] std::expected<void, StatusCode> RunFullBlock(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    HybridDecodeCache& h, std::size_t layer, std::uint64_t pos,
    HybridDecodeCache::FullKv& kv);

// Runs several tokens through the hybrid trunk in one call (batched
// GEMMs, attention with causal masking over the whole block), advancing
// the cache by tokens.size(). `logits_out`, when non-null, receives one
// vocab row per token (tokens.size() x vocab, row-major) and
// `hidden_out` the final hidden state of the last token. The cache's
// linear-attention state snapshots for each position are written to
// cache.batch->state_hist / conv_hist_hist so the caller can roll back.
[[nodiscard]] std::expected<void, StatusCode> HybridForwardBatch(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::span<const std::uint32_t> tokens,
    std::vector<float>* logits_out = nullptr,
    std::vector<float>* hidden_out = nullptr, bool all_logits = true);

}  // namespace tessera::core
