#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "core/decode.hpp"
#include "models/qwen3_5/state.hpp"
#include "tessera/model.hpp"
#include "tessera/types.hpp"

// Internal hybrid helpers shared by the hybrid decoder and the MTP head
// (both run full-attention Qwen3.5 blocks on the device).

namespace tessera::models::qwen3_5 {

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
    Backend& backend, const TransformerConfig& cfg, core::DecodeCache& cache,
    const LinearGeometry& g);

// One projection C = A x W^T for `m` rows of `a` (m x k fp32). For an
// MXFP4 weight (F4E2M1) the activation is first quantize-dequantized per
// token to FP8 E4M3 in place, the W4A8 contract the served Qwen3.8 target
// runs; other dtypes pass through unchanged. `a` is read and, for MXFP4,
// rewritten in place. `w` is n x k, `out` is m x n.
[[nodiscard]] std::expected<void, StatusCode> ProjectBatch(
    Backend& backend, Qwen35State& h, DType dtype, Buffer& a,
    const Buffer& w, Buffer& out, std::size_t m, std::size_t n,
    std::size_t k);

// The per-token FP8 QDQ used by ProjectBatch for an MXFP4 weight; a no-op
// for other dtypes. Exposed for the MTP head. `data` is rows x cols fp32.
[[nodiscard]] std::expected<void, StatusCode> QuantizeMxFp4Input(
    Backend& backend, Qwen35State& h, DType dtype, Buffer& data,
    std::size_t rows, std::size_t cols);

// Gather `tokens` embedding rows of the model's token_embd.weight into
// `out` (rows x hidden fp32) on the device. Uses a device gather kernel
// for the formats it covers (f32, bf16, Q4_K) and a host dequantize plus
// upload otherwise. `out` must hold rows * hidden floats.
[[nodiscard]] std::expected<void, StatusCode> GatherEmbeddingRows(
    Backend& backend, const Model& model, Qwen35State& h,
    std::span<const std::uint32_t> tokens, std::size_t hidden, Buffer& out);

// Runs the block's gated MLP (post-attention norm, gate/up/down, residual
// add) on h.x in place. The attention output must already be added to h.x.
[[nodiscard]] std::expected<void, StatusCode> RunFfn(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen35State& h, std::size_t layer);

// Runs one full-attention block (attn_norm, gated attention, gated MLP) on
// h.x in place, using `kv` for the key/value cache at position `pos`.
[[nodiscard]] std::expected<void, StatusCode> RunFullBlock(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen35State& h, std::size_t layer, std::uint64_t pos,
    Qwen35State::FullKv& kv);

}  // namespace tessera::models::qwen3_5
