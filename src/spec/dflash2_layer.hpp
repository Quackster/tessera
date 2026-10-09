#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"
#include "spec/dflash2_context.hpp"

namespace tessera::spec {

// One DFlash2 draft decoder layer. Pre-norm residual structure: the
// layer adds `hidden` to the carried `residual`, runs the attention half,
// adds that to the residual, runs the MLP half, and returns the MLP
// output plus the new residual. On the first layer `residual` is empty,
// so the input is used directly.
struct DraftLayerWeights {
  std::span<const float> input_norm;
  std::span<const float> attn_conv_proj;
  std::span<const float> attn_conv_base;
  std::span<const float> q_w;
  std::span<const float> k_w;
  std::span<const float> v_w;
  std::span<const float> o_w;
  std::span<const float> q_norm_w;
  std::span<const float> k_norm_w;
  std::span<const float> post_norm;
  std::span<const float> mlp_conv_proj;
  std::span<const float> mlp_conv_base;
  std::span<const float> gate_w;
  std::span<const float> up_w;
  std::span<const float> down_w;
  // Shared target-hidden norm for the context K/V (empty when no context).
  std::span<const float> hidden_norm;
};

// Host reference. `hidden` is rows x hidden_dim; `residual` is the same
// when present (empty on the first layer). `out` and `residual_out` are
// rows x hidden_dim. dims as for DraftAttentionRef (ffn for the MLP,
// window/pos_base/theta for attention, eps for the norms).
[[nodiscard]] std::expected<void, StatusCode> DraftLayerRef(
    std::span<const float> hidden, std::span<const float> residual,
    const DraftLayerWeights& w, std::span<float> out,
    std::span<float> residual_out, std::size_t rows, std::size_t hidden_dim,
    std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
    std::size_t ffn, std::size_t taps, std::size_t group_size,
    std::size_t block_size, std::size_t window, std::uint64_t pos_base,
    double theta, float eps, std::span<const float> context_hidden = {},
    bool causal = true);

// The device weight buffers (fp32) of one layer.
struct DraftLayerBuffers {
  const Buffer* input_norm = nullptr;
  const Buffer* attn_conv_proj = nullptr;
  const Buffer* attn_conv_base = nullptr;
  const Buffer* q_w = nullptr;
  const Buffer* k_w = nullptr;
  const Buffer* v_w = nullptr;
  const Buffer* o_w = nullptr;
  const Buffer* q_norm_w = nullptr;
  const Buffer* k_norm_w = nullptr;
  const Buffer* post_norm = nullptr;
  const Buffer* mlp_conv_proj = nullptr;
  const Buffer* mlp_conv_base = nullptr;
  const Buffer* gate_w = nullptr;
  const Buffer* up_w = nullptr;
  const Buffer* down_w = nullptr;
  const Buffer* hidden_norm = nullptr;
};

// Device version. `residual` is null on the first layer. Scratch buffers
// are allocated internally for this call.
[[nodiscard]] std::expected<void, StatusCode> DraftLayerDevice(
    Backend& backend, const Kernel& rmsnorm, const Kernel& gemm,
    const Kernel& conv, const Kernel& rope, const Kernel& attention,
    const Kernel& silu, const Kernel& add, const Buffer& hidden,
    const Buffer* residual,
    const DraftLayerBuffers& w, Buffer& out, Buffer& residual_out,
    std::size_t rows, std::size_t hidden_dim, std::size_t heads,
    std::size_t kv_heads, std::size_t head_dim, std::size_t ffn,
    std::size_t taps, std::size_t group_size, std::size_t block_size,
    std::size_t window, std::uint64_t pos_base, double theta, float eps,
    const Buffer* context_hidden = nullptr, std::size_t ctx = 0,
    bool causal = true, DraftContextKvView context_kv = {});

}  // namespace tessera::spec
