#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"

namespace tessera::core {

// One CLIP vision transformer block: LayerNorm, fused QKV projection,
// non-causal multi-head attention, output projection, residual; then
// LayerNorm, GELU MLP, residual. Weights are fp32: ln1/ln2 (embed),
// qkv (3*embed x embed), out (embed x embed), up (ffn x embed) and
// down (embed x ffn), each with a bias. tokens is the number of patches;
// embed = heads * head_dim.
struct VisionBlockWeights {
  std::span<const float> ln1_weight;
  std::span<const float> ln1_bias;
  std::span<const float> qkv_weight;
  std::span<const float> qkv_bias;
  std::span<const float> out_weight;
  std::span<const float> out_bias;
  std::span<const float> ln2_weight;
  std::span<const float> ln2_bias;
  std::span<const float> up_weight;
  std::span<const float> up_bias;
  std::span<const float> down_weight;
  std::span<const float> down_bias;
};

[[nodiscard]] std::expected<void, StatusCode> VisionBlockRef(
    std::span<const float> x, const VisionBlockWeights& w,
    std::span<float> out, std::size_t tokens, std::size_t embed,
    std::size_t heads, std::size_t head_dim, std::size_t ffn, float eps);

// The device weight buffers of one block.
struct VisionBlockBuffers {
  const Buffer* ln1_weight = nullptr;
  const Buffer* ln1_bias = nullptr;
  const Buffer* qkv_weight = nullptr;
  const Buffer* qkv_bias = nullptr;
  const Buffer* out_weight = nullptr;
  const Buffer* out_bias = nullptr;
  const Buffer* ln2_weight = nullptr;
  const Buffer* ln2_bias = nullptr;
  const Buffer* up_weight = nullptr;
  const Buffer* up_bias = nullptr;
  const Buffer* down_weight = nullptr;
  const Buffer* down_bias = nullptr;
};

[[nodiscard]] std::expected<void, StatusCode> VisionBlockDevice(
    Backend& backend, const Kernel& layernorm, const Kernel& gemm,
    const Kernel& attention, const Kernel& gelu, const Kernel& bias_add,
    const Kernel& add, const Buffer& x, const VisionBlockBuffers& w,
    Buffer& out, std::size_t tokens, std::size_t embed, std::size_t heads,
    std::size_t head_dim, std::size_t ffn, float eps);

}  // namespace tessera::core
