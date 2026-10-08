#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

#include "models/qwen3_5/architecture.hpp"

// The MXFP4 checkpoint stores the gated-delta value heads interleaved
// (HuggingFace order: value head p = f + factor*kh), while the internal
// (GGUF) layout groups them by the repeat index f (internal head r maps
// to checkpoint head factor*(r % key_heads) + r / key_heads). A_log is
// stored in log space and is turned into the internal F32 -exp form.
// Only the linear-attention tensors are affected; the full-attention and
// MLP tensors are already in the internal layout.

namespace tessera::models::qwen3_5 {

namespace {

// "blk.N.<tail>" -> "<tail>"; empty when the prefix is absent.
std::string_view LayerTail(std::string_view name) {
  if (name.rfind("blk.", 0) != 0) {
    return {};
  }
  const std::size_t dot = name.find('.', 4);
  if (dot == std::string_view::npos) {
    return {};
  }
  return name.substr(dot + 1);
}

}  // namespace

std::optional<WeightConversion> Qwen35Architecture::ConvertWeight(
    std::string_view internal_name, const TransformerConfig& config) const {
  const std::size_t heads = config.ssm.time_step_rank;
  const std::size_t keys = config.ssm.group_count;
  const std::size_t state = config.ssm.state_size;
  const std::size_t value_dim = config.ssm.inner_size;
  if (heads == 0 || keys == 0 || state == 0 || heads % keys != 0 ||
      value_dim == 0 || value_dim % heads != 0) {
    return std::nullopt;
  }
  const std::size_t block = value_dim / heads;  // value-head width
  const std::size_t factor = heads / keys;
  const std::size_t key_dim = state * keys;
  const auto head_src = [&](std::size_t head) {
    return factor * (head % keys) + head / keys;
  };
  // Fill src[offset .. offset+count) with the value-head permutation; the
  // caller sizes src to the full dimension first.
  const auto permute = [&](WeightConversion& conversion, std::size_t offset,
                           std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t head = i / block;
      const std::size_t within = i % block;
      conversion.src[offset + i] = offset + head_src(head) * block + within;
    }
  };
  const std::string_view tail = LayerTail(internal_name);
  if (tail == "ssm_a") {
    WeightConversion conversion;
    conversion.exp_negate = true;
    conversion.src.resize(heads);
    for (std::size_t i = 0; i < heads; ++i) {
      conversion.src[i] = head_src(i);
    }
    return conversion;
  }
  if (tail == "ssm_dt.bias" || tail == "ssm_alpha.weight" ||
      tail == "ssm_beta.weight") {
    WeightConversion conversion;
    conversion.src.resize(heads);
    for (std::size_t i = 0; i < heads; ++i) {
      conversion.src[i] = head_src(i);
    }
    return conversion;
  }
  if (tail == "attn_gate.weight") {
    WeightConversion conversion;
    conversion.src.resize(value_dim);
    permute(conversion, 0, value_dim);
    return conversion;
  }
  if (tail == "attn_qkv.weight" || tail == "ssm_conv1d.weight") {
    // The q and k blocks (key heads) stay; the value block reorders.
    const std::size_t conv_dim = key_dim * 2 + value_dim;
    WeightConversion conversion;
    conversion.src.resize(conv_dim);
    for (std::size_t i = 0; i < key_dim * 2; ++i) {
      conversion.src[i] = i;
    }
    permute(conversion, key_dim * 2, value_dim);
    return conversion;
  }
  if (tail == "ssm_out.weight") {
    WeightConversion conversion;
    conversion.inner = true;
    conversion.src.resize(value_dim);
    permute(conversion, 0, value_dim);
    return conversion;
  }
  return std::nullopt;
}

}  // namespace tessera::models::qwen3_5
