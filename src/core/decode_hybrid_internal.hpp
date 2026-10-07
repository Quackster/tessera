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

}  // namespace tessera::core
