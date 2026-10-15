#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "tessera/model.hpp"

namespace tessera::core {

// Expert-cache slot bounds and the device-memory reserve the cache leaves
// for activations, the KV cache and the staging window.
inline constexpr std::size_t kMinExpertCacheEntries = 32;
inline constexpr std::size_t kMaxExpertCacheEntries = 512;
inline constexpr std::uint64_t kExpertCacheReserveBytes = 1ull << 30;

// Expert-cache slots per (layer, kind) pair that fit `free_bytes` of
// device memory while leaving kExpertCacheReserveBytes for activations,
// the KV cache and the staging window. The cache spends one slot-set per
// pair, so the count is the budget divided by `total_per_bytes`, the sum
// of the per-expert slice bytes over every paged pair, clamped to
// [kMinExpertCacheEntries, kMaxExpertCacheEntries]. A zero
// `total_per_bytes` or a budget below the reserve yields the minimum.
//
// Usage:
//   const std::size_t slots = ExpertCacheSlots(info.free_bytes, total_per);
[[nodiscard]] constexpr std::size_t ExpertCacheSlots(
    std::uint64_t free_bytes, std::size_t total_per_bytes) {
  if (total_per_bytes == 0 || free_bytes <= kExpertCacheReserveBytes) {
    return kMinExpertCacheEntries;
  }
  const std::uint64_t budget = free_bytes - kExpertCacheReserveBytes;
  const std::size_t fit = static_cast<std::size_t>(budget / total_per_bytes);
  if (fit < kMinExpertCacheEntries) {
    return kMinExpertCacheEntries;
  }
  return fit > kMaxExpertCacheEntries ? kMaxExpertCacheEntries : fit;
}

// Split an internal `blk.<layer>.<kind>` name into its layer index and
// the kind suffix (the name after the layer prefix). Returns false when
// the name is not a block tensor; `layer` and `kind` stay untouched then.
// Either out-parameter may be null when only one half is wanted.
//
// Usage:
//   std::size_t layer = 0;
//   std::string_view kind;
//   if (SplitBlockName(name, &layer, &kind)) page kind of layer;
[[nodiscard]] bool SplitBlockName(std::string_view name, std::size_t* layer,
                                  std::string_view* kind);

// True when a tensor kind (the name after "blk.<layer>.") is a packed
// routed-expert weight of the llama/Qwen MoE convention
// ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"). Dense shared experts
// ("ffn_*_shexp") are not routed experts: every token uses them whole.
//
// Usage:
//   if (IsRoutedExpertKind(kind)) page slices, not the whole tensor;
[[nodiscard]] inline bool IsRoutedExpertKind(std::string_view kind) {
  constexpr std::string_view kSuffix = "_exps.weight";
  return kind.size() >= kSuffix.size() &&
         kind.substr(kind.size() - kSuffix.size()) == kSuffix;
}

// True when a sharded load keeps the internal tensor `name`. `trunk_layers`
// is the model's trunk block count; the MTP block sits at that index and
// belongs to the last stage. A disabled shard keeps every tensor, so the
// single-device load is unchanged.
//
// Usage:
//   if (!ShardKeepsTensor(entry.name, options.shard, layers)) continue;
[[nodiscard]] bool ShardKeepsTensor(std::string_view name,
                                     const ModelShard& shard,
                                     std::size_t trunk_layers);

// The trunk layer range [first, last) a shard owns, so a module limits its
// layer loop to it and stays in step with what the load kept. A disabled
// shard owns every layer; `last_layer` 0 means the last trunk layer.
//
// Usage:
//   const LayerRange range = ShardLayerRange(model.Shard(), cfg.layers);
//   for (std::size_t l = range.first; l < range.last; ++l) { /* layer l */ }
struct LayerRange {
  std::size_t first = 0;
  std::size_t last = 0;
};

[[nodiscard]] LayerRange ShardLayerRange(const ModelShard& shard,
                                         std::size_t trunk_layers);

// True when the `--gpu-layers` resident set keeps the tensor `name` on the
// device. `gpu_layers` 0 keeps everything (the full-upload default). A
// nonzero value keeps the output head, the embedding, the non-block globals
// and the MTP block resident, plus the last gpu_layers - 1 trunk blocks;
// earlier blocks page from system RAM per token. Values past the layer
// count keep everything resident.
//
// Usage:
//   if (!GpuLayerResident(entry.name, options.gpu_layers, layers)) page it;
[[nodiscard]] bool GpuLayerResident(std::string_view name,
                                     std::size_t gpu_layers,
                                     std::size_t trunk_layers);

// True when the `--gpu-layers` resident set keeps the tensor `name` on the
// device through the MoE dense refinement: in a model with routed experts
// (`has_moe`), every non-expert tensor of a trunk block stays resident and
// only the routed-expert weights page per token, because every token uses
// the dense weights whole while touching top_k of num_experts. In a dense
// model (or when paging is off) this is always false, so the plain block
// rule above decides alone. Values past the layer count keep everything
// resident through that rule.
//
// Usage:
//   const bool resident = GpuLayerResident(name, layers, trunk) ||
//                         PagedMoeDenseResident(name, layers, trunk, moe);
[[nodiscard]] bool PagedMoeDenseResident(std::string_view name,
                                         std::size_t gpu_layers,
                                         std::size_t trunk_layers,
                                         bool has_moe);

}  // namespace tessera::core
