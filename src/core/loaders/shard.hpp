#pragma once

#include <cstddef>
#include <string_view>

#include "tessera/model.hpp"

namespace tessera::core {

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
