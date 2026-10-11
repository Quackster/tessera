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

}  // namespace tessera::core
