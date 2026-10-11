#pragma once

#include <cstddef>
#include <string_view>

#include "tessera/model.hpp"

namespace tessera::core {

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

}  // namespace tessera::core
