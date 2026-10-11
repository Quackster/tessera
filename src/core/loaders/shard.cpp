#include "core/loaders/shard.hpp"

namespace tessera::core {

namespace {

// The layer index in an internal `blk.<N>.` name, or nullopt when the name
// is not a block tensor.
bool BlockIndex(std::string_view name, std::size_t* index) {
  constexpr std::string_view kPrefix = "blk.";
  if (name.rfind(kPrefix, 0) != 0) {
    return false;
  }
  std::size_t value = 0;
  std::size_t digits = 0;
  for (std::size_t i = kPrefix.size(); i < name.size(); ++i) {
    const char c = name[i];
    if (c == '.') {
      break;
    }
    if (c < '0' || c > '9') {
      return false;
    }
    value = value * 10 + static_cast<std::size_t>(c - '0');
    ++digits;
  }
  if (digits == 0) {
    return false;
  }
  *index = value;
  return true;
}

}  // namespace

bool ShardKeepsTensor(std::string_view name, const ModelShard& shard,
                      std::size_t trunk_layers) {
  if (!shard.enabled) {
    return true;
  }
  if (name == "token_embd.weight") {
    return shard.first_stage;
  }
  if (name == "output.weight" || name == "output_norm.weight") {
    return shard.last_stage;
  }
  std::size_t layer = 0;
  if (BlockIndex(name, &layer)) {
    // The MTP block (blk.<trunk>.*: the nextn weights and the block's own
    // attention/FFN tensors) runs after every trunk layer, so the whole
    // group belongs to the last stage. A head-only draft shard (an empty
    // trunk range with both stage flags) keeps exactly this group plus
    // the embedding and the head.
    if (trunk_layers > 0 && layer == trunk_layers) {
      return shard.last_stage;
    }
    const std::size_t end =
        shard.last_layer == 0 ? trunk_layers : shard.last_layer;
    return layer >= shard.first_layer && layer < end;
  }
  // A global tensor with no layer belongs to the first stage.
  return shard.first_stage;
}

}  // namespace tessera::core
