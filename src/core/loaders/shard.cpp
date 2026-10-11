#include "core/loaders/shard.hpp"

namespace tessera::core {

namespace {

bool SplitBlockNameInto(std::string_view name, std::size_t* layer,
                        std::string_view* kind) {
  constexpr std::string_view kPrefix = "blk.";
  if (name.rfind(kPrefix, 0) != 0) {
    return false;
  }
  std::size_t value = 0;
  std::size_t digits = 0;
  std::size_t i = kPrefix.size();
  for (; i < name.size(); ++i) {
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
  if (digits == 0 || i >= name.size() || name[i] != '.') {
    return false;
  }
  if (layer != nullptr) {
    *layer = value;
  }
  if (kind != nullptr) {
    *kind = name.substr(i + 1);
  }
  return true;
}

}  // namespace

bool SplitBlockName(std::string_view name, std::size_t* layer,
                    std::string_view* kind) {
  return SplitBlockNameInto(name, layer, kind);
}

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
  if (SplitBlockName(name, &layer, nullptr)) {
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

bool GpuLayerResident(std::string_view name, std::size_t gpu_layers,
                      std::size_t trunk_layers) {
  if (gpu_layers == 0) {
    return true;
  }
  // Without a trunk count the blocks cannot be ordered, so keep all.
  if (trunk_layers == 0) {
    return true;
  }
  // The head and the embedding are small and every token uses them.
  if (name == "token_embd.weight" || name == "output.weight" ||
      name == "output_norm.weight") {
    return true;
  }
  std::size_t layer = 0;
  if (!SplitBlockName(name, &layer, nullptr)) {
    return true;
  }
  // The MTP block runs after every trunk layer, so it stays resident.
  if (trunk_layers > 0 && layer == trunk_layers) {
    return true;
  }
  // Resident trunk blocks are the last gpu_layers - 1: gpu_layers 1 keeps
  // the output head only, gpu_layers 2 adds the last block, and so on.
  const std::size_t resident_blocks =
      gpu_layers - 1 > trunk_layers ? trunk_layers : gpu_layers - 1;
  const std::size_t first_resident = trunk_layers - resident_blocks;
  return layer >= first_resident && layer < trunk_layers;
}

bool PagedMoeDenseResident(std::string_view name, std::size_t gpu_layers,
                           std::size_t trunk_layers, bool has_moe) {
  if (!has_moe || gpu_layers == 0 || trunk_layers == 0) {
    return false;
  }
  if (GpuLayerResident(name, gpu_layers, trunk_layers)) {
    return false;
  }
  std::size_t layer = 0;
  std::string_view kind;
  if (!SplitBlockName(name, &layer, &kind) || layer >= trunk_layers) {
    return false;
  }
  return !IsRoutedExpertKind(kind);
}

}  // namespace tessera::core
