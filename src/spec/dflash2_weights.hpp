#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <memory>
#include <vector>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"
#include "spec/dflash2_config.hpp"
#include "spec/dflash2_layer.hpp"

namespace tessera::spec {

// The draft forward weights in fp32 on the device, plus the shared fc
// (target fusion), hidden_norm (context) and final norm.
struct DraftForwardWeights {
  std::vector<DraftLayerBuffers> layers;
  const Buffer* fc = nullptr;
  const Buffer* hidden_norm = nullptr;
  const Buffer* final_norm = nullptr;
};

// Loads a DFlash2 draft checkpoint: reads the safetensors file, converts
// every forward tensor from bf16 or 128x128 block-scaled fp8 to fp32, and
// uploads it. The token embedding and output weight are shared with the
// target model and are not loaded here.
class DraftWeightStore {
 public:
  DraftWeightStore() = default;
  DraftWeightStore(DraftWeightStore&&) = default;
  DraftWeightStore& operator=(DraftWeightStore&&) = default;
  DraftWeightStore(const DraftWeightStore&) = delete;
  DraftWeightStore& operator=(const DraftWeightStore&) = delete;

  // FileNotFound / MalformedFile for a bad checkpoint, UnsupportedFeature
  // for an unknown tensor dtype, OutOfMemory on upload failure.
  [[nodiscard]] static std::expected<DraftWeightStore, StatusCode> Load(
      Backend& backend, const std::filesystem::path& dir,
      const DFlash2Config& config);

  [[nodiscard]] const DraftForwardWeights& Weights() const {
    return weights_;
  }

 private:
  std::vector<std::unique_ptr<Buffer>> owned_;
  DraftForwardWeights weights_;
};

}  // namespace tessera::spec
