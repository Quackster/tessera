#include "tessera/architecture.hpp"

#include <string_view>

#include "models/qwen3_5/architecture.hpp"

namespace tessera {

std::unique_ptr<Architecture> CreateArchitecture(std::string_view arch) {
  // One line per architecture module. Add the module under
  // src/models/<arch>/ and register it here; do not edit src/core/.
  // The GGUF uses "qwen35"; the MXFP4 config.json uses the HuggingFace
  // names ("qwen3_5" model_type, "Qwen3_5ForConditionalGeneration").
  // The sparse mixture-of-experts variant (Ornith-1.5) shares the same
  // hybrid trunk and selects the MoE FFN from the config, so it maps to
  // the same module.
  if (arch == "qwen35" || arch == "qwen3_5" ||
      arch == "Qwen3_5ForConditionalGeneration") {
    return models::qwen3_5::MakeQwen35Architecture();
  }
  if (arch == "qwen35moe" || arch == "qwen3_5_moe" ||
      arch == "Qwen3_5MoeForConditionalGeneration") {
    return models::qwen3_5::MakeQwen35Architecture();
  }
  return nullptr;
}

}  // namespace tessera
