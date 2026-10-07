#include "tessera/architecture.hpp"

#include <string_view>

#include "models/qwen3_5/architecture.hpp"

namespace tessera {

std::unique_ptr<Architecture> CreateArchitecture(std::string_view arch) {
  // One line per architecture module. Add the module under
  // src/models/<arch>/ and register it here; do not edit src/core/.
  if (arch == "qwen35") {
    return models::qwen3_5::MakeQwen35Architecture();
  }
  return nullptr;
}

}  // namespace tessera
