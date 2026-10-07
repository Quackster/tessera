#include "core/decode.hpp"

#include <memory>

namespace tessera::core {

std::expected<std::uint32_t, StatusCode> DecodeStep(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token) {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  if (config->hybrid) {
    if (!cache.hybrid) {
      cache.hybrid = std::make_unique<HybridDecodeCache>();
    }
    return HybridDecodeStep(backend, model, cache, token);
  }
  return DecodeStepDevice(backend, model, cache, token);
}

}  // namespace tessera::core
