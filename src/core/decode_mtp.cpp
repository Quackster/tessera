#include "core/decode.hpp"

#include <vector>

#include "tessera/architecture.hpp"
#include "tessera/model.hpp"

namespace tessera::core {

// One multi-token-prediction draft step. Dispatched to the model's
// architecture module; UnsupportedFeature when it has no draft head.
std::expected<std::uint32_t, StatusCode> MtpDraftStep(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::span<const float> hidden, std::uint32_t token, std::uint64_t pos,
    std::vector<float>* mtp_hidden_out) {
  const Architecture* arch = model.Arch();
  if (arch == nullptr) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  return arch->Draft(backend, model, cache, hidden, token, pos, mtp_hidden_out);
}

}  // namespace tessera::core
