#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "tessera/types.hpp"

// The pluggable architecture interface. Each architecture is one module
// under src/models/<arch>/ that owns the architecture specific decode
// behavior; the generic engine, scheduler and kernels stay in src/core.
// Modules register in src/models/registry.cpp, selected at model load
// time by the file's architecture name (general.architecture). Generic
// code calls the interface and never names a model. See AGENTS.md.
//
// Usage:
//   auto* arch = model.Architecture();
//   if (arch != nullptr) arch->Draft(backend, model, cache, hidden, tok, pos);

namespace tessera {

class Backend;
class Model;

namespace core {
struct DecodeCache;
}  // namespace core

class Architecture {
 public:
  virtual ~Architecture() = default;

  // The architecture name (the model file's general.architecture).
  [[nodiscard]] virtual std::string_view Name() const = 0;

  // One multi-token-prediction draft step: fuse the token embedding and
  // the backbone hidden, run the draft block and return the drafted
  // token. `mtp_hidden_out`, when non-null, receives the draft hidden to
  // chain the next draft step. UnsupportedFeature when the architecture
  // has no draft head.
  [[nodiscard]] virtual std::expected<std::uint32_t, StatusCode> Draft(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const float> hidden, std::uint32_t token, std::uint64_t pos,
      std::vector<float>* mtp_hidden_out) const = 0;
};

// Build the architecture module for `arch`, or nullptr when no module is
// registered (the caller then falls back to the generic path).
[[nodiscard]] std::unique_ptr<Architecture> CreateArchitecture(
    std::string_view arch);

}  // namespace tessera
