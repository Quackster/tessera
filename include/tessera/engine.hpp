#pragma once

#include <memory>

#include "tessera/backend.hpp"
#include "tessera/log.hpp"
#include "tessera/model.hpp"
#include "tessera/speculative.hpp"

namespace tessera {

// Options for Engine::Create. The backend itself is selected at configure
// time (TESSERA_BACKEND); there is nothing to pick here yet.
struct EngineOptions {
  // The channel the engine (and its backend) logs through. A default
  // stderr sink is used unless the caller replaces it.
  log::Diagnostics diagnostics;
};

// Top-level facade: owns the backend and the loaded models.
//
// Usage:
//   auto engine = tessera::Engine::Create({});
//   if (!engine) return 1;
//   auto model = engine->LoadModel({"~/models/Qwen3.8-27B-GGUF/....gguf"});
class Engine {
 public:
  // Create the engine and initialize the configured backend.
  static std::expected<std::unique_ptr<Engine>, StatusCode> Create(
      const EngineOptions& options = {});

  // Load a model (see Model::Load for format detection).
  std::expected<std::unique_ptr<Model>, StatusCode> LoadModel(
      const ModelOptions& options);

  // Attach a speculative-decoding strategy (off by default). Takes
  // ownership; the strategy must not be attached elsewhere.
  std::expected<void, StatusCode> AttachSpeculative(
      std::unique_ptr<SpeculativeStrategy>&& strategy);

  // The engine's compute backend.
  [[nodiscard]] Backend& Owner();
  // nullptr when no strategy is attached.
  [[nodiscard]] const SpeculativeStrategy* Speculative() const;
  [[nodiscard]] log::Diagnostics& Diagnostics();

 private:
  Engine(std::unique_ptr<Backend> backend, log::Diagnostics diagnostics);
  std::unique_ptr<Backend> backend_;
  std::unique_ptr<SpeculativeStrategy> speculative_;
  log::Diagnostics diagnostics_;
};

}  // namespace tessera
