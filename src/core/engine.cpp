#include "tessera/engine.hpp"

#include <memory>
#include <string>
#include <utility>

namespace tessera {

Engine::Engine(std::unique_ptr<Backend> backend, log::Diagnostics diagnostics,
               std::size_t prefill_chunk_tokens)
    : backend_(std::move(backend)),
      diagnostics_(std::move(diagnostics)),
      prefill_chunk_tokens_(prefill_chunk_tokens) {}

std::expected<std::unique_ptr<Engine>, StatusCode> Engine::Create(
    const EngineOptions& options) {
  log::Diagnostics log = options.diagnostics;
  auto backend = CreateBackend();
  if (!backend) {
    log.Error("engine", "CreateBackend returned null; the build has no backend");
    return std::unexpected(StatusCode::DeviceError);
  }
  auto engine = std::unique_ptr<Engine>(
      new Engine(std::move(backend), log, options.prefill_chunk_tokens));
  // The backend logs through the engine's diagnostics copy.
  engine->backend_->SetDiagnostics(&engine->diagnostics_);
  engine->backend_->SetDeviceIndex(options.device_index);
  auto init = engine->backend_->Init();
  if (!init) {
    engine->diagnostics_.Error(
        "engine", std::string("backend init failed: ") +
                      std::string(ToString(init.error())) +
                      "; engine not started");
    return std::unexpected(init.error());
  }
  engine->diagnostics_.Info(
      "engine", std::string("engine started on ") +
                   std::string(engine->backend_->Name()) + " (device " +
                   std::to_string(options.device_index) + ": " +
                   std::string(engine->backend_->DeviceName()) + ")");
  return engine;
}

std::expected<std::unique_ptr<Model>, StatusCode> Engine::LoadModel(
    const ModelOptions& options) {
  auto model = Model::Load(*backend_, options);
  if (!model) {
    diagnostics_.Warn(
        "model", std::string("load failed for '") + options.path + "': " +
                     std::string(ToString(model.error())) +
                     "; check the path and file format");
    return model;
  }
  auto& loaded = *model;
  const char* format_name =
      loaded->Format() == ModelFormat::Gguf ? "gguf" : "mxfp4";
  diagnostics_.Info(
      "model", std::string("loaded ") + format_name + " model with " +
                   std::to_string(loaded->Tensors().size()) + " tensors");
  return model;
}

std::expected<void, StatusCode> Engine::AttachSpeculative(
    std::unique_ptr<SpeculativeStrategy>&& strategy) {
  if (!strategy) {
    diagnostics_.Warn("engine", "AttachSpeculative: null strategy");
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (speculative_) {
    diagnostics_.Warn("engine",
                     "AttachSpeculative: a strategy is already attached");
    return std::unexpected(StatusCode::InvalidArgument);
  }
  speculative_ = std::move(strategy);
  diagnostics_.Info(
      "engine", std::string("speculative strategy '") +
                   std::string(speculative_->Name()) + "' attached");
  return {};
}

Backend& Engine::Owner() {
  return *backend_;
}

const SpeculativeStrategy* Engine::Speculative() const {
  return speculative_.get();
}

log::Diagnostics& Engine::Diagnostics() {
  return diagnostics_;
}

std::size_t Engine::PrefillChunkTokens() const {
  return prefill_chunk_tokens_;
}

}  // namespace tessera
