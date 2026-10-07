#include "tessera/engine.hpp"

#include <string>
#include <vector>

#include "core/decode.hpp"

namespace tessera {

Engine::Engine(std::unique_ptr<Backend> backend, log::Diagnostics diagnostics)
    : backend_(std::move(backend)), diagnostics_(std::move(diagnostics)) {}

std::expected<std::unique_ptr<Engine>, StatusCode> Engine::Create(
    const EngineOptions& options) {
  log::Diagnostics log = options.diagnostics;
  auto backend = CreateBackend();
  if (!backend) {
    log.Error("engine", "CreateBackend returned null; the build has no backend");
    return std::unexpected(StatusCode::DeviceError);
  }
  auto engine =
      std::unique_ptr<Engine>(new Engine(std::move(backend), log));
  // The backend logs through the engine's diagnostics copy.
  engine->backend_->SetDiagnostics(&engine->diagnostics_);
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
                   std::string(engine->backend_->Name()) + " (" +
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

std::expected<std::size_t, StatusCode> Engine::GenerateStreaming(
    Model& model, const GenerateOptions& options,
    const std::function<bool(std::uint32_t)>& on_token) {
  if (options.max_tokens == 0) {
    return 0;
  }
  core::DecodeCache cache;
  std::vector<std::uint32_t> prompt = options.prompt_tokens;
  if (prompt.empty()) {
    prompt.push_back(options.first_token);
  }
  std::uint32_t next = prompt.back();
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    auto decoded = core::DecodeStep(*backend_, model, cache, prompt[i]);
    if (!decoded) {
      diagnostics_.Warn(
          "engine", std::string("prompt step ") + std::to_string(i) +
                        " failed: " + std::string(ToString(decoded.error())));
      return std::unexpected(decoded.error());
    }
    next = *decoded;
  }
  std::size_t produced = 0;
  for (std::size_t step = 0; step < options.max_tokens; ++step) {
    if (!on_token(next)) {
      break;
    }
    ++produced;
    if (step + 1 == options.max_tokens) {
      break;
    }
    auto decoded = core::DecodeStep(*backend_, model, cache, next);
    if (!decoded) {
      diagnostics_.Warn(
          "engine", std::string("generation step ") + std::to_string(step) +
                        " failed: " + std::string(ToString(decoded.error())));
      return std::unexpected(decoded.error());
    }
    next = *decoded;
  }
  return produced;
}

std::expected<std::vector<std::uint32_t>, StatusCode> Engine::Generate(
    Model& model, const GenerateOptions& options) {
  std::vector<std::uint32_t> produced;
  produced.reserve(options.max_tokens);
  auto streamed = GenerateStreaming(
      model, options, [&produced](std::uint32_t token) {
        produced.push_back(token);
        return true;
      });
  if (!streamed) {
    return std::unexpected(streamed.error());
  }
  return produced;
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

}  // namespace tessera
