#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <vector>

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

// Options for Engine::Generate.
struct GenerateOptions {
  // Greedy decode steps to run; 0 produces no tokens.
  std::size_t max_tokens = 0;
  // The first token fed to the decoder when prompt_tokens is empty.
  std::uint32_t first_token = 0;
  // Prompt tokens fed before generation; when non-empty they take the
  // place of first_token and the last one seeds generation.
  std::vector<std::uint32_t> prompt_tokens;
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

  // Greedy single-token generation on the non-speculative (reference
  // baseline) path: run `options.max_tokens` decode steps from
  // `options.first_token` and return the produced token ids. An empty
  // request returns an empty vector. MalformedFile/UnsupportedFeature
  // when the model config cannot drive the decoder.
  //
  // Usage:
  //   auto ids = engine->Generate(*model, {.max_tokens = 4});
  [[nodiscard]] std::expected<std::vector<std::uint32_t>, StatusCode>
  Generate(Model& model, const GenerateOptions& options = {});

  // Streaming variant: call `on_token` for each produced token; stop
  // early when it returns false. Returns the produced count.
  [[nodiscard]] std::expected<std::size_t, StatusCode> GenerateStreaming(
      Model& model, const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token);

  // One multi-token-prediction draft for the token after `token`: run a
  // decode step to get the backbone hidden state, then the MTP head.
  // UnsupportedFeature when the model has no MTP block.
  [[nodiscard]] std::expected<std::uint32_t, StatusCode> MtpDraft(
      Model& model, std::uint32_t token);

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
