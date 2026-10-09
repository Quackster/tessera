#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>
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
  // GPU index to run on (default 0, the first GPU). Create fails with
  // InvalidArgument when the index is out of range.
  int device_index = 0;
  // Tokens per prefill forward; 0 selects automatically from the model
  // context at generation time. Chunking is not applied yet.
  //
  // Usage:
  //   EngineOptions options;
  //   options.prefill_chunk_tokens = 512;
  std::size_t prefill_chunk_tokens = 0;
};

// Sampling parameters, with the Qwen 3.8 27B defaults. Applied in this
// order: repetition and presence penalties over the token history, then
// temperature, then min_p, top_k and top_p. temperature 0 means greedy.
struct SamplingOptions {
  float temperature = 0.6f;
  float top_p = 0.95f;
  int top_k = 20;
  float min_p = 0.0f;
  float presence_penalty = 0.0f;
  float repetition_penalty = 1.0f;
};

// Options for Engine::Generate.
struct GenerateOptions {
  // Decode steps to run; 0 fills the remaining context (see
  // Model::EffectiveMaxTokens) instead of producing no tokens.
  std::size_t max_tokens = 0;
  // The first token fed to the decoder when prompt_tokens is empty.
  std::uint32_t first_token = 0;
  // Prompt tokens fed before generation; when non-empty they take the
  // place of first_token and the last one seeds generation.
  std::vector<std::uint32_t> prompt_tokens;
  // When false (the default) the reference greedy path runs. When true,
  // tokens are drawn from `sampling` with the fixed `seed`.
  bool sample = false;
  SamplingOptions sampling;
  // RNG seed for sampling; a fixed value keeps sampling reproducible.
  std::uint64_t seed = 0;
  // Draft tokens per speculative step for the DFlash2 draft; 0 uses the
  // draft checkpoint's configured block size.
  std::size_t draft_tokens = 0;
  // Log progress every this many prefill steps (0 disables progress logs).
  std::size_t progress_every = 64;
  // Storage type of the full-attention KV cache (default fp32).
  KvCacheType kv_type = KvCacheType::F32;
  // Extra stop token ids for this request, added to the model's declared
  // stop tokens (Model::StopTokens). Generation ends when it samples one
  // of them, and the stop token is not emitted. Use this for a chat
  // template's end-of-turn token when the definition does not declare it.
  std::vector<std::uint32_t> stop_tokens;
  // Tokens per prefill forward for this request; 0 uses the engine
  // default, which itself resolves to the model context when 0.
  //
  // Usage:
  //   GenerateOptions gen;
  //   gen.prefill_chunk_tokens = 512;
  std::size_t prefill_chunk_tokens = 0;
};

// Resolve the prefill chunk for one request. The request value wins
// when nonzero; otherwise the engine default wins; otherwise the
// model context is used (current single-forward behavior).
//
// Usage:
//   const std::size_t chunk = ResolvePrefillChunkTokens(
//       engine.PrefillChunkTokens(), gen.prefill_chunk_tokens,
//       model.MaxContextLength());
[[nodiscard]] inline std::size_t ResolvePrefillChunkTokens(
    std::size_t engine_default, std::size_t request_override,
    std::size_t max_context) {
  if (request_override > 0) {
    return request_override;
  }
  if (engine_default > 0) {
    return engine_default;
  }
  return max_context > 0 ? max_context : 1;
}

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
  // `options.first_token` and return the produced token ids. A zero
  // count fills the remaining context; a prompt that already fills it
  // returns an empty vector. MalformedFile/UnsupportedFeature
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

  // Generation with image input. The prompt may contain `image_token_id`
  // placeholders; the first `image_tokens` of them consume consecutive rows
  // of `image_embeddings` (image_tokens x hidden, row-major) instead of the
  // token embedding during prefill. The rest of generation is as Generate.
  [[nodiscard]] std::expected<std::vector<std::uint32_t>, StatusCode>
  GenerateMultimodal(Model& model, const GenerateOptions& options,
                     std::span<const float> image_embeddings,
                     std::size_t image_tokens, std::uint32_t image_token_id);

  // Greedy generation with the DFlash2 draft checkpoint at `draft_path`.
  // The output equals plain greedy decoding. UnsupportedFeature when the
  // target is not hybrid or the draft does not load.
  [[nodiscard]] std::expected<std::vector<std::uint32_t>, StatusCode>
  GenerateDraft(Model& model, const GenerateOptions& options,
                const std::string& draft_path);

  // Greedy generation with MTP speculative decoding. Each step drafts one
  // token with the MTP head, verifies it against the target, and accepts
  // it when it matches; a rejected draft falls back to the target's greedy
  // token. The produced sequence is identical to Generate (speculation
  // never changes the output). UnsupportedFeature on a non-hybrid model.
  //
  // Usage:
  //   auto ids = engine->GenerateSpeculative(*model, {.max_tokens = 8});
  [[nodiscard]] std::expected<std::vector<std::uint32_t>, StatusCode>
  GenerateSpeculative(Model& model, const GenerateOptions& options = {});

  // The engine's compute backend.
  [[nodiscard]] Backend& Owner();
  // nullptr when no strategy is attached.
  [[nodiscard]] const SpeculativeStrategy* Speculative() const;
  [[nodiscard]] log::Diagnostics& Diagnostics();
  // Configured prefill chunk tokens (0 means automatic).
  //
  // Usage:
  //   const std::size_t chunk = engine.PrefillChunkTokens();
  [[nodiscard]] std::size_t PrefillChunkTokens() const;

 private:
  Engine(std::unique_ptr<Backend> backend, log::Diagnostics diagnostics,
         std::size_t prefill_chunk_tokens);
  std::unique_ptr<Backend> backend_;
  std::unique_ptr<SpeculativeStrategy> speculative_;
  log::Diagnostics diagnostics_;
  std::size_t prefill_chunk_tokens_ = 0;
};

}  // namespace tessera
