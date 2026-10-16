#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tessera/defaults.hpp"
#include "tessera/engine.hpp"
#include "tessera/types.hpp"

namespace tessera {

class Engine;
class Model;
class Tokenizer;

// Options for building a pipeline: one stage per device, the trunk layers
// split into contiguous ranges in device order.
struct PipelineOptions {
  // Path to the model (.gguf file or MXFP4 directory).
  std::string model_path;
  // Device index per stage, in stage order. At least one device; the
  // layer count must be at least the device count.
  std::vector<int> devices;
  // Trunk layers per stage, in stage order. Empty splits the layers as
  // evenly as the count allows. When set, one entry per device, each at
  // least one, summing to the model's trunk layer count. A stage that
  // carries extra weight (the output head, an MTP head) can be given fewer
  // layers so every device fits.
  std::vector<std::size_t> stage_layers;
  // Maximum context length to prepare for.
  std::size_t context = kDefaultContext;
  // Storage type of the full-attention KV cache.
  KvCacheType kv_type = KvCacheType::F32;
  // Directory of a separately fetched MTP head, loaded on the last stage,
  // which owns the head for the attached drafter. Empty when the model's
  // own GGUF carries the nextn block.
  std::string mtp_path;
  // Also keep the token embedding on the last stage. A drafter that runs
  // there and embeds the anchor tokens (DFlash2) needs it; off by default
  // because it duplicates the embedding.
  bool last_stage_embedding = false;
  // Trunk layers kept in VRAM per stage; the rest page from system RAM. 0
  // (the default) uploads a stage's whole range. Counted within a stage's
  // own layer range, so every stage pages its own earliest layers.
  std::size_t gpu_layers = 0;
};

// A model split across devices by layer range. Each stage runs its own
// layers on its own device and hands the hidden residual stream to the
// next stage with a cross-device copy. The first stage gathers the token
// embedding; the last stage runs the output head. Greedy output equals
// the single-device model.
//
// Usage:
//   PipelineOptions options;
//   options.model_path = path;
//   options.devices = {0, 1};
//   auto pipeline = Pipeline::Create(options);
//   if (pipeline) {
//     auto ids = (*pipeline)->Generate(prompt, 32);
//   }
class Pipeline {
 public:
  static std::expected<std::unique_ptr<Pipeline>, StatusCode> Create(
      const PipelineOptions& options);

  ~Pipeline();
  Pipeline(Pipeline&& other) noexcept;
  Pipeline& operator=(Pipeline&& other) noexcept;
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  // Greedy generation. `prompt` seeds the sequence; up to `max_tokens`
  // are produced (0 fills the remaining context). Stops at a declared
  // stop token. Resets the per-stage caches first, so one Pipeline can
  // serve one request at a time.
  std::expected<std::vector<std::uint32_t>, StatusCode> Generate(
      std::span<const std::uint32_t> prompt, std::size_t max_tokens);

  // Greedy generation with per-request speed tuning, timed. Applies
  // `options.kv_type`, `options.mxfp4_split_target`, `options.mxfp4_split_cap`
  // and `options.attention_split` to every stage, caps the decode at
  // `options.max_completion_tokens` (0 fills the context), and returns the
  // produced count with the prefill and decode wall times. Calibration uses
  // the times to score a setting; the produced tokens equal Generate for the
  // same input.
  //
  // Usage:
  //   auto outcome = pipeline->GenerateTimed(prompt, tuning);
  std::expected<GenerateOutcome, StatusCode> GenerateTimed(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options);

  // Streaming generation over the stages, driven by the shared decode
  // driver (core::RunDecode) so plain greedy, sampling, thinking budgets
  // and an attached strategy all behave as on the single-device engine.
  // The prompt comes from `options.prompt_tokens` (or `first_token`).
  // `on_token` returns false to abort. Resets the per-stage caches first,
  // so one Pipeline serves one request at a time; the caller serializes.
  //
  // Usage:
  //   auto outcome = pipeline->GenerateStreaming(options, on_token);
  [[nodiscard]] std::expected<GenerateOutcome, StatusCode> GenerateStreaming(
      const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token);

  // Streaming generation for a concurrent serve turn. Takes a free sequence
  // from the pool, holds the pipeline's device lock for the turn, and yields
  // the device between tokens so another in-flight turn advances beside it.
  // Safe for concurrent use by several caller threads; the pool bounds the
  // in-flight turns. `cancelled`, when set, stops a turn that is waiting for
  // a free sequence. Returns InvalidArgument when the caller cancels while
  // waiting for a sequence.
  //
  // Usage:
  //   pool.SubmitLeastLoaded(options, on_token);
  [[nodiscard]] std::expected<GenerateOutcome, StatusCode>
  GenerateStreamingConcurrent(
      const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled = {});

  // Attach a speculative strategy (off by default). The factory is called
  // once; the drafter runs on the last stage, which owns the MTP head, while
  // the target forward runs across every stage.
  //
  // Usage:
  //   pipeline->AttachSpeculative(CreateMtpStrategy, StrategyOptions{});
  std::expected<void, StatusCode> AttachSpeculative(
      const std::function<std::unique_ptr<SpeculativeStrategy>()>& factory,
      const StrategyOptions& options);

  // Greedy generation for several prompts at once, overlapped across the
  // stages: each stage runs on its own thread, so while one stage
  // processes one sequence the next stage processes another. Each
  // sequence's tokens equal a separate Generate. `max_tokens` applies per
  // prompt (0 fills the remaining context).
  //
  // Usage:
  //   std::vector<std::vector<std::uint32_t>> prompts = ...;
  //   auto out = pipeline->GenerateMany(prompts, 32);
  std::expected<std::vector<std::vector<std::uint32_t>>, StatusCode>
  GenerateMany(std::span<const std::vector<std::uint32_t>> prompts,
               std::size_t max_tokens);

  [[nodiscard]] std::size_t StageCount() const;
  // Stage `i`'s engine and model for measurement (for example per-device
  // weight bytes). The driver owns them; `i` must be below StageCount().
  [[nodiscard]] Engine& StageEngine(std::size_t i);
  [[nodiscard]] Model& StageModel(std::size_t i);
  // Owned trunk layer ranges per stage ([first, last)), in stage order,
  // for scaling per-stage budgets by layer count.
  [[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> StageRanges()
      const;
  // The model tokenizer (identical across stages); nullptr when none.
  [[nodiscard]] const Tokenizer* GetTokenizer() const;
  // Render one user message through the model chat template with the
  // generation prompt appended (same as Model::ChatPrompt).
  [[nodiscard]] std::expected<std::string, StatusCode> ChatPrompt(
      std::string_view user_text, bool enable_thinking = false,
      std::string_view reasoning_effort = "") const;

 private:
  struct Impl;
  explicit Pipeline(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace tessera
