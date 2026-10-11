#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tessera/types.hpp"

namespace tessera {

class Tokenizer;

// Options for building a pipeline: one stage per device, the trunk layers
// split into contiguous ranges in device order.
struct PipelineOptions {
  // Path to the model (.gguf file or MXFP4 directory).
  std::string model_path;
  // Device index per stage, in stage order. At least one device; the
  // layer count must be at least the device count.
  std::vector<int> devices;
  // Maximum context length to prepare for.
  std::size_t context = 4096;
  // Storage type of the full-attention KV cache.
  KvCacheType kv_type = KvCacheType::F32;
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

  [[nodiscard]] std::size_t StageCount() const;
  // The model tokenizer (identical across stages); nullptr when none.
  [[nodiscard]] const Tokenizer* GetTokenizer() const;
  // Render one user message through the model chat template with the
  // generation prompt appended (same as Model::ChatPrompt).
  [[nodiscard]] std::expected<std::string, StatusCode> ChatPrompt(
      std::string_view user_text, bool enable_thinking = false) const;

 private:
  struct Impl;
  explicit Pipeline(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace tessera
