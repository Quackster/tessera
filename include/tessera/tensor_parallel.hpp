#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tessera/defaults.hpp"
#include "tessera/engine.hpp"
#include "tessera/types.hpp"

namespace tessera {

class Engine;
class Model;
class Tokenizer;

// Options for building a tensor-parallel engine: one rank per device. The
// FFN projections are split across the ranks; everything else is
// replicated, so the ranks stay identical outside the all-reduced FFN.
struct TensorParallelOptions {
  std::string model_path;
  std::vector<int> devices;  // one rank per device, in rank order
  std::size_t context = kDefaultContext;
  KvCacheType kv_type = KvCacheType::F32;
};

// A model split across devices by tensor parallelism. Each rank loads the
// FFN projections sliced (column-parallel gate/up, row-parallel down) and
// the rest whole; every rank runs the same forward on its own thread,
// rendezvous at each FFN down projection, all-reduce the partial, and add
// the sum to the residual. Greedy output equals the single-device model.
//
// Usage:
//   TensorParallelOptions options;
//   options.model_path = path;
//   options.devices = {0, 1};
//   auto tp = TensorParallel::Create(options);
//   if (tp) { auto ids = (*tp)->Generate(prompt, 32); }
class TensorParallel {
 public:
  static std::expected<std::unique_ptr<TensorParallel>, StatusCode> Create(
      const TensorParallelOptions& options);

  ~TensorParallel();
  TensorParallel(TensorParallel&& other) noexcept;
  TensorParallel& operator=(TensorParallel&& other) noexcept;
  TensorParallel(const TensorParallel&) = delete;
  TensorParallel& operator=(const TensorParallel&) = delete;

  // Greedy generation through the ranks. `prompt` seeds; up to `max_tokens`
  // (0 fills the context). Stops at a declared stop token.
  std::expected<std::vector<std::uint32_t>, StatusCode> Generate(
      std::span<const std::uint32_t> prompt, std::size_t max_tokens);

  // Greedy generation with per-request speed tuning, timed. Applies
  // `options.kv_type`, `options.mxfp4_split_target`, `options.mxfp4_split_cap`
  // and `options.attention_split` to every rank, caps the decode at
  // `options.max_completion_tokens` (0 fills the context), and returns the
  // produced count with the prefill and decode wall times. Calibration uses
  // the times to score a setting; the produced tokens equal Generate for the
  // same input.
  //
  // Usage:
  //   auto outcome = tp->GenerateTimed(prompt, tuning);
  std::expected<GenerateOutcome, StatusCode> GenerateTimed(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options);

  [[nodiscard]] std::size_t RankCount() const;
  // Rank `i`'s engine and model for measurement (for example per-device
  // weight bytes). The driver owns them; `i` must be below RankCount().
  [[nodiscard]] Engine& RankEngine(std::size_t i);
  [[nodiscard]] Model& RankModel(std::size_t i);
  [[nodiscard]] const Tokenizer* GetTokenizer() const;
  [[nodiscard]] std::expected<std::string, StatusCode> ChatPrompt(
      std::string_view user_text, bool enable_thinking = false,
      std::string_view reasoning_effort = "") const;

 private:
  struct Impl;
  explicit TensorParallel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace tessera
