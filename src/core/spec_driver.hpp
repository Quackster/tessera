#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include "core/decode.hpp"
#include "tessera/architecture.hpp"
#include "tessera/backend.hpp"
#include "tessera/engine.hpp"
#include "tessera/log.hpp"
#include "tessera/model.hpp"
#include "tessera/speculative.hpp"
#include "tessera/types.hpp"

// The speculative decode driver and its target seam. The driver owns the
// accept rule and the token loop; a target owns the forward, the KV state
// and the drafter capture for one route. The single-device engine and each
// parallel layout implement a target, so one driver serves them all
// (AGENTS.md rule 2). Host-side; the driver runs on one generation thread.

namespace tessera::core {

// The target of one decode (greedy or speculative). `strategy` is null for
// plain greedy. All methods run on the driver's thread, so a target is
// externally synchronized and not safe for concurrent use.
class SpecTarget {
 public:
  virtual ~SpecTarget() = default;

  [[nodiscard]] virtual log::Diagnostics& diagnostics() = 0;
  [[nodiscard]] virtual Backend& backend() = 0;
  [[nodiscard]] virtual Model& model() = 0;
  [[nodiscard]] virtual DecodeCache& cache() = 0;

  // One-time trunk init, the strategy's Prepare and draft-context override,
  // and the capture setup. `strategy` may be null. Idempotent enough for one
  // driver call.
  virtual std::expected<void, StatusCode> Prepare(
      SpeculativeStrategy* strategy, const GenerateOptions& options) = 0;

  // Prefill `prompt` in the target's own chunking. When `strategy` captured
  // a drafter tail in Prepare, this also feeds the strategy. Fills the last
  // row's hidden and logits. `compute_ms`, when non-null, accumulates the
  // chunk forwards' wall time.
  virtual std::expected<void, StatusCode> Prefill(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      std::vector<float>* hidden, std::vector<float>* logits,
      double* compute_ms) = 0;

  // One anchor step returning the greedy token (device argmax where the
  // route supports it).
  virtual std::expected<std::uint32_t, StatusCode> GreedyStep(
      std::uint32_t token) = 0;

  // One anchor step returning the full distribution; `hidden`, when
  // non-null, receives the row's residual hidden for the drafter.
  virtual std::expected<std::vector<float>, StatusCode> LogitsStep(
      std::uint32_t token, std::vector<float>* hidden) = 0;

  // Verify a draft against the target: the anchor (when set) leads the
  // batch; `hidden`, when non-null, receives the last accepted row's hidden.
  // When `sampled` is non-null the target draws its token per row and
  // matches it (see SampledVerify), so the accepted prefix equals plain
  // sampling for the same seed.
  virtual std::expected<DraftVerification, StatusCode> Verify(
      std::span<const std::uint32_t> draft,
      std::span<const float> prefix_logits,
      std::optional<std::uint32_t> anchor, std::vector<float>* hidden,
      const SampledVerify* sampled) = 0;

  // True when Verify honors a non-null `sampled` (a sampled request can
  // still run the drafter). False (the default) means the target only
  // verifies greedily, so the driver drops the strategy for a sampled
  // request and samples without it.
  [[nodiscard]] virtual bool SupportsSampledVerify() const { return false; }

  // Report the target's per-cache and profiler diagnostics after the loop.
  // `loop_steps` is the decode loop's step count for the profiler.
  virtual void ReportDiagnostics(std::size_t loop_steps) = 0;
};

// Run one generation: plain greedy when `strategy` is null, speculative
// decode otherwise. `on_token` returns false to abort. The outcome carries
// the produced count and the prefill/decode wall times.
[[nodiscard]] std::expected<GenerateOutcome, StatusCode> RunDecode(
    SpecTarget& target, SpeculativeStrategy* strategy,
    const GenerateOptions& options,
    const std::function<bool(std::uint32_t)>& on_token);

// Return a copy of `options` carrying `prompt` as its prompt_tokens. An
// entry point that receives the prompt separately from the options uses
// this before RunDecode, so the driver prefills the whole prompt instead of
// falling back to the single `options.first_token`.
//
// Usage:
//   auto filled = core::WithPrompt(options, prompt);
//   core::RunDecode(target, strategy, filled, on_token);
[[nodiscard]] inline GenerateOptions WithPrompt(
    const GenerateOptions& options, std::span<const std::uint32_t> prompt) {
  GenerateOptions filled = options;
  filled.prompt_tokens.assign(prompt.begin(), prompt.end());
  return filled;
}

}  // namespace tessera::core
