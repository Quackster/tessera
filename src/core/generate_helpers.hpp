#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "core/think_budget.hpp"
#include "tessera/engine.hpp"
#include "tessera/log.hpp"
#include "tessera/model.hpp"
#include "tessera/speculative.hpp"

// Shared setup for the generation entry points (generate.cpp,
// generate_batch.cpp, generate_multimodal.cpp, scheduler.cpp): one
// canonical implementation lives here, never copied into callers.

namespace tessera {

// Sampling parameters are in range; InvalidArgument otherwise (a
// request is rejected before any device work). Warns once per call.
[[nodiscard]] StatusCode ValidateSamplingOptions(
    const SamplingOptions& sampling, log::Diagnostics& diagnostics);

// The stop set for one request: the model's declared stop tokens first,
// then the caller's extras from GenerateOptions, without duplicates.
[[nodiscard]] std::vector<std::uint32_t> StopSet(
    const Model& model, const GenerateOptions& options);

// One token from a logits row: the greedy argmax, or the per-sequence
// seeded draw when sampling (penalties read the token history).
[[nodiscard]] std::uint32_t PickToken(std::span<const float> logits,
                                      bool sample,
                                      const SamplingOptions& sampling,
                                      const std::vector<std::uint32_t>& history,
                                      std::mt19937_64& rng);

// One greedy step for several sequences: a single batched forward when
// the architecture supports it, else one DecodeToken per sequence.
// Returns the next token per input sequence, in input order. Anything
// but UnsupportedFeature from the batched path fails the call.
[[nodiscard]] std::expected<std::vector<std::uint32_t>, StatusCode>
AdvanceGreedyCaches(Backend& backend, const Model& model,
                     log::Diagnostics& diagnostics,
                     std::span<core::DecodeCache* const> caches,
                     std::span<const std::uint32_t> tokens);

// Think tags for a thinking budget: the tokenizer's special ids when
// declared, else the BPE encoding of the markers. A disengaged budget
// (plus empty close ids) when the budget is zero, the model has no
// tokenizer, or the tags do not resolve.
[[nodiscard]] std::pair<core::ThinkBudget, std::vector<std::uint32_t>>
ResolveThinkBudget(const Model& model, std::size_t budget);

// Prefill with target-hidden capture for a hidden-conditioned drafter:
// earlier positions prefill without capture (the draft keeps its recent
// window) and the tail runs chunked with engine-owned capture buffers
// plus one strategy append per chunk. Only a strategy without a batched
// append keeps the per-token loop. Every chunk stays within the
// per-launch attention work budget; the cache is empty (prefill starts
// a request), so the keys a chunk's last row sees end at its absolute
// end offset. Null progress disables chunk reports. `compute_ms`, when
// non-null, accumulates the chunk forwards' wall time in fractional
// milliseconds (the llama.cpp-comparable compute number).
[[nodiscard]] std::expected<void, StatusCode> PrefillCapturing(
    Backend& backend, Model& model, core::DecodeCache& cache,
    SpeculativeStrategy& strategy, std::span<const std::uint32_t> prompt,
    std::size_t chunk, const std::vector<std::size_t>& capture_layers,
    std::vector<Buffer*>& strategy_buffers, std::vector<float>& hidden_out,
    std::vector<float>& logits_out, const core::PrefillProgress* progress,
    double* compute_ms = nullptr);

}  // namespace tessera
