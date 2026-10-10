#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "core/think_budget.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/speculative.hpp"

// Shared setup for the generation entry points (generate.cpp,
// generate_multimodal.cpp): one canonical implementation lives here,
// never copied into the callers.

namespace tessera {

// The stop set for one request: the model's declared stop tokens first,
// then the caller's extras from GenerateOptions, without duplicates.
[[nodiscard]] std::vector<std::uint32_t> StopSet(
    const Model& model, const GenerateOptions& options);

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
// end offset. Null progress disables chunk reports.
[[nodiscard]] std::expected<void, StatusCode> PrefillCapturing(
    Backend& backend, Model& model, core::DecodeCache& cache,
    SpeculativeStrategy& strategy, std::span<const std::uint32_t> prompt,
    std::size_t chunk, const std::vector<std::size_t>& capture_layers,
    std::vector<Buffer*>& strategy_buffers, std::vector<float>& hidden_out,
    std::vector<float>& logits_out, const core::PrefillProgress* progress);

}  // namespace tessera
