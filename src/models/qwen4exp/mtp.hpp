#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "models/qwen4exp/internal.hpp"
#include "tessera/types.hpp"

// The Qwen3.8-Flash-Next MTP draft head. The served GGUF has no MTP block,
// so the module loads it from the directory the fetch wrote (see
// tessera/mtp_fetch.hpp) on the first draft. The block is one QSA attention
// plus one MoE wrapped in hyper-connections, with two input branches (the
// normed token embedding and the normed target residual stack) and a final
// mixer before the shared output head. Ported from Strata's MtpDrafter
// (see CREDITS.md).
namespace tessera::models::qwen4exp {

// Load the MTP weights and allocate the draft cache and scratch, once. A
// no-op when already ready or when the model has no MTP path. The module
// calls this from EnsureReady, before the expert cache sizes itself to the
// free device memory, so the MTP weights get their share first.
[[nodiscard]] std::expected<void, StatusCode> EnsureMtpReady(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    const Qwen4ExpGeometry& g, Qwen4ExpState& h);

// One MTP draft step: consume the target's final multi-stream residual
// `hidden` (hc * hidden) and the anchor `token`, run the MTP block at KV
// cell `pos`, and return the drafted token. When `mtp_hidden_out` is
// non-null it receives the post-FFN residual stack, which the next chained
// step consumes as its `hidden`.
[[nodiscard]] std::expected<std::uint32_t, StatusCode> RunMtpDraft(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    const Qwen4ExpGeometry& g, Qwen4ExpState& h, std::span<const float> hidden,
    std::uint32_t token, std::uint64_t pos, std::vector<float>* mtp_hidden_out);

// Committed MTP KV rows, for the strategy's truncation after a verify.
[[nodiscard]] std::size_t MtpDraftRows(const Qwen4ExpState& h);
void MtpDraftTruncate(Qwen4ExpState& h, std::size_t rows);

}  // namespace tessera::models::qwen4exp
