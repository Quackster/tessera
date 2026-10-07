#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"

namespace tessera::spec {

// The DFlash2 candidate-selector block: project the draft hidden states to
// the selector rank, then score the predecessor/successor transitions of
// the unary top-K candidates. `hidden` is rows x hidden, the projection
// weight rank x hidden, the codebooks vocab x rank, `candidate_ids` (int32)
// rows x top_k, `anchor_ids` (int32) rows, `unary` rows x top_k, and `out`
// rows x top_k x top_k.
[[nodiscard]] std::expected<void, StatusCode> DraftSelectorRef(
    std::span<const float> hidden, std::span<const float> projection_w,
    std::span<const float> predecessor, std::span<const float> successor,
    std::span<const std::int32_t> candidate_ids,
    std::span<const std::int32_t> anchor_ids, std::span<const float> unary,
    std::span<float> out, std::size_t rows, std::size_t hidden_dim,
    std::size_t rank, std::size_t vocab, std::size_t top_k);

// Device version. `gemm` is gemm_f32, `edge` the selector_edge_score
// built-in, and `proj` a rows x rank scratch buffer.
[[nodiscard]] std::expected<void, StatusCode> DraftSelectorDevice(
    Backend& backend, const Kernel& gemm, const Kernel& edge, Buffer& proj,
    const Buffer& hidden, const Buffer& projection_w,
    const Buffer& predecessor, const Buffer& successor,
    const Buffer& candidate_ids, const Buffer& anchor_ids,
    const Buffer& unary, Buffer& out, std::size_t rows, std::size_t hidden_dim,
    std::size_t rank, std::size_t vocab, std::size_t top_k);

}  // namespace tessera::spec
