#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "tessera/types.hpp"

namespace tessera::core {

// Host reference for the "selector_edge_score" built-in: the DFlash2
// candidate-selector transition score. Given the predecessor and
// successor codebooks (each vocab x rank), the per-position hidden
// projection (batch x seq x rank), the candidate token ids
// (batch x seq x top_k), the anchor token ids (batch x seq) and the
// unary top-K logits (batch x seq x top_k) it returns the re-ranked
// scores (batch x seq x top_k x top_k):
//   pred_id(l,p) = anchor(l)        if l == 0
//                = candidate(l-1,p) otherwise
//   out(l,p,c) = unary(l,p)
//              + sum_r predecessor[pred_id(l,p),r] * hidden(l,r)
//                      * successor[candidate(l,c),r]
// Every size must match and top_k, rank and vocab must be non-zero; else
// InvalidArgument.
[[nodiscard]] std::expected<void, StatusCode> SelectorEdgeScoreRef(
    std::span<const float> predecessor_codebook,
    std::span<const float> successor_codebook, std::span<const float> hidden,
    std::span<const std::int32_t> candidate_ids,
    std::span<const std::int32_t> anchor_ids, std::span<const float> unary,
    std::span<float> out, std::size_t batch, std::size_t seq,
    std::size_t top_k, std::size_t rank, std::size_t vocab);

}  // namespace tessera::core
