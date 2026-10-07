#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "tessera/types.hpp"

namespace tessera::spec {

// Extract the top-K draft candidates per row from the draft logits. For
// every row the K largest logits are written to `ids` and their values to
// `unary`, both rows x top_k, in descending order. `logits` is rows x
// vocab. top_k must be at most vocab; every size must match; else
// InvalidArgument.
[[nodiscard]] std::expected<void, StatusCode> DraftCandidates(
    std::span<const float> logits, std::span<std::uint32_t> ids,
    std::span<float> unary, std::size_t rows, std::size_t vocab,
    std::size_t top_k);

}  // namespace tessera::spec
