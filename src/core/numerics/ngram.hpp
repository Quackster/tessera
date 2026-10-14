#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "tessera/types.hpp"

// The PLE n-gram hash: sixteen PLE-table row indices per token, computed
// on the host from the last `ngram_size` token ids with 64-bit multiply
// and xor. XOR (not sum), modulo (not mask), a forward-propagating
// end-of-text cut, and a missing predecessor distinct from token 0 are
// the parts a rival reading gets plausibly wrong. Ported from
// ../Strata include/strata/kernels/ngram.hpp.

namespace tessera::core {

// `mixed = ctx[0]*mult[0] ^ ctx[1]*mult[1] ^ ...` in u64. Exposed so a
// test can assert the XOR structure directly.
[[nodiscard]] std::uint64_t NgramMixed(std::span<const std::int64_t> context,
                                       std::span<const std::uint64_t> mult);

// The row indices into the PLE table. `tokens` is n_tokens; `prev` is
// n_tokens * (ngram_size - 1) predecessors, OLDEST FIRST, with -1 where
// there is none; `out` receives n_tokens * heads indices where heads is
// (ngram_size - 1) * heads_per_ngram. A token equal to the configured
// end-of-text id cuts the context forward. A size or parameter mismatch
// is InvalidArgument.
//
// Usage:
//   NgramRows(tokens, prev, config.ple, rows);
[[nodiscard]] std::expected<void, StatusCode> NgramRows(
    std::span<const std::int32_t> tokens, std::span<const std::int32_t> prev,
    const PleParams& ple, std::span<std::uint32_t> out);

}  // namespace tessera::core
