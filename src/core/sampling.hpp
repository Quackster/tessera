#pragma once

#include <cstdint>
#include <random>
#include <span>

#include "tessera/engine.hpp"

namespace tessera::core {

// Draw one token from a logits row. Applies the penalties over `history`
// (each history token at most once for the presence penalty), then
// temperature, min_p, top_k and top_p, and samples. Deterministic for a
// given `rng` state. Returns the argmax when temperature is 0 or the
// filtered set is empty. `logits` must be nonempty.
[[nodiscard]] std::uint32_t SampleToken(
    std::span<const float> logits, const SamplingOptions& options,
    std::span<const std::uint32_t> history, std::mt19937_64& rng);

}  // namespace tessera::core
