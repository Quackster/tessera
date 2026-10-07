#pragma once

#include <memory>

#include "tessera/architecture.hpp"

namespace tessera {

// Build the Qwen3.5 module (hybrid gated attention interleaved with
// gated-delta linear attention, plus the nextn MTP draft head), from
// src/models/qwen3_5/.
[[nodiscard]] std::unique_ptr<Architecture> MakeQwen35Architecture();

}  // namespace tessera
