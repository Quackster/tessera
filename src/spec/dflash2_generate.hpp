#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "tessera/backend.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/types.hpp"

namespace tessera::spec {

// Greedy generation with the DFlash2 draft. Each step captures the target
// hidden at the draft's target_layer_ids, drafts a block of top-1
// candidates with the draft block, verifies them against the target and
// accepts the matching prefix. The output equals plain greedy decoding.
// UnsupportedFeature when the target is not hybrid or the draft does not
// load.
[[nodiscard]] std::expected<std::vector<std::uint32_t>, StatusCode>
GenerateDFlash2(Backend& backend, Model& target, const GenerateOptions& options,
                const std::string& draft_path);

}  // namespace tessera::spec
