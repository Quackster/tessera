#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "tessera/backend.hpp"
#include "tessera/engine.hpp"
#include "tessera/log.hpp"
#include "tessera/model.hpp"
#include "tessera/types.hpp"

namespace tessera::spec {

// Greedy generation with the DFlash2 draft. Each step captures the target
// hidden at the draft's target_layer_ids, drafts a block of top-1
// candidates with the draft block, verifies them against the target and
// accepts the matching prefix. The output equals plain greedy decoding.
// `stop_tokens` ends generation (and is not emitted) when the target's
// verified token is a member, so the draft path matches the reference
// path. UnsupportedFeature when the target is not hybrid or the draft does
// not load.
[[nodiscard]] std::expected<std::vector<std::uint32_t>, StatusCode>
GenerateDFlash2(Backend& backend, Model& target, const GenerateOptions& options,
                const std::string& draft_path,
                std::span<const std::uint32_t> stop_tokens,
                const log::Diagnostics* log = nullptr);

}  // namespace tessera::spec
