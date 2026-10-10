#pragma once

#include <cstddef>
#include <string_view>

#include "tessera/types.hpp"

// Small shared helpers for the tessera-cli front end (AGENTS.md rule 2:
// one canonical implementation of the flag mapping). The heavy logic
// stays in the library.

namespace tessera::cli {

// Default maximum context length (a tunable) for run, serve and calibrate.
constexpr std::size_t kDefaultContext = 4096;

// 0 keeps the draft checkpoint's configured block size. A block-diffusion
// drafter is trained for one fixed block; a mismatched block silently
// lowers acceptance (the DFlash2 checkpoint is trained for block 8).
constexpr std::size_t kDefaultDraftBlock = 0;

// 0 resolves automatically to kDefaultPrefillChunkTokens (512): long
// prompts prefill in chunk-sized forwards.
constexpr std::size_t kDefaultPrefillChunk = 0;

// Match a `--kv-*` flag and set `type`. Returns false when `arg` is not a
// KV-cache flag, so callers fall through to the next case.
inline bool MatchKvType(std::string_view arg, KvCacheType& type) {
  if (arg == "--kv-f16") {
    type = KvCacheType::F16;
    return true;
  }
  if (arg == "--kv-q8") {
    type = KvCacheType::Q8;
    return true;
  }
  if (arg == "--kv-q4") {
    type = KvCacheType::Q4;
    return true;
  }
  if (arg == "--kv-fp8") {
    type = KvCacheType::FP8;
    return true;
  }
  return false;
}

}  // namespace tessera::cli
