#pragma once

#include <cstddef>
#include <cstdlib>
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

// Match a diagnostic flag that maps to an environment override and set the
// environment, returning true when handled. `i` advances past the flag's
// value for the flags that take one. The library reads these environment
// variables, so the flag is the command-line form of the same diagnostic.
inline bool ApplyDiagnosticFlag(std::string_view arg, int& i, char** argv,
                                int argc) {
  const auto set = [](const char* name, const char* value) {
    setenv(name, value, 1);
  };
  if (arg == "--wmma") {
    set("TESSERA_MXFP4_WMMA", "1");
    return true;
  }
  if (arg == "--no-wmma") {
    set("TESSERA_MXFP4_WMMA", "0");
    return true;
  }
  if (arg == "--w4a8") {
    set("TESSERA_MXFP4_W4A8", "1");
    return true;
  }
  if (arg == "--target-bf16") {
    set("TESSERA_TARGET_BF16", "1");
    return true;
  }
  if ((arg == "--tiled-min-rows" || arg == "--tiled-min-cols" ||
       arg == "--prefill-attn-pairs") &&
      i + 1 < argc) {
    const char* value = argv[++i];
    const char* name = arg == "--tiled-min-rows"   ? "TESSERA_TILED_MIN_ROWS"
                       : arg == "--tiled-min-cols" ? "TESSERA_TILED_MIN_COLS"
                                                   : "TESSERA_PREFILL_ATTN_PAIRS";
    set(name, value);
    return true;
  }
  return false;
}

}  // namespace tessera::cli
