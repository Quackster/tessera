#pragma once

#include <cstddef>
#include <cstdint>

// Single source of truth for the default values the engine and the
// front-end tools share (AGENTS.md rule 2: one canonical definition). An
// omitted run, serve or calibrate flag falls back to the constant here,
// and every option struct default and usage string reads it, so a default
// cannot drift between the library and a tool. This header depends on
// nothing else in tessera.

namespace tessera {

// --- Context and generation shape ---

// Default maximum context length (the KV context size) for run, serve and
// calibrate when `--context` is not given.
constexpr std::size_t kDefaultContext = 32768;

// Default draft block tokens; 0 keeps the draft checkpoint's configured
// block size. A block-diffusion drafter is trained for one fixed block, so
// a mismatched block silently lowers acceptance.
constexpr std::size_t kDefaultDraftBlock = 0;

// Default prefill chunk sentinel; 0 resolves automatically to
// kDefaultPrefillChunkTokens.
constexpr std::size_t kDefaultPrefillChunk = 0;

// Tokens per prefill forward when neither the engine nor the request sets
// one (llama.cpp ubatch analog). Bounds one forward's scratch so a long
// prompt prefills in bounded memory instead of losing the device.
constexpr std::size_t kDefaultPrefillChunkTokens = 512;

// --- Sampling ---
// Defaults for SamplingOptions, matching the Qwen 3.8 27B model card.

constexpr float kDefaultTemperature = 0.6f;
constexpr float kDefaultTopP = 0.95f;
constexpr int kDefaultTopK = 20;
constexpr float kDefaultMinP = 0.0f;
constexpr float kDefaultPresencePenalty = 0.0f;
constexpr float kDefaultRepetitionPenalty = 1.0f;

// --- Serve ---

// Default bind address for `tessera-cli serve`.
constexpr const char* kDefaultServeHost = "127.0.0.1";

// Default bind port for `tessera-cli serve`.
constexpr std::uint16_t kDefaultServePort = 8080;

// Zero default completion length: a request that omits
// max_completion_tokens fills the remaining context (see
// Model::EffectiveMaxTokens) instead of stopping after a fixed budget.
constexpr std::size_t kDefaultMaxCompletionTokens = 0;

// Default live speed cadence: streamed turns report tok/s to the session
// stream and the server log every this many seconds. Zero disables both;
// --stats-interval overrides it per serve run.
constexpr std::size_t kDefaultStatsIntervalSeconds = 5;

// --- Scheduler ---

// Bound on admitted jobs when SchedulerOptions leaves it zero.
constexpr std::size_t kDefaultMaxConcurrent = 4;

// --- Kernel tuning ---

// Built-in split-K workgroup target and cap for the fp8 tensor-core MXFP4
// GEMM (the ROCm-only WMMA path). A request that leaves
// GenerateOptions::mxfp4_split_target or ::mxfp4_split_cap at 0 selects
// these.
constexpr std::size_t kDefaultMxFp4SplitTarget = 320;
constexpr std::size_t kDefaultMxFp4SplitCap = 4;

// Built-in split count for the single-token (m=1) flash-decoding
// attention path at long context (>= 1024 keys). A request that leaves
// GenerateOptions::attention_split at 0 selects it.
constexpr std::size_t kDefaultAttentionSplitChunks = 16;

// Built-in tiled-GEMM dispatch thresholds and the prefill attention pair
// budget (core::detail). Exposed so a diagnostic override can be named and
// reported; the environment variables TESSERA_TILED_MIN_ROWS,
// TESSERA_TILED_MIN_COLS and TESSERA_PREFILL_ATTN_PAIRS win over them.
constexpr std::size_t kDefaultTiledMinRows = 16;
constexpr std::size_t kDefaultTiledMinCols = 65536;
constexpr std::uint64_t kDefaultPrefillAttnPairs = 4ull * 1024 * 1024;

}  // namespace tessera
