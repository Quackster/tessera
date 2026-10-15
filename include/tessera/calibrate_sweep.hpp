#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "tessera/calibrate.hpp"
#include "tessera/engine.hpp"
#include "tessera/types.hpp"

// Shared front end for the calibration speed sweep, used by both the
// single-device `calibrate` command and its parallel (`--gpus`) path: the
// fixed prompt set, the prompt encoder, and the measure callback that turns
// a timed generation into a sweep measurement. The sweep itself stays in
// the library (tessera/calibrate.hpp).

namespace tessera {

class Tokenizer;

// Decode tokens produced per decode measurement. Enough to average out
// launch jitter.
inline constexpr std::size_t kDecodeMeasureTokens = 32;
// A decode measurement runs on the prompt truncated to this many tokens:
// the decode rate does not depend on the prompt length, and prefilling a
// long prompt for every decode point would dominate the run.
inline constexpr std::size_t kDecodeMeasurePromptTokens = 64;

// The fixed default prompt set: three base lines repeated long enough that
// the prompt exceeds the attention-split threshold (1024 keys) and the
// prefill chunk candidates, so the sweep can tell them apart.
[[nodiscard]] std::vector<std::string> DefaultCalibrationPrompts();

// Read prompts from a file, one per non-empty line. false when the file is
// missing or holds no prompt line.
[[nodiscard]] std::expected<std::vector<std::string>, bool>
CalibrationPromptsFromFile(const std::string& path);

// The encoded prompt set: the full prompts for the prefill and
// long-context settings, each truncated to kDecodeMeasurePromptTokens for
// the decode settings. All settings index the same prompt list.
struct CalibrationPrompts {
  std::vector<std::vector<std::uint32_t>> prefill;
  std::vector<std::vector<std::uint32_t>> decode;
  [[nodiscard]] std::size_t count() const { return prefill.size(); }
  // Longest full prompt, the key length the attention split needs.
  [[nodiscard]] std::size_t longest() const;
};

// Encode `prompts`; nullopt when a prompt encodes to fewer than two tokens
// (a prompt that cannot drive generation is not a usable sweep point).
[[nodiscard]] std::optional<CalibrationPrompts> EncodeCalibrationPrompts(
    const Tokenizer& tokenizer, const std::vector<std::string>& prompts);

// A timed greedy generation on one route: the prompt tokens and the speed
// tuning in, the outcome (produced count, prefill/decode times) out. The
// single-device engine and each parallel layout supply one.
using TimedGenerateFn =
    std::function<std::expected<GenerateOutcome, StatusCode>(
        std::span<const std::uint32_t>, const GenerateOptions&)>;

// Build the sweep measure callback for a route. A prefill-chunk or
// attention-split setting measures the long prompts and reports prefill
// tps; every other setting measures the short prompts and reports decode
// tps. A warmup request produces one token and is discarded by the sweep.
// `prompts` must outlive the returned callback.
[[nodiscard]] MeasureFn BuildSweepMeasure(const TimedGenerateFn& generate,
                                          const CalibrationPrompts& prompts,
                                          KvCacheType kv_type);

// Measure the chosen configuration once more: the median decode and
// prefill rates over the prompts, for the rates the entry stores.
[[nodiscard]] std::expected<std::pair<double, double>, StatusCode>
MeasureChosenRates(const MeasureFn& measure, const CalibrationConfig& chosen,
                   const CalibrationPrompts& prompts);

}  // namespace tessera
