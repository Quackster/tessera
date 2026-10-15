#include "tessera/calibrate_sweep.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tessera/tokenizer.hpp"

namespace tessera {
namespace {

// Each base line repeats to this many sentences so the prompt exceeds the
// attention-split threshold (1024 keys) and the prefill chunk half of the
// candidate list, and the sweep can tell the chunks apart.
constexpr std::size_t kPromptRepeats = 72;
const char* const kPromptBases[] = {
    "Explain step by step how a large language model computes self "
    "attention over its context. ",
    "Describe the difference between the prefill and the decode phase of a "
    "transformer inference engine. ",
    "Summarize how a grouped query attention cache is stored and read back "
    "on an accelerator. ",
};

}  // namespace

std::vector<std::string> DefaultCalibrationPrompts() {
  std::vector<std::string> prompts;
  for (const char* base : kPromptBases) {
    std::string text;
    for (std::size_t i = 0; i < kPromptRepeats; ++i) {
      text += base;
    }
    prompts.push_back(std::move(text));
  }
  return prompts;
}

std::expected<std::vector<std::string>, bool> CalibrationPromptsFromFile(
    const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    return std::unexpected(false);
  }
  std::vector<std::string> prompts;
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      prompts.push_back(line);
    }
  }
  if (prompts.empty()) {
    return std::unexpected(false);
  }
  return prompts;
}

std::size_t CalibrationPrompts::longest() const {
  std::size_t longest = 0;
  for (const auto& ids : prefill) {
    longest = std::max(longest, ids.size());
  }
  return longest;
}

std::optional<CalibrationPrompts> EncodeCalibrationPrompts(
    const Tokenizer& tokenizer, const std::vector<std::string>& prompts) {
  CalibrationPrompts encoded;
  for (const std::string& prompt : prompts) {
    auto ids = tokenizer.Encode(prompt);
    if (!ids || ids->size() < 2) {
      return std::nullopt;
    }
    std::vector<std::uint32_t> short_ids = *ids;
    if (short_ids.size() > kDecodeMeasurePromptTokens) {
      short_ids.resize(kDecodeMeasurePromptTokens);
    }
    encoded.prefill.push_back(std::move(*ids));
    encoded.decode.push_back(std::move(short_ids));
  }
  return encoded;
}

MeasureFn BuildSweepMeasure(const TimedGenerateFn& generate,
                            const CalibrationPrompts& prompts,
                            KvCacheType kv_type) {
  const CalibrationPrompts* set = &prompts;
  return [generate, set, kv_type](Setting setting,
                                  const CalibrationConfig& config,
                                  std::size_t prompt,
                                  bool warmup) -> std::expected<Measurement,
                                                                StatusCode> {
    const bool prefill = setting == Setting::PrefillChunkTokens;
    // The attention split only applies at a long key range, so it measures
    // decode on the long prompts; the other decode settings use the short
    // ones.
    const bool long_context = prefill || setting == Setting::AttentionSplit;
    GenerateOptions gen;
    gen.max_completion_tokens =
        warmup ? 1 : (prefill ? 1 : kDecodeMeasureTokens);
    gen.sample = false;
    gen.progress_every = 0;
    gen.kv_type = kv_type;
    gen.prefill_chunk_tokens = config.prefill_chunk_tokens;
    gen.draft_tokens = config.draft_tokens;
    gen.mxfp4_split_target = config.mxfp4_split_target;
    gen.mxfp4_split_cap = config.mxfp4_split_cap;
    gen.attention_split = config.attention_split;
    gen.draft_context = config.draft_context;
    gen.prompt_tokens =
        long_context ? set->prefill[prompt] : set->decode[prompt];
    auto outcome = generate(gen.prompt_tokens, gen);
    if (!outcome) {
      return std::unexpected(outcome.error());
    }
    Measurement measured;
    if (outcome->produced > 0 && outcome->decode_ms > 0.0) {
      measured.decode_tps =
          static_cast<double>(outcome->produced) * 1000.0 / outcome->decode_ms;
    }
    if (outcome->prompt_tokens > 0 && outcome->prefill_ms > 0.0) {
      measured.prefill_tps = static_cast<double>(outcome->prompt_tokens) *
                             1000.0 / outcome->prefill_ms;
    }
    return measured;
  };
}

std::expected<std::pair<double, double>, StatusCode> MeasureChosenRates(
    const MeasureFn& measure, const CalibrationConfig& chosen,
    const CalibrationPrompts& prompts) {
  std::vector<double> decode_rates;
  std::vector<double> prefill_rates;
  for (std::size_t prompt = 0; prompt < prompts.count(); ++prompt) {
    auto decode = measure(Setting::MxFp4SplitTarget, chosen, prompt, false);
    if (!decode) {
      return std::unexpected(decode.error());
    }
    decode_rates.push_back(decode->decode_tps);
    auto prefill = measure(Setting::PrefillChunkTokens, chosen, prompt, false);
    if (!prefill) {
      return std::unexpected(prefill.error());
    }
    prefill_rates.push_back(prefill->prefill_tps);
  }
  return std::pair{MedianOf(decode_rates), MedianOf(prefill_rates)};
}

}  // namespace tessera
