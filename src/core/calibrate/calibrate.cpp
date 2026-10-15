#include "tessera/calibrate.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

// The calibration sweep: the candidate lists, the pick rule, the
// three-consecutive-degradation early stop, and the KV sizing helpers. One
// canonical implementation of each (AGENTS.md rules 2 and 3). The
// measurement itself is the MeasureFn the caller supplies, so the sweep
// runs unchanged against a real engine or a stand-in. The JSON file lives
// in file.cpp and the report text in report.cpp.

namespace tessera {

namespace {

// Abort one setting's sweep once this many candidates in a row fail to beat
// the best value measured so far: the trend is down, so the remaining
// candidates would only cost measurements. The sweep then moves to the next
// setting.
constexpr std::size_t kMaxConsecutiveDegradations = 3;

// Apply one setting value onto a configuration.
void ApplySetting(CalibrationConfig& config, Setting setting,
                  std::size_t value) {
  switch (setting) {
    case Setting::MxFp4SplitTarget:
      config.mxfp4_split_target = value;
      break;
    case Setting::MxFp4SplitCap:
      config.mxfp4_split_cap = value;
      break;
    case Setting::PrefillChunkTokens:
      config.prefill_chunk_tokens = value;
      break;
    case Setting::AttentionSplit:
      config.attention_split = value;
      break;
    case Setting::DraftTokens:
      config.draft_tokens = value;
      break;
    case Setting::DraftContext:
      config.draft_context = value;
      break;
  }
}

// Median rate over the fixed prompts, for a decode or prefill setting.
std::expected<double, StatusCode> MeasureMedian(const MeasureFn& measure,
                                                 Setting setting,
                                                 const CalibrationConfig& config,
                                                 std::size_t prompt_count,
                                                 bool decode) {
  std::vector<double> rates;
  rates.reserve(prompt_count);
  for (std::size_t prompt = 0; prompt < prompt_count; ++prompt) {
    auto measured = measure(setting, config, prompt, false);
    if (!measured) {
      return std::unexpected(measured.error());
    }
    rates.push_back(decode ? measured->decode_tps : measured->prefill_tps);
  }
  return MedianOf(rates);
}

// Warm a configuration once (the result is discarded), then measure its
// median rate.
std::expected<double, StatusCode> MeasurePoint(const MeasureFn& measure,
                                               Setting setting,
                                               const CalibrationConfig& config,
                                               std::size_t prompt_count,
                                               bool decode) {
  auto warm = measure(setting, config, 0, true);
  if (!warm) {
    return std::unexpected(warm.error());
  }
  return MeasureMedian(measure, setting, config, prompt_count, decode);
}

// Sweep one setting: candidates, pick, interleaved confirmation. The other
// settings stay at their defaults while this one is probed.
std::expected<void, StatusCode> SweepSetting(
    const SweepOptions& options, const MeasureFn& measure, Setting setting,
    bool enabled, std::size_t default_value, bool decode,
    SweepOutcome& outcome) {
  if (!enabled) {
    outcome.not_applicable.push_back(setting);
    return {};
  }
  const std::vector<std::size_t> candidates =
      CandidateValues(setting, default_value);
  if (candidates.empty()) {
    outcome.not_applicable.push_back(setting);
    return {};
  }
  // Measure candidates one after another until the point budget runs out.
  // The budget counts every measured point in total, whatever the setting.
  // Budget mode measures each candidate exactly once (first prompt, no
  // warmup), so the passed count is the number of engine runs.
  const bool budgeted = options.max_points > 0;
  std::vector<std::size_t> measured;
  std::vector<double> rates;
  measured.reserve(candidates.size());
  rates.reserve(candidates.size());
  const std::size_t base = outcome.points.size();
  double best_rate = -1.0;
  std::size_t degradations = 0;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (budgeted && outcome.points.size() >= options.max_points) {
      break;
    }
    CalibrationConfig probe = options.defaults;
    ApplySetting(probe, setting, candidates[i]);
    double tps = 0.0;
    if (budgeted) {
      auto measured_once = measure(setting, probe, 0, false);
      if (!measured_once) {
        return std::unexpected(measured_once.error());
      }
      tps = decode ? measured_once->decode_tps : measured_once->prefill_tps;
    } else {
      auto rate =
          MeasurePoint(measure, setting, probe, options.prompt_count, decode);
      if (!rate) {
        return std::unexpected(rate.error());
      }
      tps = *rate;
    }
    measured.push_back(candidates[i]);
    rates.push_back(tps);
    outcome.points.push_back(SweepPoint{setting, candidates[i], tps, false});
    // Stop early on a downward trend. A candidate at or above the best so
    // far resets the streak; a tie is not a degradation.
    if (tps > best_rate) {
      best_rate = tps;
      degradations = 0;
    } else if (tps < best_rate) {
      if (++degradations >= kMaxConsecutiveDegradations) {
        break;
      }
    } else {
      degradations = 0;
    }
  }
  if (measured.empty()) {
    outcome.skipped.push_back(setting);
    return {};
  }
  std::size_t chosen = default_value;
  if (budgeted) {
    // Budget mode keeps the fastest measured value as is: no gain gate
    // and no interleaved confirmation, both cost extra measurements.
    std::size_t best = 0;
    for (std::size_t i = 1; i < measured.size(); ++i) {
      if (rates[i] > rates[best]) {
        best = i;
      }
    }
    chosen = measured[best];
    ApplySetting(outcome.config, setting, chosen);
    for (std::size_t i = 0; i < measured.size(); ++i) {
      outcome.points[base + i].kept = (measured[i] == chosen);
    }
    return {};
  }
  const std::optional<std::size_t> best =
      PickBest(measured, default_value, rates);
  if (best.has_value() && *best != default_value) {
    CalibrationConfig winner = options.defaults;
    ApplySetting(winner, setting, *best);
    CalibrationConfig floor = options.defaults;
    ApplySetting(floor, setting, default_value);
    std::vector<double> winner_rates;
    std::vector<double> floor_rates;
    for (int round = 0; round < 3; ++round) {
      auto winner_rate = MeasurePoint(measure, setting, winner,
                                      options.prompt_count, decode);
      if (!winner_rate) {
        return std::unexpected(winner_rate.error());
      }
      winner_rates.push_back(*winner_rate);
      auto floor_rate =
          MeasurePoint(measure, setting, floor, options.prompt_count, decode);
      if (!floor_rate) {
        return std::unexpected(floor_rate.error());
      }
      floor_rates.push_back(*floor_rate);
    }
    const double winner_median = MedianOf(winner_rates);
    const double floor_median = MedianOf(floor_rates);
    const double gain = floor_median > 0.0
                            ? (winner_median - floor_median) / floor_median
                            : 0.0;
    if (gain > kMinGain) {
      chosen = *best;
    }
  }
  ApplySetting(outcome.config, setting, chosen);
  for (std::size_t i = 0; i < measured.size(); ++i) {
    outcome.points[base + i].kept = (measured[i] == chosen);
  }
  return {};
}

}  // namespace

std::string_view ToString(Setting setting) {
  switch (setting) {
    case Setting::MxFp4SplitTarget:
      return "mxfp4_split_target";
    case Setting::MxFp4SplitCap:
      return "mxfp4_split_cap";
    case Setting::PrefillChunkTokens:
      return "prefill_chunk_tokens";
    case Setting::AttentionSplit:
      return "attention_split";
    case Setting::DraftTokens:
      return "draft_tokens";
    case Setting::DraftContext:
      return "draft_context";
  }
  return "unknown";
}

std::vector<std::size_t> CandidateValues(Setting setting,
                                         std::size_t default_value,
                                         std::size_t upper_bound) {
  std::vector<std::size_t> values;
  switch (setting) {
    case Setting::MxFp4SplitTarget:
      values = {120, 320, 640};
      break;
    case Setting::MxFp4SplitCap:
      values = {2, 4, 8};
      break;
    case Setting::PrefillChunkTokens:
      // Ascending multiples of 256 from the 512 default: a larger chunk
      // issues fewer launches, a smaller one bounds memory. The sweep keeps
      // the default unless a larger chunk measurably wins.
      values = {512, 768, 1024, 1536, 2048};
      break;
    case Setting::AttentionSplit:
      // Flash-decoding chunks for the single-token long-context path.
      values = {4, 8, 16, 32};
      break;
    case Setting::DraftTokens:
      // Small draft blocks. The checkpoint value is the default candidate:
      // --draft-block 0 leaves the block at the checkpoint's own size.
      values = {1, 2, 3, 4};
      break;
    case Setting::DraftContext:
      // Draft context window in rows; the checkpoint value is the default.
      values = {512, 1024, 2048};
      break;
  }
  if (default_value > 0) {
    values.push_back(default_value);
  }
  values.erase(std::remove(values.begin(), values.end(), std::size_t{0}),
               values.end());
  if (upper_bound > 0) {
    values.erase(std::remove_if(values.begin(), values.end(),
                                [upper_bound](std::size_t value) {
                                  return value > upper_bound;
                                }),
                 values.end());
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

double MedianOf(std::span<const double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::vector<double> sorted(values.begin(), values.end());
  std::sort(sorted.begin(), sorted.end());
  const std::size_t count = sorted.size();
  if (count % 2 == 1) {
    return sorted[count / 2];
  }
  return 0.5 * (sorted[count / 2 - 1] + sorted[count / 2]);
}

std::uint64_t KvBytesPerToken(const TransformerConfig& config,
                              KvCacheType kv_type) {
  if (config.layers == 0 || config.attention.kv_heads == 0 ||
      config.attention.head_dim == 0) {
    return 0;
  }
  std::size_t full_layers = 0;
  for (std::size_t layer = 0; layer < config.layers; ++layer) {
    if (config.IsFullAttentionLayer(layer)) {
      ++full_layers;
    }
  }
  if (full_layers == 0) {
    return 0;
  }
  // One key and one value per (full layer, kv head, head dim).
  const std::uint64_t elements =
      static_cast<std::uint64_t>(full_layers) * config.attention.kv_heads *
      config.attention.head_dim * 2;
  std::uint64_t numerator = 0;
  std::uint64_t denominator = 1;
  switch (kv_type) {
    case KvCacheType::F32:
      numerator = 4;
      break;
    case KvCacheType::F16:
      numerator = 2;
      break;
    case KvCacheType::Q8:
    case KvCacheType::FP8:
      numerator = 1;
      break;
    case KvCacheType::Q4:
      numerator = 1;
      denominator = 2;
      break;
  }
  return elements * numerator / denominator;
}

std::size_t MaxContextForKv(std::uint64_t available_bytes,
                            const TransformerConfig& config,
                            KvCacheType kv_type) {
  const std::uint64_t per_token = KvBytesPerToken(config, kv_type);
  if (per_token == 0) {
    return 0;
  }
  return static_cast<std::size_t>(available_bytes / per_token);
}

std::optional<std::size_t> PickBest(std::span<const std::size_t> candidates,
                                    std::size_t default_value,
                                    std::span<const double> rates) {
  if (candidates.empty() || candidates.size() != rates.size()) {
    return std::nullopt;
  }
  std::size_t best = 0;
  for (std::size_t i = 1; i < candidates.size(); ++i) {
    if (rates[i] > rates[best]) {
      best = i;
    }
  }
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i] != default_value) {
      continue;
    }
    if (rates[i] <= 0.0) {
      return default_value;
    }
    const double gain = (rates[best] - rates[i]) / rates[i];
    return gain > kMinGain ? candidates[best] : default_value;
  }
  return candidates[best];
}

std::expected<SweepOutcome, StatusCode> RunCalibrationSweep(
    const SweepOptions& options, const MeasureFn& measure) {
  if (!measure || options.prompt_count == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  SweepOutcome outcome;
  outcome.config = options.defaults;
  if (auto status = SweepSetting(options, measure, Setting::MxFp4SplitTarget,
                                 options.sweep_split_target,
                                 options.defaults.mxfp4_split_target,
                                 /*decode=*/true, outcome);
      !status) {
    return std::unexpected(status.error());
  }
  if (auto status = SweepSetting(options, measure, Setting::MxFp4SplitCap,
                                 options.sweep_split_cap,
                                 options.defaults.mxfp4_split_cap,
                                 /*decode=*/true, outcome);
      !status) {
    return std::unexpected(status.error());
  }
  if (auto status = SweepSetting(options, measure, Setting::PrefillChunkTokens,
                                 options.sweep_prefill_chunk,
                                 options.defaults.prefill_chunk_tokens,
                                 /*decode=*/false, outcome);
      !status) {
    return std::unexpected(status.error());
  }
  if (auto status = SweepSetting(options, measure, Setting::AttentionSplit,
                                 options.sweep_attention_split,
                                 options.defaults.attention_split,
                                 /*decode=*/true, outcome);
      !status) {
    return std::unexpected(status.error());
  }
  if (auto status = SweepSetting(options, measure, Setting::DraftTokens,
                                 options.sweep_draft_tokens,
                                 options.defaults.draft_tokens,
                                 /*decode=*/true, outcome);
      !status) {
    return std::unexpected(status.error());
  }
  if (auto status = SweepSetting(options, measure, Setting::DraftContext,
                                 options.sweep_draft_context,
                                 options.defaults.draft_context,
                                 /*decode=*/true, outcome);
      !status) {
    return std::unexpected(status.error());
  }
  return outcome;
}

}  // namespace tessera
