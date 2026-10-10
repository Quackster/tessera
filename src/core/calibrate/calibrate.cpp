#include "tessera/calibrate.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <utility>

#include "core/files.hpp"
#include "core/json.hpp"
#include "tessera/engine.hpp"

// The calibration module: candidate lists, the pick rule, the sweep, and
// the hardware key. One canonical implementation of each (AGENTS.md rules
// 2 and 3). The measurement itself is the MeasureFn the caller supplies, so
// the sweep runs unchanged against a real engine or a stand-in.

namespace tessera {

namespace {

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
  std::vector<double> rates(candidates.size(), 0.0);
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    CalibrationConfig probe = options.defaults;
    ApplySetting(probe, setting, candidates[i]);
    auto rate =
        MeasurePoint(measure, setting, probe, options.prompt_count, decode);
    if (!rate) {
      return std::unexpected(rate.error());
    }
    rates[i] = *rate;
  }
  std::size_t chosen = default_value;
  const std::optional<std::size_t> best =
      PickBest(candidates, default_value, rates);
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
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    outcome.points.push_back(SweepPoint{setting, candidates[i], rates[i],
                                        candidates[i] == chosen});
  }
  return {};
}

// JSON helpers: a missing or wrong-typed field leaves the output untouched
// and returns false.
bool ReadString(const core::Json& object, std::string_view key,
                std::string& out) {
  const core::Json* field = object.Find(key);
  if (field == nullptr || !field->isString()) {
    return false;
  }
  out = field->AsString();
  return true;
}

bool ReadNumber(const core::Json& object, std::string_view key, double& out) {
  const core::Json* field = object.Find(key);
  if (field == nullptr || field->type() != core::Json::Type::Number) {
    return false;
  }
  out = field->AsNumber();
  return true;
}

// Optional numeric field: absent leaves `out` unchanged. Used for fields
// added after the first calibration file version, so a file without them
// still loads.
bool ReadOptionalNumber(const core::Json& object, std::string_view key,
                        double& out) {
  const core::Json* field = object.Find(key);
  if (field == nullptr) {
    return true;
  }
  if (field->type() != core::Json::Type::Number) {
    return false;
  }
  out = field->AsNumber();
  return true;
}

// Parse one entry object. Any missing or wrong-typed field is
// MalformedFile, so a truncated file is rejected instead of half-applied.
std::expected<CalibrationEntry, StatusCode> ParseEntry(const core::Json& item) {
  if (!item.isObject()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  CalibrationEntry entry;
  double context = 0.0;
  double split = 0.0;
  double cap = 0.0;
  double prefill = 0.0;
  double attention = 0.0;
  double draft = 0.0;
  double draft_context = 0.0;
  if (!ReadString(item, "backend", entry.key.backend) ||
      !ReadString(item, "device", entry.key.device) ||
      !ReadString(item, "model", entry.key.model) ||
      !ReadNumber(item, "context", context) ||
      !ReadString(item, "kv_type", entry.key.kv_type) ||
      !ReadString(item, "strategy", entry.key.strategy) ||
      !ReadNumber(item, "mxfp4_split_target", split) ||
      !ReadOptionalNumber(item, "mxfp4_split_cap", cap) ||
      !ReadNumber(item, "prefill_chunk_tokens", prefill) ||
      !ReadOptionalNumber(item, "attention_split", attention) ||
      !ReadNumber(item, "draft_tokens", draft) ||
      !ReadOptionalNumber(item, "draft_context", draft_context) ||
      !ReadNumber(item, "decode_tps", entry.decode_tps) ||
      !ReadNumber(item, "prefill_tps", entry.prefill_tps) ||
      !ReadString(item, "date", entry.date)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (context < 0.0 || split < 0.0 || cap < 0.0 || prefill < 0.0 ||
      attention < 0.0 || draft < 0.0 || draft_context < 0.0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  entry.key.context = static_cast<std::size_t>(context);
  entry.config.mxfp4_split_target = static_cast<std::size_t>(split);
  entry.config.mxfp4_split_cap = static_cast<std::size_t>(cap);
  entry.config.prefill_chunk_tokens = static_cast<std::size_t>(prefill);
  entry.config.attention_split = static_cast<std::size_t>(attention);
  entry.config.draft_tokens = static_cast<std::size_t>(draft);
  entry.config.draft_context = static_cast<std::size_t>(draft_context);
  return entry;
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
      values = {4, 8};
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

std::size_t MaxContextForKv(std::uint64_t available_bytes,
                            const TransformerConfig& config,
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
  const std::uint64_t per_token = elements * numerator / denominator;
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

std::string HardwareKey::ToString() const {
  return backend + "|" + device + "|" + model + "|" + std::to_string(context) +
         "|" + kv_type + "|" + strategy;
}

std::expected<CalibrationFile, StatusCode> CalibrationFile::Load(
    std::string_view path) {
  auto bytes = core::ReadFile(std::filesystem::path(path));
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  const std::string text(reinterpret_cast<const char*>(bytes->data()),
                         bytes->size());
  auto document = core::Json::Parse(text);
  if (!document || !document->isObject()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const core::Json* entries = document->Find("entries");
  if (entries == nullptr || !entries->isArray()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  CalibrationFile file;
  for (const core::Json& item : entries->AsArray()) {
    auto entry = ParseEntry(item);
    if (!entry) {
      return std::unexpected(entry.error());
    }
    file.Set(std::move(*entry));
  }
  return file;
}

std::expected<void, StatusCode> CalibrationFile::Save(
    std::string_view path) const {
  core::Json root = core::Json::Object();
  root.Set("version", core::Json::Number(1));
  core::Json entries = core::Json::Array();
  for (const CalibrationEntry& entry : entries_) {
    core::Json item = core::Json::Object();
    item.Set("backend", core::Json::String(entry.key.backend));
    item.Set("device", core::Json::String(entry.key.device));
    item.Set("model", core::Json::String(entry.key.model));
    item.Set("context", core::Json::Number(
                            static_cast<double>(entry.key.context)));
    item.Set("kv_type", core::Json::String(entry.key.kv_type));
    item.Set("strategy", core::Json::String(entry.key.strategy));
    item.Set("mxfp4_split_target",
             core::Json::Number(
                 static_cast<double>(entry.config.mxfp4_split_target)));
    item.Set("mxfp4_split_cap",
             core::Json::Number(
                 static_cast<double>(entry.config.mxfp4_split_cap)));
    item.Set("prefill_chunk_tokens",
             core::Json::Number(
                 static_cast<double>(entry.config.prefill_chunk_tokens)));
    item.Set("attention_split",
             core::Json::Number(
                 static_cast<double>(entry.config.attention_split)));
    item.Set("draft_tokens", core::Json::Number(
                                 static_cast<double>(entry.config.draft_tokens)));
    item.Set("draft_context",
             core::Json::Number(
                 static_cast<double>(entry.config.draft_context)));
    item.Set("decode_tps", core::Json::Number(entry.decode_tps));
    item.Set("prefill_tps", core::Json::Number(entry.prefill_tps));
    item.Set("date", core::Json::String(entry.date));
    entries.Push(std::move(item));
  }
  root.Set("entries", std::move(entries));
  const std::string dumped = root.Dump();
  const auto bytes = std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(dumped.data()), dumped.size());
  return core::WriteFile(std::filesystem::path(path), bytes);
}

const CalibrationEntry* CalibrationFile::Find(const HardwareKey& key) const {
  for (const CalibrationEntry& entry : entries_) {
    if (entry.key == key) {
      return &entry;
    }
  }
  return nullptr;
}

void CalibrationFile::Set(CalibrationEntry entry) {
  for (CalibrationEntry& existing : entries_) {
    if (existing.key == entry.key) {
      existing = std::move(entry);
      return;
    }
  }
  entries_.push_back(std::move(entry));
}

std::span<const CalibrationEntry> CalibrationFile::Entries() const {
  return entries_;
}

HardwareKey MakeHardwareKey(std::string_view backend, std::string_view device,
                            std::string_view model, std::size_t context,
                            std::string_view kv_type,
                            std::string_view strategy) {
  HardwareKey key;
  key.backend = std::string(backend);
  key.device = std::string(device);
  key.model = std::string(model);
  key.context = context;
  key.kv_type = std::string(kv_type);
  key.strategy = std::string(strategy);
  return key;
}

CalibrationConfig ApplySavedCalibration(
    const CalibrationConfig& explicit_config, const CalibrationEntry& saved) {
  CalibrationConfig merged = explicit_config;
  if (merged.mxfp4_split_target == 0) {
    merged.mxfp4_split_target = saved.config.mxfp4_split_target;
  }
  if (merged.mxfp4_split_cap == 0) {
    merged.mxfp4_split_cap = saved.config.mxfp4_split_cap;
  }
  if (merged.prefill_chunk_tokens == 0) {
    merged.prefill_chunk_tokens = saved.config.prefill_chunk_tokens;
  }
  if (merged.attention_split == 0) {
    merged.attention_split = saved.config.attention_split;
  }
  if (merged.draft_tokens == 0) {
    merged.draft_tokens = saved.config.draft_tokens;
  }
  if (merged.draft_context == 0) {
    merged.draft_context = saved.config.draft_context;
  }
  return merged;
}

void ApplyToGenerateOptions(const CalibrationConfig& config,
                            GenerateOptions& options) {
  if (config.mxfp4_split_target != 0) {
    options.mxfp4_split_target = config.mxfp4_split_target;
  }
  if (config.mxfp4_split_cap != 0) {
    options.mxfp4_split_cap = config.mxfp4_split_cap;
  }
  if (config.prefill_chunk_tokens != 0) {
    options.prefill_chunk_tokens = config.prefill_chunk_tokens;
  }
  if (config.attention_split != 0) {
    options.attention_split = config.attention_split;
  }
  if (config.draft_tokens != 0) {
    options.draft_tokens = config.draft_tokens;
  }
  if (config.draft_context != 0) {
    options.draft_context = config.draft_context;
  }
}

std::string FormatCalibrationReport(const CalibrationReport& report) {
  const auto row = [](const std::string& label, const std::string& value,
                      std::size_t width) {
    std::string line = label;
    if (line.size() < width) {
      line.append(width - line.size(), ' ');
    }
    return line + value + "\n";
  };
  const auto default_for = [&](Setting setting) -> std::size_t {
    switch (setting) {
      case Setting::MxFp4SplitTarget:
        return report.defaults.mxfp4_split_target;
      case Setting::MxFp4SplitCap:
        return report.defaults.mxfp4_split_cap;
      case Setting::PrefillChunkTokens:
        return report.defaults.prefill_chunk_tokens;
      case Setting::AttentionSplit:
        return report.defaults.attention_split;
      case Setting::DraftTokens:
        return report.defaults.draft_tokens;
      case Setting::DraftContext:
        return report.defaults.draft_context;
    }
    return 0;
  };
  char rate[32];
  const auto format_rate = [&](double value) {
    std::snprintf(rate, sizeof(rate), "%.2f", value);
    return std::string(rate);
  };

  std::string out;
  out += "Tessera calibration report\n";
  out += "==========================\n";
  out += row("backend", report.backend, 14);
  out += row("device", report.device, 14);
  out += row("model", report.model, 14);
  out += row("kv cache", report.kv_type, 14);
  out += row("strategy", report.strategy, 14);
  out += row("key context", std::to_string(report.key_context), 14);
  out += row("max context", std::to_string(report.max_context) + " token(s), " +
                                report.kv_type + " memory bound",
             14);
  out += "\nSweep\n-----\n";
  for (Setting setting :
       {Setting::MxFp4SplitTarget, Setting::MxFp4SplitCap,
        Setting::PrefillChunkTokens, Setting::AttentionSplit,
        Setting::DraftTokens, Setting::DraftContext}) {
    out += std::string(ToString(setting)) + "\n";
    bool any = false;
    for (const SweepPoint& point : report.points) {
      if (point.setting != setting) {
        continue;
      }
      any = true;
      std::string note;
      if (point.value == default_for(setting)) {
        note += " (default)";
      }
      if (point.kept) {
        note += " (kept)";
      }
      out += row("  " + std::to_string(point.value),
                 format_rate(point.tps) + " tok/s" + note, 8);
    }
    if (!any) {
      out += "  not applicable on " + report.backend + "\n";
    }
  }
  out += "\nChosen\n------\n";
  out += row("mxfp4_split_target",
             std::to_string(report.chosen.mxfp4_split_target), 21);
  out += row("mxfp4_split_cap",
             std::to_string(report.chosen.mxfp4_split_cap), 21);
  out += row("prefill_chunk_tokens",
             std::to_string(report.chosen.prefill_chunk_tokens), 21);
  out += row("attention_split",
             std::to_string(report.chosen.attention_split), 21);
  out += row("draft_tokens",
             report.chosen.draft_tokens == 0
                 ? std::string("(none)")
                 : std::to_string(report.chosen.draft_tokens),
             21);
  out += row("draft_context",
             report.chosen.draft_context == 0
                 ? std::string("(checkpoint)")
                 : std::to_string(report.chosen.draft_context),
             21);
  out += row("decode", format_rate(report.decode_tps) + " tok/s", 21);
  out += row("prefill", format_rate(report.prefill_tps) + " tok/s", 21);
  out += "\nWrote " + report.file + "\n";
  std::string example =
      "  tessera-cli run --model " + report.model_path + " " + report.kv_flag;
  example += "\\\n      --prefill-chunk " +
             std::to_string(report.chosen.prefill_chunk_tokens);
  if (report.split_applicable) {
    example +=
        " --split-target " + std::to_string(report.chosen.mxfp4_split_target);
    example +=
        " --split-cap " + std::to_string(report.chosen.mxfp4_split_cap);
  }
  if (report.attention_applicable && report.chosen.attention_split > 0) {
    example +=
        " --attention-split " + std::to_string(report.chosen.attention_split);
  }
  if (report.draft_attached && report.chosen.draft_tokens > 0) {
    example += " --draft-block " + std::to_string(report.chosen.draft_tokens);
  }
  if (report.draft_attached && report.chosen.draft_context > 0) {
    example +=
        " --draft-context " + std::to_string(report.chosen.draft_context);
  }
  example += " --prompt-text \"Hello\"";
  out += "\nExample\n-------\n";
  out += example + "\n";
  out += "  # or apply the saved entry directly:\n";
  out += "  tessera-cli run --model " + report.model_path + " " +
         report.kv_flag + "\\\n      --calibration " + report.file +
         " --prompt-text \"Hello\"\n";
  return out;
}

}  // namespace tessera
