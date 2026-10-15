#include "tessera/calibrate.hpp"

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/files.hpp"
#include "core/json.hpp"
#include "tessera/engine.hpp"

// The calibration file: the hardware key, the strict JSON reader/writer,
// and the merge helpers that apply a saved entry to a request. One canonical
// implementation of each (AGENTS.md rules 2 and 3).

namespace tessera {

namespace {

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

// Optional string field: absent leaves `out` unchanged. Used for fields
// added after the first calibration file version, so a file without them
// still loads.
bool ReadOptionalString(const core::Json& object, std::string_view key,
                        std::string& out) {
  const core::Json* field = object.Find(key);
  if (field == nullptr) {
    return true;
  }
  if (!field->isString()) {
    return false;
  }
  out = field->AsString();
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
  double max_context = 0.0;
  if (!ReadString(item, "backend", entry.key.backend) ||
      !ReadString(item, "device", entry.key.device) ||
      !ReadString(item, "model", entry.key.model) ||
      !ReadNumber(item, "context", context) ||
      !ReadString(item, "kv_type", entry.key.kv_type) ||
      !ReadString(item, "strategy", entry.key.strategy) ||
      !ReadOptionalString(item, "parallel", entry.key.parallel) ||
      !ReadNumber(item, "mxfp4_split_target", split) ||
      !ReadOptionalNumber(item, "mxfp4_split_cap", cap) ||
      !ReadNumber(item, "prefill_chunk_tokens", prefill) ||
      !ReadOptionalNumber(item, "attention_split", attention) ||
      !ReadNumber(item, "draft_tokens", draft) ||
      !ReadOptionalNumber(item, "draft_context", draft_context) ||
      !ReadOptionalNumber(item, "max_context", max_context) ||
      !ReadNumber(item, "decode_tps", entry.decode_tps) ||
      !ReadNumber(item, "prefill_tps", entry.prefill_tps) ||
      !ReadString(item, "date", entry.date)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (context < 0.0 || split < 0.0 || cap < 0.0 || prefill < 0.0 ||
      attention < 0.0 || draft < 0.0 || draft_context < 0.0 ||
      max_context < 0.0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  entry.key.context = static_cast<std::size_t>(context);
  entry.config.mxfp4_split_target = static_cast<std::size_t>(split);
  entry.config.mxfp4_split_cap = static_cast<std::size_t>(cap);
  entry.config.prefill_chunk_tokens = static_cast<std::size_t>(prefill);
  entry.config.attention_split = static_cast<std::size_t>(attention);
  entry.config.draft_tokens = static_cast<std::size_t>(draft);
  entry.config.draft_context = static_cast<std::size_t>(draft_context);
  entry.max_context = static_cast<std::size_t>(max_context);
  return entry;
}

}  // namespace

std::string HardwareKey::ToString() const {
  return backend + "|" + device + "|" + model + "|" + std::to_string(context) +
         "|" + kv_type + "|" + strategy + "|" + parallel;
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
    item.Set("max_context",
             core::Json::Number(static_cast<double>(entry.max_context)));
    item.Set("kv_type", core::Json::String(entry.key.kv_type));
    item.Set("strategy", core::Json::String(entry.key.strategy));
    item.Set("parallel", core::Json::String(entry.key.parallel));
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
                            std::string_view strategy,
                            std::string_view parallel) {
  HardwareKey key;
  key.backend = std::string(backend);
  key.device = std::string(device);
  key.model = std::string(model);
  key.context = context;
  key.kv_type = std::string(kv_type);
  key.strategy = std::string(strategy);
  key.parallel = std::string(parallel);
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

}  // namespace tessera
