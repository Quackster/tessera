#pragma once

#include <cstddef>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tessera/types.hpp"

// Hardware calibration: measure the engine settings that depend on the
// machine, keep the fastest value, and persist it. Calibration changes only
// speed; the produced tokens are the reference and must not change. See
// docs/CALIBRATE.md.
//
// Usage:
//   SweepOptions options;
//   options.defaults.prefill_chunk_tokens = 512;
//   auto outcome = tessera::RunCalibrationSweep(options, measure);

namespace tessera {

struct GenerateOptions;

// The engine settings calibration can tune. Each is a speed-only knob:
// a change must not change the produced tokens.
enum class Setting : int {
  MxFp4SplitTarget,    // split-K workgroups for the fp8 MXFP4 GEMM, decode
  MxFp4SplitCap,       // split-K factor cap for the fp8 MXFP4 GEMM, decode
  PrefillChunkTokens,  // tokens per prefill forward, prefill
  AttentionSplit,      // single-token flash-decoding chunks, decode
  DraftTokens,         // draft block for speculative decode, decode
  DraftContext,        // DFlash2 draft context window, decode
};

// Human-readable name of a setting ("mxfp4_split_target",
// "prefill_chunk_tokens", "draft_tokens", ...).
[[nodiscard]] std::string_view ToString(Setting setting);

// One calibration configuration. A zero field means the built-in default.
struct CalibrationConfig {
  std::size_t mxfp4_split_target = 0;
  std::size_t mxfp4_split_cap = 0;
  std::size_t prefill_chunk_tokens = 0;
  std::size_t attention_split = 0;
  std::size_t draft_tokens = 0;
  std::size_t draft_context = 0;
};

// Speed of one configuration, in tokens per second.
struct Measurement {
  double decode_tps = 0.0;
  double prefill_tps = 0.0;
};

// Measures one configuration on one prompt. `setting` is the setting under
// test, so the callback can pick the prompt length and the decode budget
// that setting needs (a decode sweep uses a short prompt, a prefill sweep a
// long one). `prompt` is the 0-based index of a fixed prompt set. When
// `warmup` is true the callback runs the configuration once and the result
// is discarded (the sweep's warm step, so a fresh configuration's
// first-touch cost is not billed to a point). Rates are tokens per second;
// a failure aborts the sweep.
using MeasureFn = std::function<std::expected<Measurement, StatusCode>(
    Setting setting, const CalibrationConfig& config, std::size_t prompt,
    bool warmup)>;

// One sweep point, for the command's per-point report line.
struct SweepPoint {
  Setting setting = Setting::PrefillChunkTokens;
  std::size_t value = 0;
  double tps = 0.0;    // median rate over the prompts
  bool kept = false;   // true when this value won its setting
};

// The relative gain a winner must beat the default by to be kept: the
// noise floor (Strata uses 0.03). A single measurement is noisy by a few
// percent, so a smaller gain is treated as noise.
constexpr double kMinGain = 0.03;

// What the sweep should tune.
struct SweepOptions {
  // Number of prompts the measure callback understands (>= 1).
  std::size_t prompt_count = 1;
  // The built-in defaults each setting falls back to. The default value is
  // both a candidate and the floor the sweep must beat.
  CalibrationConfig defaults;
  // Settings to sweep. A setting left false is reported as not applicable
  // (the split target and cap on Vulkan, the draft settings when no draft
  // is attached, the attention split when the prompts are too short).
  bool sweep_split_target = true;
  bool sweep_split_cap = true;
  bool sweep_prefill_chunk = true;
  bool sweep_attention_split = true;
  bool sweep_draft_tokens = false;
  bool sweep_draft_context = false;
};

struct SweepOutcome {
  std::vector<SweepPoint> points;
  CalibrationConfig config;  // the chosen configuration
  std::vector<Setting> not_applicable;
};

// Candidate values for one setting: the fixed list from the design, the
// default value when nonzero, deduped, sorted, and bounded (`upper_bound`
// nonzero drops values above it, so a draft never exceeds the checkpoint
// block; zero values are always dropped).
[[nodiscard]] std::vector<std::size_t> CandidateValues(
    Setting setting, std::size_t default_value, std::size_t upper_bound = 0);

// Median of the values; 0.0 for an empty span. An even count averages the
// two middle values.
[[nodiscard]] double MedianOf(std::span<const double> values);

// Largest full-attention context that fits `available_bytes` of KV cache
// for `config` and `kv_type`. KV bytes per token are counted over the
// full-attention layers only (a hybrid model has fewer of them). Returns 0
// when the inputs cannot size a cache (no layers, no KV heads, no head
// dim, or no full-attention layer).
[[nodiscard]] std::size_t MaxContextForKv(std::uint64_t available_bytes,
                                          const TransformerConfig& config,
                                          KvCacheType kv_type);

// Pick the value to keep. `candidates` and `rates` are parallel. Returns
// the default when the best value does not beat it by more than kMinGain,
// the best value when no default is present in the list, and nullopt for an
// empty list (or a length mismatch).
[[nodiscard]] std::optional<std::size_t> PickBest(
    std::span<const std::size_t> candidates, std::size_t default_value,
    std::span<const double> rates);

// Run the sweep. For each enabled setting it warms and measures every
// candidate, applies the pick rule, and confirms a winner against the
// default with an interleaved re-measurement (three rounds each, median),
// so a single noisy win is rejected. A failure from `measure` aborts the
// sweep with that status and leaves the caller's defaults untouched.
//
// Usage:
//   auto outcome = RunCalibrationSweep(options, measure);
[[nodiscard]] std::expected<SweepOutcome, StatusCode> RunCalibrationSweep(
    const SweepOptions& options, const MeasureFn& measure);

// Identity of a calibrated machine and model: backend, device name, model
// name, context, KV type and strategy. A change to any field means an entry
// does not apply.
struct HardwareKey {
  std::string backend;
  std::string device;
  std::string model;
  std::size_t context = 0;
  std::string kv_type;
  std::string strategy;

  // The fields joined by '|', the file's lookup key.
  [[nodiscard]] std::string ToString() const;
  [[nodiscard]] bool operator==(const HardwareKey& other) const = default;
};

// One persisted calibration: the chosen settings, the measured rates and
// the date the measurement ran.
struct CalibrationEntry {
  HardwareKey key;
  CalibrationConfig config;
  double decode_tps = 0.0;
  double prefill_tps = 0.0;
  std::string date;
};

// A JSON calibration file: one entry per hardware key. The reader is strict
// (MalformedFile for a bad document); a missing file is FileNotFound, which
// callers treat as "no calibration yet". There is no implicit global
// lookup: a caller loads the file it was handed.
class CalibrationFile {
 public:
  // Read the file at `path`. FileNotFound when it does not exist,
  // MalformedFile when it is not a valid calibration document.
  [[nodiscard]] static std::expected<CalibrationFile, StatusCode> Load(
      std::string_view path);
  // Write the file, replacing any existing one. MalformedFile on I/O error.
  [[nodiscard]] std::expected<void, StatusCode> Save(
      std::string_view path) const;

  // The entry for `key`, or nullptr.
  [[nodiscard]] const CalibrationEntry* Find(const HardwareKey& key) const;
  // Insert, or replace the entry with the same key.
  void Set(CalibrationEntry entry);
  [[nodiscard]] std::span<const CalibrationEntry> Entries() const;

 private:
  std::vector<CalibrationEntry> entries_;
};

// Build a hardware key from the runtime facts. `model` is the model name
// (the caller falls back to the path basename when the model is unnamed).
[[nodiscard]] HardwareKey MakeHardwareKey(std::string_view backend,
                                          std::string_view device,
                                          std::string_view model,
                                          std::size_t context,
                                          std::string_view kv_type,
                                          std::string_view strategy);

// Merge a saved entry into a configuration: a field the caller set (nonzero)
// always wins; a zero field takes the saved value. Returns the merged
// configuration.
[[nodiscard]] CalibrationConfig ApplySavedCalibration(
    const CalibrationConfig& explicit_config, const CalibrationEntry& saved);

// Copy a calibration configuration onto a request's tuning fields
// (prefill chunk, draft block, split target). A zero field leaves the
// request's value unchanged, so an explicit request value always wins.
void ApplyToGenerateOptions(const CalibrationConfig& config,
                            GenerateOptions& options);

// The inputs of the human-readable calibration report.
struct CalibrationReport {
  std::string backend;
  std::string device;
  std::string model;       // model identity in the key
  std::string model_path;  // the path a run command would take
  std::string kv_type;     // display name, "int8"
  std::string kv_flag;     // run flag prefix, "--kv-q8 " (empty for fp32)
  std::string strategy;
  std::size_t key_context = 0;
  std::size_t max_context = 0;  // memory-bound context for the KV type
  CalibrationConfig defaults;
  std::vector<SweepPoint> points;
  std::vector<Setting> not_applicable;
  CalibrationConfig chosen;
  double decode_tps = 0.0;
  double prefill_tps = 0.0;
  std::string file;
  bool split_applicable = false;
  bool attention_applicable = false;
  bool draft_attached = false;
};

// Format the report as plain text: the identity, the sweep, the chosen
// settings, and an example `run` command that reproduces them.
//
// Usage:
//   std::puts(FormatCalibrationReport(report).c_str());
[[nodiscard]] std::string FormatCalibrationReport(
    const CalibrationReport& report);

}  // namespace tessera
