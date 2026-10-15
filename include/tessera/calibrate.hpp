#pragma once

#include <cstddef>
#include <cstdint>
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
  // At most this many candidate evaluations in total, counted straight
  // through the sweep whatever the setting. Each evaluation is exactly
  // one engine run (first prompt, no warmup). A setting cut short keeps
  // the fastest measured value with no gain gate and no confirmation; a
  // setting with no measured value keeps its default and is reported as
  // skipped. Zero means no budget: every enabled setting is swept.
  std::size_t max_points = 0;
};

struct SweepOutcome {
  std::vector<SweepPoint> points;
  CalibrationConfig config;  // the chosen configuration
  std::vector<Setting> not_applicable;
  // Settings with no measured value under a point budget. Unlike
  // not_applicable, these could run on this backend; the budget ran out
  // before any of their candidates was tested.
  std::vector<Setting> skipped;
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

// Bytes of one full-attention KV cache token for `config` and `kv_type`,
// counted over the full-attention layers only (a hybrid model has fewer).
// Zero when the inputs cannot size a cache (no layers, no KV heads, no
// head dim, or no full-attention layer).
[[nodiscard]] std::uint64_t KvBytesPerToken(const TransformerConfig& config,
                                            KvCacheType kv_type);

// Largest number of concurrent sessions whose KV caches at `context`
// tokens each fit `available_bytes`: available divided by the per-session
// cache size (context times the per-token bytes). Zero when any input is
// zero.
[[nodiscard]] std::size_t MaxConcurrentSessions(
    std::uint64_t available_bytes, std::uint64_t kv_bytes_per_token,
    std::size_t context);

// Throughput of one concurrent-session count, for the report.
struct ConcurrentRate {
  std::size_t sessions = 0;
  double total_tps = 0.0;        // all sessions together
  double per_session_tps = 0.0;  // total_tps divided by `sessions`
};

// Aggregate one benchmark row. `tokens` and `decode_ms` are parallel, one
// entry per session. The total rate is the total tokens over the wall time
// of the slowest session; the per-session rate divides that by the session
// count. A zero or empty input yields a zeroed rate with `sessions` set.
[[nodiscard]] ConcurrentRate SummarizeConcurrentSessions(
    std::size_t sessions, std::span<const std::size_t> tokens,
    std::span<const double> decode_ms);

// Pick the value to keep. `candidates` and `rates` are parallel. Returns
// the default when the best value does not beat it by more than kMinGain,
// the best value when no default is present in the list, and nullopt for an
// empty list (or a length mismatch).
[[nodiscard]] std::optional<std::size_t> PickBest(
    std::span<const std::size_t> candidates, std::size_t default_value,
    std::span<const double> rates);

// Run the sweep. For each enabled setting it warms and measures candidates
// in ascending order, stops that setting once three candidates in a row
// fail to beat the best value so far (a downward trend), applies the pick
// rule, and confirms a winner against the default with an interleaved
// re-measurement (three rounds each, median), so a single noisy win is
// rejected. A failure from `measure` aborts the sweep with that status and
// leaves the caller's defaults untouched.
//
// Usage:
//   auto outcome = RunCalibrationSweep(options, measure);
[[nodiscard]] std::expected<SweepOutcome, StatusCode> RunCalibrationSweep(
    const SweepOptions& options, const MeasureFn& measure);

// Identity of a calibrated machine and model: backend, device name, model
// name, context, KV type, strategy and parallel layout. A change to any
// field means an entry does not apply.
struct HardwareKey {
  std::string backend;
  std::string device;
  std::string model;
  std::size_t context = 0;
  std::string kv_type;
  std::string strategy;
  // Parallel layout the entry was measured for: "none" (one device) or
  // "tp2", "cp2", "pipe2" (route plus world size). Old files without the
  // field read as "none".
  std::string parallel = "none";

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
  // Largest full-attention context the device's free memory allows for the
  // key's KV type when the entry was written. Informational; 0 when the
  // entry predates the field.
  std::size_t max_context = 0;
  // Layers kept in VRAM when the entry was measured (llama.cpp -ngl order).
  // 0 means the full upload; a nonzero value pages the early trunk blocks.
  // An MoE target larger than VRAM needs a nonzero value to load at all, so
  // run and serve apply it. 0 when the entry predates the field.
  std::size_t gpu_layers = 0;
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
                                           std::string_view strategy,
                                           std::string_view parallel = "none");

// Merge a saved entry into a configuration: a field the caller set (nonzero)
// always wins; a zero field takes the saved value. Returns the merged
// configuration.
[[nodiscard]] CalibrationConfig ApplySavedCalibration(
    const CalibrationConfig& explicit_config, const CalibrationEntry& saved);

// The paging to load with. An explicit nonzero value always wins; a zero
// explicit value takes the saved entry's `gpu_layers`. A zero result means
// the full upload, so an explicit 0 cannot force it over a saved nonzero
// value (pass the flag only when it is set).
[[nodiscard]] std::size_t ApplySavedGpuLayers(std::size_t explicit_layers,
                                              const CalibrationEntry& saved);

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
  std::string draft_path;  // DFlash2 checkpoint dir for the example, "" when none
  std::string kv_type;     // display name, "int8"
  std::string kv_flag;     // run flag prefix, "--kv-q8 " (empty for fp32)
  std::string strategy;
  std::string parallel;  // display name, "none", "tp2", "cp2", "pipe2"
  // Flag suffix reproducing the parallel layout, "" when none.
  std::string parallel_flags;
  std::size_t key_context = 0;
  std::size_t max_context = 0;  // memory-bound context for the KV type
  // Layers kept in VRAM for the example commands; 0 is the full upload.
  std::size_t gpu_layers = 0;
  CalibrationConfig defaults;
  std::vector<SweepPoint> points;
  std::vector<Setting> not_applicable;
  std::vector<Setting> skipped;  // settings the point budget skipped
  CalibrationConfig chosen;
  double decode_tps = 0.0;
  double prefill_tps = 0.0;
  std::string file;
  bool split_applicable = false;
  bool attention_applicable = false;
  bool draft_attached = false;
  // Concurrent-session benchmark, one row per session count, plus the
  // largest count whose KV caches fit the active context for this KV type.
  // Empty when the benchmark did not run (`max_concurrent` is still set by
  // the memory bound).
  std::vector<ConcurrentRate> concurrent;
  std::size_t max_concurrent = 0;
};

// The two copy-paste run commands for a report: the explicit-flags form
// and the --calibration form. Both run at the memory-bound max context.
struct ExampleCommands {
  std::string explicit_flags;
  std::string from_file;
};

// Build the example commands for a report: the tuned flags plus the draft
// checkpoint for a dflash2 entry and --speculate for mtp. One canonical
// implementation feeding the text report.
//
// Usage:
//   ExampleCommands example = BuildExampleCommands(report);
[[nodiscard]] ExampleCommands BuildExampleCommands(
    const CalibrationReport& report);

// Format the report as plain text: the identity, the sweep, the chosen
// settings, and an example `run` command that reproduces them.
//
// Usage:
//   std::puts(FormatCalibrationReport(report).c_str());
[[nodiscard]] std::string FormatCalibrationReport(
    const CalibrationReport& report);

}  // namespace tessera
