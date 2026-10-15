#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

#include "tessera/calibrate.hpp"
#include "tessera/defaults.hpp"
#include "tessera/log.hpp"
#include "tessera/types.hpp"

// The `calibrate --gpus` front end: largest KV-cache context for one
// parallel layout. The speed sweep needs single-device tuning knobs, so
// a parallel run calibrates the memory bound only (no sweep points) and
// stores it under a key carrying the layout. Shared with the main
// calibrate command (one canonical implementation of each helper).

namespace tessera::cli {

// Headroom kept free for scratch and fragmentation when reporting the
// maximum context length.
constexpr std::uint64_t kMemoryReserveBytes = 2ull * 1024 * 1024 * 1024;
// Below this the reported maximum context is not useful.
constexpr std::size_t kMinReportedContext = 2048;

// Today's date (YYYY-MM-DD) for the entry stamp.
std::string TodayIso();

// The run flag that selects a KV cache type (empty for the fp32 default).
std::string KvFlag(KvCacheType type);

// Load the file at `path` (a missing file starts empty; a malformed one
// is reported and ignored), replace the entry with the same key, and
// save. Returns the save error, if any.
std::expected<void, StatusCode> SaveCalibrationEntry(
    log::Diagnostics& log, const std::string& path, CalibrationEntry entry);

struct ParallelCalibrateOptions {
  std::string model_path;
  std::string gpus;  // raw --gpus value, for example "0,1"
  bool tp = false;
  bool cp = false;
  bool pipeline = false;
  std::size_t context = kDefaultContext;
  KvCacheType kv_type = KvCacheType::Q8;
  std::string calibration_file;
  // Speed sweep inputs: prompts file (empty uses the fixed default set) and
  // the point budget (0 means unlimited). The parallel path sweeps the same
  // decode speed knobs the single-device path does.
  std::string prompts_path;
  std::size_t calibrate_limit = 0;
  // Attach the MTP strategy to the route and sweep the draft block. Only
  // --speculate is supported on a parallel layout; the DFlash2 --draft
  // checkpoint is single-device.
  bool speculate = false;
  bool quiet = false;
};

// Calibrate the largest KV-cache context for `--gpus` with `--tp`, `--cp`
// or `--pipeline` (exactly one): build the route, measure each rank's
// free memory and resident weight bytes, and combine the per-device
// budgets (tensor: the minimum; context: ranks times the minimum, one
// key chunk each; pipeline: per-stage budgets scaled by layer count).
// Runs one short validation generate through the route, writes the entry
// under a key carrying the layout, and prints the report. Returns a
// process exit code.
int RunParallelCalibrate(const ParallelCalibrateOptions& options);

}  // namespace tessera::cli
