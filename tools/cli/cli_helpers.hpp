#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include "tessera/calibrate.hpp"
#include "tessera/types.hpp"

// Shared front-end helpers used by the run, serve and calibrate commands:
// the model summary line, the usage text, and the calibration lookup. The
// heavy logic stays in the library.

namespace tessera {

class Engine;
class Model;

namespace cli {

// One summary line plus the first few tensor names for a loaded model, so
// both `run` and the deferred `serve` loader report it identically.
void LogModelSummary(Engine& engine, Model& model);

// The run/serve usage text (also points at the calibrate command).
void PrintUsage();

// Resolve the calibration configuration for a loaded model: build the
// hardware key, look the entry up in the calibration file, and merge it
// under the caller's explicit values (an explicit nonzero value wins). A
// missing file is not an error (no calibration yet); a malformed file is
// reported and ignored.
[[nodiscard]] CalibrationConfig ResolveCalibration(
    Engine& engine, Model& model, std::size_t context, KvCacheType kv_type,
    std::string_view strategy, const std::string& path,
    const CalibrationConfig& explicit_config);

}  // namespace cli
}  // namespace tessera
