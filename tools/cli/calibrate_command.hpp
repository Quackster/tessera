#pragma once

#include "cli_options.hpp"

// The `calibrate` subcommand: load one model, sweep the hardware-dependent
// settings (see docs/CALIBRATE.md), print one line per sweep point and a
// final report, and write the calibration file. All sweep logic lives in
// the library (tessera/calibrate.hpp); this is the thin front end.

namespace tessera::cli {

// The calibrate usage text; the caller prints it after a parse error.
void PrintCalibrateUsage();

// Run the calibrate command. `options` comes from the shared parser, so
// every flag parses; the routes calibrate does not sweep are reported as
// ignored. --model is required.
[[nodiscard]] int RunCalibrateCommand(const CliOptions& options);

}  // namespace tessera::cli
