#pragma once

// The `calibrate` subcommand: load one model, sweep the hardware-dependent
// settings (see docs/CALIBRATE.md), print one line per sweep point and a
// final report, and write the calibration file. All sweep logic lives in
// the library (tessera/calibrate.hpp); this is the thin front end.

namespace tessera::cli {

// Run `tessera-cli calibrate ...`. `argv[1]` is "calibrate" and the flags
// follow. Returns a process exit code.
int RunCalibrateCommand(int argc, char** argv);

}  // namespace tessera::cli
