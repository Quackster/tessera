#pragma once

#include "cli_options.hpp"

// The plain single-device run path: one engine, one model, streaming
// greedy generation, parallel `--prompt-text` sessions and the vision
// (`--image`/`--mmproj`) path. Kept apart from the split layouts in
// run_command.cpp so each file stays one responsibility.

namespace tessera::cli {

// Run on one device and return the process exit code. The caller
// validates --model and the layout flags; this path is only selected
// when no split flag is set.
int RunSingleDeviceGeneration(const CliOptions& options);

}  // namespace tessera::cli
