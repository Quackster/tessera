#pragma once

#include "cli_options.hpp"

// The `run` subcommand: one generation (or a batch of parallel sessions)
// on the layout the flags select. `--pipeline`, `--tp`, `--cp`,
// `--spec-split`, `--hybrid` and the plain single-device path all run
// here; the flag mapping lives in cli_options.hpp, the engine builders in
// cli_engine.hpp.

namespace tessera::cli {

// Run the command and return the process exit code. Prints its own
// diagnostics and usage text.
int RunRunCommand(const CliOptions& options);

}  // namespace tessera::cli
