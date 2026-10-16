#pragma once

#include "cli_options.hpp"

// The serve command: one HTTP server over the runner pool the layout
// flags select. Replicas are the default and the only route that pages
// and warms per device; --pipeline, --tp, --cp and --spec-split each
// build one split runner, --hybrid one runner per data-parallel group.
namespace tessera::cli {

// Run the serve command. --model is required; the caller handles the
// --gpu-layers help request before dispatch.
[[nodiscard]] int RunServeCommand(const CliOptions& options);

}  // namespace tessera::cli
