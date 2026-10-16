#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "cli_options.hpp"
#include "tessera/engine.hpp"
#include "tessera/pipeline.hpp"
#include "tessera/speculative.hpp"
#include "tessera/types.hpp"

// Shared engine construction for the run and serve commands: one place
// builds an engine, attaches the speculative strategy the flags select
// and assembles the pipeline options, so the two commands cannot drift
// (AGENTS.md rule 2: one canonical flag mapping).

namespace tessera::cli {

// One engine for `device` with the CLI's prefill chunk, the built-in
// stderr diagnostics sink and --quiet/--verbose applied before the load.
[[nodiscard]] std::expected<std::unique_ptr<Engine>, StatusCode> MakeCliEngine(
    const CliOptions& options, int device);

// Apply --quiet/--verbose to an engine a layout created internally (the
// layout's own load already ran at the default level).
void ApplyCliLogLevel(Engine& engine, const CliOptions& options);

// Attach the speculative strategy the flags select: a non-empty --draft
// attaches DFlash2, else --speculate attaches MTP. A failed attach only
// warns, so generation continues without speculative decoding. No-op
// when neither flag is set.
void AttachCliStrategy(Engine& engine, const CliOptions& options);

// One new instance of the strategy the flags select. The layout
// AttachSpeculative overloads call a factory once per rank, so this is
// the canonical factory body for run and serve.
[[nodiscard]] std::unique_ptr<SpeculativeStrategy> MakeCliStrategy(
    const CliOptions& options);

// The StrategyOptions the flag-selected strategy instances share.
[[nodiscard]] StrategyOptions CliStrategyOptions(const CliOptions& options);

// The PipelineOptions both the run and the serve pipeline paths use, so
// the two honor --pipeline-split, --gpu-layers and the last-stage
// embedding identically.
[[nodiscard]] PipelineOptions MakeCliPipelineOptions(
    const CliOptions& options, std::span<const int> devices);

// Handle `--gpu-layers help`: size the model without loading it and
// return the process exit code, or nullopt when the flag is not set.
// The caller validates --model first.
[[nodiscard]] std::optional<int> HandleGpuLayersHelp(const CliOptions& options);

}  // namespace tessera::cli
