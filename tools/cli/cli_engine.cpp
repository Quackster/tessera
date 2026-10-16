#include "cli_engine.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "cli_helpers.hpp"
#include "tessera/log.hpp"
#include "tessera/speculative.hpp"

namespace tessera::cli {

std::expected<std::unique_ptr<Engine>, StatusCode> MakeCliEngine(
    const CliOptions& options, int device) {
  EngineOptions engine_options;  // default stderr diagnostics sink
  if (options.verbose) {
    engine_options.diagnostics.SetMinLevel(log::Level::Debug);
  } else if (options.quiet) {
    engine_options.diagnostics.SetMinLevel(log::Level::Warn);
  }
  engine_options.device_index = device;
  engine_options.prefill_chunk_tokens = options.prefill_chunk;
  return Engine::Create(engine_options);
}

void ApplyCliLogLevel(Engine& engine, const CliOptions& options) {
  if (options.verbose) {
    engine.Diagnostics().SetMinLevel(log::Level::Debug);
  } else if (options.quiet) {
    engine.Diagnostics().SetMinLevel(log::Level::Warn);
  }
}

std::unique_ptr<SpeculativeStrategy> MakeCliStrategy(
    const CliOptions& options) {
  return options.draft_path.empty() ? CreateMtpStrategy()
                                    : CreateDFlash2Strategy();
}

StrategyOptions CliStrategyOptions(const CliOptions& options) {
  return StrategyOptions{options.draft_path, options.draft_block};
}

void AttachCliStrategy(Engine& engine, const CliOptions& options) {
  if (options.draft_path.empty() && !options.speculate) {
    return;
  }
  std::unique_ptr<SpeculativeStrategy> strategy = MakeCliStrategy(options);
  const std::string label = options.draft_path.empty()
                                ? "MTP"
                                : "DFlash2 ('" + options.draft_path + "')";
  auto attach = strategy->Attach(CliStrategyOptions(options));
  if (!attach) {
    engine.Diagnostics().Warn(
        "cli", label +
                   " attach failed: " + std::string(ToString(attach.error())) +
                   "; continuing without speculative decoding");
    return;
  }
  auto attached = engine.AttachSpeculative(std::move(strategy));
  if (!attached) {
    engine.Diagnostics().Warn("cli",
                              label + " strategy attach failed: " +
                                  std::string(ToString(attached.error())) +
                                  "; continuing without speculative decoding");
  }
}

PipelineOptions MakeCliPipelineOptions(const CliOptions& options,
                                       std::span<const int> devices) {
  PipelineOptions pipeline;
  pipeline.model_path = options.model_path;
  pipeline.devices.assign(devices.begin(), devices.end());
  pipeline.context = options.context;
  pipeline.kv_type = options.kv_type;
  pipeline.kv_offload = options.kv_offload;
  pipeline.mtp_path = options.mtp_path;
  pipeline.stage_layers = options.pipeline_split;
  pipeline.gpu_layers = options.gpu_layers;
  // The drafter runs on the last stage and embeds the anchor tokens
  // (DFlash2 and MTP both do), so it needs the target embedding there.
  pipeline.last_stage_embedding =
      options.speculate || !options.draft_path.empty();
  return pipeline;
}

std::optional<int> HandleGpuLayersHelp(const CliOptions& options) {
  if (!options.gpu_layers_help) {
    return std::nullopt;
  }
  auto engine_result = MakeCliEngine(options, SelectedDevices(options).front());
  if (!engine_result) {
    std::fprintf(stderr, "cli: engine create failed (%s)\n",
                 ToString(engine_result.error()).data());
    return 1;
  }
  std::unique_ptr<Engine> engine = std::move(*engine_result);
  return LogGpuLayersAdvice(*engine, options.model_path, options.context,
                            options.kv_type, CliStrategyLabel(options),
                            options.kv_offload) == 0
             ? 0
             : 1;
}

}  // namespace tessera::cli
