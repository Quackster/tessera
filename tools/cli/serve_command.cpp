#include "serve_command.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cli_common.hpp"
#include "cli_engine.hpp"
#include "cli_helpers.hpp"
#include "tessera/calibrate.hpp"
#include "tessera/context_parallel.hpp"
#include "tessera/engine.hpp"
#include "tessera/hybrid.hpp"
#include "tessera/model.hpp"
#include "tessera/pipeline.hpp"
#include "tessera/runner.hpp"
#include "tessera/serve.hpp"
#include "tessera/spec_split.hpp"
#include "tessera/speculative.hpp"
#include "tessera/tensor_parallel.hpp"
#include "tessera/types.hpp"

namespace tessera::cli {
namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;

// True when --hybrid runs at least one multi-device group (the
// model-parallel route that cannot page). An invalid mesh reports false;
// the pool builder then reports the real error.
bool HybridHasSplitGroup(const CliOptions& options) {
  HybridOptions hybrid_options;
  hybrid_options.model_path = options.model_path;
  hybrid_options.devices = SelectedDevices(options);
  hybrid_options.dp_groups = options.dp_groups;
  hybrid_options.mp_kind =
      options.cp_mode ? HybridMpKind::Context : HybridMpKind::Tensor;
  hybrid_options.context = options.context;
  hybrid_options.kv_type = options.kv_type;
  auto groups = HybridGroups(hybrid_options);
  if (!groups) {
    return false;
  }
  for (const HybridGroupDevices& group : *groups) {
    if (group.devices.size() > 1) {
      return true;
    }
  }
  return false;
}

// The flags serve cannot honor: request-level settings arrive with each
// HTTP request, so the CLI values would fight the API. Reported once
// instead of ignored in silence.
void WarnInertServeFlags(const CliOptions& options) {
  if (!options.prompt_texts.empty()) {
    WarnIgnoredFlag("--prompt-text", "prompts arrive with each HTTP request");
  }
  if (options.no_chat) {
    WarnIgnoredFlag("--no-chat",
                    "chat rendering is per request in the HTTP API");
  }
  if (!options.image_path.empty() || !options.mmproj_path.empty()) {
    WarnIgnoredFlag("--image/--mmproj", "images arrive with each chat request");
  }
  if (options.max_thinking_tokens != 0) {
    WarnIgnoredFlag("--max-thinking-tokens",
                    "budgets are per request in the HTTP API");
  }
  if (!options.hybrid_mode && options.dp_groups != 1) {
    WarnIgnoredFlag("--dp", "--dp selects the groups of --hybrid");
  }
  if (!options.pipeline_mode && !options.pipeline_split.empty()) {
    WarnIgnoredFlag("--pipeline-split", "--pipeline-split needs --pipeline");
  }
  const bool layout_runs_mp =
      options.tp_mode || options.cp_mode ||
      (options.hybrid_mode && HybridHasSplitGroup(options));
  if (layout_runs_mp && options.gpu_layers != 0) {
    WarnIgnoredFlag("--gpu-layers",
                    "the model-parallel layouts do not page; use a "
                    "single-device, --pipeline or --spec-split serve");
  }
}

// Explicit calibration values from the CLI; the saved calibration file
// merges under them (an explicit value wins).
CalibrationConfig ExplicitCalibration(const CliOptions& options) {
  CalibrationConfig applied;
  applied.prefill_chunk_tokens = options.prefill_chunk;
  applied.draft_tokens = options.draft_block;
  applied.mxfp4_split_target = options.split_target;
  applied.attention_split = options.attention_split;
  applied.draft_context = options.draft_context;
  return applied;
}

// Resolve the calibration for a loaded serve model: rank 0 of a split
// layout, or the first replica. Homogeneous devices share the settings.
CalibrationConfig ResolveServeCalibration(const CliOptions& options,
                                          Engine& engine, Model& model) {
  return ResolveCalibration(engine, model, options.context, options.kv_type,
                            CliStrategyLabel(options), options.calibration_path,
                            ExplicitCalibration(options));
}

// One greedy token through a just-created split runner with the resolved
// calibration applied, so the first real turn pays no first-touch cost.
std::expected<void, StatusCode> WarmSplitRunner(
    const Runner& runner, const CalibrationConfig& calibration,
    KvCacheType kv_type) {
  GenerateOptions warm;
  warm.max_completion_tokens = 1;
  warm.progress_every = 0;
  warm.kv_type = kv_type;
  ApplyToGenerateOptions(calibration, warm);
  auto warmed = runner.Submit(warm, [](std::uint32_t) { return true; });
  if (!warmed) {
    std::fprintf(stderr, "cli: kernel warmup failed (%s)\n",
                 ToString(warmed.error()).data());
    return std::unexpected(warmed.error());
  }
  return {};
}

// The state shared by every replica of a serve pool: the resolved paging
// value, the host bytes held by paged replicas so far, and the
// calibration resolved from the first loaded model.
struct ReplicaState {
  std::size_t gpu_layers = 0;
  std::uint64_t host_held_bytes = 0;
  CalibrationConfig calibration;
  bool loaded_one = false;
};

// Load one replica on `device` and add it to the pool. Runs the same
// pre-load checks, sampler attach and warmup as the single-device run
// path, so a serve replica equals a `run` on the same device.
std::expected<void, StatusCode> AddReplica(const CliOptions& options,
                                           int device,
                                           const ServeProgress& report,
                                           ReplicaState* state,
                                           RunnerPool& pool) {
  report("loading weights on gpu " + std::to_string(device));
  auto created = MakeCliEngine(options, device);
  if (!created) {
    return std::unexpected(created.error());
  }
  std::unique_ptr<Engine> engine_ptr = std::move(*created);
  auto& log = engine_ptr->Diagnostics();
  const std::string_view strategy = CliStrategyLabel(options);
  // Apply a calibrated paging value before the load: an MoE target
  // larger than VRAM only loads paged, and paging is a load-time
  // setting. An explicit --gpu-layers always wins.
  if (!options.calibration_path.empty()) {
    const std::size_t explicit_layers = state->gpu_layers;
    state->gpu_layers = ResolveCalibrationGpuLayers(
        *engine_ptr, options.model_path, options.context, options.kv_type,
        strategy, options.calibration_path, state->gpu_layers);
    if (state->gpu_layers != explicit_layers) {
      log.Info("cli", "calibration: --gpu-layers " +
                          std::to_string(state->gpu_layers));
    }
  }
  // Refuse a device that cannot hold the weights before the load starts,
  // so a two-device serve does not half-load. Under paging the file size
  // is not the resident size, so the resident set plus the KV cache is
  // judged instead. A zero free figure means the backend reports no free
  // memory; skip the check then and let the load fail on its own when it
  // truly cannot fit.
  const std::uint64_t needed = PathSizeBytes(options.model_path);
  const DeviceMemoryInfo memory = engine_ptr->Owner().MemoryInfo();
  if (state->gpu_layers == 0 && memory.total_bytes != 0 && needed != 0 &&
      memory.free_bytes != 0 && memory.free_bytes < needed) {
    std::fprintf(stderr,
                 "cli: gpu %d has %s MiB free but the model needs about "
                 "%s MiB; use fewer devices or a smaller quantization\n",
                 device,
                 std::to_string(memory.free_bytes / (1024 * 1024)).c_str(),
                 std::to_string(needed / (1024 * 1024)).c_str());
    return std::unexpected(StatusCode::OutOfMemory);
  }
  if (state->gpu_layers != 0) {
    if (!CheckGpuLayersFit(*engine_ptr, options.model_path, options.context,
                           options.kv_type, strategy, state->gpu_layers,
                           device)) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
  }
  // Refuse a replica whose paged weights cannot fit system RAM with the
  // ones already loaded, so a multi-device serve never drives the host
  // into the OOM killer.
  std::uint64_t replica_host_bytes = 0;
  if (!CheckHostRamFit(*engine_ptr, options.model_path, state->gpu_layers,
                       strategy, state->host_held_bytes, device,
                       &replica_host_bytes)) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  state->host_held_bytes += replica_host_bytes;
  ModelOptions model_options{options.model_path, options.context};
  model_options.gpu_layers = state->gpu_layers;
  model_options.mtp_path = options.mtp_path;
  auto model = engine_ptr->LoadModel(model_options);
  if (!model) {
    return std::unexpected(model.error());
  }
  LogModelSummary(*engine_ptr, **model);
  AttachCliStrategy(*engine_ptr, options);
  if (!state->loaded_one) {
    // The calibration key depends on the model; resolve it on the first
    // device. Homogeneous devices share the settings.
    state->calibration = ResolveServeCalibration(options, *engine_ptr, **model);
    state->loaded_one = true;
  }
  report("warming up kernels");
  GenerateOptions warm;
  warm.max_completion_tokens = 1;
  warm.progress_every = 0;
  // Warm the kernels with the serve-wide KV storage type, so the first
  // real turn pays no first-touch cost on another type.
  warm.kv_type = options.kv_type;
  ApplyToGenerateOptions(state->calibration, warm);
  auto warmed = engine_ptr->Generate(**model, warm);
  if (!warmed) {
    std::fprintf(stderr, "cli: kernel warmup failed (%s)\n",
                 ToString(warmed.error()).data());
    return std::unexpected(warmed.error());
  }
  // The pool carries the launcher's KV storage type (--kv-q8 and
  // friends); the serve layer reads it onto every turn, so an unlimited
  // turn reserves quantized rows and fits the device.
  pool.Add(std::make_unique<SingleDeviceRunner>(
      std::move(engine_ptr), std::move(*model), device, state->calibration,
      options.kv_type));
  return {};
}

// One replica per device, in list order (the default serve layout and a
// hybrid whose groups are all single-device).
std::expected<void, StatusCode> LoadReplicaPool(const CliOptions& options,
                                                std::span<const int> devices,
                                                const ServeProgress& report,
                                                RunnerPool& pool) {
  ReplicaState state;
  state.gpu_layers = options.gpu_layers;
  for (const int device : devices) {
    if (auto added = AddReplica(options, device, report, &state, pool);
        !added) {
      return added;
    }
  }
  return {};
}

// One runner over a model-parallel layout: apply the log level to every
// rank, report the summary, resolve the calibration from rank 0, wrap the
// layout in its runner and warm it.
template <typename Layout, typename MakeRunner>
std::expected<void, StatusCode> AddSplitRunner(
    const CliOptions& options, std::unique_ptr<Layout> layout, int first_device,
    const ServeProgress& report, MakeRunner make_runner, RunnerPool& pool) {
  Engine& engine0 = layout->RankEngine(0);
  Model& model0 = layout->RankModel(0);
  for (std::size_t rank = 0; rank < layout->RankCount(); ++rank) {
    ApplyCliLogLevel(layout->RankEngine(rank), options);
  }
  LogModelSummary(engine0, model0);
  const CalibrationConfig calibration =
      ResolveServeCalibration(options, engine0, model0);
  auto runner = make_runner(std::move(layout), first_device, calibration,
                            options.kv_type);
  if (auto warmed = WarmSplitRunner(*runner, calibration, options.kv_type);
      !warmed) {
    return warmed;
  }
  pool.Add(std::move(runner));
  report("ready");
  return {};
}

// The pool builder for --hybrid: one runner per data-parallel group. A
// single-device group is a plain replica; a multi-device group is one
// TensorParallel (default) or ContextParallel team over its devices.
std::expected<void, StatusCode> LoadHybridPool(const CliOptions& options,
                                               std::span<const int> devices,
                                               const ServeProgress& report,
                                               RunnerPool& pool) {
  HybridOptions hybrid_options;
  hybrid_options.model_path = options.model_path;
  hybrid_options.devices.assign(devices.begin(), devices.end());
  hybrid_options.dp_groups = options.dp_groups;
  hybrid_options.mp_kind =
      options.cp_mode ? HybridMpKind::Context : HybridMpKind::Tensor;
  hybrid_options.context = options.context;
  hybrid_options.kv_type = options.kv_type;
  auto groups = HybridGroups(hybrid_options);
  if (!groups) {
    std::fprintf(stderr, "cli: hybrid layout invalid (%s)\n",
                 ToString(groups.error()).data());
    return std::unexpected(groups.error());
  }
  ReplicaState replica_state;
  replica_state.gpu_layers = options.gpu_layers;
  for (const HybridGroupDevices& group : *groups) {
    if (group.devices.size() == 1) {
      if (auto added = AddReplica(options, group.devices.front(), report,
                                  &replica_state, pool);
          !added) {
        return added;
      }
      continue;
    }
    report("loading hybrid group " + std::to_string(group.index) + " across " +
           std::to_string(group.devices.size()) + " gpus");
    if (options.cp_mode) {
      ContextParallelOptions cp_options;
      cp_options.model_path = options.model_path;
      cp_options.devices = group.devices;
      cp_options.context = options.context;
      cp_options.kv_type = options.kv_type;
      auto cp = ContextParallel::Create(cp_options);
      if (!cp) {
        std::fprintf(stderr,
                     "cli: hybrid group %zu context-parallel create failed "
                     "(%s)\n",
                     group.index, ToString(cp.error()).data());
        return std::unexpected(cp.error());
      }
      if (!options.draft_path.empty() || options.speculate) {
        const auto factory = [&]() -> std::unique_ptr<SpeculativeStrategy> {
          return MakeCliStrategy(options);
        };
        auto attached =
            (*cp)->AttachSpeculative(factory, CliStrategyOptions(options));
        if (!attached) {
          std::fprintf(stderr, "cli: strategy attach failed (%s)\n",
                       ToString(attached.error()).data());
          return std::unexpected(attached.error());
        }
      }
      if (auto added = AddSplitRunner(
              options, std::move(*cp), group.devices.front(), report,
              [](std::unique_ptr<ContextParallel> layout, int device,
                 const CalibrationConfig& calibration, KvCacheType kv_type) {
                return MakeContextParallelRunner(std::move(layout), device,
                                                 calibration, kv_type);
              },
              pool);
          !added) {
        return added;
      }
      continue;
    }
    TensorParallelOptions tp_options;
    tp_options.model_path = options.model_path;
    tp_options.devices = group.devices;
    tp_options.context = options.context;
    tp_options.kv_type = options.kv_type;
    auto tp = TensorParallel::Create(tp_options);
    if (!tp) {
      std::fprintf(stderr,
                   "cli: hybrid group %zu tensor-parallel create failed (%s)\n",
                   group.index, ToString(tp.error()).data());
      return std::unexpected(tp.error());
    }
    if (!options.draft_path.empty() || options.speculate) {
      const auto factory = [&]() -> std::unique_ptr<SpeculativeStrategy> {
        return MakeCliStrategy(options);
      };
      auto attached =
          (*tp)->AttachSpeculative(factory, CliStrategyOptions(options));
      if (!attached) {
        std::fprintf(stderr, "cli: strategy attach failed (%s)\n",
                     ToString(attached.error()).data());
        return std::unexpected(attached.error());
      }
    }
    if (auto added = AddSplitRunner(
            options, std::move(*tp), group.devices.front(), report,
            [](std::unique_ptr<TensorParallel> layout, int device,
               const CalibrationConfig& calibration, KvCacheType kv_type) {
              return MakeTensorParallelRunner(std::move(layout), device,
                                              calibration, kv_type);
            },
            pool);
        !added) {
      return added;
    }
  }
  report("ready");
  return {};
}

}  // namespace

int RunServeCommand(const CliOptions& options) {
  if (options.model_path.empty()) {
    std::fprintf(stderr, "cli: --model is required\n");
    PrintUsage();
    return kExitUsage;
  }
  if (auto valid = ValidateLayoutFlags(options); !valid) {
    std::fprintf(stderr, "cli: %s\n", valid.error().c_str());
    PrintUsage();
    return kExitUsage;
  }
  WarnInertServeFlags(options);

  const std::vector<int> devices = SelectedDevices(options);
  const std::string routing_hint = DeviceRoutingHint(
      devices, DeviceRouteKind::kServeReplicas, SplitLayoutActive(options));
  if (!routing_hint.empty()) {
    std::fprintf(stderr, "%s\n", routing_hint.c_str());
  }

  ServeOptions serve_options;
  serve_options.host = options.host;
  serve_options.port = options.port;
  serve_options.default_max_completion_tokens = options.max_completion_tokens;
  serve_options.api_keys = options.api_keys;
  serve_options.allow_origins = options.allow_origins;
  serve_options.auto_title = options.auto_title;
  serve_options.stats_interval_seconds = options.stats_interval_seconds;
  serve_options.default_enable_thinking = options.enable_thinking;
  serve_options.default_reasoning_effort =
      options.reasoning_effort_set ? options.reasoning_effort : std::string();
  // The launcher's sampling values are the server defaults; a request
  // body may override any field. Without --seed every turn draws a fresh
  // seed, so two turns of the same prompt differ.
  serve_options.default_sampling = options.sampling;
  if (options.seed_set) {
    serve_options.default_seed = options.seed;
  }

  // Build the pool on the loader thread. Devices load and warm one at a
  // time, so only one model's weights are in host memory at once. The
  // server binds first, so a client reaches /health while loading.
  auto build_pool = [&](const ServeProgress& report)
      -> std::expected<std::unique_ptr<RunnerPool>, StatusCode> {
    auto pool = std::make_unique<RunnerPool>();
    if (options.pipeline_mode) {
      // A layer-split pipeline drives one model across every --gpus
      // device. The serve layer sees a single runner, so the split needs
      // no separate HTTP path; the pipeline resets its caches per turn and
      // Submit serializes turns.
      report("loading the layer split across the gpus");
      auto pipeline =
          Pipeline::Create(MakeCliPipelineOptions(options, devices));
      if (!pipeline) {
        std::fprintf(stderr, "cli: pipeline create failed (%s)\n",
                     ToString(pipeline.error()).data());
        return std::unexpected(pipeline.error());
      }
      // Attach the drafter to every sequence: a split serve runs several
      // turns at once, each with its own draft state, over one shared
      // read-only MTP head.
      if (options.speculate || !options.draft_path.empty()) {
        const auto factory = [&]() -> std::unique_ptr<SpeculativeStrategy> {
          return MakeCliStrategy(options);
        };
        auto attached = (*pipeline)->AttachSpeculative(
            factory, CliStrategyOptions(options));
        if (!attached) {
          std::fprintf(stderr, "cli: pipeline strategy attach failed (%s)\n",
                       ToString(attached.error()).data());
          return std::unexpected(attached.error());
        }
      }
      for (std::size_t stage = 0; stage < (*pipeline)->StageCount(); ++stage) {
        ApplyCliLogLevel((*pipeline)->StageEngine(stage), options);
      }
      report("warming up kernels");
      GenerateOptions warm;
      warm.max_completion_tokens = 1;
      warm.progress_every = 0;
      warm.kv_type = options.kv_type;
      auto warmed = (*pipeline)->GenerateStreaming(
          warm, [](std::uint32_t) { return true; });
      if (!warmed) {
        std::fprintf(stderr, "cli: kernel warmup failed (%s)\n",
                     ToString(warmed.error()).data());
        return std::unexpected(warmed.error());
      }
      pool->Add(std::make_unique<PipelineRunner>(
          std::move(*pipeline), devices.front(), CalibrationConfig{},
          options.kv_type));
      report("ready");
      return pool;
    }

    if (options.spec_split_mode) {
      // The target runs on the first --gpus device and the MTP draft
      // head on the second.
      report("loading the target and the draft head across the gpus");
      SpecSplitOptions split_options;
      split_options.model_path = options.model_path;
      split_options.devices = devices;
      split_options.context = options.context;
      split_options.kv_type = options.kv_type;
      split_options.draft_block = options.draft_block;
      split_options.prefill_chunk = options.prefill_chunk;
      split_options.mtp_path = options.mtp_path;
      split_options.gpu_layers = options.gpu_layers;
      auto split = SpecSplit::Create(split_options);
      if (!split) {
        std::fprintf(stderr, "cli: speculative-split create failed (%s)\n",
                     ToString(split.error()).data());
        return std::unexpected(split.error());
      }
      ApplyCliLogLevel((*split)->TargetEngine(), options);
      LogModelSummary((*split)->TargetEngine(), (*split)->TargetModel());
      const CalibrationConfig calibration = ResolveServeCalibration(
          options, (*split)->TargetEngine(), (*split)->TargetModel());
      auto runner = MakeSpecSplitRunner(std::move(*split), devices.front(),
                                        calibration, options.kv_type);
      if (auto warmed = WarmSplitRunner(*runner, calibration, options.kv_type);
          !warmed) {
        return std::unexpected(warmed.error());
      }
      pool->Add(std::move(runner));
      report("ready");
      return pool;
    }

    if (options.tp_mode && !options.hybrid_mode) {
      // Tensor-parallel across the selected devices; the FFN projections
      // split by rank and every rank runs the same forward with the rank
      // reduce.
      report("loading the tensor split across the gpus");
      TensorParallelOptions tp_options;
      tp_options.model_path = options.model_path;
      tp_options.devices = devices;
      tp_options.context = options.context;
      tp_options.kv_type = options.kv_type;
      auto tp = TensorParallel::Create(tp_options);
      if (!tp) {
        std::fprintf(stderr, "cli: tensor-parallel create failed (%s)\n",
                     ToString(tp.error()).data());
        return std::unexpected(tp.error());
      }
      if (options.speculate || !options.draft_path.empty()) {
        const auto factory = [&]() -> std::unique_ptr<SpeculativeStrategy> {
          return MakeCliStrategy(options);
        };
        auto attached =
            (*tp)->AttachSpeculative(factory, CliStrategyOptions(options));
        if (!attached) {
          std::fprintf(stderr, "cli: strategy attach failed (%s)\n",
                       ToString(attached.error()).data());
          return std::unexpected(attached.error());
        }
      }
      if (auto added = AddSplitRunner(
              options, std::move(*tp), devices.front(), report,
              [](std::unique_ptr<TensorParallel> layout, int device,
                 const CalibrationConfig& calibration, KvCacheType kv_type) {
                return MakeTensorParallelRunner(std::move(layout), device,
                                                calibration, kv_type);
              },
              *pool);
          !added) {
        return std::unexpected(added.error());
      }
      return pool;
    }

    if (options.cp_mode && !options.hybrid_mode) {
      // Context-parallel across the selected devices; the key range
      // splits by rank and each layer combines the partial attentions.
      report("loading the context split across the gpus");
      ContextParallelOptions cp_options;
      cp_options.model_path = options.model_path;
      cp_options.devices = devices;
      cp_options.context = options.context;
      cp_options.kv_type = options.kv_type;
      auto cp = ContextParallel::Create(cp_options);
      if (!cp) {
        std::fprintf(stderr, "cli: context-parallel create failed (%s)\n",
                     ToString(cp.error()).data());
        return std::unexpected(cp.error());
      }
      if (options.speculate || !options.draft_path.empty()) {
        const auto factory = [&]() -> std::unique_ptr<SpeculativeStrategy> {
          return MakeCliStrategy(options);
        };
        auto attached =
            (*cp)->AttachSpeculative(factory, CliStrategyOptions(options));
        if (!attached) {
          std::fprintf(stderr, "cli: strategy attach failed (%s)\n",
                       ToString(attached.error()).data());
          return std::unexpected(attached.error());
        }
      }
      if (auto added = AddSplitRunner(
              options, std::move(*cp), devices.front(), report,
              [](std::unique_ptr<ContextParallel> layout, int device,
                 const CalibrationConfig& calibration, KvCacheType kv_type) {
                return MakeContextParallelRunner(std::move(layout), device,
                                                 calibration, kv_type);
              },
              *pool);
          !added) {
        return std::unexpected(added.error());
      }
      return pool;
    }

    if (options.hybrid_mode) {
      if (auto loaded = LoadHybridPool(options, devices, report, *pool);
          !loaded) {
        return std::unexpected(loaded.error());
      }
      return pool;
    }

    // Default: one full replica per device, data-parallel behind the
    // least-loaded routing.
    if (auto loaded = LoadReplicaPool(options, devices, report, *pool);
        !loaded) {
      return std::unexpected(loaded.error());
    }
    return pool;
  };

  auto served = Serve(serve_options, std::move(build_pool));
  if (!served) {
    std::fprintf(stderr, "cli: serve failed (%s)\n",
                 ToString(served.error()).data());
    return kExitError;
  }
  return kExitOk;
}

}  // namespace tessera::cli
