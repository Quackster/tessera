#include "calibrate_command.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <expected>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "calibrate_parallel.hpp"
#include "tessera/calibrate_sweep.hpp"
#include "cli_common.hpp"
#include "cli_helpers.hpp"
#include "tessera/budget.hpp"
#include "tessera/calibrate.hpp"
#include "tessera/defaults.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/runner.hpp"
#include "tessera/speculative.hpp"
#include "tessera/types.hpp"

// The calibrate subcommand. It loads the model once, warms it, sweeps the
// candidate settings through the library's RunCalibrationSweep, prints one
// line per sweep point and a final report, and writes the calibration
// file. All sweep logic is in tessera/calibrate.hpp; this file is the thin
// front end (AGENTS.md, Where Things Live).

namespace tessera::cli {
namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;
// The attention split only applies at this many keys or more (mirrors
// core::detail::kSplitMinKeys; the CLI cannot include that header).
constexpr std::size_t kAttentionSplitMinKeys = 1024;

std::string FormatRate(double rate) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.2f", rate);
  return buffer;
}

// Samples the KV-only memory bound for `kv_type` and logs it: free
// memory minus a reserve over the per-token KV size. Call after a
// generate so lazily loaded weights (the draft) are resident;
// otherwise the bound misses their footprint.
[[nodiscard]] std::size_t LogMaxContext(Engine& engine, Model& loaded,
                                        KvCacheType kv_type, bool quiet) {
  log::Diagnostics& log = engine.Diagnostics();
  const DeviceMemoryInfo memory = engine.Owner().MemoryInfo();
  std::uint64_t model_bytes = 0;
  for (const auto& weight : loaded.Weights()) {
    model_bytes += weight.device->Size();
  }
  std::uint64_t available = 0;
  if (memory.free_bytes > 0) {
    available = memory.free_bytes;
  } else if (memory.total_bytes > model_bytes) {
    available = memory.total_bytes - model_bytes;
  }
  available = available > kMemoryReserveBytes ? available - kMemoryReserveBytes
                                              : 0;
  auto model_config = loaded.Config();
  std::size_t max_context =
      model_config ? MaxContextForKv(available, *model_config, kv_type) : 0;
  if (max_context < kMinReportedContext) {
    max_context = kMinReportedContext;
  }
  if (!quiet) {
    log.Info("calibrate",
             std::string("max context for ") + std::string(ToString(kv_type)) +
                 ": " + std::to_string(max_context) + " token(s) (" +
                 std::to_string(memory.free_bytes) + " free / " +
                 std::to_string(memory.total_bytes) + " total, reserve " +
                 std::to_string(kMemoryReserveBytes) + ")");
  }
  return max_context;
}

void PrintCalibrateUsage() {
  std::fprintf(
      stderr,
      "usage: tessera-cli calibrate --model <path> [--draft <dir>]\n"
      "       [--speculate] [--context <n>]\n"
      "       [--kv-f16|--kv-q8|--kv-q4|--kv-fp8]\n"
      "       [--calibration <file>] [--gpu <n>] [--prompts <file>]\n"
      "  --model <path>    a .gguf file or an MXFP4 model directory\n"
      "  --draft <dir>     DFlash2 draft checkpoint (enables the draft "
      "block sweep)\n"
      "  --speculate       MTP speculation from the target's own nextn head\n"
      "                    (enables the draft block sweep)\n"
      "  --context <n>     maximum context length (default %zu)\n"
      "  --kv-f16|--kv-q8|--kv-q4|--kv-fp8   KV cache type (default int8, "
      "kv8)\n"
       "  --calibration <file> calibration file to read and write (env "
       "TESSERA_CALIBRATION)\n"
       "  --gpu <n>         GPU index to use (default 0, the first)\n"
       "  --gpus <a,b>      parallel max-context: measure every listed GPU\n"
       "  --tp | --cp | --pipeline  parallel route for --gpus (exactly one)\n"
       "  --prompts <file>  fixed prompts, one per non-empty line\n"
       "  --calibrate-limit <n> measure at most n sweep points in total,\n"
       "                    in sweep order (default unlimited)\n"
       "  --concurrent      also benchmark 2, 3 and 4 concurrent sessions\n"
      "  --wmma|--no-wmma  fp8 tensor-core MXFP4 GEMM (diagnostic)\n"
      "  --w4a8            MXFP4 W4A8 activation quant (diagnostic)\n"
      "  --target-bf16     round projection outputs to bf16 (diagnostic)\n"
      "  --tiled-min-rows <n> | --tiled-min-cols <n>  tiled-GEMM thresholds "
      "(diagnostic)\n"
      "  --prefill-attn-pairs <n>  prefill attention work budget (diagnostic)\n"
      "  --quiet           suppress progress and info logs\n",
      kDefaultContext);
}

// Resolve the calibration file path: the flag, else TESSERA_CALIBRATION.
std::string ResolveCalibrationPath(const std::string& flag_value) {
  if (!flag_value.empty()) {
    return flag_value;
  }
  if (const char* env = std::getenv("TESSERA_CALIBRATION");
      env != nullptr && *env != '\0') {
    return env;
  }
  return {};
}

}  // namespace

int RunCalibrateCommand(int argc, char** argv) {
  std::string model_path;
  std::string draft_path;
  std::string calibration_path;
  std::string prompts_path;
  std::size_t calibrate_limit = 0;
  std::size_t context = kDefaultContext;
  // int8 (kv8) is the default KvType for calibration, the memory-efficient
  // cache the maximum context is found for.
  KvCacheType kv_type = KvCacheType::Q8;
  int gpu = 0;
  std::string gpus;
  bool quiet = false;
  bool concurrent = false;
  bool speculate = false;
  bool tp_mode = false;
  bool cp_mode = false;
  bool pipeline_mode = false;
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--draft" && i + 1 < argc) {
      draft_path = argv[++i];
    } else if (arg == "--speculate") {
      speculate = true;
    } else if (arg == "--calibration" && i + 1 < argc) {
      calibration_path = argv[++i];
    } else if (arg == "--prompts" && i + 1 < argc) {
      prompts_path = argv[++i];
    } else if (arg == "--calibrate-limit" && i + 1 < argc) {
      calibrate_limit = std::stoul(argv[++i]);
    } else if (arg == "--context" && i + 1 < argc) {
      context = std::stoul(argv[++i]);
    } else if (arg == "--gpu" && i + 1 < argc) {
      gpu = std::stoi(argv[++i]);
    } else if (arg == "--gpus" && i + 1 < argc) {
      gpus = argv[++i];
    } else if (arg == "--tp") {
      tp_mode = true;
    } else if (arg == "--cp") {
      cp_mode = true;
    } else if (arg == "--pipeline") {
      pipeline_mode = true;
    } else if (arg == "--quiet") {
      quiet = true;
    } else if (arg == "--concurrent") {
      concurrent = true;
    } else if (MatchKvType(arg, kv_type)) {
      // handled by MatchKvType
    } else if (ApplyDiagnosticFlag(arg, i, argv, argc)) {
      // handled by ApplyDiagnosticFlag (sets the diagnostic env var)
    } else {
      std::fprintf(stderr, "cli: unknown or unterminated argument '%s'\n",
                   arg.data());
      PrintCalibrateUsage();
      return kExitUsage;
    }
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "cli: calibrate requires --model\n");
    PrintCalibrateUsage();
    return kExitUsage;
  }
  const std::string calibration_file = ResolveCalibrationPath(calibration_path);
  if (calibration_file.empty()) {
    std::fprintf(stderr,
                 "cli: calibrate needs --calibration <file> or "
                 "TESSERA_CALIBRATION\n");
    return kExitUsage;
  }
  if (!gpus.empty() || tp_mode || cp_mode || pipeline_mode) {
    // A speculative draft needs the single-device engine: none of the
    // parallel layouts (--tp/--cp/--pipeline) attaches a strategy, so the
    // draft sweep stays single-device (see docs/CALIBRATE.md).
    if (!draft_path.empty() || speculate) {
      std::fprintf(stderr,
                   "cli: the parallel sweep has no speculative draft; "
                   "--draft/--speculate are single-device only\n");
      return kExitUsage;
    }
    // Parallel calibration: the speed sweep runs through the layout and the
    // entry also carries the largest KV-cache context the layout allows.
    ParallelCalibrateOptions parallel_options;
    parallel_options.model_path = model_path;
    parallel_options.gpus = gpus;
    parallel_options.tp = tp_mode;
    parallel_options.cp = cp_mode;
    parallel_options.pipeline = pipeline_mode;
    parallel_options.context = context;
    parallel_options.kv_type = kv_type;
    parallel_options.calibration_file = calibration_file;
    parallel_options.prompts_path = prompts_path;
    parallel_options.calibrate_limit = calibrate_limit;
    parallel_options.quiet = quiet;
    return RunParallelCalibrate(parallel_options);
  }
  std::vector<std::string> prompts = DefaultCalibrationPrompts();
  if (!prompts_path.empty()) {
    auto loaded_prompts = CalibrationPromptsFromFile(prompts_path);
    if (!loaded_prompts) {
      std::fprintf(stderr, "cli: cannot read prompts file '%s'\n",
                   prompts_path.c_str());
      return kExitError;
    }
    prompts = std::move(*loaded_prompts);
  }

  EngineOptions engine_options;
  engine_options.device_index = gpu;
  auto created = Engine::Create(engine_options);
  if (!created) {
    std::fprintf(stderr, "cli: engine create failed (%s)\n",
                 ToString(created.error()).data());
    return kExitError;
  }
  Engine& engine = **created;
  auto& log = engine.Diagnostics();

  // DFlash2 attaches from its checkpoint; MTP drafts from the target's own
  // nextn head, so it needs no path. Either enables the draft sweeps.
  bool draft_attached = false;
  bool dflash2_attached = false;
  if (!draft_path.empty() || speculate) {
    std::unique_ptr<SpeculativeStrategy> strategy =
        draft_path.empty() ? CreateMtpStrategy() : CreateDFlash2Strategy();
    const std::string label = draft_path.empty() ? "MTP" : "DFlash2";
    auto attach = strategy->Attach(StrategyOptions{draft_path, 0});
    if (!attach) {
      log.Error("calibrate", label + " attach failed (" +
                                 std::string(ToString(attach.error())) + ")");
      return kExitError;
    }
    auto attached = engine.AttachSpeculative(std::move(strategy));
    if (!attached) {
      log.Error("calibrate",
                std::string("strategy attach failed (") +
                    std::string(ToString(attached.error())) + ")");
      return kExitError;
    }
    draft_attached = true;
    dflash2_attached = !draft_path.empty();
  }

  const std::string strategy_label =
      dflash2_attached ? "dflash2" : (draft_attached ? "mtp" : "none");
  // Size the paging from the free device memory before the load: an MoE
  // target can exceed VRAM, and a full upload then fails. The chosen value
  // is stored in the entry so run and serve reuse it.
  std::size_t gpu_layers = 0;
  {
    const auto memory = engine.Owner().MemoryInfo();
    const std::uint64_t budget =
        memory.free_bytes != 0 ? memory.free_bytes : memory.total_bytes;
    auto advice = core::RecommendGpuLayers(model_path, context, kv_type, budget,
                                           strategy_label);
    if (!advice) {
      log.Error("calibrate", std::string("cannot size --gpu-layers for '") +
                                 model_path + "' (" +
                                 std::string(ToString(advice.error())) + ")");
      return kExitError;
    }
    if (!advice->has_best && budget != 0) {
      log.Error("calibrate",
                "no --gpu-layers value fits the " + std::to_string(budget) +
                    " byte device budget; lower --context, quantize the KV "
                    "cache, or use a smaller model");
      return kExitError;
    }
    gpu_layers = advice->best;
    if (!quiet) {
      if (gpu_layers == 0) {
        log.Info("calibrate", "paging: full upload fits (" +
                                  std::to_string(advice->full_bytes) +
                                  " resident bytes)");
      } else {
        log.Info("calibrate", "paging: --gpu-layers " +
                                  std::to_string(gpu_layers) + " (resident " +
                                  std::to_string(advice->best_bytes) +
                                  " bytes of " + std::to_string(budget) +
                                  " budget)");
      }
    }
  }
  ModelOptions model_options{model_path, context};
  model_options.gpu_layers = gpu_layers;
  auto model = engine.LoadModel(model_options);
  if (!model) {
    log.Error("calibrate", std::string("model load failed (") +
                               std::string(ToString(model.error())) + ")");
    return kExitError;
  }
  Model& loaded = **model;
  const Tokenizer* tokenizer = loaded.GetTokenizer();
  if (tokenizer == nullptr) {
    log.Error("calibrate",
              "this model has no tokenizer; calibration needs text prompts");
    return kExitError;
  }
  // The prefill prompts are the full fixed prompts; the decode prompts are
  // their first kDecodeMeasurePromptTokens tokens.
  auto encoded_prompts = EncodeCalibrationPrompts(*tokenizer, prompts);
  if (!encoded_prompts) {
    log.Error("calibrate", "prompt encode failed or produced no tokens");
    return kExitError;
  }
  const CalibrationPrompts& prompt_set = *encoded_prompts;
  const std::string backend(engine.Owner().Name());

  CalibrationConfig defaults;
  defaults.mxfp4_split_target = kDefaultMxFp4SplitTarget;
  defaults.mxfp4_split_cap = kDefaultMxFp4SplitCap;
  defaults.prefill_chunk_tokens = kDefaultPrefillChunkTokens;
  defaults.attention_split = kDefaultAttentionSplitChunks;
  if (draft_attached) {
    const SpeculativeStrategy* strategy = engine.Speculative();
    defaults.draft_tokens = strategy != nullptr ? strategy->DraftBlock() : 0;
  }

  // The single-device route: the engine streams one greedy generation.
  const TimedGenerateFn generate =
      [&](std::span<const std::uint32_t> prompt, const GenerateOptions& gen) {
        GenerateOptions options = gen;
        options.prompt_tokens.assign(prompt.begin(), prompt.end());
        return engine.GenerateStreaming(loaded, options,
                                        [](std::uint32_t) { return true; });
      };
  const MeasureFn measure = BuildSweepMeasure(generate, prompt_set, kv_type);

  // Warm the engine once before the first measurement (kernel compile and
  // the first-touch weight loads), so the sweep points measure steady state.
  // A limited run skips it: the passed count is the engine run count.
  if (calibrate_limit == 0) {
    if (auto warmed = measure(Setting::MxFp4SplitTarget, defaults, 0, true);
        !warmed) {
      log.Error("calibrate", std::string("engine warmup failed (") +
                                 std::string(ToString(warmed.error())) + ")");
      return kExitError;
    }
    // Report the bound early too: the draft is resident now, so the
    // number already reflects its footprint.
    (void)LogMaxContext(engine, loaded, kv_type, quiet);
  }

  SweepOptions sweep;
  sweep.prompt_count = prompt_set.count();
  // A limit measures at most that many points in total, whatever the
  // setting, and keeps the fastest measured value. Zero means no limit.
  sweep.max_points = calibrate_limit;
  sweep.defaults = defaults;
  sweep.sweep_split_target = backend == "rocm";
  sweep.sweep_split_cap = backend == "rocm";
  sweep.sweep_prefill_chunk = true;
  sweep.sweep_attention_split = prompt_set.longest() >= kAttentionSplitMinKeys;
  sweep.sweep_draft_tokens = draft_attached && defaults.draft_tokens > 0;
  // MTP has no draft context window; only DFlash2 sweeps that knob.
  sweep.sweep_draft_context = dflash2_attached;

  if (!quiet) {
    const std::string budget =
        sweep.max_points > 0
            ? ", max " + std::to_string(sweep.max_points) + " points"
            : "";
    log.Info("calibrate", std::string("sweeping on ") + backend + " (" +
                               std::string(engine.Owner().DeviceName()) +
                               "), " + std::to_string(sweep.prompt_count) +
                               " prompt(s)" + budget + ", greedy decode, " +
                               std::string(ToString(kv_type)) + " KV");
  }
  auto outcome = RunCalibrationSweep(sweep, measure);
  if (!outcome) {
    log.Error("calibrate", std::string("sweep failed (") +
                               std::string(ToString(outcome.error())) + ")");
    return kExitError;
  }
  for (const SweepPoint& point : outcome->points) {
    log.Info("calibrate", std::string(ToString(point.setting)) + " " +
                              std::to_string(point.value) + " = " +
                              FormatRate(point.tps) + " tok/s" +
                              (point.kept ? "  (kept)" : ""));
  }

  // The stored rates: a full run measures the chosen configuration once
  // more, while a limited run reuses the sweep points (decode settings
  // measured decode, the prefill setting measured prefill), so the passed
  // count stays the engine run count.
  double decode_tps = 0.0;
  double prefill_tps = 0.0;
  if (sweep.max_points == 0) {
    auto rates = MeasureChosenRates(measure, outcome->config, prompt_set);
    if (!rates) {
      log.Error("calibrate", std::string("final rate measurement failed (") +
                                 std::string(ToString(rates.error())) + ")");
      return kExitError;
    }
    decode_tps = rates->first;
    prefill_tps = rates->second;
  } else {
    std::vector<double> decode_rates;
    std::vector<double> prefill_rates;
    for (const SweepPoint& point : outcome->points) {
      if (point.setting == Setting::PrefillChunkTokens) {
        prefill_rates.push_back(point.tps);
      } else {
        decode_rates.push_back(point.tps);
      }
    }
    decode_tps = MedianOf(decode_rates);
    prefill_tps = MedianOf(prefill_rates);
  }

  // Report the maximum context length the device allows for this KV
  // cache type, measured after the sweep so the draft weights are
  // resident. KV-only and informational; the run used --context.
  const std::size_t max_context =
      LogMaxContext(engine, loaded, kv_type, quiet);

  const std::string device_name(engine.Owner().DeviceName());
  const std::string model_id =
      loaded.Name().empty()
          ? std::filesystem::path(model_path).filename().string()
          : std::string(loaded.Name());

  // The largest number of concurrent sessions whose KV caches at the
  // requested context fit the device (informational), and, when asked, the
  // measured throughput for 2, 3 and 4 sessions.
  std::vector<ConcurrentRate> concurrent_rates;
  std::size_t max_concurrent = 0;
  {
    const DeviceMemoryInfo memory = engine.Owner().MemoryInfo();
    std::uint64_t model_bytes = 0;
    for (const auto& weight : loaded.Weights()) {
      model_bytes += weight.device->Size();
    }
    std::uint64_t available = 0;
    if (memory.free_bytes > 0) {
      available = memory.free_bytes;
    } else if (memory.total_bytes > model_bytes) {
      available = memory.total_bytes - model_bytes;
    }
    available = available > kMemoryReserveBytes
                    ? available - kMemoryReserveBytes
                    : 0;
    auto model_config = loaded.Config();
    if (model_config) {
      max_concurrent = MaxConcurrentSessions(
          available, KvBytesPerToken(*model_config, kv_type), context);
    }
  }
  if (concurrent) {
    if (draft_attached) {
      log.Warn("calibrate",
               "concurrent benchmark skipped: the scheduler does not drive "
               "the attached draft strategy");
    } else {
      std::unique_ptr<Engine> engine_ptr = std::move(*created);
      std::unique_ptr<Model> model_ptr = std::move(*model);
      RunnerPool pool;
      pool.Add(std::make_unique<Runner>(std::move(engine_ptr),
                                        std::move(model_ptr), gpu,
                                        outcome->config, kv_type));
      for (const std::size_t sessions :
           {std::size_t{2}, std::size_t{3}, std::size_t{4}}) {
        std::vector<GenerateOptions> requests;
        requests.reserve(sessions);
        for (std::size_t s = 0; s < sessions; ++s) {
          GenerateOptions gen;
          gen.max_completion_tokens = kDecodeMeasureTokens;
          gen.kv_type = kv_type;
          gen.progress_every = 0;
          gen.prompt_tokens = prompt_set.decode[0];
          requests.push_back(std::move(gen));
        }
        auto results = RunConcurrentSessions(pool, requests);
        std::vector<std::size_t> tokens;
        std::vector<double> decode_ms;
        bool ok = true;
        for (const ConcurrentSessionResult& result : results) {
          if (!result.outcome) {
            ok = false;
            break;
          }
          tokens.push_back(result.tokens.size());
          decode_ms.push_back(result.outcome->decode_ms);
        }
        if (!ok) {
          log.Warn("calibrate",
                   "concurrent benchmark failed at " +
                       std::to_string(sessions) +
                       " session(s); reporting the rows measured so far");
          break;
        }
        concurrent_rates.push_back(
            SummarizeConcurrentSessions(sessions, tokens, decode_ms));
      }
      if (!quiet) {
        for (const ConcurrentRate& rate : concurrent_rates) {
          log.Info("calibrate",
                   std::to_string(rate.sessions) + " session(s): " +
                       FormatRate(rate.total_tps) + " tok/s total, " +
                       FormatRate(rate.per_session_tps) + " tok/s each");
        }
      }
    }
  }

  CalibrationEntry entry;
  entry.key = MakeHardwareKey(backend, device_name, model_id, context,
                              ToString(kv_type), strategy_label);
  entry.config = outcome->config;
  entry.decode_tps = decode_tps;
  entry.prefill_tps = prefill_tps;
  entry.max_context = max_context;
  entry.gpu_layers = gpu_layers;
  entry.date = TodayIso();

  if (auto saved =
          SaveCalibrationEntry(log, calibration_file, std::move(entry));
      !saved) {
    log.Error("calibrate", std::string("cannot write '") + calibration_file +
                               "' (" + std::string(ToString(saved.error())) +
                               ")");
    return kExitError;
  }

  // A readable report to stdout: identity, the sweep, the chosen settings,
  // and an example command that reproduces them. The formatter lives in the
  // library so it is unit tested.
  CalibrationReport report;
  report.backend = backend;
  report.device = device_name;
  report.model = model_id;
  report.model_path = model_path;
  report.draft_path = draft_path;
  report.kv_type = std::string(ToString(kv_type));
  report.kv_flag = KvFlag(kv_type);
  report.strategy = strategy_label;
  report.key_context = context;
  report.max_context = max_context;
  report.gpu_layers = gpu_layers;
  report.defaults = defaults;
  report.points = outcome->points;
  report.not_applicable = outcome->not_applicable;
  report.skipped = outcome->skipped;
  report.chosen = outcome->config;
  report.decode_tps = decode_tps;
  report.prefill_tps = prefill_tps;
  report.file = calibration_file;
  report.split_applicable = backend == "rocm";
  report.attention_applicable = sweep.sweep_attention_split;
  report.draft_attached = draft_attached;
  report.concurrent = concurrent_rates;
  report.max_concurrent = max_concurrent;
  std::fputs("\n", stdout);
  std::fputs(FormatCalibrationReport(report).c_str(), stdout);
  std::fflush(stdout);
  return kExitOk;
}

}  // namespace tessera::cli
