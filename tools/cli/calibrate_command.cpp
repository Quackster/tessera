#include "calibrate_command.hpp"

#include <chrono>
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

#include "cli_common.hpp"
#include "tessera/calibrate.hpp"
#include "tessera/engine.hpp"
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
// Decode tokens produced per decode measurement. Enough to average out
// launch jitter.
constexpr std::size_t kDecodeTokens = 32;
// Decode sweeps run on a short prompt: the decode rate does not depend on
// the prompt length, and prefilling a long prompt for every decode point
// would dominate the run.
constexpr std::size_t kDecodePromptTokens = 64;
// The fixed prompt set is a file-local constant. Each base line repeats to
// this many sentences so the prompt is longer than the prefill chunk half
// of the candidate list and the sweep can tell the chunks apart.
constexpr std::size_t kPromptRepeats = 64;
// Headroom kept free for scratch and fragmentation when reporting the
// maximum context length.
constexpr std::uint64_t kMemoryReserveBytes = 2ull * 1024 * 1024 * 1024;
// Below this the reported maximum context is not useful.
constexpr std::size_t kMinReportedContext = 2048;
// The whole sweep stops after this; a longer run means the sweep is too
// heavy or a kernel regressed.
constexpr int kSweepBudgetMinutes = 15;
const char* const kPromptBases[] = {
    "Explain step by step how a large language model computes self "
    "attention over its context. ",
    "Describe the difference between the prefill and the decode phase of a "
    "transformer inference engine. ",
    "Summarize how a grouped query attention cache is stored and read back "
    "on an accelerator. ",
};

// The three fixed prompts: each base line repeated to a length that
// exercises the prefill chunk candidates.
std::vector<std::string> DefaultPrompts() {
  std::vector<std::string> prompts;
  for (const char* base : kPromptBases) {
    std::string text;
    for (std::size_t i = 0; i < kPromptRepeats; ++i) {
      text += base;
    }
    prompts.push_back(std::move(text));
  }
  return prompts;
}

// Read prompts from a file: one prompt per non-empty line. A file with no
// prompt lines is rejected.
std::expected<std::vector<std::string>, bool> PromptsFromFile(
    const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    return std::unexpected(false);
  }
  std::vector<std::string> prompts;
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      prompts.push_back(line);
    }
  }
  if (prompts.empty()) {
    return std::unexpected(false);
  }
  return prompts;
}

std::string FormatRate(double rate) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.2f", rate);
  return buffer;
}

// The run flag that selects a KV cache type (empty for the fp32 default).
std::string KvFlag(KvCacheType type) {
  switch (type) {
    case KvCacheType::F16:
      return "--kv-f16 ";
    case KvCacheType::Q8:
      return "--kv-q8 ";
    case KvCacheType::Q4:
      return "--kv-q4 ";
    case KvCacheType::FP8:
      return "--kv-fp8 ";
    case KvCacheType::F32:
      return "";
  }
  return "";
}

std::string TodayIso() {
  const std::time_t now = std::time(nullptr);
  std::tm local{};
  localtime_r(&now, &local);
  char date[16];
  std::strftime(date, sizeof(date), "%Y-%m-%d", &local);
  return date;
}

void PrintCalibrateUsage() {
  std::fprintf(
      stderr,
      "usage: tessera-cli calibrate --model <path> [--draft <dir>]\n"
      "       [--context <n>] [--kv-f16|--kv-q8|--kv-q4|--kv-fp8]\n"
      "       [--calibration <file>] [--gpu <n>] [--prompts <file>]\n"
      "  --model <path>    a .gguf file or an MXFP4 model directory\n"
      "  --draft <dir>     DFlash2 draft checkpoint (enables the draft "
      "block sweep)\n"
      "  --context <n>     maximum context length (default %zu)\n"
      "  --kv-f16|--kv-q8|--kv-q4|--kv-fp8   KV cache type (default int8, "
      "kv8)\n"
      "  --calibration <file> calibration file to read and write (env "
      "TESSERA_CALIBRATION)\n"
      "  --gpu <n>         GPU index to use (default 0, the first)\n"
      "  --prompts <file>  fixed prompts, one per non-empty line\n"
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
  std::size_t context = kDefaultContext;
  // int8 (kv8) is the default KvType for calibration, the memory-efficient
  // cache the maximum context is found for.
  KvCacheType kv_type = KvCacheType::Q8;
  int gpu = 0;
  bool quiet = false;
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--draft" && i + 1 < argc) {
      draft_path = argv[++i];
    } else if (arg == "--calibration" && i + 1 < argc) {
      calibration_path = argv[++i];
    } else if (arg == "--prompts" && i + 1 < argc) {
      prompts_path = argv[++i];
    } else if (arg == "--context" && i + 1 < argc) {
      context = std::stoul(argv[++i]);
    } else if (arg == "--gpu" && i + 1 < argc) {
      gpu = std::stoi(argv[++i]);
    } else if (arg == "--quiet") {
      quiet = true;
    } else if (MatchKvType(arg, kv_type)) {
      // handled by MatchKvType
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
  std::vector<std::string> prompts = DefaultPrompts();
  if (!prompts_path.empty()) {
    auto loaded_prompts = PromptsFromFile(prompts_path);
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

  bool draft_attached = false;
  if (!draft_path.empty()) {
    auto strategy = CreateDFlash2Strategy();
    auto attach = strategy->Attach(StrategyOptions{draft_path, 0});
    if (!attach) {
      log.Error("calibrate", std::string("draft attach failed (") +
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
  }

  auto model = engine.LoadModel(ModelOptions{model_path, context});
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
  // their first kDecodePromptTokens tokens.
  std::vector<std::vector<std::uint32_t>> prefill_ids;
  std::vector<std::vector<std::uint32_t>> decode_ids;
  for (const std::string& prompt : prompts) {
    auto ids = tokenizer->Encode(prompt);
    if (!ids || ids->size() < 2) {
      log.Error("calibrate", "prompt encode failed or produced no tokens");
      return kExitError;
    }
    std::vector<std::uint32_t> short_ids = *ids;
    if (short_ids.size() > kDecodePromptTokens) {
      short_ids.resize(kDecodePromptTokens);
    }
    prefill_ids.push_back(std::move(*ids));
    decode_ids.push_back(std::move(short_ids));
  }
  const std::string backend(engine.Owner().Name());
  const std::string strategy_label = draft_attached ? "dflash2" : "none";

  // Report the maximum context length the device allows for this KV cache
  // type: free memory (or total memory minus the weights) minus a reserve,
  // divided by the per-token KV size. Informational; the run's context is
  // the configured --context.
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

  CalibrationConfig defaults;
  defaults.mxfp4_split_target = kDefaultMxFp4SplitTarget;
  defaults.prefill_chunk_tokens = kDefaultPrefillChunkTokens;
  if (draft_attached) {
    const SpeculativeStrategy* strategy = engine.Speculative();
    defaults.draft_tokens = strategy != nullptr ? strategy->DraftBlock() : 0;
  }

  const auto sweep_started = std::chrono::steady_clock::now();
  bool over_budget = false;
  const auto measure =
      [&](Setting setting, const CalibrationConfig& config,
          std::size_t prompt, bool warmup)
      -> std::expected<Measurement, StatusCode> {
    // Cap the whole sweep: a longer run means the sweep is too heavy or a
    // kernel regressed, so stop instead of waiting.
    if (std::chrono::steady_clock::now() - sweep_started >
        std::chrono::minutes(kSweepBudgetMinutes)) {
      if (!over_budget) {
        over_budget = true;
        log.Error("calibrate",
                  "sweep exceeded the " +
                      std::to_string(kSweepBudgetMinutes) +
                      " minute cap; the sweep is too heavy or a kernel "
                      "regressed");
      }
      return std::unexpected(StatusCode::DeviceError);
    }
    const bool decode = setting != Setting::PrefillChunkTokens;
    GenerateOptions gen;
    gen.max_completion_tokens = warmup ? 1 : (decode ? kDecodeTokens : 1);
    gen.sample = false;
    gen.progress_every = 0;
    gen.kv_type = kv_type;
    gen.prefill_chunk_tokens = config.prefill_chunk_tokens;
    gen.draft_tokens = config.draft_tokens;
    gen.mxfp4_split_target = config.mxfp4_split_target;
    gen.prompt_tokens = decode ? decode_ids[prompt] : prefill_ids[prompt];
    auto outcome =
        engine.GenerateStreaming(loaded, gen, [](std::uint32_t) { return true; });
    if (!outcome) {
      return std::unexpected(outcome.error());
    }
    Measurement measured;
    if (outcome->produced > 0 && outcome->decode_ms > 0.0) {
      measured.decode_tps =
          static_cast<double>(outcome->produced) * 1000.0 / outcome->decode_ms;
    }
    if (outcome->prompt_tokens > 0 && outcome->prefill_ms > 0.0) {
      measured.prefill_tps = static_cast<double>(outcome->prompt_tokens) *
                             1000.0 / outcome->prefill_ms;
    }
    return measured;
  };

  // Warm the engine once before the first measurement (kernel compile and
  // the first-touch weight loads), so the sweep points measure steady state.
  if (auto warmed = measure(Setting::MxFp4SplitTarget, defaults, 0, true);
      !warmed) {
    log.Error("calibrate", std::string("engine warmup failed (") +
                               std::string(ToString(warmed.error())) + ")");
    return kExitError;
  }

  SweepOptions sweep;
  sweep.prompt_count = prefill_ids.size();
  sweep.defaults = defaults;
  sweep.sweep_split_target = backend == "rocm";
  sweep.sweep_prefill_chunk = true;
  sweep.sweep_draft_tokens = draft_attached && defaults.draft_tokens > 0;

  if (!quiet) {
    log.Info("calibrate", std::string("sweeping on ") + backend + " (" +
                              std::string(engine.Owner().DeviceName()) +
                              "), " + std::to_string(sweep.prompt_count) +
                              " prompt(s), greedy decode, " +
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

  // Measure the chosen configuration once more for the stored rates: the
  // decode rate on the short prompts, the prefill rate on the long ones.
  std::vector<double> decode_rates;
  std::vector<double> prefill_rates;
  for (std::size_t prompt = 0; prompt < prefill_ids.size(); ++prompt) {
    auto decode_measured =
        measure(Setting::MxFp4SplitTarget, outcome->config, prompt, false);
    if (!decode_measured) {
      log.Error("calibrate", std::string("final decode measurement failed (") +
                                 std::string(ToString(decode_measured.error())) +
                                 ")");
      return kExitError;
    }
    decode_rates.push_back(decode_measured->decode_tps);
    auto prefill_measured =
        measure(Setting::PrefillChunkTokens, outcome->config, prompt, false);
    if (!prefill_measured) {
      log.Error("calibrate",
                std::string("final prefill measurement failed (") +
                    std::string(ToString(prefill_measured.error())) + ")");
      return kExitError;
    }
    prefill_rates.push_back(prefill_measured->prefill_tps);
  }

  CalibrationEntry entry;
  entry.key =
      MakeHardwareKey(backend, engine.Owner().DeviceName(),
                      loaded.Name().empty()
                          ? std::filesystem::path(model_path).filename().string()
                          : std::string(loaded.Name()),
                      context, ToString(kv_type), strategy_label);
  entry.config = outcome->config;
  entry.decode_tps = MedianOf(decode_rates);
  entry.prefill_tps = MedianOf(prefill_rates);
  entry.date = TodayIso();
  // Keep the identity for the report: `entry` is moved into the file below.
  const std::string model_id = entry.key.model;

  CalibrationFile file;
  auto existing = CalibrationFile::Load(calibration_file);
  if (existing) {
    file = std::move(*existing);
  } else if (existing.error() != StatusCode::FileNotFound) {
    log.Warn("calibrate", std::string("ignoring malformed calibration file '") +
                              calibration_file + "' (" +
                              std::string(ToString(existing.error())) + ")");
  }
  file.Set(std::move(entry));
  auto saved = file.Save(calibration_file);
  if (!saved) {
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
  report.device = std::string(engine.Owner().DeviceName());
  report.model = model_id;
  report.model_path = model_path;
  report.kv_type = std::string(ToString(kv_type));
  report.kv_flag = KvFlag(kv_type);
  report.strategy = strategy_label;
  report.key_context = context;
  report.max_context = max_context;
  report.defaults = defaults;
  report.points = outcome->points;
  report.not_applicable = outcome->not_applicable;
  report.chosen = outcome->config;
  report.decode_tps = MedianOf(decode_rates);
  report.prefill_tps = MedianOf(prefill_rates);
  report.file = calibration_file;
  report.split_applicable = backend == "rocm";
  report.draft_attached = draft_attached;
  std::fputs("\n", stdout);
  std::fputs(FormatCalibrationReport(report).c_str(), stdout);
  std::fflush(stdout);
  return kExitOk;
}

}  // namespace tessera::cli
