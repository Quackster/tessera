#include "calibrate_parallel.hpp"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cli_common.hpp"
#include "tessera/calibrate.hpp"
#include "tessera/context_parallel.hpp"
#include "tessera/engine.hpp"
#include "tessera/log.hpp"
#include "tessera/model.hpp"
#include "tessera/pipeline.hpp"
#include "tessera/tensor_parallel.hpp"
#include "tessera/tokenizer.hpp"
#include "tessera/types.hpp"

// Parallel max-context calibration (see calibrate_parallel.hpp). The
// speed sweep stays single-device: its knobs have no per-rank form, so a
// parallel run measures the memory bound only.

namespace tessera::cli {
namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;
// Validation generate length: enough to run every rank through one full
// step without timing anything.
constexpr std::size_t kValidateTokens = 8;

// Join device indices for logs, keys and the report's example command.
std::string JoinDevices(const std::vector<int>& devices) {
  std::string joined;
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (i > 0) {
      joined += ",";
    }
    joined += std::to_string(devices[i]);
  }
  return joined;
}

// One rank's measured budget: resident weight bytes, free bytes and the
// largest full-attention context those free bytes allow.
struct RankBudget {
  std::uint64_t weight_bytes = 0;
  std::uint64_t free_bytes = 0;
  std::size_t context = 0;
};

std::expected<RankBudget, StatusCode> MeasureRank(
    Engine& engine, Model& model, const TransformerConfig& config,
    KvCacheType kv_type) {
  RankBudget budget;
  for (const auto& weight : model.Weights()) {
    budget.weight_bytes += weight.device->Size();
  }
  const DeviceMemoryInfo memory = engine.Owner().MemoryInfo();
  budget.free_bytes = memory.free_bytes;
  std::uint64_t available = 0;
  if (memory.free_bytes > 0) {
    available = memory.free_bytes;
  } else if (memory.total_bytes > budget.weight_bytes) {
    available = memory.total_bytes - budget.weight_bytes;
  }
  available =
      available > kMemoryReserveBytes ? available - kMemoryReserveBytes : 0;
  budget.context = MaxContextForKv(available, config, kv_type);
  return budget;
}

}  // namespace

std::string TodayIso() {
  const std::time_t now = std::time(nullptr);
  std::tm local{};
  localtime_r(&now, &local);
  char date[16];
  std::strftime(date, sizeof(date), "%Y-%m-%d", &local);
  return date;
}

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

std::expected<void, StatusCode> SaveCalibrationEntry(
    log::Diagnostics& log, const std::string& path, CalibrationEntry entry) {
  CalibrationFile file;
  auto existing = CalibrationFile::Load(path);
  if (existing) {
    file = std::move(*existing);
  } else if (existing.error() != StatusCode::FileNotFound) {
    log.Warn("calibrate", std::string("ignoring malformed calibration file '") +
                              path + "' (" +
                              std::string(ToString(existing.error())) + ")");
  }
  file.Set(std::move(entry));
  return file.Save(path);
}

int RunParallelCalibrate(const ParallelCalibrateOptions& options) {
  const std::vector<int> devices = ParseGpuList(options.gpus);
  const int routes = (options.tp ? 1 : 0) + (options.cp ? 1 : 0) +
                     (options.pipeline ? 1 : 0);
  if (devices.size() < 2 || routes != 1) {
    std::fprintf(stderr, "cli: calibrate --gpus needs at least two devices "
                         "and exactly one of --tp, --cp, --pipeline\n");
    return kExitUsage;
  }
  const std::string route = options.tp ? "tp" : (options.cp ? "cp" : "pipe");
  const std::string route_flag =
      options.tp ? "--tp" : (options.cp ? "--cp" : "--pipeline");
  const std::string label = route + std::to_string(devices.size());
  const std::string device_list = JoinDevices(devices);
  auto probed = Model::ProbeConfig(options.model_path);
  if (!probed) {
    std::fprintf(stderr, "cli: cannot probe model '%s' (%s)\n",
                 options.model_path.c_str(),
                 std::string(ToString(probed.error())).c_str());
    return kExitError;
  }
  const TransformerConfig& config = *probed;
  // Build the route; every rank loads sequentially inside Create, so only
  // one model is in host memory at once.
  std::unique_ptr<TensorParallel> tp;
  std::unique_ptr<ContextParallel> cp;
  std::unique_ptr<Pipeline> pipe;
  if (options.tp) {
    TensorParallelOptions tp_options;
    tp_options.model_path = options.model_path;
    tp_options.devices = devices;
    tp_options.context = options.context;
    tp_options.kv_type = options.kv_type;
    auto built = TensorParallel::Create(tp_options);
    if (!built) {
      std::fprintf(stderr, "cli: tensor-parallel create failed (%s)\n",
                   std::string(ToString(built.error())).c_str());
      return kExitError;
    }
    tp = std::move(*built);
  } else if (options.cp) {
    ContextParallelOptions cp_options;
    cp_options.model_path = options.model_path;
    cp_options.devices = devices;
    cp_options.context = options.context;
    cp_options.kv_type = options.kv_type;
    auto built = ContextParallel::Create(cp_options);
    if (!built) {
      std::fprintf(stderr, "cli: context-parallel create failed (%s)\n",
                   std::string(ToString(built.error())).c_str());
      return kExitError;
    }
    cp = std::move(*built);
  } else {
    PipelineOptions pipe_options;
    pipe_options.model_path = options.model_path;
    pipe_options.devices = devices;
    pipe_options.context = options.context;
    pipe_options.kv_type = options.kv_type;
    auto built = Pipeline::Create(pipe_options);
    if (!built) {
      std::fprintf(stderr, "cli: pipeline create failed (%s)\n",
                   std::string(ToString(built.error())).c_str());
      return kExitError;
    }
    pipe = std::move(*built);
  }
  // Measure each rank: resident bytes and free bytes on its own device.
  const std::size_t ranks =
      tp != nullptr ? tp->RankCount()
                    : (cp != nullptr ? cp->RankCount() : pipe->StageCount());
  std::vector<RankBudget> budgets;
  budgets.reserve(ranks);
  for (std::size_t r = 0; r < ranks; ++r) {
    Engine& engine = tp != nullptr  ? tp->RankEngine(r)
                     : cp != nullptr ? cp->RankEngine(r)
                                     : pipe->StageEngine(r);
    Model& model = tp != nullptr  ? tp->RankModel(r)
                   : cp != nullptr ? cp->RankModel(r)
                                   : pipe->StageModel(r);
    auto budget = MeasureRank(engine, model, config, options.kv_type);
    if (!budget) {
      std::fprintf(stderr, "cli: rank %zu measurement failed (%s)\n", r,
                   std::string(ToString(budget.error())).c_str());
      return kExitError;
    }
    budgets.push_back(*budget);
  }
  // Combine the per-device budgets. Tensor replicates the KV cache, so
  // the smallest rank wins; context splits the keys, so each rank holds
  // 1/world of them; pipeline stages hold their layer share of the keys.
  std::size_t max_context = 0;
  if (options.tp) {
    for (const RankBudget& budget : budgets) {
      max_context = max_context == 0 ? budget.context
                                     : std::min(max_context, budget.context);
    }
  } else if (options.cp) {
    for (const RankBudget& budget : budgets) {
      max_context = max_context == 0 ? budget.context
                                     : std::min(max_context, budget.context);
    }
    max_context *= ranks;
  } else {
    const auto ranges = pipe->StageRanges();
    for (std::size_t s = 0; s < ranks; ++s) {
      const std::size_t stage_layers = ranges[s].second - ranges[s].first;
      if (stage_layers == 0 || config.layers == 0) {
        std::fprintf(stderr, "cli: stage %zu owns no layers\n", s);
        return kExitError;
      }
      const std::size_t stage_context =
          budgets[s].context * config.layers / stage_layers;
      max_context = max_context == 0 ? stage_context
                                     : std::min(max_context, stage_context);
    }
  }
  if (max_context < kMinReportedContext) {
    max_context = kMinReportedContext;
  }
  // One short validation generate through the route proves every rank
  // runs; the parallel run calibrates the memory bound, not speed.
  const Tokenizer* tokenizer = tp != nullptr  ? tp->GetTokenizer()
                               : cp != nullptr ? cp->GetTokenizer()
                                               : pipe->GetTokenizer();
  if (tokenizer == nullptr) {
    std::fprintf(stderr, "cli: this model has no tokenizer\n");
    return kExitError;
  }
  auto hello = tokenizer->Encode("Hello");
  if (!hello) {
    std::fprintf(stderr, "cli: prompt encode failed\n");
    return kExitError;
  }
  const bool validated =
      tp != nullptr   ? tp->Generate(*hello, kValidateTokens).has_value()
      : cp != nullptr ? cp->Generate(*hello, kValidateTokens).has_value()
                      : pipe->Generate(*hello, kValidateTokens).has_value();
  if (!validated) {
    std::fprintf(stderr, "cli: validation generate failed\n");
    return kExitError;
  }
  Engine& first_engine = tp != nullptr  ? tp->RankEngine(0)
                         : cp != nullptr ? cp->RankEngine(0)
                                         : pipe->StageEngine(0);
  auto& log = first_engine.Diagnostics();
  if (!options.quiet) {
    for (std::size_t r = 0; r < ranks; ++r) {
      log.Info("calibrate",
               "rank " + std::to_string(r) + ": " +
                   std::to_string(budgets[r].weight_bytes) +
                   " weight bytes, " + std::to_string(budgets[r].free_bytes) +
                   " free bytes");
    }
    log.Info("calibrate",
             std::string("max context for ") +
                 std::string(ToString(options.kv_type)) + " (" + label +
                 "): " + std::to_string(max_context) + " token(s)");
  }
  const std::string backend(first_engine.Owner().Name());
  const std::string device_name(first_engine.Owner().DeviceName());
  Model& first_model = tp != nullptr  ? tp->RankModel(0)
                       : cp != nullptr ? cp->RankModel(0)
                                       : pipe->StageModel(0);
  CalibrationEntry entry;
  entry.key = MakeHardwareKey(
      backend, device_list,
      first_model.Name().empty()
          ? std::filesystem::path(options.model_path).filename().string()
          : std::string(first_model.Name()),
      options.context, ToString(options.kv_type), "none", label);
  entry.max_context = max_context;
  entry.date = TodayIso();
  if (auto saved = SaveCalibrationEntry(log, options.calibration_file,
                                        std::move(entry));
      !saved) {
    log.Error("calibrate",
              std::string("cannot write '") + options.calibration_file +
                  "' (" + std::string(ToString(saved.error())) + ")");
    return kExitError;
  }
  CalibrationReport report;
  report.backend = backend;
  report.device = device_name + " x" + std::to_string(ranks);
  report.model = first_model.Name().empty()
                     ? std::filesystem::path(options.model_path).filename()
                           .string()
                     : std::string(first_model.Name());
  report.model_path = options.model_path;
  report.kv_type = std::string(ToString(options.kv_type));
  report.kv_flag = KvFlag(options.kv_type);
  report.strategy = "none";
  report.parallel = label;
  report.parallel_flags = route_flag + " --gpus " + device_list;
  report.key_context = options.context;
  report.max_context = max_context;
  report.file = options.calibration_file;
  std::fputs("\n", stdout);
  std::fputs(FormatCalibrationReport(report).c_str(), stdout);
  std::fflush(stdout);
  return kExitOk;
}

}  // namespace tessera::cli
