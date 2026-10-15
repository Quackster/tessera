#include "cli_helpers.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "cli_common.hpp"
#include "tessera/budget.hpp"
#include "tessera/calibrate.hpp"
#include "tessera/defaults.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/runner.hpp"
#include "tessera/serve.hpp"
#include "tessera/types.hpp"

namespace tessera::cli {
namespace {

// How many tensor names to print when summarizing a model.
constexpr std::size_t kPrintedTensorNames = 5;

// The hardware key for a loaded or probed model: the runtime backend and
// device, the model identity, the context, the KV type and the strategy.
// One canonical build for the speed-knob lookup and the pre-load paging
// lookup, so both resolve the same entry.
HardwareKey KeyFor(Engine& engine, std::string_view model_id,
                   std::size_t context, KvCacheType kv_type,
                   std::string_view strategy) {
  return MakeHardwareKey(engine.Owner().Name(), engine.Owner().DeviceName(),
                         model_id, context, ToString(kv_type), strategy);
}

// The sampler a model's card recommends, keyed by general.architecture and
// the thinking mode. nullopt when the architecture has no recommendation,
// so greedy stays the default. A thinking model loops under greedy decode,
// which is why qwen4exp supplies one.
std::optional<SamplingOptions> RecommendedSampling(std::string_view arch,
                                                   bool thinking) {
  if (arch == "qwen4exp") {
    SamplingOptions s;
    s.top_k = 20;
    s.min_p = 0.0f;
    s.repetition_penalty = 1.0f;
    if (thinking) {
      s.temperature = 1.0f;
      s.top_p = 0.95f;
      s.presence_penalty = 0.0f;
    } else {
      s.temperature = 0.7f;
      s.top_p = 0.80f;
      s.presence_penalty = 1.5f;
    }
    return s;
  }
  return std::nullopt;
}

}  // namespace

bool ApplyRecommendedSampling(const Model& model, bool thinking,
                              bool explicit_sampling,
                              SamplingOptions* sampling) {
  if (explicit_sampling) {
    return false;
  }
  const Architecture* arch = model.Arch();
  if (arch == nullptr) {
    return false;
  }
  auto recommended = RecommendedSampling(arch->Name(), thinking);
  if (!recommended) {
    return false;
  }
  *sampling = *recommended;
  return true;
}

void LogModelSummary(Engine& engine, Model& model) {
  const auto& tensors = model.Tensors();
  std::size_t total_numel = 0;
  for (const auto& tensor : tensors) {
    total_numel += tensor.shape.Numel();
  }
  std::size_t device_bytes = 0;
  for (const auto& weight : model.Weights()) {
    device_bytes += weight.device->Size();
  }
  std::string summary =
      std::string(model.Format() == ModelFormat::Gguf ? "gguf" : "mxfp4") +
      " model, " + std::to_string(tensors.size()) + " tensors, " +
      std::to_string(total_numel) + " total elements, " +
      std::to_string(device_bytes) + " bytes on device";
  if (!model.Name().empty()) {
    summary = std::string(model.Name()) + " (" + summary + ")";
  }
  auto& log = engine.Diagnostics();
  log.Info("cli", summary);
  if (!model.Paged().empty()) {
    std::size_t paged_bytes = 0;
    for (const auto& paged : model.Paged()) {
      auto sized = TensorBytes(paged.manifest.dtype,
                               paged.manifest.shape.Numel());
      if (sized) {
        paged_bytes += *sized;
      }
    }
    log.Info("cli", "paged weights: " + std::to_string(model.Paged().size()) +
                        " tensors, " + std::to_string(paged_bytes) +
                        " bytes in RAM, rest on device");
  }
  for (std::size_t i = 0; i < tensors.size() && i < kPrintedTensorNames; ++i) {
    log.Info("cli", "tensor " + std::to_string(i) + ": " + tensors[i].name +
                        " [" + std::to_string(tensors[i].shape.Numel()) + "]");
  }
}

void PrintUsage() {
  std::fprintf(stderr,
               "usage: tessera-cli run --model <path> [--draft <dir>]\n"
               "       [--context <n>] [--draft-block <n>]\n"
               "       [--prefill-chunk <n>]\n"
                "       [--gpus <a,b>] [--pipeline] [--tp] [--cp]\n"
                "       [--speculate] [--spec-split] [--hybrid] [--dp <n>]\n"
               "       --prompt-text <str> [--max-completion-tokens <n>]\n"
               "       tessera-cli serve --model <path> [--host <ip>] "
               "[--port <n>]\n"
               "       tessera-cli calibrate --model <path> [--draft <dir>]\n"
               "       [--context <n>] [--calibration <file>] [--gpu <n>]\n"
               "       [--prompts <file>]\n"
               "       tessera-cli fetch-mtp --out <dir> [inventory|fetch|"
               "verify]\n"
               "       tessera-cli --list-gpus\n"
               "  --list-gpus       list the GPUs the backend sees and exit\n"
               "  --model <path>    a .gguf file or an MXFP4 model directory\n"
               "  --draft <dir>     DFlash2 draft checkpoint directory\n"
               "  --mtp <dir>       MTP head directory (tessera-cli fetch-mtp)\n"
                "  --context <n>     maximum context length (default %zu)\n"
                "  --gpu <n>         GPU index to use (default 0, the first)\n"
                "  --gpu-layers <n|all> layers kept in VRAM (0 uploads all, "
                "default 0)\n"
                "  --gpu-layers help   size the model for --gpu-layers "
                "without loading it\n"
               "  --gpus <a,b>      serve one model replica per GPU for "
               "parallel requests\n"
               "  --pipeline        run the model split across --gpus devices "
                "(layer pipeline)\n"
                "  --calibration <file> calibrated settings to apply (env "
               "TESSERA_CALIBRATION)\n"
               "  --draft-block <n> draft block tokens (0 = checkpoint "
               "default, %zu)\n"
               "  --prefill-chunk <n> prefill tokens per forward (0 = auto, "
               "%zu)\n"
               "  --split-target <n> fp8 MXFP4 split-K workgroup target (0 = "
               "built-in, ROCm only)\n"
               "  --split-cap <n>    fp8 MXFP4 split-K factor cap (0 = built-in, "
               "ROCm only)\n"
               "  --attention-split <n> single-token flash-decoding chunks (0 = "
               "built-in, %zu)\n"
               "  --draft-context <n> DFlash2 draft context window in rows (0 = "
               "checkpoint)\n"
               "  --wmma | --no-wmma   fp8 tensor-core MXFP4 GEMM (default on; "
               "diagnostic)\n"
               "  --w4a8             MXFP4 W4A8 activation quant (diagnostic)\n"
               "  --target-bf16      round projection outputs to bf16 "
               "(diagnostic)\n"
               "  --tiled-min-rows <n> | --tiled-min-cols <n>  tiled-GEMM "
               "dispatch thresholds (diagnostic)\n"
               "  --prefill-attn-pairs <n>  prefill attention work budget "
               "(diagnostic)\n"
                "  --prompt-text <s> text prompt, repeatable for parallel\n"
                "                sessions (tokenized; needs a tokenizer)\n"
               "  --max-completion-tokens <n> completion tokens (0 fills the "
               "remaining context, default %zu)\n"
               "  --max-thinking-tokens <n> think-block budget per turn (0 "
               "leaves thinking unlimited, default 0)\n"
               "  --enable-thinking   render the chat template with thinking "
               "enabled (default off)\n"
               "  --reasoning-effort <level> thinking difficulty for the chat "
               "template (for example xhigh; ignored with --no-chat)\n"
               "  --speculate       draft with the MTP head (run and serve)\n"
               "  --mmproj <path>   vision projector (mmproj) GGUF\n"
               "  --image <path>    image (binary PPM) to prepend as tokens\n"
               "  --quiet           suppress progress and info logs\n"
               "  --verbose         enable debug logs (weight transfers, "
               "cache probes)\n"
               "  --no-chat         do not apply the chat template\n"
               "  --sample          sample instead of greedy decode\n"
               "  --temperature <f> sampling temperature (default %.1f)\n"
               "  --top-p <f>       nucleus probability (default %.2f)\n"
               "  --top-k <n>       keep the top n tokens (default %d)\n"
               "  --min-p <f>       minimum probability (default %.1f)\n"
               "  --presence-penalty <f>   presence penalty (default %.1f)\n"
               "  --repetition-penalty <f> repetition penalty (default %.1f)\n"
               "  --seed <n>        sampling RNG seed (default 0)\n"
               "  --kv-f16          store the KV cache in fp16 (default fp32)\n"
               "  --kv-q8           store the KV cache in int8\n"
               "  --kv-q4           store the KV cache in 4-bit\n"
               "  --kv-fp8          store the KV cache in FP8 E4M3\n"
                "  --host <ip>       serve bind address (default %s)\n"
                "  --port <n>        serve port (default %u)\n"
                "  --stats-interval <s> live tok/s cadence on streams and "
                "console (default %zu, 0 off)\n"
               "  --no-auto-title   keep the first user line as the chat title "
               "instead of asking the model\n"
               "  --api-key <k>     accepted API key (repeatable; env "
               "TESSERA_API_KEY)\n"
               "  --allow-origin <o> CORS origin (repeatable; * allows all)\n",
               kDefaultContext, kDefaultDraftBlock, kDefaultPrefillChunk,
               kDefaultAttentionSplitChunks, kDefaultMaxCompletionTokens,
               kDefaultTemperature, kDefaultTopP, kDefaultTopK, kDefaultMinP,
               kDefaultPresencePenalty, kDefaultRepetitionPenalty,
               kDefaultServeHost, static_cast<unsigned>(kDefaultServePort),
               kDefaultStatsIntervalSeconds);
}

CalibrationConfig ResolveCalibration(Engine& engine, Model& model,
                                     std::size_t context, KvCacheType kv_type,
                                     std::string_view strategy,
                                     const std::string& path,
                                     const CalibrationConfig& explicit_config) {
  auto& log = engine.Diagnostics();
  std::string model_id(model.Name());
  if (model_id.empty()) {
    model_id = std::filesystem::path(model.Path()).filename().string();
  }
  const HardwareKey key = KeyFor(engine, model_id, context, kv_type, strategy);
  auto file = CalibrationFile::Load(path);
  if (!file) {
    if (file.error() != StatusCode::FileNotFound) {
      log.Warn("cli", std::string("ignoring malformed calibration file '") +
                          path + "' (" + std::string(ToString(file.error())) +
                          ")");
    }
    return explicit_config;
  }
  const CalibrationEntry* entry = file->Find(key);
  if (entry == nullptr) {
    log.Info("cli", "no calibration entry for this machine and model; "
                    "using the explicit values");
    return explicit_config;
  }
  const CalibrationConfig merged =
      ApplySavedCalibration(explicit_config, *entry);
  log.Info("cli", std::string("applied calibration: mxfp4_split_target=") +
                      std::to_string(merged.mxfp4_split_target) +
                      " prefill_chunk_tokens=" +
                      std::to_string(merged.prefill_chunk_tokens) +
                      " draft_tokens=" + std::to_string(merged.draft_tokens));
  return merged;
}

std::size_t ResolveCalibrationGpuLayers(Engine& engine,
                                        const std::string& model_path,
                                        std::size_t context,
                                        KvCacheType kv_type,
                                        std::string_view strategy,
                                        const std::string& path,
                                        std::size_t explicit_layers) {
  if (path.empty()) {
    return explicit_layers;
  }
  auto name = Model::ProbeName(model_path);
  if (!name) {
    // The upcoming load reports the real error; paging stays explicit.
    return explicit_layers;
  }
  std::string model_id = *name;
  if (model_id.empty()) {
    model_id = std::filesystem::path(model_path).filename().string();
  }
  const HardwareKey key = KeyFor(engine, model_id, context, kv_type, strategy);
  auto file = CalibrationFile::Load(path);
  if (!file) {
    return explicit_layers;
  }
  const CalibrationEntry* entry = file->Find(key);
  if (entry == nullptr) {
    return explicit_layers;
  }
  return ApplySavedGpuLayers(explicit_layers, *entry);
}

namespace {

// Whole MiB, the unit the serve pre-check reports free memory in.
std::uint64_t ToMiB(std::uint64_t bytes) { return bytes / (1024 * 1024); }

}  // namespace

int LogGpuLayersAdvice(Engine& engine, const std::string& model_path,
                       std::size_t context, KvCacheType kv_type,
                       std::string_view strategy) {
  auto& log = engine.Diagnostics();
  const DeviceMemoryInfo memory = engine.Owner().MemoryInfo();
  // Vulkan reports the device total but no free figure; size against the
  // total then (an upper bound), and say so below.
  const bool budget_is_total = memory.free_bytes == 0;
  const std::uint64_t budget =
      budget_is_total ? memory.total_bytes : memory.free_bytes;
  auto advice = core::RecommendGpuLayers(model_path, context, kv_type, budget,
                                         std::string(strategy));
  if (!advice) {
    log.Warn("cli", std::string("cannot size --gpu-layers for '") + model_path +
                        "' (" + std::string(ToString(advice.error())) +
                        "); check the path and file format");
    return 1;
  }
  advice->budget_is_total = budget_is_total;
  const std::string model_id =
      std::filesystem::path(model_path).filename().string();
  log.Info("cli", model_id + ": " + std::to_string(advice->trunk_layers) +
                      " trunk blocks plus the output head (" +
                      std::to_string(advice->total_layers) +
                      " offloadable layers, 0 keeps every layer)");
  log.Info("cli",
           "context " + std::to_string(context) + ", kv " +
               std::string(ToString(kv_type)) + ": cache about " +
               std::to_string(ToMiB(advice->kv_bytes)) + " MiB (" +
               std::to_string(advice->full_layers) + " full, " +
               std::to_string(advice->linear_layers) + " linear layers)");
  log.Info("cli", "weights: full upload " +
                      std::to_string(ToMiB(advice->full_bytes)) + " MiB");
  const std::string budget_label =
      std::to_string(ToMiB(advice->free_bytes)) +
      (advice->budget_is_total ? " MiB device total (free unknown on this "
                                 "backend; other processes may hold memory)"
                               : " MiB free");
  if (!advice->has_best) {
    if (advice->free_bytes == 0) {
      log.Info("cli", "the backend reports no memory size; size "
                      "--gpu-layers by hand");
    } else {
      log.Info("cli", "no layer count fits " + budget_label +
                          "; lower --context, quantize the KV "
                          "cache, or use a smaller model");
    }
    return 0;
  }
  if (advice->best == 0) {
    log.Info("cli", "best --gpu-layers 0: the full upload fits " +
                        budget_label + "; no paging needed");
  } else {
    log.Info("cli", "best --gpu-layers " + std::to_string(advice->best) +
                        ": resident " +
                        std::to_string(ToMiB(advice->best_bytes)) +
                        " MiB plus " +
                        std::to_string(ToMiB(advice->staging_bytes)) +
                        " MiB staging of " + budget_label);
  }
  return 0;
}

bool CheckGpuLayersFit(Engine& engine, const std::string& model_path,
                       std::size_t context, KvCacheType kv_type,
                       std::string_view strategy, std::size_t gpu_layers,
                       int device) {
  auto& log = engine.Diagnostics();
  const DeviceMemoryInfo memory = engine.Owner().MemoryInfo();
  // Vulkan reports the device total but no free figure; judge against
  // the total then (a load that fails it can never fit).
  const bool budget_is_total = memory.free_bytes == 0;
  const std::uint64_t budget =
      budget_is_total ? memory.total_bytes : memory.free_bytes;
  if (budget == 0) {
    return true;
  }
  auto fit = core::FitGpuLayers(model_path, context, kv_type, budget,
                                std::string(strategy), gpu_layers);
  if (!fit) {
    log.Warn("cli", std::string("cannot size --gpu-layers for '") +
                        model_path + "' (" +
                        std::string(ToString(fit.error())) +
                        "); check the path and file format");
    return false;
  }
  fit->budget_is_total = budget_is_total;
  if (fit->fits) {
    return true;
  }
  const std::string budget_label =
      std::to_string(ToMiB(fit->budget_bytes)) +
      (fit->budget_is_total ? " MiB device total" : " MiB free");
  log.Warn("cli",
           "gpu " + std::to_string(device) + " cannot hold --gpu-layers " +
               std::to_string(gpu_layers) + ": resident about " +
               std::to_string(ToMiB(fit->resident_bytes)) + " MiB plus " +
               std::to_string(ToMiB(fit->staging_bytes)) + " MiB staging plus " +
               std::to_string(ToMiB(fit->kv_bytes)) + " MiB KV cache exceeds " +
               budget_label + "; lower --gpu-layers (try --gpu-layers help), "
                              "lower --context, or quantize the KV cache");
  return false;
}

std::vector<ConcurrentSessionResult> RunConcurrentSessions(
    const RunnerPool& pool, std::span<const GenerateOptions> requests,
    std::function<void(std::size_t, std::uint32_t)> on_token) {
  std::vector<ConcurrentSessionResult> results(requests.size());
  if (requests.empty()) {
    return results;
  }
  // One worker per session; each blocks in SubmitLeastLoaded while the
  // dispatcher time-slices the device, so all sessions decode together.
  std::vector<std::thread> workers;
  workers.reserve(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    workers.emplace_back([&, i] {
      auto outcome = pool.SubmitLeastLoaded(
          requests[i], [&](std::uint32_t token) {
            results[i].tokens.push_back(token);
            if (on_token) {
              on_token(i, token);
            }
            return true;
          });
      results[i].outcome = std::move(outcome);
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  return results;
}

}  // namespace tessera::cli
