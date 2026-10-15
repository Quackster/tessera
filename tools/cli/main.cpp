#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "calibrate_command.hpp"
#include "cli_common.hpp"
#include "cli_helpers.hpp"
#include "fetch_mtp_command.hpp"
#include "tessera/calibrate.hpp"
#include "tessera/defaults.hpp"
#include "tessera/engine.hpp"
#include "tessera/image.hpp"
#include "tessera/pipeline.hpp"
#include "tessera/tensor_parallel.hpp"
#include "tessera/context_parallel.hpp"
#include "tessera/spec_split.hpp"
#include "tessera/hybrid.hpp"
#include "tessera/vision.hpp"
#include "tessera/serve.hpp"
#include "tessera/speculative.hpp"
#include "tessera/types.hpp"

namespace {

using tessera::kDefaultContext;
using tessera::kDefaultDraftBlock;
using tessera::kDefaultPrefillChunk;
using tessera::cli::ApplyDiagnosticFlag;
using tessera::cli::LogModelSummary;
using tessera::cli::MatchKvType;
using tessera::cli::PrintUsage;
using tessera::cli::ResolveCalibration;

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;

// On-disk bytes of a model path: the file size for a GGUF, the sum of
// the regular files for an MXFP4 directory. Used to refuse a device that
// cannot hold the weights before the load starts.
std::uint64_t PathSizeBytes(const std::string& path) {
  std::error_code ec;
  const std::filesystem::path p(path);
  if (std::filesystem::is_regular_file(p, ec)) {
    return std::filesystem::file_size(p, ec);
  }
  std::uint64_t total = 0;
  if (std::filesystem::is_directory(p, ec)) {
    for (const auto& entry : std::filesystem::directory_iterator(p, ec)) {
      if (entry.is_regular_file(ec)) {
        total += entry.file_size(ec);
      }
    }
  }
  return total;
}

// Render one prompt through a chat template with the thinking options.
// Without thinking options a missing template falls back to the raw text.
// With them a render failure is an error, so an explicit thinking request
// never silently runs without thinking.
std::expected<std::string, tessera::StatusCode> RenderPromptText(
    const std::string& source, bool no_chat, bool enable_thinking,
    const std::string& reasoning_effort,
    const std::function<std::expected<std::string, tessera::StatusCode>()>&
        render) {
  if (no_chat) {
    return source;
  }
  auto chat = render();
  if (chat) {
    return *chat;
  }
  if (enable_thinking || !reasoning_effort.empty()) {
    return std::unexpected(chat.error());
  }
  return source;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage();
    return kExitUsage;
  }
  const std::string_view command = argv[1];
  if (command == "--list-gpus") {
    auto names = tessera::ListGpuNames();
    if (!names) {
      std::fprintf(stderr, "cli: cannot list GPUs (%s)\n",
                   tessera::ToString(names.error()).data());
      return kExitError;
    }
    for (std::size_t i = 0; i < names->size(); ++i) {
      std::printf("gpu %zu: %s\n", i, (*names)[i].c_str());
    }
    return kExitOk;
  }
  if (command == "calibrate") {
    return tessera::cli::RunCalibrateCommand(argc, argv);
  }
  if (command == "fetch-mtp") {
    return tessera::cli::RunFetchMtpCommand(argc, argv);
  }
  if (command != "run" && command != "serve") {
    PrintUsage();
    return kExitUsage;
  }
  std::string model_path;
  std::string draft_path;
  std::string mtp_path;
  std::vector<std::string> prompt_texts;
  std::string calibration_path;
  std::string mmproj_path;
  std::string image_path;
  std::uint32_t image_token = 0;
  bool quiet = false;
  bool verbose = false;
  bool no_chat = false;
  bool enable_thinking = false;
  std::string reasoning_effort;
  std::string host = "127.0.0.1";
  bool auto_title = true;
  std::vector<std::string> api_keys;
  std::vector<std::string> allow_origins;
  bool speculate = false;
  bool pipeline_mode = false;
  bool tp_mode = false;
  bool cp_mode = false;
  bool spec_split_mode = false;
  bool hybrid_mode = false;
  std::size_t dp_groups = 1;
  tessera::SamplingOptions sampling;
  bool sample = false;
  bool sampling_explicit = false;
  tessera::KvCacheType kv_type = tessera::KvCacheType::F32;
  std::uint64_t seed = 0;
  std::uint16_t port = tessera::kDefaultServePort;
  std::size_t stats_interval_seconds = tessera::kDefaultStatsIntervalSeconds;
  std::size_t max_completion_tokens = 0;
  std::size_t max_thinking_tokens = 0;
  std::size_t context = kDefaultContext;
  std::size_t draft_block = kDefaultDraftBlock;
  std::size_t prefill_chunk = kDefaultPrefillChunk;
  std::size_t split_target = 0;
  std::size_t split_cap = 0;
  std::size_t attention_split = 0;
  std::size_t draft_context = 0;
  std::size_t gpu_layers = 0;
  bool gpu_layers_help = false;
  // GPU selection: one index or a comma-separated list. Empty means the
  // first device.
  std::vector<int> gpus;
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--draft" && i + 1 < argc) {
      draft_path = argv[++i];
    } else if (arg == "--mtp" && i + 1 < argc) {
      mtp_path = argv[++i];
    } else if (arg == "--prompt-text" && i + 1 < argc) {
      prompt_texts.emplace_back(argv[++i]);
    } else if (arg == "--mmproj" && i + 1 < argc) {
      mmproj_path = argv[++i];
    } else if (arg == "--image" && i + 1 < argc) {
      image_path = argv[++i];
    } else if (arg == "--quiet") {
      quiet = true;
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--no-chat") {
      no_chat = true;
    } else if (arg == "--enable-thinking") {
      enable_thinking = true;
    } else if (arg == "--reasoning-effort" && i + 1 < argc) {
      reasoning_effort = argv[++i];
    } else if (arg == "--api-key" && i + 1 < argc) {
      api_keys.emplace_back(argv[++i]);
    } else if (arg == "--allow-origin" && i + 1 < argc) {
      allow_origins.emplace_back(argv[++i]);
    } else if (arg == "--host" && i + 1 < argc) {
      host = argv[++i];
    } else if (arg == "--no-auto-title") {
      auto_title = false;
    } else if (arg == "--port" && i + 1 < argc) {
      port = static_cast<std::uint16_t>(std::stoul(argv[++i]));
    } else if (arg == "--stats-interval" && i + 1 < argc) {
      stats_interval_seconds = std::stoul(argv[++i]);
    } else if (arg == "--context" && i + 1 < argc) {
      context = std::stoul(argv[++i]);
    } else if (arg == "--gpus" && i + 1 < argc) {
      // One device index or a comma-separated list, for example `0` or
      // `0,1`. serve runs one replica per device; run splits one model
      // across the devices only with a split flag.
      gpus = tessera::cli::ParseGpuList(argv[++i]);
    } else if (arg == "--calibration" && i + 1 < argc) {
      calibration_path = argv[++i];
    } else if (arg == "--draft-block" && i + 1 < argc) {
      draft_block = std::stoul(argv[++i]);
    } else if (arg == "--prefill-chunk" && i + 1 < argc) {
      prefill_chunk = std::stoul(argv[++i]);
    } else if (arg == "--split-target" && i + 1 < argc) {
      split_target = std::stoul(argv[++i]);
    } else if (arg == "--split-cap" && i + 1 < argc) {
      split_cap = std::stoul(argv[++i]);
    } else if (arg == "--attention-split" && i + 1 < argc) {
      attention_split = std::stoul(argv[++i]);
    } else if (arg == "--draft-context" && i + 1 < argc) {
      draft_context = std::stoul(argv[++i]);
    } else if (arg == "--gpu-layers" && i + 1 < argc) {
      // How many layers stay in VRAM; the rest page from system RAM per
      // token. 0 (the default) uploads everything; `all` also keeps
      // everything resident; `help` sizes the model without loading it.
      // Larger than the layer count caps cleanly.
      const std::string_view layers_value = argv[++i];
      if (layers_value == "help") {
        gpu_layers_help = true;
      } else {
        gpu_layers = layers_value == "all"
                         ? 0
                         : std::stoul(std::string(layers_value));
      }
    } else if (arg == "--speculate") {
      speculate = true;
    } else if (arg == "--pipeline") {
      pipeline_mode = true;
    } else if (arg == "--tp") {
      tp_mode = true;
    } else if (arg == "--spec-split") {
      spec_split_mode = true;
    } else if (arg == "--hybrid") {
      hybrid_mode = true;
    } else if (arg == "--dp" && i + 1 < argc) {
      dp_groups = std::stoul(argv[++i]);
    } else if (arg == "--cp") {
      cp_mode = true;
    } else if (arg == "--sample") {
      sample = true;
    } else if (MatchKvType(arg, kv_type)) {
      // handled by MatchKvType
    } else if (arg == "--temperature" && i + 1 < argc) {
      sampling.temperature = std::stof(argv[++i]);
      sampling_explicit = true;
    } else if (arg == "--top-p" && i + 1 < argc) {
      sampling.top_p = std::stof(argv[++i]);
      sampling_explicit = true;
    } else if (arg == "--top-k" && i + 1 < argc) {
      sampling.top_k = std::stoi(argv[++i]);
      sampling_explicit = true;
    } else if (arg == "--min-p" && i + 1 < argc) {
      sampling.min_p = std::stof(argv[++i]);
      sampling_explicit = true;
    } else if (arg == "--presence-penalty" && i + 1 < argc) {
      sampling.presence_penalty = std::stof(argv[++i]);
      sampling_explicit = true;
    } else if (arg == "--repetition-penalty" && i + 1 < argc) {
      sampling.repetition_penalty = std::stof(argv[++i]);
      sampling_explicit = true;
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = std::stoull(argv[++i]);
    } else if (arg == "--max-completion-tokens" && i + 1 < argc) {
      max_completion_tokens = std::stoul(argv[++i]);
    } else if (arg == "--max-thinking-tokens" && i + 1 < argc) {
      max_thinking_tokens = std::stoul(argv[++i]);
    } else if (ApplyDiagnosticFlag(arg, i, argv, argc)) {
      // handled by ApplyDiagnosticFlag (sets the diagnostic env var)
    } else {
      std::fprintf(stderr, "cli: unknown or unterminated argument '%s'\n",
                   arg.data());
      PrintUsage();
      return kExitUsage;
    }
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "cli: --model is required\n");
    PrintUsage();
    return kExitUsage;
  }
  if (api_keys.empty()) {
    if (const char* key = std::getenv("TESSERA_API_KEY");
        key != nullptr && *key != '\0') {
      api_keys.emplace_back(key);
    }
  }
  if (calibration_path.empty()) {
    if (const char* env = std::getenv("TESSERA_CALIBRATION");
        env != nullptr && *env != '\0') {
      calibration_path = env;
    }
  }

  // Device selection: an empty list means the first device; `--gpus 0`
  // selects one device and `--gpus 0,1` selects two.
  const std::vector<int> device_indices =
      gpus.empty() ? std::vector<int>{0} : gpus;

  // Explain how a multi-device `--gpus` list is routed: serve runs one
  // replica per device, while run splits one model only when a split flag
  // is set. This tells the user the split flags when they forget one.
  const bool split_active = pipeline_mode || tp_mode || cp_mode ||
                            spec_split_mode || hybrid_mode;
  // serve is replica-only by design: it loads one full model per --gpus
  // device. The split layouts are run-only, so refuse them here instead of
  // silently loading replicas the user did not ask for.
  if (command == "serve" && split_active) {
    std::fprintf(stderr,
                 "cli: serve runs one model replica per --gpus device and "
                 "cannot split one model across devices; --pipeline, --tp, "
                 "--cp, --spec-split and --hybrid are run-only. Use `run` for "
                 "a split layout, or `serve` with one --gpus device.\n");
    return kExitUsage;
  }
  const std::string routing_hint = tessera::cli::DeviceRoutingHint(
      device_indices,
      command == "serve" ? tessera::cli::DeviceRouteKind::kServeReplicas
                         : tessera::cli::DeviceRouteKind::kRunSingle,
      split_active);
  if (!routing_hint.empty()) {
    std::fprintf(stderr, "%s\n", routing_hint.c_str());
  }

  auto make_engine = [&](int device)
      -> std::expected<std::unique_ptr<tessera::Engine>, tessera::StatusCode> {
    tessera::EngineOptions engine_options;  // default stderr diagnostics sink
    engine_options.device_index = device;
    engine_options.prefill_chunk_tokens = prefill_chunk;
    return tessera::Engine::Create(engine_options);
  };
  // Attach the speculative strategy before serving or generating. DFlash2
  // drafts from its checkpoint; MTP drafts from the target's own nextn
  // head, so it needs no path. One strategy instance per engine: a
  // strategy is not shared across devices.
  auto attach_strategy = [&](tessera::Engine& engine) {
    if (draft_path.empty() && !speculate) {
      return;
    }
    std::unique_ptr<tessera::SpeculativeStrategy> strategy =
        draft_path.empty() ? tessera::CreateMtpStrategy()
                           : tessera::CreateDFlash2Strategy();
    const std::string label =
        draft_path.empty() ? "MTP" : "DFlash2 ('" + draft_path + "')";
    auto attach =
        strategy->Attach(tessera::StrategyOptions{draft_path, draft_block});
    if (!attach) {
      engine.Diagnostics().Warn(
          "cli", label + " attach failed: " +
                     std::string(tessera::ToString(attach.error())) +
                     "; continuing without speculative decoding");
      return;
    }
    auto attached = engine.AttachSpeculative(std::move(strategy));
    if (!attached) {
      engine.Diagnostics().Warn(
          "cli", label + " strategy attach failed: " +
                     std::string(tessera::ToString(attached.error())) +
                     "; continuing without speculative decoding");
    }
  };

  if (gpu_layers_help) {
    // Size the model without loading it: needs --model, one device for
    // the free-memory query, and the calibration strategy label.
    if (model_path.empty()) {
      std::fprintf(stderr, "cli: --gpu-layers help needs --model\n");
      PrintUsage();
      return kExitUsage;
    }
    auto engine_result = make_engine(device_indices.front());
    if (!engine_result) {
      std::fprintf(stderr, "cli: engine create failed (%s)\n",
                   tessera::ToString(engine_result.error()).data());
      return kExitError;
    }
    tessera::Engine& engine = **engine_result;
    const std::string strategy =
        draft_path.empty() ? (speculate ? "mtp" : "none") : "dflash2";
    return tessera::cli::LogGpuLayersAdvice(engine, model_path, context,
                                            kv_type, strategy) == 0
               ? kExitOk
               : kExitError;
  }

  if (command == "serve") {
    if (enable_thinking || !reasoning_effort.empty()) {
      std::fprintf(stderr,
                   "cli: serve needs no --enable-thinking/--reasoning-effort; "
                   "thinking is per request in the HTTP API\n");
      return kExitUsage;
    }
    tessera::ServeOptions serve_options;
    serve_options.host = host;
    serve_options.port = port;
    serve_options.api_keys = api_keys;
    serve_options.allow_origins = allow_origins;
    serve_options.auto_title = auto_title;
    serve_options.stats_interval_seconds = stats_interval_seconds;
    // Build one runner per device on the loader thread. Devices load and
    // warm one at a time, so only one model is in host memory at once.
    // The server binds first, so a client reaches /health while loading.
    auto build_pool = [&](const tessera::ServeProgress& report)
        -> std::expected<std::unique_ptr<tessera::RunnerPool>,
                         tessera::StatusCode> {
      auto pool = std::make_unique<tessera::RunnerPool>();
      tessera::CalibrationConfig calibration;
      // Paged weights are private host RAM per replica, so the pool holds
      // the sum across the devices. The device check above never sees it.
      std::uint64_t host_held_bytes = 0;
      for (std::size_t i = 0; i < device_indices.size(); ++i) {
        const int device = device_indices[i];
        report("loading weights on gpu " + std::to_string(device));
        auto created = make_engine(device);
        if (!created) {
          return std::unexpected(created.error());
        }
        std::unique_ptr<tessera::Engine> engine_ptr = std::move(*created);
        // Apply a calibrated paging value before the load: an MoE target
        // larger than VRAM only loads paged, and paging is a load-time
        // setting. An explicit --gpu-layers always wins.
        const std::string strategy =
            draft_path.empty() ? (speculate ? "mtp" : "none") : "dflash2";
        if (!calibration_path.empty()) {
          const std::size_t explicit_gpu_layers = gpu_layers;
          gpu_layers = tessera::cli::ResolveCalibrationGpuLayers(
              *engine_ptr, model_path, context, kv_type, strategy,
              calibration_path, gpu_layers);
          if (gpu_layers != explicit_gpu_layers) {
            engine_ptr->Diagnostics().Info(
                "cli", "calibration: --gpu-layers " +
                           std::to_string(gpu_layers));
          }
        }
        // Refuse a device that cannot hold the weights before the load
        // starts, so a two-device serve does not half-load. Under paging
        // the file size is not the resident size, so the resident set
        // plus the KV cache is judged instead. A zero free figure means
        // the backend reports no free memory; skip the check then and
        // let the load fail on its own when it truly cannot fit.
        const std::uint64_t needed = PathSizeBytes(model_path);
        const tessera::DeviceMemoryInfo memory =
            engine_ptr->Owner().MemoryInfo();
        if (gpu_layers == 0 && memory.total_bytes != 0 && needed != 0 &&
            memory.free_bytes != 0 && memory.free_bytes < needed) {
          std::fprintf(stderr,
                       "cli: gpu %d has %s MiB free but the model needs about "
                       "%s MiB; use fewer devices or a smaller quantization\n",
                       device,
                       std::to_string(memory.free_bytes / (1024 * 1024)).c_str(),
                       std::to_string(needed / (1024 * 1024)).c_str());
          return std::unexpected(tessera::StatusCode::OutOfMemory);
        }
        if (gpu_layers != 0) {
          if (!tessera::cli::CheckGpuLayersFit(*engine_ptr, model_path,
                                               context, kv_type, strategy,
                                               gpu_layers, device)) {
            return std::unexpected(tessera::StatusCode::OutOfMemory);
          }
        }
        // Refuse a replica whose paged weights cannot fit system RAM with
        // the ones already loaded, so a multi-device serve never drives the
        // host into the OOM killer.
        std::uint64_t replica_host_bytes = 0;
        if (!tessera::cli::CheckHostRamFit(*engine_ptr, model_path, gpu_layers,
                                           strategy, host_held_bytes, device,
                                           &replica_host_bytes)) {
          return std::unexpected(tessera::StatusCode::OutOfMemory);
        }
        host_held_bytes += replica_host_bytes;
        tessera::ModelOptions model_options{model_path, context};
        model_options.gpu_layers = gpu_layers;
        model_options.mtp_path = mtp_path;
        auto model = engine_ptr->LoadModel(model_options);
        if (!model) {
          return std::unexpected(model.error());
        }
        LogModelSummary(*engine_ptr, **model);
        if (i == 0 && tessera::cli::ApplyRecommendedSampling(
                          **model, enable_thinking, sampling_explicit,
                          &sampling)) {
          sample = true;
        }
        attach_strategy(*engine_ptr);
        if (!calibration_path.empty() && i == 0) {
          // The calibration key depends on the model; resolve it on the
          // first device. Homogeneous devices share the settings.
          tessera::CalibrationConfig explicit_config;
          explicit_config.prefill_chunk_tokens = prefill_chunk;
          explicit_config.draft_tokens = draft_block;
          calibration = ResolveCalibration(*engine_ptr, **model, context,
                                           kv_type, strategy, calibration_path,
                                           explicit_config);
        }
        report("warming up kernels");
        tessera::GenerateOptions warm;
        warm.max_completion_tokens = 1;
        warm.progress_every = 0;
        // Warm the kernels with the serve-wide KV storage type, so the
        // first real turn pays no first-touch cost on another type.
        warm.kv_type = kv_type;
        if (!calibration_path.empty()) {
          tessera::ApplyToGenerateOptions(calibration, warm);
        }
        auto warmed = engine_ptr->Generate(**model, warm);
        if (!warmed) {
          std::fprintf(stderr, "cli: kernel warmup failed (%s)\n",
                       tessera::ToString(warmed.error()).data());
          return std::unexpected(warmed.error());
        }
        // The pool carries the launcher's KV storage type (--kv-q8 and
        // friends); the serve layer reads it onto every turn, so an
        // unlimited turn reserves quantized rows and fits the device.
        pool->Add(std::make_unique<tessera::Runner>(std::move(engine_ptr),
                                                    std::move(*model), device,
                                                    calibration, kv_type));
      }
      report("ready");
      return pool;
    };
    auto served = tessera::Serve(serve_options, std::move(build_pool));
    if (!served) {
      std::fprintf(stderr, "cli: serve failed (%s)\n",
                   tessera::ToString(served.error()).data());
      return kExitError;
    }
    return kExitOk;
  }

  if (pipeline_mode) {
    // Layer-split the model across the selected devices for one request.
    if (prompt_texts.size() != 1) {
      std::fprintf(stderr, "cli: --pipeline needs exactly one --prompt-text\n");
      return kExitUsage;
    }
    const std::string& prompt_text = prompt_texts.front();
    tessera::PipelineOptions pipeline_options;
    pipeline_options.model_path = model_path;
    pipeline_options.devices = device_indices;
    pipeline_options.context = context;
    pipeline_options.kv_type = kv_type;
    pipeline_options.mtp_path = mtp_path;
    auto pipeline = tessera::Pipeline::Create(pipeline_options);
    if (!pipeline) {
      std::fprintf(stderr, "cli: pipeline create failed (%s)\n",
                   tessera::ToString(pipeline.error()).data());
      return kExitError;
    }
    // Attach the drafter to the pipeline: --draft selects DFlash2, else
    // --speculate selects MTP on the last stage, which owns the MTP head.
    if (speculate || !draft_path.empty()) {
      const bool dflash2 = !draft_path.empty();
      const auto factory =
          [dflash2]() -> std::unique_ptr<tessera::SpeculativeStrategy> {
        return dflash2 ? tessera::CreateDFlash2Strategy()
                       : tessera::CreateMtpStrategy();
      };
      auto attached = (*pipeline)->AttachSpeculative(
          factory, tessera::StrategyOptions{draft_path, draft_block});
      if (!attached) {
        std::fprintf(stderr, "cli: pipeline strategy attach failed (%s)\n",
                     tessera::ToString(attached.error()).data());
        return kExitError;
      }
    }
    const tessera::Tokenizer* tokenizer = (*pipeline)->GetTokenizer();
    if (tokenizer == nullptr) {
      std::fprintf(stderr, "cli: this model has no tokenizer\n");
      return kExitError;
    }
    const auto encode = [&](const std::string& source,
                            std::vector<std::uint32_t>* ids) -> bool {
      auto rendered = RenderPromptText(source, no_chat, enable_thinking,
                                       reasoning_effort, [&] {
                                         return (*pipeline)->ChatPrompt(
                                             source, enable_thinking,
                                             reasoning_effort);
                                       });
      if (!rendered) {
        std::fprintf(stderr, "cli: prompt render failed (%s)\n",
                     tessera::ToString(rendered.error()).data());
        return false;
      }
      std::string text = *rendered;
      auto encoded = tokenizer->Encode(text);
      if (!encoded) {
        return false;
      }
      *ids = std::move(*encoded);
      return true;
    };
    std::vector<std::uint32_t> ids;
    if (!encode(prompt_text, &ids)) {
      std::fprintf(stderr, "cli: prompt encode failed\n");
      return kExitError;
    }
    auto generated = (*pipeline)->Generate(ids, max_completion_tokens);
    if (!generated) {
      std::fprintf(stderr, "cli: pipeline generation failed (%s)\n",
                   tessera::ToString(generated.error()).data());
      return kExitError;
    }
    auto output_text = tokenizer->Decode(*generated);
    if (output_text) {
      std::fwrite(output_text->data(), 1, output_text->size(), stdout);
      std::fputc('\n', stdout);
      std::fflush(stdout);
    }
    return kExitOk;
  }

  if (tp_mode && !hybrid_mode) {
    // Tensor-parallel: split the attention and FFN across the selected
    // devices and run greedy generation with a per-rank thread and the
    // rank reduce. MoE routed experts split by expert dim on the same
    // ranks; each rank runs its local experts and reduces the sum.
    if (prompt_texts.size() != 1) {
      std::fprintf(stderr, "cli: --tp needs exactly one --prompt-text\n");
      return kExitUsage;
    }
    const std::string& prompt_text = prompt_texts.front();
    tessera::TensorParallelOptions tp_options;
    tp_options.model_path = model_path;
    tp_options.devices = device_indices;
    tp_options.context = context;
    tp_options.kv_type = kv_type;
    auto tp = tessera::TensorParallel::Create(tp_options);
    if (!tp) {
      std::fprintf(stderr, "cli: tensor-parallel create failed (%s)\n",
                   tessera::ToString(tp.error()).data());
      return kExitError;
    }
    const tessera::Tokenizer* tokenizer = (*tp)->GetTokenizer();
    if (tokenizer == nullptr) {
      std::fprintf(stderr, "cli: this model has no tokenizer\n");
      return kExitError;
    }
    auto rendered = RenderPromptText(prompt_text, no_chat, enable_thinking,
                                        reasoning_effort, [&] {
                                          return (*tp)->ChatPrompt(
                                              prompt_text, enable_thinking,
                                              reasoning_effort);
                                        });
    if (!rendered) {
      std::fprintf(stderr, "cli: prompt render failed (%s)\n",
                   tessera::ToString(rendered.error()).data());
      return kExitError;
    }
    std::string text = *rendered;
    auto ids = tokenizer->Encode(text);
    if (!ids) {
      std::fprintf(stderr, "cli: prompt encode failed\n");
      return kExitError;
    }
    auto generated = (*tp)->Generate(*ids, max_completion_tokens);
    if (!generated) {
      std::fprintf(stderr, "cli: tensor-parallel generation failed (%s)\n",
                   tessera::ToString(generated.error()).data());
      return kExitError;
    }
    auto output_text = tokenizer->Decode(*generated);
    if (output_text) {
      std::fwrite(output_text->data(), 1, output_text->size(), stdout);
      std::fputc('\n', stdout);
      std::fflush(stdout);
    }
    return kExitOk;
  }

  if (cp_mode && !hybrid_mode) {
    // Context-parallel: split the key range across the selected devices
    // and run greedy generation with a per-rank thread and a per-layer
    // partial combine.
    if (prompt_texts.size() != 1) {
      std::fprintf(stderr, "cli: --cp needs exactly one --prompt-text\n");
      return kExitUsage;
    }
    const std::string& prompt_text = prompt_texts.front();
    tessera::ContextParallelOptions cp_options;
    cp_options.model_path = model_path;
    cp_options.devices = device_indices;
    cp_options.context = context;
    cp_options.kv_type = kv_type;
    auto cp = tessera::ContextParallel::Create(cp_options);
    if (!cp) {
      std::fprintf(stderr, "cli: context-parallel create failed (%s)\n",
                   tessera::ToString(cp.error()).data());
      return kExitError;
    }
    const tessera::Tokenizer* tokenizer = (*cp)->GetTokenizer();
    if (tokenizer == nullptr) {
      std::fprintf(stderr, "cli: this model has no tokenizer\n");
      return kExitError;
    }
    auto rendered = RenderPromptText(prompt_text, no_chat, enable_thinking,
                                        reasoning_effort, [&] {
                                          return (*cp)->ChatPrompt(
                                              prompt_text, enable_thinking,
                                              reasoning_effort);
                                        });
    if (!rendered) {
      std::fprintf(stderr, "cli: prompt render failed (%s)\n",
                   tessera::ToString(rendered.error()).data());
      return kExitError;
    }
    std::string text = *rendered;
    auto ids = tokenizer->Encode(text);
    if (!ids) {
      std::fprintf(stderr, "cli: prompt encode failed\n");
      return kExitError;
    }
    auto generated = (*cp)->Generate(*ids, max_completion_tokens);
    if (!generated) {
      std::fprintf(stderr, "cli: context-parallel generation failed (%s)\n",
                   tessera::ToString(generated.error()).data());
      return kExitError;
    }
    auto output_text = tokenizer->Decode(*generated);
    if (output_text) {
      std::fwrite(output_text->data(), 1, output_text->size(), stdout);
      std::fputc('\n', stdout);
      std::fflush(stdout);
    }
    return kExitOk;
  }

  if (spec_split_mode) {
    // Speculative split: the target runs on the first --gpus device and
    // the MTP draft head on the second. Needs --speculate (MTP); other
    // strategies stay on one device.
    if (!speculate) {
      std::fprintf(stderr, "cli: --spec-split needs --speculate\n");
      return kExitUsage;
    }
    if (prompt_texts.size() != 1) {
      std::fprintf(stderr, "cli: --spec-split needs exactly one --prompt-text\n");
      return kExitUsage;
    }
    const std::string& prompt_text = prompt_texts.front();
    tessera::SpecSplitOptions split_options;
    split_options.model_path = model_path;
    split_options.devices = device_indices;
    split_options.context = context;
    split_options.kv_type = kv_type;
    split_options.draft_block = draft_block;
    split_options.prefill_chunk = prefill_chunk;
    split_options.mtp_path = mtp_path;
    split_options.gpu_layers = gpu_layers;
    auto split = tessera::SpecSplit::Create(split_options);
    if (!split) {
      std::fprintf(stderr, "cli: speculative-split create failed (%s)\n",
                   tessera::ToString(split.error()).data());
      return kExitError;
    }
    const tessera::Tokenizer* tokenizer = (*split)->GetTokenizer();
    if (tokenizer == nullptr) {
      std::fprintf(stderr, "cli: this model has no tokenizer\n");
      return kExitError;
    }
    auto rendered = RenderPromptText(prompt_text, no_chat, enable_thinking,
                                        reasoning_effort, [&] {
                                          return (*split)->ChatPrompt(
                                              prompt_text, enable_thinking,
                                              reasoning_effort);
                                        });
    if (!rendered) {
      std::fprintf(stderr, "cli: prompt render failed (%s)\n",
                   tessera::ToString(rendered.error()).data());
      return kExitError;
    }
    std::string text = *rendered;
    auto ids = tokenizer->Encode(text);
    if (!ids) {
      std::fprintf(stderr, "cli: prompt encode failed\n");
      return kExitError;
    }
    auto generated = (*split)->Generate(*ids, max_completion_tokens);
    if (!generated) {
      std::fprintf(stderr, "cli: speculative-split generation failed (%s)\n",
                   tessera::ToString(generated.error()).data());
      return kExitError;
    }
    auto output_text = tokenizer->Decode(*generated);
    if (output_text) {
      std::fwrite(output_text->data(), 1, output_text->size(), stdout);
      std::fputc('\n', stdout);
      std::fflush(stdout);
    }
    return kExitOk;
  }

  if (hybrid_mode) {
    // Hybrid: data-parallel groups of model-parallel teams behind one
    // device mesh. The one prompt runs on every group; --tp/--cp selects
    // the model-parallel route inside each multi-rank group.
    if (prompt_texts.size() != 1) {
      std::fprintf(stderr, "cli: --hybrid needs exactly one --prompt-text\n");
      return kExitUsage;
    }
    const std::string& prompt_text = prompt_texts.front();
    tessera::HybridOptions hybrid_options;
    hybrid_options.model_path = model_path;
    hybrid_options.devices = device_indices;
    hybrid_options.dp_groups = dp_groups;
    hybrid_options.mp_kind = cp_mode ? tessera::HybridMpKind::Context
                                     : tessera::HybridMpKind::Tensor;
    hybrid_options.context = context;
    hybrid_options.kv_type = kv_type;
    auto hybrid = tessera::Hybrid::Create(hybrid_options);
    if (!hybrid) {
      std::fprintf(stderr, "cli: hybrid create failed (%s)\n",
                   tessera::ToString(hybrid.error()).data());
      return kExitError;
    }
    const tessera::Tokenizer* tokenizer = (*hybrid)->GetTokenizer();
    if (tokenizer == nullptr) {
      std::fprintf(stderr, "cli: this model has no tokenizer\n");
      return kExitError;
    }
    std::vector<std::vector<std::uint32_t>> prompts;
    prompts.reserve(dp_groups);
    for (std::size_t g = 0; g < dp_groups; ++g) {
      auto rendered = RenderPromptText(prompt_text, no_chat, enable_thinking,
                                       reasoning_effort, [&] {
                                         return (*hybrid)->ChatPrompt(
                                             prompt_text, enable_thinking,
                                             reasoning_effort);
                                       });
      if (!rendered) {
        std::fprintf(stderr, "cli: prompt render failed (%s)\n",
                     tessera::ToString(rendered.error()).data());
        return kExitError;
      }
      auto encoded = tokenizer->Encode(*rendered);
      if (!encoded) {
        std::fprintf(stderr, "cli: prompt encode failed\n");
        return kExitError;
      }
      prompts.push_back(std::move(*encoded));
    }
    auto outputs = (*hybrid)->GenerateMany(prompts, max_completion_tokens);
    if (!outputs) {
      std::fprintf(stderr, "cli: hybrid generation failed (%s)\n",
                   tessera::ToString(outputs.error()).data());
      return kExitError;
    }
    for (std::size_t i = 0; i < outputs->size(); ++i) {
      auto text = tokenizer->Decode((*outputs)[i]);
      std::fprintf(stdout, "batch %zu: %s\n", i, text ? text->c_str() : "");
    }
    std::fflush(stdout);
    return kExitOk;
  }

  auto engine_result = make_engine(device_indices.front());
  if (!engine_result) {
    std::fprintf(stderr, "cli: engine create failed (%s)\n",
                 tessera::ToString(engine_result.error()).data());
    return kExitError;
  }
  tessera::Engine& engine = **engine_result;
  auto& log = engine.Diagnostics();
  if (verbose) {
    log.SetMinLevel(tessera::log::Level::Debug);
  } else if (quiet) {
    log.SetMinLevel(tessera::log::Level::Warn);
  }
  attach_strategy(engine);

  // Apply a calibrated paging value before the load: an MoE target larger
  // than VRAM only loads paged, and paging is a load-time setting. An
  // explicit --gpu-layers always wins.
  const std::string strategy =
      draft_path.empty() ? (speculate ? "mtp" : "none") : "dflash2";
  if (!calibration_path.empty()) {
    const std::size_t explicit_gpu_layers = gpu_layers;
    gpu_layers = tessera::cli::ResolveCalibrationGpuLayers(
        engine, model_path, context, kv_type, strategy, calibration_path,
        gpu_layers);
    if (gpu_layers != explicit_gpu_layers) {
      log.Info("cli", "calibration: --gpu-layers " +
                          std::to_string(gpu_layers));
    }
  }
  if (gpu_layers != 0) {
    // Refuse a paged load whose resident set plus KV cache cannot fit,
    // before the weights start moving.
    if (!tessera::cli::CheckGpuLayersFit(engine, model_path, context, kv_type,
                                         strategy, gpu_layers,
                                         device_indices.front())) {
      return kExitError;
    }
  }
  // Refuse a paged load whose host weight copy cannot fit system RAM,
  // before the weights start moving.
  std::uint64_t run_host_bytes = 0;
  if (!tessera::cli::CheckHostRamFit(engine, model_path, gpu_layers, strategy,
                                     0, device_indices.front(),
                                     &run_host_bytes)) {
    return kExitError;
  }
  tessera::ModelOptions model_options{model_path, context};
  model_options.gpu_layers = gpu_layers;
  model_options.mtp_path = mtp_path;
  auto model = engine.LoadModel(model_options);
  if (!model) {
    std::fprintf(stderr, "cli: model load failed (%s)\n",
                 tessera::ToString(model.error()).data());
    return kExitError;
  }
  tessera::Model& loaded = **model;
  LogModelSummary(engine, loaded);
  // Apply the architecture's recommended sampler unless the user set one.
  const bool thinking = enable_thinking;
  if (tessera::cli::ApplyRecommendedSampling(loaded, thinking,
                                             sampling_explicit, &sampling)) {
    sample = true;
    if (!quiet) {
      log.Info("cli", "sampling: architecture default (thinking " +
                         std::string(thinking ? "on" : "off") + ")");
    }
  }
  // Resolve any saved calibration (explicit CLI values win) before building
  // the request options.
  tessera::CalibrationConfig applied;
  applied.prefill_chunk_tokens = prefill_chunk;
  applied.draft_tokens = draft_block;
  applied.mxfp4_split_target = split_target;
  applied.attention_split = attention_split;
  applied.draft_context = draft_context;
  if (!calibration_path.empty()) {
    applied = ResolveCalibration(engine, loaded, context, kv_type, strategy,
                                 calibration_path, applied);
  }
  if (prompt_texts.size() >= 2) {
    // Parallel sessions: each --prompt-text is one session. The pool
    // time-slices the device, so both decode and stream together.
    if (!image_path.empty() || !mmproj_path.empty()) {
      std::fprintf(stderr, "cli: parallel sessions need no --image\n");
      return kExitUsage;
    }
    if (!draft_path.empty() || speculate) {
      std::fprintf(stderr,
                   "cli: parallel sessions need no --draft/--speculate\n");
      return kExitUsage;
    }
    if (max_thinking_tokens > 0) {
      std::fprintf(stderr,
                   "cli: parallel sessions need no --max-thinking-tokens\n");
      return kExitUsage;
    }
    tessera::GenerateOptions base;
    base.max_completion_tokens = max_completion_tokens;
    base.sample = sample;
    base.sampling = sampling;
    base.seed = seed;
    base.draft_tokens = applied.draft_tokens;
    base.prefill_chunk_tokens = applied.prefill_chunk_tokens;
    base.mxfp4_split_target = applied.mxfp4_split_target;
    base.mxfp4_split_cap = split_cap;
    base.attention_split = applied.attention_split;
    base.draft_context = applied.draft_context;
    base.kv_type = kv_type;
    base.progress_every = quiet ? 0 : 64;
    if (!quiet) {
      log.Info("cli", "kv cache: " +
                         std::string(tessera::ToString(kv_type)));
      log.Info("cli", sample ? "sampling: sampled" : "sampling: greedy");
      log.Info("cli", "parallel sessions: " +
                         std::to_string(prompt_texts.size()));
    }
    const tessera::Tokenizer* encode_tokenizer = loaded.GetTokenizer();
    if (encode_tokenizer == nullptr) {
      log.Warn("cli", "this model has no tokenizer");
      return kExitError;
    }
    std::vector<tessera::GenerateOptions> requests;
    requests.reserve(prompt_texts.size());
    for (const std::string& source : prompt_texts) {
      auto rendered = RenderPromptText(source, no_chat, enable_thinking,
                                       reasoning_effort, [&] {
                                         return loaded.ChatPrompt(
                                             source, enable_thinking,
                                             reasoning_effort);
                                       });
      if (!rendered) {
        log.Warn("cli", std::string("prompt render failed (") +
                            std::string(
                                tessera::ToString(rendered.error())) +
                            ")");
        return kExitError;
      }
      std::string text = *rendered;
      auto ids = encode_tokenizer->Encode(text);
      if (!ids) {
        log.Warn("cli", "prompt encode failed");
        return kExitError;
      }
      tessera::GenerateOptions one = base;
      one.prompt_tokens = std::move(*ids);
      requests.push_back(std::move(one));
    }
    std::unique_ptr<tessera::Engine> engine_ptr =
        std::move(*engine_result);
    std::unique_ptr<tessera::Model> model_ptr = std::move(*model);
    tessera::RunnerPool pool;
    pool.Add(std::make_unique<tessera::Runner>(
        std::move(engine_ptr), std::move(model_ptr),
        device_indices.front()));
    const tessera::Tokenizer* stream_tokenizer = pool.At(0).tokenizer();
    std::mutex print_mutex;
    auto results = tessera::cli::RunConcurrentSessions(
        pool, requests,
        [&](std::size_t session, std::uint32_t token) {
          if (stream_tokenizer != nullptr) {
            auto piece = stream_tokenizer->Decode(
                std::span<const std::uint32_t>(&token, 1));
            if (piece) {
              std::lock_guard<std::mutex> lock(print_mutex);
              std::fprintf(stdout, "[s%zu] %s", session, piece->c_str());
              std::fflush(stdout);
            }
          }
        });
    {
      std::lock_guard<std::mutex> lock(print_mutex);
      std::fputc('\n', stdout);
      std::fflush(stdout);
    }
    auto& pool_log = pool.At(0).engine().Diagnostics();
    bool failed = false;
    for (std::size_t i = 0; i < results.size(); ++i) {
      if (!results[i].outcome) {
        pool_log.Warn("cli", "session " + std::to_string(i) +
                                 " failed (" +
                                 std::string(tessera::ToString(
                                     results[i].outcome.error())) +
                                 ")");
        failed = true;
        continue;
      }
      if (stream_tokenizer != nullptr) {
        auto output_text = stream_tokenizer->Decode(results[i].tokens);
        if (output_text) {
          pool_log.Info("cli", "session " + std::to_string(i) +
                                   " output text: " + *output_text);
        }
      }
    }
    return failed ? kExitError : kExitOk;
  }
  if (!prompt_texts.empty() || !image_path.empty() ||
      max_completion_tokens > 0 || max_thinking_tokens > 0 || enable_thinking ||
      !reasoning_effort.empty()) {
    const std::string prompt_text =
        prompt_texts.empty() ? std::string() : prompt_texts.front();
    tessera::GenerateOptions gen;
    // Zero (the default) fills the remaining context; an explicit
    // budget caps the completion instead.
    gen.max_completion_tokens = max_completion_tokens;
    gen.max_thinking_tokens = max_thinking_tokens;
    gen.sample = sample;
    gen.sampling = sampling;
    gen.seed = seed;
    gen.draft_tokens = applied.draft_tokens;
    gen.prefill_chunk_tokens = applied.prefill_chunk_tokens;
    gen.mxfp4_split_target = applied.mxfp4_split_target;
    gen.mxfp4_split_cap = split_cap;
    gen.attention_split = applied.attention_split;
    gen.draft_context = applied.draft_context;
    gen.kv_type = kv_type;
    gen.progress_every = quiet ? 0 : 64;
    if (!quiet) {
      log.Info("cli", "kv cache: " +
                          std::string(tessera::ToString(kv_type)));
      if (enable_thinking || !reasoning_effort.empty()) {
        log.Info("cli", "thinking: enabled" +
                            (reasoning_effort.empty()
                                 ? std::string()
                                 : " (reasoning_effort=" + reasoning_effort +
                                       ")"));
      }
      if (sample) {
        log.Info("cli", "sampling: temperature=" +
                            std::to_string(sampling.temperature) +
                            " top_p=" + std::to_string(sampling.top_p) +
                            " top_k=" + std::to_string(sampling.top_k) +
                            " min_p=" + std::to_string(sampling.min_p) +
                            " presence_penalty=" +
                            std::to_string(sampling.presence_penalty) +
                            " repetition_penalty=" +
                            std::to_string(sampling.repetition_penalty) +
                            " seed=" + std::to_string(seed));
      } else {
        log.Info("cli", "sampling: greedy");
      }
    }
    if (prompt_text.empty()) {
      log.Warn("cli", "--prompt-text is required");
      return kExitUsage;
    }
    {
      const tessera::Tokenizer* tokenizer = loaded.GetTokenizer();
      if (tokenizer == nullptr) {
        log.Warn("cli",
                 "this model has no tokenizer; text input is unavailable");
        return kExitError;
      }
      std::string text = prompt_text;
      auto chat = no_chat
                      ? std::expected<std::string, tessera::StatusCode>(
                            std::unexpect, tessera::StatusCode::UnsupportedFeature)
                      : loaded.ChatPrompt(prompt_text, enable_thinking,
                                          reasoning_effort);
      if (chat.has_value()) {
        text = *chat;
        if (!quiet) {
          log.Info("cli", "applied the model chat template to the prompt");
        }
      } else if (enable_thinking || !reasoning_effort.empty()) {
        log.Warn("cli", std::string("prompt render failed (") +
                            std::string(tessera::ToString(chat.error())) +
                            ")");
        return kExitError;
      } else if (!quiet) {
        log.Info("cli", "no chat template; using the raw prompt text");
      }
      auto ids = tokenizer->Encode(text);
      if (!ids) {
        log.Warn("cli", std::string("prompt encode failed (") +
                            std::string(tessera::ToString(ids.error())) + ")");
        return kExitError;
      }
      gen.prompt_tokens = *ids;
    }
    std::expected<std::vector<std::uint32_t>, tessera::StatusCode> generated;
    bool streamed_output = false;
    if (!image_path.empty()) {
      {
        const tessera::Tokenizer* tokenizer = loaded.GetTokenizer();
        std::optional<std::uint32_t> id;
        if (tokenizer != nullptr) {
          id = tokenizer->SpecialTokenId("<|image_pad|>");
        }
        if (id.has_value()) {
          image_token = *id;
        } else {
          log.Warn("cli", "no <|image_pad|> token in this model");
          return kExitError;
        }
      }
      if (mmproj_path.empty()) {
        log.Warn("cli", "--image needs --mmproj");
        return kExitError;
      }
      if (!quiet) {
        log.Info("cli", "loading vision projector: " + mmproj_path);
      }
      auto vision = tessera::VisionModel::Load(engine.Owner(), mmproj_path);
      if (!vision) {
        log.Warn("cli", std::string("mmproj load failed (") +
                            std::string(tessera::ToString(vision.error())) +
                            ")");
        return kExitError;
      }
      if (!quiet) {
        log.Info("cli", "vision projector loaded: " +
                            std::to_string(vision->Config().block_count) +
                            " blocks, " +
                            std::to_string(vision->Config().projection_dim) +
                            " projection dim");
        log.Info("cli", "loading image: " + image_path);
      }
      auto image = tessera::LoadPpm(image_path);
      if (!image) {
        log.Warn("cli", std::string("image load failed (") +
                            std::string(tessera::ToString(image.error())) + ")");
        return kExitError;
      }
      auto resized = tessera::ResizeBilinear(*image, vision->Config().image_size,
                                             vision->Config().image_size);
      if (!quiet) {
        log.Info("cli", "image loaded: " + std::to_string(image->width) + "x" +
                            std::to_string(image->height) + ", encoding");
      }
      auto embeddings = vision->Encode(engine.Owner(), resized.pixels,
                                       resized.height, resized.width);
      if (!embeddings) {
        log.Warn("cli", std::string("image encode failed (") +
                            std::string(tessera::ToString(embeddings.error())) +
                            ")");
        return kExitError;
      }
      const std::size_t count = embeddings->size() / vision->Config().projection_dim;
      if (!quiet) {
        log.Info("cli", "image encoded: " + std::to_string(count) + " token(s)");
        // The multimodal path logs no row progress of its own, so the
        // hook shows the image-token prefill before decoding starts.
        gen.prefill_progress = [&log](std::size_t done, std::size_t total) {
          log.Info("cli", "prefill: " + std::to_string(done) + "/" +
                              std::to_string(total) + " rows");
        };
      }
      std::vector<std::uint32_t> prompt(count, image_token);
      prompt.insert(prompt.end(), gen.prompt_tokens.begin(),
                    gen.prompt_tokens.end());
      gen.prompt_tokens = prompt;
      generated = engine.GenerateMultimodal(loaded, gen, *embeddings, count,
                                            image_token);
    } else if (draft_path.empty() && !speculate) {
      // Stream plain greedy decode so each output token reaches the
      // console as it is generated. A blocking Generate would return
      // only after every step (by default the whole remaining context),
      // which looks hung after the prefill line.
      std::vector<std::uint32_t> streamed_ids;
      const tessera::Tokenizer* stream_tokenizer = loaded.GetTokenizer();
      if (!quiet) {
        const std::size_t budget = loaded.EffectiveMaxTokens(
            gen.prompt_tokens.size(), gen.max_completion_tokens);
        log.Info("cli", "generating up to " + std::to_string(budget) +
                            " token(s); streaming output below");
      }
      auto count =
          engine.GenerateStreaming(loaded, gen, [&](std::uint32_t token) {
            streamed_ids.push_back(token);
            if (stream_tokenizer != nullptr) {
              auto piece = stream_tokenizer->Decode(
                  std::span<const std::uint32_t>(&token, 1));
              if (piece) {
                std::fwrite(piece->data(), 1, piece->size(), stdout);
                std::fflush(stdout);
              }
            }
            return true;
          });
      if (!count) {
        generated = std::unexpected(count.error());
      } else {
        std::fputc('\n', stdout);
        std::fflush(stdout);
        generated = std::move(streamed_ids);
        streamed_output = true;
      }
    } else {
      generated =
          draft_path.empty()
              ? (speculate ? engine.GenerateSpeculative(loaded, gen)
                           : engine.Generate(loaded, gen))
              : engine.GenerateDraft(loaded, gen, draft_path);
    }
    if (!generated) {
      log.Warn("cli", std::string("generation failed (") +
                          std::string(tessera::ToString(generated.error())) +
                          ")");
      return kExitError;
    }
    const tessera::Tokenizer* tokenizer = loaded.GetTokenizer();
    if (tokenizer == nullptr) {
      log.Warn("cli", "cannot decode the output: this model has no tokenizer");
    } else {
      auto input_text = tokenizer->Decode(gen.prompt_tokens);
      auto output_text = tokenizer->Decode(*generated);
      if (input_text) {
        log.Info("cli", "input text: " + *input_text);
      }
      if (output_text) {
        if (!streamed_output) {
          std::fwrite(output_text->data(), 1, output_text->size(), stdout);
          std::fputc('\n', stdout);
          std::fflush(stdout);
        }
        // Split the Qwen thinking block (if any) from the response.
        const std::string close_marker = "</think>";
        const std::size_t close = output_text->find(close_marker);
        if (close == std::string::npos) {
          log.Info("cli", "output text: " + *output_text);
        } else {
          log.Info("cli", "thinking: " + output_text->substr(0, close));
          log.Info("cli", "response: " +
                              output_text->substr(close + close_marker.size()));
        }
      }
    }
  }
  return kExitOk;
}
