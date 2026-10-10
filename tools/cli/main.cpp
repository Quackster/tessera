#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "calibrate_command.hpp"
#include "cli_common.hpp"
#include "cli_helpers.hpp"
#include "tessera/calibrate.hpp"
#include "tessera/engine.hpp"
#include "tessera/image.hpp"
#include "tessera/vision.hpp"
#include "tessera/serve.hpp"
#include "tessera/speculative.hpp"
#include "tessera/types.hpp"

namespace {

using tessera::cli::kDefaultContext;
using tessera::cli::kDefaultDraftBlock;
using tessera::cli::kDefaultPrefillChunk;
using tessera::cli::LogModelSummary;
using tessera::cli::MatchKvType;
using tessera::cli::PrintUsage;
using tessera::cli::ResolveCalibration;

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;

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
  if (command != "run" && command != "serve") {
    PrintUsage();
    return kExitUsage;
  }
  std::string model_path;
  std::string draft_path;
  std::string prompt_text;
  std::string calibration_path;
  std::string mmproj_path;
  std::string image_path;
  std::uint32_t image_token = 0;
  bool quiet = false;
  bool no_chat = false;
  std::string host = "127.0.0.1";
  bool auto_title = true;
  std::vector<std::string> api_keys;
  std::vector<std::string> allow_origins;
  bool speculate = false;
  tessera::SamplingOptions sampling;
  bool sample = false;
  tessera::KvCacheType kv_type = tessera::KvCacheType::F32;
  std::uint64_t seed = 0;
  std::uint16_t port = 8080;
  std::size_t max_completion_tokens = 0;
  std::size_t max_thinking_tokens = 0;
  std::size_t context = kDefaultContext;
  std::size_t draft_block = kDefaultDraftBlock;
  std::size_t prefill_chunk = kDefaultPrefillChunk;
  int gpu = 0;
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--draft" && i + 1 < argc) {
      draft_path = argv[++i];
    } else if (arg == "--prompt-text" && i + 1 < argc) {
      prompt_text = argv[++i];
    } else if (arg == "--mmproj" && i + 1 < argc) {
      mmproj_path = argv[++i];
    } else if (arg == "--image" && i + 1 < argc) {
      image_path = argv[++i];
    } else if (arg == "--quiet") {
      quiet = true;
    } else if (arg == "--no-chat") {
      no_chat = true;
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
    } else if (arg == "--context" && i + 1 < argc) {
      context = std::stoul(argv[++i]);
    } else if (arg == "--gpu" && i + 1 < argc) {
      gpu = std::stoi(argv[++i]);
    } else if (arg == "--calibration" && i + 1 < argc) {
      calibration_path = argv[++i];
    } else if (arg == "--draft-block" && i + 1 < argc) {
      draft_block = std::stoul(argv[++i]);
    } else if (arg == "--prefill-chunk" && i + 1 < argc) {
      prefill_chunk = std::stoul(argv[++i]);
    } else if (arg == "--speculate") {
      speculate = true;
    } else if (arg == "--sample") {
      sample = true;
    } else if (MatchKvType(arg, kv_type)) {
      // handled by MatchKvType
    } else if (arg == "--temperature" && i + 1 < argc) {
      sampling.temperature = std::stof(argv[++i]);
    } else if (arg == "--top-p" && i + 1 < argc) {
      sampling.top_p = std::stof(argv[++i]);
    } else if (arg == "--top-k" && i + 1 < argc) {
      sampling.top_k = std::stoi(argv[++i]);
    } else if (arg == "--min-p" && i + 1 < argc) {
      sampling.min_p = std::stof(argv[++i]);
    } else if (arg == "--presence-penalty" && i + 1 < argc) {
      sampling.presence_penalty = std::stof(argv[++i]);
    } else if (arg == "--repetition-penalty" && i + 1 < argc) {
      sampling.repetition_penalty = std::stof(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = std::stoull(argv[++i]);
    } else if (arg == "--max-completion-tokens" && i + 1 < argc) {
      max_completion_tokens = std::stoul(argv[++i]);
    } else if (arg == "--max-thinking-tokens" && i + 1 < argc) {
      max_thinking_tokens = std::stoul(argv[++i]);
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

  tessera::EngineOptions options;  // default stderr diagnostics sink
  options.device_index = gpu;
  options.prefill_chunk_tokens = prefill_chunk;
  auto created = tessera::Engine::Create(options);
  if (!created) {
    std::fprintf(stderr, "cli: engine create failed (%s)\n",
                 tessera::ToString(created.error()).data());
    return kExitError;
  }
  tessera::Engine& engine = **created;
  auto& log = engine.Diagnostics();

  // Attach the speculative strategy before serving or generating. DFlash2
  // drafts from its checkpoint; MTP drafts from the target's own nextn
  // head, so it needs no path. Both the serve path and the plain generate
  // path read this one attachment, so --speculate serves as it runs.
  if (!draft_path.empty() || speculate) {
    std::unique_ptr<tessera::SpeculativeStrategy> strategy =
        draft_path.empty() ? tessera::CreateMtpStrategy()
                           : tessera::CreateDFlash2Strategy();
    const std::string label =
        draft_path.empty() ? "MTP" : "DFlash2 ('" + draft_path + "')";
    auto attach =
        strategy->Attach(tessera::StrategyOptions{draft_path, draft_block});
    if (!attach) {
      log.Warn("cli", label + " attach failed: " +
                           std::string(tessera::ToString(attach.error())) +
                           "; continuing without speculative decoding");
    } else {
      auto attached = engine.AttachSpeculative(std::move(strategy));
      if (!attached) {
        log.Warn("cli",
                 label + " strategy attach failed: " +
                     std::string(tessera::ToString(attached.error())) +
                     "; continuing without speculative decoding");
      }
    }
  }

  if (command == "serve") {
    tessera::ServeOptions serve_options;
    serve_options.host = host;
    serve_options.port = port;
    serve_options.api_keys = api_keys;
    serve_options.allow_origins = allow_origins;
    serve_options.auto_title = auto_title;
    if (!calibration_path.empty()) {
      // The calibration key depends on the loaded model, so resolve the
      // saved settings up front and serve the already-loaded model. Without
      // --calibration the deferred loader below binds the socket first.
      auto calibration_model =
          engine.LoadModel(tessera::ModelOptions{model_path, context});
      if (!calibration_model) {
        log.Error("cli", std::string("model load failed (") +
                             std::string(tessera::ToString(
                                 calibration_model.error())) +
                             ")");
        return kExitError;
      }
      LogModelSummary(engine, **calibration_model);
      const std::string strategy =
          draft_path.empty() ? (speculate ? "mtp" : "none") : "dflash2";
      tessera::CalibrationConfig explicit_config;
      explicit_config.prefill_chunk_tokens = prefill_chunk;
      explicit_config.draft_tokens = draft_block;
      serve_options.calibration = ResolveCalibration(
          engine, **calibration_model, context, kv_type, strategy,
          calibration_path, explicit_config);
      tessera::GenerateOptions warm;
      warm.max_completion_tokens = 1;
      warm.progress_every = 0;
      tessera::ApplyToGenerateOptions(serve_options.calibration, warm);
      auto warmed = engine.Generate(**calibration_model, warm);
      if (!warmed) {
        log.Error("cli", std::string("kernel warmup failed (") +
                             std::string(tessera::ToString(warmed.error())) +
                             "); the model cannot serve");
        return kExitError;
      }
      auto served = tessera::Serve(engine, **calibration_model, serve_options);
      if (!served) {
        log.Warn("cli", std::string("serve failed (") +
                            std::string(tessera::ToString(served.error())) +
                            ")");
        return kExitError;
      }
      return kExitOk;
    }
    // Load and warm on the serve thread: the server binds first, so a
    // client can reach /health while the weights and kernels load.
    auto loader = [&](const tessera::ServeProgress& report)
        -> std::expected<std::unique_ptr<tessera::Model>,
                         tessera::StatusCode> {
      report("loading weights");
      auto loaded = engine.LoadModel(tessera::ModelOptions{model_path, context});
      if (!loaded) {
        return std::unexpected(loaded.error());
      }
      LogModelSummary(engine, **loaded);
      report("warming up kernels");
      tessera::GenerateOptions warm;
      warm.max_completion_tokens = 1;
      warm.progress_every = 0;
      auto warmed = engine.Generate(**loaded, warm);
      if (!warmed) {
        log.Error("cli", std::string("kernel warmup failed (") +
                             std::string(tessera::ToString(warmed.error())) +
                             "); the model cannot serve");
        return std::unexpected(warmed.error());
      }
      report("ready");
      return std::move(*loaded);
    };
    auto served = tessera::Serve(engine, serve_options, std::move(loader));
    if (!served) {
      log.Warn("cli", std::string("serve failed (") +
                          std::string(tessera::ToString(served.error())) + ")");
      return kExitError;
    }
    return kExitOk;
  }

  auto model = engine.LoadModel(tessera::ModelOptions{model_path, context});
  if (!model) {
    std::fprintf(stderr, "cli: model load failed (%s)\n",
                 tessera::ToString(model.error()).data());
    return kExitError;
  }
  tessera::Model& loaded = **model;
  LogModelSummary(engine, loaded);
  // Resolve any saved calibration (explicit CLI values win) before building
  // the request options.
  tessera::CalibrationConfig applied;
  applied.prefill_chunk_tokens = prefill_chunk;
  applied.draft_tokens = draft_block;
  if (!calibration_path.empty()) {
    const std::string strategy =
        draft_path.empty() ? (speculate ? "mtp" : "none") : "dflash2";
    applied = ResolveCalibration(engine, loaded, context, kv_type, strategy,
                                 calibration_path, applied);
  }
  if (!prompt_text.empty() || !image_path.empty() ||
      max_completion_tokens > 0 || max_thinking_tokens > 0) {
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
    gen.kv_type = kv_type;
    gen.progress_every = quiet ? 0 : 64;
    if (!quiet) {
      log.Info("cli", "kv cache: " +
                          std::string(tessera::ToString(kv_type)));
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
                      : loaded.ChatPrompt(prompt_text);
      if (chat.has_value()) {
        text = *chat;
        if (!quiet) {
          log.Info("cli", "applied the model chat template to the prompt");
        }
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
