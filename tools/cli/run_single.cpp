#include "run_single.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cli_engine.hpp"
#include "cli_helpers.hpp"
#include "tessera/calibrate.hpp"
#include "tessera/engine.hpp"
#include "tessera/image.hpp"
#include "tessera/model.hpp"
#include "tessera/runner.hpp"
#include "tessera/tokenizer.hpp"
#include "tessera/types.hpp"
#include "tessera/vision.hpp"

namespace tessera::cli {
namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;

// The plain generation request the decode flags select, shared by the
// parallel-session loop and the one-shot path.
GenerateOptions BuildGenerateOptions(const CliOptions& options, bool sample,
                                     const SamplingOptions& sampling,
                                     const CalibrationConfig& applied) {
  GenerateOptions gen;
  // Zero (the default) fills the remaining context; an explicit budget
  // caps the completion instead.
  gen.max_completion_tokens = options.max_completion_tokens;
  gen.max_thinking_tokens = options.max_thinking_tokens;
  gen.sample = sample;
  gen.sampling = sampling;
  gen.seed = options.seed;
  gen.draft_tokens = applied.draft_tokens;
  gen.prefill_chunk_tokens = applied.prefill_chunk_tokens;
  gen.mxfp4_split_target = applied.mxfp4_split_target;
  gen.mxfp4_split_cap = options.split_cap;
  gen.attention_split = applied.attention_split;
  gen.draft_context = applied.draft_context;
  gen.kv_type = options.kv_type;
  gen.progress_every = options.quiet ? 0 : 64;
  return gen;
}

// The vision path: load the projector, encode the image, prepend its
// tokens and run the multimodal generation. Specific failures are logged
// here; the caller logs the generic generation failure.
std::expected<std::vector<std::uint32_t>, StatusCode> RunVisionGeneration(
    const CliOptions& options, Engine& engine, Model& model,
    GenerateOptions& gen) {
  auto& log = engine.Diagnostics();
  const Tokenizer* tokenizer = model.GetTokenizer();
  if (tokenizer == nullptr) {
    log.Warn("cli", "this model has no tokenizer");
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  auto image_token = tokenizer->SpecialTokenId("<|image_pad|>");
  if (!image_token.has_value()) {
    log.Warn("cli", "no <|image_pad|> token in this model");
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  if (options.mmproj_path.empty()) {
    log.Warn("cli", "--image needs --mmproj");
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (!options.quiet) {
    log.Info("cli", "loading vision projector: " + options.mmproj_path);
  }
  auto vision = VisionModel::Load(engine.Owner(), options.mmproj_path);
  if (!vision) {
    log.Warn("cli", std::string("mmproj load failed (") +
                        std::string(ToString(vision.error())) + ")");
    return std::unexpected(vision.error());
  }
  if (!options.quiet) {
    log.Info("cli", "vision projector loaded: " +
                        std::to_string(vision->Config().block_count) +
                        " blocks, " +
                        std::to_string(vision->Config().projection_dim) +
                        " projection dim");
    log.Info("cli", "loading image: " + options.image_path);
  }
  auto image = LoadPpm(options.image_path);
  if (!image) {
    log.Warn("cli", std::string("image load failed (") +
                        std::string(ToString(image.error())) + ")");
    return std::unexpected(image.error());
  }
  auto resized = ResizeBilinear(*image, vision->Config().image_size,
                                vision->Config().image_size);
  if (!options.quiet) {
    log.Info("cli", "image loaded: " + std::to_string(image->width) + "x" +
                        std::to_string(image->height) + ", encoding");
  }
  auto embeddings = vision->Encode(engine.Owner(), resized.pixels,
                                   resized.height, resized.width);
  if (!embeddings) {
    log.Warn("cli", std::string("image encode failed (") +
                        std::string(ToString(embeddings.error())) + ")");
    return std::unexpected(embeddings.error());
  }
  const std::size_t count =
      embeddings->size() / vision->Config().projection_dim;
  if (!options.quiet) {
    log.Info("cli", "image encoded: " + std::to_string(count) + " token(s)");
    // The multimodal path logs no row progress of its own, so the hook
    // shows the image-token prefill before decoding starts.
    gen.prefill_progress = [&log](std::size_t done, std::size_t total) {
      log.Info("cli", "prefill: " + std::to_string(done) + "/" +
                          std::to_string(total) + " rows");
    };
  }
  std::vector<std::uint32_t> prompt(count, *image_token);
  prompt.insert(prompt.end(), gen.prompt_tokens.begin(),
                gen.prompt_tokens.end());
  gen.prompt_tokens = std::move(prompt);
  return engine.GenerateMultimodal(model, gen, *embeddings, count,
                                   *image_token);
}

}  // namespace

int RunSingleDeviceGeneration(const CliOptions& options) {
  const int device = SelectedDevices(options).front();
  auto engine_result = MakeCliEngine(options, device);
  if (!engine_result) {
    std::fprintf(stderr, "cli: engine create failed (%s)\n",
                 ToString(engine_result.error()).data());
    return kExitError;
  }
  Engine& engine = **engine_result;
  auto& log = engine.Diagnostics();
  AttachCliStrategy(engine, options);

  // Apply a calibrated paging value before the load: an MoE target larger
  // than VRAM only loads paged, and paging is a load-time setting. An
  // explicit --gpu-layers always wins.
  const std::string_view strategy = CliStrategyLabel(options);
  std::size_t gpu_layers = options.gpu_layers;
  if (!options.calibration_path.empty()) {
    const std::size_t explicit_gpu_layers = gpu_layers;
    gpu_layers = ResolveCalibrationGpuLayers(
        engine, options.model_path, options.context, options.kv_type, strategy,
        options.calibration_path, gpu_layers);
    if (gpu_layers != explicit_gpu_layers) {
      log.Info("cli",
               "calibration: --gpu-layers " + std::to_string(gpu_layers));
    }
  }
  if (gpu_layers != 0) {
    // Refuse a paged load whose resident set plus KV cache cannot fit,
    // before the weights start moving.
    if (!CheckGpuLayersFit(engine, options.model_path, options.context,
                           options.kv_type, strategy, gpu_layers, device)) {
      return kExitError;
    }
  }
  // Refuse a paged load whose host weight copy cannot fit system RAM,
  // before the weights start moving.
  std::uint64_t host_bytes = 0;
  if (!CheckHostRamFit(engine, options.model_path, gpu_layers, strategy, 0,
                       device, &host_bytes)) {
    return kExitError;
  }
  ModelOptions model_options{options.model_path, options.context};
  model_options.gpu_layers = gpu_layers;
  model_options.mtp_path = options.mtp_path;
  auto model = engine.LoadModel(model_options);
  if (!model) {
    std::fprintf(stderr, "cli: model load failed (%s)\n",
                 ToString(model.error()).data());
    return kExitError;
  }
  Model& loaded = **model;
  LogModelSummary(engine, loaded);
  // Apply the architecture's recommended sampler unless the user set one.
  bool sample = options.sample;
  SamplingOptions sampling = options.sampling;
  if (ApplyRecommendedSampling(loaded, options.enable_thinking,
                               options.sampling_explicit, &sampling)) {
    sample = true;
    if (!options.quiet) {
      log.Info("cli", "sampling: architecture default (thinking " +
                          std::string(options.enable_thinking ? "on" : "off") +
                          ")");
    }
  }
  // Resolve any saved calibration (explicit CLI values win) before
  // building the request options.
  CalibrationConfig applied;
  applied.prefill_chunk_tokens = options.prefill_chunk;
  applied.draft_tokens = options.draft_block;
  applied.mxfp4_split_target = options.split_target;
  applied.attention_split = options.attention_split;
  applied.draft_context = options.draft_context;
  if (!options.calibration_path.empty()) {
    applied =
        ResolveCalibration(engine, loaded, options.context, options.kv_type,
                           strategy, options.calibration_path, applied);
  }
  if (options.prompt_texts.size() >= 2) {
    // Parallel sessions: each --prompt-text is one session. The pool
    // time-slices the device, so both decode and stream together.
    if (!options.image_path.empty() || !options.mmproj_path.empty()) {
      std::fprintf(stderr, "cli: parallel sessions need no --image\n");
      return kExitUsage;
    }
    if (!options.draft_path.empty() || options.speculate) {
      std::fprintf(stderr,
                   "cli: parallel sessions need no --draft/--speculate\n");
      return kExitUsage;
    }
    if (options.max_thinking_tokens > 0) {
      std::fprintf(stderr,
                   "cli: parallel sessions need no --max-thinking-tokens\n");
      return kExitUsage;
    }
    GenerateOptions base =
        BuildGenerateOptions(options, sample, sampling, applied);
    if (!options.quiet) {
      log.Info("cli", "kv cache: " + std::string(ToString(options.kv_type)));
      log.Info("cli", sample ? "sampling: sampled" : "sampling: greedy");
      log.Info("cli", "parallel sessions: " +
                          std::to_string(options.prompt_texts.size()));
    }
    const Tokenizer* encode_tokenizer = loaded.GetTokenizer();
    if (encode_tokenizer == nullptr) {
      log.Warn("cli", "this model has no tokenizer");
      return kExitError;
    }
    std::vector<GenerateOptions> requests;
    requests.reserve(options.prompt_texts.size());
    for (const std::string& source : options.prompt_texts) {
      auto ids =
          EncodePromptText(source, options.no_chat, *encode_tokenizer, [&] {
            return loaded.ChatPrompt(source, options.enable_thinking,
                                     options.reasoning_effort);
          });
      if (!ids) {
        log.Warn("cli", std::string("prompt encode failed (") +
                            std::string(ToString(ids.error())) + ")");
        return kExitError;
      }
      GenerateOptions one = base;
      one.prompt_tokens = std::move(*ids);
      requests.push_back(std::move(one));
    }
    std::unique_ptr<Engine> engine_ptr = std::move(*engine_result);
    std::unique_ptr<Model> model_ptr = std::move(*model);
    RunnerPool pool;
    pool.Add(std::make_unique<SingleDeviceRunner>(
        std::move(engine_ptr), std::move(model_ptr), device));
    const Tokenizer* stream_tokenizer = pool.At(0).tokenizer();
    std::mutex print_mutex;
    auto results = RunConcurrentSessions(
        pool, requests, [&](std::size_t session, std::uint32_t token) {
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
        pool_log.Warn(
            "cli", "session " + std::to_string(i) + " failed (" +
                       std::string(ToString(results[i].outcome.error())) + ")");
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
  const bool wants_generation =
      !options.prompt_texts.empty() || !options.image_path.empty() ||
      options.max_completion_tokens > 0 || options.max_thinking_tokens > 0 ||
      options.enable_thinking || !options.reasoning_effort.empty();
  if (!wants_generation) {
    return kExitOk;
  }
  const std::string prompt_text = options.prompt_texts.empty()
                                      ? std::string()
                                      : options.prompt_texts.front();
  GenerateOptions gen =
      BuildGenerateOptions(options, sample, sampling, applied);
  if (!options.quiet) {
    log.Info("cli", "kv cache: " + std::string(ToString(options.kv_type)));
    if (options.enable_thinking || !options.reasoning_effort.empty()) {
      log.Info(
          "cli",
          "thinking: enabled" +
              (options.reasoning_effort.empty()
                   ? std::string()
                   : " (reasoning_effort=" + options.reasoning_effort + ")"));
    }
    if (options.sample) {
      log.Info("cli", "sampling: " + DescribeSampling(sampling) +
                          " seed=" + std::to_string(options.seed));
    } else {
      log.Info("cli", "sampling: greedy");
    }
  }
  if (prompt_text.empty()) {
    log.Warn("cli", "--prompt-text is required");
    return kExitUsage;
  }
  const Tokenizer* tokenizer = loaded.GetTokenizer();
  if (tokenizer == nullptr) {
    log.Warn("cli", "this model has no tokenizer; text input is unavailable");
    return kExitError;
  }
  auto ids = EncodePromptText(prompt_text, options.no_chat, *tokenizer, [&] {
    return loaded.ChatPrompt(prompt_text, options.enable_thinking,
                             options.reasoning_effort);
  });
  if (!ids) {
    log.Warn("cli", std::string("prompt encode failed (") +
                        std::string(ToString(ids.error())) + ")");
    return kExitError;
  }
  gen.prompt_tokens = std::move(*ids);

  std::expected<std::vector<std::uint32_t>, StatusCode> generated;
  bool streamed_output = false;
  if (!options.image_path.empty()) {
    generated = RunVisionGeneration(options, engine, loaded, gen);
  } else if (options.draft_path.empty() && !options.speculate) {
    // Stream plain greedy decode so each output token reaches the
    // console as it is generated. A blocking Generate would return
    // only after every step (by default the whole remaining context),
    // which looks hung after the prefill line.
    std::vector<std::uint32_t> streamed_ids;
    if (!options.quiet) {
      const std::size_t budget = loaded.EffectiveMaxTokens(
          gen.prompt_tokens.size(), gen.max_completion_tokens);
      log.Info("cli", "generating up to " + std::to_string(budget) +
                          " token(s); streaming output below");
    }
    auto count =
        engine.GenerateStreaming(loaded, gen, [&](std::uint32_t token) {
          streamed_ids.push_back(token);
          auto piece =
              tokenizer->Decode(std::span<const std::uint32_t>(&token, 1));
          if (piece) {
            std::fwrite(piece->data(), 1, piece->size(), stdout);
            std::fflush(stdout);
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
        options.draft_path.empty()
            ? (options.speculate ? engine.GenerateSpeculative(loaded, gen)
                                 : engine.Generate(loaded, gen))
            : engine.GenerateDraft(loaded, gen, options.draft_path);
  }
  if (!generated) {
    log.Warn("cli", std::string("generation failed (") +
                        std::string(ToString(generated.error())) + ")");
    return kExitError;
  }
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
      log.Info("cli",
               "response: " + output_text->substr(close + close_marker.size()));
    }
  }
  return kExitOk;
}

}  // namespace tessera::cli
