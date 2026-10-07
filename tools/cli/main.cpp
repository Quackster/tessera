#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "tessera/engine.hpp"
#include "tessera/image.hpp"
#include "tessera/vision.hpp"
#include "tessera/serve.hpp"
#include "tessera/speculative.hpp"
#include "tessera/types.hpp"

namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;
// How many tensor names to print when summarizing a model.
constexpr std::size_t kPrintedTensorNames = 5;
// Defaults for the runtime options; override with the flags below.
constexpr std::size_t kDefaultContext = 4096;
constexpr std::size_t kDefaultDraftBlock = 4;

void PrintUsage() {
  std::fprintf(stderr,
               "usage: tessera-cli run --model <path> [--draft <dir>]\n"
               "       [--context <n>] [--draft-block <n>]\n"
               "       [--prompt <id>] [--prompt-text <str>] [--tokens <n>]\n"
               "       tessera-cli serve --model <path> [--host <ip>] "
               "[--port <n>]\n"
               "       tessera-cli --list-gpus\n"
               "  --list-gpus       list the GPUs the backend sees and exit\n"
               "  --model <path>    a .gguf file or an MXFP4 model directory\n"
               "  --draft <dir>     DFlash2 draft checkpoint directory\n"
               "  --context <n>     maximum context length (default %zu)\n"
               "  --gpu <n>         GPU index to use (default 0, the first)\n"
               "  --draft-block <n> draft block tokens (default %zu)\n"
               "  --prompt <id>     first token id for generation (default 0)\n"
               "  --prompt-text <s> text prompt (tokenized; needs a tokenizer)\n"
               "  --mtp <id>        print the MTP draft after this token id\n"
               "  --tokens <n>      run n greedy decode steps and print them\n"
               "  --speculate       verify MTP drafts instead of plain greedy\n"
               "  --mmproj <path>   vision projector (mmproj) GGUF\n"
               "  --image <path>    image (binary PPM) to prepend as tokens\n"
               "  --image-token <id> placeholder token id for image rows\n"
               "  --quiet           print only the generated tokens\n"
               "  --sample          sample instead of greedy decode\n"
               "  --temperature <f> sampling temperature (default 0.6)\n"
               "  --top-p <f>       nucleus probability (default 0.95)\n"
               "  --top-k <n>       keep the top n tokens (default 20)\n"
               "  --min-p <f>       minimum probability (default 0.0)\n"
               "  --presence-penalty <f>   presence penalty (default 0.0)\n"
               "  --repetition-penalty <f> repetition penalty (default 1.0)\n"
               "  --seed <n>        sampling RNG seed (default 0)\n"
               "  --kv-f16          store the KV cache in fp16 (default fp32)\n"
               "  --kv-q8           store the KV cache in int8\n"
               "  --kv-q4           store the KV cache in 4-bit\n"
               "  --host <ip>       serve bind address (default 127.0.0.1)\n"
               "  --port <n>        serve port (default 8080)\n"
               "  --api-key <k>     accepted API key (repeatable; env "
               "TESSERA_API_KEY)\n"
               "  --allow-origin <o> CORS origin (repeatable; * allows all)\n",
               kDefaultContext, kDefaultDraftBlock);
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
  if (command != "run" && command != "serve") {
    PrintUsage();
    return kExitUsage;
  }
  std::string model_path;
  std::string draft_path;
  std::string prompt_text;
  std::string mmproj_path;
  std::string image_path;
  std::uint32_t image_token = 0;
  bool quiet = false;
  std::string host = "127.0.0.1";
  std::vector<std::string> api_keys;
  std::vector<std::string> allow_origins;
  std::uint32_t prompt = 0;
  std::uint32_t mtp_token = 0;
  bool mtp = false;
  bool speculate = false;
  tessera::SamplingOptions sampling;
  bool sample = false;
  tessera::KvCacheType kv_type = tessera::KvCacheType::F32;
  std::uint64_t seed = 0;
  std::uint16_t port = 8080;
  std::size_t tokens = 0;
  std::size_t context = kDefaultContext;
  std::size_t draft_block = kDefaultDraftBlock;
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
    } else if (arg == "--image-token" && i + 1 < argc) {
      image_token = static_cast<std::uint32_t>(std::stoul(argv[++i]));
    } else if (arg == "--quiet") {
      quiet = true;
    } else if (arg == "--api-key" && i + 1 < argc) {
      api_keys.emplace_back(argv[++i]);
    } else if (arg == "--allow-origin" && i + 1 < argc) {
      allow_origins.emplace_back(argv[++i]);
    } else if (arg == "--host" && i + 1 < argc) {
      host = argv[++i];
    } else if (arg == "--port" && i + 1 < argc) {
      port = static_cast<std::uint16_t>(std::stoul(argv[++i]));
    } else if (arg == "--context" && i + 1 < argc) {
      context = std::stoul(argv[++i]);
    } else if (arg == "--gpu" && i + 1 < argc) {
      gpu = std::stoi(argv[++i]);
    } else if (arg == "--draft-block" && i + 1 < argc) {
      draft_block = std::stoul(argv[++i]);
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt = static_cast<std::uint32_t>(std::stoul(argv[++i]));
    } else if (arg == "--mtp" && i + 1 < argc) {
      mtp_token = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      mtp = true;
    } else if (arg == "--speculate") {
      speculate = true;
    } else if (arg == "--sample") {
      sample = true;
    } else if (arg == "--kv-f16") {
      kv_type = tessera::KvCacheType::F16;
    } else if (arg == "--kv-q8") {
      kv_type = tessera::KvCacheType::Q8;
    } else if (arg == "--kv-q4") {
      kv_type = tessera::KvCacheType::Q4;
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
    } else if (arg == "--tokens" && i + 1 < argc) {
      tokens = std::stoul(argv[++i]);
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

  tessera::EngineOptions options;  // default stderr diagnostics sink
  options.device_index = gpu;
  auto created = tessera::Engine::Create(options);
  if (!created) {
    std::fprintf(stderr, "cli: engine create failed (%s)\n",
                 tessera::ToString(created.error()).data());
    return kExitError;
  }
  tessera::Engine& engine = **created;
  auto& log = engine.Diagnostics();

  if (!draft_path.empty()) {
    auto strategy = tessera::CreateDFlash2Strategy();
    auto attach =
        strategy->Attach(tessera::StrategyOptions{draft_path, draft_block});
    if (!attach) {
      log.Warn("cli",
               std::string("draft attach failed for '") + draft_path + ": " +
                   std::string(tessera::ToString(attach.error())) +
                   "; continuing without speculative decoding");
    } else {
      auto attached = engine.AttachSpeculative(std::move(strategy));
      if (!attached) {
        log.Warn("cli",
                 std::string("strategy attach failed: ") +
                     std::string(tessera::ToString(attached.error())) +
                     "; continuing without speculative decoding");
      }
    }
  }

  auto model = engine.LoadModel(tessera::ModelOptions{model_path, context});
  if (!model) {
    std::fprintf(stderr, "cli: model load failed (%s)\n",
                 tessera::ToString(model.error()).data());
    return kExitError;
  }
  tessera::Model& loaded = **model;
  const auto& tensors = loaded.Tensors();
  std::size_t total_numel = 0;
  for (const auto& tensor : tensors) {
    total_numel += tensor.shape.Numel();
  }
  std::size_t device_bytes = 0;
  for (const auto& weight : loaded.Weights()) {
    device_bytes += weight.device->Size();
  }
  std::string summary =
      std::string(loaded.Format() == tessera::ModelFormat::Gguf ? "gguf"
                                                               : "mxfp4") +
      " model, " + std::to_string(tensors.size()) + " tensors, " +
      std::to_string(total_numel) + " total elements, " +
      std::to_string(device_bytes) + " bytes on device";
  if (!loaded.Name().empty()) {
    summary = std::string(loaded.Name()) + " (" + summary + ")";
  }
  log.Info("cli", summary);
  for (std::size_t i = 0; i < tensors.size() && i < kPrintedTensorNames; ++i) {
    log.Info("cli", "tensor " + std::to_string(i) + ": " +
                        tensors[i].name + " [" +
                        std::to_string(tensors[i].shape.Numel()) + "]");
  }
  if (command == "serve") {
    log.Info("cli", "serving on " + host + ":" + std::to_string(port));
    tessera::ServeOptions serve_options;
    serve_options.host = host;
    serve_options.port = port;
    serve_options.api_keys = api_keys;
    serve_options.allow_origins = allow_origins;
    auto served = tessera::Serve(engine, loaded, serve_options);
    if (!served) {
      log.Warn("cli", std::string("serve failed (") +
                          std::string(tessera::ToString(served.error())) + ")");
      return kExitError;
    }
    return kExitOk;
  }
  if (mtp) {
    auto draft = engine.MtpDraft(loaded, mtp_token);
    if (!draft) {
      log.Warn("cli", std::string("mtp draft failed (") +
                          std::string(tessera::ToString(draft.error())) + ")");
      return kExitError;
    }
    log.Info("cli", "mtp draft: " + std::to_string(*draft));
  }
  if (tokens > 0) {
    tessera::GenerateOptions gen;
    gen.max_tokens = tokens;
    gen.first_token = prompt;
    gen.sample = sample;
    gen.sampling = sampling;
    gen.seed = seed;
    gen.draft_tokens = draft_block;
    gen.kv_type = kv_type;
    gen.progress_every = quiet ? 0 : 64;
    if (!quiet) {
      log.Info("cli", "kv cache: " +
                          std::string(kv_type == tessera::KvCacheType::F16
                                          ? "fp16"
                                          : kv_type == tessera::KvCacheType::Q8
                                                ? "int8"
                                                : kv_type ==
                                                          tessera::KvCacheType::Q4
                                                      ? "4-bit"
                                                      : "fp32"));
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
    if (!prompt_text.empty()) {
      const tessera::Tokenizer* tokenizer = loaded.GetTokenizer();
      if (tokenizer == nullptr) {
        log.Warn("cli", "this model has no tokenizer; use --prompt <id>");
        return kExitError;
      }
      auto ids = tokenizer->Encode(prompt_text);
      if (!ids) {
        log.Warn("cli", std::string("prompt encode failed (") +
                            std::string(tessera::ToString(ids.error())) + ")");
        return kExitError;
      }
      gen.prompt_tokens = *ids;
    }
    std::expected<std::vector<std::uint32_t>, tessera::StatusCode> generated;
    if (!image_path.empty()) {
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
      }
      std::vector<std::uint32_t> prompt(count, image_token);
      prompt.insert(prompt.end(), gen.prompt_tokens.begin(),
                    gen.prompt_tokens.end());
      gen.prompt_tokens = prompt;
      generated = engine.GenerateMultimodal(loaded, gen, *embeddings, count,
                                            image_token);
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
    std::string produced;
    for (std::uint32_t id : *generated) {
      produced += std::to_string(id) + " ";
    }
    log.Info("cli", std::string(speculate ? "speculative tokens: "
                                         : "generated tokens: ") +
                        produced);
  }
  return kExitOk;
}
