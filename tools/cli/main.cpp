#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

#include "tessera/engine.hpp"
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
               "       [--prompt <id>] [--tokens <n>]\n"
               "  --model <path>    a .gguf file or an MXFP4 model directory\n"
               "  --draft <dir>     DFlash2 draft checkpoint directory\n"
               "  --context <n>     maximum context length (default %zu)\n"
               "  --draft-block <n> draft block tokens (default %zu)\n"
               "  --prompt <id>     first token id for generation (default 0)\n"
               "  --tokens <n>      run n greedy decode steps and print them\n",
               kDefaultContext, kDefaultDraftBlock);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || std::string_view(argv[1]) != "run") {
    PrintUsage();
    return kExitUsage;
  }
  std::string model_path;
  std::string draft_path;
  std::uint32_t prompt = 0;
  std::size_t tokens = 0;
  std::size_t context = kDefaultContext;
  std::size_t draft_block = kDefaultDraftBlock;
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--draft" && i + 1 < argc) {
      draft_path = argv[++i];
    } else if (arg == "--context" && i + 1 < argc) {
      context = std::stoul(argv[++i]);
    } else if (arg == "--draft-block" && i + 1 < argc) {
      draft_block = std::stoul(argv[++i]);
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt = static_cast<std::uint32_t>(std::stoul(argv[++i]));
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

  tessera::EngineOptions options;  // default stderr diagnostics sink
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
  if (tokens > 0) {
    auto generated =
        engine.Generate(loaded, tessera::GenerateOptions{tokens, prompt});
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
    log.Info("cli", "generated tokens: " + produced);
  }
  return kExitOk;
}
