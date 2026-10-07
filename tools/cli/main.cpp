#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

#include "core/decode.hpp"
#include "tessera/engine.hpp"
#include "tessera/speculative.hpp"
#include "tessera/types.hpp"

namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;
// How many tensor names to print when summarizing a model.
constexpr std::size_t kPrintedTensorNames = 5;

void PrintUsage() {
  std::fprintf(stderr,
               "usage: tessera-cli run --model <path> [--draft <dir>]\n"
               "       [--prompt <id>] [--tokens <n>]\n"
               "  --model <path>  a .gguf file or an MXFP4 model directory\n"
               "  --draft <dir>  DFlash2 draft checkpoint directory\n"
               "  --prompt <id>  first token id for generation (default 0)\n"
               "  --tokens <n>   run n greedy decode steps and print them\n");
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
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--draft" && i + 1 < argc) {
      draft_path = argv[++i];
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
        strategy->Attach(tessera::StrategyOptions{draft_path, 4});
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

  auto model =
      engine.LoadModel(tessera::ModelOptions{model_path, 4096});
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
    tessera::core::DecodeCache cache;
    std::uint32_t next = prompt;
    std::string produced;
    for (std::size_t step = 0; step < tokens; ++step) {
      auto decoded =
          tessera::core::DecodeStep(engine.Owner(), loaded, cache, next);
      if (!decoded) {
        log.Warn("cli", std::string("decode step ") +
                            std::to_string(step) + " failed (" +
                            std::string(tessera::ToString(decoded.error())) +
                            ")");
        return kExitError;
      }
      next = *decoded;
      produced += std::to_string(next) + " ";
    }
    log.Info("cli", "generated tokens: " + produced);
  }
  return kExitOk;
}
