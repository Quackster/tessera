#include "cli_helpers.hpp"

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>

#include "cli_common.hpp"
#include "tessera/calibrate.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/serve.hpp"
#include "tessera/types.hpp"

namespace tessera::cli {
namespace {

// How many tensor names to print when summarizing a model.
constexpr std::size_t kPrintedTensorNames = 5;

}  // namespace

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
               "       --prompt-text <str> [--max-completion-tokens <n>]\n"
               "       tessera-cli serve --model <path> [--host <ip>] "
               "[--port <n>]\n"
               "       tessera-cli calibrate --model <path> [--draft <dir>]\n"
               "       [--context <n>] [--calibration <file>] [--gpu <n>]\n"
               "       [--prompts <file>]\n"
               "       tessera-cli --list-gpus\n"
               "  --list-gpus       list the GPUs the backend sees and exit\n"
               "  --model <path>    a .gguf file or an MXFP4 model directory\n"
               "  --draft <dir>     DFlash2 draft checkpoint directory\n"
               "  --context <n>     maximum context length (default %zu)\n"
               "  --gpu <n>         GPU index to use (default 0, the first)\n"
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
               "  --prompt-text <s> text prompt (tokenized; needs a "
               "tokenizer)\n"
               "  --max-completion-tokens <n> completion tokens (0 fills the "
               "remaining context, default %zu)\n"
               "  --max-thinking-tokens <n> think-block budget per turn (0 "
               "leaves thinking unlimited, default 0)\n"
               "  --speculate       draft with the MTP head (run and serve)\n"
               "  --mmproj <path>   vision projector (mmproj) GGUF\n"
               "  --image <path>    image (binary PPM) to prepend as tokens\n"
               "  --quiet           suppress progress and info logs\n"
               "  --no-chat         do not apply the chat template\n"
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
               "  --kv-fp8          store the KV cache in FP8 E4M3\n"
               "  --host <ip>       serve bind address (default 127.0.0.1)\n"
               "  --port <n>        serve port (default 8080)\n"
               "  --no-auto-title   keep the first user line as the chat title "
               "instead of asking the model\n"
               "  --api-key <k>     accepted API key (repeatable; env "
               "TESSERA_API_KEY)\n"
               "  --allow-origin <o> CORS origin (repeatable; * allows all)\n",
               kDefaultContext, kDefaultDraftBlock, kDefaultPrefillChunk,
               kDefaultAttentionSplitChunks, kDefaultMaxCompletionTokens);
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
  const HardwareKey key =
      MakeHardwareKey(engine.Owner().Name(), engine.Owner().DeviceName(),
                      model_id, context, ToString(kv_type), strategy);
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

}  // namespace tessera::cli
