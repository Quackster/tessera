#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"

// The Qwen3.8-Flash-Next MTP head weights, loaded from the directory the
// fetch (tessera-cli fetch-mtp) wrote. The checkpoint stores the head as
// BF16 (F32 for the RMSNorm gammas and the shared-expert gate). The loader
// converts the RMSNorm gammas to the engine's convention by adding 1 (the
// model uses GemmaRMSNorm, and the qwen4exp trunk's norm kernels do not add
// 1; the main GGUF carries the same convention), keeps the projections and
// experts as BF16 device bytes for the GEMM kernels, and lifts the gate to
// F32.
//
// Thread-safety: load on one thread; the buffers are read-only afterwards
// and safe for concurrent reads.
namespace tessera::models::qwen4exp {

struct MtpWeights {
  // Every MTP tensor by its checkpoint name, ready for the forward.
  std::unordered_map<std::string, std::unique_ptr<Buffer>> tensors;
  // Expert count, routed intermediate width and hidden width.
  std::size_t experts = 0;
  std::size_t inter = 0;
  std::size_t hidden = 0;

  // The device buffer for a checkpoint tensor name, or nullptr when absent.
  [[nodiscard]] const Buffer* Get(std::string_view name) const;
};

// Load `dir` (from tessera-cli fetch-mtp). FileNotFound when the manifest or
// a tensor is missing; MalformedFile for a bad manifest, a wrong byte size
// or an unsupported dtype; OutOfMemory on upload failure.
[[nodiscard]] std::expected<MtpWeights, StatusCode> LoadMtpWeights(
    Backend& backend, const std::filesystem::path& dir);

}  // namespace tessera::models::qwen4exp
