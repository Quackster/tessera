#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "tessera/architecture.hpp"

namespace tessera {
class Backend;
class Buffer;
class Model;
namespace core {
struct DecodeCache;
}  // namespace core

namespace models::qwen4exp {

// The Qwen3.8-Flash-Next (qwen4exp) module: a hybrid trunk whose mixer is a
// gated-delta linear layer on most blocks and a sparse QSA attention on
// every fourth, wrapped in a gated residual stack (hyper-connection), with
// a per-layer n-gram embedding (PLE) and a sparse mixture-of-experts FFN.
// See docs/QWEN_FLASH_NEXT.md and AGENTS.md. The trunk is in trunk.cpp,
// the expert feed-forward in moe.cpp and the weight roles in weights.cpp.
class Qwen4ExpArchitecture final : public Architecture {
 public:
  [[nodiscard]] std::string_view Name() const override;

  [[nodiscard]] std::expected<void, StatusCode> Forward(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::uint32_t token, std::vector<float>* hidden_out,
      const std::vector<std::size_t>* capture_layers,
      std::vector<Buffer*>* capture, const Buffer* embedding) const override;

  [[nodiscard]] std::expected<std::vector<float>, StatusCode> Logits(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::uint32_t token, std::vector<float>* hidden_out,
      const std::vector<std::size_t>* capture_layers,
      std::vector<Buffer*>* capture, const Buffer* embedding) const override;

  [[nodiscard]] std::expected<void, StatusCode> ForwardBatch(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const std::uint32_t> tokens, std::vector<float>* logits_out,
      std::vector<float>* hidden_out, bool all_logits,
      const Buffer* embeddings,
      const std::vector<std::size_t>* capture_layers,
      std::vector<Buffer*>* capture,
      bool snapshot_state = false) const override;

  [[nodiscard]] std::expected<DraftVerification, StatusCode> Verify(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const std::uint32_t> draft,
      std::span<const float> prefix_logits,
      std::optional<std::uint32_t> anchor, std::vector<float>* hidden_out,
      const std::vector<std::size_t>* capture_layers,
      std::vector<Buffer*>* capture) const override;

  [[nodiscard]] std::expected<std::uint32_t, StatusCode> Draft(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const float> hidden, std::uint32_t token, std::uint64_t pos,
      std::vector<float>* mtp_hidden_out) const override;

  [[nodiscard]] std::size_t DraftRows(
      const core::DecodeCache& cache) const override;
  void DraftTruncate(core::DecodeCache& cache, std::size_t rows) const override;

  [[nodiscard]] std::expected<void, StatusCode> EnsureReady(
      Backend& backend, const Model& model,
      core::DecodeCache& cache) const override;

  [[nodiscard]] std::optional<WeightConversion> ConvertWeight(
      std::string_view internal_name,
      const TransformerConfig& config) const override;

  [[nodiscard]] ParallelKind TensorParallelKind(
      std::string_view internal_name) const override;
};

// Build the module (registered for general.architecture "qwen4exp").
[[nodiscard]] std::unique_ptr<Architecture> MakeQwen4ExpArchitecture();

}  // namespace models::qwen4exp
}  // namespace tessera
