#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
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

namespace models::qwen3_5 {

// The Qwen3.5 module: hybrid gated attention interleaved with gated-delta
// linear attention, plus the nextn multi-token-prediction draft head. See
// AGENTS.md. The trunk is in trunk.cpp and trunk_batch.cpp, the MTP head
// in mtp.cpp.
class Qwen35Architecture final : public Architecture {
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
      const Buffer* embeddings) const override;

  [[nodiscard]] std::expected<DraftVerification, StatusCode> Verify(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const std::uint32_t> draft,
      std::span<const float> prefix_logits,
      std::vector<float>* hidden_out) const override;

  [[nodiscard]] std::expected<std::uint32_t, StatusCode> Draft(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const float> hidden, std::uint32_t token, std::uint64_t pos,
      std::vector<float>* mtp_hidden_out) const override;
};

// Build the module (registered for general.architecture "qwen35").
[[nodiscard]] std::unique_ptr<Architecture> MakeQwen35Architecture();

}  // namespace models::qwen3_5
}  // namespace tessera
