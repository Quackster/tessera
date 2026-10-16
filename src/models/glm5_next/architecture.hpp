#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
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
}  // namespace tessera

namespace tessera::models::glm5_next {

// The GLM-5-Next module (GLM-5.3-Flash): KDA gated delta-rule
// linear-attention layers interleaved with NoPE MLA full-attention
// layers that attend to a k-pool selected key subset, a sparse MoE FFN
// behind manifold-constrained hyper-connections (mHC), and a
// next-token-prediction (MTP) draft block.
//
// Current state: configuration and weight-name mapping are in place;
// the trunk (trunk.cpp), the KDA/MLA/indexer kernels (kernels/) and the
// MTP head follow. Until they land the decode entry points reject with
// UnsupportedFeature, so the module is not selectable for generation.
class Glm5NextArchitecture final : public Architecture {
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
      std::vector<float>* hidden_out, bool all_logits, const Buffer* embeddings,
      const std::vector<std::size_t>* capture_layers = nullptr,
      std::vector<Buffer*>* capture = nullptr,
      bool snapshot_state = false) const override;

  [[nodiscard]] std::expected<DraftVerification, StatusCode> Verify(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const std::uint32_t> draft,
      std::span<const float> prefix_logits, std::optional<std::uint32_t> anchor,
      std::vector<float>* hidden_out,
      const std::vector<std::size_t>* capture_layers = nullptr,
      std::vector<Buffer*>* capture = nullptr) const override;

  [[nodiscard]] std::expected<std::uint32_t, StatusCode> Draft(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const float> hidden, std::uint32_t token, std::uint64_t pos,
      std::vector<float>* mtp_hidden_out) const override;

  [[nodiscard]] std::size_t DraftRows(
      const core::DecodeCache& cache) const override;

  void DraftTruncate(core::DecodeCache& cache, std::size_t rows) const override;

  // Parse the HuggingFace config.json (the multimodal checkpoint root or
  // its text_config) into the generic TransformerConfig, including the
  // GLM-specific MLA, KDA, indexer, mHC and MoE fields. Also validates
  // through GeometryOf, so a definition the trunk cannot run rejects as
  // MalformedFile here.
  [[nodiscard]] std::expected<TransformerConfig, StatusCode> ParseConfigJson(
      std::string_view json) const override;

  // Map HuggingFace safetensors names (a GLM checkpoint) to the internal
  // GGUF "blk.N.*" names. The per-expert routed tensors of an MoE
  // checkpoint are not mapped yet: the loader has no per-expert packing
  // into the stacked internal tensor, so they are ignored until it lands.
  [[nodiscard]] std::optional<std::string> MapWeightName(
      std::string_view name, const TransformerConfig& config,
      const WeightLoadOptions& options) const override;

  // A_log is stored in log space; the internal form is F32 -exp(A_log).
  [[nodiscard]] std::optional<WeightConversion> ConvertWeight(
      std::string_view internal_name,
      const TransformerConfig& config) const override;
};

std::unique_ptr<Architecture> MakeGlm5NextArchitecture();

}  // namespace tessera::models::glm5_next
