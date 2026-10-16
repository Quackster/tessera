#include "models/glm5_next/architecture.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <vector>

// Decode entry points for the GLM-5-Next module. The trunk (mHC stream
// mixing, KDA layers, sparse MLA layers with the k-pool indexer, MoE
// FFN) and its kernels are not implemented yet, so every entry point
// rejects with UnsupportedFeature. Kill the stubs as each kernel lands;
// until then a GLM model loads (config, weights) but cannot generate.

namespace tessera::models::glm5_next {

std::expected<void, StatusCode> Glm5NextArchitecture::Forward(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out,
    const std::vector<std::size_t>* capture_layers, std::vector<Buffer*>* capture,
    const Buffer* embedding) const {
  (void)backend;
  (void)model;
  (void)cache;
  (void)token;
  (void)hidden_out;
  (void)capture_layers;
  (void)capture;
  (void)embedding;
  return std::unexpected(StatusCode::UnsupportedFeature);
}

std::expected<std::vector<float>, StatusCode> Glm5NextArchitecture::Logits(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out,
    const std::vector<std::size_t>* capture_layers, std::vector<Buffer*>* capture,
    const Buffer* embedding) const {
  (void)backend;
  (void)model;
  (void)cache;
  (void)token;
  (void)hidden_out;
  (void)capture_layers;
  (void)capture;
  (void)embedding;
  return std::unexpected(StatusCode::UnsupportedFeature);
}

std::expected<void, StatusCode> Glm5NextArchitecture::ForwardBatch(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::span<const std::uint32_t> tokens, std::vector<float>* logits_out,
    std::vector<float>* hidden_out, bool all_logits, const Buffer* embeddings,
    const std::vector<std::size_t>* capture_layers, std::vector<Buffer*>* capture,
    bool snapshot_state) const {
  (void)backend;
  (void)model;
  (void)cache;
  (void)tokens;
  (void)logits_out;
  (void)hidden_out;
  (void)all_logits;
  (void)embeddings;
  (void)capture_layers;
  (void)capture;
  (void)snapshot_state;
  return std::unexpected(StatusCode::UnsupportedFeature);
}

std::expected<DraftVerification, StatusCode> Glm5NextArchitecture::Verify(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::span<const std::uint32_t> draft, std::span<const float> prefix_logits,
    std::optional<std::uint32_t> anchor, std::vector<float>* hidden_out,
    const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture) const {
  (void)backend;
  (void)model;
  (void)cache;
  (void)draft;
  (void)prefix_logits;
  (void)anchor;
  (void)hidden_out;
  (void)capture_layers;
  (void)capture;
  return std::unexpected(StatusCode::UnsupportedFeature);
}

std::expected<std::uint32_t, StatusCode> Glm5NextArchitecture::Draft(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::span<const float> hidden, std::uint32_t token, std::uint64_t pos,
    std::vector<float>* mtp_hidden_out) const {
  (void)backend;
  (void)model;
  (void)cache;
  (void)hidden;
  (void)token;
  (void)pos;
  (void)mtp_hidden_out;
  return std::unexpected(StatusCode::UnsupportedFeature);
}

std::size_t Glm5NextArchitecture::DraftRows(
    const core::DecodeCache& cache) const {
  (void)cache;
  return 0;  // no draft cache until the MTP head lands
}

void Glm5NextArchitecture::DraftTruncate(core::DecodeCache& cache,
                                         std::size_t rows) const {
  (void)cache;
  (void)rows;
}

}  // namespace tessera::models::glm5_next
