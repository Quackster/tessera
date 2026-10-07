#include "core/decode.hpp"

#include <memory>

#include "core/decode_internal.hpp"

namespace tessera::core {

std::expected<std::uint32_t, StatusCode> DecodeStep(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden) {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  if (config->hybrid) {
    if (!cache.hybrid) {
      cache.hybrid = std::make_unique<HybridDecodeCache>();
    }
    return HybridDecodeStep(backend, model, cache, token, hidden);
  }
  return DecodeStepDevice(backend, model, cache, token, hidden);
}

std::expected<std::vector<float>, StatusCode> DecodeLogits(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden) {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  if (config->hybrid) {
    if (!cache.hybrid) {
      cache.hybrid = std::make_unique<HybridDecodeCache>();
    }
    return HybridDecodeLogits(backend, model, cache, token, hidden);
  }
  return DecodeStepDeviceLogits(backend, model, cache, token, hidden);
}

std::expected<std::vector<std::vector<float>>, StatusCode> ScoreTokens(
    Backend& backend, const Model& model,
    std::span<const std::uint32_t> tokens) {
  DecodeCache cache;
  std::vector<std::vector<float>> rows;
  rows.reserve(tokens.size());
  for (const std::uint32_t token : tokens) {
    auto logits = DecodeLogits(backend, model, cache, token);
    if (!logits) {
      return std::unexpected(logits.error());
    }
    rows.push_back(std::move(*logits));
  }
  return rows;
}

std::expected<DraftVerification, StatusCode> VerifyDraft(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::span<const std::uint32_t> draft,
    std::span<const float> prefix_logits) {
  if (prefix_logits.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  DraftVerification result;
  result.logits.assign(prefix_logits.begin(), prefix_logits.end());
  // Feed a draft token only after it matches the target's greedy token, so
  // the cache advances by exactly the accepted prefix and needs no
  // rollback.
  for (const std::uint32_t token : draft) {
    if (detail::ArgMax(result.logits) != token) {
      break;
    }
    auto logits = DecodeLogits(backend, model, cache, token);
    if (!logits) {
      return std::unexpected(logits.error());
    }
    result.logits = std::move(*logits);
    ++result.accepted;
  }
  result.next_token = detail::ArgMax(result.logits);
  return result;
}

}  // namespace tessera::core
