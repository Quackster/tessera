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
  const Architecture* arch = model.Arch();
  if (arch != nullptr) {
    auto logits = arch->Logits(backend, model, cache, token, hidden, nullptr,
                               nullptr, nullptr);
    if (!logits) {
      return std::unexpected(logits.error());
    }
    return detail::ArgMax(*logits);
  }
  return DecodeStepDevice(backend, model, cache, token, hidden);
}

std::expected<void, StatusCode> DecodeForward(Backend& backend,
                                              const Model& model,
                                              DecodeCache& cache,
                                              std::uint32_t token,
                                              std::vector<float>* hidden,
                                              const Buffer* embedding) {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const Architecture* arch = model.Arch();
  if (arch != nullptr) {
    return arch->Forward(backend, model, cache, token, hidden, nullptr, nullptr,
                         embedding);
  }
  if (embedding != nullptr) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  return DecodeStepDeviceForward(backend, model, cache, token, hidden);
}

std::expected<std::vector<float>, StatusCode> DecodeLogits(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden,
    const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture, const Buffer* embedding) {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const Architecture* arch = model.Arch();
  if (arch != nullptr) {
    return arch->Logits(backend, model, cache, token, hidden, capture_layers,
                        capture, embedding);
  }
  if (embedding != nullptr) {
    return std::unexpected(StatusCode::UnsupportedFeature);
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

std::expected<std::vector<float>, StatusCode> PrefillTokens(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::span<const std::uint32_t> tokens, std::vector<float>* hidden_out,
    const Buffer* embeddings) {
  if (tokens.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const Architecture* arch = model.Arch();
  if (arch != nullptr) {
    std::vector<float> logits;
    auto status = arch->ForwardBatch(backend, model, cache, tokens, &logits,
                                     hidden_out, /*all_logits=*/false,
                                     embeddings);
    if (!status) {
      return std::unexpected(status.error());
    }
    return logits;
  }
  for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
    auto forward = DecodeForward(backend, model, cache, tokens[i]);
    if (!forward) {
      return std::unexpected(forward.error());
    }
  }
  return DecodeLogits(backend, model, cache, tokens.back(), hidden_out);
}

std::expected<std::vector<std::vector<float>>, StatusCode> DecodeLogitsBatch(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::span<const std::uint32_t> tokens) {
  if (tokens.empty()) {
    return std::vector<std::vector<float>>{};
  }
  auto config = model.Config();
  const Architecture* arch = model.Arch();
  if (arch != nullptr && config) {
    std::vector<float> flat;
    auto status = arch->ForwardBatch(backend, model, cache, tokens, &flat,
                                     nullptr, /*all_logits=*/true, nullptr);
    if (!status) {
      return std::unexpected(status.error());
    }
    std::vector<std::vector<float>> rows(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      rows[i].assign(flat.begin() + i * config->vocab_size,
                     flat.begin() + (i + 1) * config->vocab_size);
    }
    return rows;
  }
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
    std::span<const float> prefix_logits, std::vector<float>* hidden_out) {
  if (prefix_logits.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  // An architecture that batches the scoring owns the cache rollback;
  // everything else feeds one token at a time and never over-advances.
  const Architecture* arch = model.Arch();
  if (draft.size() > 1 && arch != nullptr) {
    return arch->Verify(backend, model, cache, draft, prefix_logits, hidden_out);
  }
  DraftVerification result;
  result.logits.assign(prefix_logits.begin(), prefix_logits.end());
  for (const std::uint32_t token : draft) {
    if (detail::ArgMax(result.logits) != token) {
      break;
    }
    auto logits = DecodeLogits(backend, model, cache, token, hidden_out);
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
