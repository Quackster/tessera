#include "core/decode.hpp"

#include <memory>

#include "core/decode_hybrid_internal.hpp"
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
  if (config->hybrid) {
    if (!cache.hybrid) {
      cache.hybrid = std::make_unique<HybridDecodeCache>();
    }
    return HybridForward(backend, model, cache, token, hidden, nullptr, nullptr,
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
  if (config->hybrid) {
    if (!cache.hybrid) {
      cache.hybrid = std::make_unique<HybridDecodeCache>();
    }
    return HybridDecodeLogits(backend, model, cache, token, hidden,
                              capture_layers, capture, embedding);
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
  auto config = model.Config();
  if (config && config->hybrid) {
    std::vector<float> logits;
    auto status = HybridForwardBatch(backend, model, cache, tokens, &logits,
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
  if (config && config->hybrid) {
    std::vector<float> flat;
    auto status = HybridForwardBatch(backend, model, cache, tokens, &flat);
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
  // Multi-token drafts on the hybrid path score in one batched forward and
  // roll the cache back to the accepted prefix. Single-token drafts and
  // other models feed one token at a time and never over-advance.
  if (draft.size() > 1 && config->hybrid && cache.hybrid != nullptr) {
    auto geometry = DeriveGeometry(*config);
    if (!geometry) {
      return std::unexpected(geometry.error());
    }
    const LinearGeometry& g = *geometry;
    const std::size_t state_len = g.num_v_heads * g.head_k_dim * g.head_v_dim;
    const std::size_t hist_len = g.conv_dim * (g.width - 1);
    HybridDecodeCache& h = *cache.hybrid;
    const std::size_t prefix = h.position;
    std::vector<float> flat;
    std::vector<float> last_hidden;
    auto status =
        HybridForwardBatch(backend, model, cache, draft, &flat, &last_hidden);
    if (!status) {
      return std::unexpected(status.error());
    }
    const std::size_t vocab = config->vocab_size;
    DraftVerification result;
    std::span<const float> last(prefix_logits.begin(), prefix_logits.end());
    std::size_t accepted = 0;
    for (std::size_t i = 0; i < draft.size(); ++i) {
      if (detail::ArgMax(last) != draft[i]) {
        break;
      }
      ++accepted;
      last = std::span<const float>(flat.data() + i * vocab, vocab);
    }
    result.accepted = accepted;
    result.logits.assign(last.begin(), last.end());
    result.next_token = detail::ArgMax(result.logits);
    if (accepted < draft.size()) {
      const std::size_t new_pos = prefix + accepted;
      for (auto& kv : h.full) {
        kv.rows = new_pos;
      }
      h.position = new_pos;
      for (std::size_t l = 0; l < config->layers; ++l) {
        if (config->IsFullAttentionLayer(l)) {
          continue;
        }
        if (!backend.CopyD2D(*h.batch->state_hist[l], accepted * state_len * 4,
                             *h.linear[l].state, 0, state_len * 4)) {
          return std::unexpected(StatusCode::DeviceError);
        }
        std::copy(h.batch->conv_hist_hist[l].begin() + accepted * hist_len,
                  h.batch->conv_hist_hist[l].begin() +
                      (accepted + 1) * hist_len,
                  h.conv_hist[l].begin());
      }
    }
    if (hidden_out != nullptr && accepted > 0) {
      auto xh = detail::DownloadF32(backend, *h.batch->x);
      if (xh) {
        hidden_out->resize(config->hidden_dim);
        std::copy(xh->begin() + (accepted - 1) * config->hidden_dim,
                  xh->begin() + accepted * config->hidden_dim,
                  hidden_out->begin());
      }
    }
    return result;
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
