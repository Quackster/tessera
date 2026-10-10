#include "core/decode.hpp"

#include <algorithm>
#include <memory>
#include <optional>

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

std::expected<void, StatusCode> DecodeForward(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden, const Buffer* embedding,
    const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture) {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const Architecture* arch = model.Arch();
  if (arch != nullptr) {
    return arch->Forward(backend, model, cache, token, hidden, capture_layers,
                         capture, embedding);
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

std::expected<std::uint32_t, StatusCode> DecodeToken(
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
    return arch->GreedyToken(backend, model, cache, token, hidden,
                             capture_layers, capture, embedding);
  }
  if (embedding != nullptr) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  return DecodeStepDevice(backend, model, cache, token, hidden);
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
    const Buffer* embeddings, std::size_t chunk_tokens,
    const PrefillProgress* progress) {
  if (tokens.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const Architecture* arch = model.Arch();
  if (arch != nullptr) {
    auto config = model.Config();
    if (!config) {
      return std::unexpected(config.error());
    }
    const std::size_t hidden = config->hidden_dim;
    const std::size_t heads = config->attention.heads;
    const std::size_t head_dim = config->attention.head_dim;
    const std::size_t want =
        chunk_tokens == 0 ? tokens.size() : chunk_tokens;
    // One forward per chunk, each within the attention work budget.
    // The cache is empty (prefill starts a request), so the keys the
    // chunk's last row sees end at off + len.
    const std::vector<std::size_t> lens = detail::PrefillLens(
        tokens.size(), want, /*base=*/0, heads, head_dim);
    if (lens.size() == 1 && lens[0] == tokens.size()) {
      std::vector<float> logits;
      auto status = arch->ForwardBatch(backend, model, cache, tokens, &logits,
                                       hidden_out, /*all_logits=*/false,
                                       embeddings);
      if (!status) {
        return std::unexpected(status.error());
      }
      if (progress != nullptr) {
        (*progress)(tokens.size(), tokens.size());
      }
      return logits;
    }
    // Long prompt, bounded scratch and bounded launches: one forward
    // per chunk, the head on the last chunk's last row only. The cache
    // carries the state across chunks, so the result matches one forward.
    std::unique_ptr<Buffer> staged;
    if (embeddings != nullptr) {
      auto made = backend.AllocateBuffer(want * hidden * 4,
                                         MemoryKind::Device);
      if (!made) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      staged = std::move(*made);
    }
    std::vector<float> logits;
    std::size_t off = 0;
    for (const std::size_t len : lens) {
      const bool last = off + len == tokens.size();
      const Buffer* chunk_emb = nullptr;
      if (staged) {
        if (!backend.CopyD2D(*embeddings, off * hidden * 4, *staged, 0,
                             len * hidden * 4)) {
          return std::unexpected(StatusCode::DeviceError);
        }
        chunk_emb = staged.get();
      }
      std::vector<float>* chunk_hidden = last ? hidden_out : nullptr;
      std::vector<float> chunk_logits;
      auto status = arch->ForwardBatch(
          backend, model, cache, tokens.subspan(off, len),
          last ? &chunk_logits : nullptr, chunk_hidden,
          /*all_logits=*/false, chunk_emb);
      if (!status) {
        return std::unexpected(status.error());
      }
      if (last) {
        logits = std::move(chunk_logits);
      }
      off += len;
      if (progress != nullptr) {
        (*progress)(off, tokens.size());
      }
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
    std::span<const float> prefix_logits, std::optional<std::uint32_t> anchor,
    std::vector<float>* hidden_out,
    const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture) {
  if (!anchor.has_value() && prefix_logits.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  // An architecture that batches the scoring owns the cache rollback;
  // everything else feeds one token at a time and never over-advances. An
  // anchor always takes the batch path: [anchor, drafts...] is one forward,
  // and the multi-row GEMV family reads each weight block once for the
  // whole small batch.
  const Architecture* arch = model.Arch();
  if (arch != nullptr && (draft.size() > 1 || anchor.has_value())) {
    return arch->Verify(backend, model, cache, draft, prefix_logits, anchor,
                        hidden_out, capture_layers, capture);
  }
  DraftVerification result;
  if (anchor.has_value()) {
    // The anchor rides as the batch's first row: it advances the cache and
    // its logits score draft[0]. Its hidden goes to the caller, so a folding
    // drafter keeps the last committed row's hidden even with no acceptance.
    auto anchor_logits =
        DecodeLogits(backend, model, cache, *anchor, hidden_out, capture_layers,
                     capture);
    if (!anchor_logits) {
      return std::unexpected(anchor_logits.error());
    }
    result.logits = std::move(*anchor_logits);
  } else {
    result.logits.assign(prefix_logits.begin(), prefix_logits.end());
  }
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

namespace tessera {

// The generic greedy step: download the logits and scan them on the host.
// An architecture overrides GreedyToken to argmax on the device instead.
std::expected<std::uint32_t, StatusCode> Architecture::GreedyToken(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden,
    const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture, const Buffer* embedding) const {
  auto logits =
      Logits(backend, model, cache, token, hidden, capture_layers, capture,
             embedding);
  if (!logits) {
    return std::unexpected(logits.error());
  }
  return core::detail::ArgMax(*logits);
}

}  // namespace tessera
