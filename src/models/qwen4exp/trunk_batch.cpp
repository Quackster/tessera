#include "models/qwen4exp/internal.hpp"

#include <cstring>
#include <span>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/profile.hpp"
#include "models/qwen4exp/state.hpp"

// The batched qwen4exp trunk: prefill and the multi-row verify run one
// layer across every row before moving on, so a large batch stages each
// paged layer's experts once (the union covers most experts) instead of
// once per row. The per-row residual stack lives in `h.batch_r` and is
// copied into the single-row scratch `h.R` for each row, so the layer body
// is shared with the single-token forward (RunTrunkLayer).

namespace tessera::models::qwen4exp {

namespace detail = ::tessera::core::detail;

using detail::DownloadF32;
using detail::GatherEmbedding;
using detail::NeedWeightAny;
using detail::UploadF32;

namespace {

bool CarriesPle(const TransformerConfig& cfg) {
  for (const std::uint64_t l : cfg.ple.layers) {
    if (l == 1) {
      return true;
    }
  }
  return false;
}

// Gather one embedding row per token into `h.batch_x`, either from the
// per-row `embeddings` buffer (an image row) or from the token table.
std::expected<void, StatusCode> FillBatchEmbeddings(
    Backend& backend, const Model& model, const Qwen4ExpGeometry& g,
    Qwen4ExpState& h, std::span<const std::uint32_t> tokens,
    const Buffer* embeddings) {
  const std::size_t rows = tokens.size();
  const std::size_t hidden = g.hidden;
  if (embeddings != nullptr) {
    return backend.CopyD2D(*embeddings, 0, *h.batch_x, 0, rows * hidden * 4)
               ? std::expected<void, StatusCode>{}
               : std::expected<void, StatusCode>(
                     std::unexpected(StatusCode::DeviceError));
  }
  auto embed = NeedWeightAny(model, "token_embd.weight");
  if (!embed) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::vector<float> host(rows * hidden);
  std::vector<float> row(hidden);
  for (std::size_t t = 0; t < rows; ++t) {
    if (tokens[t] >= (*embed)->manifest.shape.Numel() / hidden) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    if (auto gathered = GatherEmbedding(backend, **embed, tokens[t], hidden, row);
        !gathered) {
      return std::unexpected(gathered.error());
    }
    std::memcpy(host.data() + t * hidden, row.data(), hidden * 4);
  }
  if (!UploadF32(backend, *h.batch_x, host)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

// batch_r[row] = batch_x[row] repeated over the hc streams.
std::expected<void, StatusCode> InitBatchStack(
    Backend& backend, const Qwen4ExpGeometry& g, Qwen4ExpState& h,
    std::size_t rows) {
  for (std::size_t t = 0; t < rows; ++t) {
    for (std::size_t c = 0; c < g.hc; ++c) {
      if (!backend.CopyD2D(*h.batch_x, t * g.hidden * 4, *h.batch_r,
                           (t * g.hc + c) * g.hidden * 4, g.hidden * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
  }
  return {};
}

std::expected<void, StatusCode> GrowBatchScratch(Backend& backend,
                                                 const Qwen4ExpGeometry& g,
                                                 Qwen4ExpState& h,
                                                 std::size_t rows) {
  if (h.batch_rows >= rows && h.batch_r != nullptr && h.batch_x != nullptr) {
    return {};
  }
  auto r = backend.AllocateBuffer(rows * g.hc_dim * 4, MemoryKind::Device);
  if (!r) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  auto x = backend.AllocateBuffer(rows * g.hidden * 4, MemoryKind::Device);
  if (!x) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  h.batch_r = std::move(*r);
  h.batch_x = std::move(*x);
  h.batch_rows = rows;
  return {};
}

}  // namespace

std::expected<void, StatusCode> ForwardTrunkBatch(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::span<const std::uint32_t> tokens, std::vector<float>* logits_out,
    std::vector<float>* hidden_out, bool all_logits, const Buffer* embeddings) {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const TransformerConfig& cfg = *config;
  auto geom = DeriveGeometry(cfg);
  if (!geom) {
    return std::unexpected(geom.error());
  }
  const Qwen4ExpGeometry& g = *geom;
  if (tokens.empty()) {
    return {};
  }
  Qwen4ExpState& h = State(cache);
  h.profile = cache.profile;
  const std::size_t rows = tokens.size();
  if (auto grown = GrowBatchScratch(backend, g, h, rows); !grown) {
    return grown;
  }
  if (auto filled = FillBatchEmbeddings(backend, model, g, h, tokens,
                                        embeddings);
      !filled) {
    return filled;
  }
  if (auto init = InitBatchStack(backend, g, h, rows); !init) {
    return init;
  }
  const bool ple_layer = CarriesPle(cfg);
  // The layers run once each over every row, so RunMoeFfn's selective expert
  // staging accumulates across the rows of a layer: an expert uploaded for
  // one row is a device gather for the next, and the union per layer is
  // paid once. This beats a whole-tensor upload, which reads every expert
  // from RAM and ignores the cache.
  for (std::size_t l = 0; l < cfg.layers; ++l) {
    for (std::size_t t = 0; t < rows; ++t) {
      if (!backend.CopyD2D(*h.batch_r, t * g.hc_dim * 4, *h.R, 0,
                           g.hc_dim * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      if (auto layer = RunTrunkLayer(backend, model, cfg, g, h, l,
                                     h.position + t, tokens[t], ple_layer);
          !layer) {
        return layer;
      }
      if (!backend.CopyD2D(*h.R, 0, *h.batch_r, t * g.hc_dim * 4,
                           g.hc_dim * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
  }
  const auto emit_row = [&](std::size_t t)
      -> std::expected<void, StatusCode> {
    if (!backend.CopyD2D(*h.batch_r, t * g.hc_dim * 4, *h.R, 0,
                         g.hc_dim * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (auto head = RunTrunkHead(backend, model, cfg, g, h); !head) {
      return head;
    }
    auto logits = DownloadF32(backend, *h.logits);
    if (!logits) {
      return std::unexpected(logits.error());
    }
    if (logits_out != nullptr) {
      logits_out->insert(logits_out->end(), logits->begin(), logits->end());
    }
    return {};
  };
  if (logits_out != nullptr) {
    if (all_logits) {
      for (std::size_t t = 0; t < rows; ++t) {
        if (auto emitted = emit_row(t); !emitted) {
          return emitted;
        }
      }
    } else if (auto emitted = emit_row(rows - 1); !emitted) {
      return emitted;
    }
  }
  if (hidden_out != nullptr) {
    if (!backend.CopyD2D(*h.batch_r, (rows - 1) * g.hc_dim * 4, *h.R, 0,
                         g.hc_dim * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    auto snap = DownloadF32(backend, *h.R);
    if (!snap) {
      return std::unexpected(snap.error());
    }
    *hidden_out = std::move(*snap);
  }
  h.position += rows;
  return {};
}

}  // namespace tessera::models::qwen4exp
