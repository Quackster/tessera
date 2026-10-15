#include "models/qwen4exp/internal.hpp"

#include <algorithm>
#include <cstring>
#include <span>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/loaders/shard.hpp"
#include "core/profile.hpp"
#include "models/qwen4exp/state.hpp"

// The batched qwen4exp trunk: prefill and the multi-row verify run one
// layer across every row before moving on. The mixer half of a layer runs
// per row (the recurrence and the paged cache are sequential), then every
// row's MoE router runs, the selected ids are read back once, the union of
// experts stages once, and the expert half runs per row. This pays the
// per-layer id readback and expert transfer once per layer instead of once
// per row. The per-row residual stack lives in `h.batch_r`.

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

// Allocates the per-layer gated-delta state snapshots for a verify: one
// slot per committed position (rows + 1, slot 0 is the pre-batch state),
// for every owned recurrent layer.
std::expected<void, StatusCode> AllocSnapshotScratch(
    Backend& backend, const TransformerConfig& cfg, const Qwen4ExpGeometry& g,
    const ModelShard& shard, Qwen4ExpState& h, std::size_t rows) {
  const core::LayerRange range = core::ShardLayerRange(shard, cfg.layers);
  const std::size_t state_len =
      g.gdn_num_v_heads * g.gdn_head_k_dim * g.gdn_head_v_dim;
  const std::size_t hist_len = g.gdn_conv_dim * (g.gdn_width - 1);
  h.batch_gdn_state.clear();
  h.batch_gdn_state.resize(cfg.layers);
  h.batch_gdn_conv.clear();
  h.batch_gdn_conv.resize(cfg.layers);
  for (std::size_t l = range.first; l < range.last; ++l) {
    if (cfg.IsFullAttentionLayer(l)) {
      continue;
    }
    auto state =
        backend.AllocateBuffer((rows + 1) * state_len * 4, MemoryKind::Device);
    auto conv =
        backend.AllocateBuffer((rows + 1) * hist_len * 4, MemoryKind::Device);
    if (!state || !conv) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    h.batch_gdn_state[l] = std::move(*state);
    h.batch_gdn_conv[l] = std::move(*conv);
  }
  h.batch_snapshot_rows = rows;
  return {};
}

std::expected<void, StatusCode> GrowBatchScratch(Backend& backend,
                                                 const TransformerConfig& cfg,
                                                 const Qwen4ExpGeometry& g,
                                                 const ModelShard& shard,
                                                 Qwen4ExpState& h,
                                                 std::size_t rows,
                                                 bool snapshot_state) {
  if (h.batch_rows >= rows && h.batch_r != nullptr) {
    if (snapshot_state && h.batch_snapshot_rows < rows &&
        !AllocSnapshotScratch(backend, cfg, g, shard, h, rows)) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    return {};
  }
  const std::size_t top_k = cfg.experts_per_tok;
  auto alloc = [&](std::unique_ptr<Buffer>& slot,
                   std::size_t bytes) -> std::expected<void, StatusCode> {
    auto buffer = backend.AllocateBuffer(bytes, MemoryKind::Device);
    if (!buffer) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    slot = std::move(*buffer);
    return {};
  };
  if (auto r = alloc(h.batch_r, rows * g.hc_dim * 4); !r) return r;
  if (auto x = alloc(h.batch_x, rows * g.hidden * 4); !x) return x;
  if (auto m = alloc(h.batch_mixed, rows * g.hidden * 4); !m) return m;
  if (auto i = alloc(h.batch_inject2, rows * g.hc * 4); !i) return i;
  if (auto id = alloc(h.batch_ids, rows * top_k * 4); !id) return id;
  if (auto w = alloc(h.batch_wts, rows * top_k * 4); !w) return w;
  if (auto s = alloc(h.batch_sig, rows * 4); !s) return s;
  h.batch_rows = rows;
  if (snapshot_state) {
    if (auto snap = AllocSnapshotScratch(backend, cfg, g, shard, h, rows);
        !snap) {
      return snap;
    }
  }
  return {};
}

}  // namespace

std::expected<void, StatusCode> ForwardTrunkBatch(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::span<const std::uint32_t> tokens, std::vector<float>* logits_out,
    std::vector<float>* hidden_out, bool all_logits, const Buffer* embeddings,
    const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture, bool snapshot_state) {
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
  const std::size_t hidden = g.hidden;
  const std::size_t top_k = cfg.experts_per_tok;
  const std::size_t gdn_state_len =
      g.gdn_num_v_heads * g.gdn_head_k_dim * g.gdn_head_v_dim;
  const std::size_t gdn_hist_len = g.gdn_conv_dim * (g.gdn_width - 1);
  const ModelShard& shard = model.Shard();
  const core::LayerRange range = core::ShardLayerRange(shard, cfg.layers);
  if (range.first >= range.last) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (auto grown = GrowBatchScratch(backend, cfg, g, shard, h, rows,
                                    snapshot_state);
      !grown) {
    return grown;
  }
  // The first stage gathers one embedding row per token; a later stage
  // receives the previous stage's whole residual stack and loads it whole.
  if (shard.enabled && !shard.first_stage) {
    if (embeddings == nullptr ||
        embeddings->Size() < rows * g.hc_dim * 4 ||
        !backend.CopyD2D(*embeddings, 0, *h.batch_r, 0, rows * g.hc_dim * 4)) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
  } else {
    if (auto filled = FillBatchEmbeddings(backend, model, g, h, tokens,
                                          embeddings);
        !filled) {
      return filled;
    }
    if (auto init = InitBatchStack(backend, g, h, rows); !init) {
      return init;
    }
  }
  h.batch_snapshot_start = h.position;
  // Snapshot slot 0 (the pre-batch gated-delta state) per owned layer.
  if (snapshot_state) {
    for (std::size_t l = range.first; l < range.last; ++l) {
      if (cfg.IsFullAttentionLayer(l)) {
        continue;
      }
      if (!backend.CopyD2D(*h.gdn[l].state, 0, *h.batch_gdn_state[l], 0,
                           gdn_state_len * 4) ||
          !backend.CopyD2D(*h.gdn[l].conv_hist, 0, *h.batch_gdn_conv[l], 0,
                           gdn_hist_len * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
  }
  const bool ple_layer = CarriesPle(cfg);
  std::vector<std::uint32_t> host_ids(rows * top_k);
  std::vector<std::uint32_t> union_ids;
  for (std::size_t l = range.first; l < range.last; ++l) {
    // Mixer half: the attention hyper-connection, the mixer and the FFN
    // hyper-connection read, per row (the state is sequential). Store the
    // FFN input and the injection for the expert half.
    for (std::size_t t = 0; t < rows; ++t) {
      if (!backend.CopyD2D(*h.batch_r, t * g.hc_dim * 4, *h.R, 0,
                           g.hc_dim * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      if (auto mixer = RunTrunkLayerMixer(backend, model, cfg, g, h, l,
                                          h.position + t, tokens[t], ple_layer);
          !mixer) {
        return mixer;
      }
      // Row t's post-step recurrent state lands in slot t + 1, so a verify
      // can restore the state at any committed prefix.
      if (snapshot_state && !cfg.IsFullAttentionLayer(l) &&
          (!backend.CopyD2D(*h.gdn[l].state, 0, *h.batch_gdn_state[l],
                            (t + 1) * gdn_state_len * 4, gdn_state_len * 4) ||
           !backend.CopyD2D(*h.gdn[l].conv_hist, 0, *h.batch_gdn_conv[l],
                            (t + 1) * gdn_hist_len * 4, gdn_hist_len * 4))) {
        return std::unexpected(StatusCode::DeviceError);
      }
      if (!backend.CopyD2D(*h.mixed, 0, *h.batch_mixed, t * hidden * 4,
                           hidden * 4) ||
          !backend.CopyD2D(*h.inject2, 0, *h.batch_inject2, t * g.hc * 4,
                           g.hc * 4) ||
          !backend.CopyD2D(*h.R, 0, *h.batch_r, t * g.hc_dim * 4,
                           g.hc_dim * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
    // Router for every row, then one readback of the whole selection.
    for (std::size_t t = 0; t < rows; ++t) {
      if (!backend.CopyD2D(*h.batch_mixed, t * hidden * 4, *h.mixed, 0,
                           hidden * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      if (auto router = RunMoeRouter(backend, model, cfg, h, l, *h.mixed);
          !router) {
        return router;
      }
      if (!backend.CopyD2D(*h.moe_ids, 0, *h.batch_ids, t * top_k * 4,
                           top_k * 4) ||
          !backend.CopyD2D(*h.moe_wts, 0, *h.batch_wts, t * top_k * 4,
                           top_k * 4) ||
          !backend.CopyD2D(*h.moe_sig, 0, *h.batch_sig, t * 4, 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
    if (model.ExpertsPaged(l)) {
      if (!backend.CopyD2H(*h.batch_ids,
                           reinterpret_cast<std::byte*>(host_ids.data()),
                           host_ids.size() * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      union_ids.assign(host_ids.begin(), host_ids.end());
      std::sort(union_ids.begin(), union_ids.end());
      union_ids.erase(std::unique(union_ids.begin(), union_ids.end()),
                      union_ids.end());
      if (auto staged = model.EnsureLayerExperts(
              l, std::span<const std::uint32_t>(union_ids));
          !staged) {
        return std::unexpected(staged.error());
      }
    }
    // Expert half: the shared expert, the routed experts from the staged
    // union, and the FFN write back into the stack, per row.
    for (std::size_t t = 0; t < rows; ++t) {
      if (!backend.CopyD2D(*h.batch_mixed, t * hidden * 4, *h.mixed, 0,
                           hidden * 4) ||
          !backend.CopyD2D(*h.batch_inject2, t * g.hc * 4, *h.inject2, 0,
                           g.hc * 4) ||
          !backend.CopyD2D(*h.batch_ids, t * top_k * 4, *h.moe_ids, 0,
                           top_k * 4) ||
          !backend.CopyD2D(*h.batch_wts, t * top_k * 4, *h.moe_wts, 0,
                           top_k * 4) ||
          !backend.CopyD2D(*h.batch_sig, t * 4, *h.moe_sig, 0, 4) ||
          !backend.CopyD2D(*h.batch_r, t * g.hc_dim * 4, *h.R, 0,
                           g.hc_dim * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      if (auto experts = RunMoeExperts(backend, model, cfg, h, l, *h.mixed,
                                       top_k, {}, {});
          !experts) {
        return experts;
      }
      if (auto write = HyperWrite(backend, h, g, *h.R, *h.block_out, *h.inject2);
          !write) {
        return write;
      }
      if (!backend.CopyD2D(*h.R, 0, *h.batch_r, t * g.hc_dim * 4,
                           g.hc_dim * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
    // Capture the per-row residual stack after the boundary layer for the
    // next stage's handoff.
    if (capture_layers != nullptr && capture != nullptr) {
      for (std::size_t i = 0; i < capture_layers->size(); ++i) {
        if ((*capture_layers)[i] == l &&
            !backend.CopyD2D(*h.batch_r, 0, *(*capture)[i], 0,
                             rows * g.hc_dim * 4)) {
          return std::unexpected(StatusCode::DeviceError);
        }
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
