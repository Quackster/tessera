#include "core/decode_hybrid_internal.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/conv.hpp"

namespace tessera::core {

namespace {

using detail::AddDevice;
using detail::AppendKv;
using detail::AttentionDevice;
using detail::AttentionQuantDevice;
using detail::DeltaStepHeadsDevice;
using detail::DownloadF32;
using detail::DownloadF32Cached;
using detail::GatherEmbedding;
using detail::GemmFor;
using detail::L2NormDevice;
using detail::MropeDevice;
using detail::NeedWeight;
using detail::NeedWeightAny;
using detail::ProjectDevice;
using detail::QGateSplitDevice;
using detail::RepeatHeadsDevice;
using detail::RmsNormDevice;
using detail::RmsNormGatedDevice;
using detail::SigmoidGateDevice;
using detail::SiluMulDevice;
using detail::SsmGateDevice;
using detail::UploadF32;

// Allocates (or reallocates) the batched scratch for `rows` tokens.
std::expected<void, StatusCode> AllocBatch(Backend& backend,
                                           const TransformerConfig& cfg,
                                           const LinearGeometry& g,
                                           HybridDecodeCache& h,
                                           std::size_t rows) {
  if (h.batch && h.batch->capacity >= rows) {
    return {};
  }
  auto b = std::make_unique<HybridBatchScratch>();
  b->capacity = rows;
  auto alloc = [&backend](std::unique_ptr<Buffer>& slot,
                          std::size_t elements) -> bool {
    auto buf = backend.AllocateBuffer(elements * 4, MemoryKind::Device);
    if (!buf) {
      return false;
    }
    slot = std::move(*buf);
    return true;
  };
  const std::size_t q_dim = cfg.attention.heads * cfg.attention.head_dim;
  const std::size_t kv_dim = cfg.attention.kv_heads * cfg.attention.head_dim;
  const std::size_t state_len = g.num_v_heads * g.head_k_dim * g.head_v_dim;
  const std::size_t hist_len = g.conv_dim * (g.width - 1);
  if (!alloc(b->x, rows * cfg.hidden_dim) ||
      !alloc(b->xn, rows * cfg.hidden_dim) ||
      !alloc(b->proj, rows * cfg.hidden_dim) ||
      !alloc(b->logits, rows * cfg.vocab_size) ||
      !alloc(b->fused, rows * q_dim * 2) || !alloc(b->q, rows * q_dim) ||
      !alloc(b->gate, rows * q_dim) || !alloc(b->kf, rows * kv_dim) ||
      !alloc(b->vf, rows * kv_dim) || !alloc(b->attn, rows * q_dim) ||
      !alloc(b->fgate, rows * cfg.ffn_dim) ||
      !alloc(b->fup, rows * cfg.ffn_dim) ||
      !alloc(b->fmlp, rows * cfg.ffn_dim) ||
      !alloc(b->qkv, rows * g.conv_dim) ||
      !alloc(b->z, rows * g.value_dim) ||
      !alloc(b->alpha_raw, rows * g.num_v_heads) ||
      !alloc(b->beta_raw, rows * g.num_v_heads) ||
      !alloc(b->alpha, rows * g.num_v_heads) ||
      !alloc(b->beta, rows * g.num_v_heads) ||
      !alloc(b->out, rows * g.value_dim)) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  auto pos = backend.AllocateBuffer(rows * 3 * 8, MemoryKind::Device);
  if (!pos) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  b->pos = std::move(*pos);
  b->state_hist.resize(cfg.layers);
  b->conv_hist_hist.resize(cfg.layers);
  for (std::size_t l = 0; l < cfg.layers; ++l) {
    if (!cfg.IsFullAttentionLayer(l)) {
      if (!alloc(b->state_hist[l], (rows + 1) * state_len)) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      b->conv_hist_hist[l].assign((rows + 1) * hist_len, 0.0f);
    }
  }
  h.batch = std::move(b);
  return {};
}

// Batched gated MLP over `rows` rows of h.batch->x in place.
std::expected<void, StatusCode> RunFfnBatch(Backend& backend,
                                            const Model& model,
                                            const TransformerConfig& cfg,
                                            HybridDecodeCache& h,
                                            std::size_t layer,
                                            std::size_t rows) {
  const std::string base = "blk." + std::to_string(layer) + ".";
  auto mlp_norm =
      NeedWeight(model, base + "post_attention_norm.weight", DType::F32);
  auto fg = NeedWeightAny(model, base + "ffn_gate.weight");
  auto fu = NeedWeightAny(model, base + "ffn_up.weight");
  auto fd = NeedWeightAny(model, base + "ffn_down.weight");
  if (!mlp_norm || !fg || !fu || !fd) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto gemm_fgate = GemmFor(backend, h.gemms, (*fg)->manifest.dtype);
  auto gemm_fup = GemmFor(backend, h.gemms, (*fu)->manifest.dtype);
  auto gemm_fdown = GemmFor(backend, h.gemms, (*fd)->manifest.dtype);
  if (!gemm_fgate || !gemm_fup || !gemm_fdown) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  HybridBatchScratch& b = *h.batch;
  if (!RmsNormDevice(backend, *h.rmsnorm_kernel, *b.x, *(*mlp_norm)->device,
                     *b.xn, rows, cfg.hidden_dim, cfg.norm_eps) ||
      !ProjectDevice(backend, *(*gemm_fgate), *b.xn, *(*fg)->device, *b.fgate,
                     rows, cfg.ffn_dim, cfg.hidden_dim) ||
      !ProjectDevice(backend, *(*gemm_fup), *b.xn, *(*fu)->device, *b.fup,
                     rows, cfg.ffn_dim, cfg.hidden_dim) ||
      !SiluMulDevice(backend, *h.silu_mul_kernel, *b.fgate, *b.fup, *b.fmlp,
                     rows * cfg.ffn_dim) ||
      !ProjectDevice(backend, *(*gemm_fdown), *b.fmlp, *(*fd)->device, *b.proj,
                     rows, cfg.hidden_dim, cfg.ffn_dim) ||
      !AddDevice(backend, *h.add_kernel, *b.x, *b.proj, *b.x,
                 rows * cfg.hidden_dim)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

// Batched full-attention block over `rows` rows ending in h.batch->x.
std::expected<void, StatusCode> RunFullBlockBatch(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    HybridDecodeCache& h, std::size_t layer, std::uint64_t pos,
    HybridDecodeCache::FullKv& kv, std::size_t rows) {
  const std::size_t hidden = cfg.hidden_dim;
  const std::size_t heads = cfg.attention.heads;
  const std::size_t kv_heads = cfg.attention.kv_heads;
  const std::size_t head_dim = cfg.attention.head_dim;
  const std::size_t kv_dim = kv_heads * head_dim;
  const std::size_t rope_dim = cfg.attention.rope_dim;
  const std::size_t sec_t = static_cast<std::size_t>(cfg.rope_sections[0]);
  const std::size_t sec_h = static_cast<std::size_t>(cfg.rope_sections[1]);
  const std::size_t sec_w = static_cast<std::size_t>(cfg.rope_sections[2]);
  const std::string base = "blk." + std::to_string(layer) + ".";
  auto norm = NeedWeight(model, base + "attn_norm.weight", DType::F32);
  auto wq = NeedWeightAny(model, base + "attn_q.weight");
  auto wk = NeedWeightAny(model, base + "attn_k.weight");
  auto wv = NeedWeightAny(model, base + "attn_v.weight");
  auto wo = NeedWeightAny(model, base + "attn_output.weight");
  auto q_norm = NeedWeight(model, base + "attn_q_norm.weight", DType::F32);
  auto k_norm = NeedWeight(model, base + "attn_k_norm.weight", DType::F32);
  if (!norm || !wq || !wk || !wv || !wo || !q_norm || !k_norm) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto gemm_q = GemmFor(backend, h.gemms, (*wq)->manifest.dtype);
  auto gemm_k = GemmFor(backend, h.gemms, (*wk)->manifest.dtype);
  auto gemm_v = GemmFor(backend, h.gemms, (*wv)->manifest.dtype);
  auto gemm_o = GemmFor(backend, h.gemms, (*wo)->manifest.dtype);
  if (!gemm_q || !gemm_k || !gemm_v || !gemm_o) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  HybridBatchScratch& b = *h.batch;
  if (!RmsNormDevice(backend, *h.rmsnorm_kernel, *b.x, *(*norm)->device, *b.xn,
                     rows, hidden, cfg.norm_eps) ||
      !ProjectDevice(backend, *(*gemm_q), *b.xn, *(*wq)->device, *b.fused,
                     rows, heads * head_dim * 2, hidden) ||
      !QGateSplitDevice(backend, *h.qgate_split_kernel, *b.fused, *b.q, *b.gate,
                        heads, head_dim, rows) ||
      !ProjectDevice(backend, *(*gemm_k), *b.xn, *(*wk)->device, *b.kf, rows,
                     kv_dim, hidden) ||
      !ProjectDevice(backend, *(*gemm_v), *b.xn, *(*wv)->device, *b.vf, rows,
                     kv_dim, hidden) ||
      !RmsNormDevice(backend, *h.rmsnorm_kernel, *b.q, *(*q_norm)->device, *b.q,
                     rows * heads, head_dim, cfg.norm_eps) ||
      !RmsNormDevice(backend, *h.rmsnorm_kernel, *b.kf, *(*k_norm)->device,
                     *b.kf, rows * kv_heads, head_dim, cfg.norm_eps) ||
      !MropeDevice(backend, *h.mrope_kernel, *b.q, *b.pos, rows, heads,
                   head_dim, rope_dim, sec_t, sec_h, sec_w,
                   cfg.attention.rope_theta) ||
      !MropeDevice(backend, *h.mrope_kernel, *b.kf, *b.pos, rows, kv_heads,
                   head_dim, rope_dim, sec_t, sec_h, sec_w,
                   cfg.attention.rope_theta)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  for (std::size_t t = 0; t < rows; ++t) {
    if (!backend.CopyD2D(*b.kf, t * kv_dim * 4, *h.kf, 0, kv_dim * 4) ||
        !backend.CopyD2D(*b.vf, t * kv_dim * 4, *h.vf, 0, kv_dim * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (auto appended =
            AppendKv(backend, h.cast_kernel.get(), h.quant_kernel.get(),
                     h.kv_scratch.get(), h.kv_scratch.get(),
                     h.scale_scratch.get(), kv, *h.kf, *h.vf, kv_dim);
        !appended) {
      return std::unexpected(appended.error());
    }
  }
  const bool quantized =
      kv.type == KvCacheType::Q8 || kv.type == KvCacheType::Q4;
  const bool attention_ok =
      quantized
          ? detail::AttentionQuantDevice(backend, *h.attention_kernel, *b.q,
                                         *kv.k, *kv.v, *kv.k_scale,
                                         *kv.v_scale, *b.attn, kv.rows, heads,
                                         kv_heads, head_dim, pos, 0, rows)
                .has_value()
          : AttentionDevice(backend, *h.attention_kernel, *b.q, *kv.k, *kv.v,
                            *b.attn, kv.rows, heads, kv_heads, head_dim, pos,
                            0, rows, kv.type == KvCacheType::F16)
                .has_value();
  if (!attention_ok ||
      !SigmoidGateDevice(backend, *h.sigmoid_gate_kernel, *b.attn, *b.gate,
                         *b.attn, rows * heads * head_dim) ||
      !ProjectDevice(backend, *(*gemm_o), *b.attn, *(*wo)->device, *b.proj,
                     rows, hidden, heads * head_dim) ||
      !AddDevice(backend, *h.add_kernel, *b.x, *b.proj, *b.x,
                 rows * hidden)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return RunFfnBatch(backend, model, cfg, h, layer, rows);
}

// Batched linear-attention block. The projections run batched; the causal
// conv and the gated-delta scan run token by token with the shared state.
std::expected<void, StatusCode> RunLinearBlockBatch(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    HybridDecodeCache& h, std::size_t layer, const LinearGeometry& g,
    std::size_t rows) {
  const std::size_t hidden = cfg.hidden_dim;
  const std::string base = "blk." + std::to_string(layer) + ".";
  auto norm = NeedWeight(model, base + "attn_norm.weight", DType::F32);
  auto w_qkv = NeedWeightAny(model, base + "attn_qkv.weight");
  auto w_gate = NeedWeightAny(model, base + "attn_gate.weight");
  auto w_conv = NeedWeight(model, base + "ssm_conv1d.weight", DType::F32);
  auto w_alpha = NeedWeightAny(model, base + "ssm_alpha.weight");
  auto w_beta = NeedWeightAny(model, base + "ssm_beta.weight");
  auto w_a = NeedWeight(model, base + "ssm_a", DType::F32);
  auto w_dt = NeedWeight(model, base + "ssm_dt.bias", DType::F32);
  auto w_norm = NeedWeight(model, base + "ssm_norm.weight", DType::F32);
  auto w_out = NeedWeightAny(model, base + "ssm_out.weight");
  if (!norm || !w_qkv || !w_gate || !w_conv || !w_alpha || !w_beta || !w_a ||
      !w_dt || !w_norm || !w_out) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto gemm_qkv = GemmFor(backend, h.gemms, (*w_qkv)->manifest.dtype);
  auto gemm_gate = GemmFor(backend, h.gemms, (*w_gate)->manifest.dtype);
  auto gemm_alpha = GemmFor(backend, h.gemms, (*w_alpha)->manifest.dtype);
  auto gemm_beta = GemmFor(backend, h.gemms, (*w_beta)->manifest.dtype);
  auto gemm_out = GemmFor(backend, h.gemms, (*w_out)->manifest.dtype);
  if (!gemm_qkv || !gemm_gate || !gemm_alpha || !gemm_beta || !gemm_out) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  HybridBatchScratch& b = *h.batch;
  if (!RmsNormDevice(backend, *h.rmsnorm_kernel, *b.x, *(*norm)->device, *b.xn,
                     rows, hidden, cfg.norm_eps) ||
      !ProjectDevice(backend, *(*gemm_qkv), *b.xn, *(*w_qkv)->device, *b.qkv,
                     rows, g.conv_dim, hidden) ||
      !ProjectDevice(backend, *(*gemm_gate), *b.xn, *(*w_gate)->device, *b.z,
                     rows, g.value_dim, hidden) ||
      !ProjectDevice(backend, *(*gemm_beta), *b.xn, *(*w_beta)->device,
                     *b.beta_raw, rows, g.num_v_heads, hidden) ||
      !ProjectDevice(backend, *(*gemm_alpha), *b.xn, *(*w_alpha)->device,
                     *b.alpha_raw, rows, g.num_v_heads, hidden) ||
      !SsmGateDevice(backend, *h.ssm_gate_kernel, *(*w_a)->device,
                     *(*w_dt)->device, *b.alpha_raw, *b.beta_raw, *b.alpha,
                     *b.beta, g.num_v_heads, rows)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  std::vector<float> qkv(rows * g.conv_dim);
  if (!backend.CopyD2H(*b.qkv, reinterpret_cast<std::byte*>(qkv.data()),
                       qkv.size() * 4)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  auto conv_w = DownloadF32Cached(backend, h.host_weights,
                                  base + "ssm_conv1d.weight",
                                  *(*w_conv)->device);
  if (!conv_w) {
    return std::unexpected(conv_w.error());
  }
  const float q_scale = 1.0 / std::sqrt(static_cast<double>(g.head_k_dim));
  const std::size_t state_len = g.num_v_heads * g.head_k_dim * g.head_v_dim;
  const std::size_t hist_len = g.conv_dim * (g.width - 1);
  // Snapshot the state before the block (slot 0); each token appends its
  // post-token state to slot t+1 so a verification can roll back.
  if (!backend.CopyD2D(*h.linear[layer].state, 0, *b.state_hist[layer], 0,
                       state_len * 4)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  std::copy(h.conv_hist[layer].begin(), h.conv_hist[layer].end(),
            b.conv_hist_hist[layer].begin());
  for (std::size_t t = 0; t < rows; ++t) {
    const float* row = qkv.data() + t * g.conv_dim;
    std::vector<float> xr(g.conv_dim * g.width);
    for (std::size_t c = 0; c < g.conv_dim; ++c) {
      xr[c * g.width] = row[c];
      for (std::size_t i = 1; i < g.width; ++i) {
        xr[c * g.width + i] =
            h.conv_hist[layer][c * (g.width - 1) + (i - 1)];
      }
    }
    std::vector<float> mixed(g.conv_dim);
    std::vector<float> rev_w((*conv_w)->size());
    for (std::size_t c = 0; c < g.conv_dim; ++c) {
      for (std::size_t i = 0; i < g.width; ++i) {
        rev_w[c * g.width + i] = (**conv_w)[c * g.width + (g.width - 1 - i)];
      }
    }
    auto conv = Conv1dStepRef(std::span<const float>(xr),
                              std::span<const float>(rev_w),
                              std::span<float>(mixed), g.conv_dim, g.width);
    if (!conv) {
      return std::unexpected(conv.error());
    }
    for (float& value : mixed) {
      value = value / (1.0f + std::exp(-value));
    }
    for (std::size_t c = 0; c < g.conv_dim; ++c) {
      for (std::size_t i = g.width - 1; i > 1; --i) {
        h.conv_hist[layer][c * (g.width - 1) + (i - 1)] =
            h.conv_hist[layer][c * (g.width - 1) + (i - 2)];
      }
      if (g.width > 1) {
        h.conv_hist[layer][c * (g.width - 1)] = row[c];
      }
    }
    if (!UploadF32(backend, *h.conv_mixed, mixed) ||
        !backend.CopyD2D(*h.conv_mixed, 0, *h.q_l, 0, g.key_dim * 4) ||
        !backend.CopyD2D(*h.conv_mixed, g.key_dim * 4, *h.k_l, 0,
                         g.key_dim * 4) ||
        !backend.CopyD2D(*h.conv_mixed, 2 * g.key_dim * 4, *h.v_l, 0,
                         g.value_dim * 4) ||
        !L2NormDevice(backend, *h.l2norm_kernel, *h.q_l, *h.q_l, g.num_k_heads,
                      g.head_k_dim, 1e-6f, q_scale) ||
        !L2NormDevice(backend, *h.l2norm_kernel, *h.k_l, *h.k_l, g.num_k_heads,
                      g.head_k_dim, 1e-6f) ||
        !RepeatHeadsDevice(backend, *h.repeat_heads_kernel, *h.q_l, *h.q_exp,
                           g.num_v_heads, g.head_k_dim, g.factor) ||
        !RepeatHeadsDevice(backend, *h.repeat_heads_kernel, *h.k_l, *h.k_exp,
                           g.num_v_heads, g.head_k_dim, g.factor) ||
        !backend.CopyD2D(*b.alpha, t * g.num_v_heads * 4, *h.alpha, 0,
                         g.num_v_heads * 4) ||
        !backend.CopyD2D(*b.beta, t * g.num_v_heads * 4, *h.beta, 0,
                         g.num_v_heads * 4) ||
        !DeltaStepHeadsDevice(backend, *h.delta_step_heads_kernel,
                              *h.linear[layer].state, *h.k_exp, *h.v_l,
                              *h.q_exp, *h.core, *h.alpha, *h.beta,
                              g.num_v_heads, g.head_k_dim, g.head_v_dim) ||
        !backend.CopyD2D(*b.z, t * g.value_dim * 4, *h.z, 0, g.value_dim * 4) ||
        !RmsNormGatedDevice(backend, *h.rmsnorm_gated_kernel, *h.core,
                            *(*w_norm)->device, *h.z, *h.out, g.num_v_heads,
                            g.head_v_dim, cfg.norm_eps) ||
        !backend.CopyD2D(*h.out, 0, *b.out, t * g.value_dim * 4,
                         g.value_dim * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    // Snapshot the recurrent state and conv history after this token so a
    // verification can roll back to any accepted prefix.
    if (!backend.CopyD2D(*h.linear[layer].state, 0,
                         *b.state_hist[layer], (t + 1) * state_len * 4,
                         state_len * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    std::copy(h.conv_hist[layer].begin(), h.conv_hist[layer].end(),
              b.conv_hist_hist[layer].begin() + (t + 1) * hist_len);
  }
  if (!ProjectDevice(backend, *(*gemm_out), *b.out, *(*w_out)->device, *b.proj,
                     rows, hidden, g.value_dim) ||
      !AddDevice(backend, *h.add_kernel, *b.x, *b.proj, *b.x,
                 rows * hidden)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return RunFfnBatch(backend, model, cfg, h, layer, rows);
}

}  // namespace

std::expected<void, StatusCode> HybridForwardBatch(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::span<const std::uint32_t> tokens, std::vector<float>* logits_out,
    std::vector<float>* hidden_out, bool all_logits) {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const TransformerConfig& cfg = *config;
  if (!cfg.hybrid || cfg.full_attention_interval == 0 ||
      cfg.rope_sections.size() != 3) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const std::size_t rows = tokens.size();
  if (rows == 0) {
    return {};
  }
  auto geometry = DeriveGeometry(cfg);
  if (!geometry) {
    return std::unexpected(geometry.error());
  }
  const LinearGeometry& g = *geometry;
  auto ready = EnsureHybridReady(backend, cfg, cache, g);
  if (!ready) {
    return std::unexpected(ready.error());
  }
  HybridDecodeCache& h = *cache.hybrid;
  auto alloc = AllocBatch(backend, cfg, g, h, rows);
  if (!alloc) {
    return std::unexpected(alloc.error());
  }
  const std::size_t hidden = cfg.hidden_dim;
  auto embed = NeedWeightAny(model, "token_embd.weight");
  if (!embed) {
    return std::unexpected(embed.error());
  }
  // Embed the tokens into a host staging buffer (quant weights gather on
  // the host) and upload once.
  std::vector<float> staged(rows * hidden);
  std::vector<float> row(hidden);
  for (std::size_t t = 0; t < rows; ++t) {
    if (tokens[t] >= cfg.vocab_size) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    auto gathered =
        GatherEmbedding(backend, **embed, tokens[t], hidden, row);
    if (!gathered) {
      return std::unexpected(gathered.error());
    }
    std::copy(row.begin(), row.end(), staged.begin() + t * hidden);
  }
  if (!UploadF32(backend, *h.batch->x, staged)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  // Position triples for mRoPE (text rows: t == h == w).
  std::vector<std::uint64_t> triples(rows * 3);
  for (std::size_t t = 0; t < rows; ++t) {
    const std::uint64_t p = h.position + t;
    triples[t * 3 + 0] = p;
    triples[t * 3 + 1] = p;
    triples[t * 3 + 2] = p;
  }
  if (!backend.CopyH2D(*h.batch->pos, std::span<const std::byte>(
                                  reinterpret_cast<const std::byte*>(
                                      triples.data()),
                                  triples.size() * 8))) {
    return std::unexpected(StatusCode::DeviceError);
  }
  const std::uint64_t start = h.position;
  for (std::size_t l = 0; l < cfg.layers; ++l) {
    if (cfg.IsFullAttentionLayer(l)) {
      auto block = RunFullBlockBatch(backend, model, cfg, h, l, start,
                                     h.full[l], rows);
      if (!block) {
        return std::unexpected(block.error());
      }
      continue;
    }
    auto block = RunLinearBlockBatch(backend, model, cfg, h, l, g, rows);
    if (!block) {
      return std::unexpected(block.error());
    }
  }
  h.position += rows;
  if (hidden_out != nullptr) {
    hidden_out->resize(hidden);
    auto last = detail::DownloadF32(backend, *h.batch->x);
    if (!last) {
      return std::unexpected(last.error());
    }
    std::copy(last->end() - hidden, last->end(), hidden_out->begin());
  }
  if (logits_out != nullptr) {
    auto out_norm = NeedWeight(model, "output_norm.weight", DType::F32);
    auto output = NeedWeightAny(model, "output.weight");
    if (!out_norm || !output) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    auto gemm_out = GemmFor(backend, h.gemms, (*output)->manifest.dtype);
    if (!gemm_out) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    if (all_logits && rows > 1) {
      if (!RmsNormDevice(backend, *h.rmsnorm_kernel, *h.batch->x,
                         *(*out_norm)->device, *h.batch->xn, rows, hidden,
                         cfg.norm_eps) ||
          !ProjectDevice(backend, *(*gemm_out), *h.batch->xn, *(*output)->device,
                         *h.batch->logits, rows, cfg.vocab_size, hidden)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      logits_out->resize(rows * cfg.vocab_size);
      if (!backend.CopyD2H(*h.batch->logits,
                           reinterpret_cast<std::byte*>(logits_out->data()),
                           rows * cfg.vocab_size * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    } else {
      // Only the last row needs logits (prefill): head on that row alone.
      if (!backend.CopyD2D(*h.batch->x, (rows - 1) * hidden * 4, *h.x, 0,
                           hidden * 4) ||
          !RmsNormDevice(backend, *h.rmsnorm_kernel, *h.x, *(*out_norm)->device,
                         *h.xn, 1, hidden, cfg.norm_eps) ||
          !ProjectDevice(backend, *(*gemm_out), *h.xn, *(*output)->device,
                         *h.logits, 1, cfg.vocab_size, hidden)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      logits_out->resize(cfg.vocab_size);
      if (!backend.CopyD2H(*h.logits,
                           reinterpret_cast<std::byte*>(logits_out->data()),
                           cfg.vocab_size * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
  }
  return {};
}

}  // namespace tessera::core
