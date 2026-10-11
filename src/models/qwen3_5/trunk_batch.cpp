#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/profile.hpp"
#include "core/numerics/conv.hpp"
#include "models/qwen3_5/architecture.hpp"
#include "models/qwen3_5/internal.hpp"
#include "models/qwen3_5/state.hpp"

namespace tessera::models::qwen3_5 {

namespace {

namespace detail = ::tessera::core::detail;

using core::Conv1dStepRef;
using core::DecodeCache;
using detail::AddDevice;
using detail::AppendKv;
using detail::AttentionDevice;
using detail::AttentionQuantDevice;
using detail::Conv1dStateDevice;
using detail::DeltaStepHeadsDevice;
using detail::DownloadF32;
using detail::DownloadF32Cached;
using detail::L2NormDevice;
using detail::MropeDevice;
using detail::NeedWeight;
using detail::NeedWeightAny;
using detail::QGateSplitDevice;
using detail::RepeatHeadsDevice;
using detail::RmsNormDevice;
using detail::RmsNormGatedDevice;
using detail::SigmoidGateDevice;
using detail::SiluMulDevice;
using detail::SsmGateDevice;

// Sizes (or resizes) a batched scratch for `rows` tokens. Two scratches are
// sized by need, not `rows`, because a long prompt must not allocate their
// full rows: the logits scratch (prefill scores only the last token) and the
// per-layer linear state snapshots (only a verification reads them).
std::expected<void, StatusCode> EnsureScratch(Backend& backend,
                                              const TransformerConfig& cfg,
                                              const LinearGeometry& g,
                                              Qwen35BatchScratch& s,
                                              std::size_t rows,
                                              std::size_t logits_rows,
                                              bool need_state_hist) {
  auto alloc = [&backend](std::unique_ptr<Buffer>& slot,
                          std::size_t elements) -> bool {
    auto buf = backend.AllocateBuffer(elements * 4, MemoryKind::Device);
    if (!buf) {
      return false;
    }
    slot = std::move(*buf);
    return true;
  };
  const std::size_t state_len = g.num_v_heads * g.head_k_dim * g.head_v_dim;
  const std::size_t hist_len = g.conv_dim * (g.width - 1);
  const auto alloc_hist = [&](Qwen35BatchScratch& t, std::size_t hist_rows) {
    t.state_hist.resize(cfg.layers);
    t.conv_hist_hist.resize(cfg.layers);
    for (std::size_t l = 0; l < cfg.layers; ++l) {
      if (!cfg.IsFullAttentionLayer(l)) {
        if (!alloc(t.state_hist[l], (hist_rows + 1) * state_len) ||
            !alloc(t.conv_hist_hist[l], (hist_rows + 1) * hist_len)) {
          return false;
        }
      }
    }
    t.hist_rows = hist_rows;
    return true;
  };
  if (s.capacity >= rows) {
    s.snapshot_states = need_state_hist;
    if (s.logits_capacity < logits_rows) {
      if (!alloc(s.logits, logits_rows * cfg.vocab_size)) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      s.logits_capacity = logits_rows;
    }
    if (need_state_hist && s.hist_rows < rows && !alloc_hist(s, rows)) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    return {};
  }
  s = Qwen35BatchScratch{};
  s.capacity = rows;
  s.logits_capacity = logits_rows;
  s.snapshot_states = need_state_hist;
  const std::size_t q_dim = cfg.attention.heads * cfg.attention.head_dim;
  const std::size_t kv_dim = cfg.attention.kv_heads * cfg.attention.head_dim;
  const std::size_t ffn = cfg.ffn_dim != 0 ? cfg.ffn_dim : 1;
  if (!alloc(s.x, rows * cfg.hidden_dim) ||
      !alloc(s.xn, rows * cfg.hidden_dim) ||
      !alloc(s.proj, rows * cfg.hidden_dim) ||
      !alloc(s.logits, logits_rows * cfg.vocab_size) ||
      !alloc(s.fused, rows * q_dim * 2) || !alloc(s.q, rows * q_dim) ||
      !alloc(s.gate, rows * q_dim) || !alloc(s.kf, rows * kv_dim) ||
      !alloc(s.vf, rows * kv_dim) || !alloc(s.attn, rows * q_dim) ||
      !alloc(s.fgate, rows * ffn) || !alloc(s.fup, rows * ffn) ||
      !alloc(s.fmlp, rows * ffn) || !alloc(s.qkv, rows * g.conv_dim) ||
      !alloc(s.z, rows * g.value_dim) ||
      !alloc(s.alpha_raw, rows * g.num_v_heads) ||
      !alloc(s.beta_raw, rows * g.num_v_heads) ||
      !alloc(s.alpha, rows * g.num_v_heads) ||
      !alloc(s.beta, rows * g.num_v_heads) || !alloc(s.out, rows * g.value_dim) ||
      !alloc(s.q_all, rows * g.key_dim) ||
      !alloc(s.k_all, rows * g.key_dim) ||
      !alloc(s.v_all, rows * g.value_dim) ||
      !alloc(s.q_exp_all, rows * g.num_v_heads * g.head_k_dim) ||
      !alloc(s.k_exp_all, rows * g.num_v_heads * g.head_k_dim) ||
      !alloc(s.core_all, rows * g.value_dim) ||
      (cfg.IsMoe() && !alloc(s.moe_out, rows * cfg.hidden_dim))) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  auto pos = backend.AllocateBuffer(rows * 3 * 8, MemoryKind::Device);
  if (!pos) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  s.pos = std::move(*pos);
  if (need_state_hist && !alloc_hist(s, rows)) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  return {};
}

// Sizes the per-cache batch scratch, creating it on first use.
std::expected<void, StatusCode> AllocBatch(Backend& backend,
                                           const TransformerConfig& cfg,
                                           const LinearGeometry& g,
                                           Qwen35State& h, std::size_t rows,
                                           std::size_t logits_rows,
                                           bool need_state_hist) {
  if (!h.batch) {
    h.batch = std::make_unique<Qwen35BatchScratch>();
  }
  return EnsureScratch(backend, cfg, g, *h.batch, rows, logits_rows,
                       need_state_hist);
}

// Batched gated MLP over `rows` rows of `b.x` in place. The scratch is
// passed explicitly so the multi-sequence path can share one scratch across
// sequences while the single-sequence path uses its own.
std::expected<void, StatusCode> RunFfnBatch(Backend& backend,
                                            const Model& model,
                                            const TransformerConfig& cfg,
                                            Qwen35State& h,
                                            Qwen35BatchScratch& b,
                                            std::size_t layer,
                                            std::size_t rows) {
  if (cfg.IsMoe()) {
    return RunMoeFfnBatch(backend, model, cfg, h, layer, rows);
  }
  core::Profile* const deep =
      h.profile != nullptr && h.profile->Deep() ? h.profile : nullptr;
  core::PhaseScope ffn_scope(deep, core::Phase::kFfn);
  const std::string base = "blk." + std::to_string(layer) + ".";
  auto mlp_norm =
      NeedWeight(model, base + "post_attention_norm.weight", DType::F32);
  auto fg = NeedWeightAny(model, base + "ffn_gate.weight");
  auto fu = NeedWeightAny(model, base + "ffn_up.weight");
  auto fd = NeedWeightAny(model, base + "ffn_down.weight");
  if (!mlp_norm || !fg || !fu || !fd) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (!RmsNormDevice(backend, *h.rmsnorm_kernel, *b.x, *(*mlp_norm)->device,
                     *b.xn, rows, cfg.hidden_dim, cfg.norm_eps)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  InvalidateActivationPack(h);
  if (!ProjectBatch(backend, h, (*fg)->manifest.dtype, *b.xn, *(*fg)->device,
                    *b.fgate, rows, cfg.ffn_dim, cfg.hidden_dim) ||
      !ProjectBatch(backend, h, (*fu)->manifest.dtype, *b.xn, *(*fu)->device,
                    *b.fup, rows, cfg.ffn_dim, cfg.hidden_dim) ||
      !SiluMulDevice(backend, *h.silu_mul_kernel, *b.fgate, *b.fup, *b.fmlp,
                     rows * cfg.ffn_dim) ||
      !ProjectBatch(backend, h, (*fd)->manifest.dtype, *b.fmlp, *(*fd)->device,
                    *b.proj, rows, cfg.hidden_dim, cfg.ffn_dim) ||
      !AddDevice(backend, *h.add_kernel, *b.x, *b.proj, *b.x,
                 rows * cfg.hidden_dim)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (deep != nullptr) {
    backend.Synchronize();  // profile attribution only
  }
  return {};
}

// Batched full-attention block over `rows` rows ending in h.batch->x.
std::expected<void, StatusCode> RunFullBlockBatch(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen35State& h, Qwen35BatchScratch& b, std::size_t layer, std::uint64_t pos,
    Qwen35State::FullKv& kv, std::size_t rows) {
  core::Profile* const deep =
      h.profile != nullptr && h.profile->Deep() ? h.profile : nullptr;
  core::PhaseScope layer_scope(deep, core::Phase::kFullLayer);
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
  if (!RmsNormDevice(backend, *h.rmsnorm_kernel, *b.x, *(*norm)->device, *b.xn,
                     rows, hidden, cfg.norm_eps)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  InvalidateActivationPack(h);
  if (!ProjectBatch(backend, h, (*wq)->manifest.dtype, *b.xn, *(*wq)->device,
                    *b.fused, rows, heads * head_dim * 2, hidden) ||
      !QGateSplitDevice(backend, *h.qgate_split_kernel, *b.fused, *b.q, *b.gate,
                        heads, head_dim, rows) ||
      !ProjectBatch(backend, h, (*wk)->manifest.dtype, *b.xn, *(*wk)->device,
                    *b.kf, rows, kv_dim, hidden) ||
      !ProjectBatch(backend, h, (*wv)->manifest.dtype, *b.xn, *(*wv)->device,
                    *b.vf, rows, kv_dim, hidden) ||
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
  // F32 caches append straight from the batch K/V at row t (no scratch
  // round-trip); the cast/quant paths still copy the row first.
  const bool kv_f32 = kv.type == KvCacheType::F32;
  for (std::size_t t = 0; t < rows; ++t) {
    const Buffer* ksrc = b.kf.get();
    const Buffer* vsrc = b.vf.get();
    std::size_t row_offset = 0;
    if (kv_f32) {
      row_offset = t * kv_dim * 4;
    } else {
      if (!backend.CopyD2D(*b.kf, t * kv_dim * 4, *h.kf, 0, kv_dim * 4) ||
          !backend.CopyD2D(*b.vf, t * kv_dim * 4, *h.vf, 0, kv_dim * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      ksrc = h.kf.get();
      vsrc = h.vf.get();
    }
    if (auto appended =
            AppendKv(backend, h.cast_kernel.get(), h.quant_kernel.get(),
                     h.kv_scratch.get(), h.kv_scratch.get(),
                     h.scale_scratch.get(), kv, *ksrc, *vsrc, kv_dim,
                     row_offset);
        !appended) {
      return std::unexpected(appended.error());
    }
  }
  const bool quantized = kv.type == KvCacheType::Q8 ||
                         kv.type == KvCacheType::Q4 ||
                         kv.type == KvCacheType::FP8;
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
      !ProjectBatch(backend, h, (*wo)->manifest.dtype, *b.attn, *(*wo)->device,
                    *b.proj, rows, hidden, heads * head_dim) ||
      !AddDevice(backend, *h.add_kernel, *b.x, *b.proj, *b.x,
                 rows * hidden)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (deep != nullptr) {
    backend.Synchronize();  // profile attribution only
  }
  return RunFfnBatch(backend, model, cfg, h, b, layer, rows);
}

// Batched linear-attention block. The projections run batched; the causal
// conv and the gated-delta scan run token by token with the shared state.
std::expected<void, StatusCode> RunLinearBlockBatch(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen35State& h, Qwen35BatchScratch& b, std::size_t layer,
    const LinearGeometry& g, std::size_t rows) {
  core::Profile* const deep =
      h.profile != nullptr && h.profile->Deep() ? h.profile : nullptr;
  core::PhaseScope layer_scope(deep, core::Phase::kLinearLayer);
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
  if (!RmsNormDevice(backend, *h.rmsnorm_kernel, *b.x, *(*norm)->device, *b.xn,
                     rows, hidden, cfg.norm_eps)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  InvalidateActivationPack(h);
  if (!ProjectBatch(backend, h, (*w_qkv)->manifest.dtype, *b.xn,
                    *(*w_qkv)->device, *b.qkv, rows, g.conv_dim, hidden) ||
      !ProjectBatch(backend, h, (*w_gate)->manifest.dtype, *b.xn,
                    *(*w_gate)->device, *b.z, rows, g.value_dim, hidden) ||
      !ProjectBatch(backend, h, (*w_beta)->manifest.dtype, *b.xn,
                    *(*w_beta)->device, *b.beta_raw, rows, g.num_v_heads,
                    hidden) ||
      !ProjectBatch(backend, h, (*w_alpha)->manifest.dtype, *b.xn,
                    *(*w_alpha)->device, *b.alpha_raw, rows, g.num_v_heads,
                    hidden) ||
      !SsmGateDevice(backend, *h.ssm_gate_kernel, *(*w_a)->device,
                     *(*w_dt)->device, *b.alpha_raw, *b.beta_raw, *b.alpha,
                     *b.beta, g.num_v_heads, rows)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  const float q_scale = 1.0 / std::sqrt(static_cast<double>(g.head_k_dim));
  const std::size_t state_len = g.num_v_heads * g.head_k_dim * g.head_v_dim;
  const std::size_t hist_len = g.conv_dim * (g.width - 1);
  // Snapshot the pre-block state and conv history (slot 0). The batched conv
  // and DeltaStep then write each row's post-step state to its own slot (row
  // t reads slot t, writes slot t+1) so the commit restores any accepted
  // prefix. Prefill never rolls back, so it skips both writes.
  if (b.snapshot_states &&
      (!backend.CopyD2D(*h.linear[layer].state, 0, *b.state_hist[layer], 0,
                        state_len * 4) ||
       !backend.CopyD2D(*h.linear[layer].conv_hist, 0,
                        *b.conv_hist_hist[layer], 0, hist_len * 4))) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (rows > 1) {
    // Prefill and the batched verify run one launch per kernel for all rows
    // instead of about twelve launches per row: the conv advances the shared
    // history across the rows, then the L2 norms, head repeat, delta scan and
    // gated norm run over all rows.
    Buffer& state =
        b.snapshot_states ? *b.state_hist[layer] : *h.linear[layer].state;
    if (!Conv1dStateDevice(
            backend, *h.conv1d_state_kernel, *b.qkv, *(*w_conv)->device,
            *h.linear[layer].conv_hist, *b.q_all, *b.k_all, *b.v_all,
            g.conv_dim, g.width, g.key_dim, 0, rows,
            b.snapshot_states ? b.conv_hist_hist[layer].get() : nullptr,
            b.snapshot_states ? hist_len : 0) ||
        !L2NormDevice(backend, *h.l2norm_kernel, *b.q_all, *b.q_all,
                      rows * g.num_k_heads, g.head_k_dim, 1e-6f, q_scale) ||
        !L2NormDevice(backend, *h.l2norm_kernel, *b.k_all, *b.k_all,
                      rows * g.num_k_heads, g.head_k_dim, 1e-6f) ||
        !RepeatHeadsDevice(backend, *h.repeat_heads_kernel, *b.q_all,
                           *b.q_exp_all, g.num_v_heads, g.head_k_dim, g.factor,
                           rows) ||
        !RepeatHeadsDevice(backend, *h.repeat_heads_kernel, *b.k_all,
                           *b.k_exp_all, g.num_v_heads, g.head_k_dim, g.factor,
                           rows) ||
        !DeltaStepHeadsDevice(backend, *h.delta_step_heads_kernel, state,
                              *b.k_exp_all, *b.v_all, *b.q_exp_all, *b.core_all,
                              *b.alpha, *b.beta, g.num_v_heads, g.head_k_dim,
                              g.head_v_dim, rows, 0,
                              b.snapshot_states ? state_len : 0, 0) ||
        !RmsNormGatedDevice(backend, *h.rmsnorm_gated_kernel, *b.core_all,
                            *(*w_norm)->device, *b.z, *b.out,
                            rows * g.num_v_heads, g.head_v_dim, cfg.norm_eps) ||
        !ProjectBatch(backend, h, (*w_out)->manifest.dtype, *b.out,
                      *(*w_out)->device, *b.proj, rows, hidden, g.value_dim) ||
        !AddDevice(backend, *h.add_kernel, *b.x, *b.proj, *b.x, rows * hidden)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (deep != nullptr) {
      backend.Synchronize();  // profile attribution only
    }
    return RunFfnBatch(backend, model, cfg, h, b, layer, rows);
  }
  for (std::size_t t = 0; t < rows; ++t) {
    // The causal conv, SiLU and q/k/v split run on the device; the
    // history lives in linear[layer].conv_hist and the kernel shifts it.
    if (!Conv1dStateDevice(backend, *h.conv1d_state_kernel, *b.qkv,
                           *(*w_conv)->device, *h.linear[layer].conv_hist,
                           *h.q_l, *h.k_l, *h.v_l, g.conv_dim, g.width,
                           g.key_dim, t * g.conv_dim) ||
        !L2NormDevice(backend, *h.l2norm_kernel, *h.q_l, *h.q_l, g.num_k_heads,
                      g.head_k_dim, 1e-6f, q_scale) ||
        !L2NormDevice(backend, *h.l2norm_kernel, *h.k_l, *h.k_l, g.num_k_heads,
                      g.head_k_dim, 1e-6f) ||
        !RepeatHeadsDevice(backend, *h.repeat_heads_kernel, *h.q_l, *h.q_exp,
                           g.num_v_heads, g.head_k_dim, g.factor) ||
        !RepeatHeadsDevice(backend, *h.repeat_heads_kernel, *h.k_l, *h.k_exp,
                           g.num_v_heads, g.head_k_dim, g.factor) ||
        !DeltaStepHeadsDevice(
            backend, *h.delta_step_heads_kernel,
            b.snapshot_states ? *b.state_hist[layer]
                              : *h.linear[layer].state,
            *h.k_exp, *h.v_l, *h.q_exp, *h.core, *b.alpha, *b.beta,
            g.num_v_heads, g.head_k_dim, g.head_v_dim, 1,
            b.snapshot_states ? t * state_len : 0,
            b.snapshot_states ? state_len : 0, t * g.num_v_heads) ||
        !RmsNormGatedDevice(backend, *h.rmsnorm_gated_kernel, *h.core,
                            *(*w_norm)->device, *b.z, *b.out, g.num_v_heads,
                            g.head_v_dim, cfg.norm_eps, t * g.value_dim,
                            t * g.value_dim)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    // The DeltaStep already wrote this token's post-state to slot t+1 of the
    // state history, so only the conv history needs a copy for rollback.
    if (b.snapshot_states &&
        !backend.CopyD2D(*h.linear[layer].conv_hist, 0,
                         *b.conv_hist_hist[layer], (t + 1) * hist_len * 4,
                         hist_len * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  if (!ProjectBatch(backend, h, (*w_out)->manifest.dtype, *b.out,
                    *(*w_out)->device, *b.proj, rows, hidden, g.value_dim) ||
      !AddDevice(backend, *h.add_kernel, *b.x, *b.proj, *b.x,
                 rows * hidden)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (deep != nullptr) {
    backend.Synchronize();  // profile attribution only
  }
  return RunFfnBatch(backend, model, cfg, h, b, layer, rows);
}

// Runs the vocab head over the batch. `all_rows` projects every row (the
// batched verify with all_logits); otherwise only the last row (prefill).
// The device logits land in h.batch->logits / h.logits; when `download` is
// non-null they are copied to the host, otherwise they stay on the device
// for a device-side argmax.
std::expected<void, StatusCode> RunBatchHead(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen35State& h, Qwen35BatchScratch& b, std::size_t rows, bool all_rows,
    std::vector<float>* download) {
  auto out_norm = NeedWeight(model, "output_norm.weight", DType::F32);
  auto output = NeedWeightAny(model, "output.weight");
  if (!out_norm || !output) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const std::size_t hidden = cfg.hidden_dim;
  if (all_rows && rows > 1) {
    if (!RmsNormDevice(backend, *h.rmsnorm_kernel, *b.x, *(*out_norm)->device,
                       *b.xn, rows, hidden, cfg.norm_eps) ||
        !ProjectBatch(backend, h, (*output)->manifest.dtype, *b.xn,
                      *(*output)->device, *b.logits, rows, cfg.vocab_size,
                      hidden)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (download != nullptr) {
      download->resize(rows * cfg.vocab_size);
      if (!backend.CopyD2H(*b.logits,
                           reinterpret_cast<std::byte*>(download->data()),
                           rows * cfg.vocab_size * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
  } else {
    if (!backend.CopyD2D(*b.x, (rows - 1) * hidden * 4, *h.x, 0, hidden * 4) ||
        !RmsNormDevice(backend, *h.rmsnorm_kernel, *h.x, *(*out_norm)->device,
                       *h.xn, 1, hidden, cfg.norm_eps) ||
        !ProjectBatch(backend, h, (*output)->manifest.dtype, *h.xn,
                      *(*output)->device, *h.logits, 1, cfg.vocab_size,
                      hidden)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (download != nullptr) {
      download->resize(cfg.vocab_size);
      if (!backend.CopyD2H(*h.logits,
                           reinterpret_cast<std::byte*>(download->data()),
                           cfg.vocab_size * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
  }
  return {};
}

// Full-attention block for a multi-sequence batch: `rows` sequences each
// contribute one token. The projections, norms and RoPE run over all rows in
// `b`; the attention runs per sequence over that sequence's own KV, with
// `row_base = t` selecting row t of the shared q/attn buffers and `pos[t]`
// the sequence's position.
std::expected<void, StatusCode> RunFullBlockBatchMulti(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen35State& driver, Qwen35BatchScratch& b, std::size_t layer,
    std::span<Qwen35State* const> seqs, std::span<const std::uint64_t> pos,
    std::size_t rows) {
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
  if (!RmsNormDevice(backend, *driver.rmsnorm_kernel, *b.x, *(*norm)->device,
                     *b.xn, rows, hidden, cfg.norm_eps)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  InvalidateActivationPack(driver);
  if (!ProjectBatch(backend, driver, (*wq)->manifest.dtype, *b.xn,
                    *(*wq)->device, *b.fused, rows, heads * head_dim * 2,
                    hidden) ||
      !QGateSplitDevice(backend, *driver.qgate_split_kernel, *b.fused, *b.q,
                        *b.gate, heads, head_dim, rows) ||
      !ProjectBatch(backend, driver, (*wk)->manifest.dtype, *b.xn,
                    *(*wk)->device, *b.kf, rows, kv_dim, hidden) ||
      !ProjectBatch(backend, driver, (*wv)->manifest.dtype, *b.xn,
                    *(*wv)->device, *b.vf, rows, kv_dim, hidden) ||
      !RmsNormDevice(backend, *driver.rmsnorm_kernel, *b.q, *(*q_norm)->device,
                     *b.q, rows * heads, head_dim, cfg.norm_eps) ||
      !RmsNormDevice(backend, *driver.rmsnorm_kernel, *b.kf, *(*k_norm)->device,
                     *b.kf, rows * kv_heads, head_dim, cfg.norm_eps) ||
      !detail::MropeDevice(backend, *driver.mrope_kernel, *b.q, *b.pos, rows,
                           heads, head_dim, rope_dim, sec_t, sec_h, sec_w,
                           cfg.attention.rope_theta) ||
      !detail::MropeDevice(backend, *driver.mrope_kernel, *b.kf, *b.pos, rows,
                           kv_heads, head_dim, rope_dim, sec_t, sec_h, sec_w,
                           cfg.attention.rope_theta)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  // Append each sequence's key/value row to its own cache.
  for (std::size_t t = 0; t < rows; ++t) {
    Qwen35State& s = *seqs[t];
    Qwen35State::FullKv& kv = s.full[layer];
    const Buffer* ksrc = b.kf.get();
    const Buffer* vsrc = b.vf.get();
    std::size_t row_offset = 0;
    if (kv.type == KvCacheType::F32) {
      row_offset = t * kv_dim * 4;
    } else {
      if (!backend.CopyD2D(*b.kf, t * kv_dim * 4, *s.kf, 0, kv_dim * 4) ||
          !backend.CopyD2D(*b.vf, t * kv_dim * 4, *s.vf, 0, kv_dim * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      ksrc = s.kf.get();
      vsrc = s.vf.get();
    }
    if (auto appended =
            detail::AppendKv(backend, s.cast_kernel.get(), s.quant_kernel.get(),
                             s.kv_scratch.get(), s.kv_scratch.get(),
                             s.scale_scratch.get(), kv, *ksrc, *vsrc, kv_dim,
                             row_offset);
        !appended) {
      return std::unexpected(appended.error());
    }
  }
  // Attention per sequence over its own KV, one row via row_base.
  for (std::size_t t = 0; t < rows; ++t) {
    Qwen35State& s = *seqs[t];
    Qwen35State::FullKv& kv = s.full[layer];
    const bool quantized = kv.type == KvCacheType::Q8 ||
                           kv.type == KvCacheType::Q4 ||
                           kv.type == KvCacheType::FP8;
    const bool ok =
        quantized
            ? detail::AttentionQuantDevice(
                  backend, *driver.attention_kernel, *b.q, *kv.k, *kv.v,
                  *kv.k_scale, *kv.v_scale, *b.attn, kv.rows, heads, kv_heads,
                  head_dim, pos[t], 0, 1, t)
                  .has_value()
            : AttentionDevice(backend, *driver.attention_kernel, *b.q, *kv.k,
                              *kv.v, *b.attn, kv.rows, heads, kv_heads,
                              head_dim, pos[t], 0, 1,
                              kv.type == KvCacheType::F16, true, t)
                  .has_value();
    if (!ok) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  if (!SigmoidGateDevice(backend, *driver.sigmoid_gate_kernel, *b.attn,
                         *b.gate, *b.attn, rows * heads * head_dim) ||
      !ProjectBatch(backend, driver, (*wo)->manifest.dtype, *b.attn,
                    *(*wo)->device, *b.proj, rows, hidden, heads * head_dim) ||
      !AddDevice(backend, *driver.add_kernel, *b.x, *b.proj, *b.x,
                 rows * hidden)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return RunFfnBatch(backend, model, cfg, driver, b, layer, rows);
}

}  // namespace

std::expected<void, StatusCode> Qwen35Architecture::ForwardBatch(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::span<const std::uint32_t> tokens, std::vector<float>* logits_out,
    std::vector<float>* hidden_out, bool all_logits,
    const Buffer* embeddings, const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture) const {
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
  auto ready = EnsureHybridReady(backend, cfg, cache, g, model.Shard());
  if (!ready) {
    return std::unexpected(ready.error());
  }
  Qwen35State& h = State(cache);
  const std::size_t logits_rows = (all_logits && rows > 1) ? rows : 1;
  // A verification may roll back to a fresh state even for a one-token
  // draft, so it always needs the pre-block snapshot (slot 0).
  const bool need_state_hist = all_logits;
  auto alloc = AllocBatch(backend, cfg, g, h, rows, logits_rows,
                          need_state_hist);
  if (!alloc) {
    return std::unexpected(alloc.error());
  }
  const std::size_t hidden = cfg.hidden_dim;
  if (embeddings != nullptr) {
    // A caller-supplied embedding per row (text rows and image rows).
    if (embeddings->Size() < rows * hidden * 4 ||
        !backend.CopyD2D(*embeddings, 0, *h.batch->x, 0, rows * hidden * 4)) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
  } else {
    auto gathered =
        GatherEmbeddingRows(backend, model, h, tokens, hidden, *h.batch->x);
    if (!gathered) {
      return std::unexpected(gathered.error());
    }
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
  // Optionally copy the per-row residual hidden after selected layers, for
  // the DFlash2 draft's target hidden states at every scored position.
  const auto capture_layer = [&](std::size_t layer) -> StatusCode {
    if (capture_layers == nullptr || capture == nullptr) {
      return StatusCode::Ok;
    }
    for (std::size_t i = 0; i < capture_layers->size(); ++i) {
      if ((*capture_layers)[i] == layer &&
          !backend.CopyD2D(*h.batch->x, 0, *(*capture)[i], 0,
                           rows * hidden * 4)) {
        return StatusCode::DeviceError;
      }
    }
    return StatusCode::Ok;
  };
  const std::uint64_t start = h.position;
  const ModelShard& shard = model.Shard();
  const std::size_t layer_begin = shard.enabled ? shard.first_layer : 0;
  std::size_t layer_end = cfg.layers;
  if (shard.enabled && shard.last_layer != 0 && shard.last_layer < layer_end) {
    layer_end = shard.last_layer;
  }
  for (std::size_t l = layer_begin; l < layer_end; ++l) {
    if (cfg.IsFullAttentionLayer(l)) {
      auto block = RunFullBlockBatch(backend, model, cfg, h, *h.batch, l, start,
                                     h.full[l], rows);
      if (!block) {
        return std::unexpected(block.error());
      }
    } else {
      auto block =
          RunLinearBlockBatch(backend, model, cfg, h, *h.batch, l, g, rows);
      if (!block) {
        return std::unexpected(block.error());
      }
    }
    if (capture_layer(l) != StatusCode::Ok) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  h.position += rows;
  if (hidden_out != nullptr) {
    // The batch scratch outlives its rows (a later smaller batch reuses
    // the capacity), so the last row is copied out first: reading the
    // buffer tail would return a stale row.
    hidden_out->resize(hidden);
    if (!backend.CopyD2D(*h.batch->x, (rows - 1) * hidden * 4, *h.x, 0,
                         hidden * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    auto last = detail::DownloadF32(backend, *h.x);
    if (!last) {
      return std::unexpected(last.error());
    }
    std::copy(last->begin(), last->end(), hidden_out->begin());
  }
  if (logits_out != nullptr) {
    auto head = RunBatchHead(backend, model, cfg, h, *h.batch, rows, all_logits,
                             logits_out);
    if (!head) {
      return std::unexpected(head.error());
    }
  }
  return {};
}

// Score a multi-token draft in one batched forward and roll the cache
// back to the accepted prefix (full-attention KV rows, linear-attention
// state, conv history and position).
std::expected<DraftVerification, StatusCode> Qwen35Architecture::Verify(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::span<const std::uint32_t> draft,
    std::span<const float> prefix_logits, std::optional<std::uint32_t> anchor,
    std::vector<float>* hidden_out,
    const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture) const {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  auto geometry = DeriveGeometry(*config);
  if (!geometry) {
    return std::unexpected(geometry.error());
  }
  const LinearGeometry& g = *geometry;
  const std::size_t state_len = g.num_v_heads * g.head_k_dim * g.head_v_dim;
  const std::size_t hist_len = g.conv_dim * (g.width - 1);
  Qwen35State& h = State(cache);
  const std::size_t prefix = h.position;
  // The scored batch: [anchor, drafts...] when the anchor rides the verify,
  // otherwise just the drafts (the prefix logits score draft[0]).
  const std::size_t lead = anchor.has_value() ? 1 : 0;
  std::vector<std::uint32_t> anchored;
  std::span<const std::uint32_t> tokens = draft;
  if (lead == 1) {
    anchored.reserve(draft.size() + 1);
    anchored.push_back(*anchor);
    anchored.insert(anchored.end(), draft.begin(), draft.end());
    tokens = anchored;
  }
  const std::size_t vocab = config->vocab_size;
  DraftVerification result;
  std::size_t accepted = 0;
  if (lead == 1) {
    // The anchor rides the batch. The engine reads only `accepted` and
    // `next_token`, so argmax every scored row on the device and read back
    // `rows` ids instead of downloading rows x vocab floats and scanning them
    // on the host (about 9 MB per step at the draft size).
    auto status = ForwardBatch(backend, model, cache, tokens, nullptr, nullptr,
                               /*all_logits=*/true, nullptr, capture_layers,
                               capture);
    if (!status) {
      return std::unexpected(status.error());
    }
    auto head = RunBatchHead(backend, model, *config, h, *h.batch,
                             tokens.size(), true, nullptr);
    if (!head) {
      return std::unexpected(head.error());
    }
    auto best =
        DeviceRowArgMax(backend, h, *h.batch->logits, tokens.size(), vocab);
    if (!best) {
      return std::unexpected(best.error());
    }
    for (std::size_t i = 0; i < draft.size(); ++i) {
      if ((*best)[i] != draft[i]) {
        break;
      }
      ++accepted;
    }
    result.accepted = accepted;
    result.next_token = (*best)[accepted];
  } else {
    // No anchor: the caller's prefix logits score draft[0], and the caller
    // reads `result.logits` as the continuation row, so keep the host path.
    std::vector<float> flat;
    auto status = ForwardBatch(backend, model, cache, tokens, &flat, nullptr,
                               /*all_logits=*/true, nullptr, capture_layers,
                               capture);
    if (!status) {
      return std::unexpected(status.error());
    }
    std::span<const float> last =
        std::span<const float>(prefix_logits.begin(), prefix_logits.end());
    for (std::size_t i = 0; i < draft.size(); ++i) {
      if (detail::ArgMax(last) != draft[i]) {
        break;
      }
      ++accepted;
      last = std::span<const float>(flat.data() + (i + lead) * vocab, vocab);
    }
    result.accepted = accepted;
    result.logits.assign(last.begin(), last.end());
    result.next_token = detail::ArgMax(result.logits);
  }
  // Keep the anchor (when present) plus the accepted drafts; drop the rest.
  // The restore always runs: the batched verify advances the recurrent state
  // through the history slots (each DeltaStep writes its own slot), so the
  // live state buffer is stale even when every draft is accepted.
  const std::size_t committed = lead + accepted;
  {
    const std::size_t new_pos = prefix + committed;
    for (auto& kv : h.full) {
      kv.rows = new_pos;
    }
    h.position = new_pos;
    for (std::size_t l = 0; l < config->layers; ++l) {
      if (config->IsFullAttentionLayer(l)) {
        continue;
      }
      if (!backend.CopyD2D(*h.batch->state_hist[l], committed * state_len * 4,
                           *h.linear[l].state, 0, state_len * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      if (!backend.CopyD2D(*h.batch->conv_hist_hist[l],
                           committed * hist_len * 4, *h.linear[l].conv_hist, 0,
                           hist_len * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
  }
  if (hidden_out != nullptr && (accepted > 0 || lead == 1)) {
    // The last committed row: the accepted draft, the anchor with no
    // accepted draft, or the accepted prefix's last draft without one.
    auto xh = detail::DownloadF32(backend, *h.batch->x);
    if (xh) {
      hidden_out->resize(config->hidden_dim);
      const std::size_t row = lead == 1 ? accepted : accepted - 1;
      std::copy(xh->begin() + row * config->hidden_dim,
                xh->begin() + (row + 1) * config->hidden_dim,
                hidden_out->begin());
    }
  }
  return result;
}

// Batched greedy decode of several independent sequences. The dense
// projections and FFN run over all sequences in one shared scratch; the
// full-attention layers run the attention per sequence over that
// sequence's own KV with `row_base`; the hybrid linear layers run per
// sequence because their conv/delta state is per sequence. Returns the
// greedy token per sequence.
std::expected<std::vector<std::uint32_t>, StatusCode>
Qwen35Architecture::GreedyBatchMulti(
    Backend& backend, const Model& model,
    std::span<core::DecodeCache* const> caches,
    std::span<const std::uint32_t> tokens) const {
  const std::size_t rows = caches.size();
  if (rows == 0 || tokens.size() != rows) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const TransformerConfig& cfg = *config;
  if (!cfg.hybrid || cfg.full_attention_interval == 0 ||
      cfg.rope_sections.size() != 3) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (cfg.IsMoe()) {
    // The MoE FFN batch path is not multi-sequence yet.
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  auto geometry = DeriveGeometry(cfg);
  if (!geometry) {
    return std::unexpected(geometry.error());
  }
  const LinearGeometry& g = *geometry;
  for (std::size_t t = 0; t < rows; ++t) {
    if (caches[t] == nullptr) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    auto ready = EnsureHybridReady(backend, cfg, *caches[t], g, model.Shard());
    if (!ready) {
      return std::unexpected(ready.error());
    }
  }
  Qwen35State& driver = State(*caches[0]);
  const std::size_t hidden = cfg.hidden_dim;
  if (!driver.multi) {
    driver.multi = std::make_unique<Qwen35BatchScratch>();
  }
  auto scratch = EnsureScratch(backend, cfg, g, *driver.multi, rows, 1, false);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  Qwen35BatchScratch& b = *driver.multi;
  auto gathered =
      GatherEmbeddingRows(backend, model, driver, tokens, hidden, *b.x);
  if (!gathered) {
    return std::unexpected(gathered.error());
  }
  std::vector<Qwen35State*> seqs(rows);
  std::vector<std::uint64_t> pos(rows);
  std::vector<std::uint64_t> triples(rows * 3);
  for (std::size_t t = 0; t < rows; ++t) {
    Qwen35State& s = State(*caches[t]);
    seqs[t] = &s;
    if (!s.batch) {
      auto sized = AllocBatch(backend, cfg, g, s, 1, 1, false);
      if (!sized) {
        return std::unexpected(sized.error());
      }
    }
    pos[t] = s.position;
    triples[3 * t] = pos[t];
    triples[3 * t + 1] = pos[t];
    triples[3 * t + 2] = pos[t];
  }
  if (!backend.CopyH2D(*b.pos,
                       std::span<const std::byte>(
                           reinterpret_cast<const std::byte*>(triples.data()),
                           triples.size() * sizeof(std::uint64_t)))) {
    return std::unexpected(StatusCode::DeviceError);
  }
  const ModelShard& shard = model.Shard();
  const std::size_t layer_begin = shard.enabled ? shard.first_layer : 0;
  std::size_t layer_end = cfg.layers;
  if (shard.enabled && shard.last_layer != 0 && shard.last_layer < layer_end) {
    layer_end = shard.last_layer;
  }
  for (std::size_t l = layer_begin; l < layer_end; ++l) {
    if (cfg.IsFullAttentionLayer(l)) {
      auto block = RunFullBlockBatchMulti(backend, model, cfg, driver, b, l,
                                          seqs, pos, rows);
      if (!block) {
        return std::unexpected(block.error());
      }
      continue;
    }
    for (std::size_t t = 0; t < rows; ++t) {
      Qwen35State& s = *seqs[t];
      if (!backend.CopyD2D(*b.x, t * hidden * 4, *s.batch->x, 0, hidden * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      auto lin = RunLinearBlockBatch(backend, model, cfg, s, *s.batch, l, g, 1);
      if (!lin) {
        return std::unexpected(lin.error());
      }
      if (!backend.CopyD2D(*s.batch->x, 0, *b.x, t * hidden * 4, hidden * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
  }
  auto out_norm = NeedWeight(model, "output_norm.weight", DType::F32);
  auto output = NeedWeightAny(model, "output.weight");
  if (!out_norm || !output) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::vector<std::uint32_t> result(rows);
  for (std::size_t t = 0; t < rows; ++t) {
    Qwen35State& s = *seqs[t];
    if (!backend.CopyD2D(*b.x, t * hidden * 4, *s.x, 0, hidden * 4) ||
        !RmsNormDevice(backend, *driver.rmsnorm_kernel, *s.x,
                       *(*out_norm)->device, *s.xn, 1, hidden, cfg.norm_eps) ||
        !ProjectBatch(backend, driver, (*output)->manifest.dtype, *s.xn,
                      *(*output)->device, *s.logits, 1, cfg.vocab_size,
                      hidden)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    auto best = DeviceRowArgMax(backend, s, *s.logits, 1, cfg.vocab_size);
    if (!best) {
      return std::unexpected(best.error());
    }
    result[t] = (*best)[0];
    s.position += 1;
  }
  return result;
}

}  // namespace tessera::models::qwen3_5
