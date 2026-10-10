#include "core/decode.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/conv.hpp"
#include "core/profile.hpp"
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
using detail::Conv1dStateDevice;
using detail::DeltaStepHeadsDevice;
using detail::DownloadF32;
using detail::DownloadF32Cached;
using detail::GatherEmbedding;
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
using detail::UploadF32;

std::expected<void, StatusCode> InitScratch(Backend& backend,
                                            const TransformerConfig& cfg,
                                            Qwen35State& h,
                                            const LinearGeometry& g) {
  const std::size_t q_dim = cfg.attention.heads * cfg.attention.head_dim;
  const std::size_t kv_dim = cfg.attention.kv_heads * cfg.attention.head_dim;
  auto alloc = [&backend](std::size_t bytes) {
    return backend.AllocateBuffer(bytes, MemoryKind::Device);
  };
  auto make = [&](std::unique_ptr<Buffer>& slot,
                  std::size_t elements) -> std::expected<void, StatusCode> {
    auto buf = alloc(elements * 4);
    if (!buf) {
      return std::unexpected(buf.error());
    }
    slot = std::move(*buf);
    return {};
  };
  if (!make(h.x, cfg.hidden_dim) || !make(h.xn, cfg.hidden_dim) ||
      !make(h.proj, cfg.hidden_dim) || !make(h.logits, cfg.vocab_size) ||
      !make(h.fused, q_dim * 2) || !make(h.q, q_dim) || !make(h.gate, q_dim) ||
      !make(h.kf, kv_dim) || !make(h.vf, kv_dim) || !make(h.attn, q_dim) ||
      !make(h.fgate, cfg.ffn_dim) || !make(h.fup, cfg.ffn_dim) ||
      !make(h.fmlp, cfg.ffn_dim) || !make(h.qkv, g.conv_dim) ||
      !make(h.z, g.value_dim) || !make(h.alpha_raw, g.num_v_heads) ||
      !make(h.beta_raw, g.num_v_heads) || !make(h.alpha, g.num_v_heads) ||
      !make(h.beta, g.num_v_heads) ||
      !make(h.q_l, g.key_dim) || !make(h.k_l, g.key_dim) ||
      !make(h.q_exp, g.num_v_heads * g.head_k_dim) ||
      !make(h.k_exp, g.num_v_heads * g.head_k_dim) ||
      !make(h.v_l, g.value_dim) || !make(h.core, g.value_dim) ||
      !make(h.out, g.value_dim) || !make(h.mtp_fused, 2 * cfg.hidden_dim) ||
      !make(h.mtp_h, cfg.hidden_dim)) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  auto pos = alloc(3 * 8);  // three u64 triples
  if (!pos) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  h.pos = std::move(*pos);
  h.full.resize(cfg.layers);
  h.linear.resize(cfg.layers);
  const std::size_t hist_len = g.conv_dim * (g.width - 1);
  const std::size_t state_len =
      g.num_v_heads * g.head_k_dim * g.head_v_dim;
  std::vector<float> zeros;
  for (std::size_t l = 0; l < cfg.layers; ++l) {
    if (!cfg.IsFullAttentionLayer(l)) {
      // The conv history lives on the device; the kernel shifts it.
      auto conv = alloc(hist_len * 4);
      if (!conv) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      zeros.assign(hist_len, 0.0f);
      if (!UploadF32(backend, **conv, zeros)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      h.linear[l].conv_hist = std::move(*conv);
      auto state = alloc(state_len * 4);
      if (!state) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      zeros.assign(state_len, 0.0f);
      if (!UploadF32(backend, **state, zeros)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      h.linear[l].state = std::move(*state);
    }
  }
  return {};
}

// Load the split-N kernels and (re)allocate the partial scratch for the
// single-token path. The kernels and scratch live in the state and are
// reused by every full-attention layer.
std::expected<void, StatusCode> EnsureSplitReady(Backend& backend,
                                                 const TransformerConfig& cfg,
                                                 Qwen35State& h,
                                                 KvCacheType type) {
  const std::size_t heads = cfg.attention.heads;
  const std::size_t head_dim = cfg.attention.head_dim;
  const std::size_t chunks = detail::kSplitChunks;
  if (h.split_acc == nullptr || h.split_heads != heads ||
      h.split_dim != head_dim || h.split_chunks != chunks) {
    const std::size_t part = heads * chunks;
    auto acc = backend.AllocateBuffer(part * head_dim * 4, MemoryKind::Device);
    auto mx = backend.AllocateBuffer(part * 4, MemoryKind::Device);
    auto sm = backend.AllocateBuffer(part * 4, MemoryKind::Device);
    if (!acc || !mx || !sm) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    h.split_acc = std::move(*acc);
    h.split_max = std::move(*mx);
    h.split_sum = std::move(*sm);
    h.split_heads = heads;
    h.split_dim = head_dim;
    h.split_chunks = chunks;
  }
  if (h.attention_combine_kernel == nullptr) {
    auto kernel = backend.LoadKernel("attention_combine", {});
    if (!kernel) {
      return std::unexpected(kernel.error());
    }
    h.attention_combine_kernel = std::move(*kernel);
  }
  if (h.attention_split_kernel == nullptr) {
    const char* name = type == KvCacheType::Q8    ? "attention_q8_split"
                       : type == KvCacheType::Q4  ? "attention_q4_split"
                       : type == KvCacheType::FP8 ? "attention_fp8_split"
                                                  : "attention_split";
    auto kernel = backend.LoadKernel(name, {});
    if (!kernel) {
      return std::unexpected(kernel.error());
    }
    h.attention_split_kernel = std::move(*kernel);
  }
  return {};
}

// Runs the output head (final norm plus the vocab projection) for the
// current hidden h.x, leaving the logits in h.logits on the device.
std::expected<void, StatusCode> RunOutputHead(Backend& backend,
                                              const Model& model,
                                              const TransformerConfig& cfg,
                                              Qwen35State& h) {
  const std::size_t hidden = cfg.hidden_dim;
  auto out_norm = NeedWeight(model, "output_norm.weight", DType::F32);
  auto output = NeedWeightAny(model, "output.weight");
  if (!out_norm || !output) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (!RmsNormDevice(backend, *h.rmsnorm_kernel, *h.x, *(*out_norm)->device,
                     *h.xn, 1, hidden, cfg.norm_eps) ||
      !ProjectBatch(backend, h, (*output)->manifest.dtype, *h.xn,
                    *(*output)->device, *h.logits, 1, cfg.vocab_size, hidden)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

}  // namespace

std::expected<void, StatusCode> RunFfn(Backend& backend, const Model& model,
                                       const TransformerConfig& cfg,
                                       Qwen35State& h,
                                       std::size_t layer) {
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
  if (!RmsNormDevice(backend, *h.rmsnorm_kernel, *h.x, *(*mlp_norm)->device,
                     *h.xn, 1, cfg.hidden_dim, cfg.norm_eps) ||
      !ProjectBatch(backend, h, (*fg)->manifest.dtype, *h.xn, *(*fg)->device,
                    *h.fgate, 1, cfg.ffn_dim, cfg.hidden_dim) ||
      !ProjectBatch(backend, h, (*fu)->manifest.dtype, *h.xn, *(*fu)->device,
                    *h.fup, 1, cfg.ffn_dim, cfg.hidden_dim) ||
      !SiluMulDevice(backend, *h.silu_mul_kernel, *h.fgate, *h.fup, *h.fmlp,
                     cfg.ffn_dim) ||
      !ProjectBatch(backend, h, (*fd)->manifest.dtype, *h.fmlp, *(*fd)->device,
                    *h.proj, 1, cfg.hidden_dim, cfg.ffn_dim) ||
      !AddDevice(backend, *h.add_kernel, *h.x, *h.proj, *h.x, cfg.hidden_dim)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (deep != nullptr) {
    backend.Synchronize();  // profile attribution only
  }
  return {};
}

std::expected<void, StatusCode> RunFullBlock(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen35State& h, std::size_t layer, std::uint64_t pos,
    Qwen35State::FullKv& kv) {
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
  if (!RmsNormDevice(backend, *h.rmsnorm_kernel, *h.x, *(*norm)->device, *h.xn,
                     1, hidden, cfg.norm_eps) ||
      !ProjectBatch(backend, h, (*wq)->manifest.dtype, *h.xn, *(*wq)->device,
                    *h.fused, 1, heads * head_dim * 2, hidden) ||
      !QGateSplitDevice(backend, *h.qgate_split_kernel, *h.fused, *h.q, *h.gate,
                        heads, head_dim) ||
      !ProjectBatch(backend, h, (*wk)->manifest.dtype, *h.xn, *(*wk)->device,
                    *h.kf, 1, kv_dim, hidden) ||
      !ProjectBatch(backend, h, (*wv)->manifest.dtype, *h.xn, *(*wv)->device,
                    *h.vf, 1, kv_dim, hidden) ||
      !RmsNormDevice(backend, *h.rmsnorm_kernel, *h.q, *(*q_norm)->device, *h.q,
                     heads, head_dim, cfg.norm_eps) ||
      !RmsNormDevice(backend, *h.rmsnorm_kernel, *h.kf, *(*k_norm)->device,
                     *h.kf, kv_heads, head_dim, cfg.norm_eps) ||
      !MropeDevice(backend, *h.mrope_kernel, *h.q, *h.pos, 1, heads, head_dim,
                   rope_dim, sec_t, sec_h, sec_w, cfg.attention.rope_theta) ||
      !MropeDevice(backend, *h.mrope_kernel, *h.kf, *h.pos, 1, kv_heads,
                   head_dim, rope_dim, sec_t, sec_h, sec_w,
                   cfg.attention.rope_theta)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (auto appended =
          AppendKv(backend, h.cast_kernel.get(), h.quant_kernel.get(),
                   h.kv_scratch.get(), h.kv_scratch.get(),
                   h.scale_scratch.get(), kv, *h.kf, *h.vf, kv_dim);
      !appended) {
    return std::unexpected(appended.error());
  }
  const bool quantized = kv.type == KvCacheType::Q8 ||
                         kv.type == KvCacheType::Q4 ||
                         kv.type == KvCacheType::FP8;
  // The single-token path splits the key range (flash decoding): the key
  // range is one serial 128-tile dependency chain per workgroup, so a long
  // context is latency-bound. Batches (prefill, verify) keep the tiled
  // kernel, which is faster at their row counts.
  const std::size_t split = detail::SplitFor(1, kv.rows);
  bool attention_ok = false;
  if (split > 0) {
    auto ready = EnsureSplitReady(backend, cfg, h, kv.type);
    if (!ready) {
      return std::unexpected(ready.error());
    }
    attention_ok =
        quantized
            ? detail::AttentionQuantSplitDevice(
                  backend, *h.attention_split_kernel,
                  *h.attention_combine_kernel, *h.q, *kv.k, *kv.v, *kv.k_scale,
                  *kv.v_scale, *h.attn, *h.split_acc, *h.split_max,
                  *h.split_sum, kv.rows, heads, kv_heads, head_dim, pos, 0, 1,
                  split)
                  .has_value()
            : detail::AttentionSplitDevice(
                  backend, *h.attention_split_kernel,
                  *h.attention_combine_kernel, *h.q, *kv.k, *kv.v, *h.attn,
                  *h.split_acc, *h.split_max, *h.split_sum, kv.rows, heads,
                  kv_heads, head_dim, pos, 0, 1, split,
                  kv.type == KvCacheType::F16)
                  .has_value();
  } else {
    attention_ok =
        quantized
            ? detail::AttentionQuantDevice(backend, *h.attention_kernel, *h.q,
                                           *kv.k, *kv.v, *kv.k_scale,
                                           *kv.v_scale, *h.attn, kv.rows, heads,
                                           kv_heads, head_dim, pos, 0, 1)
                  .has_value()
            : AttentionDevice(backend, *h.attention_kernel, *h.q, *kv.k, *kv.v,
                              *h.attn, kv.rows, heads, kv_heads, head_dim, pos,
                              0, 1, kv.type == KvCacheType::F16)
                  .has_value();
  }
  if (!attention_ok ||
      !SigmoidGateDevice(backend, *h.sigmoid_gate_kernel, *h.attn, *h.gate,
                         *h.attn, heads * head_dim) ||
      !ProjectBatch(backend, h, (*wo)->manifest.dtype, *h.attn, *(*wo)->device,
                    *h.proj, 1, hidden, heads * head_dim) ||
      !AddDevice(backend, *h.add_kernel, *h.x, *h.proj, *h.x, hidden)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return RunFfn(backend, model, cfg, h, layer);
}

std::expected<void, StatusCode> EnsureHybridReady(
    Backend& backend, const TransformerConfig& cfg, DecodeCache& cache,
    const LinearGeometry& g) {
  Qwen35State& h = State(cache);
  h.profile = cache.profile;
  if (!h.ready) {
    auto load = [&backend](std::unique_ptr<Kernel>& slot,
                           std::string_view name) -> bool {
      auto kernel = backend.LoadKernel(name, {});
      if (!kernel) {
        return false;
      }
      slot = std::move(*kernel);
      return true;
    };
    if (!load(h.rmsnorm_kernel, "rmsnorm") || !load(h.add_kernel, "add") ||
        !load(h.silu_mul_kernel, "silu_mul") ||
        !load(h.sigmoid_gate_kernel, "sigmoid_gate") ||
        !load(h.qgate_split_kernel, "qgate_split") ||
        !load(h.mrope_kernel, "mrope") ||
        !load(h.attention_kernel,
              cache.kv_type == KvCacheType::Q8    ? "attention_q8"
              : cache.kv_type == KvCacheType::Q4  ? "attention_q4"
              : cache.kv_type == KvCacheType::FP8 ? "attention_fp8"
                                                  : "attention") ||
        !load(h.repeat_heads_kernel, "repeat_heads") ||
        !load(h.l2norm_kernel, "l2norm") ||
        !load(h.ssm_gate_kernel, "ssm_gate") ||
        !load(h.delta_step_heads_kernel, "delta_step_heads") ||
        !load(h.rmsnorm_gated_kernel, "rmsnorm_gated") ||
        !load(h.conv1d_state_kernel, "conv1d_state")) {
      return std::unexpected(StatusCode::DeviceError);
    }
    auto scratch = InitScratch(backend, cfg, h, g);
    if (!scratch) {
      return std::unexpected(scratch.error());
    }
    const std::size_t cache_kv_dim =
        cfg.attention.kv_heads * cfg.attention.head_dim;
    if (cache.kv_type == KvCacheType::F16) {
      auto cast = backend.LoadKernel("cast_f32_f16", {});
      if (!cast) {
        return std::unexpected(cast.error());
      }
      h.cast_kernel = std::move(*cast);
      auto kv_scratch = backend.AllocateBuffer(cache_kv_dim * 2,
                                               MemoryKind::Device);
      if (!kv_scratch) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      h.kv_scratch = std::move(*kv_scratch);
    } else if (cache.kv_type == KvCacheType::Q8 ||
               cache.kv_type == KvCacheType::Q4 ||
               cache.kv_type == KvCacheType::FP8) {
      const bool q4 = cache.kv_type == KvCacheType::Q4;
      auto quant = backend.LoadKernel(
          cache.kv_type == KvCacheType::Q8    ? "quantize_q8"
          : cache.kv_type == KvCacheType::Q4  ? "quantize_q4"
                                              : "quantize_fp8_pack",
          {});
      if (!quant) {
        return std::unexpected(quant.error());
      }
      h.quant_kernel = std::move(*quant);
      auto kv_scratch = backend.AllocateBuffer(
          q4 ? cache_kv_dim / 2 : cache_kv_dim, MemoryKind::Device);
      auto scale = backend.AllocateBuffer(4, MemoryKind::Device);
      if (!kv_scratch || !scale) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      h.kv_scratch = std::move(*kv_scratch);
      h.scale_scratch = std::move(*scale);
    }
    if (cache.kv_type != KvCacheType::F32) {
      for (auto& kv : h.full) {
        kv.type = cache.kv_type;
      }
    }
    h.ready = true;
  }
  return {};
}

std::expected<void, StatusCode> GatherEmbeddingRows(
    Backend& backend, const Model& model, Qwen35State& h,
    std::span<const std::uint32_t> tokens, std::size_t hidden, Buffer& out) {
  auto embed = NeedWeightAny(model, "token_embd.weight");
  if (!embed) {
    return std::unexpected(embed.error());
  }
  const DType dtype = (*embed)->manifest.dtype;
  const std::size_t rows = tokens.size();
  const std::size_t vocab = (*embed)->manifest.shape.Numel() / hidden;
  for (const std::uint32_t token : tokens) {
    if (token >= vocab) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
  }
  const std::string_view name = detail::EmbeddingKernelName(dtype);
  // A Q4_K table is quantized in 256-element blocks, so the width must
  // align with the block.
  const bool aligned = dtype != DType::Q4K || hidden % 256 == 0;
  if (!name.empty() && aligned) {
    auto kernel = detail::CachedKernel(backend, h.embed_kernels,
                                       static_cast<int>(dtype), name);
    if (!kernel) {
      return std::unexpected(kernel.error());
    }
    if (h.embed_ids == nullptr || h.embed_ids->Size() < rows * 4) {
      auto allocated = backend.AllocateBuffer(rows * 4, MemoryKind::Device);
      if (!allocated) {
        return std::unexpected(allocated.error());
      }
      h.embed_ids = std::move(*allocated);
    }
    if (!backend.CopyH2D(*h.embed_ids, std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(tokens.data()), rows * 4))) {
      return std::unexpected(StatusCode::DeviceError);
    }
    return detail::GatherEmbeddingDevice(backend, **kernel, *h.embed_ids,
                                         *(*embed)->device, out, rows, hidden,
                                         vocab);
  }
  // Host fallback: dequantize each row and upload.
  std::vector<float> staged(rows * hidden);
  std::vector<float> row(hidden);
  for (std::size_t t = 0; t < rows; ++t) {
    auto gathered =
        GatherEmbedding(backend, **embed, tokens[t], hidden, row);
    if (!gathered) {
      return std::unexpected(gathered.error());
    }
    std::copy(row.begin(), row.end(), staged.begin() + t * hidden);
  }
  return UploadF32(backend, out, staged);
}

std::expected<void, StatusCode> Qwen35Architecture::Forward(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out,
    const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture, const Buffer* embedding) const {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const TransformerConfig& cfg = *config;
  if (!cfg.hybrid || cfg.full_attention_interval == 0 ||
      cfg.rope_sections.size() != 3) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (token >= cfg.vocab_size) {
    return std::unexpected(StatusCode::InvalidArgument);
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
  Qwen35State& h = State(cache);
  const std::size_t hidden = cfg.hidden_dim;
  if (embedding != nullptr) {
    // A precomputed embedding (an image token) replaces the gathered row.
    if (embedding->Size() < hidden * 4 ||
        !backend.CopyD2D(*embedding, 0, *h.x, 0, hidden * 4)) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
  } else {
    const std::uint32_t id = token;
    auto gathered = GatherEmbeddingRows(
        backend, model, h, std::span<const std::uint32_t>(&id, 1), hidden,
        *h.x);
    if (!gathered) {
      return std::unexpected(gathered.error());
    }
  }
  const std::uint64_t p = h.position;
  const std::uint64_t triples[3] = {p, p, p};
  if (!backend.CopyH2D(*h.pos, std::span<const std::byte>(
                                   reinterpret_cast<const std::byte*>(triples),
                                   sizeof(triples)))) {
    return std::unexpected(StatusCode::DeviceError);
  }
  // Optionally copy the residual-stream hidden after selected layers, for
  // the DFlash2 draft's target hidden states.
  const auto capture_layer = [&](std::size_t layer) -> StatusCode {
    if (capture_layers == nullptr || capture == nullptr) {
      return StatusCode::Ok;
    }
    for (std::size_t i = 0; i < capture_layers->size(); ++i) {
      if ((*capture_layers)[i] == layer) {
        if (!backend.CopyD2D(*h.x, 0, *(*capture)[i], 0, hidden * 4)) {
          return StatusCode::DeviceError;
        }
      }
    }
    return StatusCode::Ok;
  };
  core::Profile* const deep_profile =
      h.profile != nullptr && h.profile->Deep() ? h.profile : nullptr;
  for (std::size_t l = 0; l < cfg.layers; ++l) {
    if (cfg.IsFullAttentionLayer(l)) {
      core::PhaseScope layer_scope(deep_profile, core::Phase::kFullLayer);
      if (auto block = RunFullBlock(backend, model, cfg, h, l, p, h.full[l]);
          !block) {
        return std::unexpected(block.error());
      }
      if (deep_profile != nullptr) {
        backend.Synchronize();  // profile attribution only
      }
      if (capture_layer(l) != StatusCode::Ok) {
        return std::unexpected(StatusCode::DeviceError);
      }
      continue;
    }
    core::PhaseScope layer_scope(deep_profile, core::Phase::kLinearLayer);
    const std::string base = "blk." + std::to_string(l) + ".";
    auto norm = NeedWeight(model, base + "attn_norm.weight", DType::F32);
    if (!norm) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    if (!RmsNormDevice(backend, *h.rmsnorm_kernel, *h.x, *(*norm)->device,
                       *h.xn, 1, hidden, cfg.norm_eps)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    auto w_qkv = NeedWeightAny(model, base + "attn_qkv.weight");
    auto w_gate = NeedWeightAny(model, base + "attn_gate.weight");
    auto w_conv = NeedWeight(model, base + "ssm_conv1d.weight", DType::F32);
    auto w_alpha = NeedWeightAny(model, base + "ssm_alpha.weight");
    auto w_beta = NeedWeightAny(model, base + "ssm_beta.weight");
    auto w_a = NeedWeight(model, base + "ssm_a", DType::F32);
    auto w_dt = NeedWeight(model, base + "ssm_dt.bias", DType::F32);
    auto w_norm = NeedWeight(model, base + "ssm_norm.weight", DType::F32);
    auto w_out = NeedWeightAny(model, base + "ssm_out.weight");
    if (!w_qkv || !w_gate || !w_conv || !w_alpha || !w_beta || !w_a || !w_dt ||
        !w_norm || !w_out) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    if (!ProjectBatch(backend, h, (*w_qkv)->manifest.dtype, *h.xn,
                      *(*w_qkv)->device, *h.qkv, 1, g.conv_dim, hidden) ||
        !ProjectBatch(backend, h, (*w_gate)->manifest.dtype, *h.xn,
                      *(*w_gate)->device, *h.z, 1, g.value_dim, hidden) ||
        !ProjectBatch(backend, h, (*w_beta)->manifest.dtype, *h.xn,
                      *(*w_beta)->device, *h.beta_raw, 1, g.num_v_heads,
                      hidden) ||
        !ProjectBatch(backend, h, (*w_alpha)->manifest.dtype, *h.xn,
                      *(*w_alpha)->device, *h.alpha_raw, 1, g.num_v_heads,
                      hidden) ||
        !SsmGateDevice(backend, *h.ssm_gate_kernel, *(*w_a)->device,
                       *(*w_dt)->device, *h.alpha_raw, *h.beta_raw, *h.alpha,
                       *h.beta, g.num_v_heads)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    // The causal conv, SiLU and q/k/v split run on the device; the
    // history lives in linear[l].conv_hist and the kernel shifts it.
    if (!Conv1dStateDevice(backend, *h.conv1d_state_kernel, *h.qkv,
                           *(*w_conv)->device, *h.linear[l].conv_hist, *h.q_l,
                           *h.k_l, *h.v_l, g.conv_dim, g.width, g.key_dim) ||
        !L2NormDevice(backend, *h.l2norm_kernel, *h.q_l, *h.q_l, g.num_k_heads,
                      g.head_k_dim, 1e-6f,
                      1.0 / std::sqrt(static_cast<double>(g.head_k_dim))) ||
        !L2NormDevice(backend, *h.l2norm_kernel, *h.k_l, *h.k_l, g.num_k_heads,
                      g.head_k_dim, 1e-6f) ||
        !RepeatHeadsDevice(backend, *h.repeat_heads_kernel, *h.q_l, *h.q_exp,
                           g.num_v_heads, g.head_k_dim, g.factor) ||
        !RepeatHeadsDevice(backend, *h.repeat_heads_kernel, *h.k_l, *h.k_exp,
                           g.num_v_heads, g.head_k_dim, g.factor) ||
        !DeltaStepHeadsDevice(backend, *h.delta_step_heads_kernel,
                              *h.linear[l].state, *h.k_exp, *h.v_l, *h.q_exp,
                              *h.core, *h.alpha, *h.beta, g.num_v_heads,
                              g.head_k_dim, g.head_v_dim) ||
        !RmsNormGatedDevice(backend, *h.rmsnorm_gated_kernel, *h.core,
                            *(*w_norm)->device, *h.z, *h.out, g.num_v_heads,
                            g.head_v_dim, cfg.norm_eps) ||
        !ProjectBatch(backend, h, (*w_out)->manifest.dtype, *h.out,
                      *(*w_out)->device, *h.proj, 1, hidden, g.value_dim) ||
        !AddDevice(backend, *h.add_kernel, *h.x, *h.proj, *h.x, hidden)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (auto ffn = RunFfn(backend, model, cfg, h, l); !ffn) {
      return std::unexpected(ffn.error());
    }
    if (capture_layer(l) != StatusCode::Ok) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  if (hidden_out != nullptr) {
    auto snapshot = detail::DownloadF32(backend, *h.x);
    if (!snapshot) {
      return std::unexpected(snapshot.error());
    }
    *hidden_out = std::move(*snapshot);
  }
  ++h.position;
  return {};
}

// The output head is separate so a prompt prefill can run the block
// forward for all but the last token without the vocab-sized projection.
std::expected<std::vector<float>, StatusCode> Qwen35Architecture::Logits(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out,
    const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture, const Buffer* embedding) const {
  core::Profile* profile = cache.profile;
  {
    core::PhaseScope scope(profile, core::Phase::kForward);
    auto forward = Forward(backend, model, cache, token, hidden_out,
                           capture_layers, capture, embedding);
    if (!forward) {
      return std::unexpected(forward.error());
    }
    // Profile attribution only: the download below would sync anyway.
    if (profile != nullptr) {
      backend.Synchronize();
    }
  }
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const TransformerConfig& cfg = *config;
  Qwen35State& h = State(cache);
  {
    core::PhaseScope scope(profile, core::Phase::kHead);
    auto head = RunOutputHead(backend, model, cfg, h);
    if (!head) {
      return std::unexpected(head.error());
    }
    if (profile != nullptr) {
      backend.Synchronize();
    }
  }
  core::PhaseScope scope(profile, core::Phase::kDownload);
  backend.Synchronize();
  return detail::DownloadF32(backend, *h.logits);
}

std::expected<std::uint32_t, StatusCode> Qwen35Architecture::GreedyToken(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out,
    const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture, const Buffer* embedding) const {
  core::Profile* profile = cache.profile;
  {
    core::PhaseScope scope(profile, core::Phase::kForward);
    auto forward = Forward(backend, model, cache, token, hidden_out,
                           capture_layers, capture, embedding);
    if (!forward) {
      return std::unexpected(forward.error());
    }
    if (profile != nullptr) {
      backend.Synchronize();
    }
  }
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const TransformerConfig& cfg = *config;
  Qwen35State& h = State(cache);
  {
    core::PhaseScope scope(profile, core::Phase::kHead);
    auto head = RunOutputHead(backend, model, cfg, h);
    if (!head) {
      return std::unexpected(head.error());
    }
    if (profile != nullptr) {
      backend.Synchronize();
    }
  }
  // Top-1 on the device: read back one id instead of the whole vocabulary.
  core::PhaseScope scope(profile, core::Phase::kDownload);
  auto best = DeviceRowArgMax(backend, h, *h.logits, 1, cfg.vocab_size);
  if (!best) {
    return std::unexpected(best.error());
  }
  return (*best)[0];
}

std::string_view Qwen35Architecture::Name() const { return "qwen35"; }

}  // namespace tessera::models::qwen3_5
