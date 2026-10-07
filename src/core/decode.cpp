#include "core/decode.hpp"
#include "core/decode_internal.hpp"

#include <cmath>
#include <cstring>
#include <string>

namespace tessera::core {

namespace {

using detail::Attend;
using detail::DownloadF32;
using detail::DownloadF32Cached;
using detail::GatherEmbedding;
using detail::NeedWeight;
using detail::NeedWeightAny;
using detail::Project;
using detail::RmsNormInto;
using detail::SiluMulInto;

// RoPE over an uploaded row buffer, in place on the device.
std::expected<void, StatusCode> RopeRows(
    Backend& backend, const Kernel& rope, Buffer& io, std::size_t rows,
    std::size_t heads, std::size_t head_dim, std::size_t rope_dim,
    std::uint64_t pos_base, double theta) {
  const float theta_f = static_cast<float>(theta);
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(theta_f));
  std::memcpy(&bits, &theta_f, sizeof(bits));
  KernelLaunch launch;
  launch.grid_x =
      static_cast<std::uint32_t>((rows * heads * (rope_dim / 2) + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&io};
  launch.scalars = {rows, heads, head_dim,
                    rope_dim, pos_base, bits};
  auto ran = backend.LaunchKernel(rope, launch);
  if (!ran) {
    return std::unexpected(ran.error());
  }
  backend.Synchronize();
  return {};
}

}  // namespace

std::expected<std::uint32_t, StatusCode> DecodeStep(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token) {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const TransformerConfig& cfg = *config;
  // Hybrid attention/SSM blocks run their own walk.
  if (cfg.hybrid) {
    if (!cache.hybrid) {
      cache.hybrid = std::make_unique<HybridDecodeCache>();
    }
    return HybridDecodeStep(backend, model, cache, token);
  }
  const std::size_t heads = cfg.attention.heads;
  const std::size_t kv_heads = cfg.attention.kv_heads;
  const std::size_t head_dim = cfg.attention.head_dim;
  const std::size_t hidden = cfg.hidden_dim;
  const std::size_t kv_dim = kv_heads * head_dim;
  if (token >= cfg.vocab_size) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (cache.layers.size() != cfg.layers) {
    cache.layers.assign(cfg.layers, DecodeCache::LayerCache{});
    auto gemm = backend.LoadKernel("gemm_q4k", {});
    auto rope = backend.LoadKernel("rope", {});
    auto attention = backend.LoadKernel("attention", {});
    if (!gemm || !rope || !attention) {
      return std::unexpected(StatusCode::DeviceError);
    }
    cache.gemm_kernel = std::move(*gemm);
    cache.rope_kernel = std::move(*rope);
    cache.attention_kernel = std::move(*attention);
  }
  auto embed = NeedWeightAny(model, "token_embd.weight");
  if (!embed) {
    return std::unexpected(embed.error());
  }
  std::vector<float> x(hidden);
  if (auto gathered = GatherEmbedding(backend, **embed, token, hidden, x);
      !gathered) {
    return std::unexpected(gathered.error());
  }
  std::vector<float> work(hidden);
  std::vector<float> mlp_work(cfg.ffn_dim);
  for (std::size_t l = 0; l < cfg.layers; ++l) {
    const std::string base = "blk." + std::to_string(l) + ".";
    auto norm = NeedWeight(model, base + "attn_norm.weight", DType::F32);
    auto wq = NeedWeight(model, base + "attn_q.weight", DType::Q4K);
    auto wk = NeedWeight(model, base + "attn_k.weight", DType::Q4K);
    auto wv = NeedWeight(model, base + "attn_v.weight", DType::Q4K);
    auto wo = NeedWeight(model, base + "attn_output.weight", DType::Q4K);
    auto mlp_norm = NeedWeight(model, base + "ffn_norm.weight", DType::F32);
    auto gate = NeedWeight(model, base + "ffn_gate.weight", DType::Q4K);
    auto up = NeedWeight(model, base + "ffn_up.weight", DType::Q4K);
    auto down = NeedWeight(model, base + "ffn_down.weight", DType::Q4K);
    if (!norm || !wq || !wk || !wv || !wo || !mlp_norm || !gate || !up ||
        !down) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    auto norm_w = DownloadF32Cached(backend, cache.host_weights,
                                 base + "attn_norm.weight",
                                 *(*norm)->device);
    if (!norm_w) {
      return std::unexpected(norm_w.error());
    }
    RmsNormInto(x, (*norm_w)->data(), cfg.norm_eps, work);
    auto q = Project(backend, *cache.gemm_kernel, work, *(*wq)->device,
                     heads * head_dim);
    auto k_row = Project(backend, *cache.gemm_kernel, work, *(*wk)->device,
                         kv_dim);
    auto v_row = Project(backend, *cache.gemm_kernel, work, *(*wv)->device,
                         kv_dim);
    if (!q) {
      return std::unexpected(q.error());
    }
    if (!k_row) {
      return std::unexpected(k_row.error());
    }
    if (!v_row) {
      return std::unexpected(v_row.error());
    }
    DecodeCache::LayerCache& layer = cache.layers[l];
    const std::size_t pos = layer.k.size() / kv_dim;
    auto q_buf = backend.AllocateBuffer(q->size() * 4, MemoryKind::Device);
    auto k_buf =
        backend.AllocateBuffer(k_row->size() * 4, MemoryKind::Device);
    if (!q_buf || !k_buf) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    auto up_q = backend.CopyH2D(
        **q_buf, std::span<const std::byte>(
                     reinterpret_cast<const std::byte*>(q->data()),
                     q->size() * 4));
    auto up_k = backend.CopyH2D(
        **k_buf, std::span<const std::byte>(
                     reinterpret_cast<const std::byte*>(k_row->data()),
                     k_row->size() * 4));
    if (!up_q || !up_k) {
      return std::unexpected(StatusCode::DeviceError);
    }
    auto roped_q = RopeRows(backend, *cache.rope_kernel, **q_buf, 1, heads,
                            head_dim, cfg.attention.rope_dim, pos,
                            cfg.attention.rope_theta);
    auto roped_k = RopeRows(backend, *cache.rope_kernel, **k_buf, 1,
                            kv_heads, head_dim, cfg.attention.rope_dim, pos,
                            cfg.attention.rope_theta);
    if (!roped_q || !roped_k) {
      return std::unexpected(StatusCode::DeviceError);
    }
    std::vector<std::byte> q_raw(q->size() * 4);
    std::vector<std::byte> k_raw(k_row->size() * 4);
    if (!backend.CopyD2H(**q_buf, q_raw.data(), q_raw.size()) ||
        !backend.CopyD2H(**k_buf, k_raw.data(), k_raw.size())) {
      return std::unexpected(StatusCode::DeviceError);
    }
    std::memcpy(q->data(), q_raw.data(), q_raw.size());
    layer.k.resize(layer.k.size() + k_row->size());
    std::memcpy(layer.k.data() + pos * kv_dim, k_raw.data(), k_raw.size());
    layer.v.insert(layer.v.end(), v_row->begin(), v_row->end());
    auto attn = Attend(backend, *cache.attention_kernel, *q, layer.k,
                       layer.v, heads, kv_heads, head_dim, pos);
    if (!attn) {
      return std::unexpected(attn.error());
    }
    auto proj_out = Project(backend, *cache.gemm_kernel, *attn,
                            *(*wo)->device, hidden);
    if (!proj_out) {
      return std::unexpected(proj_out.error());
    }
    for (std::size_t i = 0; i < hidden; ++i) {
      x[i] += (*proj_out)[i];
    }
    auto mlp_norm_w = DownloadF32Cached(backend, cache.host_weights,
                                    base + "ffn_norm.weight",
                                    *(*mlp_norm)->device);
    if (!mlp_norm_w) {
      return std::unexpected(mlp_norm_w.error());
    }
    RmsNormInto(x, (*mlp_norm_w)->data(), cfg.norm_eps, work);
    auto g = Project(backend, *cache.gemm_kernel, work, *(*gate)->device,
                     cfg.ffn_dim);
    auto u = Project(backend, *cache.gemm_kernel, work, *(*up)->device,
                     cfg.ffn_dim);
    if (!g) {
      return std::unexpected(g.error());
    }
    if (!u) {
      return std::unexpected(u.error());
    }
    SiluMulInto(*g, *u, mlp_work);
    auto d = Project(backend, *cache.gemm_kernel, mlp_work,
                     *(*down)->device, hidden);
    if (!d) {
      return std::unexpected(d.error());
    }
    for (std::size_t i = 0; i < hidden; ++i) {
      x[i] += (*d)[i];
    }
  }
  auto out_norm = NeedWeight(model, "output_norm.weight", DType::F32);
  auto output = NeedWeight(model, "output.weight", DType::Q4K);
  if (!out_norm || !output) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto out_norm_w = DownloadF32Cached(backend, cache.host_weights,
                                    "output_norm.weight",
                                    *(*out_norm)->device);
  if (!out_norm_w) {
    return std::unexpected(out_norm_w.error());
  }
  RmsNormInto(x, (*out_norm_w)->data(), cfg.norm_eps, work);
  auto logits =
      Project(backend, *cache.gemm_kernel, work, *(*output)->device,
              cfg.vocab_size);
  if (!logits) {
    return std::unexpected(logits.error());
  }
  std::uint32_t best = 0;
  for (std::uint32_t i = 1; i < logits->size(); ++i) {
    if ((*logits)[i] > (*logits)[best]) {
      best = i;
    }
  }
  return best;
}

}  // namespace tessera::core
