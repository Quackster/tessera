#include "core/decode.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/hybrid_ops.hpp"
#include "core/numerics/norm.hpp"

namespace tessera::core {

namespace {

using detail::Attend;
using detail::DeriveGeometry;
using detail::DownloadF32;
using detail::DownloadF32Cached;
using detail::GatherEmbedding;
using detail::GemmFor;
using detail::LinearGeometry;
using detail::NeedWeight;
using detail::NeedWeightAny;
using detail::NormHeads;
using detail::Project;
using detail::RmsNormInto;
using detail::RunConv1d;
using detail::RunDeltaStep;
using detail::RunMrope;
using detail::SiluMulInto;
using detail::UploadF32;

// The gated full-attention token mixer; `work` is the pre-normed input.
std::expected<std::vector<float>, StatusCode> FullAttentionLayer(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    DecodeCache& cache, HybridDecodeCache& hc, std::size_t l,
    const std::vector<float>& work) {
  const std::size_t heads = cfg.attention.heads;
  const std::size_t kv_heads = cfg.attention.kv_heads;
  const std::size_t head_dim = cfg.attention.head_dim;
  const std::size_t hidden = cfg.hidden_dim;
  const std::size_t kv_dim = kv_heads * head_dim;
  const std::size_t rope_dim = cfg.attention.rope_dim;
  const std::size_t sec_t = static_cast<std::size_t>(cfg.rope_sections[0]);
  const std::size_t sec_h = static_cast<std::size_t>(cfg.rope_sections[1]);
  const std::size_t sec_w = static_cast<std::size_t>(cfg.rope_sections[2]);
  const std::string base = "blk." + std::to_string(l) + ".";
  auto wq = NeedWeightAny(model, base + "attn_q.weight");
  auto wk = NeedWeightAny(model, base + "attn_k.weight");
  auto wv = NeedWeightAny(model, base + "attn_v.weight");
  auto wo = NeedWeightAny(model, base + "attn_output.weight");
  auto q_norm = NeedWeight(model, base + "attn_q_norm.weight", DType::F32);
  auto k_norm = NeedWeight(model, base + "attn_k_norm.weight", DType::F32);
  if (!wq || !wk || !wv || !wo || !q_norm || !k_norm) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto gemm_q = GemmFor(backend, hc, (*wq)->manifest.dtype);
  auto gemm_k = GemmFor(backend, hc, (*wk)->manifest.dtype);
  auto gemm_v = GemmFor(backend, hc, (*wv)->manifest.dtype);
  auto gemm_o = GemmFor(backend, hc, (*wo)->manifest.dtype);
  if (!gemm_q || !gemm_k || !gemm_v || !gemm_o) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  const std::size_t q_dim = heads * head_dim;
  auto qg = Project(backend, *(*gemm_q), work, *(*wq)->device, q_dim * 2);
  if (!qg) {
    return std::unexpected(qg.error());
  }
  std::vector<float> q(q_dim);
  std::vector<float> gate(q_dim);
  for (std::size_t h = 0; h < heads; ++h) {
    std::memcpy(q.data() + h * head_dim, qg->data() + h * 2 * head_dim,
                head_dim * 4);
    std::memcpy(gate.data() + h * head_dim,
                qg->data() + h * 2 * head_dim + head_dim, head_dim * 4);
  }
  auto k_row = Project(backend, *(*gemm_k), work, *(*wk)->device, kv_dim);
  auto v_row = Project(backend, *(*gemm_v), work, *(*wv)->device, kv_dim);
  if (!k_row || !v_row) {
    return std::unexpected(StatusCode::DeviceError);
  }
  auto q_norm_w = DownloadF32Cached(backend, hc.host_weights,
                                    base + "attn_q_norm.weight",
                                    *(*q_norm)->device);
  auto k_norm_w = DownloadF32Cached(backend, hc.host_weights,
                                    base + "attn_k_norm.weight",
                                    *(*k_norm)->device);
  if (!q_norm_w || !k_norm_w) {
    return std::unexpected(StatusCode::DeviceError);
  }
  NormHeads(q, heads, head_dim, **q_norm_w, cfg.norm_eps);
  NormHeads(*k_row, kv_heads, head_dim, **k_norm_w, cfg.norm_eps);
  HybridDecodeCache::FullLayer& full = hc.full[l];
  const std::size_t pos = full.k.size() / kv_dim;
  const std::vector<std::uint64_t> triple = {pos, pos, pos};
  if (!RunMrope(backend, *hc.mrope_kernel, q, triple, 1, heads, head_dim,
                rope_dim, sec_t, sec_h, sec_w, cfg.attention.rope_theta) ||
      !RunMrope(backend, *hc.mrope_kernel, *k_row, triple, 1, kv_heads,
                head_dim, rope_dim, sec_t, sec_h, sec_w,
                cfg.attention.rope_theta)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  full.k.resize(full.k.size() + kv_dim);
  std::memcpy(full.k.data() + pos * kv_dim, k_row->data(), kv_dim * 4);
  full.v.insert(full.v.end(), v_row->begin(), v_row->end());
  auto attn = Attend(backend, *cache.attention_kernel, q, full.k, full.v,
                     heads, kv_heads, head_dim, pos);
  if (!attn) {
    return std::unexpected(attn.error());
  }
  std::vector<float> gated(q_dim);
  for (std::size_t i = 0; i < q_dim; ++i) {
    gated[i] = (*attn)[i] / (1.0f + std::exp(-gate[i]));
  }
  auto proj = Project(backend, *(*gemm_o), gated, *(*wo)->device, hidden);
  if (!proj) {
    return std::unexpected(proj.error());
  }
  return proj;
}

// The gated-delta linear-attention token mixer; `work` is the pre-normed
// input.
std::expected<std::vector<float>, StatusCode> LinearAttentionLayer(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    HybridDecodeCache& hc, std::size_t l, const std::vector<float>& work) {
  auto geometry = DeriveGeometry(cfg);
  if (!geometry) {
    return std::unexpected(geometry.error());
  }
  const LinearGeometry& g = *geometry;
  const std::size_t hidden = cfg.hidden_dim;
  const std::string base = "blk." + std::to_string(l) + ".";
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
  auto gemm_qkv = GemmFor(backend, hc, (*w_qkv)->manifest.dtype);
  auto gemm_gate = GemmFor(backend, hc, (*w_gate)->manifest.dtype);
  auto gemm_alpha = GemmFor(backend, hc, (*w_alpha)->manifest.dtype);
  auto gemm_beta = GemmFor(backend, hc, (*w_beta)->manifest.dtype);
  auto gemm_out = GemmFor(backend, hc, (*w_out)->manifest.dtype);
  if (!gemm_qkv || !gemm_gate || !gemm_alpha || !gemm_beta || !gemm_out) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  auto qkv = Project(backend, *(*gemm_qkv), work, *(*w_qkv)->device,
                     g.conv_dim);
  auto z = Project(backend, *(*gemm_gate), work, *(*w_gate)->device,
                   g.value_dim);
  auto beta_raw = Project(backend, *(*gemm_beta), work, *(*w_beta)->device,
                          g.num_v_heads);
  auto alpha_raw = Project(backend, *(*gemm_alpha), work, *(*w_alpha)->device,
                           g.num_v_heads);
  if (!qkv || !z || !beta_raw || !alpha_raw) {
    return std::unexpected(StatusCode::DeviceError);
  }
  auto a = DownloadF32Cached(backend, hc.host_weights, base + "ssm_a",
                             *(*w_a)->device);
  auto dt = DownloadF32Cached(backend, hc.host_weights, base + "ssm_dt.bias",
                              *(*w_dt)->device);
  if (!a || !dt) {
    return std::unexpected(StatusCode::DeviceError);
  }
  HybridDecodeCache::LinearLayer& st = hc.linear[l];
  // Causal conv1d over the last `width` qkv vectors (zero history), then
  // SiLU; take the newest output.
  std::vector<float> seq(g.conv_dim * g.width);
  for (std::size_t m = 0; m < g.width; ++m) {
    for (std::size_t c = 0; c < g.conv_dim; ++c) {
      seq[c * g.width + m] =
          m + 1 == g.width ? (*qkv)[c] : st.conv_hist[m * g.conv_dim + c];
    }
  }
  auto conv_y = RunConv1d(backend, *hc.conv_kernel, seq, *(*w_conv)->device,
                          g.conv_dim, g.width, g.width);
  if (!conv_y) {
    return std::unexpected(conv_y.error());
  }
  std::vector<float> mixed(g.conv_dim);
  for (std::size_t c = 0; c < g.conv_dim; ++c) {
    const float v = (*conv_y)[c * g.width + (g.width - 1)];
    mixed[c] = v / (1.0f + std::exp(-v));
  }
  // Update the conv history (drop oldest, append the newest qkv).
  if (g.width > 1) {
    for (std::size_t m = 0; m + 1 < g.width; ++m) {
      std::memcpy(st.conv_hist.data() + m * g.conv_dim,
                  st.conv_hist.data() + (m + 1) * g.conv_dim,
                  g.conv_dim * 4);
    }
    std::memcpy(st.conv_hist.data() + (g.width - 2) * g.conv_dim,
                qkv->data(), g.conv_dim * 4);
  }
  // Split into q, k (key_dim each) and v (value_dim), then L2-normalize
  // q and k per key head.
  std::vector<float> q(mixed.begin(), mixed.begin() + g.key_dim);
  std::vector<float> k(mixed.begin() + g.key_dim,
                       mixed.begin() + 2 * g.key_dim);
  std::vector<float> v(mixed.begin() + 2 * g.key_dim, mixed.end());
  std::vector<float> normed(g.key_dim);
  if (auto s = L2NormRef(std::span<const float>(q), std::span<float>(normed),
                         g.num_k_heads, g.head_k_dim, 1e-6f);
      !s) {
    return std::unexpected(s.error());
  }
  q = normed;
  if (auto s = L2NormRef(std::span<const float>(k), std::span<float>(normed),
                         g.num_k_heads, g.head_k_dim, 1e-6f);
      !s) {
    return std::unexpected(s.error());
  }
  k = normed;
  // Gated-delta scan per value head (q and k repeat to the value heads).
  std::vector<float> core(g.value_dim);
  for (std::size_t h = 0; h < g.num_v_heads; ++h) {
    const std::size_t ks = (h / g.factor) * g.head_k_dim;
    std::vector<float> q_h(q.begin() + ks, q.begin() + ks + g.head_k_dim);
    std::vector<float> k_h(k.begin() + ks, k.begin() + ks + g.head_k_dim);
    std::vector<float> v_h(v.begin() + h * g.head_v_dim,
                           v.begin() + (h + 1) * g.head_v_dim);
    const float raw = (*alpha_raw)[h] + (**dt)[h];
    const float softplus = raw > 20.0f ? raw : std::log1p(std::exp(raw));
    const float alph = std::exp(-std::exp((**a)[h]) * softplus);
    const float bet = 1.0f / (1.0f + std::exp(-(*beta_raw)[h]));
    std::vector<float> state_slice(
        st.state.begin() + h * g.head_k_dim * g.head_v_dim,
        st.state.begin() + (h + 1) * g.head_k_dim * g.head_v_dim);
    auto o = RunDeltaStep(backend, *hc.delta_kernel, state_slice, k_h, v_h, q_h,
                          g.head_k_dim, g.head_v_dim, alph, bet);
    if (!o) {
      return std::unexpected(o.error());
    }
    std::memcpy(st.state.data() + h * g.head_k_dim * g.head_v_dim,
                state_slice.data(), state_slice.size() * 4);
    std::memcpy(core.data() + h * g.head_v_dim, o->data(), g.head_v_dim * 4);
  }
  // Gated RMS norm over the value heads, gated by the z projection.
  std::vector<float> out(g.value_dim);
  auto norm_w = DownloadF32Cached(backend, hc.host_weights,
                                  base + "ssm_norm.weight",
                                  *(*w_norm)->device);
  if (!norm_w) {
    return std::unexpected(norm_w.error());
  }
  if (auto s = RmsNormGatedRef(std::span<const float>(core),
                               std::span<const float>(**norm_w),
                               std::span<const float>(*z), std::span<float>(out),
                               g.num_v_heads, g.head_v_dim, cfg.norm_eps);
      !s) {
    return std::unexpected(s.error());
  }
  auto proj = Project(backend, *(*gemm_out), out, *(*w_out)->device, hidden);
  if (!proj) {
    return std::unexpected(proj.error());
  }
  return proj;
}

// post_attention_norm + gated MLP + residual, in place on x.
std::expected<void, StatusCode> ApplyFfn(Backend& backend, const Model& model,
                                         const TransformerConfig& cfg,
                                         HybridDecodeCache& hc, std::size_t l,
                                         std::vector<float>& x) {
  const std::size_t hidden = cfg.hidden_dim;
  const std::string base = "blk." + std::to_string(l) + ".";
  auto post =
      NeedWeight(model, base + "post_attention_norm.weight", DType::F32);
  auto w_gate = NeedWeightAny(model, base + "ffn_gate.weight");
  auto w_up = NeedWeightAny(model, base + "ffn_up.weight");
  auto w_down = NeedWeightAny(model, base + "ffn_down.weight");
  if (!post || !w_gate || !w_up || !w_down) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto gemm_gate = GemmFor(backend, hc, (*w_gate)->manifest.dtype);
  auto gemm_up = GemmFor(backend, hc, (*w_up)->manifest.dtype);
  auto gemm_down = GemmFor(backend, hc, (*w_down)->manifest.dtype);
  if (!gemm_gate || !gemm_up || !gemm_down) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  auto post_w = DownloadF32Cached(backend, hc.host_weights,
                                  base + "post_attention_norm.weight",
                                  *(*post)->device);
  if (!post_w) {
    return std::unexpected(post_w.error());
  }
  std::vector<float> work(hidden);
  RmsNormInto(x, (*post_w)->data(), cfg.norm_eps, work);
  auto gate = Project(backend, *(*gemm_gate), work, *(*w_gate)->device,
                      cfg.ffn_dim);
  auto up =
      Project(backend, *(*gemm_up), work, *(*w_up)->device, cfg.ffn_dim);
  if (!gate || !up) {
    return std::unexpected(StatusCode::DeviceError);
  }
  std::vector<float> mlp(cfg.ffn_dim);
  SiluMulInto(*gate, *up, mlp);
  auto down = Project(backend, *(*gemm_down), mlp, *(*w_down)->device, hidden);
  if (!down) {
    return std::unexpected(down.error());
  }
  for (std::size_t i = 0; i < hidden; ++i) {
    x[i] += (*down)[i];
  }
  return {};
}

}  // namespace

std::expected<std::uint32_t, StatusCode> HybridDecodeStep(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token) {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const TransformerConfig& cfg = *config;
  if (!cfg.hybrid || cfg.full_attention_interval == 0 ||
      cfg.rope_sections.size() != 3) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const std::size_t hidden = cfg.hidden_dim;
  if (token >= cfg.vocab_size) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  HybridDecodeCache& hc = *cache.hybrid;
  if (!hc.ready) {
    hc.full.assign(cfg.layers, HybridDecodeCache::FullLayer{});
    hc.linear.assign(cfg.layers, HybridDecodeCache::LinearLayer{});
    auto mrope = backend.LoadKernel("mrope", {});
    auto conv = backend.LoadKernel("conv1d", {});
    auto delta = backend.LoadKernel("delta_step", {});
    if (!mrope || !conv || !delta) {
      return std::unexpected(StatusCode::DeviceError);
    }
    hc.mrope_kernel = std::move(*mrope);
    hc.conv_kernel = std::move(*conv);
    hc.delta_kernel = std::move(*delta);
    if (!cache.attention_kernel) {
      auto attention = backend.LoadKernel("attention", {});
      if (!attention) {
        return std::unexpected(attention.error());
      }
      cache.attention_kernel = std::move(*attention);
    }
    auto geometry = DeriveGeometry(cfg);
    if (!geometry) {
      return std::unexpected(geometry.error());
    }
    const LinearGeometry& g = *geometry;
    for (std::size_t l = 0; l < cfg.layers; ++l) {
      if (!cfg.IsFullAttentionLayer(l)) {
        hc.linear[l].conv_hist.assign(g.conv_dim * (g.width - 1), 0.0f);
        hc.linear[l].state.assign(
            g.num_v_heads * g.head_k_dim * g.head_v_dim, 0.0f);
      }
    }
    hc.ready = true;
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
  for (std::size_t l = 0; l < cfg.layers; ++l) {
    const std::string base = "blk." + std::to_string(l) + ".";
    auto norm = NeedWeight(model, base + "attn_norm.weight", DType::F32);
    if (!norm) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    auto norm_w = DownloadF32Cached(backend, hc.host_weights,
                                    base + "attn_norm.weight",
                                    *(*norm)->device);
    if (!norm_w) {
      return std::unexpected(norm_w.error());
    }
    RmsNormInto(x, (*norm_w)->data(), cfg.norm_eps, work);
    std::expected<std::vector<float>, StatusCode> mix =
        cfg.IsFullAttentionLayer(l)
            ? FullAttentionLayer(backend, model, cfg, cache, hc, l, work)
            : LinearAttentionLayer(backend, model, cfg, hc, l, work);
    if (!mix) {
      return std::unexpected(mix.error());
    }
    for (std::size_t i = 0; i < hidden; ++i) {
      x[i] += (*mix)[i];
    }
    auto ffn = ApplyFfn(backend, model, cfg, hc, l, x);
    if (!ffn) {
      return std::unexpected(ffn.error());
    }
  }
  auto out_norm = NeedWeight(model, "output_norm.weight", DType::F32);
  auto output = NeedWeightAny(model, "output.weight");
  if (!out_norm || !output) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto gemm_out = GemmFor(backend, hc, (*output)->manifest.dtype);
  if (!gemm_out) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  auto out_norm_w = DownloadF32Cached(backend, hc.host_weights,
                                      "output_norm.weight",
                                      *(*out_norm)->device);
  if (!out_norm_w) {
    return std::unexpected(out_norm_w.error());
  }
  RmsNormInto(x, (*out_norm_w)->data(), cfg.norm_eps, work);
  auto logits = Project(backend, *(*gemm_out), work, *(*output)->device,
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

