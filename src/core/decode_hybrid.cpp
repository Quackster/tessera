#include "core/decode.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "core/decode_internal.hpp"

namespace tessera::core {

namespace {

using detail::Attend;
using detail::DownloadF32;
using detail::NeedWeight;
using detail::Project;
using detail::RmsNormInto;
using detail::SiluMulInto;
using detail::UploadF32;

// Gemm kernel id for a projection dtype (the formats the GGUF target
// uses). Empty string when the dtype has no GEMM kernel.
std::string_view GemmKernelName(DType dtype) {
  switch (dtype) {
    case DType::Q4K: return "gemm_q4k";
    case DType::Q5K: return "gemm_q5k";
    case DType::Q6K: return "gemm_q6k";
    case DType::Q3K: return "gemm_q3k";
    case DType::Q80: return "gemm_q80";
    case DType::IQ4_NL: return "gemm_iq4nl";
    case DType::IQ4_XS: return "gemm_iq4xs";
    case DType::IQ3_S: return "gemm_iq3s";
    default: return {};
  }
}

// Load (once) and return the GEMM kernel for a projection dtype.
std::expected<Kernel*, StatusCode> GemmFor(Backend& backend,
                                           HybridDecodeCache& cache,
                                           DType dtype) {
  auto it = cache.gemms.find(static_cast<int>(dtype));
  if (it != cache.gemms.end()) {
    return it->second.get();
  }
  const std::string_view name = GemmKernelName(dtype);
  if (name.empty()) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  auto kernel = backend.LoadKernel(name, {});
  if (!kernel) {
    return std::unexpected(kernel.error());
  }
  Kernel* raw = kernel->get();
  cache.gemms.emplace(static_cast<int>(dtype), std::move(*kernel));
  return raw;
}

// In-place mRoPE over rows x heads x head_dim with per-row (t, h, w)
// triples, on the device.
std::expected<void, StatusCode> RunMrope(
    Backend& backend, const Kernel& mrope, std::vector<float>& io,
    const std::vector<std::uint64_t>& pos, std::size_t rows, std::size_t heads,
    std::size_t head_dim, std::size_t rope_dim, std::size_t sec_t,
    std::size_t sec_h, std::size_t sec_w, double theta) {
  auto io_buf = backend.AllocateBuffer(io.size() * 4, MemoryKind::Device);
  auto pos_buf = backend.AllocateBuffer(pos.size() * 8, MemoryKind::Device);
  if (!io_buf || !pos_buf) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  if (!UploadF32(backend, **io_buf, io) ||
      !backend.CopyH2D(**pos_buf,
                       std::span<const std::byte>(
                           reinterpret_cast<const std::byte*>(pos.data()),
                           pos.size() * 8))) {
    return std::unexpected(StatusCode::DeviceError);
  }
  const float theta_f = static_cast<float>(theta);
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(theta_f));
  std::memcpy(&bits, &theta_f, sizeof(bits));
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>(
      (rows * heads * (rope_dim / 2) + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {(*io_buf).get(), (*pos_buf).get()};
  launch.scalars = {rows, heads, head_dim, rope_dim, bits,
                    sec_t, sec_h, sec_w};
  auto ran = backend.LaunchKernel(mrope, launch);
  if (!ran) {
    return std::unexpected(ran.error());
  }
  backend.Synchronize();
  auto down = DownloadF32(backend, **io_buf);
  if (!down) {
    return std::unexpected(down.error());
  }
  io = std::move(*down);
  return {};
}

// QK-Norm: RMSNorm over head_dim applied per head (host, ascending).
void NormHeads(std::vector<float>& x, std::size_t heads, std::size_t head_dim,
               const std::vector<float>& w, double eps) {
  std::vector<float> slice(head_dim);
  std::vector<float> out(head_dim);
  for (std::size_t h = 0; h < heads; ++h) {
    std::memcpy(slice.data(), x.data() + h * head_dim, head_dim * 4);
    RmsNormInto(slice, w.data(), eps, out);
    std::memcpy(x.data() + h * head_dim, out.data(), head_dim * 4);
  }
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
  const std::size_t heads = cfg.attention.heads;
  const std::size_t kv_heads = cfg.attention.kv_heads;
  const std::size_t head_dim = cfg.attention.head_dim;
  const std::size_t hidden = cfg.hidden_dim;
  const std::size_t kv_dim = kv_heads * head_dim;
  const std::size_t rope_dim = cfg.attention.rope_dim;
  const std::size_t sec_t = static_cast<std::size_t>(cfg.rope_sections[0]);
  const std::size_t sec_h = static_cast<std::size_t>(cfg.rope_sections[1]);
  const std::size_t sec_w = static_cast<std::size_t>(cfg.rope_sections[2]);
  if (token >= cfg.vocab_size) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  // The recurrent (linear-attention) path is not wired yet; reject
  // models that need it up front rather than mid-walk.
  for (std::size_t l = 0; l < cfg.layers; ++l) {
    if (!cfg.IsFullAttentionLayer(l)) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
  }
  HybridDecodeCache& hc = *cache.hybrid;
  if (!hc.ready) {
    hc.full.assign(cfg.layers, HybridDecodeCache::FullLayer{});
    auto mrope = backend.LoadKernel("mrope", {});
    if (!mrope) {
      return std::unexpected(mrope.error());
    }
    hc.mrope_kernel = std::move(*mrope);
    if (!cache.attention_kernel) {
      auto attention = backend.LoadKernel("attention", {});
      if (!attention) {
        return std::unexpected(attention.error());
      }
      cache.attention_kernel = std::move(*attention);
    }
    hc.ready = true;
  }
  auto embed = NeedWeight(model, "token_embd.weight", DType::F32);
  if (!embed) {
    return std::unexpected(embed.error());
  }
  std::vector<float> x(hidden);
  auto row = backend.CopyD2HAt(
      *(*embed)->device, static_cast<std::size_t>(token) * hidden * 4,
      reinterpret_cast<std::byte*>(x.data()), hidden * 4);
  if (!row) {
    return std::unexpected(row.error());
  }
  std::vector<float> work(hidden);
  std::vector<float> mlp_work(cfg.ffn_dim);
  for (std::size_t l = 0; l < cfg.layers; ++l) {
    if (!cfg.IsFullAttentionLayer(l) || l >= hc.full.size()) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    const std::string base = "blk." + std::to_string(l) + ".";
    auto norm = NeedWeight(model, base + "attn_norm.weight", DType::F32);
    auto wq = NeedWeight(model, base + "attn_q.weight", DType::Q4K);
    auto wk = NeedWeight(model, base + "attn_k.weight", DType::Q4K);
    auto wv = NeedWeight(model, base + "attn_v.weight", DType::Q4K);
    auto wo = NeedWeight(model, base + "attn_output.weight", DType::Q4K);
    auto q_norm = NeedWeight(model, base + "attn_q_norm.weight", DType::F32);
    auto k_norm = NeedWeight(model, base + "attn_k_norm.weight", DType::F32);
    auto post_norm =
        NeedWeight(model, base + "post_attention_norm.weight", DType::F32);
    auto ffn_gate = NeedWeight(model, base + "ffn_gate.weight", DType::Q4K);
    auto ffn_up = NeedWeight(model, base + "ffn_up.weight", DType::Q4K);
    auto ffn_down = NeedWeight(model, base + "ffn_down.weight", DType::Q4K);
    if (!norm || !wq || !wk || !wv || !wo || !q_norm || !k_norm ||
        !post_norm || !ffn_gate || !ffn_up || !ffn_down) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    // Projections may use any of the supported quant formats; the
    // weight dtype selects the kernel.
    auto gemm_q = GemmFor(backend, hc, (*wq)->manifest.dtype);
    auto gemm_k = GemmFor(backend, hc, (*wk)->manifest.dtype);
    auto gemm_v = GemmFor(backend, hc, (*wv)->manifest.dtype);
    auto gemm_o = GemmFor(backend, hc, (*wo)->manifest.dtype);
    auto gemm_gate = GemmFor(backend, hc, (*ffn_gate)->manifest.dtype);
    auto gemm_up = GemmFor(backend, hc, (*ffn_up)->manifest.dtype);
    auto gemm_down = GemmFor(backend, hc, (*ffn_down)->manifest.dtype);
    if (!gemm_q || !gemm_k || !gemm_v || !gemm_o || !gemm_gate || !gemm_up ||
        !gemm_down) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    // Attention pre-norm.
    auto norm_w = DownloadF32(backend, *(*norm)->device);
    if (!norm_w) {
      return std::unexpected(norm_w.error());
    }
    RmsNormInto(x, norm_w->data(), cfg.norm_eps, work);
    // Fused Q-plus-gate projection, split per head (query then gate).
    const std::size_t q_dim = heads * head_dim;
    auto qg = Project(backend, *(*gemm_q), work, *(*wq)->device, q_dim * 2);
    if (!qg) {
      return std::unexpected(qg.error());
    }
    std::vector<float> q(q_dim);
    std::vector<float> gate(q_dim);
    for (std::size_t h = 0; h < heads; ++h) {
      std::memcpy(q.data() + h * head_dim,
                  qg->data() + h * 2 * head_dim, head_dim * 4);
      std::memcpy(gate.data() + h * head_dim,
                  qg->data() + h * 2 * head_dim + head_dim, head_dim * 4);
    }
    auto k_row = Project(backend, *(*gemm_k), work, *(*wk)->device, kv_dim);
    auto v_row = Project(backend, *(*gemm_v), work, *(*wv)->device, kv_dim);
    if (!k_row || !v_row) {
      return std::unexpected(StatusCode::DeviceError);
    }
    // QK-Norm over the head dim.
    auto q_norm_w = DownloadF32(backend, *(*q_norm)->device);
    auto k_norm_w = DownloadF32(backend, *(*k_norm)->device);
    if (!q_norm_w || !k_norm_w) {
      return std::unexpected(StatusCode::DeviceError);
    }
    NormHeads(q, heads, head_dim, *q_norm_w, cfg.norm_eps);
    NormHeads(*k_row, kv_heads, head_dim, *k_norm_w, cfg.norm_eps);
    // mRoPE (text rows repeat the same id three times).
    HybridDecodeCache::FullLayer& full = hc.full[l];
    const std::size_t pos = full.k.size() / kv_dim;
    const std::vector<std::uint64_t> triple = {pos, pos, pos};
    auto roped_q = RunMrope(backend, *hc.mrope_kernel, q, triple, 1, heads,
                            head_dim, rope_dim, sec_t, sec_h, sec_w,
                            cfg.attention.rope_theta);
    auto roped_k = RunMrope(backend, *hc.mrope_kernel, *k_row, triple, 1,
                            kv_heads, head_dim, rope_dim, sec_t, sec_h, sec_w,
                            cfg.attention.rope_theta);
    if (!roped_q || !roped_k) {
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
    // Sigmoid output gate, then the output projection and residual.
    std::vector<float> gated(q_dim);
    for (std::size_t i = 0; i < q_dim; ++i) {
      gated[i] = (*attn)[i] / (1.0f + std::exp(-gate[i]));
    }
    auto proj_out = Project(backend, *(*gemm_o), gated, *(*wo)->device, hidden);
    if (!proj_out) {
      return std::unexpected(proj_out.error());
    }
    for (std::size_t i = 0; i < hidden; ++i) {
      x[i] += (*proj_out)[i];
    }
    // Post-attention norm and the gated MLP, with residual.
    auto post_w = DownloadF32(backend, *(*post_norm)->device);
    if (!post_w) {
      return std::unexpected(post_w.error());
    }
    RmsNormInto(x, post_w->data(), cfg.norm_eps, work);
    auto g = Project(backend, *(*gemm_gate), work, *(*ffn_gate)->device,
                     cfg.ffn_dim);
    auto u = Project(backend, *(*gemm_up), work, *(*ffn_up)->device,
                     cfg.ffn_dim);
    if (!g || !u) {
      return std::unexpected(StatusCode::DeviceError);
    }
    SiluMulInto(*g, *u, mlp_work);
    auto d = Project(backend, *(*gemm_down), mlp_work, *(*ffn_down)->device,
                     hidden);
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
  auto gemm_out = GemmFor(backend, hc, (*output)->manifest.dtype);
  if (!gemm_out) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  auto out_norm_w = DownloadF32(backend, *(*out_norm)->device);
  if (!out_norm_w) {
    return std::unexpected(out_norm_w.error());
  }
  RmsNormInto(x, out_norm_w->data(), cfg.norm_eps, work);
  auto logits =
      Project(backend, *(*gemm_out), work, *(*output)->device,
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
