#include "core/decode.hpp"

#include <cmath>
#include <cstring>
#include <string>

namespace tessera::core {

namespace {

// A required weight: missing is MalformedFile, a wrong layout is
// UnsupportedFeature (only Q4_K projections and F32 vectors run).
std::expected<const DeviceTensor*, StatusCode> NeedWeight(
    const Model& model, std::string_view name, DType dtype) {
  for (const auto& weight : model.Weights()) {
    if (weight.manifest.name == name) {
      if (weight.manifest.dtype != dtype) {
        return std::unexpected(StatusCode::UnsupportedFeature);
      }
      return &weight;
    }
  }
  return std::unexpected(StatusCode::MalformedFile);
}

void RmsNormInto(const std::vector<float>& x, const float* w, double eps,
                 std::vector<float>& out) {
  float mean = 0.0f;
  for (float v : x) {
    mean += v * v;
  }
  mean /= static_cast<float>(x.size());
  const float gain =
      1.0f / std::sqrt(mean + static_cast<float>(eps));
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = x[i] * gain * w[i];
  }
}

void SiluMulInto(const std::vector<float>& gate, const std::vector<float>& up,
                 std::vector<float>& out) {
  for (std::size_t i = 0; i < gate.size(); ++i) {
    const float silu = gate[i] / (1.0f + std::exp(-gate[i]));
    out[i] = silu * up[i];
  }
}

// Small F32 vectors (norms) download to the host; elementwise ops run
// there until their device kernels land.
std::expected<std::vector<float>, StatusCode> DownloadF32(
    Backend& backend, const Buffer& buf) {
  std::vector<std::byte> raw(buf.Size());
  auto down = backend.CopyD2H(buf, raw.data(), raw.size());
  if (!down) {
    return std::unexpected(down.error());
  }
  std::vector<float> out(buf.Size() / 4);
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

// y = A(1 x k) times dequant(W) on the device (W is Q4_K, n x k).
std::expected<std::vector<float>, StatusCode> Project(
    Backend& backend, const Kernel& gemm, const std::vector<float>& x,
    const Buffer& weights, std::size_t n) {
  const std::size_t k = x.size();
  auto x_buf = backend.AllocateBuffer(k * 4, MemoryKind::Device);
  if (!x_buf) {
    return std::unexpected(x_buf.error());
  }
  auto out_buf = backend.AllocateBuffer(n * 4, MemoryKind::Device);
  if (!out_buf) {
    return std::unexpected(out_buf.error());
  }
  auto up = backend.CopyH2D(
      **x_buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(x.data()), k * 4));
  if (!up) {
    return std::unexpected(up.error());
  }
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((n + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {(*x_buf).get(), &weights, (*out_buf).get()};
  launch.scalars = {k, n, 1};
  auto ran = backend.LaunchKernel(gemm, launch);
  if (!ran) {
    return std::unexpected(ran.error());
  }
  backend.Synchronize();
  std::vector<std::byte> raw(n * 4);
  auto down = backend.CopyD2H(**out_buf, raw.data(), raw.size());
  if (!down) {
    return std::unexpected(down.error());
  }
  std::vector<float> out(n);
  std::memcpy(out.data(), raw.data(), n * 4);
  return out;
}

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

// Causal attention over one query row (device); the full key/value
// matrices upload fresh every step.
std::expected<std::vector<float>, StatusCode> Attend(
    Backend& backend, const Kernel& attention, const std::vector<float>& q,
    const std::vector<float>& k, const std::vector<float>& v,
    std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
    std::uint64_t q_base) {
  const std::size_t kv_dim = kv_heads * head_dim;
  const std::size_t n = k.size() / kv_dim;
  auto q_buf = backend.AllocateBuffer(q.size() * 4, MemoryKind::Device);
  auto k_buf = backend.AllocateBuffer(k.size() * 4, MemoryKind::Device);
  auto v_buf = backend.AllocateBuffer(v.size() * 4, MemoryKind::Device);
  auto out_buf =
      backend.AllocateBuffer(heads * head_dim * 4, MemoryKind::Device);
  if (!q_buf || !k_buf || !v_buf || !out_buf) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend.CopyH2D(
        **buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(data.data()),
                   data.size() * 4));
  };
  if (!upload(q_buf, q) || !upload(k_buf, k) || !upload(v_buf, v)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  KernelLaunch launch;
  launch.grid_x =
      static_cast<std::uint32_t>((heads * head_dim + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {(*q_buf).get(), (*k_buf).get(), (*v_buf).get(),
                    (*out_buf).get()};
  launch.scalars = {1, n, heads, kv_heads, head_dim, q_base};
  auto ran = backend.LaunchKernel(attention, launch);
  if (!ran) {
    return std::unexpected(ran.error());
  }
  backend.Synchronize();
  std::vector<std::byte> raw(heads * head_dim * 4);
  auto down = backend.CopyD2H(**out_buf, raw.data(), raw.size());
  if (!down) {
    return std::unexpected(down.error());
  }
  std::vector<float> out(heads * head_dim);
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
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
  // Hybrid attention/SSM blocks need the recurrent kernels; the
  // vanilla walk below cannot run them, so fail fast and loud.
  if (cfg.hybrid) {
    return std::unexpected(StatusCode::UnsupportedFeature);
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
    auto norm_w = DownloadF32(backend, *(*norm)->device);
    if (!norm_w) {
      return std::unexpected(norm_w.error());
    }
    RmsNormInto(x, norm_w->data(), cfg.norm_eps, work);
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
    auto mlp_norm_w = DownloadF32(backend, *(*mlp_norm)->device);
    if (!mlp_norm_w) {
      return std::unexpected(mlp_norm_w.error());
    }
    RmsNormInto(x, mlp_norm_w->data(), cfg.norm_eps, work);
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
  auto out_norm_w = DownloadF32(backend, *(*out_norm)->device);
  if (!out_norm_w) {
    return std::unexpected(out_norm_w.error());
  }
  RmsNormInto(x, out_norm_w->data(), cfg.norm_eps, work);
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
