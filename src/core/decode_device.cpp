#include "core/decode.hpp"

#include <cstring>
#include <string>
#include <vector>

#include "core/decode_internal.hpp"

namespace tessera::core {

namespace {

using detail::AppendKv;
using detail::DownloadF32;
using detail::GatherEmbedding;
using detail::AttentionDevice;
using detail::GemmFor;
using detail::RopeDevice;
using detail::NeedWeight;
using detail::NeedWeightAny;
using detail::UploadF32;

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

}  // namespace

std::expected<void, StatusCode> DecodeStepDeviceForward(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out) {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const TransformerConfig& cfg = *config;
  if (cfg.hybrid) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  const std::size_t hidden = cfg.hidden_dim;
  const std::size_t heads = cfg.attention.heads;
  const std::size_t kv_heads = cfg.attention.kv_heads;
  const std::size_t head_dim = cfg.attention.head_dim;
  const std::size_t kv_dim = kv_heads * head_dim;
  if (token >= cfg.vocab_size) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (cache.device == nullptr) {
    auto state = std::make_unique<DeviceDecodeState>();
    auto load = [&backend](std::unique_ptr<Kernel>& slot,
                           std::string_view name) -> bool {
      auto kernel = backend.LoadKernel(name, {});
      if (!kernel) {
        return false;
      }
      slot = std::move(*kernel);
      return true;
    };
    if (!load(state->rmsnorm_kernel, "rmsnorm") ||
        !load(state->add_kernel, "add") ||
        !load(state->silu_mul_kernel, "silu_mul") ||
        !load(state->rope_kernel, "rope") ||
        !load(state->attention_kernel, "attention")) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (cache.kv_f16) {
      if (!load(state->cast_kernel, "cast_f32_f16")) {
        return std::unexpected(StatusCode::DeviceError);
      }
      auto scratch = backend.AllocateBuffer(kv_dim * 2, MemoryKind::Device);
      if (!scratch) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      state->kv_scratch = std::move(*scratch);
    }
    const std::size_t h4 = hidden * 4;
    const std::size_t q4 = heads * head_dim * 4;
    const std::size_t kv4 = kv_dim * 4;
    const std::size_t f4 = cfg.ffn_dim * 4;
    auto alloc = [&backend](std::size_t bytes) {
      return backend.AllocateBuffer(bytes, MemoryKind::Device);
    };
    auto x = alloc(h4);
    auto xn = alloc(h4);
    auto proj = alloc(h4);
    auto q = alloc(q4);
    auto k = alloc(kv4);
    auto v = alloc(kv4);
    auto attn = alloc(q4);
    auto gate = alloc(f4);
    auto up = alloc(f4);
    auto mlp = alloc(f4);
    auto logits = alloc(cfg.vocab_size * 4);
    if (!x || !xn || !proj || !q || !k || !v || !attn || !gate || !up || !mlp ||
        !logits) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    state->x = std::move(*x);
    state->xn = std::move(*xn);
    state->proj = std::move(*proj);
    state->q = std::move(*q);
    state->k = std::move(*k);
    state->v = std::move(*v);
    state->attn = std::move(*attn);
    state->gate = std::move(*gate);
    state->up = std::move(*up);
    state->mlp = std::move(*mlp);
    state->logits = std::move(*logits);
    state->kv.resize(cfg.layers);
    for (auto& kv : state->kv) {
      kv.f16 = cache.kv_f16;
    }
    cache.device = std::move(state);
  }
  DeviceDecodeState& st = *cache.device;
  auto embed = NeedWeightAny(model, "token_embd.weight");
  if (!embed) {
    return std::unexpected(embed.error());
  }
  if ((*embed)->manifest.dtype == DType::F32) {
    const std::size_t offset = static_cast<std::size_t>(token) * hidden * 4;
    auto copy = backend.CopyD2D(*(*embed)->device, offset, *st.x, 0, hidden * 4);
    if (!copy) {
      return std::unexpected(copy.error());
    }
  } else {
    std::vector<float> row(hidden);
    auto gathered = GatherEmbedding(backend, **embed, token, hidden, row);
    if (!gathered) {
      return std::unexpected(gathered.error());
    }
    if (!UploadF32(backend, *st.x, row)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  for (std::size_t l = 0; l < cfg.layers; ++l) {
    const std::string base = "blk." + std::to_string(l) + ".";
    auto norm = NeedWeight(model, base + "attn_norm.weight", DType::F32);
    auto wq = NeedWeightAny(model, base + "attn_q.weight");
    auto wk = NeedWeightAny(model, base + "attn_k.weight");
    auto wv = NeedWeightAny(model, base + "attn_v.weight");
    auto wo = NeedWeightAny(model, base + "attn_output.weight");
    auto mlp_norm = NeedWeight(model, base + "ffn_norm.weight", DType::F32);
    auto gate_w = NeedWeightAny(model, base + "ffn_gate.weight");
    auto up_w = NeedWeightAny(model, base + "ffn_up.weight");
    auto down_w = NeedWeightAny(model, base + "ffn_down.weight");
    if (!norm || !wq || !wk || !wv || !wo || !mlp_norm || !gate_w || !up_w ||
        !down_w) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    auto gemm_q = GemmFor(backend, st.gemms, (*wq)->manifest.dtype);
    auto gemm_k = GemmFor(backend, st.gemms, (*wk)->manifest.dtype);
    auto gemm_v = GemmFor(backend, st.gemms, (*wv)->manifest.dtype);
    auto gemm_o = GemmFor(backend, st.gemms, (*wo)->manifest.dtype);
    auto gemm_gate = GemmFor(backend, st.gemms, (*gate_w)->manifest.dtype);
    auto gemm_up = GemmFor(backend, st.gemms, (*up_w)->manifest.dtype);
    auto gemm_down = GemmFor(backend, st.gemms, (*down_w)->manifest.dtype);
    if (!gemm_q || !gemm_k || !gemm_v || !gemm_o || !gemm_gate || !gemm_up ||
        !gemm_down) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    DeviceDecodeState::Kv& kv = st.kv[l];
    const std::size_t pos = kv.rows;
    if (!detail::RmsNormDevice(backend, *st.rmsnorm_kernel, *st.x,
                               *(*norm)->device, *st.xn, 1, hidden,
                               cfg.norm_eps)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (!detail::ProjectDevice(backend, *(*gemm_q), *st.xn, *(*wq)->device,
                               *st.q, 1, heads * head_dim, hidden) ||
        !detail::ProjectDevice(backend, *(*gemm_k), *st.xn, *(*wk)->device,
                               *st.k, 1, kv_dim, hidden) ||
        !detail::ProjectDevice(backend, *(*gemm_v), *st.xn, *(*wv)->device,
                               *st.v, 1, kv_dim, hidden)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (!RopeDevice(backend, *st.rope_kernel, *st.q, heads, head_dim,
                    cfg.attention.rope_dim, pos, cfg.attention.rope_theta) ||
        !RopeDevice(backend, *st.rope_kernel, *st.k, kv_heads, head_dim,
                    cfg.attention.rope_dim, pos, cfg.attention.rope_theta)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (auto appended = AppendKv(backend, st.cast_kernel.get(),
                                 st.kv_scratch.get(), kv, *st.k, *st.v,
                                 kv_dim);
        !appended) {
      return std::unexpected(appended.error());
    }
    if (!AttentionDevice(backend, *st.attention_kernel, *st.q, *kv.k, *kv.v,
                         *st.attn, kv.rows, heads, kv_heads, head_dim, pos, 0,
                         1, kv.f16)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (!detail::ProjectDevice(backend, *(*gemm_o), *st.attn, *(*wo)->device,
                               *st.proj, 1, hidden, heads * head_dim)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (!detail::AddDevice(backend, *st.add_kernel, *st.x, *st.proj, *st.x,
                           hidden) ||
        !detail::RmsNormDevice(backend, *st.rmsnorm_kernel, *st.x,
                               *(*mlp_norm)->device, *st.xn, 1, hidden,
                               cfg.norm_eps) ||
        !detail::ProjectDevice(backend, *(*gemm_gate), *st.xn,
                               *(*gate_w)->device, *st.gate, 1, cfg.ffn_dim,
                               hidden) ||
        !detail::ProjectDevice(backend, *(*gemm_up), *st.xn, *(*up_w)->device,
                               *st.up, 1, cfg.ffn_dim, hidden) ||
        !detail::SiluMulDevice(backend, *st.silu_mul_kernel, *st.gate, *st.up,
                               *st.mlp, cfg.ffn_dim) ||
        !detail::ProjectDevice(backend, *(*gemm_down), *st.mlp,
                               *(*down_w)->device, *st.proj, 1, hidden,
                               cfg.ffn_dim) ||
        !detail::AddDevice(backend, *st.add_kernel, *st.x, *st.proj, *st.x,
                           hidden)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  if (hidden_out != nullptr) {
    auto snapshot = detail::DownloadF32(backend, *st.x);
    if (!snapshot) {
      return std::unexpected(snapshot.error());
    }
    *hidden_out = std::move(*snapshot);
  }
  return {};
}

std::expected<std::vector<float>, StatusCode> DecodeStepDeviceLogits(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out) {
  auto forward =
      DecodeStepDeviceForward(backend, model, cache, token, hidden_out);
  if (!forward) {
    return std::unexpected(forward.error());
  }
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const TransformerConfig& cfg = *config;
  DeviceDecodeState& st = *cache.device;
  const std::size_t hidden = cfg.hidden_dim;
  auto out_norm = NeedWeight(model, "output_norm.weight", DType::F32);
  auto output = NeedWeightAny(model, "output.weight");
  if (!out_norm || !output) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto gemm_out = GemmFor(backend, st.gemms, (*output)->manifest.dtype);
  if (!gemm_out) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  if (!detail::RmsNormDevice(backend, *st.rmsnorm_kernel, *st.x,
                             *(*out_norm)->device, *st.xn, 1, hidden,
                             cfg.norm_eps) ||
      !detail::ProjectDevice(backend, *(*gemm_out), *st.xn,
                             *(*output)->device, *st.logits, 1, cfg.vocab_size,
                             hidden)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  backend.Synchronize();
  return DownloadF32(backend, *st.logits);
}

std::expected<std::uint32_t, StatusCode> DecodeStepDevice(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out) {
  auto logits =
      DecodeStepDeviceLogits(backend, model, cache, token, hidden_out);
  if (!logits) {
    return std::unexpected(logits.error());
  }
  return detail::ArgMax(*logits);
}

}  // namespace tessera::core
