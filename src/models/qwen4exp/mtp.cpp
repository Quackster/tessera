#include "models/qwen4exp/mtp.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/profile.hpp"
#include "models/qwen4exp/architecture.hpp"
#include "models/qwen4exp/mtp_weights.hpp"
#include "models/qwen4exp/state.hpp"
#include "tessera/log.hpp"

// The qwen4exp MTP draft block. See mtp.hpp for the contract. The attention
// is dense over the MTP's own paged FP16 cache (the sparse indexer only
// changes which cells are read; below the selection bound it reads every
// cell, so the draft is identical). The routed experts are resident BF16 and
// run one expert at a time through the generic GEMM: the draft decides no
// emitted token, so its speed is secondary to the target's.

namespace tessera::models::qwen4exp {

namespace detail = ::tessera::core::detail;

using detail::AddDevice;
using detail::DownloadF32;
using detail::GatherEmbedding;
using detail::GemmFor;
using detail::NeedWeightAny;
using detail::ProjectDevice;
using detail::QGateSplitDevice;
using detail::RmsNormDevice;
using detail::RopeDevice;
using detail::SiluMulDevice;
using detail::TopKRowsDevice;
using detail::UploadF32;

namespace {


// The MTP draft cache geometry. The pool covers the model's context (the
// CLI default), so the draft never faults before the trunk; the dense
// attention reads at most the most recent cells, which bounds the softmax
// scratch (qsa_attend's cap).
constexpr std::size_t kMtpCells = 32768;
constexpr std::size_t kMtpPageSize = 4;
constexpr std::size_t kMtpPages = kMtpCells / kMtpPageSize;
constexpr std::size_t kMtpMaxAttend = 4096;

std::uint32_t Grid(std::size_t elements) {
  return static_cast<std::uint32_t>((elements + 255) / 256);
}

std::expected<void, StatusCode> Alloc(Backend& backend,
                                      std::unique_ptr<Buffer>& slot,
                                      std::size_t bytes, MemoryKind memory) {
  auto buffer = backend.AllocateBuffer(bytes, memory);
  if (!buffer) {
    return std::unexpected(buffer.error());
  }
  slot = std::move(*buffer);
  return {};
}

// out = a + b * weight[idx], the routed-expert accumulation.
std::expected<void, StatusCode> ScaleAdd(Backend& backend, Qwen4ExpState& h,
                                         const Buffer& a, const Buffer& b,
                                         const Buffer& weight, Buffer& out,
                                         std::size_t n, std::size_t idx) {
  KernelLaunch launch;
  launch.grid_x = Grid(n);
  launch.block_x = 256;
  launch.buffers = {&a, &b, &weight, &out};
  launch.scalars = {n, idx};
  return backend.LaunchKernel(*h.moe_scale_add, launch);
}

}  // namespace

// Load the MTP weights and allocate the draft cache and scratch once.
std::expected<void, StatusCode> EnsureMtpReady(Backend& backend,
                                               const Model& model,
                                               const TransformerConfig& cfg,
                                               const Qwen4ExpGeometry& g,
                                               Qwen4ExpState& h) {
  if (h.mtp_ready) {
    return {};
  }
  if (model.MtpPath().empty()) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  const auto* module =
      static_cast<const Qwen4ExpArchitecture*>(model.Arch());
  if (module == nullptr) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  auto shared = module->SharedMtpWeights(backend, model);
  if (!shared) {
    return std::unexpected(shared.error());
  }
  h.mtp_weights = *shared;
  const std::size_t pool_bytes =
      kMtpPages * g.kv_heads * kMtpPageSize * g.head_dim * 2;
  const std::size_t attend_bytes =
      kMtpMaxAttend * g.kv_heads * g.head_dim * 2;
  if (!Alloc(backend, h.mtp_k_pool, pool_bytes, h.kv_memory) ||
      !Alloc(backend, h.mtp_v_pool, pool_bytes, h.kv_memory) ||
      !Alloc(backend, h.mtp_page_table, kMtpPages * 4, MemoryKind::Device) ||
      !Alloc(backend, h.mtp_ids, kMtpMaxAttend * 4, MemoryKind::Device) ||
      !Alloc(backend, h.mtp_k_scratch, attend_bytes, MemoryKind::Device) ||
      !Alloc(backend, h.mtp_v_scratch, attend_bytes, MemoryKind::Device) ||
      !Alloc(backend, h.mtp_expert_scratch,
             2 * h.mtp_weights->inter * h.mtp_weights->hidden * 2,
             MemoryKind::Device) ||
      !Alloc(backend, h.mtp_hidden_branch, g.hc_dim * 4, MemoryKind::Device) ||
      !Alloc(backend, h.mtp_streams, g.hc_dim * 4, MemoryKind::Device)) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  // The draft's paged K/V pools start empty, so an unwritten cell reads as
  // zero rather than whatever the allocator returned.
  const std::vector<std::byte> mtp_pool_zeros(pool_bytes, std::byte{0});
  if (!backend.CopyH2D(*h.mtp_k_pool, mtp_pool_zeros) ||
      !backend.CopyH2D(*h.mtp_v_pool, mtp_pool_zeros)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  // The MTP MoE reuses the trunk's shared MoE scratch. On a head-only draft
  // device the trunk never ran, so those buffers were never grown; allocate
  // them here with the sizes the MTP and the trunk both use.
  const std::size_t ne = h.mtp_weights->experts;
  const std::size_t inter = h.mtp_weights->inter;
  const std::size_t hidden = h.mtp_weights->hidden;
  const std::size_t top_k = cfg.experts_per_tok;
  if (!Alloc(backend, h.moe_logits, ne * 4, MemoryKind::Device) ||
      !Alloc(backend, h.moe_ids, top_k * 4, MemoryKind::Device) ||
      !Alloc(backend, h.moe_vals, top_k * 4, MemoryKind::Device) ||
      !Alloc(backend, h.moe_wts, top_k * 4, MemoryKind::Device) ||
      !Alloc(backend, h.moe_sig, 4, MemoryKind::Device) ||
      !Alloc(backend, h.moe_acc, hidden * 4, MemoryKind::Device) ||
      !Alloc(backend, h.moe_shared, hidden * 4, MemoryKind::Device) ||
      !Alloc(backend, h.moe_gate_out, inter * 4, MemoryKind::Device) ||
      !Alloc(backend, h.moe_up_out, inter * 4, MemoryKind::Device) ||
      !Alloc(backend, h.moe_inter, inter * 4, MemoryKind::Device) ||
      !Alloc(backend, h.moe_phi, top_k * inter * 4, MemoryKind::Device)) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  std::vector<std::int32_t> table(kMtpPages);
  for (std::size_t p = 0; p < kMtpPages; ++p) {
    table[p] = static_cast<std::int32_t>(p);
  }
  if (!backend.CopyH2D(
          *h.mtp_page_table,
          std::span<const std::byte>(
              reinterpret_cast<const std::byte*>(table.data()),
              table.size() * sizeof(table[0])))) {
    return std::unexpected(StatusCode::DeviceError);
  }
  h.mtp_ready = true;
  return {};
}

namespace {

// Dense QSA attention over the MTP's own cache: project q (with gate), k
// and v, norm and rotate, append the cell, gather the attended cells and
// run the softmax GQA with the output gate.
std::expected<void, StatusCode> RunMtpAttention(
    Backend& backend, const TransformerConfig& cfg, const Qwen4ExpGeometry& g,
    Qwen4ExpState& h, const MtpWeights& w, std::uint64_t pos,
    const Buffer& mixed, Buffer& block_out) {
  const Buffer* wq = w.Get("mtp.layers.0.self_attn.q_proj.weight");
  const Buffer* wk = w.Get("mtp.layers.0.self_attn.k_proj.weight");
  const Buffer* wv = w.Get("mtp.layers.0.self_attn.v_proj.weight");
  const Buffer* wo = w.Get("mtp.layers.0.self_attn.o_proj.weight");
  const Buffer* qn = w.Get("mtp.layers.0.self_attn.q_norm.weight");
  const Buffer* kn = w.Get("mtp.layers.0.self_attn.k_norm.weight");
  if (wq == nullptr || wk == nullptr || wv == nullptr || wo == nullptr ||
      qn == nullptr || kn == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (!Project(backend, h, DType::BF16, mixed, *wq, *h.q_full, 1,
               g.heads * 2 * g.head_dim, g.hidden) ||
      !QGateSplitDevice(backend, *h.qgate_split, *h.q_full, *h.q_attn,
                        *h.gate_buf, g.heads, g.head_dim) ||
      !Project(backend, h, DType::BF16, mixed, *wk, *h.kcur, 1, g.kv_dim,
               g.hidden) ||
      !Project(backend, h, DType::BF16, mixed, *wv, *h.vcur, 1, g.kv_dim,
               g.hidden) ||
      !RmsNormDevice(backend, *h.rmsnorm, *h.q_attn, *qn, *h.q_attn, g.heads,
                     g.head_dim, cfg.norm_eps) ||
      !RmsNormDevice(backend, *h.rmsnorm, *h.kcur, *kn, *h.kcur, g.kv_heads,
                     g.head_dim, cfg.norm_eps) ||
      !RopeDevice(backend, *h.rope, *h.q_attn, g.heads, g.head_dim, g.rope_dim,
                  pos, g.rope_theta) ||
      !RopeDevice(backend, *h.rope, *h.kcur, g.kv_heads, g.head_dim, g.rope_dim,
                  pos, g.rope_theta)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  {
    KernelLaunch append;
    append.grid_x = Grid(g.kv_heads * g.head_dim / 2);
    append.block_x = 256;
    append.buffers = {h.mtp_k_pool.get(), h.mtp_v_pool.get(),
                      h.mtp_page_table.get(), h.kcur.get(), h.vcur.get()};
    append.scalars = {pos, g.kv_heads, g.head_dim, kMtpPageSize};
    if (!backend.LaunchKernel(*h.kv_append, append)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  const std::uint64_t n_kv = std::min<std::uint64_t>(pos + 1, kMtpCells);
  const std::uint64_t start =
      n_kv > kMtpMaxAttend ? n_kv - kMtpMaxAttend : 0;
  const std::uint64_t n_ids = n_kv - start;
  {
    std::vector<std::int32_t> ids(n_ids);
    for (std::size_t i = 0; i < n_ids; ++i) {
      ids[i] = static_cast<std::int32_t>(start + i);
    }
    if (!backend.CopyH2D(
            *h.mtp_ids,
            std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(ids.data()),
                ids.size() * sizeof(ids[0])))) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  {
    KernelLaunch gather;
    gather.grid_x = Grid(n_ids * g.kv_heads * g.head_dim);
    gather.block_x = 256;
    gather.buffers = {h.mtp_k_pool.get(),   h.mtp_v_pool.get(),
                      h.mtp_page_table.get(), h.mtp_ids.get(),
                      h.mtp_k_scratch.get(), h.mtp_v_scratch.get()};
    gather.scalars = {n_ids, g.kv_heads, g.head_dim, kMtpPageSize};
    if (!backend.LaunchKernel(*h.kv_gather, gather)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  {
    KernelLaunch attend;
    attend.grid_x = static_cast<std::uint32_t>(g.heads);
    attend.block_x = 256;
    attend.buffers = {h.q_attn.get(), h.mtp_k_scratch.get(),
                      h.mtp_v_scratch.get(), h.q_attn.get()};
    attend.scalars = {n_ids, g.heads, g.kv_heads, g.head_dim};
    if (!backend.LaunchKernel(*h.qsa_attend, attend)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  {
    KernelLaunch gate;
    gate.grid_x = Grid(g.heads * g.head_dim / 2);
    gate.block_x = 256;
    gate.buffers = {h.q_attn.get(), h.q_full.get(), h.attn_f16.get()};
    gate.scalars = {g.heads, g.head_dim};
    if (!backend.LaunchKernel(*h.qsa_gate_apply, gate)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  {
    const std::size_t n = g.heads * g.head_dim;
    KernelLaunch cast;
    cast.grid_x = Grid(n / 2);
    cast.block_x = 256;
    cast.buffers = {h.attn_f16.get(), h.attn_in.get()};
    cast.scalars = {n};
    if (!backend.LaunchKernel(*h.cast_f16_f32, cast)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  if (!Project(backend, h, DType::BF16, *h.attn_in, *wo, block_out, 1, g.hidden,
               g.heads * g.head_dim)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

// The MTP MoE: router, the always-on shared expert, the top-k resident BF16
// experts one at a time, and the combine into `block_out`.
std::expected<void, StatusCode> RunMtpMoe(Backend& backend,
                                          const TransformerConfig& cfg,
                                          Qwen4ExpState& h, const MtpWeights& w,
                                          const Buffer& mixed) {
  const std::size_t hidden = w.hidden;
  const std::size_t ne = w.experts;
  const std::size_t inter = w.inter;
  const std::size_t top_k = cfg.experts_per_tok;
  const Buffer* router = w.Get("mtp.layers.0.mlp.gate.weight");
  const Buffer* shexp_gate = w.Get("mtp.layers.0.mlp.shared_expert_gate.weight");
  const Buffer* gate_sh = w.Get("mtp.layers.0.mlp.shared_expert.gate_proj.weight");
  const Buffer* up_sh = w.Get("mtp.layers.0.mlp.shared_expert.up_proj.weight");
  const Buffer* down_sh = w.Get("mtp.layers.0.mlp.shared_expert.down_proj.weight");
  const Buffer* gate_up = w.Get("mtp.layers.0.mlp.experts.gate_up_proj");
  const Buffer* down = w.Get("mtp.layers.0.mlp.experts.down_proj");
  if (router == nullptr || shexp_gate == nullptr || gate_sh == nullptr ||
      up_sh == nullptr || down_sh == nullptr || gate_up == nullptr ||
      down == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (!Project(backend, h, DType::BF16, mixed, *router, *h.moe_logits, 1, ne,
               hidden) ||
      !TopKRowsDevice(backend, *h.moe_topk, *h.moe_logits, *h.moe_ids,
                      *h.moe_vals, 1, ne, top_k)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  {
    KernelLaunch gate;
    gate.grid_x = 1;
    gate.block_x = 256;
    gate.buffers = {h.moe_vals.get(), const_cast<Buffer*>(&mixed),
                    const_cast<Buffer*>(shexp_gate), h.moe_wts.get(),
                    h.moe_sig.get()};
    gate.scalars = {top_k, hidden};
    if (!backend.LaunchKernel(*h.moe_gate, gate)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  const std::size_t shared_inter = inter;
  if (!Project(backend, h, DType::BF16, mixed, *gate_sh, *h.moe_gate_out, 1,
               shared_inter, hidden) ||
      !Project(backend, h, DType::BF16, mixed, *up_sh, *h.moe_up_out, 1,
               shared_inter, hidden) ||
      !SiluMulDevice(backend, *h.silu_mul, *h.moe_gate_out, *h.moe_up_out,
                     *h.moe_inter, shared_inter) ||
      !Project(backend, h, DType::BF16, *h.moe_inter, *down_sh, *h.moe_shared, 1,
               hidden, shared_inter)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  std::vector<std::uint32_t> ids(top_k, 0);
  if (!backend.CopyD2H(*h.moe_ids, reinterpret_cast<std::byte*>(ids.data()),
                       top_k * 4)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  std::vector<float> zeros(hidden, 0.0f);
  if (!UploadF32(backend, *h.moe_acc, zeros)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  const std::size_t gu_bytes = 2 * inter * hidden * 2;
  const std::size_t down_bytes = hidden * inter * 2;
  for (std::size_t s = 0; s < top_k; ++s) {
    const std::size_t e = ids[s];
    if (e >= ne) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    if (!backend.CopyD2D(*gate_up, e * gu_bytes, *h.mtp_expert_scratch, 0,
                         gu_bytes) ||
        !Project(backend, h, DType::BF16, mixed, *h.mtp_expert_scratch,
                 *h.moe_phi, 1, 2 * inter, hidden)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (!backend.CopyD2D(*h.moe_phi, 0, *h.moe_gate_out, 0, inter * 4) ||
        !backend.CopyD2D(*h.moe_phi, inter * 4, *h.moe_up_out, 0, inter * 4) ||
        !SiluMulDevice(backend, *h.silu_mul, *h.moe_gate_out, *h.moe_up_out,
                       *h.moe_inter, inter) ||
        !backend.CopyD2D(*down, e * down_bytes, *h.mtp_expert_scratch, 0,
                         down_bytes) ||
        !Project(backend, h, DType::BF16, *h.moe_inter, *h.mtp_expert_scratch,
                 *h.proj, 1, hidden, inter) ||
        !ScaleAdd(backend, h, *h.moe_acc, *h.proj, *h.moe_wts, *h.moe_acc,
                  hidden, s)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  if (!ScaleAdd(backend, h, *h.moe_acc, *h.moe_shared, *h.moe_sig, *h.block_out,
                hidden, 0)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

}  // namespace

std::expected<std::uint32_t, StatusCode> RunMtpDraft(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    const Qwen4ExpGeometry& g, Qwen4ExpState& h, std::span<const float> hidden,
    std::uint32_t token, std::uint64_t pos, std::vector<float>* mtp_hidden_out,
    std::vector<float>* logits_out) {
  if (auto ready = EnsureMtpReady(backend, model, cfg, g, h); !ready) {
    return std::unexpected(ready.error());
  }
  const MtpWeights& w = *h.mtp_weights;
  if (hidden.size() != g.hc_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const Buffer* fc_embedding = w.Get("mtp.fc_embedding.weight");
  const Buffer* fc_hidden = w.Get("mtp.fc_hidden.weight");
  const Buffer* norm_embed = w.Get("mtp.pre_fc_norm_embedding.weight");
  const Buffer* norm_hidden = w.Get("mtp.pre_fc_norm_hidden.weight");
  const Buffer* attn_norm =
      w.Get("mtp.layers.0.attn_hyper_connection.hc_norm.weight");
  const Buffer* attn_down = w.Get(
      "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight");
  const Buffer* attn_up = w.Get(
      "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight");
  const Buffer* attn_inject = w.Get(
      "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight");
  const Buffer* mlp_norm =
      w.Get("mtp.layers.0.mlp_hyper_connection.hc_norm.weight");
  const Buffer* mlp_down = w.Get(
      "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight");
  const Buffer* mlp_up = w.Get(
      "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight");
  const Buffer* mlp_inject = w.Get(
      "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight");
  const Buffer* mix_norm = w.Get("mtp.hyper_connection_mixer.hc_norm.weight");
  const Buffer* mix_down =
      w.Get("mtp.hyper_connection_mixer.input_mix_weight_down.weight");
  const Buffer* mix_up =
      w.Get("mtp.hyper_connection_mixer.input_mix_weight_up.weight");
  auto embed = NeedWeightAny(model, "token_embd.weight");
  auto output = NeedWeightAny(model, "output.weight");
  if (fc_embedding == nullptr || fc_hidden == nullptr || norm_embed == nullptr ||
      norm_hidden == nullptr || attn_norm == nullptr || attn_down == nullptr ||
      attn_up == nullptr || attn_inject == nullptr || mlp_norm == nullptr ||
      mlp_down == nullptr || mlp_up == nullptr || mlp_inject == nullptr ||
      mix_norm == nullptr || mix_down == nullptr || mix_up == nullptr ||
      !embed || !output) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  // The two input branches. e = fc_embedding(rms(embed(tok)) * (1+w));
  // h = fc_hidden(rms_10240(R) * (1+w)); R = h + e broadcast over streams.
  {
    std::vector<float> row(g.hidden);
    if (auto gathered = GatherEmbedding(backend, **embed, token, g.hidden, row);
        !gathered) {
      return std::unexpected(gathered.error());
    }
    if (!UploadF32(backend, *h.x, row)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (!RmsNormDevice(backend, *h.rmsnorm, *h.x, *norm_embed, *h.mixed, 1,
                       g.hidden, cfg.norm_eps)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (!Project(backend, h, DType::BF16, *h.mixed, *fc_embedding, *h.proj, 1,
                 g.hidden, g.hidden)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (!UploadF32(backend, *h.mtp_hidden_branch,
                   std::vector<float>(hidden.begin(), hidden.end()))) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (!RmsNormDevice(backend, *h.rmsnorm, *h.mtp_hidden_branch, *norm_hidden,
                       *h.mtp_hidden_branch, 1, g.hc_dim, cfg.norm_eps)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (!Project(backend, h, DType::BF16, *h.mtp_hidden_branch, *fc_hidden,
                 *h.mtp_streams, g.hc, g.hidden, g.hidden)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    for (std::size_t c = 0; c < g.hc; ++c) {
      if (!backend.CopyD2D(*h.proj, 0, *h.R, c * g.hidden * 4, g.hidden * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
    if (!AddDevice(backend, *h.add, *h.mtp_streams, *h.R, *h.R, g.hc_dim)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  if (auto read = HyperRead(backend, h, cfg, g, *attn_norm, *attn_down,
                            *attn_up, attn_inject, *h.R, *h.mixed,
                            h.inject.get());
      !read) {
    return std::unexpected(read.error());
  }
  if (auto attn = RunMtpAttention(backend, cfg, g, h, w, pos, *h.mixed,
                                  *h.block_out);
      !attn) {
    return std::unexpected(attn.error());
  }
  if (auto write = HyperWrite(backend, h, g, *h.R, *h.block_out, *h.inject);
      !write) {
    return std::unexpected(write.error());
  }
  if (auto read = HyperRead(backend, h, cfg, g, *mlp_norm, *mlp_down, *mlp_up,
                            mlp_inject, *h.R, *h.mixed, h.inject2.get());
      !read) {
    return std::unexpected(read.error());
  }
  if (auto moe = RunMtpMoe(backend, cfg, h, w, *h.mixed); !moe) {
    return std::unexpected(moe.error());
  }
  if (auto write = HyperWrite(backend, h, g, *h.R, *h.block_out, *h.inject2);
      !write) {
    return std::unexpected(write.error());
  }
  if (mtp_hidden_out != nullptr) {
    auto snap = DownloadF32(backend, *h.R);
    if (!snap) {
      return std::unexpected(snap.error());
    }
    *mtp_hidden_out = std::move(*snap);
  }
  // The final mixer read, then the shared output head.
  if (auto read = HyperRead(backend, h, cfg, g, *mix_norm, *mix_down, *mix_up,
                            nullptr, *h.R, *h.mixed, nullptr);
      !read) {
    return std::unexpected(read.error());
  }
  if (!Project(backend, h, (*output)->manifest.dtype, *h.mixed,
               *(*output)->device, *h.logits, 1, cfg.vocab_size, g.hidden)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  backend.Synchronize();
  auto logits = DownloadF32(backend, *h.logits);
  if (!logits) {
    return std::unexpected(logits.error());
  }
  std::uint32_t best = 0;
  for (std::uint32_t i = 1; i < logits->size(); ++i) {
    if ((*logits)[i] > (*logits)[best]) {
      best = i;
    }
  }
  if (logits_out != nullptr) {
    *logits_out = std::move(*logits);
  }
  h.mtp_rows = pos + 1;
  return best;
}

std::size_t MtpDraftRows(const Qwen4ExpState& h) {
  return static_cast<std::size_t>(h.mtp_rows);
}

void MtpDraftTruncate(Qwen4ExpState& h, std::size_t rows) {
  h.mtp_rows = rows;
}

std::expected<const MtpWeights*, StatusCode>
Qwen4ExpArchitecture::SharedMtpWeights(Backend& backend,
                                       const Model& model) const {
  std::lock_guard<std::mutex> lock(mtp_mutex_);
  if (shared_mtp_weights_ == nullptr) {
    if (model.MtpPath().empty()) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    if (const auto* log = backend.Diagnostics()) {
      log->Info("mtp", "loading shared MTP head from '" + model.MtpPath() +
                           "' on device " +
                           std::to_string(backend.DeviceIndex()));
    }
    auto weights = LoadMtpWeights(backend, model.MtpPath());
    if (!weights) {
      return std::unexpected(weights.error());
    }
    shared_mtp_weights_ = std::make_unique<MtpWeights>(std::move(*weights));
  }
  return shared_mtp_weights_.get();
}

}  // namespace tessera::models::qwen4exp
