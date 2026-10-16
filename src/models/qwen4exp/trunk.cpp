#include "models/qwen4exp/internal.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/loaders/shard.hpp"
#include "core/numerics/ngram.hpp"
#include "core/numerics/qsa.hpp"
#include "core/profile.hpp"
#include "core/sampling.hpp"
#include "models/qwen4exp/architecture.hpp"
#include "models/qwen4exp/mtp.hpp"
#include "models/qwen4exp/ple_table.hpp"
#include "models/qwen4exp/state.hpp"

// The qwen4exp trunk: the gated residual stack ("hyper-connection") wraps a
// gated-delta linear mixer or a sparse QSA attention, then a sparse
// mixture-of-experts feed-forward. The PLE n-gram block updates the stack
// before layer 1. See docs/QWEN_FLASH_NEXT.md.

namespace tessera::models::qwen4exp {

namespace detail = ::tessera::core::detail;

using detail::AddDevice;
using detail::DownloadF32;
using detail::GatherEmbeddingDevice;
using detail::GemmFor;
using detail::L2NormDevice;
using detail::NeedWeight;
using detail::NeedWeightAny;
using detail::ProjectDevice;
using detail::RmsNormDevice;
using detail::RopeDevice;
using detail::SsmGateDevice;
using detail::UploadF32;

namespace {

// QSA paged KV capacity and geometry (the served artifact's contract).
// The capacity must cover the context; raise it together with --context.
// The QSA paged-cache capacity is the model context, floored at this and
// rounded to a page, so a short context keeps the old buffer sizes and a
// long one (up to the model maximum) covers every position.
constexpr std::size_t kQsaMinCells = 8192;
constexpr std::size_t kQsaPageSize = 4;
constexpr std::size_t kQsaIdxBlock = 4;
constexpr std::size_t kQsaNRot = 64;
constexpr std::size_t kPleHeads = 16;  // (ngram_size - 1) * heads_per_ngram

std::uint32_t Grid(std::size_t elements) {
  return static_cast<std::uint32_t>((elements + 255) / 256);
}

std::uint32_t EpsBits(double eps) {
  const float f = static_cast<float>(eps);
  std::uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  return bits;
}

// QSA paged-cache cell count: the model context, floored at the minimum and
// rounded up to a page, so the page table and the pooled index divide evenly.
std::size_t QsaCapacity(std::size_t context) {
  std::size_t cells = context < kQsaMinCells ? kQsaMinCells : context;
  const std::size_t rem = cells % kQsaPageSize;
  if (rem != 0) {
    cells += kQsaPageSize - rem;
  }
  return cells;
}

}  // namespace

std::expected<Qwen4ExpGeometry, StatusCode> DeriveGeometry(
    const TransformerConfig& cfg) {
  Qwen4ExpGeometry g;
  g.hidden = cfg.hidden_dim;
  g.heads = cfg.attention.heads;
  g.kv_heads = cfg.attention.kv_heads;
  g.head_dim = cfg.attention.head_dim;
  g.rope_dim = cfg.attention.rope_dim;
  g.rope_theta = cfg.attention.rope_theta;
  g.hc = cfg.hyper_connection.count;
  g.hc_lr = cfg.hyper_connection.low_rank;
  g.gdn_head_k_dim = cfg.ssm.state_size;
  g.gdn_num_k_heads = cfg.ssm.group_count;
  g.gdn_value_dim = cfg.ssm.inner_size;
  g.gdn_num_v_heads = cfg.ssm.time_step_rank;
  g.gdn_width = cfg.ssm.conv_kernel;
  g.idx_heads = cfg.indexer.heads;
  g.idx_dim = cfg.indexer.key_length;
  g.idx_top_k = cfg.indexer.top_k;
  if (g.hidden == 0 || g.heads == 0 || g.kv_heads == 0 || g.head_dim == 0 ||
      g.rope_dim == 0 || g.hc == 0 || g.hc_lr == 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (g.hc % 2 != 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  g.q_dim = g.heads * g.head_dim;
  g.kv_dim = g.kv_heads * g.head_dim;
  g.hc_dim = g.hc * g.hidden;
  if (g.gdn_head_k_dim == 0 || g.gdn_num_k_heads == 0 ||
      g.gdn_value_dim == 0 || g.gdn_num_v_heads == 0 || g.gdn_width == 0 ||
      g.gdn_value_dim % g.gdn_num_v_heads != 0 ||
      g.gdn_num_v_heads % g.gdn_num_k_heads != 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  g.gdn_key_dim = g.gdn_head_k_dim * g.gdn_num_k_heads;
  g.gdn_head_v_dim = g.gdn_value_dim / g.gdn_num_v_heads;
  g.gdn_factor = g.gdn_num_v_heads / g.gdn_num_k_heads;
  g.gdn_conv_dim = g.gdn_key_dim * 2 + g.gdn_value_dim;
  if (g.idx_heads == 0 || g.idx_dim == 0 || g.idx_top_k == 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return g;
}

std::expected<void, StatusCode> Project(Backend& backend, Qwen4ExpState& h,
                                        DType dtype, const Buffer& a,
                                        const Buffer& w, Buffer& out,
                                        std::size_t m, std::size_t n,
                                        std::size_t k) {
  auto gemm = GemmFor(backend, h.gemms, dtype);
  if (!gemm) {
    return std::unexpected(gemm.error());
  }
  return ProjectDevice(backend, **gemm, a, w, out, m, n, k);
}

// One gated residual read: normalize the stack into the sub-block input
// `mixed`, and (when `w_inject` is non-null) the per-stream injection.
std::expected<void, StatusCode> HyperRead(
    Backend& backend, Qwen4ExpState& h, const TransformerConfig& cfg,
    const Qwen4ExpGeometry& g, const Buffer& w_norm, const Buffer& w_down,
    const Buffer& w_up, const Buffer* w_inject, Buffer& r, Buffer& mixed,
    Buffer* inject) {
  const std::uint64_t eps = EpsBits(cfg.norm_eps);
  KernelLaunch norm;
  norm.grid_x = static_cast<std::uint32_t>(g.hc);
  norm.block_x = 256;
  norm.buffers = {&r, const_cast<Buffer*>(&w_norm), h.hc_xn.get(),
                  h.hc_xq.get()};
  norm.scalars = {g.hidden, g.hc, eps};
  if (!backend.LaunchKernel(*h.gr_norm, norm)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  KernelLaunch down;
  down.grid_x = static_cast<std::uint32_t>(g.hc_lr);
  // Sixteen warps shorten the per-output reduction; the kernel's block
  // reduce is sized for them.
  down.block_x = 512;
  down.buffers = {h.hc_xq.get(), const_cast<Buffer*>(&w_down), h.hc_lq.get()};
  down.scalars = {g.hc_dim, g.hc_lr, g.hc};
  if (!backend.LaunchKernel(*h.gr_down, down)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  KernelLaunch gate;
  // One warp per output element: block 256 holds 8 warps, so the grid is
  // ceil(hc_dim / 8). The warp reads the weight row coalesced.
  gate.grid_x = static_cast<std::uint32_t>((g.hc_dim + 7) / 8);
  gate.block_x = 256;
  gate.buffers = {h.hc_lq.get(), const_cast<Buffer*>(&w_up), h.hc_xn.get(),
                  h.hc_gated.get()};
  gate.scalars = {g.hc_dim, g.hc_lr};
  if (!backend.LaunchKernel(*h.gr_gate, gate)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  KernelLaunch mean;
  mean.grid_x = Grid(g.hidden);
  mean.block_x = 256;
  mean.buffers = {h.hc_gated.get(), &mixed};
  mean.scalars = {g.hidden, g.hc};
  if (!backend.LaunchKernel(*h.gr_mean, mean)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (w_inject != nullptr) {
    KernelLaunch inj;
    inj.grid_x = static_cast<std::uint32_t>(g.hc);
    inj.block_x = 256;
    inj.buffers = {h.hc_xq.get(), const_cast<Buffer*>(w_inject), inject};
    inj.scalars = {g.hc_dim, g.hc};
    if (!backend.LaunchKernel(*h.gr_inject, inj)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  return {};
}

// One gated residual write: fold `block_out` back into the stack with the
// per-stream injection.
std::expected<void, StatusCode> HyperWrite(Backend& backend, Qwen4ExpState& h,
                                           const Qwen4ExpGeometry& g, Buffer& r,
                                           const Buffer& block_out,
                                           const Buffer& inject) {
  KernelLaunch write;
  write.grid_x = Grid(g.hc_dim);
  write.block_x = 256;
  write.buffers = {&r, &block_out, &inject, &r};
  write.scalars = {g.hidden, g.hc};
  if (!backend.LaunchKernel(*h.gr_write, write)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

namespace {

std::expected<void, StatusCode> LoadKernel(Backend& backend,
                                           std::unique_ptr<Kernel>& slot,
                                           const char* name) {
  if (slot != nullptr) {
    return {};
  }
  auto kernel = backend.LoadKernel(name, {});
  if (!kernel) {
    return std::unexpected(kernel.error());
  }
  slot = std::move(*kernel);
  return {};
}

std::expected<void, StatusCode> Alloc(Backend& backend,
                                      std::unique_ptr<Buffer>& slot,
                                      std::size_t bytes, MemoryKind memory) {
  auto buf = backend.AllocateBuffer(bytes, memory);
  if (!buf) {
    return std::unexpected(buf.error());
  }
  slot = std::move(*buf);
  return {};
}

}  // namespace

namespace {

// A gated-delta (GDN) layer: project qkv, gate, alpha and beta from the
// mixed activation, run the causal conv, the L2 norms, the recurrence and
// the gated output norm, then project the block output.
std::expected<void, StatusCode> RunGdnLayer(Backend& backend,
                                            const Model& model,
                                            const TransformerConfig& cfg,
                                            const Qwen4ExpGeometry& g,
                                            Qwen4ExpState& h, std::size_t layer,
                                            const Buffer& mixed,
                                            Buffer& block_out) {
  const std::string base = "blk." + std::to_string(layer) + ".";
  auto w_qkv = NeedWeightAny(model, base + "attn_qkv.weight");
  auto w_gate = NeedWeightAny(model, base + "attn_gate.weight");
  auto w_alpha = NeedWeightAny(model, base + "ssm_alpha.weight");
  auto w_beta = NeedWeightAny(model, base + "ssm_beta.weight");
  auto w_a = NeedWeight(model, base + "ssm_a", DType::F32);
  auto w_dt = NeedWeight(model, base + "ssm_dt.bias", DType::F32);
  auto w_conv = NeedWeight(model, base + "ssm_conv1d.weight", DType::F32);
  auto w_norm = NeedWeight(model, base + "ssm_norm.weight", DType::F32);
  auto w_out = NeedWeightAny(model, base + "ssm_out.weight");
  if (!w_qkv || !w_gate || !w_alpha || !w_beta || !w_a || !w_dt || !w_conv ||
      !w_norm || !w_out) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (!Project(backend, h, (*w_qkv)->manifest.dtype, mixed,
               *(*w_qkv)->device, *h.qkv, 1, g.gdn_conv_dim, g.hidden) ||
      !Project(backend, h, (*w_gate)->manifest.dtype, mixed,
               *(*w_gate)->device, *h.z, 1, g.gdn_value_dim, g.hidden) ||
      !Project(backend, h, (*w_alpha)->manifest.dtype, mixed,
               *(*w_alpha)->device, *h.alpha_raw, 1, g.gdn_num_v_heads,
               g.hidden) ||
      !Project(backend, h, (*w_beta)->manifest.dtype, mixed,
               *(*w_beta)->device, *h.beta_raw, 1, g.gdn_num_v_heads, g.hidden) ||
      !SsmGateDevice(backend, *h.ssm_gate, *(*w_a)->device, *(*w_dt)->device,
                     *h.alpha_raw, *h.beta_raw, *h.alpha, *h.beta,
                     g.gdn_num_v_heads)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  Qwen4ExpState::GdnLayer& st = h.gdn[layer];
  if (!detail::Conv1dStateDevice(backend, *h.conv1d_state, *h.qkv,
                                 *(*w_conv)->device, *st.conv_hist, *h.q_l,
                                 *h.k_l, *h.v_l, g.gdn_conv_dim, g.gdn_width,
                                 g.gdn_key_dim)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (!L2NormDevice(backend, *h.l2norm, *h.q_l, *h.q_l, g.gdn_num_k_heads,
                    g.gdn_head_k_dim, 1e-6,
                    1.0 / std::sqrt(static_cast<double>(g.gdn_head_k_dim))) ||
      !L2NormDevice(backend, *h.l2norm, *h.k_l, *h.k_l, g.gdn_num_k_heads,
                    g.gdn_head_k_dim, 1e-6)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (!detail::RepeatHeadsDevice(backend, *h.repeat_heads, *h.q_l, *h.q_exp,
                                 g.gdn_num_v_heads, g.gdn_head_k_dim,
                                 g.gdn_factor) ||
      !detail::RepeatHeadsDevice(backend, *h.repeat_heads, *h.k_l, *h.k_exp,
                                 g.gdn_num_v_heads, g.gdn_head_k_dim,
                                 g.gdn_factor)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (!detail::DeltaStepHeadsDevice(backend, *h.delta_step_heads, *st.state,
                                    *h.k_exp, *h.v_l, *h.q_exp, *h.core,
                                    *h.alpha, *h.beta, g.gdn_num_v_heads,
                                    g.gdn_head_k_dim, g.gdn_head_v_dim)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (!detail::GdnOutNormDevice(backend, *h.gdn_out_norm, *h.core, *h.z,
                                *(*w_norm)->device, *h.out, g.gdn_num_v_heads,
                                g.gdn_head_v_dim, cfg.norm_eps)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (!Project(backend, h, (*w_out)->manifest.dtype, *h.out, *(*w_out)->device,
               block_out, 1, g.hidden, g.gdn_value_dim)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

// A sparse QSA attention layer: project q (with gate), k and v; norm and
// rotate q and k; append k and v to the paged FP16 cache; run the indexer
// (raw key, pooling, scoring, top-k), gather the selected cells, attend,
// apply the gate and project the output.
std::expected<void, StatusCode> RunQsaLayer(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    const Qwen4ExpGeometry& g, Qwen4ExpState& h, std::size_t layer,
    std::uint64_t pos, const Buffer& mixed, Buffer& block_out) {
  const std::string base = "blk." + std::to_string(layer) + ".";
  auto w_q = NeedWeightAny(model, base + "attn_q.weight");
  auto w_k = NeedWeightAny(model, base + "attn_k.weight");
  auto w_v = NeedWeightAny(model, base + "attn_v.weight");
  auto w_o = NeedWeightAny(model, base + "attn_output.weight");
  auto w_qn = NeedWeight(model, base + "attn_q_norm.weight", DType::F32);
  auto w_kn = NeedWeight(model, base + "attn_k_norm.weight", DType::F32);
  auto w_idxk = NeedWeightAny(model, base + "indexer.k_proj.weight");
  auto w_idxq = NeedWeightAny(model, base + "indexer.q_proj.weight");
  auto w_iqn = NeedWeight(model, base + "indexer.q_norm.weight", DType::F32);
  auto w_ikn = NeedWeight(model, base + "indexer.k_norm.weight", DType::F32);
  if (!w_q || !w_k || !w_v || !w_o || !w_qn || !w_kn || !w_idxk || !w_idxq ||
      !w_iqn || !w_ikn) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  Qwen4ExpState::QsaLayer& st = h.qsa[layer];
  if (!Project(backend, h, (*w_q)->manifest.dtype, mixed, *(*w_q)->device,
               *h.q_full, 1, g.heads * 2 * g.head_dim, g.hidden) ||
      !detail::QGateSplitDevice(backend, *h.qgate_split, *h.q_full, *h.q_attn,
                                *h.gate_buf, g.heads, g.head_dim) ||
      !Project(backend, h, (*w_k)->manifest.dtype, mixed, *(*w_k)->device,
               *h.kcur, 1, g.kv_dim, g.hidden) ||
      !Project(backend, h, (*w_v)->manifest.dtype, mixed, *(*w_v)->device,
               *h.vcur, 1, g.kv_dim, g.hidden) ||
      !RmsNormDevice(backend, *h.rmsnorm, *h.q_attn, *(*w_qn)->device,
                     *h.q_attn, g.heads, g.head_dim, cfg.norm_eps) ||
      !RmsNormDevice(backend, *h.rmsnorm, *h.kcur, *(*w_kn)->device, *h.kcur,
                     g.kv_heads, g.head_dim, cfg.norm_eps) ||
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
    append.buffers = {st.k_pool.get(), st.v_pool.get(), st.page_table.get(),
                      h.kcur.get(), h.vcur.get()};
    append.scalars = {pos, g.kv_heads, g.head_dim, kQsaPageSize};
    if (!backend.LaunchKernel(*h.kv_append, append)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  // Indexer: the raw key, its pooling/rotation, the query, then scoring.
  if (!Project(backend, h, (*w_idxk)->manifest.dtype, mixed,
               *(*w_idxk)->device, *h.idx_raw, 1, g.idx_dim, g.hidden) ||
      !Project(backend, h, (*w_idxq)->manifest.dtype, mixed,
               *(*w_idxq)->device, *h.q_idx, 1, g.idx_heads * g.idx_dim,
               g.hidden) ||
      !RmsNormDevice(backend, *h.rmsnorm, *h.q_idx, *(*w_iqn)->device, *h.q_idx,
                     g.idx_heads, g.idx_dim, cfg.norm_eps) ||
      !RopeDevice(backend, *h.rope, *h.q_idx, g.idx_heads, g.idx_dim, kQsaNRot,
                  pos, g.rope_theta)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  const std::int32_t pos_i = static_cast<std::int32_t>(pos);
  {
    KernelLaunch idx;
    idx.grid_x = 1;
    idx.block_x = 256;
    idx.buffers = {h.idx_raw.get(),        (*w_ikn)->device.get(),
                   st.idx_tail.get(),      st.idx_dead.get(),
                   st.idx_pooled.get(),    st.block_pos.get(),
                   st.idx_cos.get(),       st.idx_sin.get()};
    idx.scalars = {0,
                   EpsBits(cfg.norm_eps),
                   g.idx_dim,
                   kQsaIdxBlock,
                   kQsaNRot,
                   static_cast<std::uint64_t>(pos_i)};
    if (!backend.LaunchKernel(*h.indexer_key_append, idx)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  std::int32_t step[core::kQsaStepCount] = {0, 0, 0, 0};
  core::QsaStepFill(std::span<std::int32_t>(step, core::kQsaStepCount), pos_i,
                    kQsaIdxBlock, g.idx_top_k);
  const std::uint64_t n_kv = static_cast<std::uint64_t>(step[1]);
  const std::uint64_t n_bid = static_cast<std::uint64_t>(step[2]);
  const std::uint64_t width = static_cast<std::uint64_t>(step[3]);
  {
    KernelLaunch score;
    score.grid_x = Grid(n_kv);
    score.block_x = 256;
    score.buffers = {st.idx_pooled.get(), h.q_idx.get(), h.idx_bias.get(),
                     h.cell_scores.get()};
    score.scalars = {g.idx_heads, g.idx_dim, kQsaIdxBlock, n_bid, n_kv};
    if (!backend.LaunchKernel(*h.qsa_index, score)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  {
    KernelLaunch topk;
    topk.grid_x = 1;
    topk.block_x = 256;
    topk.buffers = {h.cell_scores.get(), h.ids.get()};
    topk.scalars = {n_kv, width};
    if (!backend.LaunchKernel(*h.qsa_topk, topk)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  const std::uint64_t n_ids = width;
  {
    KernelLaunch gather;
    gather.grid_x = Grid(n_ids * g.kv_heads * g.head_dim);
    gather.block_x = 256;
    gather.buffers = {st.k_pool.get(),      st.v_pool.get(),
                      st.page_table.get(),  h.ids.get(),
                      h.k_scratch.get(),    h.v_scratch.get()};
    gather.scalars = {n_ids, g.kv_heads, g.head_dim, kQsaPageSize};
    if (!backend.LaunchKernel(*h.kv_gather, gather)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  {
    KernelLaunch attend;
    attend.grid_x = static_cast<std::uint32_t>(g.heads);
    attend.block_x = 256;
    attend.buffers = {h.q_attn.get(), h.k_scratch.get(), h.v_scratch.get(),
                      h.q_attn.get()};
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
  if (!Project(backend, h, (*w_o)->manifest.dtype, *h.attn_in,
               *(*w_o)->device, block_out, 1, g.hidden, g.heads * g.head_dim)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

// The PLE block: hash the last n-gram, gather the sixteen table rows,
// project a key and a value, gate the value per stream, run the grouped
// norm and the dilated causal convolution, and add the result to the
// residual stack.
std::expected<void, StatusCode> RunPleBlock(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    const Qwen4ExpGeometry& g, Qwen4ExpState& h, std::uint32_t token,
    Buffer& r) {
  if (h.ple_table == nullptr) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  const std::string base = "blk.1.";
  auto w_key = NeedWeightAny(model, base + "ple_key.weight");
  auto w_value = NeedWeightAny(model, base + "ple_value.weight");
  auto w_conv = NeedWeight(model, base + "ple_conv1d.weight", DType::F16);
  auto w_nk = NeedWeight(model, base + "ple_norm_key.weight", DType::F32);
  auto w_nq = NeedWeight(model, base + "ple_norm_query.weight", DType::F32);
  auto w_nc = NeedWeight(model, base + "ple_norm_conv.weight", DType::F32);
  if (!w_key || !w_value || !w_conv || !w_nk || !w_nq || !w_nc) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const std::int32_t tok = static_cast<std::int32_t>(token);
  // The two predecessors, oldest first (-1 when absent).
  std::int32_t prev[2] = {h.ple_prev.size() > 1 ? h.ple_prev[0] : -1,
                          h.ple_prev.size() > 1 ? h.ple_prev[1] : -1};
  std::vector<std::uint32_t> rows(kPleHeads);
  auto hashed = core::NgramRows(std::span<const std::int32_t>(&tok, 1),
                                std::span<const std::int32_t>(prev, 2),
                                cfg.ple, std::span<std::uint32_t>(rows));
  if (!hashed) {
    return std::unexpected(hashed.error());
  }
  std::vector<float> emb(kPleHeads * kPleRowElements);
  auto gathered = h.ple_table->Gather(std::span<const std::uint32_t>(rows),
                                      std::span<float>(emb));
  if (!gathered) {
    return std::unexpected(gathered.error());
  }
  if (!UploadF32(backend, *h.ple_emb, emb)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  h.ple_prev.clear();
  h.ple_prev.push_back(prev[1]);
  h.ple_prev.push_back(tok);
  const std::size_t hist = (cfg.ple.conv_kernel - 1) * cfg.ple.ngram_size;
  const std::uint64_t eps = EpsBits(cfg.norm_eps);
  if (!Project(backend, h, (*w_key)->manifest.dtype, *h.ple_emb,
               *(*w_key)->device, *h.ple_key, 1, g.hc_dim, g.hidden) ||
      !Project(backend, h, (*w_value)->manifest.dtype, *h.ple_emb,
               *(*w_value)->device, *h.ple_value, 1, g.hidden, g.hidden)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  auto rows_norm = [&](const Buffer& x, const Buffer& w, Buffer& y)
      -> std::expected<void, StatusCode> {
    KernelLaunch n;
    n.grid_x = static_cast<std::uint32_t>(g.hc);
    n.block_x = 256;
    n.buffers = {&x, &w, &y};
    n.scalars = {g.hc, g.hidden, eps};
    return backend.LaunchKernel(*h.rmsnorm_rows, n);
  };
  if (!rows_norm(*h.ple_key, *(*w_nk)->device, *h.ple_key) ||
      !rows_norm(r, *(*w_nq)->device, *h.ple_query)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  {
    KernelLaunch gate;
    gate.grid_x = static_cast<std::uint32_t>(g.hc);
    gate.block_x = 256;
    gate.buffers = {h.ple_key.get(), h.ple_query.get(), h.inject.get()};
    gate.scalars = {g.hidden, g.hc};
    if (!backend.LaunchKernel(*h.ple_gate, gate)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  {
    KernelLaunch bcast;
    bcast.grid_x = Grid(g.hc_dim);
    bcast.block_x = 256;
    bcast.buffers = {h.ple_value.get(), h.inject.get(), h.ple_gated.get()};
    bcast.scalars = {g.hc_dim, g.hidden};
    if (!backend.LaunchKernel(*h.ple_broadcast, bcast)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  if (!rows_norm(*h.ple_gated, *(*w_nc)->device, *h.ple_normalized)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  {
    KernelLaunch conv;
    conv.grid_x = Grid(g.hc_dim);
    conv.block_x = 256;
    conv.buffers = {h.ple_hist.get(),  h.ple_normalized.get(),
                    (*w_conv)->device.get(), &r,
                    h.ple_gated.get(),  h.ple_conv_out.get(),
                    &r};
    conv.scalars = {g.hc_dim, hist, cfg.ple.conv_kernel};
    if (!backend.LaunchKernel(*h.ple_conv, conv)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  {
    KernelLaunch adv;
    adv.grid_x = Grid(g.hc_dim);
    adv.block_x = 256;
    adv.buffers = {h.ple_hist.get(), h.ple_normalized.get()};
    adv.scalars = {g.hc_dim, hist};
    if (!backend.LaunchKernel(*h.ple_hist_advance, adv)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  return {};
}

std::int32_t ArgMaxToken(std::span<const float> logits) {
  std::int32_t best = 0;
  float best_value = logits.empty() ? 0.0f : logits[0];
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > best_value) {
      best_value = logits[i];
      best = static_cast<std::int32_t>(i);
    }
  }
  return best;
}

}  // namespace

// The output head: a hyper-connection read of the stack with the
// output_hc_* weights, then the vocabulary projection into `h.logits`.
std::expected<void, StatusCode> RunTrunkHead(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    const Qwen4ExpGeometry& g, Qwen4ExpState& h) {
  auto w_norm = NeedWeight(model, "output_hc_norm.weight", DType::F32);
  auto w_down = NeedWeight(model, "output_hc_down.weight", DType::BF16);
  auto w_up = NeedWeight(model, "output_hc_up.weight", DType::BF16);
  auto w_out = NeedWeightAny(model, "output.weight");
  if (!w_norm || !w_down || !w_up || !w_out) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (auto read = HyperRead(backend, h, cfg, g, *(*w_norm)->device,
                            *(*w_down)->device, *(*w_up)->device, nullptr, *h.R,
                            *h.mixed, nullptr);
      !read) {
    return read;
  }
  return Project(backend, h, (*w_out)->manifest.dtype, *h.mixed,
                 *(*w_out)->device, *h.logits, 1, cfg.vocab_size, g.hidden);
}

// The mixer half of a trunk block: the attention hyper-connection read, the
// QSA or gated-delta mixer, the attention write back into the stack, and the
// FFN hyper-connection read. Leaves the FFN input in `h.mixed` and the
// per-stream injection in `h.inject2` for the FFN half. The PLE block runs
// here when this layer carries it.
std::expected<void, StatusCode> RunTrunkLayerMixer(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    const Qwen4ExpGeometry& g, Qwen4ExpState& h, std::size_t l,
    std::uint64_t pos, std::uint32_t token, bool ple_layer) {
  if (auto ensured = model.EnsureLayerResident(l); !ensured) {
    return std::unexpected(ensured.error());
  }
  if (ple_layer && l == 1) {
    if (auto ple = RunPleBlock(backend, model, cfg, g, h, token, *h.R);
        !ple) {
      return ple;
    }
  }
  const std::string base = "blk." + std::to_string(l) + ".";
  auto an = NeedWeight(model, base + "hc_attn_norm.weight", DType::F32);
  auto ad = NeedWeight(model, base + "hc_attn_down.weight", DType::BF16);
  auto au = NeedWeight(model, base + "hc_attn_up.weight", DType::BF16);
  auto ai = NeedWeight(model, base + "hc_attn_inject.weight", DType::BF16);
  if (!an || !ad || !au || !ai) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (auto read = HyperRead(backend, h, cfg, g, *(*an)->device, *(*ad)->device,
                            *(*au)->device, (*ai)->device.get(), *h.R, *h.mixed,
                            h.inject.get());
      !read) {
    return read;
  }
  std::expected<void, StatusCode> mixer =
      cfg.IsFullAttentionLayer(l)
          ? RunQsaLayer(backend, model, cfg, g, h, l, pos, *h.mixed,
                        *h.block_out)
          : RunGdnLayer(backend, model, cfg, g, h, l, *h.mixed, *h.block_out);
  if (!mixer) {
    return mixer;
  }
  if (auto write = HyperWrite(backend, h, g, *h.R, *h.block_out, *h.inject);
      !write) {
    return write;
  }
  auto fn = NeedWeight(model, base + "hc_ffn_norm.weight", DType::F32);
  auto fd = NeedWeight(model, base + "hc_ffn_down.weight", DType::BF16);
  auto fu = NeedWeight(model, base + "hc_ffn_up.weight", DType::BF16);
  auto fi = NeedWeight(model, base + "hc_ffn_inject.weight", DType::BF16);
  if (!fn || !fd || !fu || !fi) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return HyperRead(backend, h, cfg, g, *(*fn)->device, *(*fd)->device,
                   *(*fu)->device, (*fi)->device.get(), *h.R, *h.mixed,
                   h.inject2.get());
}

// The FFN half: run the MoE on `h.mixed` and fold the block output into the
// stack with `h.inject2`. The mixer half must have run first.
std::expected<void, StatusCode> RunTrunkLayerFfn(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    const Qwen4ExpGeometry& g, Qwen4ExpState& h, std::size_t l,
    core::Profile* deep_profile) {
  core::PhaseScope ffn_scope(deep_profile, core::Phase::kFfn);
  if (auto moe = RunMoeFfn(backend, model, cfg, h, l, *h.mixed); !moe) {
    return moe;
  }
  return HyperWrite(backend, h, g, *h.R, *h.block_out, *h.inject2);
}

// One trunk block on the current residual stack `h.R`, at absolute position
// `pos`, with `token` feeding the PLE block when this layer carries it. The
// mixer is QSA on a full-attention layer and gated-delta otherwise. Used by
// the single-token Forward and by the batched forward, which drives one row
// at a time so both share this body.
std::expected<void, StatusCode> RunTrunkLayer(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    const Qwen4ExpGeometry& g, Qwen4ExpState& h, std::size_t l,
    std::uint64_t pos, std::uint32_t token, bool ple_layer) {
  core::Profile* const deep_profile =
      h.profile != nullptr && h.profile->Deep() ? h.profile : nullptr;
  core::PhaseScope layer_scope(
      deep_profile, cfg.IsFullAttentionLayer(l) ? core::Phase::kFullLayer
                                                : core::Phase::kLinearLayer);
  if (auto mixer = RunTrunkLayerMixer(backend, model, cfg, g, h, l, pos, token,
                                      ple_layer);
      !mixer) {
    return mixer;
  }
  if (auto ffn = RunTrunkLayerFfn(backend, model, cfg, g, h, l, deep_profile);
      !ffn) {
    return ffn;
  }
  if (deep_profile != nullptr) {
    backend.Synchronize();  // profile attribution only
  }
  return {};
}

// Derive the last split shard's path from a "...-NNNNN-of-MMMMM.gguf"
// name; the PLE table lives there. Empty when the name has no split
// suffix.
std::string PleTablePath(const std::string& path) {
  const std::size_t of_pos = path.rfind("-of-");
  if (of_pos == std::string::npos) {
    return {};
  }
  std::size_t first = of_pos;
  while (first > 0 &&
         std::isdigit(static_cast<unsigned char>(path[first - 1]))) {
    --first;
  }
  const std::size_t total_start = of_pos + 4;
  std::size_t total_end = total_start;
  while (total_end < path.size() &&
         std::isdigit(static_cast<unsigned char>(path[total_end]))) {
    ++total_end;
  }
  if (first == of_pos || total_end == total_start) {
    return {};
  }
  // Replace the shard index with the total, keeping "-of-MMMMM.gguf".
  return path.substr(0, first) +
         path.substr(total_start, total_end - total_start) +
         path.substr(of_pos);
}

std::expected<Qwen4ExpState*, StatusCode> InitState(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    const Qwen4ExpGeometry& g, core::DecodeCache& cache) {
  Qwen4ExpState& h = State(cache);
  if (h.ready) {
    return &h;
  }
  // A sharded stage allocates per-layer state only for the layers it owns;
  // the vectors stay sized to the trunk so indexing by the global layer
  // number is unchanged.
  const core::LayerRange range =
      core::ShardLayerRange(model.Shard(), cfg.layers);
  if (auto r = LoadKernel(backend, h.rmsnorm, "rmsnorm"); !r)
    return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.rmsnorm_rows, "rmsnorm_rows"); !r)
    return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.add, "add"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.silu_mul, "silu_mul"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.sigmoid_gate, "sigmoid_gate"); !r)
    return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.qgate_split, "qgate_split"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.rope, "rope"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.cast_f16_f32, "cast_f16_f32"); !r)
    return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.gr_norm, "gr_norm"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.gr_down, "gr_down"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.gr_gate, "gr_gate"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.gr_mean, "gr_mean"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.gr_inject, "gr_inject"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.gr_write, "gr_write"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.conv1d_state, "conv1d_state"); !r)
    return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.l2norm, "l2norm"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.ssm_gate, "ssm_gate"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.repeat_heads, "repeat_heads"); !r)
    return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.delta_step_heads, "delta_step_heads");
      !r) {
    return std::unexpected(r.error());
  }
  if (auto r = LoadKernel(backend, h.rmsnorm_gated, "rmsnorm_gated"); !r)
    return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.gdn_out_norm, "gdn_out_norm"); !r)
    return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.kv_append, "kv_append"); !r)
    return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.indexer_key_append, "indexer_key_append");
      !r) {
    return std::unexpected(r.error());
  }
  if (auto r = LoadKernel(backend, h.qsa_index, "qsa_index"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.qsa_topk, "qsa_topk"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.kv_gather, "kv_gather"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.qsa_attend, "qsa_attend"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.qsa_gate_apply, "qsa_gate_apply"); !r)
    return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.ple_gate, "ple_gate"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.ple_broadcast, "ple_broadcast"); !r)
    return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.ple_conv, "ple_conv"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.ple_hist_advance, "ple_hist_advance");
      !r) {
    return std::unexpected(r.error());
  }
  if (auto r = LoadKernel(backend, h.moe_gate, "moe_gate"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.moe_scale_add, "moe_scale_add"); !r)
    return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.moe_topk, "top_k_rows"); !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.moe_gate_up_iq2s, "moe_experts_gate_up_iq2s");
      !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.moe_gate_up_iq3xxs, "moe_experts_gate_up_iq3xxs");
      !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.moe_gate_up_iq3s, "moe_experts_gate_up_iq3s");
      !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.moe_gate_up_iq4xs, "moe_experts_gate_up_iq4xs");
      !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.moe_down_iq4nl, "moe_experts_down_iq4nl");
      !r) return std::unexpected(r.error());
  if (auto r = LoadKernel(backend, h.moe_down_q20, "moe_experts_down_q20");
      !r) return std::unexpected(r.error());

  const std::size_t f = 4;
  auto make = [&](std::unique_ptr<Buffer>& slot,
                  std::size_t bytes) -> std::expected<void, StatusCode> {
    return Alloc(backend, slot, bytes, MemoryKind::Device);
  };
  const std::size_t hc_dim = g.hc_dim;
  const std::size_t q_dim = g.heads * g.head_dim;
  const std::size_t id_cap = cfg.indexer.top_k + kQsaIdxBlock - 1;
  // The QSA paged cache covers the model context, so a 262144-token run
  // allocates 262144 cells and a short run keeps the old buffers.
  const std::size_t qsa_cells = QsaCapacity(model.MaxContextLength());
  const std::size_t qsa_pages = qsa_cells / kQsaPageSize;
  const std::size_t qsa_pooled = qsa_cells / kQsaIdxBlock + 2;
  if (!make(h.x, g.hidden * f) || !make(h.proj, g.hidden * f) ||
      !make(h.logits, cfg.vocab_size * f) || !make(h.R, hc_dim * f) ||
      !make(h.mixed, g.hidden * f) || !make(h.block_out, g.hidden * f) ||
      !make(h.inject, g.hc * f) || !make(h.inject2, g.hc * f) ||
      !make(h.hc_xn, hc_dim * f) || !make(h.hc_xq, hc_dim * f) ||
      !make(h.hc_lq, g.hc_lr * f) || !make(h.hc_gated, hc_dim * f) ||
      !make(h.qkv, g.gdn_conv_dim * f) || !make(h.z, g.gdn_value_dim * f) ||
      !make(h.alpha_raw, g.gdn_num_v_heads * f) ||
      !make(h.beta_raw, g.gdn_num_v_heads * f) ||
      !make(h.alpha, g.gdn_num_v_heads * f) ||
      !make(h.beta, g.gdn_num_v_heads * f) ||
      !make(h.q_l, g.gdn_key_dim * f) || !make(h.k_l, g.gdn_key_dim * f) ||
      !make(h.q_exp, g.gdn_num_v_heads * g.gdn_head_k_dim * f) ||
      !make(h.k_exp, g.gdn_num_v_heads * g.gdn_head_k_dim * f) ||
      !make(h.v_l, g.gdn_value_dim * f) || !make(h.core, g.gdn_value_dim * f) ||
      !make(h.out, g.gdn_value_dim * f) || !make(h.q_full, q_dim * 2 * f) ||
      !make(h.q_attn, q_dim * f) || !make(h.gate_buf, q_dim * f) ||
      !make(h.kcur, g.kv_dim * f) || !make(h.vcur, g.kv_dim * f) ||
      !make(h.idx_raw, g.idx_dim * f) ||
      !make(h.q_idx, g.idx_heads * g.idx_dim * f) ||
      !make(h.cell_scores, qsa_cells * f) || !make(h.ids, id_cap * f) ||
      !make(h.k_scratch, id_cap * g.kv_heads * g.head_dim * 2) ||
      !make(h.v_scratch, id_cap * g.kv_heads * g.head_dim * 2) ||
      !make(h.idx_bias, qsa_pooled * f) ||
      !make(h.attn_f16, q_dim * 2) || !make(h.attn_in, q_dim * f) ||
      !make(h.embed_ids, 4)) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  {
    std::vector<float> zeros(qsa_pooled, 0.0f);
    if (!UploadF32(backend, *h.idx_bias, zeros)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }

  const std::size_t pool_bytes =
      qsa_pages * g.kv_heads * kQsaPageSize * g.head_dim * 2;
  // An offloaded cache keeps the paged K/V pools in system RAM; everything
  // else stays in device memory. The MTP draft shares the choice.
  h.kv_memory =
      cache.kv_offload ? MemoryKind::HostVisible : MemoryKind::Device;
  // The paged K/V pools start empty. The indexer can select a cell in the
  // current block that the append has not written yet, so an unwritten pool
  // must read as zeros, never as whatever the allocator returned.
  const std::vector<std::byte> pool_zeros(pool_bytes, std::byte{0});
  const std::size_t cos_elems = qsa_cells * (kQsaNRot / 2);
  std::vector<float> cos_tab(cos_elems);
  std::vector<float> sin_tab(cos_elems);
  for (std::size_t p = 0; p < qsa_cells; ++p) {
    for (std::size_t j = 0; j < kQsaNRot / 2; ++j) {
      const double freq =
          std::pow(g.rope_theta, -2.0 * static_cast<double>(j) /
                                    static_cast<double>(kQsaNRot));
      const double angle = static_cast<double>(p) * freq;
      cos_tab[p * (kQsaNRot / 2) + j] = static_cast<float>(std::cos(angle));
      sin_tab[p * (kQsaNRot / 2) + j] = static_cast<float>(std::sin(angle));
    }
  }
  h.gdn.resize(cfg.layers);
  h.qsa.resize(cfg.layers);
  std::vector<float> zeros_state;
  for (std::size_t l = range.first; l < range.last; ++l) {
    if (cfg.IsFullAttentionLayer(l)) {
      Qwen4ExpState::QsaLayer& st = h.qsa[l];
      if (!Alloc(backend, st.k_pool, pool_bytes, h.kv_memory) ||
          !Alloc(backend, st.v_pool, pool_bytes, h.kv_memory) ||
          !make(st.page_table, qsa_pages * 4) ||
          !make(st.idx_tail, (kQsaIdxBlock - 1) * g.idx_dim * f) ||
          !make(st.idx_dead, g.idx_dim * f) ||
          !make(st.idx_pooled, qsa_pooled * g.idx_dim * f) ||
          !make(st.idx_cos, cos_elems * f) || !make(st.idx_sin, cos_elems * f) ||
          !make(st.block_pos, 4)) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      if (!backend.CopyH2D(*st.k_pool, pool_zeros) ||
          !backend.CopyH2D(*st.v_pool, pool_zeros)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      std::vector<std::int32_t> table(qsa_pages);
      for (std::size_t p = 0; p < qsa_pages; ++p) {
        table[p] = static_cast<std::int32_t>(p);
      }
      if (!backend.CopyH2D(
              *st.page_table,
              std::span<const std::byte>(
                  reinterpret_cast<const std::byte*>(table.data()),
                  table.size() * sizeof(table[0])))) {
        return std::unexpected(StatusCode::DeviceError);
      }
      if (!UploadF32(backend, *st.idx_cos, cos_tab) ||
          !UploadF32(backend, *st.idx_sin, sin_tab)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      std::vector<float> zeros_pooled(qsa_pooled * g.idx_dim, 0.0f);
      std::vector<float> zeros_tail((kQsaIdxBlock - 1) * g.idx_dim, 0.0f);
      std::vector<float> zeros_dead(g.idx_dim, 0.0f);
      if (!UploadF32(backend, *st.idx_pooled, zeros_pooled) ||
          !UploadF32(backend, *st.idx_tail, zeros_tail) ||
          !UploadF32(backend, *st.idx_dead, zeros_dead)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    } else {
      Qwen4ExpState::GdnLayer& st = h.gdn[l];
      const std::size_t hist = g.gdn_conv_dim * (g.gdn_width - 1);
      const std::size_t state =
          g.gdn_num_v_heads * g.gdn_head_k_dim * g.gdn_head_v_dim;
      if (!make(st.conv_hist, hist * f) || !make(st.state, state * f)) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      zeros_state.assign(hist, 0.0f);
      if (!UploadF32(backend, *st.conv_hist, zeros_state)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      zeros_state.assign(state, 0.0f);
      if (!UploadF32(backend, *st.state, zeros_state)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
  }

  // The PLE block runs once, before layer 1, so only the stage that owns
  // layer 1 allocates the n-gram table and its history.
  const bool owns_ple =
      !cfg.ple.layers.empty() && range.first <= 1 && range.last > 1;
  if (owns_ple) {
    const std::string table_path = PleTablePath(model.Path());
    if (!table_path.empty()) {
      auto table = PleTable::Open(table_path);
      if (table) {
        h.ple_table = std::shared_ptr<PleTable>(std::move(*table));
      }
    }
    const std::size_t hist = (cfg.ple.conv_kernel - 1) * cfg.ple.ngram_size;
    if (!make(h.ple_hist, hc_dim * hist * f) || !make(h.ple_key, hc_dim * f) ||
        !make(h.ple_query, hc_dim * f) || !make(h.ple_value, g.hidden * f) ||
        !make(h.ple_gated, hc_dim * f) ||
        !make(h.ple_normalized, hc_dim * f) ||
        !make(h.ple_conv_out, hc_dim * f) || !make(h.ple_emb, g.hidden * f)) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    zeros_state.assign(hc_dim * hist, 0.0f);
    if (!UploadF32(backend, *h.ple_hist, zeros_state)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  h.ready = true;
  return &h;
}

std::expected<void, StatusCode> Qwen4ExpArchitecture::EnsureReady(
    Backend& backend, const Model& model, core::DecodeCache& cache) const {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  auto geom = DeriveGeometry(*config);
  if (!geom) {
    return std::unexpected(geom.error());
  }
  auto state = InitState(backend, model, *config, *geom, cache);
  if (!state) {
    return std::unexpected(state.error());
  }
  // Load the MTP head now, before the expert cache sizes itself to the free
  // device memory, so the MTP weights get their share of VRAM first.
  if (!model.MtpPath().empty()) {
    if (auto mtp = EnsureMtpReady(backend, model, *config, *geom, **state); !mtp) {
      return std::unexpected(mtp.error());
    }
  }
  return {};
}

std::expected<void, StatusCode> Qwen4ExpArchitecture::Forward(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out,
    const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture, const Buffer* embedding) const {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const TransformerConfig& cfg = *config;
  if (token >= cfg.vocab_size) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto geom = DeriveGeometry(cfg);
  if (!geom) {
    return std::unexpected(geom.error());
  }
  const Qwen4ExpGeometry& g = *geom;
  auto ready = EnsureReady(backend, model, cache);
  if (!ready) {
    return ready;
  }
  Qwen4ExpState& h = State(cache);
  h.profile = cache.profile;
  const ModelShard& shard = model.Shard();
  const core::LayerRange range = core::ShardLayerRange(shard, cfg.layers);
  if (range.first >= range.last) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  // The first stage gathers the token embedding (or a caller embedding). A
  // later stage receives the previous stage's whole residual stack through
  // `embedding`: the hyper-connection keeps `hc` parallel streams across
  // the split, so the handoff is hc*hidden and is loaded whole.
  if (shard.enabled && !shard.first_stage) {
    if (embedding == nullptr || embedding->Size() < g.hc_dim * 4 ||
        !backend.CopyD2D(*embedding, 0, *h.R, 0, g.hc_dim * 4)) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
  } else {
    if (embedding != nullptr) {
      if (!backend.CopyD2D(*embedding, 0, *h.x, 0, g.hidden * 4)) {
        return std::unexpected(StatusCode::InvalidArgument);
      }
    } else {
      auto embed = NeedWeightAny(model, "token_embd.weight");
      if (!embed) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      std::vector<float> row(g.hidden);
      if (auto gathered =
              detail::GatherEmbedding(backend, **embed, token, g.hidden, row);
          !gathered) {
        return std::unexpected(gathered.error());
      }
      if (!UploadF32(backend, *h.x, row)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
    for (std::size_t c = 0; c < g.hc; ++c) {
      if (!backend.CopyD2D(*h.x, 0, *h.R, c * g.hidden * 4, g.hidden * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
  }
  const std::uint64_t pos = h.position;
  bool ple_layer = false;
  for (const std::uint64_t l : cfg.ple.layers) {
    if (l == 1) {
      ple_layer = true;
    }
  }
  for (std::size_t l = range.first; l < range.last; ++l) {
    if (auto layer = RunTrunkLayer(backend, model, cfg, g, h, l, pos, token,
                                   ple_layer);
        !layer) {
      return layer;
    }
    // Capture the residual stack after the boundary layer for the next
    // stage's handoff (the pipeline asks for one boundary layer).
    if (capture_layers != nullptr && capture != nullptr) {
      for (std::size_t i = 0; i < capture_layers->size(); ++i) {
        if ((*capture_layers)[i] == l &&
            !backend.CopyD2D(*h.R, 0, *(*capture)[i], 0, g.hc_dim * 4)) {
          return std::unexpected(StatusCode::DeviceError);
        }
      }
    }
  }
  if (hidden_out != nullptr) {
    // The MTP head consumes the final multi-stream residual stack (hc *
    // hidden), not the single mixed activation: the anchor pairs the token
    // with the stack at the position before it.
    auto snap = DownloadF32(backend, *h.R);
    if (!snap) {
      return std::unexpected(snap.error());
    }
    *hidden_out = std::move(*snap);
  }
  ++h.position;
  return {};
}

std::expected<std::vector<float>, StatusCode> Qwen4ExpArchitecture::Logits(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out,
    const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture, const Buffer* embedding) const {
  core::Profile* const profile = cache.profile;
  {
    core::PhaseScope scope(profile, core::Phase::kForward);
    if (auto forward = Forward(backend, model, cache, token, hidden_out,
                               capture_layers, capture, embedding);
        !forward) {
      return std::unexpected(forward.error());
    }
    if (profile != nullptr) {
      backend.Synchronize();  // profile attribution only
    }
  }
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const TransformerConfig& cfg = *config;
  auto geom = DeriveGeometry(cfg);
  if (!geom) {
    return std::unexpected(geom.error());
  }
  Qwen4ExpState& h = State(cache);
  {
    core::PhaseScope scope(profile, core::Phase::kHead);
    if (auto head = RunTrunkHead(backend, model, cfg, *geom, h); !head) {
      return std::unexpected(head.error());
    }
    if (profile != nullptr) {
      backend.Synchronize();  // profile attribution only
    }
  }
  core::PhaseScope scope(profile, core::Phase::kDownload);
  backend.Synchronize();
  return DownloadF32(backend, *h.logits);
}

std::expected<void, StatusCode> Qwen4ExpArchitecture::ForwardBatch(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::span<const std::uint32_t> tokens, std::vector<float>* logits_out,
    std::vector<float>* hidden_out, bool all_logits, const Buffer* embeddings,
    const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture, bool snapshot_state) const {
  if (tokens.empty()) {
    return {};
  }
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const std::size_t vocab = config->vocab_size;
  if (logits_out != nullptr && all_logits) {
    logits_out->clear();
    logits_out->reserve(tokens.size() * vocab);
  }
  // A multi-row batch runs the layers once each over all rows, so the paged
  // experts stage once per layer instead of once per row. A verify that
  // requests state snapshots also takes the batched path, so a one-row
  // verify still records the pre/post recurrent state for rollback. The
  // plain single-token case keeps the per-token path.
  if (tokens.size() > 1 || snapshot_state) {
    auto ready = EnsureReady(backend, model, cache);
    if (!ready) {
      return ready;
    }
    return ForwardTrunkBatch(backend, model, cache, tokens, logits_out,
                             hidden_out, all_logits, embeddings, capture_layers,
                             capture, snapshot_state);
  }
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const bool last = i + 1 == tokens.size();
    if (all_logits || last) {
      auto logits = Logits(backend, model, cache, tokens[i], hidden_out,
                           capture_layers, capture, embeddings);
      if (!logits) {
        return std::unexpected(logits.error());
      }
      if (logits_out != nullptr) {
        logits_out->insert(logits_out->end(), logits->begin(), logits->end());
      }
    } else {
      auto forward = Forward(backend, model, cache, tokens[i], hidden_out,
                             capture_layers, capture, embeddings);
      if (!forward) {
        return forward;
      }
    }
  }
  return {};
}

std::expected<void, StatusCode> Qwen4ExpArchitecture::TruncateTarget(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::size_t keep_rows) const {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  auto geom = DeriveGeometry(*config);
  if (!geom) {
    return std::unexpected(geom.error());
  }
  const Qwen4ExpGeometry& g = *geom;
  Qwen4ExpState& h = State(cache);
  if (h.batch_gdn_state.empty()) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  if (keep_rows < h.batch_snapshot_start ||
      keep_rows - h.batch_snapshot_start > h.batch_snapshot_rows) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t slot =
      static_cast<std::size_t>(keep_rows - h.batch_snapshot_start);
  const std::size_t state_len =
      g.gdn_num_v_heads * g.gdn_head_k_dim * g.gdn_head_v_dim;
  const std::size_t hist_len = g.gdn_conv_dim * (g.gdn_width - 1);
  // A sharded stage owns only its layers; the full-attention layers roll
  // back through the position alone because their paged cache is bounded
  // by the position, so only the recurrent state needs restoring.
  const core::LayerRange range =
      core::ShardLayerRange(model.Shard(), config->layers);
  for (std::size_t l = range.first; l < range.last; ++l) {
    if (config->IsFullAttentionLayer(l) || h.batch_gdn_state[l] == nullptr) {
      continue;
    }
    if (!backend.CopyD2D(*h.batch_gdn_state[l], slot * state_len * 4,
                         *h.gdn[l].state, 0, state_len * 4) ||
        !backend.CopyD2D(*h.batch_gdn_conv[l], slot * hist_len * 4,
                         *h.gdn[l].conv_hist, 0, hist_len * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  h.position = keep_rows;
  return {};
}

std::expected<DraftVerification, StatusCode> Qwen4ExpArchitecture::Verify(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::span<const std::uint32_t> draft, std::span<const float> prefix_logits,
    std::optional<std::uint32_t> anchor, std::vector<float>* hidden_out,
    const std::vector<std::size_t>* capture_layers,
    std::vector<Buffer*>* capture, const SampledVerify* sampled) const {
  if (!anchor.has_value() && prefix_logits.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const bool sample = sampled != nullptr && sampled->sampling != nullptr &&
                      sampled->rng != nullptr;
  std::vector<std::uint32_t> hist;
  if (sample) {
    hist.assign(sampled->history.begin(), sampled->history.end());
  }
  DraftVerification result;
  if (anchor.has_value()) {
    auto logits = Logits(backend, model, cache, *anchor, hidden_out,
                         capture_layers, capture, nullptr);
    if (!logits) {
      return std::unexpected(logits.error());
    }
    result.logits = std::move(*logits);
  } else {
    result.logits.assign(prefix_logits.begin(), prefix_logits.end());
  }
  const bool rejection =
      sample && sampled->draft_probs != nullptr &&
      sampled->draft_vocab == result.logits.size() &&
      sampled->draft_probs->size() >= draft.size() * sampled->draft_vocab;
  bool all = true;
  for (std::size_t i = 0; i < draft.size(); ++i) {
    const std::uint32_t token = draft[i];
    if (rejection) {
      const std::vector<double> target = core::BuildDenseDistribution(
          result.logits, *sampled->sampling, hist);
      const std::span<const double> q(sampled->draft_probs->data() +
                                          i * sampled->draft_vocab,
                                      sampled->draft_vocab);
      const core::RejectionResult step =
          core::RejectionStep(target, q, token, *sampled->rng);
      if (!step.accepted) {
        result.next_token = step.token;
        all = false;
        break;
      }
    } else {
      const std::uint32_t y =
          sample ? core::SampleToken(result.logits, *sampled->sampling, hist,
                                     *sampled->rng)
                 : static_cast<std::uint32_t>(ArgMaxToken(result.logits));
      if (y != token) {
        result.next_token = y;
        all = false;
        break;
      }
    }
    ++result.accepted;
    hist.push_back(token);
    auto logits = Logits(backend, model, cache, token, hidden_out,
                         capture_layers, capture, nullptr);
    if (!logits) {
      return std::unexpected(logits.error());
    }
    result.logits = std::move(*logits);
  }
  if (all) {
    if (rejection) {
      result.next_token = core::SampleFromDense(
          core::BuildDenseDistribution(result.logits, *sampled->sampling, hist),
          *sampled->rng);
    } else {
      result.next_token =
          sample ? core::SampleToken(result.logits, *sampled->sampling, hist,
                                     *sampled->rng)
                 : static_cast<std::uint32_t>(ArgMaxToken(result.logits));
    }
  }
  return result;
}

std::expected<std::uint32_t, StatusCode> Qwen4ExpArchitecture::Draft(
    Backend& backend, const Model& model, core::DecodeCache& cache,
    std::span<const float> hidden, std::uint32_t token, std::uint64_t pos,
    std::vector<float>* mtp_hidden_out, std::vector<float>* logits_out) const {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  auto geom = DeriveGeometry(*config);
  if (!geom) {
    return std::unexpected(geom.error());
  }
  auto ready = EnsureReady(backend, model, cache);
  if (!ready) {
    return std::unexpected(ready.error());
  }
  Qwen4ExpState& h = State(cache);
  return RunMtpDraft(backend, model, *config, *geom, h, hidden, token, pos,
                     mtp_hidden_out, logits_out);
}

std::size_t Qwen4ExpArchitecture::DraftRows(
    const core::DecodeCache& cache) const {
  if (cache.arch == nullptr) {
    return 0;
  }
  return MtpDraftRows(*static_cast<const Qwen4ExpState*>(cache.arch.get()));
}

void Qwen4ExpArchitecture::DraftTruncate(core::DecodeCache& cache,
                                         std::size_t rows) const {
  if (cache.arch != nullptr) {
    MtpDraftTruncate(*static_cast<Qwen4ExpState*>(cache.arch.get()), rows);
  }
}

}  // namespace tessera::models::qwen4exp




