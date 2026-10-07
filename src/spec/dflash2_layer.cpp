#include "spec/dflash2_layer.hpp"

#include <memory>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/norm.hpp"
#include "spec/dflash2_attention.hpp"
#include "spec/dflash2_context.hpp"
#include "spec/dflash2_mlp.hpp"

namespace tessera::spec {

namespace {

using core::detail::AddDevice;

// Allocate one fp32 scratch buffer of `elements` floats.
std::expected<std::unique_ptr<Buffer>, StatusCode> Scratch(Backend& backend,
                                                          std::size_t elements) {
  return backend.AllocateBuffer(elements * 4, MemoryKind::Device);
}

}  // namespace

std::expected<void, StatusCode> DraftLayerRef(
    std::span<const float> hidden, std::span<const float> residual,
    const DraftLayerWeights& w, std::span<float> out,
    std::span<float> residual_out, std::size_t rows, std::size_t hidden_dim,
    std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
    std::size_t ffn, std::size_t taps, std::size_t group_size,
    std::size_t block_size, std::size_t window, std::uint64_t pos_base,
    double theta, float eps, std::span<const float> context_hidden) {
  const std::size_t elements = rows * hidden_dim;
  if (hidden.size() != elements || out.size() != elements ||
      residual_out.size() != elements ||
      (!residual.empty() && residual.size() != elements)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const bool first = residual.empty();
  std::vector<float> pre(elements);
  if (first) {
    pre = std::vector<float>(hidden.begin(), hidden.end());
  } else if (!core::AddRef(hidden, residual, std::span<float>(pre), elements)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::size_t ctx = 0;
  std::vector<float> ctx_k;
  std::vector<float> ctx_v;
  if (!context_hidden.empty()) {
    const std::size_t kv_dim = kv_heads * head_dim;
    if (context_hidden.size() % hidden_dim != 0) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    ctx = context_hidden.size() / hidden_dim;
    ctx_k.resize(ctx * kv_dim);
    ctx_v.resize(ctx * kv_dim);
    auto context = DraftContextKvRef(
        context_hidden, w.hidden_norm, w.k_w, w.v_w, w.k_norm_w,
        std::span<float>(ctx_k), std::span<float>(ctx_v), ctx, hidden_dim,
        kv_heads, head_dim, pos_base, theta, eps);
    if (!context) {
      return std::unexpected(context.error());
    }
  }
  std::vector<float> attn_out(elements);
  auto attn = DraftAttentionRef(
      std::span<const float>(pre), w.input_norm, w.attn_conv_proj,
      w.attn_conv_base, w.q_w, w.k_w, w.v_w, w.o_w, w.q_norm_w, w.k_norm_w,
      std::span<float>(attn_out), rows, hidden_dim, heads, kv_heads, head_dim,
      taps, group_size, block_size, window, pos_base, theta, eps,
      std::span<const float>(ctx_k), std::span<const float>(ctx_v), ctx);
  if (!attn) {
    return std::unexpected(attn.error());
  }
  std::vector<float> post(elements);
  if (!core::AddRef(std::span<const float>(attn_out), std::span<const float>(pre),
                    std::span<float>(post), elements)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto mlp = DraftMlpRef(std::span<const float>(post), w.post_norm,
                         w.mlp_conv_proj, w.mlp_conv_base, w.gate_w, w.up_w,
                         w.down_w, out, rows, hidden_dim, ffn, taps,
                         group_size, block_size, eps);
  if (!mlp) {
    return std::unexpected(mlp.error());
  }
  for (std::size_t i = 0; i < elements; ++i) {
    residual_out[i] = post[i];
  }
  return {};
}

std::expected<void, StatusCode> DraftLayerDevice(
    Backend& backend, const Kernel& rmsnorm, const Kernel& gemm,
    const Kernel& conv, const Kernel& rope, const Kernel& attention,
    const Kernel& silu, const Kernel& add, const Buffer& hidden,
    const Buffer* residual,
    const DraftLayerBuffers& w, Buffer& out, Buffer& residual_out,
    std::size_t rows, std::size_t hidden_dim, std::size_t heads,
    std::size_t kv_heads, std::size_t head_dim, std::size_t ffn,
    std::size_t taps, std::size_t group_size, std::size_t block_size,
    std::size_t window, std::uint64_t pos_base, double theta, float eps,
    const Buffer* context_hidden, std::size_t ctx) {
  if (rows == 0 || hidden_dim == 0 || heads == 0 || kv_heads == 0 ||
      head_dim == 0 || ffn == 0 || taps == 0 || group_size == 0 ||
      block_size == 0 || hidden_dim % group_size != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t num_groups = hidden_dim / group_size;
  const std::size_t proj_n = 2 * taps * num_groups;
  const std::size_t q_dim = heads * head_dim;
  const std::size_t kv_dim = kv_heads * head_dim;
  const std::size_t elements = rows * hidden_dim;
  std::vector<std::unique_ptr<Buffer>> buffers;
  auto alloc = [&](std::size_t count) -> Buffer* {
    auto buffer = Scratch(backend, count);
    if (!buffer) {
      return nullptr;
    }
    buffers.push_back(std::move(*buffer));
    return buffers.back().get();
  };
  Buffer* pre = alloc(elements);
  Buffer* attn_out = alloc(elements);
  Buffer* post = alloc(elements);
  Buffer* axn = alloc(elements);
  Buffer* aproj = alloc(rows * proj_n);
  Buffer* ah1 = alloc(elements);
  Buffer* q = alloc(rows * q_dim);
  Buffer* k = alloc(rows * kv_dim);
  Buffer* v = alloc(rows * kv_dim);
  Buffer* attn = alloc(rows * q_dim);
  Buffer* oproj = alloc(elements);
  Buffer* aside = alloc(taps * hidden_dim);
  Buffer* mxn = alloc(elements);
  Buffer* mproj = alloc(rows * proj_n);
  Buffer* mh1 = alloc(elements);
  Buffer* gate = alloc(rows * ffn);
  Buffer* up = alloc(rows * ffn);
  Buffer* act = alloc(rows * ffn);
  Buffer* down = alloc(elements);
  Buffer* mside = alloc(taps * hidden_dim);
  if (pre == nullptr || attn_out == nullptr || post == nullptr || axn == nullptr ||
      aproj == nullptr || ah1 == nullptr || q == nullptr || k == nullptr ||
      v == nullptr || attn == nullptr || oproj == nullptr || aside == nullptr ||
      mxn == nullptr || mproj == nullptr || mh1 == nullptr || gate == nullptr ||
      up == nullptr || act == nullptr || down == nullptr || mside == nullptr) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  if (residual == nullptr) {
    if (!backend.CopyD2D(hidden, 0, *pre, 0, elements * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  } else if (!AddDevice(backend, add, hidden, *residual, *pre, elements)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  Buffer* ctx_k = nullptr;
  Buffer* ctx_v = nullptr;
  if (ctx > 0) {
    if (context_hidden == nullptr || w.hidden_norm == nullptr) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    Buffer* ctx_normed = alloc(ctx * hidden_dim);
    ctx_k = alloc(ctx * kv_dim);
    ctx_v = alloc(ctx * kv_dim);
    if (ctx_normed == nullptr || ctx_k == nullptr || ctx_v == nullptr) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    auto context = DraftContextKvDevice(
        backend, rmsnorm, gemm, rope, *ctx_normed, *context_hidden,
        *w.hidden_norm, *w.k_w, *w.v_w, *w.k_norm_w, *ctx_k, *ctx_v, ctx,
        hidden_dim, kv_heads, head_dim, pos_base, theta, eps);
    if (!context) {
      return std::unexpected(context.error());
    }
  }
  auto attn_status = DraftAttentionDevice(
      backend, rmsnorm, gemm, conv, rope, attention, *axn, *aproj, *ah1, *q,
      *k, *v, *attn, *oproj, *aside, *pre, *w.input_norm, *w.attn_conv_proj,
      *w.attn_conv_base, *w.q_w, *w.k_w, *w.v_w, *w.o_w, *w.q_norm_w,
      *w.k_norm_w, *attn_out, rows, hidden_dim, heads, kv_heads, head_dim,
      taps, group_size, block_size, window, pos_base, theta, eps, ctx_k, ctx_v,
      ctx);
  if (!attn_status) {
    return std::unexpected(attn_status.error());
  }
  if (!AddDevice(backend, add, *attn_out, *pre, *post, elements)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  auto mlp_status = DraftMlpDevice(
      backend, rmsnorm, gemm, conv, silu, *mxn, *mproj, *mh1, *gate, *up, *act,
      *down, *mside, *post, *w.post_norm, *w.mlp_conv_proj, *w.mlp_conv_base,
      *w.gate_w, *w.up_w, *w.down_w, out, rows, hidden_dim, ffn, taps,
      group_size, block_size, eps);
  if (!mlp_status) {
    return std::unexpected(mlp_status.error());
  }
  if (!backend.CopyD2D(*post, 0, residual_out, 0, elements * 4)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

}  // namespace tessera::spec
