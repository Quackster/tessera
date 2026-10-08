#include "spec/dflash2_attention.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/attention.hpp"
#include "core/numerics/gemm.hpp"
#include "core/numerics/norm.hpp"
#include "spec/dflash2_conv.hpp"

namespace tessera::spec {

namespace {

using core::detail::AttentionDevice;
using core::detail::ProjectDevice;
using core::detail::RmsNormDevice;
using core::detail::RopeDevice;

std::expected<void, StatusCode> Validate(
    std::size_t rows, std::size_t hidden, std::size_t heads,
    std::size_t kv_heads, std::size_t head_dim, std::size_t taps,
    std::size_t group_size, std::size_t block_size) {
  if (rows == 0 || hidden == 0 || heads == 0 || kv_heads == 0 ||
      head_dim == 0 || taps == 0 || group_size == 0 || block_size == 0 ||
      hidden % group_size != 0 || heads % kv_heads != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return {};
}

}  // namespace

std::expected<void, StatusCode> DraftAttentionRef(
    std::span<const float> x, std::span<const float> input_norm_w,
    std::span<const float> conv_proj_w, std::span<const float> conv_base,
    std::span<const float> q_w, std::span<const float> k_w,
    std::span<const float> v_w, std::span<const float> o_w,
    std::span<const float> q_norm_w, std::span<const float> k_norm_w,
    std::span<float> out, std::size_t rows, std::size_t hidden,
    std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
    std::size_t taps, std::size_t group_size, std::size_t block_size,
    std::size_t window, std::uint64_t pos_base, double theta, float eps,
    std::span<const float> context_k, std::span<const float> context_v,
    std::size_t ctx, bool causal) {
  if (auto valid = Validate(rows, hidden, heads, kv_heads, head_dim, taps,
                            group_size, block_size);
      !valid) {
    return valid;
  }
  const std::size_t q_dim = heads * head_dim;
  const std::size_t kv_dim = kv_heads * head_dim;
  const std::size_t num_groups = hidden / group_size;
  const std::size_t proj_n = 2 * taps * num_groups;
  if (x.size() != rows * hidden || out.size() != rows * hidden ||
      input_norm_w.size() != hidden ||
      conv_proj_w.size() != proj_n * hidden ||
      conv_base.size() != 2 * taps * hidden || q_w.size() != q_dim * hidden ||
      k_w.size() != kv_dim * hidden || v_w.size() != kv_dim * hidden ||
      o_w.size() != hidden * q_dim || q_norm_w.size() != head_dim ||
      k_norm_w.size() != head_dim ||
      (ctx > 0 && (context_k.size() != ctx * kv_dim ||
                   context_v.size() != ctx * kv_dim))) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> xn(rows * hidden);
  std::vector<float> proj(rows * proj_n);
  std::vector<float> h1(rows * hidden);
  std::vector<float> q(rows * q_dim);
  std::vector<float> k(rows * kv_dim);
  std::vector<float> v(rows * kv_dim);
  std::vector<float> attn(rows * q_dim);
  std::vector<float> oproj(rows * hidden);
  std::span<const float> base0(conv_base.data(), taps * hidden);
  std::span<const float> base1(conv_base.data() + taps * hidden, taps * hidden);
  const std::uint64_t query_base = pos_base + ctx;
  if (!core::RmsNormRef(x, input_norm_w, std::span<float>(xn), rows, hidden,
                        eps) ||
      !core::GemmF32Ref(std::span<const float>(xn), conv_proj_w,
                        std::span<float>(proj), rows, proj_n, hidden) ||
      !GroupedConvFinishRef(std::span<const float>(xn),
                            std::span<const float>(proj), base0,
                            std::span<float>(h1), rows, hidden, taps,
                            group_size, block_size, 0) ||
      !core::GemmF32Ref(std::span<const float>(h1), q_w, std::span<float>(q),
                        rows, q_dim, hidden) ||
      !core::GemmF32Ref(std::span<const float>(h1), k_w, std::span<float>(k),
                        rows, kv_dim, hidden) ||
      !core::GemmF32Ref(std::span<const float>(h1), v_w, std::span<float>(v),
                        rows, kv_dim, hidden) ||
      !core::RmsNormRef(std::span<const float>(q), q_norm_w,
                        std::span<float>(q), rows * heads, head_dim, eps) ||
      !core::RmsNormRef(std::span<const float>(k), k_norm_w,
                        std::span<float>(k), rows * kv_heads, head_dim, eps) ||
      !core::RopeRef(std::span<float>(q), rows, heads, head_dim, head_dim,
                     query_base, theta) ||
      !core::RopeRef(std::span<float>(k), rows, kv_heads, head_dim, head_dim,
                     query_base, theta)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> full_k;
  std::vector<float> full_v;
  if (ctx > 0) {
    full_k.resize((ctx + rows) * kv_dim);
    full_v.resize((ctx + rows) * kv_dim);
    std::copy(context_k.begin(), context_k.end(), full_k.begin());
    std::copy(context_v.begin(), context_v.end(), full_v.begin());
    std::copy(k.begin(), k.end(), full_k.begin() + ctx * kv_dim);
    std::copy(v.begin(), v.end(), full_v.begin() + ctx * kv_dim);
  }
  std::span<const float> keys =
      ctx > 0 ? std::span<const float>(full_k) : std::span<const float>(k);
  std::span<const float> values =
      ctx > 0 ? std::span<const float>(full_v) : std::span<const float>(v);
  const std::size_t n = ctx + rows;
  if (!core::AttentionRef(std::span<const float>(q), keys, values,
                          std::span<float>(attn), rows, n, heads, kv_heads,
                          head_dim, query_base, window, causal) ||
      !core::GemmF32Ref(std::span<const float>(attn), o_w,
                        std::span<float>(oproj), rows, hidden, q_dim) ||
      !GroupedConvFinishRef(std::span<const float>(oproj),
                            std::span<const float>(proj), base1, out, rows,
                            hidden, taps, group_size, block_size, 1)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return {};
}

std::expected<void, StatusCode> DraftAttentionDevice(
    Backend& backend, const Kernel& rmsnorm, const Kernel& gemm,
    const Kernel& conv, const Kernel& rope, const Kernel& attention,
    Buffer& xn, Buffer& proj, Buffer& h1, Buffer& q, Buffer& k, Buffer& v,
    Buffer& attn, Buffer& oproj, Buffer& side_base, const Buffer& x,
    const Buffer& input_norm_w, const Buffer& conv_proj_w,
    const Buffer& conv_base, const Buffer& q_w, const Buffer& k_w,
    const Buffer& v_w, const Buffer& o_w, const Buffer& q_norm_w,
    const Buffer& k_norm_w, Buffer& out, std::size_t rows, std::size_t hidden,
    std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
    std::size_t taps, std::size_t group_size, std::size_t block_size,
    std::size_t window, std::uint64_t pos_base, double theta, float eps,
    const Buffer* context_k, const Buffer* context_v, std::size_t ctx,
    bool causal) {
  if (auto valid = Validate(rows, hidden, heads, kv_heads, head_dim, taps,
                            group_size, block_size);
      !valid) {
    return valid;
  }
  if (ctx > 0 && (context_k == nullptr || context_v == nullptr)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t q_dim = heads * head_dim;
  const std::size_t kv_dim = kv_heads * head_dim;
  const std::size_t num_groups = hidden / group_size;
  const std::size_t proj_n = 2 * taps * num_groups;
  const std::uint64_t query_base = pos_base + ctx;
  if (!RmsNormDevice(backend, rmsnorm, x, input_norm_w, xn, rows, hidden,
                     eps) ||
      !ProjectDevice(backend, gemm, xn, conv_proj_w, proj, rows, proj_n,
                     hidden) ||
      !GroupedConvFinishDevice(backend, conv, xn, proj, conv_base, side_base,
                               h1, rows, hidden, taps, group_size, block_size,
                               0) ||
      !ProjectDevice(backend, gemm, h1, q_w, q, rows, q_dim, hidden) ||
      !ProjectDevice(backend, gemm, h1, k_w, k, rows, kv_dim, hidden) ||
      !ProjectDevice(backend, gemm, h1, v_w, v, rows, kv_dim, hidden) ||
      !RmsNormDevice(backend, rmsnorm, q, q_norm_w, q, rows * heads, head_dim,
                     eps) ||
      !RmsNormDevice(backend, rmsnorm, k, k_norm_w, k, rows * kv_heads,
                     head_dim, eps) ||
      !RopeDevice(backend, rope, q, heads, head_dim, head_dim, query_base,
                  theta, rows) ||
      !RopeDevice(backend, rope, k, kv_heads, head_dim, head_dim, query_base,
                  theta, rows)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  std::unique_ptr<Buffer> full_k;
  std::unique_ptr<Buffer> full_v;
  const Buffer* keys = &k;
  const Buffer* values = &v;
  std::size_t n = rows;
  if (ctx > 0) {
    auto fk = backend.AllocateBuffer((ctx + rows) * kv_dim * 4,
                                     MemoryKind::Device);
    auto fv = backend.AllocateBuffer((ctx + rows) * kv_dim * 4,
                                     MemoryKind::Device);
    if (!fk || !fv) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    full_k = std::move(*fk);
    full_v = std::move(*fv);
    if (!backend.CopyD2D(*context_k, 0, *full_k, 0, ctx * kv_dim * 4) ||
        !backend.CopyD2D(*context_v, 0, *full_v, 0, ctx * kv_dim * 4) ||
        !backend.CopyD2D(k, 0, *full_k, ctx * kv_dim * 4, rows * kv_dim * 4) ||
        !backend.CopyD2D(v, 0, *full_v, ctx * kv_dim * 4, rows * kv_dim * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    keys = full_k.get();
    values = full_v.get();
    n = ctx + rows;
  }
  if (!AttentionDevice(backend, attention, q, *keys, *values, attn, n, heads,
                       kv_heads, head_dim, query_base, window, rows,
                       /*kv_f16=*/false, causal) ||
      !ProjectDevice(backend, gemm, attn, o_w, oproj, rows, hidden, q_dim) ||
      !GroupedConvFinishDevice(backend, conv, oproj, proj, conv_base,
                               side_base, out, rows, hidden, taps, group_size,
                               block_size, 1)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

}  // namespace tessera::spec
