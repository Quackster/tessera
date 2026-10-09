#include "spec/dflash2_context.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/attention.hpp"
#include "core/numerics/concat.hpp"
#include "core/numerics/gemm.hpp"
#include "core/numerics/norm.hpp"
#include "core/numerics/quant.hpp"
#include "spec/dflash2_fuse.hpp"

namespace tessera::spec {

namespace {

using core::detail::ProjectDevice;
using core::detail::RmsNormDevice;
using core::detail::RopeDevice;

}  // namespace

std::expected<void, StatusCode> DraftContextKvRef(
    std::span<const float> context, std::span<const float> hidden_norm,
    std::span<const float> k_proj, std::span<const float> v_proj,
    std::span<const float> k_norm, std::span<float> out_k,
    std::span<float> out_v, std::size_t ctx, std::size_t hidden_dim,
    std::size_t kv_heads, std::size_t head_dim, std::uint64_t pos_base,
    double theta, float eps) {
  if (ctx == 0 || hidden_dim == 0 || kv_heads == 0 || head_dim == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t kv_dim = kv_heads * head_dim;
  if (context.size() != ctx * hidden_dim ||
      hidden_norm.size() != hidden_dim ||
      k_proj.size() != kv_dim * hidden_dim ||
      v_proj.size() != kv_dim * hidden_dim || k_norm.size() != head_dim ||
      out_k.size() != ctx * kv_dim || out_v.size() != ctx * kv_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> normed(ctx * hidden_dim);
  if (!core::RmsNormRef(context, hidden_norm, std::span<float>(normed), ctx,
                        hidden_dim, eps) ||
      !core::GemmF32Ref(std::span<const float>(normed), k_proj, out_k, ctx,
                        kv_dim, hidden_dim) ||
      !core::GemmF32Ref(std::span<const float>(normed), v_proj, out_v, ctx,
                        kv_dim, hidden_dim) ||
      !core::RmsNormRef(std::span<const float>(out_k), k_norm, out_k,
                        ctx * kv_heads, head_dim, eps) ||
      !core::RopeRef(out_k, ctx, kv_heads, head_dim, head_dim, pos_base,
                     theta)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return {};
}

std::expected<void, StatusCode> DraftContextKvDevice(
    Backend& backend, const Kernel& rmsnorm, const Kernel& gemm,
    const Kernel& rope, Buffer& normed, const Buffer& context,
    const Buffer& hidden_norm, const Buffer& k_proj, const Buffer& v_proj,
    const Buffer& k_norm, Buffer& out_k, Buffer& out_v, std::size_t ctx,
    std::size_t hidden_dim, std::size_t kv_heads, std::size_t head_dim,
    std::uint64_t pos_base, double theta, float eps) {
  if (ctx == 0 || hidden_dim == 0 || kv_heads == 0 || head_dim == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t kv_dim = kv_heads * head_dim;
  if (normed.Size() < ctx * hidden_dim * 4) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (!RmsNormDevice(backend, rmsnorm, context, hidden_norm, normed, ctx,
                     hidden_dim, eps) ||
      !ProjectDevice(backend, gemm, normed, k_proj, out_k, ctx, kv_dim,
                     hidden_dim) ||
      !ProjectDevice(backend, gemm, normed, v_proj, out_v, ctx, kv_dim,
                     hidden_dim) ||
      !RmsNormDevice(backend, rmsnorm, out_k, k_norm, out_k, ctx * kv_heads,
                     head_dim, eps) ||
      !RopeDevice(backend, rope, out_k, kv_heads, head_dim, head_dim,
                  pos_base, theta, ctx)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

std::expected<void, StatusCode> DraftContextAppendDevice(
    Backend& backend, const Kernel& rmsnorm, const Kernel& gemm,
    const Kernel& rope, const Kernel& concat, const Kernel& quantize,
    DraftContextCache& cache, const std::vector<const Buffer*>& aux,
    const Buffer& fc_w, const std::vector<const Buffer*>& hidden_norm,
    const std::vector<const Buffer*>& k_w,
    const std::vector<const Buffer*>& v_w,
    const std::vector<const Buffer*>& k_norm, std::size_t n,
    std::size_t features, std::size_t new_rows, std::size_t hidden_dim,
    std::size_t kv_heads, std::size_t head_dim, double theta, float eps,
    std::size_t limit) {
  if (n == 0 || features == 0 || new_rows == 0 || hidden_dim == 0 ||
      kv_heads == 0 || head_dim == 0 || aux.size() != n ||
      hidden_norm.size() != n || k_w.size() != n || v_w.size() != n ||
      k_norm.size() != n) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (cache.layers.empty()) {
    cache.layers.resize(n);
  }
  if (cache.layers.size() != n) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t kv_dim = kv_heads * head_dim;
  // Drop the oldest rows so only the most recent `limit` survive.
  if (limit > 0 && cache.rows + new_rows > limit) {
    const std::size_t drop = std::min(cache.rows, cache.rows + new_rows - limit);
    if (drop > 0) {
      for (DraftContextKvBuffers& layer : cache.layers) {
        if (!backend.CopyD2D(*layer.k, drop * kv_dim * 4, *layer.k, 0,
                             (cache.rows - drop) * kv_dim * 4) ||
            !backend.CopyD2D(*layer.v, drop * kv_dim * 4, *layer.v, 0,
                             (cache.rows - drop) * kv_dim * 4)) {
          return std::unexpected(StatusCode::DeviceError);
        }
      }
      cache.rows -= drop;
      cache.base += drop;
    }
  }
  // Grow geometrically.
  if (cache.rows + new_rows > cache.capacity) {
    std::size_t capacity = cache.capacity == 0 ? 16 : cache.capacity;
    while (capacity < cache.rows + new_rows) {
      capacity *= 2;
    }
    for (std::size_t l = 0; l < n; ++l) {
      auto grown_k =
          backend.AllocateBuffer(capacity * kv_dim * 4, MemoryKind::Device);
      auto grown_v =
          backend.AllocateBuffer(capacity * kv_dim * 4, MemoryKind::Device);
      if (!grown_k || !grown_v) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      if (cache.rows > 0 &&
          (!backend.CopyD2D(*cache.layers[l].k, 0, **grown_k, 0,
                            cache.rows * kv_dim * 4) ||
           !backend.CopyD2D(*cache.layers[l].v, 0, **grown_v, 0,
                            cache.rows * kv_dim * 4))) {
        return std::unexpected(StatusCode::DeviceError);
      }
      cache.layers[l].k = std::move(*grown_k);
      cache.layers[l].v = std::move(*grown_v);
    }
    cache.capacity = capacity;
  }
  // Stage the concatenated aux as [n][new_rows][features].
  const std::size_t width = n * features;
  auto aux_stage =
      backend.AllocateBuffer(n * new_rows * features * 4, MemoryKind::Device);
  auto fused =
      backend.AllocateBuffer(new_rows * hidden_dim * 4, MemoryKind::Device);
  auto concat_scratch =
      backend.AllocateBuffer(new_rows * width * 4, MemoryKind::Device);
  auto qscale = backend.AllocateBuffer(new_rows * 4, MemoryKind::Device);
  auto normed =
      backend.AllocateBuffer(new_rows * hidden_dim * 4, MemoryKind::Device);
  auto tmp_k = backend.AllocateBuffer(new_rows * kv_dim * 4, MemoryKind::Device);
  auto tmp_v = backend.AllocateBuffer(new_rows * kv_dim * 4, MemoryKind::Device);
  if (!aux_stage || !fused || !concat_scratch || !qscale || !normed || !tmp_k ||
      !tmp_v) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!backend.CopyD2D(*aux[i], 0, **aux_stage, i * new_rows * features * 4,
                         new_rows * features * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  if (auto ok = DraftFuseDevice(backend, gemm, concat, quantize,
                                **concat_scratch, **qscale, **aux_stage, fc_w,
                                **fused, n, new_rows, features, hidden_dim);
      !ok) {
    return std::unexpected(ok.error());
  }
  const std::uint64_t pos_base =
      static_cast<std::uint64_t>(cache.base + cache.rows);
  for (std::size_t l = 0; l < n; ++l) {
    if (auto kv = DraftContextKvDevice(
            backend, rmsnorm, gemm, rope, **normed, **fused, *hidden_norm[l],
            *k_w[l], *v_w[l], *k_norm[l], **tmp_k, **tmp_v, new_rows,
            hidden_dim, kv_heads, head_dim, pos_base, theta, eps);
        !kv) {
      return std::unexpected(kv.error());
    }
    if (!backend.CopyD2D(**tmp_k, 0, *cache.layers[l].k,
                         cache.rows * kv_dim * 4, new_rows * kv_dim * 4) ||
        !backend.CopyD2D(**tmp_v, 0, *cache.layers[l].v,
                         cache.rows * kv_dim * 4, new_rows * kv_dim * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  cache.rows += new_rows;
  return {};
}

}  // namespace tessera::spec
