#include "spec/dflash2_context.hpp"

#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/attention.hpp"
#include "core/numerics/gemm.hpp"
#include "core/numerics/norm.hpp"

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

}  // namespace tessera::spec
