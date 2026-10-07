#include "core/vision_block.hpp"

#include <memory>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/attention.hpp"
#include "core/numerics/gemm.hpp"
#include "core/numerics/norm.hpp"

namespace tessera::core {

namespace {

using detail::AddDevice;
using detail::AttentionDevice;
using detail::BiasAddDevice;
using detail::LayerNormDevice;
using detail::ProjectDevice;

std::expected<void, StatusCode> GeluDevice(Backend& backend,
                                           const Kernel& kernel,
                                           const Buffer& x, Buffer& y,
                                           std::size_t n) {
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((n + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&x, &y};
  launch.scalars = {n};
  return backend.LaunchKernel(kernel, launch);
}

}  // namespace

std::expected<void, StatusCode> VisionBlockRef(
    std::span<const float> x, const VisionBlockWeights& w,
    std::span<float> out, std::size_t tokens, std::size_t embed,
    std::size_t heads, std::size_t head_dim, std::size_t ffn, float eps) {
  if (tokens == 0 || embed == 0 || heads == 0 || head_dim == 0 || ffn == 0 ||
      heads * head_dim != embed || x.size() != tokens * embed ||
      out.size() != tokens * embed) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> xn(tokens * embed);
  std::vector<float> qkv(tokens * 3 * embed);
  std::vector<float> q(tokens * embed);
  std::vector<float> k(tokens * embed);
  std::vector<float> v(tokens * embed);
  std::vector<float> attn(tokens * embed);
  std::vector<float> oproj(tokens * embed);
  std::vector<float> h(tokens * embed);
  std::vector<float> xn2(tokens * embed);
  std::vector<float> up(tokens * ffn);
  std::vector<float> down(tokens * embed);
  if (!LayerNormRef(x, w.ln1_weight, w.ln1_bias, std::span<float>(xn), tokens,
                    embed, eps) ||
      !GemmF32Ref(std::span<const float>(xn), w.qkv_weight,
                  std::span<float>(qkv), tokens, 3 * embed, embed) ||
      !BiasAddRef(std::span<const float>(qkv), w.qkv_bias,
                  std::span<float>(qkv), tokens, 3 * embed)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t t = 0; t < tokens; ++t) {
    for (std::size_t c = 0; c < embed; ++c) {
      q[t * embed + c] = qkv[t * 3 * embed + c];
      k[t * embed + c] = qkv[t * 3 * embed + embed + c];
      v[t * embed + c] = qkv[t * 3 * embed + 2 * embed + c];
    }
  }
  if (!AttentionRef(std::span<const float>(q), std::span<const float>(k),
                    std::span<const float>(v), std::span<float>(attn), tokens,
                    tokens, heads, heads, head_dim, 0, 0, false) ||
      !GemmF32Ref(std::span<const float>(attn), w.out_weight,
                  std::span<float>(oproj), tokens, embed, embed) ||
      !BiasAddRef(std::span<const float>(oproj), w.out_bias,
                  std::span<float>(oproj), tokens, embed) ||
      !AddRef(x, std::span<const float>(oproj), std::span<float>(h),
              tokens * embed) ||
      !LayerNormRef(std::span<const float>(h), w.ln2_weight, w.ln2_bias,
                    std::span<float>(xn2), tokens, embed, eps) ||
      !GemmF32Ref(std::span<const float>(xn2), w.up_weight,
                  std::span<float>(up), tokens, ffn, embed) ||
      !BiasAddRef(std::span<const float>(up), w.up_bias, std::span<float>(up),
                  tokens, ffn) ||
      !GeluRef(std::span<const float>(up), std::span<float>(up), tokens * ffn) ||
      !GemmF32Ref(std::span<const float>(up), w.down_weight,
                  std::span<float>(down), tokens, embed, ffn) ||
      !BiasAddRef(std::span<const float>(down), w.down_bias,
                  std::span<float>(down), tokens, embed) ||
      !AddRef(std::span<const float>(h), std::span<const float>(down), out,
              tokens * embed)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return {};
}

std::expected<void, StatusCode> VisionBlockDevice(
    Backend& backend, const Kernel& layernorm, const Kernel& gemm,
    const Kernel& attention, const Kernel& gelu, const Kernel& bias_add,
    const Kernel& add, const Buffer& x, const VisionBlockBuffers& w,
    Buffer& out, std::size_t tokens, std::size_t embed, std::size_t heads,
    std::size_t head_dim, std::size_t ffn, float eps) {
  if (tokens == 0 || embed == 0 || heads == 0 || head_dim == 0 || ffn == 0 ||
      heads * head_dim != embed) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t elements = tokens * embed;
  std::vector<std::unique_ptr<Buffer>> owned;
  auto alloc = [&](std::size_t count) -> Buffer* {
    auto buffer = backend.AllocateBuffer(count * 4, MemoryKind::Device);
    if (!buffer) {
      return nullptr;
    }
    owned.push_back(std::move(*buffer));
    return owned.back().get();
  };
  Buffer* xn = alloc(elements);
  Buffer* qkv = alloc(elements * 3);
  Buffer* q = alloc(elements);
  Buffer* k = alloc(elements);
  Buffer* v = alloc(elements);
  Buffer* attn = alloc(elements);
  Buffer* oproj = alloc(elements);
  Buffer* h = alloc(elements);
  Buffer* xn2 = alloc(elements);
  Buffer* up = alloc(tokens * ffn);
  Buffer* down = alloc(elements);
  if (xn == nullptr || qkv == nullptr || q == nullptr || k == nullptr ||
      v == nullptr || attn == nullptr || oproj == nullptr || h == nullptr ||
      xn2 == nullptr || up == nullptr || down == nullptr) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  const auto norm = [&](const Buffer& src, const Buffer* weight,
                        const Buffer* bias, Buffer& dst) {
    return LayerNormDevice(backend, layernorm, src, *weight, *bias, dst, tokens,
                           embed, eps)
        .has_value();
  };
  const auto gemm_to = [&](const Buffer& a, const Buffer* weight, Buffer& c,
                           std::size_t n, std::size_t kk) {
    return ProjectDevice(backend, gemm, a, *weight, c, tokens, n, kk)
        .has_value();
  };
  const auto bias = [&](Buffer& t, const Buffer* b, std::size_t cols) {
    return BiasAddDevice(backend, bias_add, t, *b, t, tokens, cols).has_value();
  };
  // qkv is [tokens, 3*embed]; q/k/v are the three column blocks, so copy
  // each row's slices (a strided slice is not one contiguous range).
  const auto split_qkv = [&]() {
    for (std::size_t t = 0; t < tokens; ++t) {
      const std::size_t row = t * 3 * embed * 4;
      if (!backend.CopyD2D(*qkv, row, *q, t * embed * 4, embed * 4) ||
          !backend.CopyD2D(*qkv, row + embed * 4, *k, t * embed * 4,
                           embed * 4) ||
          !backend.CopyD2D(*qkv, row + 2 * embed * 4, *v, t * embed * 4,
                           embed * 4)) {
        return false;
      }
    }
    return true;
  };
  if (!norm(x, w.ln1_weight, w.ln1_bias, *xn) ||
      !gemm_to(*xn, w.qkv_weight, *qkv, 3 * embed, embed) ||
      !bias(*qkv, w.qkv_bias, 3 * embed) ||
      !split_qkv() ||
      !AttentionDevice(backend, attention, *q, *k, *v, *attn, tokens, heads,
                       heads, head_dim, 0, 0, tokens, false, false)
           .has_value() ||
      !gemm_to(*attn, w.out_weight, *oproj, embed, embed) ||
      !bias(*oproj, w.out_bias, embed) ||
      !AddDevice(backend, add, x, *oproj, *h, elements).has_value() ||
      !norm(*h, w.ln2_weight, w.ln2_bias, *xn2) ||
      !gemm_to(*xn2, w.up_weight, *up, ffn, embed) ||
      !bias(*up, w.up_bias, ffn) ||
      !GeluDevice(backend, gelu, *up, *up, tokens * ffn).has_value() ||
      !gemm_to(*up, w.down_weight, *down, embed, ffn) ||
      !bias(*down, w.down_bias, embed) ||
      !AddDevice(backend, add, *h, *down, out, elements).has_value()) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

}  // namespace tessera::core
