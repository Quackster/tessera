#include "spec/dflash2_mlp.hpp"

#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/gemm.hpp"
#include "core/numerics/norm.hpp"
#include "spec/dflash2_conv.hpp"

namespace tessera::spec {

namespace {

using core::detail::ProjectDevice;
using core::detail::RmsNormDevice;
using core::detail::SiluMulDevice;

}  // namespace

std::expected<void, StatusCode> DraftMlpRef(
    std::span<const float> x, std::span<const float> post_norm_w,
    std::span<const float> conv_proj_w, std::span<const float> conv_base,
    std::span<const float> gate_w, std::span<const float> up_w,
    std::span<const float> down_w, std::span<float> out, std::size_t rows,
    std::size_t hidden, std::size_t ffn, std::size_t taps,
    std::size_t group_size, std::size_t block_size, float eps) {
  if (rows == 0 || hidden == 0 || ffn == 0 || taps == 0 || group_size == 0 ||
      block_size == 0 || hidden % group_size != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t num_groups = hidden / group_size;
  const std::size_t proj_n = 2 * taps * num_groups;
  if (x.size() != rows * hidden || out.size() != rows * hidden ||
      post_norm_w.size() != hidden ||
      conv_proj_w.size() != proj_n * hidden ||
      conv_base.size() != 2 * taps * hidden ||
      gate_w.size() != ffn * hidden || up_w.size() != ffn * hidden ||
      down_w.size() != hidden * ffn) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> xn(rows * hidden);
  std::vector<float> proj(rows * proj_n);
  std::vector<float> h1(rows * hidden);
  std::vector<float> gate(rows * ffn);
  std::vector<float> up(rows * ffn);
  std::vector<float> act(rows * ffn);
  std::vector<float> down(rows * hidden);
  std::span<const float> base_side0(conv_base.data(), taps * hidden);
  std::span<const float> base_side1(conv_base.data() + taps * hidden,
                                    taps * hidden);
  if (!core::RmsNormRef(x, post_norm_w, std::span<float>(xn), rows, hidden,
                        eps) ||
      !core::GemmF32Ref(std::span<const float>(xn), conv_proj_w,
                        std::span<float>(proj), rows, proj_n, hidden) ||
      !GroupedConvFinishRef(std::span<const float>(xn),
                            std::span<const float>(proj), base_side0,
                            std::span<float>(h1), rows, hidden, taps,
                            group_size, block_size, 0) ||
      !core::GemmF32Ref(std::span<const float>(h1), gate_w,
                        std::span<float>(gate), rows, ffn, hidden) ||
      !core::GemmF32Ref(std::span<const float>(h1), up_w,
                        std::span<float>(up), rows, ffn, hidden) ||
      !core::SiluMulRef(std::span<const float>(gate),
                        std::span<const float>(up), std::span<float>(act),
                        rows * ffn) ||
      !core::GemmF32Ref(std::span<const float>(act), down_w,
                        std::span<float>(down), rows, hidden, ffn) ||
      !GroupedConvFinishRef(std::span<const float>(down),
                            std::span<const float>(proj), base_side1, out,
                            rows, hidden, taps, group_size, block_size, 1)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return {};
}

std::expected<void, StatusCode> DraftMlpDevice(
    Backend& backend, const Kernel& rmsnorm, const Kernel& gemm,
    const Kernel& conv, const Kernel& silu, Buffer& xn, Buffer& proj,
    Buffer& h1, Buffer& gate, Buffer& up, Buffer& act, Buffer& down,
    Buffer& side_base, const Buffer& x, const Buffer& post_norm_w,
    const Buffer& conv_proj_w, const Buffer& conv_base, const Buffer& gate_w,
    const Buffer& up_w, const Buffer& down_w, Buffer& out, std::size_t rows,
    std::size_t hidden, std::size_t ffn, std::size_t taps,
    std::size_t group_size, std::size_t block_size, float eps) {
  if (rows == 0 || hidden == 0 || ffn == 0 || taps == 0 || group_size == 0 ||
      block_size == 0 || hidden % group_size != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t num_groups = hidden / group_size;
  const std::size_t proj_n = 2 * taps * num_groups;
  if (!RmsNormDevice(backend, rmsnorm, x, post_norm_w, xn, rows, hidden,
                     eps) ||
      !ProjectDevice(backend, gemm, xn, conv_proj_w, proj, rows, proj_n,
                     hidden) ||
      !GroupedConvFinishDevice(backend, conv, xn, proj, conv_base, side_base,
                               h1, rows, hidden, taps, group_size, block_size,
                               0) ||
      !ProjectDevice(backend, gemm, h1, gate_w, gate, rows, ffn, hidden) ||
      !ProjectDevice(backend, gemm, h1, up_w, up, rows, ffn, hidden) ||
      !SiluMulDevice(backend, silu, gate, up, act, rows * ffn) ||
      !ProjectDevice(backend, gemm, act, down_w, down, rows, hidden, ffn) ||
      !GroupedConvFinishDevice(backend, conv, down, proj, conv_base,
                               side_base, out, rows, hidden, taps, group_size,
                               block_size, 1)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

}  // namespace tessera::spec
