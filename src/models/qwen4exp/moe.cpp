#include "models/qwen4exp/internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/decode_internal.hpp"
#include "models/qwen4exp/state.hpp"

// Sparse mixture-of-experts feed-forward for qwen4exp. The router projects
// the normed activation to one logit per expert, the top-k experts are
// selected and softmaxed, each runs a gated MLP with its own rank-3 packed
// weights, and an always-on shared expert contributes a sigmoid-gated
// dense MLP. The routed expert formats (IQ3_XXS gate/up, IQ4_NL down) have
// no fused packed kernel, so each selected expert is gathered device to
// device and run through the generic GEMM. A fused packed-expert kernel
// and the residency cache are the throughput follow-up.

namespace tessera::models::qwen4exp {

namespace detail = ::tessera::core::detail;

using detail::NeedWeight;
using detail::NeedWeightAny;
using detail::ProjectDevice;
using detail::SiluMulDevice;
using detail::TopKRowsDevice;

namespace {

constexpr std::size_t kMoeThreads = 256;

std::uint32_t MoeGrid(std::size_t elements) {
  return static_cast<std::uint32_t>((elements + kMoeThreads - 1) / kMoeThreads);
}

std::expected<void, StatusCode> Grow(Backend& backend,
                                     std::unique_ptr<Buffer>& slot,
                                     std::size_t bytes) {
  if (slot != nullptr && slot->Size() >= bytes) {
    return {};
  }
  auto buffer = backend.AllocateBuffer(bytes, MemoryKind::Device);
  if (!buffer) {
    return std::unexpected(buffer.error());
  }
  slot = std::move(*buffer);
  return {};
}

std::expected<std::size_t, StatusCode> ExpertSliceBytes(
    const DeviceTensor& weight, std::size_t num_experts) {
  if (num_experts == 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const std::size_t total = weight.manifest.shape.Numel();
  if (total % num_experts != 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const std::size_t per = total / num_experts;
  if (per == 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return TensorBytes(weight.manifest.dtype, per);
}

std::expected<void, StatusCode> ZeroBuffer(Backend& backend, Qwen4ExpState& h,
                                           Buffer& buffer, std::size_t n) {
  KernelLaunch zero;
  zero.grid_x = MoeGrid(n);
  zero.block_x = kMoeThreads;
  zero.buffers = {&buffer};
  zero.scalars = {0, n};
  if (!backend.LaunchKernel(*h.moe_fill, zero)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

std::expected<void, StatusCode> ScaleAdd(Backend& backend, Qwen4ExpState& h,
                                         const Buffer& a, const Buffer& b,
                                         const Buffer& weight, Buffer& out,
                                         std::size_t n, std::size_t idx) {
  KernelLaunch add;
  add.grid_x = MoeGrid(n);
  add.block_x = kMoeThreads;
  add.buffers = {&a, &b, &weight, &out};
  add.scalars = {n, idx};
  if (!backend.LaunchKernel(*h.moe_scale_add, add)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

}  // namespace

std::expected<void, StatusCode> RunMoeFfn(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen4ExpState& h, std::size_t layer, const Buffer& mixed) {
  const std::size_t hidden = cfg.hidden_dim;
  const std::size_t ne = cfg.num_experts;
  const std::size_t top_k = cfg.experts_per_tok;
  const std::size_t inter = std::max(cfg.moe_intermediate,
                                     cfg.shared_expert_intermediate);
  const std::string base = "blk." + std::to_string(layer) + ".";

  auto router = NeedWeightAny(model, base + "ffn_gate_inp.weight");
  auto shexp_gate = NeedWeightAny(model, base + "ffn_gate_inp_shexp.weight");
  auto gate_sh = NeedWeightAny(model, base + "ffn_gate_shexp.weight");
  auto up_sh = NeedWeightAny(model, base + "ffn_up_shexp.weight");
  auto down_sh = NeedWeightAny(model, base + "ffn_down_shexp.weight");
  if (!router || !shexp_gate || !gate_sh || !up_sh || !down_sh) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (!Grow(backend, h.moe_logits, ne * 4) ||
      !Grow(backend, h.moe_ids, top_k * 4) ||
      !Grow(backend, h.moe_vals, top_k * 4) ||
      !Grow(backend, h.moe_wts, top_k * 4) || !Grow(backend, h.moe_sig, 4) ||
      !Grow(backend, h.moe_acc, hidden * 4) ||
      !Grow(backend, h.moe_shared, hidden * 4) ||
      !Grow(backend, h.moe_gate_out, inter * 4) ||
      !Grow(backend, h.moe_up_out, inter * 4) ||
      !Grow(backend, h.moe_inter, inter * 4) ||
      !Grow(backend, h.moe_down_out, hidden * 4) ||
      !Grow(backend, h.moe_shexp_w, hidden * 4)) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  // The moe_gate kernel reads the shared-expert gate as fp32; the
  // checkpoint stores it bf16, so stage a converted copy in that case.
  Buffer* shexp_w = (*shexp_gate)->device.get();
  if ((*shexp_gate)->manifest.dtype == DType::BF16) {
    std::vector<std::byte> raw(hidden * 2);
    if (!backend.CopyD2H(*(*shexp_gate)->device, raw.data(), raw.size())) {
      return std::unexpected(StatusCode::DeviceError);
    }
    std::vector<float> conv(hidden);
    for (std::size_t i = 0; i < hidden; ++i) {
      std::uint16_t half = 0;
      std::memcpy(&half, raw.data() + i * 2, 2);
      const std::uint32_t bits = static_cast<std::uint32_t>(half) << 16;
      std::memcpy(&conv[i], &bits, sizeof(conv[i]));
    }
    if (!detail::UploadF32(backend, *h.moe_shexp_w, conv)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    shexp_w = h.moe_shexp_w.get();
  }
  if (!Project(backend, h, (*router)->manifest.dtype, mixed,
               *(*router)->device, *h.moe_logits, 1, ne, hidden)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (auto top = TopKRowsDevice(backend, *h.moe_topk, *h.moe_logits, *h.moe_ids,
                                *h.moe_vals, 1, ne, top_k);
      !top) {
    return std::unexpected(top.error());
  }
  if (layer == 0 && std::getenv("TESSERA_QWEN4EXP_DUMP")) {
    std::vector<std::uint32_t> ids(top_k, 0);
    std::vector<float> lv(ne, 0.0f);
    backend.CopyD2H(*h.moe_ids, reinterpret_cast<std::byte*>(ids.data()),
                    top_k * 4);
    backend.CopyD2H(*h.moe_logits, reinterpret_cast<std::byte*>(lv.data()),
                    ne * 4);
    float m2[4] = {0, 0, 0, 0};
    backend.CopyD2H(mixed, reinterpret_cast<std::byte*>(m2), 16);
    std::fprintf(stderr,
                 "q4e dump moein: %.5f %.5f %.5f %.5f  ids", m2[0], m2[1], m2[2],
                 m2[3]);
    for (std::size_t k = 0; k < top_k; ++k) {
      std::fprintf(stderr, " %u(%.3f)", ids[k], lv[ids[k]]);
    }
    std::fprintf(stderr, "\n");
    std::vector<float> full(hidden, 0.0f);
    backend.CopyD2H(mixed, reinterpret_cast<std::byte*>(full.data()),
                    hidden * 4);
    if (FILE* fp = std::fopen("/tmp/q4e_moein.bin", "wb")) {
      std::fwrite(full.data(), 4, full.size(), fp);
      std::fclose(fp);
    }
  }
  {
    KernelLaunch gate;
    gate.grid_x = 1;
    gate.block_x = kMoeThreads;
    gate.buffers = {h.moe_vals.get(), const_cast<Buffer*>(&mixed), shexp_w,
                    h.moe_wts.get(), h.moe_sig.get()};
    gate.scalars = {top_k, hidden};
    if (!backend.LaunchKernel(*h.moe_gate, gate)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }

  std::vector<std::uint32_t> ids(top_k, 0);
  if (!backend.CopyD2H(*h.moe_ids, reinterpret_cast<std::byte*>(ids.data()),
                       top_k * 4)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  // Paged experts: make the routed selection resident before the slices
  // are resolved. The shared expert is dense and stays resident.
  if (model.ExpertsPaged(layer)) {
    if (auto staged = model.EnsureLayerExperts(
            layer, std::span<const std::uint32_t>(ids));
        !staged) {
      return std::unexpected(staged.error());
    }
  }
  const DeviceTensor* gate_exps =
      model.FindDeviceTensor(base + "ffn_gate_exps.weight");
  const DeviceTensor* up_exps =
      model.FindDeviceTensor(base + "ffn_up_exps.weight");
  const DeviceTensor* down_exps =
      model.FindDeviceTensor(base + "ffn_down_exps.weight");
  if (gate_exps == nullptr || up_exps == nullptr || down_exps == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto gate_bytes = ExpertSliceBytes(*gate_exps, ne);
  auto up_bytes = ExpertSliceBytes(*up_exps, ne);
  auto down_bytes = ExpertSliceBytes(*down_exps, ne);
  if (!gate_bytes || !up_bytes || !down_bytes) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const DType gate_dt = gate_exps->manifest.dtype;
  const DType up_dt = up_exps->manifest.dtype;
  const DType down_dt = down_exps->manifest.dtype;
  if (!Grow(backend, h.moe_gate_w, *gate_bytes) ||
      !Grow(backend, h.moe_up_w, *up_bytes) ||
      !Grow(backend, h.moe_down_w, *down_bytes) ||
      !ZeroBuffer(backend, h, *h.moe_acc, hidden)) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  const std::size_t routed_inter = cfg.moe_intermediate;
  for (std::size_t k = 0; k < top_k; ++k) {
    const std::uint32_t expert = ids[k];
    if (expert >= ne) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    if (!backend.CopyD2D(*gate_exps->device, expert * *gate_bytes,
                         *h.moe_gate_w, 0, *gate_bytes) ||
        !backend.CopyD2D(*up_exps->device, expert * *up_bytes, *h.moe_up_w, 0,
                         *up_bytes) ||
        !Project(backend, h, gate_dt, mixed, *h.moe_gate_w, *h.moe_gate_out, 1,
                 routed_inter, hidden) ||
        !Project(backend, h, up_dt, mixed, *h.moe_up_w, *h.moe_up_out, 1,
                 routed_inter, hidden) ||
        !SiluMulDevice(backend, *h.silu_mul, *h.moe_gate_out, *h.moe_up_out,
                       *h.moe_inter, routed_inter) ||
        !backend.CopyD2D(*down_exps->device, expert * *down_bytes,
                         *h.moe_down_w, 0, *down_bytes) ||
        !Project(backend, h, down_dt, *h.moe_inter, *h.moe_down_w,
                 *h.moe_down_out, 1, hidden, routed_inter)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (auto scaled = ScaleAdd(backend, h, *h.moe_acc, *h.moe_down_out,
                               *h.moe_wts, *h.moe_acc, hidden, k);
        !scaled) {
      return scaled;
    }
  }

  const std::size_t shared = cfg.shared_expert_intermediate;
  if (!Project(backend, h, (*gate_sh)->manifest.dtype, mixed,
               *(*gate_sh)->device, *h.moe_gate_out, 1, shared, hidden) ||
      !Project(backend, h, (*up_sh)->manifest.dtype, mixed, *(*up_sh)->device,
               *h.moe_up_out, 1, shared, hidden) ||
      !SiluMulDevice(backend, *h.silu_mul, *h.moe_gate_out, *h.moe_up_out,
                     *h.moe_inter, shared) ||
      !Project(backend, h, (*down_sh)->manifest.dtype, *h.moe_inter,
               *(*down_sh)->device, *h.moe_shared, 1, hidden, shared)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (auto combined =
          ScaleAdd(backend, h, *h.moe_acc, *h.moe_shared, *h.moe_sig,
                   *h.block_out, hidden, 0);
      !combined) {
    return combined;
  }
  if (layer == 0 && std::getenv("TESSERA_QWEN4EXP_DUMP")) {
    std::vector<float> mout(hidden, 0.0f);
    backend.CopyD2H(*h.block_out, reinterpret_cast<std::byte*>(mout.data()),
                    hidden * 4);
    if (FILE* fp = std::fopen("/tmp/q4e_moeout.bin", "wb")) {
      std::fwrite(mout.data(), 4, mout.size(), fp);
      std::fclose(fp);
    }
  }
  return {};
}

}  // namespace tessera::models::qwen4exp
