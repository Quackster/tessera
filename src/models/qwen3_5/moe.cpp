#include "models/qwen3_5/internal.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "core/decode_internal.hpp"
#include "models/qwen3_5/state.hpp"

// Sparse mixture-of-experts feed-forward for the Qwen3.5-MoE family
// (Ornith-1.5). The router projects the normed hidden to one logit per
// expert, the top-k experts are selected and softmaxed, each selected
// expert runs a gated MLP with its own weights, and an always-on shared
// expert contributes a sigmoid-gated dense MLP. The expert weights are
// rank-3 packed tensors; one expert's slice is a contiguous block, so the
// generic GEMM helpers run each expert after a device-to-device gather.

namespace tessera::models::qwen3_5 {

namespace detail = ::tessera::core::detail;

using detail::AddDevice;
using detail::NeedWeight;
using detail::NeedWeightAny;
using detail::RmsNormDevice;
using detail::SiluMulDevice;
using detail::TopKRowsDevice;

namespace {

// The Qwen3-family MoE renormalizes the selected top-k routing weights
// (norm_topk_prob). See docs/ORNITH-1.5.md for the reference note.
constexpr std::uint64_t kMoeRenormalize = 1;

// Elementwise threads per MoE launch (matches the built-in kernels).
constexpr std::size_t kMoeThreads = 256;

std::uint32_t MoeGrid(std::size_t elements) {
  return static_cast<std::uint32_t>((elements + kMoeThreads - 1) / kMoeThreads);
}

// Grows `slot` to at least `bytes` on the device, keeping it when already
// large enough.
std::expected<void, StatusCode> GrowBuffer(Backend& backend,
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

// The byte size of one expert's slice of a rank-3 packed weight tensor.
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

std::expected<void, StatusCode> LoadMoeKernels(Backend& backend,
                                               Qwen35State& h) {
  const auto load = [&backend](std::unique_ptr<Kernel>& slot,
                               const char* name)
      -> std::expected<void, StatusCode> {
    if (slot != nullptr) {
      return {};
    }
    auto kernel = backend.LoadKernel(name, {});
    if (!kernel) {
      return std::unexpected(kernel.error());
    }
    slot = std::move(*kernel);
    return {};
  };
  if (auto r = load(h.moe_gate_kernel, "moe_gate"); !r) return r;
  if (auto r = load(h.moe_scale_kernel, "moe_scale_add"); !r) return r;
  if (auto r = load(h.moe_topk_kernel, "top_k_rows"); !r) return r;
  if (auto r = load(h.moe_fill_kernel, "fill"); !r) return r;
  return {};
}

std::expected<void, StatusCode> EnsureMoeReady(
    Backend& backend, const TransformerConfig& cfg, Qwen35State& h) {
  if (h.moe_ready) {
    return {};
  }
  if (auto r = LoadMoeKernels(backend, h); !r) {
    return r;
  }
  const std::size_t hidden = cfg.hidden_dim;
  const std::size_t ne = cfg.num_experts;
  const std::size_t top_k = cfg.experts_per_tok;
  const std::size_t inter = std::max(cfg.moe_intermediate,
                                     cfg.shared_expert_intermediate);
  const auto alloc = [&backend](std::unique_ptr<Buffer>& slot,
                                std::size_t bytes)
      -> std::expected<void, StatusCode> {
    auto buffer = backend.AllocateBuffer(bytes, MemoryKind::Device);
    if (!buffer) {
      return std::unexpected(buffer.error());
    }
    slot = std::move(*buffer);
    return {};
  };
  if (auto r = alloc(h.moe_logits, ne * 4); !r) return r;
  if (auto r = alloc(h.moe_shexp_gate, 4); !r) return r;
  if (auto r = alloc(h.moe_sig, 4); !r) return r;
  if (auto r = alloc(h.moe_ids, top_k * 4); !r) return r;
  if (auto r = alloc(h.moe_vals, top_k * 4); !r) return r;
  if (auto r = alloc(h.moe_wts, top_k * 4); !r) return r;
  if (auto r = alloc(h.moe_acc, hidden * 4); !r) return r;
  if (auto r = alloc(h.moe_gate_out, inter * 4); !r) return r;
  if (auto r = alloc(h.moe_up_out, inter * 4); !r) return r;
  if (auto r = alloc(h.moe_inter, inter * 4); !r) return r;
  if (auto r = alloc(h.moe_down_out, hidden * 4); !r) return r;
  if (auto r = alloc(h.moe_shared, hidden * 4); !r) return r;
  if (auto r = alloc(h.moe_row_in, hidden * 4); !r) return r;
  if (auto r = alloc(h.moe_row_out, hidden * 4); !r) return r;
  h.moe_ready = true;
  return {};
}

std::expected<void, StatusCode> ZeroBuffer(Backend& backend, Qwen35State& h,
                                           Buffer& buffer, std::size_t n) {
  KernelLaunch zero;
  zero.grid_x = MoeGrid(n);
  zero.block_x = kMoeThreads;
  zero.buffers = {&buffer};
  zero.scalars = {0, n};
  if (!backend.LaunchKernel(*h.moe_fill_kernel, zero)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

std::expected<void, StatusCode> ScaleAdd(Backend& backend, Qwen35State& h,
                                         const Buffer& a, const Buffer& b,
                                         const Buffer& weight, Buffer& out,
                                         std::size_t n, std::size_t idx) {
  KernelLaunch add;
  add.grid_x = MoeGrid(n);
  add.block_x = kMoeThreads;
  add.buffers = {&a, &b, &weight, &out};
  add.scalars = {n, idx};
  if (!backend.LaunchKernel(*h.moe_scale_kernel, add)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

// One row of the MoE feed-forward: norm `x_row` into h.xn, route, run the
// experts and the shared expert, and write the summed feed-forward output
// (without the residual) to `out_row`.
std::expected<void, StatusCode> MoeFfnRow(Backend& backend,
                                          const Model& model,
                                          const TransformerConfig& cfg,
                                          Qwen35State& h, std::size_t layer,
                                          const Buffer& x_row,
                                          Buffer& out_row) {
  const std::size_t hidden = cfg.hidden_dim;
  const std::size_t ne = cfg.num_experts;
  const std::size_t top_k = cfg.experts_per_tok;
  const std::size_t inter = cfg.moe_intermediate;
  const std::size_t shared = cfg.shared_expert_intermediate;
  const std::string base = "blk." + std::to_string(layer) + ".";

  auto mlp_norm =
      NeedWeight(model, base + "post_attention_norm.weight", DType::F32);
  auto router = NeedWeight(model, base + "ffn_gate_inp.weight", DType::F32);
  auto shexp_gate =
      NeedWeight(model, base + "ffn_gate_inp_shexp.weight", DType::F32);
  auto gate_exps = NeedWeightAny(model, base + "ffn_gate_exps.weight");
  auto up_exps = NeedWeightAny(model, base + "ffn_up_exps.weight");
  auto down_exps = NeedWeightAny(model, base + "ffn_down_exps.weight");
  auto gate_sh = NeedWeightAny(model, base + "ffn_gate_shexp.weight");
  auto up_sh = NeedWeightAny(model, base + "ffn_up_shexp.weight");
  auto down_sh = NeedWeightAny(model, base + "ffn_down_shexp.weight");
  if (!mlp_norm || !router || !shexp_gate || !gate_exps || !up_exps ||
      !down_exps || !gate_sh || !up_sh || !down_sh) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (!RmsNormDevice(backend, *h.rmsnorm_kernel, x_row, *(*mlp_norm)->device,
                     *h.xn, 1, hidden, cfg.norm_eps) ||
      !ProjectBatch(backend, h, (*router)->manifest.dtype, *h.xn,
                    *(*router)->device, *h.moe_logits, 1, ne, hidden) ||
      !ProjectBatch(backend, h, (*shexp_gate)->manifest.dtype, *h.xn,
                    *(*shexp_gate)->device, *h.moe_shexp_gate, 1, 1, hidden)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (auto top = TopKRowsDevice(backend, *h.moe_topk_kernel, *h.moe_logits,
                                *h.moe_ids, *h.moe_vals, 1, ne, top_k);
      !top) {
    return std::unexpected(top.error());
  }
  {
    KernelLaunch gate;
    gate.grid_x = 1;
    gate.block_x = kMoeThreads;
    gate.buffers = {h.moe_logits.get(), h.moe_vals.get(),
                    h.moe_shexp_gate.get(), h.moe_wts.get(), h.moe_sig.get()};
    gate.scalars = {ne, top_k, kMoeRenormalize};
    if (!backend.LaunchKernel(*h.moe_gate_kernel, gate)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  std::vector<std::uint32_t> ids(top_k, 0);
  if (!backend.CopyD2H(*h.moe_ids, reinterpret_cast<std::byte*>(ids.data()),
                       top_k * 4)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (auto r = ZeroBuffer(backend, h, *h.moe_acc, hidden); !r) {
    return r;
  }
  auto gate_bytes = ExpertSliceBytes(**gate_exps, ne);
  auto up_bytes = ExpertSliceBytes(**up_exps, ne);
  auto down_bytes = ExpertSliceBytes(**down_exps, ne);
  if (!gate_bytes || !up_bytes || !down_bytes) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (auto r = GrowBuffer(backend, h.moe_gate_w, *gate_bytes); !r) return r;
  if (auto r = GrowBuffer(backend, h.moe_up_w, *up_bytes); !r) return r;
  if (auto r = GrowBuffer(backend, h.moe_down_w, *down_bytes); !r) return r;
  for (std::size_t k = 0; k < top_k; ++k) {
    const std::uint32_t expert = ids[k];
    if (expert >= ne) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    if (!backend.CopyD2D(*(*gate_exps)->device, expert * *gate_bytes,
                         *h.moe_gate_w, 0, *gate_bytes) ||
        !backend.CopyD2D(*(*up_exps)->device, expert * *up_bytes, *h.moe_up_w,
                         0, *up_bytes) ||
        !ProjectBatch(backend, h, (*gate_exps)->manifest.dtype, *h.xn,
                      *h.moe_gate_w, *h.moe_gate_out, 1, inter, hidden) ||
        !ProjectBatch(backend, h, (*up_exps)->manifest.dtype, *h.xn,
                      *h.moe_up_w, *h.moe_up_out, 1, inter, hidden) ||
        !SiluMulDevice(backend, *h.silu_mul_kernel, *h.moe_gate_out,
                       *h.moe_up_out, *h.moe_inter, inter) ||
        !backend.CopyD2D(*(*down_exps)->device, expert * *down_bytes,
                         *h.moe_down_w, 0, *down_bytes) ||
        !ProjectBatch(backend, h, (*down_exps)->manifest.dtype, *h.moe_inter,
                      *h.moe_down_w, *h.moe_down_out, 1, hidden, inter)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (auto r = ScaleAdd(backend, h, *h.moe_acc, *h.moe_down_out, *h.moe_wts,
                          *h.moe_acc, hidden, k);
        !r) {
      return r;
    }
  }
  if (!ProjectBatch(backend, h, (*gate_sh)->manifest.dtype, *h.xn,
                    *(*gate_sh)->device, *h.moe_gate_out, 1, shared, hidden) ||
      !ProjectBatch(backend, h, (*up_sh)->manifest.dtype, *h.xn,
                    *(*up_sh)->device, *h.moe_up_out, 1, shared, hidden) ||
      !SiluMulDevice(backend, *h.silu_mul_kernel, *h.moe_gate_out,
                     *h.moe_up_out, *h.moe_inter, shared) ||
      !ProjectBatch(backend, h, (*down_sh)->manifest.dtype, *h.moe_inter,
                    *(*down_sh)->device, *h.moe_shared, 1, hidden, shared)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (auto r = ScaleAdd(backend, h, *h.moe_acc, *h.moe_shared, *h.moe_sig,
                        *h.moe_acc, hidden, 0);
      !r) {
    return r;
  }
  if (!backend.CopyD2D(*h.moe_acc, 0, out_row, 0, hidden * 4)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

}  // namespace

std::expected<void, StatusCode> RunMoeFfn(Backend& backend,
                                          const Model& model,
                                          const TransformerConfig& cfg,
                                          Qwen35State& h, std::size_t layer) {
  if (auto ready = EnsureMoeReady(backend, cfg, h); !ready) {
    return ready;
  }
  auto row = MoeFfnRow(backend, model, cfg, h, layer, *h.x, *h.proj);
  if (!row) {
    return row;
  }
  if (!AddDevice(backend, *h.add_kernel, *h.x, *h.proj, *h.x, cfg.hidden_dim)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

std::expected<void, StatusCode> RunMoeFfnBatch(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen35State& h, std::size_t layer, std::size_t rows) {
  if (auto ready = EnsureMoeReady(backend, cfg, h); !ready) {
    return ready;
  }
  Qwen35BatchScratch& b = *h.batch;
  const std::size_t hidden = cfg.hidden_dim;
  for (std::size_t t = 0; t < rows; ++t) {
    if (!backend.CopyD2D(*b.x, t * hidden * 4, *h.moe_row_in, 0, hidden * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (auto row = MoeFfnRow(backend, model, cfg, h, layer, *h.moe_row_in,
                             *h.moe_row_out);
        !row) {
      return row;
    }
    if (!backend.CopyD2D(*h.moe_row_out, 0, *b.moe_out, t * hidden * 4,
                         hidden * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  if (!AddDevice(backend, *h.add_kernel, *b.x, *b.moe_out, *b.x,
                 rows * hidden)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

}  // namespace tessera::models::qwen3_5
