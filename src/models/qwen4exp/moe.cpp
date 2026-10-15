#include "models/qwen4exp/internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/moe_cpu.hpp"
#include "core/profile.hpp"
#include "models/qwen4exp/state.hpp"

// Sparse mixture-of-experts feed-forward for qwen4exp. The router projects
// the normed activation to one logit per expert, the top-k experts are
// selected and softmaxed, each runs a gated MLP with its own rank-3 packed
// weights, and an always-on shared expert contributes a sigmoid-gated
// dense MLP. The phase is split into a router (selection) and an expert
// phase so a batched caller can run every row's router, read the ids back
// once, stage the union and then run the experts per row.

namespace tessera::models::qwen4exp {

namespace detail = ::tessera::core::detail;

using detail::NeedWeight;
using detail::NeedWeightAny;
using detail::ProjectDevice;
using detail::SiluMulDevice;
using detail::TopKRowsDevice;

namespace {

constexpr std::size_t kMoeThreads = 256;

// Routed experts per layer the CPU computes (TESSERA_CPU_MOE), the CPU side
// of the CPU+GPU hybrid MoE. Zero disables it and stages every expert on the
// GPU. Read once from the environment.
const std::size_t kCpuMoeExpertsPerLayer = []() {
  const char* value = std::getenv("TESSERA_CPU_MOE");
  if (value == nullptr || *value == '\0') {
    return std::size_t{0};
  }
  return static_cast<std::size_t>(std::strtoul(value, nullptr, 10));
}();

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

std::expected<void, StatusCode> RunMoeRouter(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen4ExpState& h, std::size_t layer, const Buffer& mixed) {
  const std::size_t hidden = cfg.hidden_dim;
  const std::size_t ne = cfg.num_experts;
  const std::size_t top_k = cfg.experts_per_tok;
  const std::string base = "blk." + std::to_string(layer) + ".";
  auto router = NeedWeightAny(model, base + "ffn_gate_inp.weight");
  auto shexp_gate = NeedWeightAny(model, base + "ffn_gate_inp_shexp.weight");
  if (!router || !shexp_gate) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (!Grow(backend, h.moe_logits, ne * 4) ||
      !Grow(backend, h.moe_ids, top_k * 4) ||
      !Grow(backend, h.moe_vals, top_k * 4) ||
      !Grow(backend, h.moe_wts, top_k * 4) || !Grow(backend, h.moe_sig, 4)) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  // The moe_gate kernel reads the shared-expert gate as fp32; the checkpoint
  // stores it bf16, so convert it once per layer and reuse the copy. Doing
  // this every token cost a device-to-host and host-to-device round trip per
  // layer, which dominated the MoE host time.
  Buffer* shexp_w = (*shexp_gate)->device.get();
  if ((*shexp_gate)->manifest.dtype == DType::BF16) {
    if (h.moe_shexp_w.size() <= layer) {
      h.moe_shexp_w.resize(layer + 1);
    }
    if (h.moe_shexp_w[layer] == nullptr) {
      auto buf = backend.AllocateBuffer(hidden * 4, MemoryKind::Device);
      if (!buf) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
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
      h.moe_shexp_w[layer] = std::move(*buf);
      if (!detail::UploadF32(backend, *h.moe_shexp_w[layer], conv)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
    shexp_w = h.moe_shexp_w[layer].get();
  }
  core::PhaseScope scope(h.profile, core::Phase::kMoeRouter);
  if (!Project(backend, h, (*router)->manifest.dtype, mixed,
               *(*router)->device, *h.moe_logits, 1, ne, hidden)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (auto top = TopKRowsDevice(backend, *h.moe_topk, *h.moe_logits, *h.moe_ids,
                                *h.moe_vals, 1, ne, top_k);
      !top) {
    return std::unexpected(top.error());
  }
  KernelLaunch gate;
  gate.grid_x = 1;
  gate.block_x = kMoeThreads;
  gate.buffers = {h.moe_vals.get(), const_cast<Buffer*>(&mixed), shexp_w,
                  h.moe_wts.get(), h.moe_sig.get()};
  gate.scalars = {top_k, hidden};
  if (!backend.LaunchKernel(*h.moe_gate, gate)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

std::expected<void, StatusCode> RunMoeExperts(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen4ExpState& h, std::size_t layer, const Buffer& mixed,
    std::size_t n_gpu, std::span<const std::uint32_t> cpu_ids,
    std::span<const float> cpu_wts) {
  const std::size_t hidden = cfg.hidden_dim;
  const std::size_t ne = cfg.num_experts;
  const std::size_t top_k = cfg.experts_per_tok;
  const std::size_t inter = std::max(cfg.moe_intermediate,
                                     cfg.shared_expert_intermediate);
  const std::string base = "blk." + std::to_string(layer) + ".";
  auto gate_sh = NeedWeightAny(model, base + "ffn_gate_shexp.weight");
  auto up_sh = NeedWeightAny(model, base + "ffn_up_shexp.weight");
  auto down_sh = NeedWeightAny(model, base + "ffn_down_shexp.weight");
  if (!gate_sh || !up_sh || !down_sh) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (!Grow(backend, h.moe_acc, hidden * 4) ||
      !Grow(backend, h.moe_shared, hidden * 4) ||
      !Grow(backend, h.moe_gate_out, inter * 4) ||
      !Grow(backend, h.moe_up_out, inter * 4) ||
      !Grow(backend, h.moe_inter, inter * 4)) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  // The shared expert reads only the activation, so run it before the routed
  // expert kernels: its GPU work overlaps the host staging copy and upload
  // when the caller stages after this call.
  const std::size_t shared = cfg.shared_expert_intermediate;
  {
    core::PhaseScope scope(h.profile, core::Phase::kMoeShared);
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
  }
  // The caller staged the routed experts, so the slice geometry and the
  // fused kernel are known now.
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
  // The fused routed-expert kernels read each slice straight from the
  // staging tensor at ids[slot]*stride, so no per-expert copy is made.
  const Kernel* gate_up = nullptr;
  switch (gate_exps->manifest.dtype) {
  case DType::IQ2_S: gate_up = h.moe_gate_up_iq2s.get(); break;
  case DType::IQ3_XXS: gate_up = h.moe_gate_up_iq3xxs.get(); break;
  case DType::IQ3_S: gate_up = h.moe_gate_up_iq3s.get(); break;
  case DType::IQ4_XS: gate_up = h.moe_gate_up_iq4xs.get(); break;
  default: break;
  }
  const Kernel* down = nullptr;
  switch (down_exps->manifest.dtype) {
  case DType::IQ4_NL: down = h.moe_down_iq4nl.get(); break;
  case DType::Q2_0: down = h.moe_down_q20.get(); break;
  default: break;
  }
  if (gate_up == nullptr || down == nullptr) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  const std::size_t routed_inter = cfg.moe_intermediate;
  if (!Grow(backend, h.moe_phi, top_k * routed_inter * 4)) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  core::PhaseScope compute_scope(h.profile, core::Phase::kMoeCompute);
  {
    KernelLaunch gu;
    gu.grid_x = static_cast<std::uint32_t>((routed_inter * n_gpu + 7) / 8);
    gu.block_x = kMoeThreads;
    gu.buffers = {&mixed, gate_exps->device.get(), up_exps->device.get(),
                  h.moe_ids.get(), h.moe_phi.get()};
    gu.scalars = {hidden, routed_inter, n_gpu, *gate_bytes, *up_bytes};
    if (!backend.LaunchKernel(*gate_up, gu)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  {
    KernelLaunch dn;
    dn.grid_x = static_cast<std::uint32_t>((hidden + 7) / 8);
    dn.block_x = kMoeThreads;
    dn.buffers = {h.moe_phi.get(), down_exps->device.get(), h.moe_ids.get(),
                  h.moe_wts.get(), h.moe_acc.get()};
    dn.scalars = {hidden, routed_inter, n_gpu, *down_bytes, 0};
    if (!backend.LaunchKernel(*down, dn)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  // CPU side of the hybrid: compute the uncached experts the CPU took and add
  // their weighted outputs to the routed sum.
  if (!cpu_ids.empty()) {
    if (h.cpu_x.size() < hidden) {
      h.cpu_x.resize(hidden);
    }
    if (!backend.CopyD2H(mixed, reinterpret_cast<std::byte*>(h.cpu_x.data()),
                         hidden * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    const std::span<const std::byte> gate_host =
        model.PagedTensorHost(base + "ffn_gate_exps.weight");
    const std::span<const std::byte> up_host =
        model.PagedTensorHost(base + "ffn_up_exps.weight");
    const std::span<const std::byte> down_host =
        model.PagedTensorHost(base + "ffn_down_exps.weight");
    h.cpu_acc.assign(hidden, 0.0f);
    if (auto computed = core::CpuMoeExpertsForward(
            std::span<const float>(h.cpu_x), cpu_ids, cpu_wts,
            gate_exps->manifest.dtype, gate_host, *gate_bytes,
            up_exps->manifest.dtype, up_host, *up_bytes,
            down_exps->manifest.dtype, down_host, *down_bytes, hidden,
            routed_inter, h.cpu_acc);
        !computed) {
      return std::unexpected(computed.error());
    }
    if (n_gpu > 0) {
      if (!Grow(backend, h.moe_cpu_acc, hidden * 4) ||
          !detail::UploadF32(backend, *h.moe_cpu_acc, h.cpu_acc) ||
          !detail::AddDevice(backend, *h.add, *h.moe_acc, *h.moe_cpu_acc,
                             *h.moe_acc, hidden)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    } else if (!detail::UploadF32(backend, *h.moe_acc, h.cpu_acc)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  return ScaleAdd(backend, h, *h.moe_acc, *h.moe_shared, *h.moe_sig,
                  *h.block_out, hidden, 0);
}

// Read back the selected ids, split the uncached experts to the CPU when the
// hybrid is on, and stage the GPU experts. `n_gpu`, `cpu_ids` and `cpu_wts`
// describe the split for the expert phase.
std::expected<void, StatusCode> StageMoeSelection(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen4ExpState& h, std::size_t layer, std::size_t* n_gpu,
    std::vector<std::uint32_t>* cpu_ids, std::vector<float>* cpu_wts) {
  const std::size_t top_k = cfg.experts_per_tok;
  *n_gpu = top_k;
  cpu_ids->clear();
  cpu_wts->clear();
  if (!model.ExpertsPaged(layer)) {
    return {};
  }
  core::PhaseScope staging_scope(h.profile, core::Phase::kMoeStaging);
  std::vector<std::uint32_t> ids(top_k, 0);
  {
    core::PhaseScope scope(h.profile, core::Phase::kMoeReadback);
    if (!backend.CopyD2H(*h.moe_ids, reinterpret_cast<std::byte*>(ids.data()),
                         top_k * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  // Keep at least one expert on the GPU so the staging views stay resolved.
  const std::size_t cpu_target =
      std::min(kCpuMoeExpertsPerLayer, top_k > 0 ? top_k - 1 : 0);
  if (cpu_target == 0) {
    core::PhaseScope scope(h.profile, core::Phase::kMoeUpload);
    if (auto staged = model.EnsureLayerExperts(
            layer, std::span<const std::uint32_t>(ids));
        !staged) {
      return std::unexpected(staged.error());
    }
    return {};
  }
  std::vector<float> wts(top_k, 0.0f);
  if (!backend.CopyD2H(*h.moe_wts, reinterpret_cast<std::byte*>(wts.data()),
                       top_k * 4)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  std::vector<std::uint32_t> gpu;
  std::vector<float> gpu_wts;
  for (std::size_t s = 0; s < top_k; ++s) {
    if (model.ExpertCached(layer, ids[s]) || cpu_ids->size() >= cpu_target) {
      gpu.push_back(ids[s]);
      gpu_wts.push_back(wts[s]);
    } else {
      cpu_ids->push_back(ids[s]);
      cpu_wts->push_back(wts[s]);
    }
  }
  *n_gpu = gpu.size();
  core::PhaseScope scope(h.profile, core::Phase::kMoeUpload);
  if (auto staged =
          model.EnsureLayerExperts(layer,
                                   std::span<const std::uint32_t>(gpu));
      !staged) {
    return std::unexpected(staged.error());
  }
  if (!backend.CopyH2D(
          *h.moe_ids,
          std::span<const std::byte>(
              reinterpret_cast<const std::byte*>(gpu.data()), gpu.size() * 4)) ||
      !backend.CopyH2D(*h.moe_wts,
                       std::span<const std::byte>(
                           reinterpret_cast<const std::byte*>(gpu_wts.data()),
                           gpu_wts.size() * 4))) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

std::expected<void, StatusCode> RunMoeFfn(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen4ExpState& h, std::size_t layer, const Buffer& mixed) {
  if (auto routed = RunMoeRouter(backend, model, cfg, h, layer, mixed);
      !routed) {
    return routed;
  }
  std::size_t n_gpu = 0;
  std::vector<std::uint32_t> cpu_ids;
  std::vector<float> cpu_wts;
  if (auto staged = StageMoeSelection(backend, model, cfg, h, layer, &n_gpu,
                                      &cpu_ids, &cpu_wts);
      !staged) {
    return staged;
  }
  return RunMoeExperts(backend, model, cfg, h, layer, mixed, n_gpu, cpu_ids,
                       cpu_wts);
}

}  // namespace tessera::models::qwen4exp
