#include "spec/dflash2_drafter.hpp"

#include <string>

#include "spec/dflash2_block.hpp"

namespace tessera::spec {

std::expected<DFlash2Drafter, StatusCode> DFlash2Drafter::Create(
    Backend& backend, const std::filesystem::path& dir,
    const DFlash2Config& config) {
  DFlash2Drafter drafter;
  auto store = DraftWeightStore::Load(backend, dir, config);
  if (!store) {
    return std::unexpected(store.error());
  }
  drafter.store_ = std::move(*store);
  auto load = [&backend](std::unique_ptr<Kernel>& slot,
                         const char* name) -> StatusCode {
    auto kernel = backend.LoadKernel(name, {});
    if (!kernel) {
      return kernel.error();
    }
    slot = std::move(*kernel);
    return StatusCode::Ok;
  };
  const std::pair<std::unique_ptr<Kernel>*, const char*> kernels[] = {
      {&drafter.rmsnorm_, "rmsnorm"},  {&drafter.gemm_, "gemm_f32"},
      {&drafter.conv_, "dflash_conv"}, {&drafter.rope_, "rope"},
      {&drafter.attention_, "attention"}, {&drafter.silu_, "silu_mul"},
      {&drafter.add_, "add"},          {&drafter.concat_, "concat_features"},
  };
  for (const auto& [slot, name] : kernels) {
    const StatusCode status = load(*slot, name);
    if (status != StatusCode::Ok) {
      return std::unexpected(status);
    }
  }
  drafter.config_ = config;
  return drafter;
}

std::expected<void, StatusCode> DFlash2Drafter::Run(
    Backend& backend, const Buffer& mask_embeds, const Buffer& aux,
    const Buffer& output_w, const Kernel& head_gemm, Buffer& logits,
    std::size_t rows, std::size_t ctx, std::size_t vocab) const {
  const std::size_t n = config_.target_layer_ids.size();
  // The draft layer weights are fp32 after conversion, so gemm_f32 runs
  // them; `head_gemm` is accepted for the target's quantized shared head.
  (void)head_gemm;
  return DraftBlockDevice(
      backend, *rmsnorm_, *gemm_, *conv_, *rope_, *attention_, *silu_, *add_,
      *concat_, mask_embeds, aux, *store_.Weights().fc,
      store_.Weights().layers, *store_.Weights().final_norm, output_w, logits,
      rows, ctx, config_.hidden_size, n, config_.hidden_size, vocab,
      config_.num_heads, config_.num_kv_heads, config_.head_dim,
      config_.intermediate_size, config_.conv_kernel_size,
      config_.conv_group_size, config_.block_size, config_.sliding_window, 0,
      config_.rope_theta, static_cast<float>(config_.rms_norm_eps));
}

}  // namespace tessera::spec
