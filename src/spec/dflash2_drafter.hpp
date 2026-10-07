#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <memory>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"
#include "spec/dflash2_config.hpp"
#include "spec/dflash2_weights.hpp"

namespace tessera::spec {

// The loaded DFlash2 drafter: the draft weights in fp32 plus the kernels
// the draft block needs. `Run` executes one draft block: the target hidden
// states are fused with `fc`, the mask-token queries run the layer stack
// with that context, and the final hidden is projected to logits with the
// supplied output weight (normally the target model's shared head).
class DFlash2Drafter {
 public:
  DFlash2Drafter() = default;
  DFlash2Drafter(DFlash2Drafter&&) = default;
  DFlash2Drafter& operator=(DFlash2Drafter&&) = default;
  DFlash2Drafter(const DFlash2Drafter&) = delete;
  DFlash2Drafter& operator=(const DFlash2Drafter&) = delete;

  [[nodiscard]] static std::expected<DFlash2Drafter, StatusCode> Create(
      Backend& backend, const std::filesystem::path& dir,
      const DFlash2Config& config);

  [[nodiscard]] const DFlash2Config& Config() const { return config_; }
  [[nodiscard]] const DraftWeightStore& Weights() const { return store_; }

  // One draft block. `mask_embeds` is rows x hidden, `aux` is
  // (target_layer_ids.size()) x ctx x hidden, `output_w` is vocab x hidden,
  // `logits` is rows x vocab.
  [[nodiscard]] std::expected<void, StatusCode> Run(
      Backend& backend, const Buffer& mask_embeds, const Buffer& aux,
      const Buffer& output_w, const Kernel& head_gemm, Buffer& logits,
      std::size_t rows, std::size_t ctx, std::size_t vocab,
      Buffer* hidden_out = nullptr) const;

 private:
  DFlash2Config config_;
  DraftWeightStore store_;
  std::unique_ptr<Kernel> rmsnorm_;
  std::unique_ptr<Kernel> gemm_;
  std::unique_ptr<Kernel> conv_;
  std::unique_ptr<Kernel> rope_;
  std::unique_ptr<Kernel> attention_;
  std::unique_ptr<Kernel> silu_;
  std::unique_ptr<Kernel> add_;
  std::unique_ptr<Kernel> concat_;
};

}  // namespace tessera::spec
