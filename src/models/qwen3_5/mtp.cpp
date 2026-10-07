#include "models/qwen3_5/architecture.hpp"

#include <cstring>
#include <string>
#include <vector>

#include "core/decode.hpp"
#include "core/decode_hybrid_internal.hpp"
#include "core/decode_internal.hpp"

namespace tessera {

namespace {

using core::HybridDecodeCache;
using core::RunFullBlock;
using core::detail::DownloadF32;
using core::detail::GatherEmbedding;
using core::detail::GemmFor;
using core::detail::NeedWeight;
using core::detail::NeedWeightAny;
using core::detail::ProjectDevice;
using core::detail::RmsNormDevice;
using core::detail::UploadF32;

// The Qwen3.5 module: the nextn multi-token-prediction draft head. The
// trunk (hybrid gated attention plus gated-delta linear attention) is
// still in src/core/decode_hybrid.cpp and moves here next.
class Qwen35Architecture final : public Architecture {
 public:
  [[nodiscard]] std::string_view Name() const override { return "qwen35"; }

  // The Qwen3.5 MTP head (public layout, vLLM qwen3_5_mtp): concat the
  // normed token embedding and the normed backbone hidden, project with
  // nextn.eh_proj to one hidden vector, run one full-attention block, then
  // the shared head norm and the shared output weight.
  [[nodiscard]] std::expected<std::uint32_t, StatusCode> Draft(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const float> hidden, std::uint32_t token, std::uint64_t pos,
      std::vector<float>* mtp_hidden_out) const override {
    auto config = model.Config();
    if (!config) {
      return std::unexpected(config.error());
    }
    const TransformerConfig& cfg = *config;
    if (!cfg.hybrid) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    if (!cache.hybrid || !cache.hybrid->ready) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (token >= cfg.vocab_size || hidden.size() != cfg.hidden_dim) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    HybridDecodeCache& h = *cache.hybrid;
    const std::size_t hidden_dim = cfg.hidden_dim;
    const std::size_t layer = cfg.layers;
    const std::string base = "blk." + std::to_string(layer) + ".";
    auto eh = NeedWeightAny(model, base + "nextn.eh_proj.weight");
    auto enorm = NeedWeight(model, base + "nextn.enorm.weight", DType::F32);
    auto hnorm = NeedWeight(model, base + "nextn.hnorm.weight", DType::F32);
    auto shnorm =
        NeedWeight(model, base + "nextn.shared_head_norm.weight", DType::F32);
    auto output = NeedWeightAny(model, "output.weight");
    auto embed = NeedWeightAny(model, "token_embd.weight");
    if (!eh || !enorm || !hnorm || !shnorm || !output || !embed) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    auto gemm_eh = GemmFor(backend, h.gemms, (*eh)->manifest.dtype);
    auto gemm_out = GemmFor(backend, h.gemms, (*output)->manifest.dtype);
    if (!gemm_eh || !gemm_out) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    // e_n = RMSNorm(embed(token), enorm).
    if ((*embed)->manifest.dtype == DType::F32) {
      auto copy = backend.CopyD2D(*(*embed)->device, token * hidden_dim * 4,
                                  *h.mtp_h, 0, hidden_dim * 4);
      if (!copy) {
        return std::unexpected(copy.error());
      }
    } else {
      std::vector<float> row(hidden_dim);
      auto gathered = GatherEmbedding(backend, **embed, token, hidden_dim, row);
      if (!gathered) {
        return std::unexpected(gathered.error());
      }
      if (!UploadF32(backend, *h.mtp_h, row)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
    if (!RmsNormDevice(backend, *h.rmsnorm_kernel, *h.mtp_h, *(*enorm)->device,
                       *h.mtp_h, 1, hidden_dim, cfg.norm_eps)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    // h_n = RMSNorm(hidden, hnorm).
    if (!UploadF32(backend, *h.xn,
                   std::vector<float>(hidden.begin(), hidden.end())) ||
        !RmsNormDevice(backend, *h.rmsnorm_kernel, *h.xn, *(*hnorm)->device,
                       *h.xn, 1, hidden_dim, cfg.norm_eps)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    // fused = concat([e_n, h_n]); x = eh_proj(fused).
    if (!backend.CopyD2D(*h.mtp_h, 0, *h.mtp_fused, 0, hidden_dim * 4) ||
        !backend.CopyD2D(*h.xn, 0, *h.mtp_fused, hidden_dim * 4,
                         hidden_dim * 4) ||
        !ProjectDevice(backend, *(*gemm_eh), *h.mtp_fused, *(*eh)->device, *h.x,
                       1, hidden_dim, 2 * hidden_dim)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    const std::uint64_t triples[3] = {pos, pos, pos};
    if (!backend.CopyH2D(*h.pos,
                         std::span<const std::byte>(
                             reinterpret_cast<const std::byte*>(triples),
                             sizeof(triples)))) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (auto block = RunFullBlock(backend, model, cfg, h, layer, pos, h.mtp_kv);
        !block) {
      return std::unexpected(block.error());
    }
    if (!RmsNormDevice(backend, *h.rmsnorm_kernel, *h.x, *(*shnorm)->device,
                       *h.xn, 1, hidden_dim, cfg.norm_eps) ||
        !ProjectDevice(backend, *(*gemm_out), *h.xn, *(*output)->device,
                       *h.logits, 1, cfg.vocab_size, hidden_dim)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    if (mtp_hidden_out != nullptr) {
      // The shared-head norm output is the MTP hidden to chain the next
      // draft.
      auto down = DownloadF32(backend, *h.xn);
      if (!down) {
        return std::unexpected(down.error());
      }
      *mtp_hidden_out = std::move(*down);
    }
    backend.Synchronize();
    std::vector<float> logits(cfg.vocab_size);
    auto down = backend.CopyD2H(*h.logits,
                                reinterpret_cast<std::byte*>(logits.data()),
                                logits.size() * 4);
    if (!down) {
      return std::unexpected(down.error());
    }
    std::uint32_t best = 0;
    for (std::uint32_t i = 1; i < logits.size(); ++i) {
      if (logits[i] > logits[best]) {
        best = i;
      }
    }
    return best;
  }
};

}  // namespace

std::unique_ptr<Architecture> MakeQwen35Architecture() {
  return std::make_unique<Qwen35Architecture>();
}

}  // namespace tessera
