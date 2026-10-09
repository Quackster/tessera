#include "tessera/speculative.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/decode.hpp"
#include "tessera/architecture.hpp"
#include "tessera/model.hpp"

namespace tessera::spec {

namespace {

// MTP drafts per batched verification when the caller sets no block.
constexpr std::size_t kDefaultMtpBlock = 4;

// Multi-token-prediction strategy. Stateless beyond the draft KV row base and
// the anchor hidden: the MTP head lives in the target model (Qwen3.5 module).
class MtpStrategy final : public SpeculativeStrategy {
 public:
  std::string_view Name() const override { return "mtp"; }

  std::expected<void, StatusCode> Attach(
      const StrategyOptions& options) override {
    if (options.draft_block_size > 0) {
      block_ = options.draft_block_size;
    }
    return {};
  }

  std::expected<void, StatusCode> Prepare(Backend& backend,
                                          Model& target) override {
    (void)backend;
    arch_ = target.Arch();
    if (arch_ == nullptr) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    return {};
  }

  std::size_t DraftBlock() const override { return block_; }
  std::span<const std::size_t> CaptureLayers() const override { return {}; }
  std::span<Buffer* const> CaptureBuffers() override { return {}; }

  std::expected<void, StatusCode> OnAnchor(
      Backend& backend, Model& target, core::DecodeCache& cache,
      std::uint32_t token, std::uint64_t position,
      std::span<const float> hidden) override {
    (void)backend;
    (void)target;
    anchor_ = token;
    anchor_pos_ = position;
    chain_hidden_.assign(hidden.begin(), hidden.end());
    mtp_base_ = arch_->DraftRows(cache);
    return {};
  }

  std::expected<std::size_t, StatusCode> Draft(
      Backend& backend, Model& target, core::DecodeCache& cache,
      std::span<const float> current_logits, std::uint32_t anchor,
      std::span<std::uint32_t> drafts) override {
    (void)current_logits;
    std::vector<float> chain = chain_hidden_;
    std::uint32_t tok = anchor;
    std::uint64_t pos = anchor_pos_ + 1;
    std::size_t count = 0;
    for (std::size_t i = 0; i < block_ && i < drafts.size(); ++i) {
      std::vector<float> next_chain;
      auto draft =
          core::MtpDraftStep(backend, target, cache, chain, tok, pos,
                             &next_chain);
      if (!draft) {
        if (draft.error() == StatusCode::UnsupportedFeature) {
          break;
        }
        return std::unexpected(draft.error());
      }
      drafts[i] = *draft;
      chain = std::move(next_chain);
      tok = *draft;
      ++pos;
      ++count;
    }
    return count;
  }

  std::expected<void, StatusCode> Commit(Backend& backend, Model& target,
                                         core::DecodeCache& cache,
                                         std::size_t accepted) override {
    (void)backend;
    (void)target;
    arch_->DraftTruncate(cache, mtp_base_ + accepted);
    return {};
  }

 private:
  std::size_t block_ = kDefaultMtpBlock;
  const Architecture* arch_ = nullptr;
  std::uint64_t mtp_base_ = 0;
  std::uint64_t anchor_pos_ = 0;
  std::uint32_t anchor_ = 0;
  std::vector<float> chain_hidden_;
};

}  // namespace

}  // namespace tessera::spec

namespace tessera {

std::unique_ptr<SpeculativeStrategy> CreateMtpStrategy() {
  return std::make_unique<spec::MtpStrategy>();
}

}  // namespace tessera
