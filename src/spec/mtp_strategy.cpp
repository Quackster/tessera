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
// Two is the measured balance on the 27B Q4_K_M GGUF (ROCm, 7900 XTX)
// after the anchor folds into the verify and the multi-row GEMV reads each
// weight block once per column: a 2-row verify costs about 1.1x a one-row
// forward, and the second draft verifies for almost free, so the decode is
// about 30-34 tok/s at block 2 against 27-30 at block 1 on prose, a
// planet list, and a repetitive loop. Block 3 wins slightly more on
// structured text (35) but wastes drafts on uncertain text; block 4 both
// falls off the multi-row kernel's row range and accepts under half its
// drafts, so it is much slower. Callers can still override with
// `--draft-block`.
constexpr std::size_t kDefaultMtpBlock = 2;

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
  // The anchor rides the verify batch (the reference MTP cycle): the draft
  // runs before the target sees the anchor, pairing the anchor token with
  // the hidden one position before, and the verify scores
  // [anchor, drafts...] in one batched forward. OnAnchor is therefore
  // called after the verify with the next anchor.
  [[nodiscard]] bool FoldsAnchor() const override { return true; }
  [[nodiscard]] bool NeedsPrefillAnchor() const override { return true; }
  std::span<const std::size_t> CaptureLayers() const override { return {}; }
  std::span<Buffer* const> CaptureBuffers() override { return {}; }

  // The MTP head chains from one anchor hidden, so it has no batched
  // prefill: the engine keeps the per-token OnAnchor path for it. An
  // empty probe still returns Ok per the interface contract.
  std::expected<void, StatusCode> AppendPrefill(
      Backend& backend, Model& target, core::DecodeCache& cache,
      std::span<const std::uint32_t> tokens, std::uint64_t position,
      std::span<Buffer* const> captures) override {
    (void)backend;
    (void)target;
    (void)cache;
    (void)position;
    (void)captures;
    if (tokens.empty()) {
      return {};
    }
    return std::unexpected(StatusCode::UnsupportedFeature);
  }

  std::expected<void, StatusCode> OnAnchor(
      Backend& backend, Model& target, core::DecodeCache& cache,
      std::uint32_t token, std::uint64_t position,
      std::span<const float> hidden) override {
    (void)backend;
    (void)target;
    // Folding contract: `token` is the next anchor at `position` and
    // `hidden` is the target residual hidden at `position - 1`; the first
    // MTP step consumes that pair at KV slot `position`.
    anchor_ = token;
    anchor_pos_ = position;
    chain_hidden_.assign(hidden.begin(), hidden.end());
    mtp_base_ = arch_->DraftRows(cache);
    chain_rows_ = 0;
    return {};
  }

  std::expected<std::size_t, StatusCode> Draft(
      Backend& backend, Model& target, core::DecodeCache& cache,
      std::span<const float> current_logits, std::uint32_t anchor,
      std::span<std::uint32_t> drafts) override {
    (void)current_logits;
    std::vector<float> chain = chain_hidden_;
    std::uint32_t tok = anchor;
    std::size_t count = 0;
    for (std::size_t i = 0; i < block_ && i < drafts.size(); ++i) {
      // The last draft in the step is not chained, so skip its hidden-state
      // download: only the draft token is needed from it.
      const bool last = (i + 1 >= block_) || (i + 1 >= drafts.size());
      std::vector<float> next_chain;
      // Slot `anchor_pos_ + i`: the first step writes the anchor's own
      // slot with (anchor token, hidden at anchor_pos_ - 1) and predicts
      // the next token; later steps chain the previous MTP output hidden,
      // exactly like the reference cycle.
      auto draft =
          core::MtpDraftStep(backend, target, cache, chain, tok,
                             anchor_pos_ + i, last ? nullptr : &next_chain);
      if (!draft) {
        if (draft.error() == StatusCode::UnsupportedFeature) {
          break;
        }
        return std::unexpected(draft.error());
      }
      drafts[i] = *draft;
      if (!last) {
        chain = std::move(next_chain);
      }
      tok = *draft;
      ++count;
      ++chain_rows_;
    }
    return count;
  }

  std::expected<void, StatusCode> Commit(Backend& backend, Model& target,
                                         core::DecodeCache& cache,
                                         std::size_t accepted) override {
    (void)backend;
    (void)target;
    // Keep the rows for the committed positions: the anchor's row (step 0)
    // plus one per accepted draft, never more than the chain wrote. The
    // next cycle rewrites the following slot in order, so rejected rows
    // never enter the attention window.
    const std::size_t keep = std::min(chain_rows_, accepted + 1);
    arch_->DraftTruncate(cache, mtp_base_ + keep);
    return {};
  }

 private:
  std::size_t block_ = kDefaultMtpBlock;
  const Architecture* arch_ = nullptr;
  std::uint64_t mtp_base_ = 0;
  std::uint64_t anchor_pos_ = 0;
  std::uint32_t anchor_ = 0;
  // Rows the current chain wrote (one per MtpDraftStep); Commit keeps at
  // most that many plus the anchor.
  std::size_t chain_rows_ = 0;
  std::vector<float> chain_hidden_;
};

}  // namespace

}  // namespace tessera::spec

namespace tessera {

std::unique_ptr<SpeculativeStrategy> CreateMtpStrategy() {
  return std::make_unique<spec::MtpStrategy>();
}

}  // namespace tessera
