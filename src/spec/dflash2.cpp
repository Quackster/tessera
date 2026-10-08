#include "tessera/speculative.hpp"

#include "core/loaders/safetensors.hpp"

namespace tessera::spec {

namespace {

// DFlash2 draft block size must stay in the reference range; the candidate
// selector and dynamic convolution assume small blocks. 0 means "use the
// checkpoint's configured block size", which is what the drafter was
// trained for.
constexpr std::size_t kMinDraftBlockSize = 0;
constexpr std::size_t kMaxDraftBlockSize = 8;

// DFlash2 strategy: local dynamic convolution (grouped causal convolutions
// around attention and the MLP inside the parallel draft block) plus the
// low-rank candidate selector. The draft/verification algorithms land in
// milestone 7 (docs/PROGRESS.md); today the strategy validates the draft
// checkpoint layout.
class DFlash2Strategy final : public SpeculativeStrategy {
 public:
  std::string_view Name() const override {
    return "dflash2";
  }

  std::expected<void, StatusCode> Attach(
      const StrategyOptions& options) override {
    if (attached_) {
      return {};
    }
    if (options.draft_path.empty()) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    if (options.draft_block_size < kMinDraftBlockSize ||
        options.draft_block_size > kMaxDraftBlockSize) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    // The draft checkpoint is a model directory (config.json + weights).
    auto layout = core::InspectMxFp4Directory(options.draft_path);
    if (!layout) {
      return std::unexpected(layout.error());
    }
    draft_path_ = options.draft_path;
    draft_block_size_ = options.draft_block_size;
    attached_ = true;
    return {};
  }

 private:
  std::string draft_path_;
  std::size_t draft_block_size_ = 4;
  bool attached_ = false;
};

}  // namespace

}  // namespace tessera::spec

namespace tessera {

std::unique_ptr<SpeculativeStrategy> CreateDFlash2Strategy() {
  return std::make_unique<spec::DFlash2Strategy>();
}

}  // namespace tessera
