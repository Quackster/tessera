#include "tessera/speculative.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "core/loaders/safetensors.hpp"
#include "spec/dflash2_candidates.hpp"
#include "spec/dflash2_config.hpp"
#include "spec/dflash2_context.hpp"
#include "spec/dflash2_drafter.hpp"
#include "spec/dflash2_mask.hpp"
#include "spec/dflash2_selector.hpp"
#include "tessera/model.hpp"

namespace tessera::spec {

namespace {

// DFlash2 was trained for a small block; the candidate selector and dynamic
// convolution assume it. 0 means "use the checkpoint's block size".
constexpr std::size_t kMaxDraftBlockSize = 8;

// DFlash2 strategy: local dynamic convolution (grouped causal convolutions
// around attention and the MLP in the parallel draft block) plus the low-rank
// candidate selector. Hidden-conditioned: the draft context is the target's
// residual hidden at `target_layer_ids`.
class DFlash2Strategy final : public SpeculativeStrategy {
 public:
  std::string_view Name() const override { return "dflash2"; }

  std::expected<void, StatusCode> Attach(
      const StrategyOptions& options) override {
    if (attached_) {
      return {};
    }
    if (options.draft_path.empty()) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    if (options.draft_block_size > kMaxDraftBlockSize) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    auto layout = core::InspectMxFp4Directory(options.draft_path);
    if (!layout) {
      return std::unexpected(layout.error());
    }
    draft_path_ = options.draft_path;
    draft_block_size_ = options.draft_block_size;
    attached_ = true;
    return {};
  }

  std::expected<void, StatusCode> Prepare(Backend& backend,
                                          Model& target) override {
    if (prepared_) {
      return {};
    }
    auto config = LoadDFlash2Config(draft_path_);
    if (!config) {
      return std::unexpected(config.error());
    }
    config_ = std::move(*config);
    auto drafter = DFlash2Drafter::Create(backend, draft_path_, config_);
    if (!drafter) {
      return std::unexpected(drafter.error());
    }
    drafter_ = std::move(*drafter);
    auto target_config = target.Config();
    if (!target_config) {
      return std::unexpected(target_config.error());
    }
    for (const DeviceTensor& weight : target.Weights()) {
      if (weight.manifest.name == "token_embd.weight") {
        embed_ = &weight;
      } else if (weight.manifest.name == "output.weight") {
        output_ = &weight;
      }
    }
    if (embed_ == nullptr || output_ == nullptr) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    auto head_gemm =
        core::detail::GemmFor(backend, gemms_, output_->manifest.dtype);
    if (!head_gemm) {
      return std::unexpected(head_gemm.error());
    }
    head_gemm_ = *head_gemm;
    hidden_ = target_config->hidden_dim;
    vocab_ = config_.vocab_size;
    block_ = config_.block_size - 1;
    if (draft_block_size_ > 0 && draft_block_size_ < block_) {
      block_ = draft_block_size_;
    }
    if (block_ == 0) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    query_rows_ = block_ + 1;
    n_ = config_.target_layer_ids.size();
    rank_ = config_.selector_rank;
    topk_ = config_.selector_top_k;
    ctx_window_ = config_.sliding_window;
    if (const char* env = std::getenv("TESSERA_DFLASH2_CTX"); env != nullptr) {
      const int v = std::atoi(env);
      if (v > 0) {
        ctx_window_ = static_cast<std::size_t>(v);
      }
    }
    drafter_.SetContextLimit(ctx_window_ == 0 ? 0 : ctx_window_ + 1);
    capture_layers_ = std::vector<std::size_t>(
        config_.target_layer_ids.begin(), config_.target_layer_ids.end());
    for (std::size_t i = 0; i < n_; ++i) {
      auto buffer = backend.AllocateBuffer(block_ * hidden_ * 4,
                                           MemoryKind::Device);
      if (!buffer) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      capture_storage_.push_back(std::move(*buffer));
    }
    for (auto& buffer : capture_storage_) {
      capture_ptrs_.push_back(buffer.get());
      capture_const_.push_back(buffer.get());
    }
    auto logits = backend.AllocateBuffer(query_rows_ * vocab_ * 4,
                                         MemoryKind::Device);
    auto draft_hidden = backend.AllocateBuffer(query_rows_ * hidden_ * 4,
                                               MemoryKind::Device);
    auto sel_hidden = backend.AllocateBuffer(block_ * hidden_ * 4,
                                             MemoryKind::Device);
    auto query = backend.AllocateBuffer(query_rows_ * hidden_ * 4,
                                        MemoryKind::Device);
    if (!logits || !draft_hidden || !sel_hidden || !query) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    logits_ = std::move(*logits);
    draft_hidden_ = std::move(*draft_hidden);
    sel_hidden_ = std::move(*sel_hidden);
    query_slot_ = std::move(*query);
    auto selector_gemm = backend.LoadKernel("gemm_f32_batched", {});
    auto selector_kernel = backend.LoadKernel("selector_edge_score", {});
    if (selector_gemm) {
      selector_gemm_ = std::move(*selector_gemm);
    }
    if (selector_kernel) {
      selector_kernel_ = std::move(*selector_kernel);
    }
    prepared_ = true;
    return {};
  }

  std::size_t DraftBlock() const override { return block_; }

  std::span<const std::size_t> CaptureLayers() const override {
    return capture_layers_;
  }

  std::span<Buffer* const> CaptureBuffers() override { return capture_ptrs_; }

  std::expected<void, StatusCode> OnAnchor(
      Backend& backend, Model& target, core::DecodeCache& cache,
      std::uint32_t token, std::uint64_t position,
      std::span<const float> hidden) override {
    (void)target;
    (void)cache;
    (void)position;
    (void)hidden;
    anchor_ = token;
    return drafter_.AppendContext(backend, capture_const_, 1);
  }

  std::expected<std::size_t, StatusCode> Draft(
      Backend& backend, Model& target, core::DecodeCache& cache,
      std::span<const float> current_logits, std::uint32_t anchor,
      std::span<std::uint32_t> drafts) override {
    (void)target;
    (void)cache;
    (void)current_logits;
    const std::size_t ctx = drafter_.ContextRows() - 1;
    if (ctx == 0) {
      return std::unexpected(StatusCode::DeviceError);
    }
    auto query = QueryEmbeddings(backend, *embed_, anchor,
                                 config_.mask_token_id, block_, hidden_);
    if (!query) {
      return std::unexpected(query.error());
    }
    auto run = drafter_.Run(backend, **query, *output_->device, *head_gemm_,
                            *logits_, query_rows_, ctx, drafter_.ContextBase(),
                            vocab_, draft_hidden_.get());
    if (!run) {
      return std::unexpected(run.error());
    }
    backend.Synchronize();
    std::vector<float> draft_logits(query_rows_ * vocab_);
    if (!backend.CopyD2H(*logits_,
                         reinterpret_cast<std::byte*>(draft_logits.data()),
                         draft_logits.size() * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    std::span<const float> mask_logits(draft_logits);
    mask_logits = mask_logits.subspan(vocab_);
    const DraftSelectorWeights& selector = drafter_.Weights().Selector();
    if (selector.projection != nullptr && selector_gemm_ &&
        selector_kernel_) {
      std::vector<std::uint32_t> cand_ids(block_ * topk_);
      std::vector<float> unary(block_ * topk_);
      auto candidates =
          DraftCandidates(mask_logits, std::span<std::uint32_t>(cand_ids),
                          std::span<float>(unary), block_, vocab_, topk_);
      if (!candidates) {
        return std::unexpected(candidates.error());
      }
      std::vector<std::int32_t> anchors(block_,
                                        static_cast<std::int32_t>(anchor));
      auto upload = [&](const void* data, std::size_t bytes) {
        auto buffer = backend.AllocateBuffer(bytes, MemoryKind::Device);
        if (buffer) {
          backend.CopyH2D(**buffer, std::span<const std::byte>(
                                       reinterpret_cast<const std::byte*>(data),
                                       bytes));
        }
        return buffer;
      };
      auto cand_buf = upload(cand_ids.data(), cand_ids.size() * 4);
      auto anchor_buf = upload(anchors.data(), anchors.size() * 4);
      auto unary_buf = upload(unary.data(), unary.size() * 4);
      auto proj_buf =
          backend.AllocateBuffer(block_ * rank_ * 4, MemoryKind::Device);
      auto scores_buf =
          backend.AllocateBuffer(block_ * topk_ * topk_ * 4, MemoryKind::Device);
      if (!cand_buf || !anchor_buf || !unary_buf || !proj_buf || !scores_buf) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      if (!backend.CopyD2D(*draft_hidden_, hidden_ * 4, *sel_hidden_, 0,
                           block_ * hidden_ * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      auto scored = DraftSelectorDevice(
          backend, *selector_gemm_, *selector_kernel_, **proj_buf,
          *sel_hidden_, *selector.projection, *selector.predecessor,
          *selector.successor, **cand_buf, **anchor_buf, **unary_buf,
          **scores_buf, block_, hidden_, rank_, vocab_, topk_);
      if (!scored) {
        return std::unexpected(scored.error());
      }
      backend.Synchronize();
      std::vector<float> scores(block_ * topk_ * topk_);
      if (!backend.CopyD2H(**scores_buf,
                           reinterpret_cast<std::byte*>(scores.data()),
                           scores.size() * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      std::size_t predecessor = 0;
      for (std::size_t r = 0; r < block_ && r < drafts.size(); ++r) {
        const float* row = scores.data() + (r * topk_ + predecessor) * topk_;
        std::size_t best = 0;
        for (std::size_t c = 1; c < topk_; ++c) {
          if (row[c] > row[best]) {
            best = c;
          }
        }
        drafts[r] = cand_ids[r * topk_ + best];
        predecessor = best;
      }
      return block_;
    }
    for (std::size_t r = 0; r < block_ && r < drafts.size(); ++r) {
      std::span<const float> row(mask_logits.data() + r * vocab_, vocab_);
      drafts[r] = core::detail::ArgMax(row);
    }
    return block_;
  }

  std::expected<void, StatusCode> Commit(Backend& backend, Model& target,
                                         core::DecodeCache& cache,
                                         std::size_t accepted) override {
    (void)target;
    (void)cache;
    if (accepted == 0) {
      return {};
    }
    return drafter_.AppendContext(backend, capture_const_, accepted);
  }

 private:
  std::string draft_path_;
  std::size_t draft_block_size_ = 0;
  bool attached_ = false;
  bool prepared_ = false;
  spec::DFlash2Config config_;
  spec::DFlash2Drafter drafter_;
  const DeviceTensor* embed_ = nullptr;
  const DeviceTensor* output_ = nullptr;
  std::unordered_map<int, std::unique_ptr<Kernel>> gemms_;
  Kernel* head_gemm_ = nullptr;
  std::unique_ptr<Kernel> selector_gemm_;
  std::unique_ptr<Kernel> selector_kernel_;
  std::size_t hidden_ = 0;
  std::size_t vocab_ = 0;
  std::size_t block_ = 0;
  std::size_t query_rows_ = 0;
  std::size_t n_ = 0;
  std::size_t rank_ = 0;
  std::size_t topk_ = 0;
  std::size_t ctx_window_ = 0;
  std::uint32_t anchor_ = 0;
  std::vector<std::size_t> capture_layers_;
  std::vector<std::unique_ptr<Buffer>> capture_storage_;
  std::vector<Buffer*> capture_ptrs_;
  std::vector<const Buffer*> capture_const_;
  std::unique_ptr<Buffer> logits_;
  std::unique_ptr<Buffer> draft_hidden_;
  std::unique_ptr<Buffer> sel_hidden_;
  std::unique_ptr<Buffer> query_slot_;
};

}  // namespace

}  // namespace tessera::spec

namespace tessera {

std::unique_ptr<SpeculativeStrategy> CreateDFlash2Strategy() {
  return std::make_unique<spec::DFlash2Strategy>();
}

}  // namespace tessera
