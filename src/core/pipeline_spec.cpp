#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "core/pipeline_impl.hpp"
#include "core/spec_driver.hpp"
#include "tessera/architecture.hpp"
#include "tessera/backend.hpp"
#include "tessera/model.hpp"
#include "tessera/speculative.hpp"

// Speculative decoding for the pipeline route. The decode driver runs on one
// thread; the target drives the stages in order, so the drafter (the MTP
// head) runs on the last stage and every target forward runs across all
// stages.

namespace tessera {

// The single-thread pipeline decode target.
struct Pipeline::Impl::Target : public core::SpecTarget {
  explicit Target(Impl& impl) : impl_(impl) {}

  log::Diagnostics& diagnostics() override {
    return impl_.stages.back().engine->Diagnostics();
  }
  Backend& backend() override { return impl_.stages.back().engine->Owner(); }
  Model& model() override { return *impl_.stages.back().model; }
  core::DecodeCache& cache() override { return *impl_.stages.back().cache; }

  std::expected<void, StatusCode> Prepare(
      SpeculativeStrategy* strategy, const GenerateOptions& options) override {
    impl_.ResetCaches(options);
    if (strategy != nullptr) {
      auto prepared = strategy->Prepare(backend(), model());
      if (!prepared) {
        return std::unexpected(prepared.error());
      }
      strategy->SetDraftContext(options.draft_context);
    }
    return {};
  }

  std::expected<void, StatusCode> Prefill(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      std::vector<float>* hidden, std::vector<float>* logits,
      double* compute_ms) override {
    (void)compute_ms;
    for (std::size_t i = 0; i < prompt.size(); ++i) {
      const bool last = (i + 1 == prompt.size());
      auto out = impl_.Step(prompt[i], last, last ? hidden : nullptr);
      if (!out) {
        return std::unexpected(out.error());
      }
      if (last) {
        *logits = std::move(*out);
      }
      if (options.prefill_progress) {
        options.prefill_progress(i + 1, prompt.size());
      }
    }
    position_ = prompt.size();
    return {};
  }

  std::expected<std::uint32_t, StatusCode> GreedyStep(
      std::uint32_t token) override {
    auto out = impl_.Step(token, true, nullptr);
    if (!out) {
      return std::unexpected(out.error());
    }
    ++position_;
    return core::detail::ArgMax(*out);
  }

  std::expected<std::vector<float>, StatusCode> LogitsStep(
      std::uint32_t token, std::vector<float>* hidden) override {
    auto out = impl_.Step(token, true, hidden);
    if (!out) {
      return std::unexpected(out.error());
    }
    ++position_;
    return std::move(*out);
  }

  std::expected<DraftVerification, StatusCode> Verify(
      std::span<const std::uint32_t> draft,
      std::span<const float> prefix_logits, std::optional<std::uint32_t> anchor,
      std::vector<float>* hidden) override {
    const std::size_t prefix = position_;
    auto verified = impl_.VerifyBatch(draft, prefix_logits, anchor, hidden,
                                      prefix);
    if (!verified) {
      return std::unexpected(verified.error());
    }
    position_ += (anchor.has_value() ? 1 : 0) + verified->accepted;
    return verified;
  }

  void ReportDiagnostics(std::size_t loop_steps) override {
    (void)loop_steps;
  }

 private:
  Impl& impl_;
  std::size_t position_ = 0;
};

std::expected<DraftVerification, StatusCode> Pipeline::Impl::VerifyBatch(
    std::span<const std::uint32_t> draft, std::span<const float> prefix_logits,
    std::optional<std::uint32_t> anchor, std::vector<float>* hidden_out,
    std::size_t prefix_position) {
  const std::size_t n = stages.size();
  const std::size_t lead = anchor.has_value() ? 1 : 0;
  std::vector<std::uint32_t> tokens;
  tokens.reserve(draft.size() + lead);
  if (lead == 1) {
    tokens.push_back(*anchor);
  }
  tokens.insert(tokens.end(), draft.begin(), draft.end());
  const std::size_t rows = tokens.size();
  if (rows == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto config = stages.back().model->Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const std::size_t hidden = hidden_dim;
  const std::size_t vocab = config->vocab_size;
  // One outgoing boundary buffer per stage but the last, one incoming per
  // stage but the first, and the final per-row hidden on the last stage.
  std::vector<std::unique_ptr<Buffer>> outgoing(n);
  std::vector<std::unique_ptr<Buffer>> incoming(n);
  for (std::size_t i = 0; i + 1 < n; ++i) {
    auto out = stages[i].engine->Owner().AllocateBuffer(rows * hidden * 4,
                                                        MemoryKind::Device);
    auto in = stages[i + 1].engine->Owner().AllocateBuffer(
        rows * hidden * 4, MemoryKind::Device);
    if (!out || !in) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    outgoing[i] = std::move(*out);
    incoming[i + 1] = std::move(*in);
  }
  auto final_made = stages.back().engine->Owner().AllocateBuffer(
      rows * hidden * 4, MemoryKind::Device);
  if (!final_made) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  std::unique_ptr<Buffer> final_hidden = std::move(*final_made);

  std::vector<float> flat;
  for (std::size_t i = 0; i < n; ++i) {
    Stage& stage = stages[i];
    const Architecture* arch = stage.model->Arch();
    if (arch == nullptr) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    Backend& backend = stage.engine->Owner();
    const bool last = (i + 1 == n);
    // ForwardBatch reads `tokens.size()` for the row count even when
    // `embeddings` replaces the gathered rows, so a later stage passes a
    // same-sized placeholder.
    std::vector<std::uint32_t> placeholder;
    std::span<const std::uint32_t> stage_tokens = tokens;
    if (i > 0) {
      placeholder.assign(rows, 0);
      stage_tokens = placeholder;
    }
    const Buffer* embeddings = i > 0 ? incoming[i].get() : nullptr;
    std::vector<std::size_t> cap_layers{stage.boundary_layer};
    std::vector<Buffer*> cap_bufs{last ? final_hidden.get()
                                       : outgoing[i].get()};
    auto status = arch->ForwardBatch(
        backend, *stage.model, *stage.cache, stage_tokens,
        last ? &flat : nullptr, nullptr, last, embeddings, &cap_layers,
        &cap_bufs, /*snapshot_state=*/true);
    if (!status) {
      return std::unexpected(status.error());
    }
    if (!last) {
      auto copied = CopyAcrossDevices(*outgoing[i], 0, *incoming[i + 1], 0,
                                      rows * hidden * 4);
      if (!copied) {
        return std::unexpected(copied.error());
      }
    }
  }

  DraftVerification result;
  std::size_t accepted = 0;
  const auto logits_row = [&](std::size_t row) {
    return std::span<const float>(flat.data() + row * vocab, vocab);
  };
  if (lead == 1) {
    for (std::size_t i = 0; i < draft.size(); ++i) {
      if (core::detail::ArgMax(logits_row(i)) != draft[i]) {
        break;
      }
      ++accepted;
    }
    result.accepted = accepted;
    result.next_token = core::detail::ArgMax(logits_row(accepted));
  } else {
    std::span<const float> last = prefix_logits;
    for (std::size_t i = 0; i < draft.size(); ++i) {
      if (core::detail::ArgMax(last) != draft[i]) {
        break;
      }
      ++accepted;
      last = logits_row(i + lead);
    }
    result.accepted = accepted;
    result.logits.assign(last.begin(), last.end());
    result.next_token = core::detail::ArgMax(result.logits);
  }
  if (hidden_out != nullptr && (accepted > 0 || lead == 1)) {
    const std::size_t row = lead == 1 ? accepted : accepted - 1;
    std::vector<float> all(rows * hidden);
    if (!stages.back().engine->Owner().CopyD2H(
            *final_hidden, reinterpret_cast<std::byte*>(all.data()),
            all.size() * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    hidden_out->assign(all.begin() + row * hidden,
                       all.begin() + (row + 1) * hidden);
  }
  // Keep the anchor plus the accepted drafts on every stage.
  const std::size_t keep = prefix_position + lead + accepted;
  for (Stage& stage : stages) {
    const Architecture* arch = stage.model->Arch();
    if (arch == nullptr) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    auto truncated = arch->TruncateTarget(stage.engine->Owner(), *stage.model,
                                          *stage.cache, keep);
    if (!truncated) {
      return std::unexpected(truncated.error());
    }
  }
  return result;
}

std::expected<GenerateOutcome, StatusCode> Pipeline::Impl::RunSpeculative(
    std::span<const std::uint32_t> prompt, const GenerateOptions& options,
    std::vector<std::uint32_t>* produced) {
  if (prompt.empty() || prompt.size() > context) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  produced->clear();
  Target target(*this);
  auto outcome = core::RunDecode(
      target, strategy.get(), options,
      [produced](std::uint32_t token) {
        produced->push_back(token);
        return true;
      });
  if (!outcome) {
    return std::unexpected(outcome.error());
  }
  return *outcome;
}

std::expected<void, StatusCode> Pipeline::AttachSpeculative(
    const std::function<std::unique_ptr<SpeculativeStrategy>()>& factory,
    const StrategyOptions& options) {
  if (!factory) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto strategy = factory();
  if (strategy == nullptr) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto attached = strategy->Attach(options);
  if (!attached) {
    return std::unexpected(attached.error());
  }
  impl_->strategy = std::move(strategy);
  return {};
}

}  // namespace tessera
